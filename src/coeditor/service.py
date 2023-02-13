# End-user API as an editing suggestion tool.

import difflib
import io
import sys
import textwrap
from pprint import pprint

import jedi
import parso
import torch
from parso.python import tree as ptree

from coeditor.c3problem import (
    C3GeneratorCache,
    C3Problem,
    C3ProblemTokenizer,
    JediUsageAnalyzer,
    SrcInfo,
    TkC3Problem,
)
from coeditor.change import (
    Added,
    Change,
    Deleted,
    Modified,
    default_show_diff,
    show_change,
)
from coeditor.common import *
from coeditor.encoding import (
    Newline_id,
    StrDelta,
    TkDelta,
    is_extra_id,
    tk_splitlines,
    tokens_to_change,
)
from coeditor.model import (
    BatchArgs,
    C3DataLoader,
    DecodingArgs,
    RetrievalDecodingResult,
    RetrievalEditorModel,
    RetrievalModelPrediction,
)
from coeditor.scoped_changes import (
    ChangedSpan,
    DefaultIgnoreDirs,
    JModule,
    JModuleChange,
    StatementSpan,
    get_python_files,
    parse_module_script,
)

from .git import file_content_from_commit

_tlogger = TimeLogger()

CommitHash = str
SysTime = float


@dataclass
class ChangeDetector:
    project: Path
    untracked_as_additions: bool = True
    ignore_dirs: Collection[str] = field(default_factory=lambda: DefaultIgnoreDirs)
    # if only the first target line is specified, how many following lines to edit.
    max_lines_to_edit: int = 30

    def __post_init__(self):
        self.script_cache = TimedCache()
        self.analyzer = JediUsageAnalyzer()
        self._index_cache = TimedCache[RelPath, JModule, CommitHash]()
        self._now_cache = TimedCache[RelPath, tuple[JModule, jedi.Script], SysTime]()
        proj = self.project
        self.jproj = jedi.Project(path=proj, added_sys_path=[proj / "src"])

        self._updated_now_modules = set[ModuleName]()
        self._updated_index_modules = set[ModuleName]()
        self.gcache = C3GeneratorCache({})
        # preemptively parse all moduels
        self.get_current_modules()

    def _get_index_hash(self, path: RelPath) -> CommitHash:
        out = run_command(["git", "ls-files", "-s", path.as_posix()], cwd=self.project)
        hash = out.split(" ")[1]
        assert_eq(len(hash), 40)
        return hash

    def _get_mod_time(self, path: RelPath) -> SysTime:
        return os.stat(self.project / path).st_mtime

    def _parse_index_module(self, path: RelPath) -> JModule:
        code = file_content_from_commit(self.project, "", path.as_posix())
        mod = parso.parse(code)
        assert isinstance(mod, ptree.Module)
        mname = path_to_module_name(path)
        self._updated_index_modules.add(mname)
        return JModule(mname, mod)

    def _parse_now_module_script(self, path: RelPath):
        m, s = parse_module_script(self.jproj, self.project / path)
        self._updated_now_modules.add(m.mname)
        return m, s

    def get_index_module(self, path: RelPath) -> JModule:
        stamp = self._get_index_hash(path)
        return self._index_cache.cached(
            path, stamp, lambda: self._parse_index_module(path)
        )

    def get_current_module(self, path: RelPath) -> JModule:
        stamp = self._get_mod_time(path)
        mod, _ = self._now_cache.cached(
            path, stamp, lambda: self._parse_now_module_script(path)
        )
        return mod

    def get_current_script(self, path: RelPath) -> jedi.Script:
        stamp = self._get_mod_time(path)
        _, script = self._now_cache.cached(
            path, stamp, lambda: self._parse_now_module_script(path)
        )
        return script

    def get_current_modules(self) -> dict[RelPath, JModule]:
        files = get_python_files(self.project)
        return {f: self.get_current_module(f) for f in files}

    def get_problem(
        self,
        target_file: RelPath,
        target_lines: Sequence[int] | int,
    ) -> tuple[C3Problem, StatementSpan]:
        def is_src(path_s: str) -> bool:
            path = Path(path_s)
            return path.suffix == ".py" and all(
                p not in self.ignore_dirs for p in path.parts
            )

        if isinstance(target_lines, int):
            first_line = target_lines
        else:
            first_line = target_lines[0]
        changed_files = run_command(
            ["git", "status", "--porcelain"], cwd=self.project
        ).splitlines()

        path_changes = set[Change[str]]()

        for change_line in changed_files:
            if not change_line:
                continue
            if change_line[2] == " ":
                tag = change_line[:2]
                path = change_line[3:]
                if not is_src(path):
                    continue
                if tag.endswith("M") or tag.endswith("A") or tag == "??":
                    if tag == "??" and not self.untracked_as_additions:
                        continue
                    if tag.endswith("A"):
                        path_changes.add(Added(path))
                    elif tag.endswith("D"):
                        path_changes.add(Deleted(path))
                    if tag.endswith("M"):
                        path_changes.add(Modified(path, path))
            else:
                tag, path1, path2 = change_line.split(" ")
                assert tag.startswith("R")
                if is_src(path1):
                    path_changes.add(Deleted(path1))
                if is_src(path2):
                    path_changes.add(Added(path2))

        # use inverse changes so that we can locate spans using their current locations
        rev_changed = dict[ModuleName, JModuleChange]()
        for path_change in path_changes:
            path = self.project / path_change.earlier
            rel_path = to_rel_path(path.relative_to(self.project))
            if not isinstance(path_change, Added) and not path.exists():
                warnings.warn(f"File missing: {rel_path}")
                if isinstance(path_change, Deleted):
                    continue
                elif isinstance(path_change, Modified):
                    path_change = Added(path_change.after)
            match path_change:
                case Added():
                    mod = self.get_current_module(rel_path)
                    rev_changed[mod.mname] = JModuleChange.from_modules(
                        Deleted(mod), only_ast_changes=False
                    )
                case Deleted():
                    mod = self.get_index_module(rel_path)
                    rev_changed[mod.mname] = JModuleChange.from_modules(
                        Added(mod), only_ast_changes=False
                    )
                case Modified(path1, path2):
                    assert path1 == path2
                    mod_old = self.get_index_module(rel_path)
                    mod_new = self.get_current_module(rel_path)
                    rev_changed[mod_new.mname] = JModuleChange.from_modules(
                        Modified(mod_new, mod_old), only_ast_changes=False
                    )

        target_mod = self.get_current_module(target_file)
        span = target_mod.as_scope.search_span_by_line(first_line)
        if span is None:
            print_err("Target scope:")
            print_err(target_mod.as_scope)
            raise ValueError(f"Could not find a statement span at line {first_line}.")

        if target_mod.mname not in rev_changed:
            print(f"Target module '{target_mod.mname}' has not changed.")
            rev_changed[target_mod.mname] = JModuleChange(
                Modified.from_unchanged(target_mod), []
            )

        cspans = [
            c
            for c in rev_changed[target_mod.mname].changed
            if first_line in c.line_range
        ]
        if len(cspans) == 0:
            # Create a trivial change for the target module if it wasn't changed.
            print(f"Target span has not changed. Creating a trivial change.")
            parents = [Modified.from_unchanged(s) for s in span.scope.ancestors()]
            cspan = ChangedSpan(
                Modified.from_unchanged(span.code), parents, span.line_range
            )
        else:
            if len(cspans) > 1:
                warnings.warn(
                    f"Multiple spans at line {first_line}. Using only the first one."
                )
            cspan = cspans[0]

        with _tlogger.timed("usage analysis"):
            script = self.get_current_script(target_file)
            lines_to_analyze = set(cspan.line_range.to_range())
            lines_to_analyze.update(cspan.header_line_range.to_range())
            self.analyzer.error_counts.clear()
            target_usages = self.analyzer.get_line_usages(
                script, lines_to_analyze, silent=False
            )
            if errors := self.analyzer.error_counts:
                print("Errors during usage analysis:")
                pprint(errors)

        src_info = SrcInfo(project=str(self.project), commit=None)
        changed = {m: c.inverse() for m, c in rev_changed.items()}
        cspan = cspan.inverse()
        if isinstance(target_lines, int):
            n_above = max(1, self.max_lines_to_edit // 2)
            edit_start = max(cspan.line_range[0], first_line - n_above)
            edit_stop = min(edit_start + self.max_lines_to_edit, cspan.line_range[1])
            target_lines = range(edit_start, edit_stop)

        modules = self.get_current_modules()
        self.gcache.set_module_map({m.mname: m for m in modules.values()})
        self.gcache.clear_caches(self._updated_index_modules, self._updated_now_modules)
        self._updated_index_modules.clear()
        self._updated_now_modules.clear()

        prob = self.gcache.create_problem(
            cspan, target_lines, changed, target_usages, src_info
        )
        return prob, span


# add, delete, replace, equal
StatusTag = Literal["A", "D", "R", " "]


class EditSuggestion(TypedDict):
    score: float
    change_preview: str
    new_code: str
    line_status: list[tuple[int, StatusTag]]


def path_to_module_name(rel_path: RelPath) -> ModuleName:
    parts = rel_path.parts
    assert parts[-1].endswith(".py"), f"not a python file: {rel_path}"
    if parts[0] == "src":
        parts = parts[1:]
    if parts[-1] == "__init__.py":
        return ".".join(parts[:-1])
    else:
        name = parts[-1][:-3]
        return ".".join((*parts[:-1], name))


@dataclass
class ServiceResponse:
    target_file: str
    edit_start: tuple[int, int]
    edit_end: tuple[int, int]
    target_lines: Sequence[int]
    input_code: str
    suggestions: list[EditSuggestion]

    def to_json(self):
        return {
            "target_file": self.target_file,
            "edit_start": self.edit_start,
            "edit_end": self.edit_end,
            "old_code": self.input_code,
            "suggestions": [s for s in self.suggestions],
            "target_lines": list(self.target_lines),
        }

    def print(self, file=sys.stdout):
        print(f"Target file: {self.target_file}", file=file)
        print(f"Edit range: {self.edit_start} - {self.edit_end}", file=file)
        target_lines = self.target_lines
        if target_lines:
            target_lines = f"{target_lines[0]}--{target_lines[-1]}"
        print(f"Target lines: {target_lines}", file=file)
        for i, s in enumerate(self.suggestions):
            print(
                f"\t--------------- Suggestion {i} (score: {s['score']:.3g}) ---------------",
                file=file,
            )
            print(textwrap.indent(s["change_preview"], "\t"), file=file)
        # print(f"Input code:", file=file)
        # print(self.input_code, file=file)

    def __str__(self) -> str:
        # use the print above
        s = io.StringIO()
        self.print(s)
        return s.getvalue()


ErrorStr = str


@dataclass
class _EditRegion:
    current_code: str
    target_lines: Sequence[int]
    target_line_ids: Sequence[int]


@dataclass
class EditPredictionService:
    def __init__(
        self,
        detector: ChangeDetector,
        model: RetrievalEditorModel,
        c3_tkn: C3ProblemTokenizer = C3ProblemTokenizer(
            max_query_tks=1024,
            max_ref_tks=1024,
            max_output_tks=512,
            max_ref_tks_sum=1024 * 12,
        ),
        dec_args: DecodingArgs = DecodingArgs(),
    ) -> None:
        self.project = detector.project
        self.detector = detector
        self.model = model
        self.c3_tkn = c3_tkn
        self.dec_args = dec_args
        self.show_max_solutions = 3
        self.tlogger = _tlogger
        model.tlogger = _tlogger

    def _suggest_edit_two_steps(
        self,
        file: RelPath,
        edit_lines: Sequence[int] | int,
        log_dir: Path | None = Path(".coeditor_logs"),
    ) -> tuple[_EditRegion, Callable[[], ServiceResponse]]:
        timed = self.tlogger.timed

        with timed("get c3 problem"):
            problem, span = self.detector.get_problem(file, edit_lines)

        with timed("tokenize c3 problem"):
            tk_prob = self.c3_tkn.tokenize_problem(problem)

        target = self.get_target_code(span.code, problem, tk_prob)

        def next_step():
            batch = C3DataLoader.pack_batch([tk_prob])
            original = problem.span.original.tolist()

            with timed("run model"), torch.autocast("cuda"):
                predictions = self.model.predict_on_batch(
                    batch, [original], self.dec_args, self.show_max_solutions
                )
                assert_eq(len(predictions), 1)
                predictions = predictions[0]
                assert predictions

            if log_dir is not None:
                log_dir.mkdir(exist_ok=True)
                input_tks = batch["input_ids"][0]
                references = batch["references"]
                output_truth = batch["labels"][0]
                print(f"Writing logs to: {log_dir}")
                for i, pred in enumerate(predictions):
                    with (log_dir / f"solution-{i}.txt").open("w") as f:
                        pred_tks = pred.out_tks
                        score = pred.score
                        print(f"{problem.edit_line_ids=}", file=f)
                        print(f"{len(input_tks)=}", file=f)
                        print(f"{len(references)=}", file=f)
                        print(f"Solution score: {score:.3g}", file=f)
                        print(f"Marginalized samples:", pred.n_samples, file=f)
                        pred = RetrievalModelPrediction(
                            input_ids=input_tks,
                            output_ids=pred_tks,
                            labels=output_truth,
                            references=references,
                        )
                        pred_str = RetrievalDecodingResult.show_prediction(
                            problem, pred
                        )
                        print(pred_str, file=f)

            target_lines = target.target_lines
            suggestions = list[EditSuggestion]()
            for pred in predictions:
                pred_change = self.apply_edit_to_elem(
                    target,
                    problem,
                    pred.out_tks,
                )
                preview = "\n".join(
                    compute_line_diffs_fast(
                        splitlines(pred_change.before),
                        splitlines(pred_change.after),
                    )
                )
                diff_ops = get_diff_ops(
                    splitlines(pred_change.before), splitlines(pred_change.after)
                )
                line_status = dict[int, StatusTag]()
                for tag, (i1, i2), _ in diff_ops:
                    if tag == "A":
                        line_status[i1] = "A"
                        continue
                    for i in range(i1, i2):
                        if i not in line_status:
                            line_status[i] = tag
                line_status = [
                    (i + target_lines[0], tag) for i, tag in line_status.items()
                ]

                suggestion = EditSuggestion(
                    score=pred.score,
                    change_preview=preview,
                    new_code=pred_change.after,
                    line_status=line_status[: len(target_lines)],
                )
                suggestions.append(suggestion)

            return ServiceResponse(
                target_file=str(self.project / file),
                edit_start=(target_lines[0], 0),
                edit_end=(target_lines[-1] + 1, 0),
                target_lines=target.target_lines,
                input_code=target.current_code,
                suggestions=suggestions,
            )

        return target, next_step

    def suggest_edit(
        self,
        file: RelPath,
        edit_lines: Sequence[int] | int,
        log_dir: Path | None = Path(".coeditor_logs"),
    ) -> ServiceResponse:
        _, f = self._suggest_edit_two_steps(file, edit_lines, log_dir)
        return f()

    @staticmethod
    def get_target_code(
        current_code: str,
        problem: C3Problem,
        tk_prob: TkC3Problem,
    ) -> _EditRegion:
        n_out_segs = sum(1 for tk in tk_prob.output_tks if is_extra_id(tk))
        edit_line_ids = problem.edit_line_ids[:n_out_segs]
        target_lines = problem.line_ids_to_input_lines(edit_line_ids)
        current_start = target_lines[0] - problem.span.line_range[0]
        current_stop = target_lines[-1] - problem.span.line_range[0] + 1
        current_lines = current_code.split("\n")[current_start:current_stop]
        current_lines.append("")
        current_code = "\n".join(current_lines)
        return _EditRegion(current_code, target_lines, edit_line_ids)

    @staticmethod
    def apply_edit_to_elem(
        target: _EditRegion,
        problem: C3Problem,
        out_tks: TokenSeq,
    ) -> Modified[str]:
        edit_line_ids = target.target_line_ids
        edit_start = edit_line_ids[0]
        edit_stop = edit_line_ids[-1] + 1

        delta = (
            TkDelta.from_output_tks(problem.edit_line_ids, out_tks)
            .for_input_range((edit_start, edit_stop + 1))
            .shifted(-edit_start)
        )

        change1_tks = get_tk_lines(
            problem.span.original.tolist(), range(edit_start, edit_stop)
        )
        change1 = tokens_to_change(change1_tks)
        change2_tks = delta.apply_to_change(change1_tks)
        change2 = tokens_to_change(change2_tks)
        # change2 is supposed to be the change we want. However, the tokenizer
        # sometimes does not perfectly encode the input, hence we extract the
        # delta and directly apply it to the current code to avoid unnecessary
        # tokenization.
        _, delta2 = StrDelta.from_change(Modified(change1.after, change2.after))

        new_code = delta2.apply_to_input(target.current_code)
        return Modified(target.current_code, new_code)


_tag_map: dict[str, StatusTag] = {
    "insert": "A",
    "delete": "D",
    "replace": "R",
    "equal": " ",
}


def get_diff_ops(
    before: Sequence[str], after: Sequence[str]
) -> list[tuple[StatusTag, tuple[int, int], tuple[int, int]]]:
    matcher = difflib.SequenceMatcher(None, before, after)
    return [
        (_tag_map[tag], (i1, i2), (j1, j2))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
    ]


def get_tk_lines(tks: TokenSeq, line_ids: Sequence[int]) -> TokenSeq:
    lines = tk_splitlines(tks)
    return join_list((lines[i] for i in line_ids), Newline_id)


def show_location(loc: CodePosition):
    return f"{loc[0]}:{loc[1]}"
