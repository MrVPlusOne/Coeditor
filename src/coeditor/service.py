# End-user API as an editing suggestion tool.

import io
import sys
import textwrap

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
)
from coeditor.change import Added, Change, Deleted, Modified, default_show_diff
from coeditor.common import *
from coeditor.encoding import TkDelta, input_lines_from_tks, tokens_to_change
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
    get_python_files,
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

    def __post_init__(self):
        self.script_cache = TimedCache()
        self.analyzer = JediUsageAnalyzer()
        self._index_cache = TimedCache[RelPath, JModule, CommitHash]()
        self._now_cache = TimedCache[RelPath, JModule, SysTime]()

    def _get_index_content(self, path: RelPath):
        return file_content_from_commit(self.project, "", path.as_posix())

    def _get_index_stamp(self, path: RelPath) -> CommitHash:
        out = run_command(["git", "ls-files", "-s", path.as_posix()], cwd=self.project)
        hash = out.split(" ")[1]
        assert_eq(len(hash), 40)
        return hash

    def _get_mod_time(self, path: RelPath) -> SysTime:
        return os.stat(self.project / path).st_mtime

    def _parse_index_module(self, path: RelPath) -> JModule:
        code = self._get_index_content(path)
        mod = parso.parse(code)
        assert isinstance(mod, ptree.Module)
        mname = path_to_module_name(path)
        return JModule(mname, mod)

    def _get_index_module(self, path: RelPath) -> JModule:
        stamp = self._get_index_stamp(path)
        return self._index_cache.cached(
            path, stamp, lambda: self._parse_index_module(path)
        )

    def _parse_current_module(self, path: RelPath) -> JModule:
        code = (self.project / path).read_text()
        mod = parso.parse(code)
        assert isinstance(mod, ptree.Module)
        mname = path_to_module_name(path)
        return JModule(mname, mod)

    def get_current_module(self, path: RelPath) -> JModule:
        stamp = self._get_mod_time(path)
        return self._now_cache.cached(
            path, stamp, lambda: self._parse_current_module(path)
        )

    def get_current_modules(self) -> dict[RelPath, JModule]:
        files = get_python_files(self.project)
        return {f: self.get_current_module(f) for f in files}

    def get_problem(
        self,
        target_file: RelPath,
        target_lines: Sequence[int] | int,
    ) -> C3Problem:
        def is_src(path_s: str) -> bool:
            path = Path(path_s)
            return path.suffix == ".py" and all(
                p not in self.ignore_dirs for p in path.parts
            )

        if isinstance(target_lines, int):
            first_line = target_lines
            edit_lines = None
        else:
            first_line = target_lines[0]
            edit_lines = target_lines
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
                    mod = self._get_index_module(rel_path)
                    rev_changed[mod.mname] = JModuleChange.from_modules(
                        Added(mod), only_ast_changes=False
                    )
                case Modified(path1, path2):
                    assert path1 == path2
                    mod_old = self._get_index_module(rel_path)
                    mod_new = self.get_current_module(rel_path)
                    rev_changed[mod_new.mname] = JModuleChange.from_modules(
                        Modified(mod_new, mod_old), only_ast_changes=False
                    )
        modules = self.get_current_modules()
        gcache = C3GeneratorCache({m.mname: m for m in modules.values()})

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
        if len(cspans) != 1:
            # Create a trivial change for the target module if it wasn't changed.
            print(f"Target span has not changed. Creating a trivial change.")
            parents = [Modified.from_unchanged(s) for s in span.scope.ancestors()]
            cspan = ChangedSpan(
                Modified.from_unchanged(span.code), parents, span.line_range
            )
        else:
            cspan = cspans[0]

        with _tlogger.timed("usage analysis"):
            script = jedi.Script(path=self.project / target_file)
            lines_to_analyze = set(cspan.line_range.to_range())
            lines_to_analyze.update(cspan.header_line_range.to_range())
            target_usages = self.analyzer.get_line_usages(
                script, lines_to_analyze, silent=True
            )
        src_info = SrcInfo(project=str(self.project), commit=None)
        changed = {m: c.inverse() for m, c in rev_changed.items()}
        cspan = cspan.inverse()
        prob = gcache.create_problem(
            cspan, edit_lines, changed, target_usages, src_info
        )
        return prob


@dataclass
class EditSuggestion:
    score: float
    change_preview: str
    new_code: str

    def to_json(self):
        return {
            "score": self.score,
            "change_preview": self.change_preview,
            "new_code": self.new_code,
        }


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
            "suggestions": [s.to_json() for s in self.suggestions],
        }

    def print(self, file=sys.stdout):
        print(f"Target file: {self.target_file}", file=file)
        print(f"Edit range: {self.edit_start} - {self.edit_end}", file=file)
        print(f"Target lines: {self.target_lines}", file=file)
        for i, s in enumerate(self.suggestions):
            print(
                f"\t--------------- Suggestion {i} (score: {s.score:.3g}) ---------------",
                file=file,
            )
            print(textwrap.indent(s.change_preview, "\t"), file=file)
        print(f"Input code:", file=file)
        print(self.input_code, file=file)

    def __str__(self) -> str:
        # use the print above
        s = io.StringIO()
        self.print(s)
        return s.getvalue()


ErrorStr = str


@dataclass
class EditPredictionService:
    def __init__(
        self,
        detector: ChangeDetector,
        model: RetrievalEditorModel,
        batch_args: BatchArgs = BatchArgs.service_default(),
        c3_tkn: C3ProblemTokenizer = C3ProblemTokenizer(),
        dec_args: DecodingArgs = DecodingArgs(),
    ) -> None:
        self.project = detector.project
        self.detector = detector
        self.model = model
        self.batch_args = batch_args
        self.c3_tkn = c3_tkn
        self.dec_args = dec_args
        self.show_max_solutions = 3
        self.tlogger = _tlogger
        model.tlogger = _tlogger

    def suggest_edit(
        self,
        file: Path,
        edit_lines: Sequence[int] | int,
        log_dir: Path | None = Path(".coeditor_logs"),
    ) -> ServiceResponse:
        timed = self.tlogger.timed
        project = self.project

        if file.is_absolute():
            file = file.relative_to(project)
        file = to_rel_path(file)

        with timed("get c3 problem"):
            problem = self.detector.get_problem(file, edit_lines)

        with timed("tokenize c3 problem"):
            tk_prob = self.c3_tkn.tokenize_problem(problem)
            target_begin = problem.span.line_range[0]
            target_lines = input_lines_from_tks(tk_prob.main_input.tolist())
            target_lines = [target_begin + l for l in target_lines]
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
                    print(f"{problem.edit_lines=}", file=f)
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
                    pred_str = RetrievalDecodingResult.show_prediction(problem, pred)
                    print(pred_str, file=f)

        suggestions = list[EditSuggestion]()
        for pred in predictions:
            suggested_change, preview = self.apply_edit_to_elem(
                problem,
                pred.out_tks,
            )
            suggestion = EditSuggestion(
                score=pred.score,
                change_preview=preview,
                new_code=suggested_change.after,
            )
            suggestions.append(suggestion)

        span = problem.span
        old_code = tokens_to_change(span.original.tolist()).after

        return ServiceResponse(
            target_file=file.as_posix(),
            edit_start=(span.line_range[0], 0),
            edit_end=(span.line_range[1], 0),
            target_lines=target_lines,
            input_code=old_code,
            suggestions=suggestions,
        )

    @staticmethod
    def apply_edit_to_elem(
        problem: C3Problem,
        out_tks: TokenSeq,
    ) -> tuple[Modified[str], str]:
        change_tks = problem.span.original.tolist()
        delta = TkDelta.from_output_tks(problem.edit_lines, out_tks)
        new_change_tks = delta.apply_to_change(change_tks)
        new_change = tokens_to_change(new_change_tks)
        current_code = tokens_to_change(change_tks).after
        preview = default_show_diff(current_code, new_change.after)
        return new_change, preview


def replace_lines(text: str, span: CodeRange, replacement: str):
    start_ln, end_ln = span[0][0] - 1, span[1][0]
    replacemnet = textwrap.indent(textwrap.dedent(replacement), " " * span[0][1])
    old_lines = text.split("\n")
    new_lines = old_lines[:start_ln] + [replacemnet] + old_lines[end_ln + 1 :]
    return "\n".join(new_lines)


def get_span(text: str, span: CodeRange):
    start_ln, end_ln = span[0][0] - 1, span[1][0]
    old_lines = text.split("\n")
    new_lines = old_lines[start_ln : end_ln + 1]
    new_lines[0] = new_lines[0][span[0][1] :]
    new_lines[-1] = new_lines[-1][: span[1][1]]
    return "\n".join(new_lines)


def show_location(loc: CodePosition):
    return f"{loc[0]}:{loc[1]}"
