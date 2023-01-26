# End-user API as an editing suggestion tool.

import io
import logging
import sys
import textwrap

import torch
from libcst.metadata import CodePosition, CodeRange

from coeditor.common import *
from coeditor.encoders import (
    EditRequest,
    QueryRefEditEncoder,
    apply_output_tks_to_change,
    change_tks_to_query_context,
)
from coeditor.encoding import (
    Add_id,
    Del_id,
    change_to_tokens,
    extra_id_to_number,
    get_extra_id,
    inline_output_tokens,
    is_extra_id,
    tokens_to_change,
)
from coeditor.history import (
    Added,
    Modified,
    ModuleEdit,
    ProjectEdit,
    default_show_diff,
    file_content_from_commit,
    get_change_path,
)
from coeditor.model import DecodingArgs
from coeditor.retrieval_model import (
    BatchArgs,
    RetrievalDecodingResult,
    RetrievalEditorModel,
    RetrievalModelPrediction,
)
from spot.data import output_ids_as_seqs
from spot.static_analysis import (
    CommentRemover,
    ModuleName,
    PythonElem,
    PythonFunction,
    PythonModule,
    PythonProject,
    remove_comments,
)
from spot.utils import add_line_numbers

DropComment = bool


@dataclass
class ChangeDetectionConfig:
    untracked_as_additions: bool = True
    ignore_dirs: Collection[str] = field(
        default_factory=lambda: PythonProject.DefaultIgnoreDirs
    )
    drop_comments: DropComment = True

    def get_index_content(self, project: Path, path_s: str):
        return file_content_from_commit(project, "", path_s)

    def get_index_stamp(self, project_root: Path, path_s: str):
        out = run_command(["git", "ls-files", "-s", path_s], cwd=project_root)
        hash = out.split(" ")[1]
        assert_eq(len(hash), 40)
        return hash

    def get_pedit(
        self,
        project_root: Path,
        target_file: Path,
        prev_cache: TimedCache[tuple[ModuleName, DropComment], PythonModule, str],
        now_cache: TimedCache[tuple[ModuleName, DropComment], PythonModule, float],
    ) -> ProjectEdit:
        def is_src(path_s: str) -> bool:
            path = Path(path_s)
            return path.suffix == ".py" and all(
                p not in self.ignore_dirs for p in path.parts
            )

        src_map = dict[ModuleName, Path]()

        def get_module_path(file_s: str) -> ModuleName:
            path = Path(file_s)
            mname = PythonProject.rel_path_to_module_name(Path(path))
            src_map[mname] = path
            return mname

        changed_files = run_command(
            ["git", "status", "--porcelain"], cwd=project_root
        ).splitlines()

        prev_module2file = dict[ModuleName, str]()
        current_module2file = dict[ModuleName, str | None]()

        for line in changed_files:
            if not line:
                continue
            if line[2] == " ":
                tag = line[:2]
                path = line[3:]
                if not is_src(path):
                    continue
                if tag.endswith("M") or tag.endswith("A") or tag == "??":
                    if tag == "??" and not self.untracked_as_additions:
                        continue
                    epath = get_module_path(path)
                    if tag.endswith("M"):
                        prev_module2file[epath] = path
                    current_module2file[epath] = path
            else:
                tag, path1, path2 = line.split(" ")
                assert tag.startswith("R")
                if not is_src(path1) or not is_src(path2):
                    continue
                current_module2file[get_module_path(path2)] = path2
                current_module2file[get_module_path(path1)] = None
                prev_module2file[get_module_path(path1)] = path1

        if target_file.is_absolute():
            target_file = target_file.relative_to(project_root)
        target_mname = get_module_path(target_file.as_posix())
        if target_mname not in prev_module2file:
            prev_module2file[target_mname] = target_file.as_posix()

        prev_modules = dict[ModuleName, PythonModule]()
        for mname, file_prev in prev_module2file.items():
            stamp = self.get_index_stamp(project_root, file_prev)
            prev_modules[mname] = prev_cache.cached(
                (mname, self.drop_comments),
                stamp,
                lambda: PythonModule.from_cst(
                    cst.parse_module(self.get_index_content(project_root, file_prev)),
                    mname,
                    self.drop_comments,
                ),
            )

        now_modules = dict[ModuleName, PythonModule]()
        for mname, file_now in current_module2file.items():
            if file_now is None:
                continue
            path_now = project_root / file_now
            mtime = os.stat(path_now).st_mtime
            (project_root / path_now).read_text()
            now_modules[mname] = now_cache.cached(
                (mname, self.drop_comments),
                mtime,
                lambda: PythonModule.from_cst(
                    cst.parse_module(path_now.read_text()), mname, self.drop_comments
                ),
            )

        prev_project = PythonProject.from_modules(
            project_root.resolve(),
            modules=prev_modules.values(),
            src_map=src_map,
        )

        return ProjectEdit.from_module_changes(prev_project, now_modules)


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


@dataclass
class ServiceResponse:
    target_file: str
    edit_start: tuple[int, int]
    edit_end: tuple[int, int]
    old_code: str
    suggestions: list[EditSuggestion]

    def to_json(self):
        return {
            "target_file": self.target_file,
            "edit_start": self.edit_start,
            "edit_end": self.edit_end,
            "old_code": self.old_code,
            "suggestions": [s.to_json() for s in self.suggestions],
        }

    def print(self, file=sys.stdout):
        print(f"Target file: {self.target_file}", file=file)
        print(f"Edit range: {self.edit_start} - {self.edit_end}", file=file)
        for i, s in enumerate(self.suggestions):
            print(
                f"\t--------------- Suggestion {i} (score: {s.score:.3g}) ---------------",
                file=file,
            )
            print(textwrap.indent(s.change_preview, "\t"), file=file)
        print(f"original code:", file=file)

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
        project: Path,
        model: RetrievalEditorModel,
        batch_args: BatchArgs = BatchArgs(shuffle_extra_ids=False),
        encoder: QueryRefEditEncoder = QueryRefEditEncoder(),
        dec_args: DecodingArgs = DecodingArgs(),
        config: ChangeDetectionConfig = ChangeDetectionConfig(),
    ) -> None:
        self.project = project
        self.model = model
        self.batch_args = batch_args
        self.encoder = encoder
        self.dec_args = dec_args
        self.config = config
        self.show_max_solutions = 3
        self.preview_ctx_lines = 3

        # caches file contents from the index using its sha1 hash
        self.prev_cache = TimedCache[
            tuple[ModuleName, DropComment], PythonModule, str
        ]()
        # caches file contents from the working tree using its mtime
        self.now_cache = TimedCache[
            tuple[ModuleName, DropComment], PythonModule, float
        ]()
        self.stub_cache = TimedCache[ModuleName, list[TokenSeq], int]()
        self.tlogger = model.tlogger

    def suggest_edit(
        self,
        file: Path,
        line: int,
        log_dir: Path | None = Path(".coeditor_logs"),
    ) -> ServiceResponse:
        """Make the suggestion in-place at the given location."""
        timed = self.tlogger.timed
        project = self.project

        if not file.is_absolute():
            file = project / file

        with timed("get target element"):
            mname = PythonProject.rel_path_to_module_name(file.relative_to(project))
            stamp = os.stat(file).st_mtime
            now_code = file.read_text()
            now_mod = self.now_cache.cached(
                (mname, False),
                stamp,
                lambda: PythonModule.from_cst(
                    cst.parse_module(now_code), mname, drop_comments=False
                ),
            )
            now_elem = get_elem_by_line(now_mod, line)
            if now_elem is None:
                raise ValueError(
                    f"No code element found at line {line} in file {file}."
                )
            if not isinstance(now_elem, PythonFunction):
                raise ValueError(f"Only functions can be edited by the model.")

        with timed("construct project edit"):
            pedit = self.config.get_pedit(
                project, file, self.prev_cache, self.now_cache
            )
            if mname not in pedit.changes:
                assert mname in pedit.before.modules
                assert mname in pedit.after.modules
                pedit.changes[mname] = ModuleEdit.from_no_change(
                    pedit.before.modules[mname]
                )
        match [
            c for c in pedit.all_elem_changes() if get_change_path(c) == now_elem.path
        ]:
            case [Modified(PythonFunction(), PythonFunction()) as mf]:
                elem_change = cast(Modified[PythonFunction], mf)
            case [Added(PythonFunction()) as mf]:
                elem_change = cast(Added[PythonFunction], mf)
            case _:
                if self.config.drop_comments:
                    trans_tree = remove_comments(now_elem.tree)
                    trans_elem = PythonFunction(
                        now_elem.name, now_elem.path, now_elem.parent_class, trans_tree
                    )
                else:
                    trans_elem = now_elem
                elem_change = Modified(trans_elem, trans_elem)

        with timed("encode edits"):
            respect_lines = (
                self.compute_offset(now_mod, now_elem, line, drop_comments=True) + 1
            )
            print(f"{respect_lines = }")
            req = EditRequest(elem_change, respect_lines)
            qedits = list(
                self.encoder.encode_pedit(
                    pedit,
                    self.stub_cache,
                    queries=[req],
                    training=False,
                )
            )
            if qedits[0].tk_pedit.module_stubs:
                print("stub files:", qedits[0].tk_pedit.module_stubs.keys())
            assert len(qedits) == 1
            batches = tk_edits_to_batches(qedits, self.batch_args)
            assert len(batches) == 1
            batch = batches[0]

        with timed("run model"), torch.autocast("cuda"):
            predictions = self.model.predict_on_batch(
                batch, [req], self.dec_args, self.show_max_solutions
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
                    print(f"{respect_lines = }", file=f)
                    print(f"{len(input_tks) = }", file=f)
                    print(f"{len(references) = }", file=f)
                    print(f"Solution score: {score:.3g}", file=f)
                    print(f"Marginalized samples:", pred.n_samples, file=f)
                    pred = RetrievalModelPrediction(
                        input_ids=input_tks,
                        output_ids=pred_tks,
                        labels=output_truth,
                        references=references,
                    )
                    pred_str = RetrievalDecodingResult.show_prediction(None, pred)
                    print(pred_str, file=f)

        now_span = now_mod.location_map[now_elem.tree]
        # old_elem_code = get_span(now_code, now_span)
        old_elem_code = now_elem.code
        respect_lines = (
            self.compute_offset(now_mod, now_elem, line, drop_comments=False) + 1
        )

        suggestions = list[EditSuggestion]()
        for pred in predictions:
            suggested_change, preview = self.apply_edit_to_elem(
                file,
                now_mod,
                now_elem,
                line,
                pred.out_tks,
            )
            suggestion = EditSuggestion(
                score=pred.score,
                change_preview=preview,
                new_code=suggested_change.after,
            )
            suggestions.append(suggestion)

        def as_tuple(x: CodePosition):
            return (x.line, x.column)

        return ServiceResponse(
            target_file=file.as_posix(),
            edit_start=(now_span.start.line, 0),
            edit_end=as_tuple(now_span.end),
            old_code=old_elem_code,
            suggestions=suggestions,
        )

    def compute_offset(
        self,
        now_mod: PythonModule,
        now_elem: PythonElem,
        line: int,
        drop_comments: bool,
    ):
        "Compute the relative offset of a given line w.r.t. the beginning of a function."
        start_line = now_mod.location_map[now_elem.tree].start.line
        origin_offset = line - start_line
        if not drop_comments:
            return origin_offset
        else:
            removed_lines = 0
            remover = CommentRemover(now_mod.location_map)
            now_elem.tree.visit(remover)
            for c in remover.removed_lines:
                span = remover.src_map[c]
                if span.end.line < line:
                    removed_lines += span.end.line - span.start.line + 1
            return origin_offset - removed_lines

    def preview_changes(
        self,
        change: Modified[str],
        respect_lines: int,
    ) -> str:
        change_tks = change_to_tokens(change)
        (input_tks, output_tks), _ = change_tks_to_query_context(
            change_tks, respect_lines
        )
        new_change = tokens_to_change(inline_output_tokens(input_tks, output_tks))
        change_str = default_show_diff(new_change.before, new_change.after)
        return change_str

    def apply_edit_to_elem(
        self,
        file: Path,
        now_mod: PythonModule,
        now_elem: PythonElem,
        cursor_line: int,
        out_tks: TokenSeq,
    ) -> tuple[Modified[str], str]:
        mname = now_elem.path.module
        path_s = file.relative_to(self.project).as_posix()
        prev_mod = self.prev_cache.cached(
            (path_s, False),
            self.config.get_index_stamp(self.project, path_s),
            lambda: PythonModule.from_cst(
                cst.parse_module(file_content_from_commit(self.project, "", path_s)),
                mname,
                drop_comments=False,
            ),
        )
        now_code = now_elem.code
        prev_elem = prev_mod.elems_dict.get(now_elem.path.path)
        if prev_elem is None:
            code_change = Added(now_code)
        else:
            code_change = Modified(prev_elem.code, now_code)
        lines_with_comment = (
            self.compute_offset(now_mod, now_elem, cursor_line, drop_comments=False) + 1
        )
        logging.info("Now respect lines:", lines_with_comment)
        if self.config.drop_comments:
            # map the changes to the original code locations with comments
            remover = CommentRemover(prev_mod.location_map)
            if prev_elem is not None:
                elem1 = prev_elem.tree.visit(remover)
                assert isinstance(elem1, cst.CSTNode)
                line_map = remover.line_map(elem1)
            else:
                line_map = {0: 0, 1: 1}
            n_lines = len(line_map)
            line_map[n_lines] = line_map[n_lines - 1] + 1
            lines_no_comment = (
                self.compute_offset(now_mod, now_elem, cursor_line, drop_comments=True)
                + 1
            )
            new_out_tks = TokenSeq()
            for k, seg in output_ids_as_seqs(out_tks).items():
                rel_line = extra_id_to_number(k) + lines_no_comment
                if rel_line not in line_map:
                    messages = []
                    messages.append(
                        f"predicted relative line {rel_line} (extra_id_{extra_id_to_number(k)}) is out of range."
                    )
                    messages.append(
                        f"{n_lines = }, {lines_no_comment = }, {lines_with_comment = }"
                    )
                    messages.append(f"{line_map = }")
                    if isinstance(code_change, Modified):
                        messages.append("Prev element:")
                        messages.append(add_line_numbers(code_change.before))
                    e = ValueError("\n".join(messages))
                    raise e
                line = line_map[rel_line]
                k1 = get_extra_id(line - lines_with_comment)
                new_out_tks.append(k1)
                new_out_tks.extend(seg)
            out_tks = new_out_tks

        change_tks = change_to_tokens(code_change)
        new_change = apply_output_tks_to_change(change_tks, lines_with_comment, out_tks)
        new_change = new_change.map(lambda s: s.strip("\n"))
        preview = self.preview_changes(new_change, lines_with_comment)
        return new_change, preview


def replace_lines(text: str, span: CodeRange, replacement: str):
    start_ln, end_ln = span.start.line - 1, span.end.line
    replacemnet = textwrap.indent(textwrap.dedent(replacement), " " * span.start.column)
    old_lines = text.split("\n")
    new_lines = old_lines[:start_ln] + [replacemnet] + old_lines[end_ln + 1 :]
    return "\n".join(new_lines)


def get_span(text: str, span: CodeRange):
    start_ln, end_ln = span.start.line - 1, span.end.line
    old_lines = text.split("\n")
    new_lines = old_lines[start_ln : end_ln + 1]
    new_lines[0] = new_lines[0][span.start.column :]
    new_lines[-1] = new_lines[-1][: span.end.column]
    return "\n".join(new_lines)


def get_elem_by_line(module: PythonModule, line: int) -> PythonElem | None:
    def in_span(line: int, span: CodeRange):
        return span.start.line <= line <= span.end.line

    for e in module.all_elements():
        span = module.location_map[e.tree]
        if in_span(line, span):
            return e
    return None


def show_location(loc: CodePosition):
    return f"{loc.line}:{loc.column}"


def split_label_by_post_edit_line(
    label_tks: TokenSeq, post_line: int
) -> tuple[TokenSeq, TokenSeq]:
    """Split the label into two parts: before and after the given
    post-edit line."""
    assert label_tks, "label should not be empty."
    line_counter = -1
    split_pos = 0
    for pos, tk in enumerate(label_tks):
        if is_extra_id(tk):
            line_counter += 1
        elif tk == Add_id:
            line_counter += 1
        elif tk == Del_id:
            line_counter -= 1
        if line_counter <= post_line:
            split_pos = pos
    return label_tks[:split_pos], label_tks[split_pos:]
