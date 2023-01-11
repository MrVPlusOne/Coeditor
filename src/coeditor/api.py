# End-user API as an editing suggestion tool.

import copy
import io
import logging
import sys
import torch
from coeditor.common import *
from libcst.metadata import CodePosition, CodeRange
from coeditor.encoders import (
    QueryRefEditEncoder,
    EditRequest,
    apply_output_tks_to_change,
    change_tks_to_query_context,
)
from coeditor.encoding import (
    Add_id,
    Del_id,
    Newline_id,
    change_to_tokens,
    decode_tokens,
    extra_id_to_number,
    extract_edit_change,
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
    get_commit_history,
    parse_cst_module,
    show_change,
)
from coeditor.model import CoeditorModel, DecodingArgs
from coeditor.retrieval_model import (
    BatchArgs,
    RetrievalDecodingResult,
    RetrievalEditorModel,
    RetrievalModelPrediction,
    query_edits_to_batches,
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

import textwrap


@dataclass
class ChangeDetectionConfig:
    untracked_as_additions: bool = True
    ignore_dirs: Collection[str] = field(
        default_factory=lambda: PythonProject.DefaultIgnoreDirs
    )
    prev_commit: str = "HEAD"
    drop_comments: bool = True

    def get_prev_content(self, project: Path, path_s: str):
        return file_content_from_commit(project, self.prev_commit, path_s)

    def get_pedit(
        self,
        project_root: Path,
        target_file: Path,
        prev_cache: TimedCache[ModuleName, PythonModule, str],
        now_cache: TimedCache,
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

        # get the previous commit timestamp
        commit_stamp = run_command(
            ["git", "log", "-1", "--format=%ci"], cwd=project_root
        )
        assert (
            self.prev_commit == "HEAD"
        ), "Currently only prev_commit=HEAD is supported."

        changed_files = run_command(
            ["git", "status", "--porcelain"], cwd=project_root
        ).splitlines()

        prev_module2file = dict[ModuleName, str]()
        current_module2file = dict[ModuleName, str | None]()

        for line in changed_files:
            segs = line.strip().split(" ")
            match segs:
                case ["D", path] if is_src(path):
                    epath = get_module_path(path)
                    prev_module2file[epath] = path
                    current_module2file[epath] = None
                case [("M" | "A" | "??") as tag, path] if is_src(path):
                    if tag == "??" and not self.untracked_as_additions:
                        continue
                    epath = get_module_path(path)
                    if tag == "M":
                        prev_module2file[epath] = path
                    current_module2file[epath] = path
                case [tag, path1, path2] if (
                    tag.startswith("R") and is_src(path1) and is_src(path2)
                ):
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
            prev_modules[mname] = prev_cache.cached(
                mname,
                commit_stamp,
                lambda: PythonModule.from_cst(
                    cst.parse_module(self.get_prev_content(project_root, file_prev)),
                    mname,
                    self.drop_comments,
                ),
            )

        now_modules = dict[ModuleName, PythonModule]()
        for mname, file_now in current_module2file.items():
            if file_now is None:
                continue
            path_now = project_root / file_now
            mtime = str(os.stat(path_now).st_mtime)
            (project_root / path_now).read_text()
            now_modules[mname] = now_cache.cached(
                mname,
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
                f"\t--------------- Suggestion {i} (score: {s.score:.2f}) ---------------",
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
        batch_args: BatchArgs = BatchArgs(max_ref_dropout=0.0, shuffle_extra_ids=False),
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

        self.prev_cache = TimedCache[ModuleName, PythonModule, str]()
        self.now_cache = TimedCache[ModuleName, PythonModule, float]()
        self.parse_cache = TimedCache[ModuleName, PythonModule, float]()
        self.prev_parse_cache = TimedCache[ModuleName, PythonModule, str]()
        self.stub_cache = TimedCache[ModuleName, list[TokenSeq], int]()
        self.tlogger = model.tlogger

    def suggest_edit(
        self,
        file: Path,
        line: int,
        log_file: Path | None = Path("coeditor-log.txt"),
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
            now_mod = self.parse_cache.cached(
                mname,
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
                trans_elem = copy.copy(now_elem)
                if self.config.drop_comments:
                    trans_elem.tree = remove_comments(trans_elem.tree)
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
            batches = query_edits_to_batches(qedits, self.batch_args)
            assert len(batches) == 1
            batch = batches[0]

        with timed("run model"), torch.autocast("cuda"):
            predictions = self.model.predict_on_batch(
                batch, [req], self.dec_args, self.show_max_solutions
            )
            assert_eq(len(predictions), 1)
            predictions = predictions[0]
            assert predictions

        # for i, (pred_change, _, score) in enumerate(predictions):
        #     print("=" * 10, f"Sugeestion {i}", "=" * 10)
        #     print(f"score: {score:.4g}")
        #     print(show_change(pred_change))

        best_output = predictions[0].out_tks

        if log_file is not None:
            with log_file.open("w") as f:
                input_tks = batch["input_ids"][0]
                references = batch["references"]
                output_truth = batch["labels"][0]
                print(f"{respect_lines = }", file=f)
                print(f"{len(input_tks) = }", file=f)
                print(f"{len(references) = }", file=f)
                pred = RetrievalModelPrediction(
                    input_ids=input_tks,
                    output_ids=best_output,
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
            new_elem_code = self.apply_edit_to_elem(
                file,
                now_mod,
                now_elem,
                line,
                pred.out_tks,
            )
            preview = self.preview_changes(
                Modified(old_elem_code, new_elem_code), respect_lines
            )
            suggestion = EditSuggestion(
                score=pred.score,
                change_preview=preview,
                new_code=new_elem_code,
            )
            suggestions.append(suggestion)

        def as_tuple(x: CodePosition):
            return (x.line, x.column)

        return ServiceResponse(
            target_file=file.relative_to(project).as_posix(),
            edit_start=as_tuple(now_span.start),
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

        ctx_start = max(0, respect_lines - self.preview_ctx_lines)
        ctx_code = "\n".join(change.before.split("\n")[ctx_start:respect_lines])
        if ctx_code:
            ctx_code = textwrap.indent(ctx_code, "  ") + "\nfocus>\n"
        new_change = tokens_to_change(inline_output_tokens(input_tks, output_tks))
        change_str = default_show_diff(new_change.before, new_change.after)
        return ctx_code + change_str

    def apply_edit_to_elem(
        self,
        file: Path,
        now_mod: PythonModule,
        now_elem: PythonElem,
        cursor_line: int,
        out_tks: TokenSeq,
    ) -> str:
        mname = now_elem.path.module
        path_s = file.relative_to(self.project).as_posix()
        prev_mod = self.prev_parse_cache.cached(
            path_s,
            self.config.prev_commit,
            lambda: PythonModule.from_cst(
                cst.parse_module(
                    file_content_from_commit(
                        self.project, self.config.prev_commit, path_s
                    )
                ),
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
            remover = CommentRemover(now_mod.location_map)
            elem1 = now_elem.tree.visit(remover)
            assert isinstance(elem1, cst.CSTNode)
            line_map = remover.line_map(elem1)
            n_lines = len(line_map)
            line_map[n_lines] = line_map[n_lines - 1] + 1
            lines_no_comment = (
                self.compute_offset(now_mod, now_elem, cursor_line, drop_comments=True)
                + 1
            )
            new_out_tks = TokenSeq()
            for k, seg in output_ids_as_seqs(out_tks).items():
                line = line_map[extra_id_to_number(k) + lines_no_comment]
                k1 = get_extra_id(line - lines_with_comment)
                new_out_tks.append(k1)
                new_out_tks.extend(seg)
            out_tks = new_out_tks

        change_tks = change_to_tokens(code_change)
        new_change = apply_output_tks_to_change(change_tks, lines_with_comment, out_tks)
        return new_change.after.strip("\n")


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
