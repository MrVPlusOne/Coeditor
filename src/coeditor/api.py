# End-user API as an editing suggestion tool.

import copy
import logging
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
    is_extra_id,
)

from coeditor.history import (
    Added,
    Modified,
    ModuleEdit,
    ProjectEdit,
    file_content_from_commit,
    get_change_path,
    get_commit_history,
    parse_cst_module,
    show_change,
)
from coeditor.model import CoeditorModel, DecodingArgs
from coeditor.retrieval_model import (
    BatchArgs,
    RetrievalEditorModel,
    edit_groups_to_batches,
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

    def get_pedit(
        self,
        project_root: Path,
        prev_cache: TimedCache[ModuleName, PythonModule, str],
        now_cache: TimedCache,
    ) -> ProjectEdit:
        def is_src(path_s: str) -> bool:
            path = Path(path_s)
            return path.suffix == ".py" and all(
                p not in self.ignore_dirs for p in path.parts
            )

        def get_prev_content(path_s: str):
            return file_content_from_commit(project_root, self.prev_commit, path_s)

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

        prev_modules = dict[ModuleName, PythonModule]()
        for mname, file_prev in prev_module2file.items():
            prev_modules[mname] = prev_cache.cached(
                mname,
                commit_stamp,
                lambda: PythonModule.from_cst(
                    cst.parse_module(get_prev_content(file_prev)),
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
        self.model = model
        self.encoder = encoder
        self.dec_args = dec_args
        self.config = config
        self.show_max_solutions = 3

        self.prev_cache = TimedCache[ModuleName, PythonModule, str]()
        self.now_cache = TimedCache[ModuleName, PythonModule, float]()
        self.parse_cache = TimedCache[ModuleName, PythonModule, float]()
        self.prev_parse_cache = TimedCache[ModuleName, PythonModule, str]()
        self.stub_cache = TimedCache[ModuleName, list[TokenSeq], int]()
        self.tlogger = TimeLogger()

    def suggest_edit(
        self,
        file: Path,
        line: int,
        log_file: Path | None = Path("coeditor-log.txt"),
        apply_edit: bool = False,
    ) -> bool | None:
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
            pedit = self.config.get_pedit(project, self.prev_cache, self.now_cache)
            now_trans_mod = self.now_cache.cached(
                mname,
                stamp,
                lambda: PythonModule.from_cst(
                    now_mod.tree, mname, self.config.drop_comments
                ),
            )
            if mname not in pedit.after.modules:
                pedit.after.modules[mname] = now_trans_mod
                pedit.changes[mname] = ModuleEdit.from_no_change(now_trans_mod)
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
            batches = edit_groups_to_batches([qedits], self.batch_args)
            assert len(batches) == 1
            batch = batches[0]

        with timed("run model"), torch.autocast("cuda"):
            dec_args = self.dec_args.to_model_args()
            input_tks = batch["input_ids"][0]
            references = batch["references"]
            output_truth = batch["labels"][0]
            gen_out = self.model.generate(
                self.model.encode_token_seqs([input_tks]),
                references=references,
                query_ref_list=batch["query_ref_list"],
                output_scores=True,
                return_dict_in_generate=True,
                num_return_sequences=self.dec_args.num_beams,
                **dec_args,
            )
            assert not isinstance(gen_out, torch.LongTensor)
        for i in range(gen_out.sequences.size(0))[: self.show_max_solutions]:
            out_tks = gen_out.sequences[i].tolist()
            pred_change = extract_edit_change(input_tks, out_tks)
            print("=" * 10, f"Sugeestion {i}", "=" * 10)
            if (scores := getattr(gen_out, "sequences_scores", None)) is not None:
                print(f"score: {scores[i].item():.4g}")
            print(show_change(pred_change))

        out_tks = gen_out.sequences[0].tolist()
        changed = None
        if apply_edit:
            new_elem_code = self.apply_edit_to_elem(
                file,
                now_elem,
                self.compute_offset(now_mod, now_elem, line, drop_comments=False) + 1,
                out_tks,
            )
            now_span = now_mod.location_map[now_elem.tree]
            new_code = replace_lines(now_code, now_span, new_elem_code)
            changed = new_code != now_code
            if changed:
                file.write_text(new_code)
                print("Edit applied to source.")

        if log_file is None:
            return changed
        header = lambda s: "=" * 10 + s + "=" * 10
        indent = lambda s: textwrap.indent(s, "    ")
        with log_file.open("w") as f:
            print(f"{respect_lines = }", file=f)
            print(f"{len(input_tks) = }", file=f)
            print(f"{len(references) = }", file=f)
            print(header("Ground truth"), file=f)
            print(indent(decode_tokens(output_truth)), file=f)
            print(header("Predicted"), file=f)
            print(indent(decode_tokens(out_tks)), file=f)
            print(header("Input"), file=f)
            print(indent(decode_tokens(input_tks)), file=f)
            print(header("References"), file=f)
            for i, ref in enumerate(references):
                print("-" * 6 + f"Reference {i}" + "-" * 6, file=f)
                print(indent(decode_tokens(ref)), file=f)
        return changed

    def compute_offset(
        self,
        now_mod: PythonModule,
        now_elem: PythonFunction,
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

    def apply_edit_to_elem(
        self,
        file: Path,
        now_elem: PythonElem,
        respect_lines: int,
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
        logging.info("Now respect lines:", respect_lines)
        change_tks = change_to_tokens(code_change)
        new_change = apply_output_tks_to_change(change_tks, respect_lines, out_tks)
        return new_change.after


def replace_lines(text: str, span: CodeRange, replacement: str):
    start_ln, end_ln = span.start.line - 1, span.end.line
    replacemnet = textwrap.indent(textwrap.dedent(replacement), " " * span.start.column)
    old_lines = text.split("\n")
    new_lines = old_lines[:start_ln] + [replacemnet] + old_lines[end_ln + 1 :]
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
