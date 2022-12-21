# End-user API as an editing suggestion tool.

import torch
from coeditor.common import *
from libcst.metadata import CodePosition, CodeRange
from coeditor.encoders import BasicQueryEditEncoder
from coeditor.encoding import (
    Add_id,
    Del_id,
    decode_tokens,
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
from spot.static_analysis import (
    CommentRemover,
    ModuleName,
    PythonElem,
    PythonFunction,
    PythonModule,
    PythonProject,
    remove_comments,
)
from transformers.generation_utils import (
    GreedySearchOutput,
    BeamSearchOutput,
    BeamSearchEncoderDecoderOutput,
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
        self, project_root: Path, prev_cache: TimedCache, now_cache: TimedCache
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
        encoder: BasicQueryEditEncoder = BasicQueryEditEncoder(),
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

        self.prev_cache = TimedCache()
        self.now_cache = TimedCache()
        self.parse_cache = TimedCache()
        self.stub_cache = TimedCache()
        self.tlogger = TimeLogger()

    def suggest_edit(
        self,
        file: Path,
        line: int,
        log_file: Path | None = Path("coeditor-log.txt"),
    ) -> None:
        """Make the suggestion in-place at the given location."""
        timed = self.tlogger.timed
        project = self.project

        if not file.is_absolute():
            file = project / file

        with timed("get target element"):
            mname = PythonProject.rel_path_to_module_name(file.relative_to(project))
            stamp = str(os.stat(file).st_mtime)
            mod = self.parse_cache.cached(
                mname,
                stamp,
                lambda: PythonModule.from_cst(
                    cst.parse_module(file.read_text()), mname, drop_comments=False
                ),
            )
            elem = get_elem_by_line(mod, line)
            if elem is None:
                raise ValueError(
                    f"No code element found at line {line} in file {file}."
                )
            if not isinstance(elem, PythonFunction):
                raise ValueError(f"Only functions can be edited by the model.")
        cursor_offset = self.compute_offset(mod, elem, line)

        with timed("construct project edit"):
            pedit = self.config.get_pedit(project, self.prev_cache, self.now_cache)
            this_file_changed = mname in pedit.changes
            if mname not in pedit.after.modules:
                this_module = self.now_cache.cached(
                    mname,
                    stamp,
                    lambda: PythonModule.from_cst(
                        mod.tree, mname, self.config.drop_comments
                    ),
                )
                pedit.after.modules[mname] = this_module
                pedit.changes[mname] = ModuleEdit.from_no_change(this_module)
        match [c for c in pedit.all_elem_changes() if get_change_path(c) == elem.path]:
            case [Modified(PythonFunction(), PythonFunction()) as mf]:
                elem_change = cast(Modified[PythonFunction], mf)
            case [Added(PythonFunction()) as mf]:
                elem_change = cast(Added[PythonFunction], mf)
            case _:
                elem_change = Modified(elem, elem)
        with timed("encode edits"):
            qedits = list(
                self.encoder.encode_pedit(pedit, self.stub_cache, queries=[elem_change])
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
            output_prefix, output_truth = split_label_by_post_edit_line(
                batch["labels"][0], cursor_offset
            )
            gen_out = self.model.generate(
                self.model.encode_token_seqs([input_tks]),
                references=references,
                query_ref_list=batch["query_ref_list"],
                prefix_allowed_tokens_fn=(
                    CoeditorModel._prefix_constraint([output_prefix])
                    if output_prefix
                    else None
                ),
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

        if log_file is None:
            return
        header = lambda s: "=" * 10 + s + "=" * 10
        indent = lambda s: textwrap.indent(s, "    ")
        with log_file.open("w") as f:
            print(f"{cursor_offset = }", file=f)
            print(f"{len(input_tks) = }", file=f)
            print(f"{len(references) = }", file=f)
            print(header("User prefix"), file=f)
            print(indent(decode_tokens(output_prefix)), file=f)
            print(header("Ground truth"), file=f)
            print(indent(decode_tokens(output_truth)), file=f)
            print(header("Predicted"), file=f)
            out_tks = gen_out.sequences[0].tolist()
            print(indent(decode_tokens(out_tks[len(output_prefix) + 1 :])), file=f)
            print(header("Input"), file=f)
            print(indent(decode_tokens(input_tks)), file=f)
            print(header("References"), file=f)
            for i, ref in enumerate(references):
                print("-" * 6 + f"Reference {i}" + "-" * 6, file=f)
                print(indent(decode_tokens(ref)), file=f)

    def compute_offset(self, mod: PythonModule, elem: PythonFunction, line: int):
        "Compute the relative offset of a given line in the body of a function."
        origin_offset = line - mod.location_map[elem.tree.body].start.line + 1
        if not self.config.drop_comments:
            return origin_offset
        else:
            removed_lines = 0
            remover = CommentRemover(mod.location_map)
            elem.tree.visit(remover)
            for c in remover.removed:
                span = remover.src_map[c]
                if span.end.line < line:
                    removed_lines += span.end.line - span.start.line + 1
            return origin_offset - removed_lines


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
