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
    Modified,
    ProjectEdit,
    file_content_from_commit,
    get_change_path,
    get_commit_history,
    parse_cst_module,
    show_change,
)
from coeditor.model import CoeditorModel, DecodingArgs
from coeditor.retrieval_model import BatchArgs, RetrievalEditorModel, edits_to_batches
from spot.static_analysis import (
    ModuleName,
    PythonElem,
    PythonFunction,
    PythonModule,
    PythonProject,
    remove_comments,
)


class TimedCache(Generic[T1, T2]):
    """Store the time-stamped results to avoid recomputation."""

    def __init__(self) -> None:
        self.cache = dict[T1, tuple[str, T2]]()

    def get(self, key: T1, stamp: str) -> T2 | None:
        match self.cache.get(key):
            case None:
                return None
            case (s, value) if stamp == s:
                return value

    def set(self, key: T1, value: T2, stamp: str) -> None:
        self.cache.pop(key, None)
        self.cache[key] = (stamp, value)


@dataclass
class ChangeDetectionConfig:
    untracked_as_additions: bool = True
    ignore_dirs: Collection[str] = field(
        default_factory=lambda: PythonProject.DefaultIgnoreDirs
    )
    prev_commit: str = "HEAD"
    module_preprocess: Callable[[cst.Module], cst.Module] = remove_comments

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
        for mname, path in prev_module2file.items():
            if (m := prev_cache.get(mname, commit_stamp)) is None:
                cst_m = self.src2module(get_prev_content(path))
                m = PythonModule.from_cst(cst_m, mname)
            prev_cache.set(mname, m, commit_stamp)
            prev_modules[mname] = m

        now_modules = dict[ModuleName, PythonModule]()
        for mname, path in current_module2file.items():
            if path is None:
                continue
            path = project_root / path
            mtime = str(os.stat(path).st_mtime)
            (project_root / path).read_text()
            if (m := now_cache.get(mname, mtime)) is None:
                cst_m = self.src2module(path.read_text())
                m = PythonModule.from_cst(cst_m, mname)
            now_cache.set(mname, m, mtime)
            now_modules[mname] = m

        prev_project = PythonProject.from_modules(
            project_root.resolve(),
            modules=prev_modules.values(),
            src_map=src_map,
        )

        return ProjectEdit.from_module_changes(prev_project, now_modules)

    def src2module(self, src: str) -> cst.Module:
        return self.module_preprocess(cst.parse_module(src))


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

        self.prev_cache = TimedCache()
        self.now_cache = TimedCache()
        self.parse_cache = TimedCache()
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
            if (
                mod := self.parse_cache.get(mname, str(os.stat(file).st_mtime))
            ) is None:
                mod = PythonModule.from_cst(cst.parse_module(file.read_text()), mname)
            elem = get_elem_by_line(mod, line)
            if elem is None:
                raise ValueError(
                    f"No code element found at line {line} in file {file}."
                )
            if not isinstance(elem, PythonFunction):
                raise ValueError(f"Only functions can be edited by the model.")
        cursor_offset = line - mod.location_map[elem.tree.body].start.line

        with timed("construct project edit"):
            pedit = self.config.get_pedit(project, self.prev_cache, self.now_cache)
            if mname not in pedit.after.modules:
                pedit.after.modules[mname] = PythonModule.from_cst(
                    self.config.module_preprocess(mod.tree), mname
                )
        match [c for c in pedit.all_elem_changes() if get_change_path(c) == elem.path]:
            case [Modified(PythonFunction(), PythonFunction()) as mf]:
                elem_change = cast(Modified[PythonFunction], mf)
            case _:
                elem_change = Modified(elem, elem)
        with timed("encode edits"):
            qedits = list(self.encoder.encode_pedit(pedit, queries=[elem_change]))
            assert len(qedits) == 1
            batches = edits_to_batches([qedits], self.batch_args)
            assert len(batches) == 1
            batch = batches[0]

        with timed("run model"), torch.autocast("cuda"):
            dec_args = {
                "max_length": self.dec_args.max_output_tks,
                "do_sample": self.dec_args.do_sample,
                "top_p": self.dec_args.top_p,
                "num_beams": self.dec_args.num_beams,
                "length_penalty": self.dec_args.length_penalty,
            }
            input_tks = batch["input_ids"][0]
            references = batch["references"]
            output_prefix, _ = split_label_by_post_edit_line(
                batch["labels"][0], cursor_offset
            )
            out_tks = self.model.generate(
                self.model.encode_token_seqs([input_tks]),
                references=references,
                query_ref_list=batch["query_ref_list"],
                prefix_allowed_tokens_fn=(
                    CoeditorModel._prefix_constraint([output_prefix])
                    if output_prefix
                    else None
                ),
                **dec_args,
            )[0].tolist()
            out_tks = cast(TokenSeq, out_tks)
            pred_change = extract_edit_change(input_tks, out_tks)
            print("=" * 10, "Predicted code change", "=" * 10)
            print(show_change(pred_change))

        if log_file is not None:
            with log_file.open("w") as f:
                print(f"{len(input_tks) = }")
                print(f"{len(references) = }")
                print("User prefix:", decode_tokens(output_prefix), file=f)
                assert (
                    not self.batch_args.shuffle_extra_ids
                ), "Ids cannot be shuffled for this to work for now"
                print(qedits[0].show_prediction(out_tks), file=f)


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
    split_pos = 0 if post_line < 0 else len(label_tks)
    for pos, tk in enumerate(label_tks):
        if is_extra_id(tk):
            line_counter += 1
        elif tk == Add_id:
            line_counter += 1
        elif tk == Del_id:
            line_counter += 1
        if line_counter == post_line + 1:
            split_pos = pos
    return label_tks[:split_pos], label_tks[split_pos:]
