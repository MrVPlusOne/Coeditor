# End-user API as an editing suggestion tool.

import torch
from coeditor.common import *
from libcst.metadata import CodePosition, CodeRange
from coeditor.encoders import BasicQueryEditEncoder
from coeditor.encoding import decode_tokens, extract_edit_change

from coeditor.history import (
    Modified,
    ProjectEdit,
    file_content_from_commit,
    get_change_path,
    get_commit_history,
    parse_cst_module,
    show_change,
)
from coeditor.model import DecodingArgs
from coeditor.retrieval_model import BatchArgs, RetrievalEditorModel, edits_to_batches
from spot.static_analysis import (
    ModuleName,
    PythonElem,
    PythonFunction,
    PythonModule,
    PythonProject,
)


@dataclass
class ChangeDetectionConfig:
    untracked_as_additions: bool = True
    ignore_dirs: Collection[str] = field(
        default_factory=lambda: PythonProject.DefaultIgnoreDirs
    )
    prev_commit: str = "HEAD"
    src2module: Callable[[str], cst.Module] = parse_cst_module

    def get_pedit(self, project_root: Path) -> ProjectEdit:
        def is_src(path_s: str) -> bool:
            path = Path(path_s)
            return path.suffix == ".py" and all(
                p not in self.ignore_dirs for p in path.parts
            )

        def get_content(path_s: str):
            return (project_root / path_s).read_text()

        def get_prev_content(path_s: str):
            return file_content_from_commit(project_root, self.prev_commit, path_s)

        changed_files = run_command(
            ["git", "status", "--porcelain"], cwd=project_root
        ).splitlines()

        prev_modules = dict[ModuleName, str]()
        current_modules = dict[ModuleName, str | None]()
        src_map = dict[ModuleName, Path]()

        def get_module_path(file_s: str) -> ModuleName:
            path = Path(file_s)
            mname = PythonProject.rel_path_to_module_name(Path(path))
            src_map[mname] = path
            return mname

        for line in changed_files:
            segs = line.strip().split(" ")
            match segs:
                case ["D", path] if is_src(path):
                    epath = get_module_path(path)
                    prev_modules[epath] = get_prev_content(path)
                    current_modules[epath] = None
                case [("M" | "A" | "??") as tag, path] if is_src(path):
                    if tag == "??" and not self.untracked_as_additions:
                        continue
                    epath = get_module_path(path)
                    if tag == "M":
                        prev_modules[epath] = get_prev_content(path)
                    current_modules[epath] = get_content(path)
                case [tag, path1, path2] if (
                    tag.startswith("R") and is_src(path1) and is_src(path2)
                ):
                    current_modules[get_module_path(path2)] = get_content(path2)
                    current_modules[get_module_path(path1)] = None
                    prev_modules[get_module_path(path1)] = get_prev_content(path1)

        prev_project = PythonProject.from_modules(
            project_root.resolve(),
            modules=[
                PythonModule.from_cst(self.src2module(v), k)
                for k, v in prev_modules.items()
            ],
            src_map=src_map,
        )

        return ProjectEdit.from_code_changes(
            prev_project,
            current_modules,
            src2module=self.src2module,
        )


@dataclass
class EditPredictionService:
    def __init__(
        self,
        model: RetrievalEditorModel,
        batch_args: BatchArgs = BatchArgs(max_ref_dropout=0.0, shuffle_extra_ids=False),
        encoder: BasicQueryEditEncoder = BasicQueryEditEncoder(),
        dec_args: DecodingArgs = DecodingArgs(),
        config: ChangeDetectionConfig = ChangeDetectionConfig(),
    ) -> None:
        self.model = model
        self.batch_args = batch_args
        self.model = model
        self.encoder = encoder
        self.dec_args = dec_args
        self.config = config

        self.tlogger = TimeLogger()

    def suggest_edit(
        self,
        project: Path,
        file: Path,
        loc: CodePosition | tuple[int, int],
        log_file: Path | None = Path("coeditor-log.txt"),
    ) -> None:
        """Make the suggestion in-place at the given location."""
        timed = self.tlogger.timed

        if isinstance(loc, tuple):
            loc = CodePosition(*loc)
        if not file.is_absolute():
            file = project / file

        with timed("get target element"):
            mname = PythonProject.rel_path_to_module_name(file.relative_to(project))
            mod = PythonModule.from_cst(cst.parse_module(file.read_text()), mname)
            elem = get_elem_by_location(mod, loc)
        if elem is None:
            raise ValueError(
                f"No code element found at {show_location(loc)} in file {file}."
            )
        if not isinstance(elem, PythonFunction):
            raise ValueError(f"Only functions can be edited")

        with timed("construct project edit"):
            pedit = self.config.get_pedit(project)
        old_elems = [
            c for c in pedit.all_elem_changes() if get_change_path(c) == elem.path
        ]
        match old_elems:
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
            print(f"{len(input_tks) = }")
            print(f"{len(references) = }")
            out_tks = self.model.generate(
                self.model.encode_token_seqs([input_tks]),
                references=references,
                query_ref_list=batch["query_ref_list"],
                **dec_args,
            )[0].tolist()
            out_tks = cast(TokenSeq, out_tks)
            pred_change = extract_edit_change(input_tks, out_tks)
            print("=" * 10, "Predicted code change", "=" * 10)
            print(show_change(pred_change))

        if log_file is not None:
            with log_file.open("w") as f:
                print("=" * 10, "Input code", "=" * 10, file=f)
                print(decode_tokens(input_tks), file=f)
                for i, ref in enumerate(references):
                    print(f"References {i}:", file=f)
                    print(decode_tokens(ref), file=f)


def get_elem_by_location(module: PythonModule, loc: CodePosition) -> PythonElem | None:
    def to_tuple(pos: CodePosition):
        return pos.line, pos.column

    def in_span(loc: CodePosition, span: CodeRange):
        return to_tuple(span.start) <= to_tuple(loc) < to_tuple(span.end)

    for e in module.all_elements():
        span = module.location_map[e.tree]
        if in_span(loc, span):
            return e
    return None


def show_location(loc: CodePosition):
    return f"{loc.line}:{loc.column}"
