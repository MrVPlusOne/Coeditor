import logging
from coeditor.encoding import (
    Add_id,
    BOS_id,
    CtxEncoder,
    Del_id,
    EOS_id,
    EditEncoder,
    Newline_id,
    TokenizedEdit,
    TruncateAt,
    break_into_chunks,
    change_to_input_output,
    change_to_tokens,
    collapse_code,
    encode_basic,
    truncate_output_tks,
    truncate_section,
)
from spot.static_analysis import ProjectPath, PythonElem
from .history import (
    Added,
    Change,
    Deleted,
    Modified,
    ProjectEdit,
    get_change_path,
    parse_cst_module,
)
from .common import *

TQueryEdit = TypeVar("TQueryEdit")


@dataclass
class TkProjectEdit(Generic[TQueryEdit]):
    """
    Args:
    - `tk_references`:
    """

    tk_references: Mapping[ProjectPath, Sequence[TokenSeq]]
    qedits: Mapping[ProjectPath, TQueryEdit]

    @property
    def stats(self) -> Mapping[str, int]:
        return {
            "references": len(self.tk_references),
            "ref_size_sum": sum(
                len(tks) for segs in self.tk_references.values() for tks in segs
            ),
        }


@dataclass
class BasicTkQueryEdit(TokenizedEdit):
    input_tks: TokenSeq
    output_tks: TokenSeq
    path: ProjectPath
    change_type: Change[None]
    tk_pedit: TkProjectEdit["BasicTkQueryEdit"]

    @property
    def main_tks(self):
        return self.input_tks

    def show(self) -> str:
        return self.show_prediction(None)

    def all_ctxs(self) -> dict[str, TokenSeq]:
        return {
            str(path) + (f" ({i})" if len(segs) > 1 else ""): seg
            for path, segs in self.tk_pedit.tk_references.items()
            if path != self.path
            for i, seg in enumerate(segs)
        }

    def meta_data_lines(self) -> list[str]:
        return [
            f"n_references: {len(self.tk_pedit.tk_references)}",
            f"n_ref_blocks: {sum(len(segs) for segs in self.tk_pedit.tk_references.values())}",
        ]

    def stats(self) -> Mapping[str, int | float]:
        return {
            "input_tks": len(self.input_tks),
            "output_tks": len(self.output_tks),
        } | self.tk_pedit.stats


@dataclass
class BasicQueryEditEncoder(EditEncoder[BasicTkQueryEdit]):
    "Only use changed elements in a commit as references."
    VERSION=1
    max_ref_tks: int = 256
    ref_block_overlap: int = 32
    max_query_tks: int = 512
    max_output_tks: int = 256
    add_truncate_bos: bool = False
    collapse_unchanged: bool = True

    def encode_pedit(
        self,
        pedit: ProjectEdit,
        include_additions: bool = False,
    ) -> Iterable[BasicTkQueryEdit]:
        ctx_enc = CtxEncoder(pedit, self.collapse_unchanged)
        moved = find_moved(pedit.all_elem_changes())
        moved_paths = {a for a, b in moved} | {b for a, b in moved}
        tk_refs = {
            get_change_path(c): self.encode_elem_change(c)
            for c in pedit.all_elem_changes()
            if get_change_path(c) not in moved_paths
        }
        for (d, a), code in moved.items():
            tk_refs[d] = self.encode_elem_move(d, a, code)

        query_data = dict[ProjectPath, BasicTkQueryEdit]()
        tk_pedit = TkProjectEdit(tk_references=tk_refs, qedits=query_data)
        for mf in pedit.modified_functions(ast_must_change=True, body_must_change=True):
            body_change = mf.map(lambda x: x.header_body_code[1])
            if count_lines(body_change.before) > 99:
                continue  # skip large functions
            input_tks, output_tks = change_to_input_output(body_change)
            path = get_change_path(cast(Change, mf))
            path_tks = encode_basic(f"# path: {path}")
            header_tks = change_to_tokens(mf.map(lambda x: x.header_body_code[0]))
            cls_tks = tuple()
            if (cls_p := mf.after.parent_class) is not None:
                cls_tks = (ctx_enc.encode_ctx_element(cls_p),)
            input_tks = join_list(
                (path_tks, *cls_tks, header_tks, input_tks), sep=Newline_id
            )
            input_tks = truncate_section(
                input_tks,
                TruncateAt.Right,
                self.max_query_tks,
                add_bos=self.add_truncate_bos,
            )
            output_tks = truncate_output_tks(input_tks, output_tks)
            output_tks = truncate_section(
                output_tks,
                TruncateAt.Right,
                self.max_output_tks,
                add_bos=self.add_truncate_bos,
            )
            if not output_tks:
                # can happen if input too long
                continue
            query_data[path] = BasicTkQueryEdit(
                input_tks=input_tks,
                output_tks=output_tks,
                path=path,
                change_type=mf.map(lambda _: None),
                tk_pedit=tk_pedit,
            )
        if query_data:
            yield from query_data.values()

    def encode_elem_change(self, c: Change[PythonElem]) -> list[TokenSeq]:
        code_change = c.map(lambda e: f"# {e.path.pop()}\n" + dedent(e.code))
        change_tks = change_to_tokens(code_change)
        all_tks = self.maybe_wrap_bos(change_tks)
        # all_tks = truncate_section(
        #     all_tks, TruncateAt.Right, self.max_ref_tks, add_bos=self.add_truncate_bos
        # )
        chunks = break_into_chunks(
            all_tks,
            self.max_ref_tks,
            overlap=self.ref_block_overlap,
            add_bos=self.add_truncate_bos,
        )
        return [c for c in chunks if has_change(c)]

    def encode_elem_move(
        self, old_path: ProjectPath, new_path: ProjectPath, code: str
    ) -> list[TokenSeq]:
        code = dedent(code)
        if self.collapse_unchanged:
            try:
                mod = collapse_code(parse_cst_module(code))
                assert isinstance(mod, cst.Module)
                code = dedent(mod.code).strip()
            except cst.ParserSyntaxError:
                logging.warn("Unable to parse the following code:\n" + code)
        code_tks = encode_basic(code)
        before_prefix = f"# old: {old_path}\n"
        after_prefix = f"# new: {new_path}\n"
        prefix_tks = change_to_tokens(Modified(before_prefix, after_prefix))
        all_tks = self.maybe_wrap_bos(prefix_tks + code_tks + [Newline_id])
        # all_tks = truncate_section(
        #     all_tks, TruncateAt.Right, self.max_ref_tks, add_bos=self.add_truncate_bos
        # )
        chunks = break_into_chunks(
            all_tks,
            self.max_ref_tks,
            overlap=self.ref_block_overlap,
            add_bos=self.add_truncate_bos,
        )
        return [c for c in chunks if has_change(c)]


def has_change(tks: TokenSeq) -> bool:
    return Add_id in tks or Del_id in tks


def find_moved(
    changes: Iterable[Change[PythonElem]],
):
    added = dict[str, ProjectPath]()
    deleted = dict[str, ProjectPath]()
    moved = dict[tuple[ProjectPath, ProjectPath], str]()
    for c in changes:
        path = get_change_path(c)
        if isinstance(c, Added):
            added[(code := c.after.code)] = path
            if code in deleted:
                moved[(deleted[code], path)] = code
        elif isinstance(c, Deleted):
            deleted[(code := c.before.code)] = path
            if code in added:
                moved[(path, added[code])] = code
    return moved
