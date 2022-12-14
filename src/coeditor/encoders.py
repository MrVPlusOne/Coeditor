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
from spot.static_analysis import ProjectPath, PythonElem, PythonFunction, PythonVariable
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
    VERSION = 2
    max_ref_tks: int = 256
    ref_block_overlap: int = 0
    max_ref_blocks: int = 5
    max_query_tks: int = 512
    max_output_tks: int = 256
    add_truncate_bos: bool = True
    collapse_unchanged: bool = True

    def encode_pedit(
        self,
        pedit: ProjectEdit,
        include_additions: bool = False,
    ) -> Iterable[BasicTkQueryEdit]:
        ctx_enc = CtxEncoder(pedit, self.collapse_unchanged, indent_in_class=False)
        moved = find_moved(pedit.all_elem_changes())
        moved_paths = {a for a, b in moved} | {b for a, b in moved}
        tk_refs = {
            get_change_path(c): self.encode_elem_change(c, ctx_enc)
            for c in pedit.all_elem_changes()
            if get_change_path(c) not in moved_paths
        }
        for (d, a), change in moved.items():
            tk_refs[d] = self.encode_elem_move(d, a, change)

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

    def encode_elem_change(
        self, c: Change[PythonElem], ctx_encoder: CtxEncoder
    ) -> list[TokenSeq]:
        path_tks = change_to_tokens(c.map(lambda e: f"# {e.path.pop()}"))
        path_tks = truncate_section(
            path_tks,
            TruncateAt.Left,
            self.max_ref_tks // 4,
            add_bos=self.add_truncate_bos,
        )
        path_tks.append(Newline_id)
        change_tks = ctx_encoder.encode_ctx_element(get_change_path(c))
        change_tks = self.maybe_wrap_bos(change_tks)
        # all_tks = truncate_section(
        #     all_tks, TruncateAt.Right, self.max_ref_tks, add_bos=self.add_truncate_bos
        # )
        chunks = break_into_chunks(
            change_tks,
            self.max_ref_tks - len(path_tks),
            overlap=self.ref_block_overlap,
            add_bos=self.add_truncate_bos,
        )
        return [path_tks + c for c in chunks if has_change(c)][: self.max_ref_blocks]

    def encode_elem_move(
        self,
        old_path: ProjectPath,
        new_path: ProjectPath,
        change: Modified[PythonElem],
    ) -> list[TokenSeq]:
        def elem2code(e: PythonElem) -> str:
            if self.collapse_unchanged:
                code = show_expr(collapse_code(e.tree), quoted=False)
            else:
                code = e.code
            return dedent(code)

        code_change = change.map(elem2code)
        code_tks = change_to_tokens(code_change)
        before_prefix = f"# old: {old_path}\n"
        after_prefix = f"# new: {new_path}\n"
        prefix_tks = change_to_tokens(Modified(before_prefix, after_prefix))
        prefix_tks = truncate_section(
            prefix_tks,
            TruncateAt.Left,
            self.max_ref_tks // 2,
            add_bos=self.add_truncate_bos,
        )
        # all_tks = self.maybe_wrap_bos(prefix_tks + code_tks + [Newline_id])
        # all_tks = truncate_section(
        #     all_tks, TruncateAt.Right, self.max_ref_tks, add_bos=self.add_truncate_bos
        # )
        chunks = break_into_chunks(
            code_tks,
            self.max_ref_tks - len(prefix_tks),
            overlap=self.ref_block_overlap,
            add_bos=self.add_truncate_bos,
        )
        return [prefix_tks + c for c in chunks if has_change(c)]


def has_change(tks: TokenSeq) -> bool:
    return Add_id in tks or Del_id in tks


def find_moved(
    changes: Iterable[Change[PythonElem]],
):
    def get_body_code(e: PythonElem):
        if isinstance(e, PythonVariable):
            return dedent(e.code)
        assert isinstance(e, PythonFunction)
        return dedent(e.header_body_code[1])

    path2change = {get_change_path(c): c for c in changes}
    added = dict[str, ProjectPath]()
    deleted = dict[str, ProjectPath]()
    moved = dict[tuple[ProjectPath, ProjectPath], Modified[PythonElem]]()
    for path, c in path2change.items():
        if isinstance(c, Added):
            code = get_body_code(c.after)
            if (old_path := deleted.pop(code, None)) is not None:
                e_before = cast(Deleted, path2change[old_path]).before
                moved[(old_path, path)] = Modified(e_before, c.after)
            else:
                added[code] = path
        elif isinstance(c, Deleted):
            code = get_body_code(c.before)
            if (new_path := added.pop(code, None)) is not None:
                e_after = cast(Added, path2change[new_path]).after
                moved[(path, new_path)] = Modified(c.before, e_after)
            else:
                deleted[code] = path
    return moved
