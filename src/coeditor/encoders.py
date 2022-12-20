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
from spot.static_analysis import (
    ModuleName,
    ProjectPath,
    PythonElem,
    PythonFunction,
    PythonModule,
    PythonVariable,
    stub_from_module,
)
from .history import (
    Added,
    Change,
    Deleted,
    Modified,
    ProjectEdit,
    get_change_path,
    parse_cst_module,
    show_change,
    to_modified_function,
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
    module_stubs: Mapping[ModuleName, Sequence[TokenSeq]] | None = None

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
    is_rename_update: bool | None = None

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
        if self.is_rename_update is None:
            is_rename_update = float("nan")
        else:
            is_rename_update = int(self.is_rename_update)
        return {
            "input_tks": len(self.input_tks),
            "output_tks": len(self.output_tks),
            "is_rename_update": is_rename_update,
        } | self.tk_pedit.stats


@dataclass
class BasicQueryEditEncoder(EditEncoder[BasicTkQueryEdit]):
    "Only use changed elements in a commit as references."
    VERSION = 3
    max_ref_tks: int = 512
    ref_chunk_overlap: int = 16
    max_chunks_per_ref: int = 4
    max_query_tks: int = 512
    max_output_tks: int = 256
    add_stubs: bool = True
    add_truncate_bos: bool = True
    collapse_unchanged: bool = True

    def encode_pedits(
        self,
        pedits: Sequence[ProjectEdit],
        include_additions: bool = False,
    ) -> Iterable[BasicTkQueryEdit]:
        stub_cache = TimedCache()
        for pedit in pedits:
            yield from self.encode_pedit(
                pedit, stub_cache, include_additions=include_additions
            )

    def encode_pedit(
        self,
        pedit: ProjectEdit,
        stub_cache: TimedCache[ModuleName, list[TokenSeq], int],
        include_additions: bool = False,
        queries: Sequence[Change[PythonFunction]] | None = None,
    ) -> Iterable[BasicTkQueryEdit]:
        """
        Args:
            - query_changes: The changes to be encoded as queries. If None, all
            modified functions in the pedit will be used as queries.

        """
        ctx_enc = CtxEncoder(pedit, self.collapse_unchanged)
        renamed = find_renamed(pedit.all_elem_changes())
        renamed_paths = {a for a, b in renamed} | {b for a, b in renamed}
        after_to_mf = {
            b: mf
            for (a, b), change in renamed.items()
            if (mf := to_modified_function(change))
        }
        module_stubs = None
        if self.add_stubs:
            module_stubs = {
                name: stub_cache.cached(
                    name, id(pymod), lambda: self.encode_module_stub(not_none(pymod))
                )[: self.max_chunks_per_ref]
                for name in pedit.changes
                if (pymod := pedit.after.modules.get(name)) is not None
            }
        tk_refs = {
            get_change_path(c): list(self.encode_elem_change(c, ctx_enc))[
                : self.max_chunks_per_ref
            ]
            for c in pedit.all_elem_changes()
            if get_change_path(c) not in renamed_paths
        }
        for (d, a), change in renamed.items():
            tk_refs[d] = list(self.encode_elem_move(d, a, change))[
                : self.max_chunks_per_ref
            ]

        query_data = dict[ProjectPath, BasicTkQueryEdit]()
        tk_pedit = TkProjectEdit(
            tk_references=tk_refs, qedits=query_data, module_stubs=module_stubs
        )
        for_training = queries is None
        if queries is None:
            queries = list(
                pedit.modified_functions(ast_must_change=True, body_must_change=True)
            )
        renamed_updates = {
            get_change_path(c)
            for c in find_rename_updates(
                renamed, [q for q in queries if isinstance(q, Modified)]
            )
        }
        # for r in renamed.values():
        #     if isinstance(r.after, PythonVariable) and r.before.name != r.after.name:
        #         print("Renamed var:", r.after.path)
        #         print(show_change(r))
        #         print("Rename updates:", renamed_updates)
        for mf in queries:
            assert not isinstance(mf, Deleted)
            if mf.after.path in renamed_paths:
                mf = after_to_mf[mf.after.path]
            body_change = mf.map(lambda x: x.header_body_code[1])
            if (
                isinstance(body_change, Modified)
                and count_lines(body_change.before) > 99
            ):
                if for_training:
                    continue  # skip large functions during training
                else:
                    warnings.warn(
                        "Function has more than 99 lines, only the first 100 lines will be edited."
                    )
            input_tks, output_tks = change_to_input_output(body_change)
            path = get_change_path(cast(Change, mf))
            path_tks = encode_basic(f"# edit: {path}")
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
            if for_training and not output_tks:
                # can happen if input too long
                continue
            query_data[path] = BasicTkQueryEdit(
                input_tks=input_tks,
                output_tks=output_tks,
                path=path,
                change_type=mf.map(lambda _: None),
                tk_pedit=tk_pedit,
                is_rename_update=path in renamed_updates,
            )
        if query_data:
            yield from query_data.values()

    def encode_elem_change(
        self, c: Change[PythonElem], ctx_encoder: CtxEncoder
    ) -> Iterable[TokenSeq]:
        path_tks = change_to_tokens(c.map(lambda e: f"# {e.path}"))
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
            overlap=self.ref_chunk_overlap,
            add_bos=self.add_truncate_bos,
        )
        for i, tks in enumerate(chunks):
            to_check = tks if i == 0 else tks[self.ref_chunk_overlap :]
            if has_change(to_check):
                yield path_tks + tks

    def encode_elem_move(
        self,
        old_path: ProjectPath,
        new_path: ProjectPath,
        change: Modified[PythonElem],
    ) -> Iterable[TokenSeq]:
        def elem2code(e: PythonElem) -> str:
            if self.collapse_unchanged:
                code = show_expr(collapse_code(e.tree))
            else:
                code = e.code
            return code

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
        chunks = break_into_chunks(
            code_tks,
            self.max_ref_tks - len(prefix_tks),
            overlap=self.ref_chunk_overlap,
            add_bos=self.add_truncate_bos,
        )
        for i, tks in enumerate(chunks):
            to_check = tks if i == 0 else tks[self.ref_chunk_overlap :]
            if has_change(to_check):
                yield prefix_tks + tks

    def encode_module_stub(self, module: PythonModule) -> list[TokenSeq]:
        name_tks = encode_basic(f"# stub: {module.name}\n")

        stub_tks = encode_basic(
            stub_from_module(module.tree, lightweight=False, keep_types=True).code
        )
        chunks = break_into_chunks(
            stub_tks,
            self.max_ref_tks - len(name_tks),
            overlap=self.ref_chunk_overlap,
            add_bos=self.add_truncate_bos,
        )
        return [name_tks + tks for tks in chunks]


def has_change(tks: TokenSeq) -> bool:
    return Add_id in tks or Del_id in tks


def find_renamed(
    changes: Iterable[Change[PythonElem]],
):
    """Use a simple heuristic to guess renamed elements."""

    def get_body_code(e: PythonElem):
        if isinstance(e, PythonVariable):
            rhs_list = list(e.iter_rhs())
            if rhs_list:
                # requires in the same parent and have the same rhs exprs
                path_str = cst.SimpleString(repr(str(e.path.pop())))
                lines = [cst.SimpleStatementLine([cst.Expr(path_str)])]
                rhs_lines = [cst.SimpleStatementLine([cst.Expr(x)]) for x in rhs_list]
                return cst.Module(lines + rhs_lines).code
            else:
                # won't match anything else
                return repr(str(e.path))
        assert isinstance(e, PythonFunction)
        return dedent(e.header_body_code[1])

    path2change = {get_change_path(c): c for c in changes}
    added = dict[str, ProjectPath]()
    deleted = dict[str, ProjectPath]()
    moved = dict[tuple[ProjectPath, ProjectPath], Modified[PythonElem]]()
    for path, c in path2change.items():
        if isinstance(c, Added):
            code = normalize_code_by_ast(get_body_code(c.after))
            if (old_path := deleted.pop(code, None)) is not None:
                e_before = cast(Deleted, path2change[old_path]).before
                moved[(old_path, path)] = Modified(e_before, c.after)
            else:
                added[code] = path
        elif isinstance(c, Deleted):
            code = normalize_code_by_ast(get_body_code(c.before))
            if (new_path := added.pop(code, None)) is not None:
                e_after = cast(Added, path2change[new_path]).after
                moved[(path, new_path)] = Modified(c.before, e_after)
            else:
                deleted[code] = path
    return moved


def find_rename_updates(
    rename_map: Mapping[tuple[ProjectPath, ProjectPath], Modified[PythonElem]],
    changes: Iterable[Modified[PythonElem]],
) -> Iterable[Modified[PythonElem]]:
    """Given a map of renamed elements, guess which modifications are caused
    only by these renamings using a simple heuristic."""

    name_maps = {
        m.before.name: m.after.name
        for m in rename_map.values()
        if m.before.name != m.after.name
    }

    class RenameSymbols(cst.CSTTransformer):
        def leave_Name(self, node: "cst.Name", updated: "cst.Name"):
            if (new_name := name_maps.get(updated.value)) is not None:
                return cst.Name(new_name)
            return updated

    for m in changes:
        tree1 = cast(cst.CSTNode, m.before.tree.visit(RenameSymbols()))
        tree2 = cast(cst.CSTNode, m.after.tree.visit(RenameSymbols()))
        code1 = normalize_code_by_ast(show_expr(tree1))
        code2 = normalize_code_by_ast(show_expr(tree2))
        if code1 == code2:
            yield m
