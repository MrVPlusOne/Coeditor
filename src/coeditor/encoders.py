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
    change_tks_to_input_output,
    change_to_tokens,
    collapse_code,
    encode_basic,
    inline_output_tokens,
    tokens_to_change,
    truncate_output_tks,
    truncate_section,
    truncate_sections,
)
from spot.static_analysis import (
    ModuleName,
    ProjectPath,
    PythonElem,
    PythonFunction,
    PythonModule,
    PythonVariable,
    show_element,
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
            "n_references": len(self.tk_references),
            "ref_size_max": max(
                len(tks) for segs in self.tk_references.values() for tks in segs
            ),
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
    prev_chunks: Sequence[TokenSeq]
    tk_pedit: TkProjectEdit["BasicTkQueryEdit"]
    is_rename_update: bool | None = None

    @property
    def main_tks(self):
        return self.input_tks

    def show(self) -> str:
        return self.show_prediction(None)

    def all_ctxs(self) -> dict[str, TokenSeq]:
        prev_segs = {self.path: self.prev_chunks}
        ref_segs = {
            path: seg
            for path, seg in self.tk_pedit.tk_references.items()
            if path != self.path
        }
        return {
            str(path) + (f" ({i})" if len(segs) > 1 else ""): seg
            for path, segs in (prev_segs | ref_segs).items()
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
            "prev_chunks": len(self.prev_chunks),
            "is_rename_update": is_rename_update,
        } | self.tk_pedit.stats


@dataclass
class EditRequest:
    target: Change[PythonFunction]
    respect_lines: int


@dataclass
class QueryRefEditEncoder(EditEncoder[BasicTkQueryEdit]):
    """Encode edits as queries and references.
    # Args
    - `max_ref_tks`: The maximum number of tokens in each reference.
    - `ref_chunk_overlap`: The number of tokens to overlap between reference
    or query chunks. When a reference/query is longer than its limit, it will
    be broken into chunks.
    - `max_chunks_per_ref`: The maximum number of chunks to use for each reference.
    More chunks will be discarded.
    - `max_lines_per_function`: Functions larger than this limit will be discarded
    during training.
    - `max_query_tks`: The maximum number of tokens in each query.
    - `max_output_tks`: The maximum number of tokens in each output sequence. Exceeding
    parts will be cut off during training.
    - `add_stubs`: Whether to add stubs of changed modules as references.
    - `add_truncate_bos`: Whether to add a BOS and EOS tokens when truncating.
    - `collapse_unchanged`: Whether to omit of the body of unchanged definitions
    when generating the references.
    """

    VERSION = 5
    max_ref_tks: int = 512
    ref_chunk_overlap: int = 16
    max_chunks_per_ref: int = 4
    max_lines_per_function: int = 500
    max_query_tks: int = 512
    max_output_tks: int = 256
    add_stubs: bool = True
    add_truncate_bos: bool = True
    collapse_unchanged: bool = True
    ast_mask_prob: float = 0.0

    def encode_pedits(
        self,
        pedits: Sequence[ProjectEdit],
        training: bool,
    ) -> Iterable[BasicTkQueryEdit]:
        stub_cache = TimedCache()
        for pedit in pedits:
            yield from self.encode_pedit(pedit, stub_cache, training=training)

    def encode_pedit(
        self,
        pedit: ProjectEdit,
        stub_cache: TimedCache[ModuleName, list[TokenSeq], int],
        training: bool,
        queries: Sequence[EditRequest] | None = None,
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
        for refs in tk_refs.values():
            for seg in refs:
                assert len(seg) <= self.max_ref_tks

        query_data = dict[ProjectPath, BasicTkQueryEdit]()
        tk_pedit = TkProjectEdit(
            tk_references=tk_refs, qedits=query_data, module_stubs=module_stubs
        )
        no_queries = queries is None
        if queries is None:
            queries = [
                req
                for mf in pedit.modified_functions(
                    ast_must_change=True, body_must_change=True
                )
                for req in self.sample_requests(mf, training)
            ]
        renamed_updates = {
            get_change_path(c)
            for c in find_rename_updates(
                renamed, [q for q in queries if isinstance(q, Modified)]
            )
        }

        for request in queries:
            mf = request.target
            assert not isinstance(mf, Deleted)
            if mf.after.path in renamed_paths:
                mf = after_to_mf[mf.after.path]

            if (
                no_queries
                and not training
                and isinstance(mf, Modified)
                and count_lines(mf.before.header_body_code[1]) > 99
            ):
                # skip large functions during evaluation
                continue

            if training and self.ast_mask_prob > 0 and isinstance(mf, Modified):
                tree_before = random_mask_ast(
                    mf.before.tree,
                    mask_prob=self.ast_mask_prob,
                    max_span_size=6,
                    mask_name="MASKED",
                )
                try:
                    code_before = show_element(tree_before, mf.before.in_class)
                except Exception as e:
                    # e.add_note(f"CST:\n{tree_before}")
                    # raise
                    warnings.warn("Failed to show masked code: " + str(e))
                    code_before = mf.before.code
                code_change = Modified(code_before, mf.after.code)
            else:
                code_change = mf.map(lambda x: x.code)
            change_tks = change_to_tokens(code_change)
            (input_tks, output_tks), context = change_tks_to_query_context(
                change_tks, request.respect_lines
            )

            path = get_change_path(mf)
            path_tks = encode_basic(f"# edit: {path}")
            cls_tks = tuple()
            if (cls_p := mf.after.parent_class) is not None:
                cls_tks = (ctx_enc.encode_ctx_element(cls_p),)
            context = join_list((*cls_tks, context), sep=Newline_id)

            input_tks, used_context = truncate_sections(
                self.max_query_tks - len(path_tks) - 2,
                (input_tks, TruncateAt.Right),
                (context, TruncateAt.Left),
                add_bos=self.add_truncate_bos,
            )
            input_tks = join_list((path_tks, used_context, input_tks), sep=Newline_id)
            if len(used_context) == len(context):
                prev_chunks = []
            else:
                to_keep = len(context) - len(used_context) + self.ref_chunk_overlap + 1
                remaining_context = context[:to_keep]
                remaining_context.append(EOS_id)
                prev_chunks = list(self.encode_previous_chunks(path, remaining_context))
            output_tks = truncate_output_tks(input_tks, output_tks)
            output_tks = truncate_section(
                output_tks,
                TruncateAt.Right,
                self.max_output_tks,
                add_bos=self.add_truncate_bos,
            )
            if no_queries and not output_tks:
                continue
            assert len(input_tks) <= self.max_query_tks
            assert len(output_tks) <= self.max_output_tks
            query_data[path] = BasicTkQueryEdit(
                input_tks=input_tks,
                output_tks=output_tks,
                path=path,
                change_type=mf.map(lambda _: None),
                prev_chunks=prev_chunks,
                tk_pedit=tk_pedit,
                is_rename_update=path in renamed_updates,
            )
        if query_data:
            yield from query_data.values()

    def sample_requests(
        self, mf: Modified[PythonFunction], training: bool
    ) -> Iterable[EditRequest]:
        if not training:
            # keep the signature changes at test time
            yield EditRequest(mf, len(mf.after.header_body_code[0]))
        else:
            if (
                len(mf.before.code.split("\n")) > self.max_lines_per_function
                or len(mf.after.code.split("\n")) > self.max_lines_per_function
            ):
                return  # skip oversized functions
            lines_per_request = 50
            min_lines_to_edit = 3
            # split it into chunks
            focus_max = max(1, len(mf.after.code.split("\n")) - min_lines_to_edit)
            for start in range(0, focus_max, lines_per_request):
                x = random.random()
                end = min(focus_max, start + lines_per_request)
                # bias focus toward start
                focus = int(x * x * (end - start) + 0.5) + start
                yield EditRequest(mf, focus)

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

    def encode_previous_chunks(
        self, path: ProjectPath, context_tks: TokenSeq
    ) -> Iterable[TokenSeq]:
        "Encode the changes immediately before the current editing focus."
        if not context_tks:
            return
        path_tks = encode_basic(f"# edit: {str(path)}")
        path_tks = truncate_section(
            path_tks,
            TruncateAt.Left,
            self.max_ref_tks // 4,
            add_bos=self.add_truncate_bos,
        )
        path_tks.append(Newline_id)
        chunks = break_into_chunks(
            context_tks,
            self.max_ref_tks - len(path_tks),
            overlap=self.ref_chunk_overlap,
            add_bos=self.add_truncate_bos,
        )
        for tks in chunks:
            # these important context should always be seen by the model
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


def change_tks_to_query_context(change_tks: TokenSeq, respect_lines: int):
    lines = split_list(change_tks, Newline_id)
    spliter = 0
    result_lines = 0
    for i, l in enumerate(lines):
        if l and l[0] == Del_id:
            pass
        else:
            result_lines += 1
        if result_lines == respect_lines:
            spliter = i + 1

    context = join_list(lines[:spliter], Newline_id)
    query = change_tks_to_input_output(join_list(lines[spliter:], Newline_id))
    return query, context


def apply_output_tks_to_change(
    change_tks: TokenSeq,
    respect_lines: int,
    out_tks: TokenSeq,
) -> Modified[str]:
    (input_tks, _), context = change_tks_to_query_context(change_tks, respect_lines)
    change_tks = (
        context
        + [Newline_id]
        + inline_output_tokens(input_tks, out_tks, leave_unpredicted=False)
    )
    return tokens_to_change(change_tks)


def compute_node_size(node: cst.CSTNode) -> Mapping[cst.CSTNode, int]:
    class Counter(cst.CSTVisitor):
        def __init__(self):
            self.counts: dict[cst.CSTNode, int] = {}
            self.counter = 0

        def on_visit(self, node: cst.CSTNode) -> bool:
            self.counts[node] = self.counter
            if isinstance(node, cst.BaseExpression):
                self.counter += 1
            return True

        def on_leave(self, node: cst.CSTNode):
            self.counts[node] = self.counter - self.counts[node]

    counter = Counter()
    node.visit(counter)
    return counter.counts


def random_mask_ast(
    node: cst.CSTNode,
    mask_prob: float,
    max_span_size: int,
    rng: random.Random | None = None,
    mask_name: str = "MASKED",
) -> cst.CSTNode:
    if rng is None:
        rng = random._inst

    size_map = compute_node_size(node)

    class Masker(cst.CSTTransformer):
        def visit_ConcatenatedString(self, node: cst.ConcatenatedString):
            return False

        def on_leave(self, original: cst.CSTNode, updated):
            updated = super().on_leave(original, updated)
            if (
                isinstance(
                    original,
                    (
                        cst.Name,
                        cst.Call,
                        cst.Attribute,
                        cst.Subscript,
                        cst.Lambda,
                        cst.BinaryOperation,
                        cst.UnaryOperation,
                        cst.Comparison,
                        cst.BaseList,
                        cst.SimpleString,
                        cst.SimpleStatementLine,
                    ),
                )
                and 0 < (size := size_map[original]) <= max_span_size
            ):
                if cast(random.Random, rng).random() < mask_prob / size:
                    mask = cst.Name(mask_name)
                    if isinstance(original, cst.SimpleStatementLine):
                        mask = cst.SimpleStatementLine([cst.Expr(mask)])
                    elif isinstance(original, cst.SimpleString):
                        mask = cst.SimpleString(repr(mask_name))
                    return mask
            return updated

    try:
        new_node = node.visit(Masker())
    except Exception as e:
        warnings.warn(f"Failed during random masking: {str(e)}")
        return node
    assert isinstance(new_node, cst.CSTNode)
    return new_node
