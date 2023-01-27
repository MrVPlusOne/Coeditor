import jedi
import jedi.cache
import jedi.parser_utils
import parso
import parso.cache
from cachetools import LRUCache
from jedi.api import classes, convert_names, helpers
from parso.python import tree
from parso.python import tree as ptree

from spot.static_analysis import (
    ModuleHierarchy,
    ModuleName,
    ProjectPath,
    sort_modules_by_imports,
)
from spot.utils import scalar_stats

from .change import Added, Change, Modified
from .common import *
from .encoding import (
    Del_id,
    Newline_id,
    TkDelta,
    TokenizedEdit,
    TruncateAt,
    break_into_chunks,
    change_tks_to_original_delta,
    change_to_line_diffs,
    change_to_tokens,
    encode_basic,
    get_extra_id,
    line_diffs_to_original_delta,
    truncate_output_tks,
    truncate_section,
    truncate_sections,
)
from .git import CommitInfo
from .scoped_changes import (
    ChangeScope,
    JModule,
    JModuleChange,
    JProjectChange,
    LineRange,
    ProjectChangeProcessor,
    ProjectState,
    StatementSpan,
)
from .tk_array import TkArray


@dataclass(frozen=True)
class ChangedHeader:
    """Represents the changes made to a header.
    This format does not store parent syntax nodes and is more suitable for serialization.
    """

    change_tks: TkArray
    # below are pre-edit attributes
    type: str
    line_range: LineRange
    path: ProjectPath

    def __repr__(self) -> str:
        return (
            f"ChangedHeader(path={self.path}, range={self.line_range}, "
            f"change={self.change_tks})"
        )


@dataclass(frozen=True)
class ChangedCodeSpan:
    """Represents the changes made to a span of code.
    This format does not store parent syntax nodes and is more suitable for serialization.
    """

    headers: Sequence[ChangedHeader]
    change_tks: TkArray
    # below are pre-edit attributes
    line_range: LineRange
    module: ModuleName


class SrcInfo(TypedDict):
    project: str
    commit: CommitInfo | None


@dataclass(frozen=True)
class C3Problem:
    "Contextual code change prediction problem."
    span: ChangedCodeSpan
    # the lines to be edited
    edit_lines: Collection[int]
    # most relevant to least relevant
    relevant_changes: Sequence[ChangedCodeSpan]
    # most relevant to least relevant
    relevant_unchanged: Sequence[ChangedCodeSpan]
    # some optional information about how the problem was generated
    change_type: Change[None]
    src_info: SrcInfo

    def meta_data_lines(self) -> list[str]:
        return [
            f"path: {self.span.headers[-1].path}",
            f"project: {self.src_info['project']}",
            f"commit: {self.src_info['commit']}",
        ]


PyFullName = NewType("PyFullName", str)


@dataclass(frozen=True)
class PyDefinition:
    """Note that the module and positions can be referring to either the import
    statement or the actual definition."""

    full_name: PyFullName
    start_pos: tuple[int, int]
    end_pos: tuple[int, int]

    @staticmethod
    def from_name(name: classes.BaseName) -> Iterable["PyDefinition"]:
        if (
            not name.in_builtin_module()
            and (full_name := name.full_name)
            # and (import_module := name.module_name)
            and (start_pos := name.get_definition_start_position())
            and (end_pos := name.get_definition_end_position())
        ):
            full_name = PyFullName(full_name)
            # if not full_name.startswith(import_module):
            #     raise ValueError(f"Inconsistent module: {full_name=}, {import_module=}")
            yield PyDefinition(full_name, start_pos, end_pos)


@dataclass(frozen=True)
class LineUsageAnalysis:
    line2usages: Mapping[int, set[PyDefinition]]


class C3ProblemGenerator(ProjectChangeProcessor[C3Problem]):
    VERSION = "2.1"
    # change spans with more than this many lines will be ignored
    max_span_lines: int = 500
    max_lines_to_edit: int = 25
    max_problems_per_elem: int = 4

    def __init__(self, analyzer: "JediUsageAnalyzer | None" = None):
        if analyzer is None:
            analyzer = JediUsageAnalyzer()

        self.analyzer = analyzer
        # whether to only generate problems for editing functions
        self._is_training: bool = False

    def __repr__(self) -> str:
        return repr_modified_args(self)

    def append_stats(self, stats: dict[str, Any]) -> None:
        rec_add_dict_to(stats, {"analyzer_errors": self.analyzer.error_counts})

    def clear_stats(self) -> None:
        return self.analyzer.error_counts.clear()

    def set_training(self, is_training: bool) -> None:
        self._is_training = is_training

    def pre_edit_analysis(
        self,
        pstate: ProjectState,
        modules: Mapping[RelPath, JModule],
        changes: Mapping[ModuleName, JModuleChange],
    ) -> Mapping[ModuleName, LineUsageAnalysis]:
        "Return the definition usages of each line."
        project = pstate.project
        result = dict[ModuleName, LineUsageAnalysis]()

        src_map = {m.mname: f for f, m in modules.items()}
        for mname, mchange in changes.items():
            if not isinstance(mchange.module_change, Modified):
                continue

            lines_to_analyze = set[int]()
            for span in mchange.changed.values():
                if span.change is Added:
                    continue
                lines_to_analyze.update(range(*span.line_range))
                lines_to_analyze.update(range(*span.header_line_range))

            mod_path = src_map[mname]
            script = pstate.scripts[mod_path]
            line_usages = self.analyzer.get_line_usages(
                script, project.path, lines_to_analyze, silent=True
            )
            result[mname] = line_usages
        return result

    def post_edit_analysis(
        self,
        pstate: ProjectState,
        modules: Mapping[RelPath, JModule],
        changes: Mapping[ModuleName, JModuleChange],
    ) -> list[ModuleName]:
        "Return the topological order among the modules."
        # sort modules topologically
        module_deps = dict[ModuleName, set[ModuleName]]()
        for rel_path, module in modules.items():
            names = {n for n in module.imported_names}
            script = pstate.scripts[rel_path]
            deps = module_deps.setdefault(module.mname, set())
            for n in names:
                try:
                    srcs = _fast_goto(
                        script, n, follow_imports=True, follow_builtin_imports=False
                    )
                except Exception as e:
                    self.analyzer.add_error(str(e))
                    continue
                for source in srcs:
                    deps.add(source.module_name)
        module_order = sort_modules_by_imports(module_deps)
        return module_order

    def process_change(
        self,
        pchange: JProjectChange,
        mod2usages: Mapping[ModuleName, LineUsageAnalysis],
        module_order: Sequence[ModuleName],
    ) -> Sequence[C3Problem]:
        before_mod_map = {m.mname: m for m in pchange.all_modules.before}
        mod_hier = ModuleHierarchy.from_modules(before_mod_map)
        cspan_cache = dict[PyDefinition, list[ChangedCodeSpan]]()

        def get_def_spans(used: PyDefinition) -> list[ChangedCodeSpan]:
            "Get the (pre-edit) spans for the given definition."
            if used.full_name in cspan_cache:
                return cspan_cache[used.full_name]
            path = mod_hier.resolve_path(split_dots(used.full_name))
            cspans = list[ChangedCodeSpan]()
            if path is None or (jmod := before_mod_map.get(path.module)) is None:
                cspan_cache[used] = cspans
                return cspans
            scope = jmod.as_scope
            elem = scope._search(path.path, used.start_pos[0])
            func_scopes = list[ChangeScope]()
            stmt_spans = list[StatementSpan]()
            match elem:
                case ChangeScope(tree=ptree.Function()):
                    func_scopes.append(elem)
                case ChangeScope(tree=ptree.Class()):
                    # add all attrs and methods
                    stmt_spans.extend(elem.spans)
                    func_scopes.extend(
                        s
                        for s in elem.subscopes.values()
                        if isinstance(s.tree, ptree.Function)
                    )
                case StatementSpan():
                    stmt_spans.append(elem)

            # add collapsed functions
            for f_scope in func_scopes:
                ancestors = f_scope.ancestors()
                stmts = f_scope.spans[-1].statements
                body_code = stmts[-1].get_code()
                if len(stmts) > 1:
                    ellipsis = "    " * (len(ancestors) - 1) + "# ...\n"
                    body_code = ellipsis + body_code
                cspan = ChangedCodeSpan(
                    [to_header(Modified.from_unchanged(s)) for s in ancestors],
                    TkArray.new(encode_basic(body_code)),
                    f_scope.spans[-1].line_range,
                    f_scope.path.module,
                )
                cspans.append(cspan)

            # add statement spans
            for stmt_span in stmt_spans:
                ancestors = stmt_span.scope.ancestors()
                stmts = stmt_span.statements
                match stmts:
                    case [
                        ptree.PythonNode(
                            type="simple_stmt",
                            children=[ptree.String(), ptree.Newline()],
                        ),
                        *rest,
                    ]:
                        if not rest:
                            continue
                        stmts = rest
                body_code = "".join(s.get_code() for s in stmts).lstrip("\n")
                cspan = ChangedCodeSpan(
                    [to_header(Modified.from_unchanged(s)) for s in ancestors],
                    TkArray.new(encode_basic(body_code)),
                    stmt_span.line_range,
                    stmt_span.scope.path.module,
                )
                cspans.append(cspan)

            cspan_cache[used] = cspans
            return cspans

        def get_relevant_unchanged(
            this_change: ChangedCodeSpan, other_changes: Collection[ChangedCodeSpan]
        ):
            module = this_change.module
            line_usages = mod2usages[module]
            # parent defs are also considered as used
            parent_defs = [
                PyDefinition(
                    PyFullName(f"{c.path.module}.{c.path.path}"),
                    (c.line_range[0], 0),
                    (c.line_range[1], 0),
                )
                for c in this_change.headers
            ]
            # immediate parents are more relevant
            sorted_defs = list(reversed(parent_defs))
            used_defs = set(sorted_defs)
            all_lines = set(range(*this_change.line_range))
            all_lines.update(range(*this_change.headers[-1].line_range))
            for l in all_lines:
                for pydef in line_usages.line2usages.get(l, set()):
                    if (
                        pydef.full_name.startswith(module)
                        and pydef.start_pos[0] in all_lines
                    ):
                        # skip self references
                        continue
                    if pydef not in used_defs:
                        used_defs.add(pydef)
                        sorted_defs.append(pydef)

            # return unique cspans
            seen = set[tuple[ModuleName, LineRange]]()
            # we don't need to show the changed parts again
            for cspan in (this_change, *other_changes):
                seen.add((cspan.module, cspan.line_range))
            result = list[ChangedCodeSpan]()
            for used in sorted_defs:
                for cspan in get_def_spans(used):
                    key = (cspan.module, cspan.line_range)
                    if key not in seen:
                        result.append(cspan)
                        seen.add(key)
            return result

        header_cache = dict[ProjectPath, ChangedHeader]()

        def to_header(cs: Change[ChangeScope]) -> ChangedHeader:
            path = cs.earlier().path
            if (ch := header_cache.get(path)) is None:
                header_change = cs.map(lambda s: s.header_code.strip("\n"))
                ch = ChangedHeader(
                    TkArray.new(change_to_tokens(header_change)),
                    cs.earlier().tree.type,
                    cs.earlier().header_line_range,
                    cs.earlier().path,
                )
                header_cache[path] = ch
            return ch

        processed_cspans = list[ChangedCodeSpan]()
        problems = list[C3Problem]()
        for m in module_order:
            if (mchange := pchange.changed.get(m)) is None:
                continue
            for span in mchange.changed.values():
                original, delta = line_diffs_to_original_delta(
                    change_to_line_diffs(span.change)
                )
                change_tks = change_to_tokens(span.change)
                n_lines = count_lines(original)

                def change_counts(r: range) -> int:
                    return delta.for_input_range((r.start, r.stop)).num_changes()

                code_span = ChangedCodeSpan(
                    headers=[to_header(cs) for cs in span.parent_scopes],
                    change_tks=TkArray.new(change_tks),
                    line_range=span.line_range,
                    module=span.path.module,
                )
                should_mk_problem = (
                    (span.change.as_char() == Modified.as_char())
                    and (self._is_training or span._is_func_body())
                    and (count_lines(span.change.earlier()) <= self.max_span_lines)
                    and (count_lines(span.change.later()) <= self.max_span_lines)
                )
                if should_mk_problem:
                    # latest changes are more relevant
                    relevant_changes = list(reversed(processed_cspans))
                    relevant_unchanged = get_relevant_unchanged(
                        code_span, relevant_changes
                    )
                    src_info: SrcInfo = {
                        "project": pchange.project_name,
                        "commit": pchange.commit_info,
                    }
                    edit_ranges = list[range]()
                    for i in range(0, n_lines, self.max_lines_to_edit):
                        r = range(i, min(n_lines, i + self.max_lines_to_edit))
                        c = change_counts(r)
                        if c > 0:
                            edit_ranges.append(r)
                        if len(edit_ranges) >= self.max_problems_per_elem:
                            break

                    for r in edit_ranges:
                        prob = C3Problem(
                            code_span,
                            r,
                            relevant_changes=relevant_changes,
                            relevant_unchanged=relevant_unchanged,
                            change_type=span.change.map(lambda _: None),
                            src_info=src_info,
                        )
                        problems.append(prob)
                processed_cspans.append(code_span)
        return problems


@dataclass(frozen=True)
class TkC3Problem(TokenizedEdit):
    "Tokenized contextual code change prediction problem."
    input: TkArray
    output: TkArray
    path: ProjectPath
    change_type: Change[None]
    # most relevant to least relevant
    named_references: Sequence[tuple[str, TkArray]]
    project: str
    commit: CommitInfo | None

    @property
    def references(self) -> Sequence[TkArray]:
        return [ref for _, ref in self.named_references]

    def __repr__(self):
        return f"TkC3Problem(path={self.path}, type={self.change_type.as_char()}, stats={self.stats()})"

    @property
    def input_tks(self) -> TokenSeq:
        return self.input.tolist()

    @property
    def output_tks(self) -> TokenSeq:
        return self.output.tolist()

    @property
    def main_tks(self) -> TokenSeq:
        return self.input_tks

    def show(self) -> str:
        return self.show_prediction(None)

    def all_ctxs(self) -> dict[str, TokenSeq]:
        return {name: ref.tolist() for name, ref in self.named_references}

    def meta_data_lines(self) -> list[str]:
        return [
            f"path: {self.path}",
            f"n_references: {len(self.references)}",
            f"total_reference_tks: {sum(len(ref) for ref in self.references)}",
            f"project: {self.project}",
            f"commit: {self.commit}",
        ]

    def stats(self) -> Mapping[str, int | float]:
        return {
            "input_tks": len(self.input_tks),
            "output_tks": len(self.output_tks),
            "n_references": len(self.references),
            "total_reference_tks": sum(len(ref) for ref in self.references),
        }


class C3TokenizerArgs(TypedDict):
    max_ref_tks: int
    max_query_tks: int
    max_output_tks: int
    max_scope_tks: int
    max_ref_tks_sum: int
    ref_chunk_overlap: int


@dataclass
class C3ProblemTokenizer:
    VERSION = "2.2"
    max_ref_tks: int = 512
    max_query_tks: int = 512
    max_output_tks: int = 256
    max_scope_tks: int = 128
    max_ref_tks_sum: int = 512 * 12
    ref_chunk_overlap: int = 32

    def get_args(self):
        return C3TokenizerArgs(
            max_ref_tks=self.max_ref_tks,
            max_query_tks=self.max_query_tks,
            max_output_tks=self.max_output_tks,
            max_scope_tks=self.max_scope_tks,
            max_ref_tks_sum=self.max_ref_tks_sum,
            ref_chunk_overlap=self.ref_chunk_overlap,
        )

    @classmethod
    def from_args(cls, args: C3TokenizerArgs) -> Self:
        return cls(**args)

    def __post_init__(self):
        self._offset_cache = LRUCache[int, TkArray](maxsize=100)

    def tokenize_problem(
        self,
        problem: C3Problem,
    ) -> TkC3Problem:
        span = problem.span

        original: TokenSeq
        tk_delta: TkDelta
        original, tk_delta = change_tks_to_original_delta(span.change_tks.tolist())
        origin_lines = split_list(original, Newline_id)
        assert isinstance(problem.edit_lines, range), "Only support range for now"
        edit_start = problem.edit_lines[0]
        edit_end = problem.edit_lines[-1] + 1
        scope_tks = self._encode_headers(span.headers, 0)
        input_limit = self.max_query_tks - len(scope_tks)

        chunk_input = TokenSeq()
        chunk_output = TokenSeq()

        for i in range(len(problem.edit_lines)):
            chunk_input.append(get_extra_id(i))
            l = edit_start + i
            if l < len(origin_lines):
                chunk_input.extend(origin_lines[l])
                chunk_input.append(Newline_id)
            line_change = join_list(tk_delta.get_line_change(l), Newline_id)
            chunk_output.append(get_extra_id(i))
            chunk_output.extend(line_change)
            if line_change and line_change[-1] != Del_id:
                chunk_output.append(Newline_id)

        # limit the input size if it's too long
        chunk_input = truncate_section(
            chunk_input, TruncateAt.Right, input_limit, inplace=True
        )
        chunk_output = truncate_output_tks(chunk_input, chunk_output)

        # try move some prev_change_tks into the input
        above_tks = join_list(origin_lines[:edit_start] + [TokenSeq()], Newline_id)
        above_tks = tk_delta.for_input_range((0, edit_start)).apply_to_input(above_tks)
        below_tks = join_list(origin_lines[edit_end:] + [TokenSeq()], Newline_id)
        chunk_input, above_tks, below_tks = self._inline_some_context(
            chunk_input, above_tks, below_tks, input_limit
        )

        chunk_output = truncate_section(
            chunk_output,
            TruncateAt.Right,
            self.max_output_tks,
            add_bos=False,
            inplace=True,
        )

        above_chunks = break_into_chunks(
            above_tks,
            lambda i: self._encode_headers(span.headers, -1 - i),
            chunk_size=self.max_ref_tks,
            overlap=self.ref_chunk_overlap,
            right_to_left=True,
        )
        if not below_tks:
            below_chunks = []
        else:
            below_chunks = break_into_chunks(
                below_tks,
                lambda i: self._encode_headers(span.headers, i + 1),
                chunk_size=self.max_ref_tks,
                overlap=self.ref_chunk_overlap,
            )
        above_chunks = [
            (f"above chunk {i}", TkArray.new(chunk))
            for i, chunk in enumerate(above_chunks)
        ]
        below_chunks = [
            (f"below chunk {i}", TkArray.new(chunk))
            for i, chunk in enumerate(below_chunks)
        ]
        all_refs = above_chunks + below_chunks
        ref_size_sum = sum(len(ref) for _, ref in all_refs)

        # compute the references that are relevant to this span
        if ref_size_sum < self.max_ref_tks_sum:
            changed = self._group_encode_changed_refs(problem.relevant_changes)
            for i, chunk in enumerate(changed):
                all_refs.append((f"changed ref {i}", TkArray.new(chunk)))
            ref_size_sum += sum(len(x) for x in changed)
        if ref_size_sum < self.max_ref_tks_sum:
            unchanged = self._group_encode_unchanged_refs(problem.relevant_unchanged)
            for i, chunk in enumerate(unchanged):
                all_refs.append((f"unchanged ref {i}", TkArray.new(chunk)))

        # take until we hit the limit
        ref_size_sum = 0
        kept_refs = list[tuple[str, TkArray]]()
        for (name, ref) in all_refs:
            if ref_size_sum + len(ref) > self.max_ref_tks_sum:
                continue
            ref_size_sum += len(ref)
            kept_refs.append((name, ref))

        return TkC3Problem(
            TkArray.new(scope_tks + chunk_input),
            TkArray.new(chunk_output),
            path=span.headers[-1].path,
            change_type=problem.change_type,
            named_references=kept_refs,
            project=problem.src_info["project"],
            commit=problem.src_info["commit"],
        )

    def _encode_headers(
        self, scope_changes: Sequence[ChangedHeader], offset: int
    ) -> TokenSeq:
        segs = [c.change_tks.tolist() for c in scope_changes]
        if offset != 0:
            segs.append(self._get_offset_tks(offset).tolist())
        segs.append([])
        scope_tks = join_list(segs, Newline_id)
        scope_tks = truncate_section(
            scope_tks, TruncateAt.Left, self.max_scope_tks, inplace=True
        )
        return scope_tks

    def _get_offset_tks(self, offset: int) -> TkArray:
        if (tks := self._offset_cache.get(offset)) is None:
            tks = TkArray.new(encode_basic(f"# offset: {offset}"))
            self._offset_cache[offset] = tks
        return tks

    def _inline_some_context(
        self,
        input: TokenSeq,
        above_ctx: TokenSeq,
        below_ctx: TokenSeq,
        size_limit: int,
    ) -> tuple[TokenSeq, TokenSeq, TokenSeq]:
        "try move some some of the ctx tokens into the input if there's space."
        extra_space = size_limit - len(input)
        if (above_ctx or below_ctx) and extra_space > 0:
            truncated_above, truncated_below = truncate_sections(
                extra_space,
                (above_ctx, TruncateAt.Left),
                (below_ctx, TruncateAt.Right),
                add_bos=True,
            )
            above_left = len(above_ctx) - len(truncated_above)
            if above_left > 0:
                above_ctx = truncate_section(
                    above_ctx, TruncateAt.Right, above_left + self.ref_chunk_overlap
                )
            else:
                above_ctx = TokenSeq()
            below_left = len(below_ctx) - len(truncated_below)
            if below_left > 0:
                below_ctx = truncate_section(
                    below_ctx, TruncateAt.Left, below_left + self.ref_chunk_overlap
                )
            else:
                below_ctx = TokenSeq()
            input = truncated_above + input + truncated_below
        return input, above_ctx, below_ctx

    def _group_encode_unchanged_refs(
        self, elems: Sequence[ChangedCodeSpan]
    ) -> Sequence[TokenSeq]:
        return self._group_encode_changed_refs(elems)

    def _group_encode_changed_refs(
        self, changes: Sequence[ChangedCodeSpan]
    ) -> Sequence[TokenSeq]:
        module2changes = groupby(changes, lambda c: c.module)
        all_chunks = list[TokenSeq]()
        for change_group in module2changes.values():
            change_group.sort(key=lambda c: c.line_range[0])
            segs = list[TokenSeq]()
            # we'll add module as the chunk header, so we start within the module
            last_scope = change_group[0].headers[:1]
            for c in change_group:
                header_diff = list[ChangedHeader]()
                for i, h in enumerate(c.headers):
                    if i >= len(last_scope) or h.path != last_scope[i].path:
                        header_diff.append(h)
                if header_diff:
                    header_tks = self._encode_headers(header_diff, 0)
                    segs.append(header_tks)
                segs.append(c.change_tks.tolist())
                segs.append([Newline_id, Newline_id])
                last_scope = c.headers
            segs.append([Newline_id])
            mod_change = change_group[0].headers[:1]
            mod_chunks = break_into_chunks(
                join_list(segs),
                lambda i: self._encode_headers(mod_change, i),
                self.max_ref_tks,
                overlap=self.ref_chunk_overlap,
            )
            all_chunks.extend(mod_chunks)
        return all_chunks

    def _compute_stats(self, problems: Sequence[C3Problem]):
        all_stats = pmap(self._tokenize_stats, problems)
        if not all_stats:
            return dict()
        keys = all_stats[0].keys()
        stats = dict[str, dict]()
        for k in keys:
            stats[k] = scalar_stats([s[k] for s in all_stats])
        return stats

    def _tokenize_stats(self, problem: C3Problem):
        return self.tokenize_problem(problem).stats()

    def __repr__(self):
        return repr_modified_args(self)


@dataclass
class JediUsageAnalyzer:
    _KnownJediErrors = {
        "not enough values to unpack (expected 2",
        "'Newline' object has no attribute 'children'",
        "trailer_op is actually ",
        "There's a scope that was not managed: <Module",
        "maximum recursion depth exceeded",
        "'NoneType' object has no attribute 'type'",
    }

    def __post_init__(self):
        self.error_counts = dict[str, int]()
        self.tlogger: TimeLogger = TimeLogger()

    def get_line_usages(
        self,
        script: jedi.Script,
        proj_root: Path,
        lines_to_analyze: Collection[int],
        silent: bool = False,
    ):
        jmod: tree.Module = script._module_node
        line2usages = dict[int, set[PyDefinition]]()
        all_names = [
            name for k, names in jmod.get_used_names()._dict.items() for name in names
        ]
        all_names.sort(key=lambda x: x.start_pos)
        for name in tqdm(all_names, f"Analyzing {script.path}", disable=silent):
            name: tree.Name
            line = name.start_pos[0]
            if line not in lines_to_analyze:
                continue
            usages = line2usages.setdefault(line, set())
            try:
                defs = _fast_goto(
                    script,
                    name,
                    follow_imports=True,
                    follow_builtin_imports=False,
                )
                for d in defs:
                    usages.update(PyDefinition.from_name(d))

            except Exception as e:
                err_text = repr(e)
                str_limit = 40
                if len(err_text) > str_limit:
                    err_text = err_text[:str_limit] + "..."
                self.add_error(err_text)
        return LineUsageAnalysis(line2usages)

    def add_error(self, err_text: str):
        self.error_counts[err_text] = self.error_counts.get(err_text, 0) + 1

    @staticmethod
    def is_known_error(err_text: str):
        return any(k in err_text for k in JediUsageAnalyzer._KnownJediErrors)


def _fast_goto(
    script: jedi.Script,
    tree_name: tree.Name,
    *,
    follow_imports=False,
    follow_builtin_imports=False,
    only_stubs=False,
    prefer_stubs=False,
) -> set[classes.Name]:
    """
    Goes to the name that defined the object under the cursor. Optionally
    you can follow imports.
    Multiple objects may be returned, depending on an if you can have two
    different versions of a function.

    :param follow_imports: The method will follow imports.
    :param follow_builtin_imports: If ``follow_imports`` is True will try
        to look up names in builtins (i.e. compiled or extension modules).
    :param only_stubs: Only return stubs for this method.
    :param prefer_stubs: Prefer stubs to Python objects for this method.
    :rtype: list of :class:`.Name`
    """
    name = script._get_module_context().create_name(tree_name)

    # Make it possible to goto the super class function/attribute
    # definitions, when they are overwritten.
    names = []
    if name.tree_name.is_definition() and name.parent_context.is_class():
        class_node = name.parent_context.tree_node
        class_value = script._get_module_context().create_value(class_node)
        mro = class_value.py__mro__()
        next(mro)  # Ignore the first entry, because it's the class itself.
        for cls in mro:
            names = cls.goto(tree_name.value)
            if names:
                break

    if not names:
        names = list(name.goto())

    if follow_imports:
        names = helpers.filter_follow_imports(names, follow_builtin_imports)
    names = convert_names(
        names,
        only_stubs=only_stubs,
        prefer_stubs=prefer_stubs,
    )

    return {classes.Name(script._inference_state, d) for d in set(names)}


# fix jedi cache error
def _get_parso_cache_node(grammar, path):
    try:
        return jedi.parser_utils.parser_cache[grammar._hashed].get(path)
    except KeyError:
        return None


jedi.parser_utils.get_parso_cache_node = _get_parso_cache_node


def fix_jedi_cache(cache_dir: Path):
    jedi.parser_utils.get_parso_cache_node = _get_parso_cache_node
    jedi.settings.cache_directory = cache_dir / "jedi_cache"
    parso.cache._default_cache_path = cache_dir / "parso_cache"