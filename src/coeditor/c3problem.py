from dataclasses import replace
from pprint import pprint
from textwrap import indent

import jedi
import jedi.cache
import jedi.parser_utils
import parso
import parso.cache
from jedi.api import classes, convert_names, helpers
from parso.python import tree
from parso.python import tree as ptree

from coeditor._utils import scalar_stats

from .change import Added, Change, Modified, show_change
from .common import *
from .encoding import (
    Add_id,
    Del_id,
    Newline_id,
    TkDelta,
    TokenizedEdit,
    TruncateAt,
    break_into_chunks,
    change_to_line_diffs,
    change_to_tokens,
    decode_tokens,
    encode_lines_join,
    encode_single_line,
    get_extra_id,
    line_diffs_to_original_delta,
    tk_splitlines,
    tokens_to_change,
    truncate_output_tks,
    truncate_section,
    truncate_sections,
)
from .git import CommitInfo
from .scoped_changes import (
    ChangedSpan,
    ChangeScope,
    JModule,
    JModuleChange,
    JProjectChange,
    LineRange,
    ModuleName,
    ProjectChangeProcessor,
    ProjectPath,
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
    original: TkArray
    delta: TkDelta
    # below are pre-edit attributes
    line_range: LineRange
    module: ModuleName

    def get_change(self) -> Modified[str]:
        change_tks = self.delta.apply_to_change(self.original.tolist())
        return tokens_to_change(change_tks)

    def change_size(self) -> int:
        return len(self.original) + self.delta.change_size()


class SrcInfo(TypedDict):
    project: str
    commit: CommitInfo | None


@dataclass(frozen=True)
class C3Problem:
    "Contextual code change prediction problem."
    span: ChangedCodeSpan
    # The line ids in the change tks that should be edited
    edit_line_ids: Sequence[int]
    # most relevant to least relevant
    relevant_changes: Sequence[ChangedCodeSpan]
    # most relevant to least relevant
    relevant_unchanged: Mapping["PyFullName", "PyDefinition"]
    # some optional information about how the problem was generated
    change_type: Change[None]
    src_info: SrcInfo
    transformations: tuple[str, ...] = ()

    def restrict_span_changes(self):
        "restrict the changes in the span to the edit lines"
        eids = self.edit_line_ids
        delta = self.span.delta.for_input_range((eids[0], eids[-1] + 1))
        span = replace(self.span, delta=delta)
        return replace(self, span=span)

    @property
    def path(self) -> ProjectPath:
        return self.span.headers[-1].path

    def meta_data_lines(self) -> list[str]:
        return [
            f"path: {self.span.headers[-1].path}",
            f"project: {self.src_info['project']}",
            f"commit: {self.src_info['commit']}",
        ]

    def summary(self) -> str:
        return "\n".join(self.meta_data_lines())

    def print(self):
        main_change = self.span.delta.apply_to_change(self.span.original.tolist())
        print_sections(
            ("summary", self.summary()),
            ("main change", decode_tokens(main_change)),
            ("edit_line_ids", str(self.edit_line_ids)),
        )

    def line_ids_to_input_lines(self, line_ids: Sequence[int]) -> Sequence[int]:
        """Convert the edit lines (which are line ids including deleted lines) into
        normal line numbers that do not include deleted lines."""
        change_tks = self.span.delta.apply_to_change(self.span.original.tolist())
        input_l = self.span.line_range[0]
        input_lines = list[int]()
        for i, tks in enumerate(tk_splitlines(change_tks)):
            if tks and tks[0] == Del_id:
                continue
            if i in line_ids:
                input_lines.append(input_l)
            input_l += 1

        return input_lines


PyFullName = NewType("PyFullName", str)


@dataclass
class PyDefinition:
    full_name: PyFullName
    start_locs: set[tuple[int, int]]
    signatures: set[str]

    def __post_init__(self):
        self.parent = ".".join(split_dots(self.full_name)[:-1])

    def update(self, name: classes.BaseName):
        if name.type not in ("function", "statement", "class"):
            return
        assert_eq(name.full_name, self.full_name)
        if loc := name.get_definition_start_position():
            self.start_locs.add(loc)

        if name.type == "statement":
            stmt = name._name.tree_name.search_ancestor("simple_stmt")
            if stmt:
                assert isinstance(stmt, ptree.PythonNode)
                self.signatures.add(stmt.get_code(include_prefix=False).strip())
            return

        for sig in name._get_signatures(for_docstring=True):
            self.signatures.add(sig.to_string().strip())


@dataclass(frozen=True)
class LineUsageAnalysis:
    line2usages: Mapping[int, Sequence[PyDefinition]]


class C3ProblemGenerator(ProjectChangeProcessor[C3Problem]):
    """
    ### Change log
    - v2.9: Add sibling usages for class members. Improve statement signatures.
    - v2.8: Fix module usages in `pre_edit_analysis`. Sort changes using heuristic.
    - v2.7: Use new PyDefiniton that includes signatures.
    - v2.6: fix missing changes in `JModuleChanges`. Rename to edit_line_ids.
    - v2.5: fix newline encoding bug.
    - v2.4: fix buggy encoding of `Added` and `Deleted` changes.
    - v2.3: always generate problems with full editing range and move the problem
    splitting logic elsewhere. Also changed the data format of `ChangedCodeSpan`.
    """

    VERSION = "2.9"
    # change spans with more than this many lines will be ignored
    max_span_lines: int = 500
    # change spans with more than this many characters will be ignored
    max_span_chars: int = 6000

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
        result = dict[ModuleName, LineUsageAnalysis]()

        src_map = {m.mname: f for f, m in modules.items()}
        for mname, mchange in changes.items():
            result[mname] = LineUsageAnalysis({})
            if not isinstance(mchange.module_change, Modified):
                continue

            lines_to_analyze = set[int]()
            for span in mchange.changed:
                if not isinstance(span.change, Modified):
                    continue
                lines_to_analyze.update(span.line_range.to_range())
                lines_to_analyze.update(span.header_line_range.to_range())
            if not lines_to_analyze:
                continue

            mod_path = src_map[mname]
            script = pstate.scripts[mod_path]
            line_usages = self.analyzer.get_line_usages(
                script, lines_to_analyze, silent=True
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
        """
        Return (untransformed) c3 problems for the given project change.
        Each problem contains a code change and a list of previous changes.
        """
        before_mod_map = {m.mname: m for m in pchange.all_modules.before}
        cache = C3GeneratorCache(before_mod_map)

        processed_cspans = list[ChangedCodeSpan]()
        problems = list[C3Problem]()
        for m in module_order:
            if (mchange := pchange.changed.get(m)) is None:
                continue
            if not (usages := mod2usages.get(m)):
                usages = LineUsageAnalysis({})
                warnings.warn("Unexpected: usages missing for module: " + str(m))
            for span in mchange.changed:
                code_span = cache.to_code_span(span)
                should_mk_problem = (
                    (span.change.as_char() == Modified.as_char())
                    and (self._is_training or span._is_func_body())
                    and (len(span.change.earlier) <= self.max_span_chars)
                    and (len(span.change.later) <= self.max_span_chars)
                    and (count_lines(span.change.earlier) <= self.max_span_lines)
                    and (count_lines(span.change.later) <= self.max_span_lines)
                )
                if should_mk_problem:
                    # latest changes are more relevant
                    relevant_unchanged = cache.get_relevant_unchanged(code_span, usages)
                    relevant_changes = list(reversed(processed_cspans))
                    relevant_changes = cache.sort_changes(
                        code_span, relevant_unchanged, relevant_changes
                    )
                    src_info: SrcInfo = {
                        "project": pchange.project_name,
                        "commit": pchange.commit_info,
                    }
                    n_lines = span.line_range[1] - span.line_range[0]
                    prob = C3Problem(
                        code_span,
                        range(0, n_lines + 1),  # one additional line for appending
                        relevant_changes=relevant_changes,
                        relevant_unchanged=relevant_unchanged,
                        change_type=span.change.map(lambda _: None),
                        src_info=src_info,
                    )
                    problems.append(prob)

                processed_cspans.append(code_span)
        return problems


class C3GeneratorCache:
    """
    Cache various information needed for constructing C3 problems.
    A new cache should be created for each project change (i.e., each commit).
    """

    def __init__(self, pre_module_map: Mapping[ModuleName, JModule]):
        # stores the changed headers
        self._header_cache = dict[ProjectPath, ChangedHeader]()
        # stores the definitions pre-edit
        self._pre_def_cache = dict[ProjectPath, list[ChangedCodeSpan]]()
        # stores the changed code spans
        self._cspan_cache = dict[tuple[ModuleName, LineRange], ChangedCodeSpan]()
        self._module_map = pre_module_map
        self._mod_hier = ModuleHierarchy.from_modules(pre_module_map)

    def set_module_map(self, pre_module_map: Mapping[ModuleName, JModule]):
        self._module_map = pre_module_map
        self._mod_hier = ModuleHierarchy.from_modules(pre_module_map)

    def clear_caches(
        self, pre_changed: set[ModuleName], post_changed: set[ModuleName]
    ) -> None:
        "Clear outdated caches."
        for k in tuple(self._header_cache):
            if k.module in pre_changed or k.module in post_changed:
                del self._header_cache[k]
        for k in tuple(self._pre_def_cache):
            if k.module in pre_changed:
                del self._pre_def_cache[k]
        for k in tuple(self._cspan_cache):
            if k[0] in pre_changed or k[0] in post_changed:
                del self._cspan_cache[k]

    def create_problem(
        self,
        target: ChangedSpan,
        target_lines: Sequence[int],
        changed: Mapping[ModuleName, JModuleChange],
        target_usages: LineUsageAnalysis,
        src_info: SrcInfo,
    ) -> C3Problem:
        relevant_changes = list[ChangedCodeSpan]()
        changed = dict(changed)
        module = target.module
        target_mc = changed.pop(module)
        all_mc = [target_mc] + list(changed.values())
        for mc in all_mc:
            is_target_mc = mc.module_change.earlier.mname == module
            for cspan in mc.changed:
                if not is_target_mc or cspan.line_range != target.line_range:
                    relevant_changes.append(self.to_code_span(cspan))

        code_span = self.to_code_span(target)
        changed_code = code_span.delta.apply_to_change(code_span.original.tolist())
        target_set = set(target_lines)
        line_ids = list[int]()
        input_l = target.line_range[0]
        for i, tks in enumerate(tk_splitlines(changed_code)):
            if tks and tks[0] == Del_id:
                continue
            if input_l in target_set:
                line_ids.append(i)
            input_l += 1
        code_span = replace(
            code_span, original=TkArray.new(changed_code), delta=TkDelta.empty()
        )
        relevant_unchanged = self.get_relevant_unchanged(code_span, target_usages)

        relevant_changes = self.sort_changes(
            code_span, relevant_unchanged, relevant_changes
        )

        prob = C3Problem(
            code_span,
            line_ids,
            relevant_changes=relevant_changes,
            relevant_unchanged=relevant_unchanged,
            change_type=target.change.map(lambda _: None),
            src_info=src_info,
        )
        return prob

    def get_relevant_unchanged(
        self,
        this_change: ChangedCodeSpan,
        line_usages: LineUsageAnalysis,
    ):
        module = this_change.module
        # parent defs are also considered as used
        name2def = dict[PyFullName, PyDefinition]()
        all_lines = set(this_change.line_range.to_range())
        all_lines.update(this_change.headers[-1].line_range.to_range())
        for l in all_lines:
            for pydef in line_usages.line2usages.get(l, []):
                if pydef.full_name.startswith(module) and any(
                    l in all_lines for l in pydef.start_locs
                ):
                    # skip self references
                    continue
                name2def.setdefault(pydef.full_name, pydef)
        return {k: name2def[k] for k in sorted(name2def.keys())}

    max_distance_penalty = 1000
    usage_bonus = 2000

    def sort_changes(
        self,
        target: ChangedCodeSpan,
        used_defs: Mapping[PyFullName, PyDefinition],
        changed: Sequence[ChangedCodeSpan],
    ) -> Sequence[ChangedCodeSpan]:
        def distance_penalty(cspan: ChangedCodeSpan) -> int:
            if cspan.module != target.module:
                return self.max_distance_penalty
            dis_above = abs(target.line_range[0] - cspan.line_range[1])
            dis_below = abs(cspan.line_range[0] - target.line_range[1])
            return min(self.max_distance_penalty, dis_above, dis_below)

        def usage_penalty(cspan: ChangedCodeSpan) -> int:
            path = cspan.headers[-1].path
            fullname = path.module + "." + path.path
            if fullname in used_defs:
                return -self.usage_bonus
            return 0

        def length_penalty(cspan: ChangedCodeSpan) -> int:
            return len(cspan.original) + cspan.delta.change_size()

        result = list(changed)
        result.sort(
            key=lambda x: distance_penalty(x) + usage_penalty(x) + length_penalty(x)
        )
        return result

    def to_header(self, cs: Change[ChangeScope]) -> ChangedHeader:
        path = cs.earlier.path
        if (ch := self._header_cache.get(path)) is None:
            header_change = cs.map(lambda s: s.header_code.strip("\n"))
            ch = ChangedHeader(
                TkArray.new(change_to_tokens(header_change)),
                cs.earlier.tree.type,
                cs.earlier.header_line_range,
                cs.earlier.path,
            )
            self._header_cache[path] = ch
        return ch

    def to_code_span(self, span: ChangedSpan):
        mod = span.parent_scopes[0].later.path.module
        key = (mod, span.line_range)
        if (cs := self._cspan_cache.get(key)) is not None:
            return cs
        original, delta = line_diffs_to_original_delta(
            change_to_line_diffs(span.change)
        )
        result = ChangedCodeSpan(
            headers=[self.to_header(cs) for cs in span.parent_scopes],
            original=TkArray.new(encode_lines_join(original)),
            delta=delta.to_tk_delta(),
            line_range=span.line_range,
            module=span.module,
        )
        self._cspan_cache[key] = result
        return result


class C3ProblemTransform(ABC):
    "A strategy to generate new C3 problems from the orginal ones."

    @abstractmethod
    def transform(self, prob: C3Problem) -> Sequence[C3Problem]:
        ...


@dataclass
class C3ProblemSimpleSplit(C3ProblemTransform):
    "Simply split the problem into fixed-sized editing ranges."
    max_lines_to_edit: int = 25
    max_split_factor: int = 4

    def transform(self, prob: C3Problem) -> Sequence[C3Problem]:
        delta = prob.span.delta
        l_range = prob.edit_line_ids
        assert isinstance(l_range, range)
        start, stop = l_range.start, l_range.stop
        problems = list[C3Problem]()
        new_trans = prob.transformations + ("split",)
        for i in range(start, stop, self.max_lines_to_edit):
            j = min(i + self.max_lines_to_edit, stop)
            sub_delta = delta.for_input_range((i, j))
            if sub_delta.num_changes() > 0:
                sub_prob = replace(
                    prob, edit_line_ids=range(i, j), transformations=new_trans
                )
                problems.append(sub_prob)
            if len(problems) >= self.max_split_factor:
                break
        return problems


@dataclass
class C3ProblemChangeInlining(C3ProblemTransform):
    """Split the problem into fixed-sized editing ranges like `C3ProblemSimpleSplit`,
    but also randomly keep some subset of changes in the input.

    ### Change log
    - v1.3: make `random_subset` truely random.
    - v1.2: fix newline encoding bug.
    - v1.1
        - Dropout changes using change groups instead of individual change actions.
        - Perform dropout at entire problem level ratehr than chunk level. This way,
    changes in later chunks will be visible as well.
        - Removed `dropout_prob`.
    """

    VERSION = "1.3"

    max_lines_to_edit: int = 25
    max_split_factor: int = 4
    # when dropping the changes into the input, the biggest ratio of changes to drop
    max_dropout_ratio: float = 0.5
    _test_prob: float = 0.01

    def __post_init__(self):
        self._rng = random.Random()

    def transform(self, prob: C3Problem) -> Sequence[C3Problem]:
        original = prob.span.original
        delta = prob.span.delta
        l_range = prob.edit_line_ids
        assert isinstance(l_range, range)
        start, stop = l_range.start, l_range.stop

        grouped_keys = delta.change_groups()
        should_dropout = len(grouped_keys) >= 2
        if should_dropout:
            n_to_drop = int(
                len(grouped_keys) * random.random() * self.max_dropout_ratio
            )
            assert n_to_drop < len(grouped_keys)
            keys_to_drop = join_list(
                random_subset(grouped_keys, n_to_drop, rng=self._rng)
            )
        else:
            keys_to_drop = []
        if keys_to_drop:
            delta1, delta2 = delta.decompose_for_change(keys_to_drop)
            if random.random() < self._test_prob:
                result1 = delta2.apply_to_change(
                    delta1.apply_to_change(original.tolist())
                )
                result2 = delta.apply_to_change(original.tolist())
                code1 = tokens_to_change(result1).after
                code2 = tokens_to_change(result2).after
                if code1 != code2:
                    print_sections(
                        ("result1", decode_tokens(result1)),
                        ("result2", decode_tokens(result2)),
                        ("delta", str(delta)),
                        ("keys_to_drop", str(keys_to_drop)),
                        ("delta1", str(delta1)),
                        ("delta2", str(delta2)),
                    )
                    raise AssertionError("decompose_for_change failed.")
            delta2_groups = delta2.change_groups()
            if not delta2_groups:
                print_err(f"{delta=}, {keys_to_drop=}, {delta1=}")
                raise AssertionError("Empty delta2_groups")
            new_original = TkArray.new(delta1.apply_to_change(original.tolist()))
            new_trans = prob.transformations + ("split", "dropout")
            new_span = replace(prob.span, original=new_original, delta=delta2)
        else:
            new_trans = prob.transformations + ("split",)
            new_span = prob.span
            delta1 = None
            delta2_groups = delta.change_groups()

        prob_and_n = list[tuple[C3Problem, int]]()
        for i in range(start, stop, self.max_lines_to_edit):
            j = min(i + self.max_lines_to_edit, stop)
            edit_line_ids = range(i, j)
            if delta1 is not None:
                edit_line_ids = delta1.get_new_line_ids(edit_line_ids)
            line_set = set(edit_line_ids)
            n_groups = sum(any(key[0] in line_set for key in g) for g in delta2_groups)
            if n_groups > 0:
                sub_prob = replace(
                    prob,
                    span=new_span,
                    edit_line_ids=edit_line_ids,
                    transformations=new_trans,
                )
                prob_and_n.append((sub_prob, n_groups))
        # return the problems with the most changes
        prob_and_n.sort(key=lambda p: p[1], reverse=True)
        probs = [p[0] for p in prob_and_n]
        return probs[: self.max_split_factor]


@dataclass
class C3ToCodeCompletion(C3ProblemTransform):
    """Convert the C3 problem into an edit-oriented code completion problem by
    randomly picking a changed line as the completion target, deleting its
    old version, and treating the new version as the desired output.

    ### Change log
    - empty
    """

    VERSION = "1.0"
    min_target_size = 6

    def __post_init__(self):
        self._rng = random.Random()

    def transform(self, prob: C3Problem) -> Sequence[C3Problem]:
        original = prob.span.original
        delta = prob.span.delta

        def group_filter(group: tuple) -> bool:
            segs = [delta[k] for k in group]
            return (
                segs[0][0] == Add_id
                and sum(len(s) for s in segs) >= self.min_target_size
            )

        add_groups = [ks for ks in delta.change_groups() if group_filter(ks)]
        if not add_groups:
            return []
        # sample the completion target
        target = self._rng.choice(add_groups)

        prev_changes = [k for k in delta.keys() if k < target[0]]
        if delta[target[-1]][0] == Del_id:
            # if the last change is a deletion, move it into prev_changesine into before_changes
            prev_changes.append(target[-1])
            target = target[:-1]
            assert target
        prev_delta, rest_delta = delta.decompose_for_change(prev_changes)
        new_original = prev_delta.apply_to_change(original.tolist())
        new_delta_keys = tuple(rest_delta.keys())[: len(target)]
        new_delta = rest_delta.for_keys(new_delta_keys)
        assert new_delta, "the remaining delta should not be empty"
        new_span = replace(
            prob.span, original=TkArray.new(new_original), delta=new_delta
        )
        new_trans = prob.transformations + ("code_completion",)
        new_lines = tuple(set(k[0] for k in new_delta.keys()))
        new_prob = replace(
            prob,
            span=new_span,
            edit_line_ids=new_lines,
            transformations=new_trans,
        )
        return [new_prob]


@dataclass(frozen=True)
class TkC3Problem(TokenizedEdit):
    "Tokenized contextual code change prediction problem."
    main_input: TkArray
    header: TkArray
    output: TkArray
    path: ProjectPath
    change_type: Change[None]
    # most relevant to least relevant
    named_references: Sequence[tuple[str, TkArray]]
    project: str
    commit: CommitInfo | None
    truncated: bool

    @property
    def references(self) -> Sequence[TkArray]:
        return [ref for _, ref in self.named_references]

    def __repr__(self):
        return f"TkC3Problem(path={self.path}, type={self.change_type.as_char()}, stats={self.stats()})"

    @property
    def input_tks(self) -> TokenSeq:
        return self.header.tolist() + self.main_input.tolist()

    @property
    def output_tks(self) -> TokenSeq:
        return self.output.tolist()

    @property
    def main_tks(self) -> TokenSeq:
        return self.input_tks

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
    """
    ## Change log
    - 2.6: increase max_ref_tks_sum from 512 * 12 to 512 * 16.
    - 2.5: Sort used references by path.
    - 2.4: Encode each changed reference individually. Encode signatures for unchanged.
    """

    VERSION = "2.6"
    max_ref_tks: int = 512
    max_query_tks: int = 512
    max_output_tks: int = 256
    max_scope_tks: int = 128
    max_ref_tks_sum: int = 512 * 16
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
        self._offset_cache = dict[int, TkArray]()

    def tokenize_problem(
        self,
        problem: C3Problem,
    ) -> TkC3Problem:
        span = problem.span

        original: TokenSeq = span.original.tolist()
        tk_delta: TkDelta = span.delta
        origin_lines = tk_splitlines(original)
        edit_start = problem.edit_line_ids[0]
        scope_tks = self._encode_headers(span.headers, 0)
        input_limit = self.max_query_tks - len(scope_tks)

        chunk_input = TokenSeq()
        chunk_output = TokenSeq()
        last_line = edit_start

        for i, l in enumerate(problem.edit_line_ids):
            for line in origin_lines[last_line + 1 : l]:
                chunk_input.extend(line)
                chunk_input.append(Newline_id)

            chunk_input.append(get_extra_id(i))
            if l < len(origin_lines):
                chunk_input.extend(origin_lines[l])
                chunk_input.append(Newline_id)
                last_line = l
            line_change = join_list(tk_delta.get_line_change(l), Newline_id)
            chunk_output.append(get_extra_id(i))
            chunk_output.extend(line_change)
            if line_change and line_change[-1] != Del_id:
                chunk_output.append(Newline_id)
            if len(chunk_input) > input_limit:
                break
        edit_stop = last_line + 1

        # limit the input size if it's too long
        chunk_input = truncate_section(
            chunk_input, TruncateAt.Right, input_limit, inplace=True
        )
        chunk_output = truncate_output_tks(chunk_input, chunk_output)

        # try move some prev_change_tks into the input
        above_tks = join_list(origin_lines[:edit_start] + [TokenSeq()], Newline_id)
        above_tks = tk_delta.for_input_range((0, edit_start)).apply_to_change(above_tks)
        below_tks = join_list(origin_lines[edit_stop:] + [TokenSeq()], Newline_id)
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

        truncated = False
        if ref_size_sum < self.max_ref_tks_sum:
            unchanged = self._group_encode_unchanged_refs(problem.relevant_unchanged)
            for i, chunk in enumerate(unchanged):
                all_refs.append((f"unchanged ref {i}", chunk))
        else:
            truncated = True

        if ref_size_sum < self.max_ref_tks_sum:
            changed = self._group_encode_changed_refs(problem.relevant_changes)
            for i, chunk in enumerate(changed):
                all_refs.append((f"changed ref {i}", chunk))
            ref_size_sum += sum(len(x) for x in changed)
        else:
            truncated = True

        # take until we hit the limit
        ref_size_sum = 0
        kept_refs = list[tuple[str, TkArray]]()
        for name, ref in all_refs:
            if ref_size_sum + len(ref) > self.max_ref_tks_sum:
                truncated = True
                break
            ref_size_sum += len(ref)
            kept_refs.append((name, ref))

        return TkC3Problem(
            TkArray.new(chunk_input),
            TkArray.new(scope_tks),
            TkArray.new(chunk_output),
            path=span.headers[-1].path,
            change_type=problem.change_type,
            named_references=kept_refs,
            project=problem.src_info["project"],
            commit=problem.src_info["commit"],
            truncated=truncated,
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
            tks = TkArray.new(encode_single_line(f"# offset: {offset}"))
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
        self, elems: Mapping[PyFullName, PyDefinition]
    ) -> Sequence[TkArray]:
        def sort_key(e: PyDefinition):
            return (e.parent, min(e.start_locs, default=(0, 0)))

        results = list[TkArray]()
        this_chunk = TokenSeq()
        sorted_elems = [e for e in elems.values() if e.signatures and e.parent]
        sorted_elems.sort(key=sort_key)
        last_parent = None
        for defn in sorted_elems:
            parent = defn.parent
            header = f"at: {parent}\n" if parent != last_parent else ""
            text = header + indent("\n".join(s for s in defn.signatures), TAB) + "\n\n"
            tks = encode_lines_join(text)
            tks = truncate_section(
                tks, TruncateAt.Right, self.max_ref_tks, inplace=True
            )
            if len(this_chunk) + len(tks) > self.max_ref_tks:
                results.append(TkArray.new(this_chunk))
                this_chunk = tks
            else:
                this_chunk.extend(tks)
            last_parent = parent
        if this_chunk:
            results.append(TkArray.new(this_chunk))
        return results

    def _group_encode_changed_refs(
        self, changes: Sequence[ChangedCodeSpan]
    ) -> Sequence[TkArray]:
        return join_list(self._encode_changed_ref(x) for x in changes)

    def _encode_changed_ref(self, cspan: ChangedCodeSpan) -> Iterable[TkArray]:
        "encode a single changed reference"
        cspan_tks = cspan.delta.apply_to_change(cspan.original.tolist())
        for chunk in break_into_chunks(
            cspan_tks,
            lambda i: self._encode_headers(cspan.headers, i),
            self.max_ref_tks,
            overlap=self.ref_chunk_overlap,
        ):
            yield TkArray.new(chunk)

    def compute_stats(self, problems: Sequence[C3Problem]):
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
    include_parent_usages: bool = True

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
        lines_to_analyze: Collection[int],
        silent: bool = False,
    ):
        jmod: tree.Module = script._module_node
        name2def_node = dict[PyFullName, list[classes.Name]]()
        line2usages = dict[int, set[PyFullName]]()
        registered_classes = set[PyFullName]()

        def register_usage(cname: classes.Name, usages: set[PyFullName]):
            fname = cname.full_name
            if fname is None:
                return
            fname = PyFullName(fname)
            usages.add(fname)
            name2def_node.setdefault(fname, list()).append(cname)
            # FIXME: wait until the jedi bug is fixed
            # if self.include_parent_usages and (parent := cname.parent()):
            # if cname.type == "statement" and parent.type == "class":
            # register_usage(parent, usages)

        def register_class_usage(cname: classes.Name, usages: set[PyFullName]):
            assert_eq(cname.type, "class")
            if not cname.full_name or cname.full_name in registered_classes:
                return
            for n in cname.defined_names():
                if n.type == "statement":
                    register_usage(n, usages)
            registered_classes.add(PyFullName(cname.full_name))

        all_names = [
            name for names in jmod.get_used_names()._dict.values() for name in names
        ]
        all_names.sort(key=lambda x: x.start_pos)
        all_names = [n for n in all_names if n.start_pos[0] in lines_to_analyze]
        name2pydef = dict[PyFullName, PyDefinition]()

        try:
            for name in tqdm(all_names, f"Analyzing {script.path}", disable=silent):
                name: tree.Name
                line = name.start_pos[0]
                usages = line2usages.setdefault(line, set())
                cnames = _fast_goto(
                    script,
                    name,
                    follow_imports=True,
                    follow_builtin_imports=False,
                )
                for cname in cnames:
                    register_usage(cname, usages)
                    if (parent := cname.parent()) and parent.type == "class":
                        register_class_usage(parent, usages)

            for fname, nodes in name2def_node.items():
                pdef = PyDefinition(fname, set(), set())
                for n in nodes:
                    pdef.update(n)
                name2pydef[fname] = pdef
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            err_text = repr(e)
            str_limit = 80
            if len(err_text) > str_limit:
                err_text = err_text[:str_limit] + "..."
            self.add_error(err_text)

        return LineUsageAnalysis(
            {
                k: [name2pydef[n] for n in sorted(names) if n in name2pydef]
                for k, names in line2usages.items()
            }
        )

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


class ModuleHierarchy:
    def __init__(self):
        self.children = dict[str, "ModuleHierarchy"]()
        # maps from implcit relative imports to the modules that they actually refer to
        self._implicit_imports: dict[tuple[ModuleName, ModuleName], ModuleName] = dict()

    def __repr__(self):
        return f"ModuleNamespace({self.children})"

    def add_module(self, segs: Sequence[str]) -> None:
        namespace = self
        for s in segs:
            if s in namespace.children:
                namespace = namespace.children[s]
            else:
                namespace.children[s] = ModuleHierarchy()
                namespace = namespace.children[s]

    def has_module(self, segs: Sequence[str]) -> bool:
        namespace = self
        for s in segs:
            if s in namespace.children:
                namespace = namespace.children[s]
            else:
                return False
        return True

    def resolve_path(self, segs: Sequence[str]) -> ProjectPath | None:
        if len(segs) < 2:
            return None
        namespace = self
        matched = 0
        for s in segs[:-1]:
            if s in namespace.children:
                namespace = namespace.children[s]
                matched += 1
            else:
                break
        if matched == 0:
            return None
        return ProjectPath(".".join(segs[:matched]), ".".join(segs[matched:]))

    @staticmethod
    def from_modules(modules: Iterable[str]) -> "ModuleHierarchy":
        root = ModuleHierarchy()
        for m in modules:
            root.add_module(split_dots(m))
        return root


def sort_modules_by_imports(
    imports: Mapping[ModuleName, set[ModuleName]]
) -> list[ModuleName]:
    "Sort modules topologically according to imports"
    sorted_modules = list[str]()
    visited = set[str]()

    def visit(m: str) -> None:
        if m in visited or m not in imports:
            return
        visited.add(m)
        if m in imports:
            for m2 in imports[m]:
                visit(m2)
        sorted_modules.append(m)

    for m in imports:
        visit(m)
    return sorted_modules


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
