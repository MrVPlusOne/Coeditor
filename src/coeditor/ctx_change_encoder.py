from os import PathLike
from coeditor.encoders import has_change
from coeditor.encoding import (
    Del_id,
    Newline_id,
    TokenizedEdit,
    TruncateAt,
    break_into_chunks,
    change_to_line_diffs,
    change_to_tokens,
    encode_basic,
    get_extra_id,
    line_diffs_to_original_delta,
    truncate_output_tks,
    truncate_section,
    truncate_sections,
)
from coeditor.history import Added, Change, Modified
from spot.static_analysis import (
    ModuleName,
    ProjectPath,
    PythonProject,
    sort_modules_by_imports,
)
from .common import *
import jedi, parso
from parso.python import tree
from jedi.api import helpers, convert_names, classes
from jedi import cache
from .code_change import (
    ChangeScope,
    ChangedSpan,
    JModule,
    JProjectChange,
    BasicTkQueryEdit,
    ProjectChangeProcessor,
    PyNode,
)
from parso.python import tree as ptree
from cachetools import FIFOCache
import parso.tree as parso_tree
import jedi.cache

jedi.cache.clear_time_caches = lambda: None


@dataclass
class CtxCodeChangeProblem:
    span: ChangedSpan
    # most relevant to least relevant
    relevant_changes: list[ChangedSpan]
    # most relevant to least relevant
    relevant_unchanged: list[tuple[ProjectPath, str]]


PyFullName = NewType("PyPathStr", str)


@dataclass
class LineUsageAnalysis:
    line2usages: dict[int, set[PyFullName]]


@dataclass
class CtxCodeChangeProblemGenerator(ProjectChangeProcessor[CtxCodeChangeProblem]):
    def pre_edit_analysis(
        self,
        project: jedi.Project,
        modules: Mapping[RelPath, JModule],
        file_changes: Sequence[Change[str]],
    ) -> Mapping[ModuleName, LineUsageAnalysis]:
        "Return the definition usages of each line."
        analysis = JediUsageAnalysis(project)
        # proot = Path(project._path)
        result = dict[ModuleName, LineUsageAnalysis]()
        for change in file_changes:
            if not isinstance(change, Modified):
                continue
            mod_path = RelPath(Path(change.before))
            jmod = modules[mod_path]
            assert (project.path / mod_path).exists()
            script = jedi.Script(path=project.path / mod_path, project=project)
            line_usages, script = analysis.get_module_usages(script, silent=True)
            result[jmod.mname] = line_usages
        return result

    def post_edit_analysis(
        self,
        project: jedi.Project,
        modules: Mapping[RelPath, JModule],
        file_changes,
    ) -> list[ModuleName]:
        "Return the topological order among the modules."
        # sort modules topologically
        module_deps = dict[ModuleName, set[ModuleName]]()
        for rel_path, module in modules.items():
            assert (project.path / rel_path).exists()
            script = jedi.Script(path=project._path / rel_path, project=project)
            deps = module_deps.setdefault(module.mname, set())
            for n in module.imported_names:
                n = script._module_node.get_name_of_position(n.start_pos)
                for source in fast_goto(
                    script, n, follow_imports=True, follow_builtin_imports=False
                ):
                    deps.add(source.module_name)
        module_order = sort_modules_by_imports(module_deps)
        return module_order

    def encode_change(
        self,
        pchange: JProjectChange,
        mod2usages: Mapping[ModuleName, LineUsageAnalysis],
        module_order: Sequence[ModuleName],
    ) -> Iterable[CtxCodeChangeProblem]:
        def _get_relevant(span: ChangedSpan):
            if isinstance(span.change, Added):
                # nothing to analyze
                return []
            path = span.scope_change.earlier().path
            line_usages = mod2usages[path.module]
            all_used = set[PyFullName]()
            for l in range(*span.line_range):
                all_used.update(line_usages.line2usages.get(l, tuple()))

            result = list[tuple[ProjectPath, str]]()
            for used in all_used:
                result.append((ProjectPath("?", used), used))
            return result

        sorted_cspans = list[ChangedSpan]()
        for m in module_order:
            if (mchange := pchange.changed.get(m)) is None:
                continue
            for span in mchange.changed.values():
                if span.change.as_char() == Modified.as_char():
                    yield CtxCodeChangeProblem(
                        span,
                        relevant_changes=sorted_cspans.copy(),
                        relevant_unchanged=_get_relevant(span),
                    )
                sorted_cspans.append(span)


@dataclass
class TkCtxCodeChangeProblem(TokenizedEdit):
    input_tks: TokenSeq
    output_tks: TokenSeq
    path: ProjectPath
    change_type: Change[None]
    # most relevant to least relevant
    named_references: Sequence[tuple[str, TokenSeq]]

    @property
    def references(self) -> Sequence[TokenSeq]:
        return [ref for name, ref in self.named_references]

    def __repr__(self):
        return f"TkCtxCodeChangeProblem(path={self.path}, type={self.change_type.as_char()}, stats={self.stats()})"

    @property
    def main_tks(self):
        return self.input_tks

    def show(self) -> str:
        return self.show_prediction(None)

    def all_ctxs(self) -> dict[str, TokenSeq]:
        return {name: ref for name, ref in self.named_references}

    def meta_data_lines(self) -> list[str]:
        return [
            f"path: {self.path}",
            f"n_references: {len(self.references)}",
            f"total_reference_tks: {sum(len(ref) for ref in self.references)}",
        ]

    def stats(self) -> Mapping[str, int | float]:
        return {
            "input_tks": len(self.input_tks),
            "output_tks": len(self.output_tks),
            "n_references": len(self.references),
            "total_reference_tks": sum(len(ref) for ref in self.references),
        }


_ObjId = NewType("_ObjId", int)


@dataclass
class TkCtxCodeChangeEncoder:
    VERSION = "0.0"
    max_ref_tks: int = 512
    max_query_tks: int = 512
    max_output_tks: int = 256
    max_lines_to_edit: int = 20
    ref_chunk_overlap: int = 32
    max_chunks_per_ref: int = 4
    max_lines_per_function: int = 500
    skip_unchanged_problems: bool = True

    def __post_init__(self):
        self._id_cache = FIFOCache[_ObjId, Sequence[TokenSeq]](maxsize=1000)
        self._value_cache = FIFOCache[Any, Sequence[TokenSeq]](maxsize=1000)

    def encode_problem(
        self,
        problem: CtxCodeChangeProblem,
    ) -> Iterable[TkCtxCodeChangeProblem]:
        span = problem.span
        named_references = list[tuple[str, TokenSeq]]()
        # compute the references that are relevant to this span
        for ref_span in problem.relevant_changes:
            for i, ref in enumerate(self._encode_changed_ref(ref_span)):
                named_references.append((f"changed ref {i}", ref))
        for path, unchanged in problem.relevant_unchanged:
            for i, ref in enumerate(self._encode_unchanged_ref(path, unchanged)):
                named_references.append((f"unchanged ref {i}", ref))

        diffs = change_to_line_diffs(span.change)
        original, delta = line_diffs_to_original_delta(diffs)
        origin_lines = split_list(encode_basic(original), Newline_id)
        tk_delta = delta.to_tk_delta()
        chunk_id = 0
        chunk_start_l = 0
        scope_tks = self._encode_scope(span.scope_change, 0)
        chunk_input = TokenSeq()
        input_limit = self.max_query_tks - len(scope_tks)
        chunk_lines = 0
        chunk_output = TokenSeq()
        prev_change_tks = TokenSeq()

        def get_problem(chunk_input, chunk_output):
            # try move some prev_change_tks into the input
            above_tks = prev_change_tks
            below_tks = join_list(origin_lines[l:], Newline_id)
            chunk_input, above_tks, below_tks = self._inline_some_context(
                chunk_input, above_tks, below_tks, input_limit
            )

            # limit the input size if it's too long (can happen for later chunks)
            chunk_input = truncate_section(chunk_input, TruncateAt.Right, input_limit)
            chunk_output = truncate_output_tks(chunk_input, chunk_output)
            chunk_output = truncate_section(
                chunk_output, TruncateAt.Right, self.max_output_tks, add_bos=False
            )

            above_chunks = break_into_chunks(
                above_tks,
                lambda i: self._encode_scope(span.scope_change, -1 - i),
                chunk_size=self.max_ref_tks,
                overlap=self.ref_chunk_overlap,
                right_to_left=True,
            )
            if finished:
                below_chunks = []
            else:
                below_chunks = break_into_chunks(
                    below_tks,
                    lambda i: self._encode_scope(span.scope_change, i + 1),
                    chunk_size=self.max_ref_tks,
                    overlap=self.ref_chunk_overlap,
                )
            above_chunks = [
                (f"above chunk {i}", chunk) for i, chunk in enumerate(above_chunks)
            ]
            below_chunks = [
                (f"below chunk {i}", chunk) for i, chunk in enumerate(below_chunks)
            ]
            return TkCtxCodeChangeProblem(
                scope_tks + chunk_input,
                chunk_output,
                path=span.scope_change.earlier().path,
                change_type=span.change.map(lambda _: None),
                named_references=above_chunks + below_chunks + named_references,
            )

        for l in range(len(tk_delta.deltas) + 1):
            finished = l == len(tk_delta.deltas)
            input_growth = len(origin_lines[l]) + 2 if l < len(origin_lines) else 1
            if (
                finished
                or chunk_lines >= self.max_lines_to_edit
                or len(chunk_input) + input_growth > input_limit
            ):
                if has_change(chunk_output):
                    yield get_problem(chunk_input, chunk_output)

                if finished:
                    break

                chunk_main_input = join_list(origin_lines[chunk_start_l:l], Newline_id)
                chunk_main_delta = tk_delta.for_input_range((chunk_start_l, l))
                chunk_main_change = chunk_main_delta.to_change_tks(chunk_main_input)
                prev_change_tks.extend(chunk_main_change)
                prev_change_tks.append(Newline_id)
                chunk_id += 1
                chunk_input = TokenSeq()
                chunk_lines = 0
                chunk_output = TokenSeq()
                chunk_start_l = l

            chunk_input.append(get_extra_id(chunk_lines))
            if l < len(origin_lines):
                chunk_input.extend(origin_lines[l])
                chunk_input.append(Newline_id)
            line_change = join_list(tk_delta.deltas[l], Newline_id)
            chunk_output.append(get_extra_id(chunk_lines))
            chunk_output.extend(line_change)
            if line_change and line_change[-1] != Del_id:
                chunk_output.append(Newline_id)
            chunk_lines += 1

    def _encode_scope(self, scope_change: Change[ChangeScope], offset: int) -> TokenSeq:
        hchange = scope_change.map(lambda s: s.header_code)
        if offset != 0:
            ending = encode_basic(f"\n# offset: {offset}\n")
        else:
            ending = [Newline_id]
        return change_to_tokens(hchange) + ending

    def _inline_some_context(
        self,
        input: TokenSeq,
        above_ctx: TokenSeq,
        below_ctx: TokenSeq,
        size_limit: int,
    ) -> tuple[TokenSeq, TokenSeq, TokenSeq]:
        "try move some some of the ctx tokens into the input if there's space."
        extra_space = size_limit - len(input)
        if above_ctx and extra_space > 0:
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

    def _encode_unchanged_ref(
        self, path: ProjectPath, content: str
    ) -> Iterable[TokenSeq]:
        if (key := (path, content)) in self._value_cache:
            return self._value_cache[key]
        main_tks = encode_basic(f"#{str(path)}\n{content}")
        ref_chunks = (truncate_section(main_tks, TruncateAt.Right, self.max_ref_tks),)
        self._value_cache[key] = ref_chunks
        return ref_chunks

    def _encode_changed_ref(self, changed: ChangedSpan) -> Sequence[TokenSeq]:
        if (key := _ObjId(id(changed))) in self._id_cache:
            return self._id_cache[key]
        change_tks = change_to_tokens(changed.change)
        ref_chunks = break_into_chunks(
            change_tks,
            lambda i: self._encode_scope(changed.scope_change, i),
            self.max_ref_tks,
            overlap=self.ref_chunk_overlap,
            max_return_chunks=self.max_chunks_per_ref,
        )
        self._id_cache[key] = ref_chunks
        return ref_chunks


@dataclass
class JediUsageAnalysis:
    jproj: jedi.Project
    follow_imports: bool = True
    only_same_project_usages: bool = True

    def __post_init__(self):
        self.errors = dict[str, int]()
        self.tlogger: TimeLogger = TimeLogger()
        self.proj_root = self.jproj._path
        assert isinstance(self.proj_root, Path)

    def get_module_usages(self, script: jedi.Script, silent: bool = False):
        jmod: tree.Module = script._module_node
        line_usages = dict[int, set[PyFullName]]()
        all_names = [
            name for k, names in jmod.get_used_names()._dict.items() for name in names
        ]
        all_names.sort(key=lambda x: x.start_pos)
        errors = self.errors
        for name in tqdm(all_names, f"Analyzing {script.path}", disable=silent):
            name: tree.Name
            line = name.start_pos[0]
            line_usages.setdefault(line, set())
            try:
                defs = fast_goto(
                    script,
                    name,
                    follow_imports=self.follow_imports,
                    follow_builtin_imports=False,
                )
                for d in defs:
                    if (
                        d.module_path
                        and d.full_name
                        and (
                            not self.only_same_project_usages
                            or (self.proj_root in d.module_path.parents)
                        )
                    ):
                        line_usages[line].add(PyFullName(d.full_name))
            except (AttributeError, AssertionError) as e:
                text = str(e)
                errors[text] = errors.setdefault(text, 0) + 1
        return LineUsageAnalysis(line_usages), script


def fast_goto(
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
