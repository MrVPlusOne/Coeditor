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


@dataclass(unsafe_hash=True)
class PyDefinition:
    full_name: PyFullName
    module: ModuleName
    file: Path
    start_pos: tuple[int, int]
    end_pos: tuple[int, int]

    @staticmethod
    def from_signatures(
        name: classes.BaseName, project: Path | None = None
    ) -> Iterable["PyDefinition"]:
        if name.in_builtin_module():
            return
        for sig in name.get_signatures():
            if (
                not sig.in_builtin_module()
                and (full_name := sig.full_name)
                and (file := sig.module_path)
                and (project in file.parents)
                and (module := sig.module_name)
                and (start_pos := sig.get_definition_start_position())
                and (end_pos := sig.get_definition_end_position())
            ):
                full_name = PyFullName(full_name)
                yield PyDefinition(full_name, module, file, start_pos, end_pos)


@dataclass
class LineUsageAnalysis:
    line2usages: Mapping[int, set[PyDefinition]]


class CtxCodeChangeProblemGenerator(ProjectChangeProcessor[CtxCodeChangeProblem]):
    def __init__(self, analysis: "JediUsageAnalysis | None"):
        if analysis is None:
            analysis = JediUsageAnalysis()
        self.analysis = analysis

    def pre_edit_analysis(
        self,
        project: jedi.Project,
        modules: Mapping[RelPath, JModule],
        file_changes: Sequence[Change[str]],
    ) -> Mapping[ModuleName, LineUsageAnalysis]:
        "Return the definition usages of each line."
        # proot = Path(project._path)
        result = dict[ModuleName, LineUsageAnalysis]()
        for change in file_changes:
            if not isinstance(change, Modified):
                continue
            mod_path = RelPath(Path(change.before))
            jmod = modules[mod_path]
            assert (project.path / mod_path).exists()
            script = jedi.Script(path=project.path / mod_path, project=project)
            line_usages = self.analysis.get_module_usages(
                script, project.path, silent=True
            )
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
            path = span.parent_scopes[-1].earlier().path
            line_usages = mod2usages[path.module]
            all_used = set[PyDefinition]()
            l_start, l_end = span.line_range
            for l in range(l_start, l_end + 1):
                for pydef in line_usages.line2usages.get(l, set()):
                    if (
                        pydef.module == path.module
                        and l_start <= pydef.start_pos[0] <= l_end
                    ):
                        # skip self references
                        print(f"Skip: {pydef}")
                        continue
                    all_used.add(pydef)

            result = list[tuple[ProjectPath, str]]()
            for used in all_used:
                result.append((ProjectPath(used.full_name, ""), str(used)))
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
    max_scope_tks: int = 128
    max_lines_to_edit: int = 20
    ref_chunk_overlap: int = 32
    max_chunks_per_ref: int = 4
    max_lines_per_function: int = 500
    skip_unchanged_problems: bool = True

    def __post_init__(self):
        self._id_cache = FIFOCache[_ObjId, TokenSeq](maxsize=1000)
        self._scope_cache = FIFOCache[_ObjId, TokenSeq](maxsize=1000)
        self._value_cache = FIFOCache[Any, Sequence[TokenSeq]](maxsize=1000)

    def encode_problem(
        self,
        problem: CtxCodeChangeProblem,
    ) -> Iterable[TkCtxCodeChangeProblem]:
        span = problem.span
        named_references = list[tuple[str, TokenSeq]]()
        # compute the references that are relevant to this span
        relevant_chunks = self._group_encode_changed_refs(problem.relevant_changes)
        for i, chunk in enumerate(relevant_chunks):
            named_references.append((f"changed ref {i}", chunk))
        for i, (path, unchanged) in enumerate(problem.relevant_unchanged):
            for j, ref in enumerate(self._encode_unchanged_ref(path, unchanged)):
                named_references.append((f"unchanged ref {i}-{j}", ref))

        diffs = change_to_line_diffs(span.change)
        original, delta = line_diffs_to_original_delta(diffs)
        origin_lines = split_list(encode_basic(original), Newline_id)
        tk_delta = delta.to_tk_delta()
        chunk_id = 0
        chunk_start_l = 0
        scope_tks = self._encode_parent_scopes(span.parent_scopes, 0)
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
                lambda i: self._encode_parent_scopes(span.parent_scopes, -1 - i),
                chunk_size=self.max_ref_tks,
                overlap=self.ref_chunk_overlap,
                right_to_left=True,
            )
            if finished:
                below_chunks = []
            else:
                below_chunks = break_into_chunks(
                    below_tks,
                    lambda i: self._encode_parent_scopes(span.parent_scopes, i + 1),
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
                path=span.parent_scopes[-1].earlier().path,
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

    def _encode_scope_change(self, c: Change[ChangeScope]) -> TokenSeq:
        if (key := _ObjId(id(c))) in self._scope_cache:
            return self._scope_cache[key]
        hchange = c.map(lambda s: s.header_code)
        tks = truncate_section(
            change_to_tokens(hchange), TruncateAt.Left, self.max_scope_tks
        )
        self._scope_cache[key] = tks
        return tks

    def _encode_parent_scopes(
        self, scope_changes: Sequence[Change[ChangeScope]], offset: int
    ) -> TokenSeq:
        scope_tks = join_list(
            (self._encode_scope_change(c) for c in scope_changes), sep=Newline_id
        )
        if offset != 0:
            ending = encode_basic(f"\n# offset: {offset}\n")
        else:
            ending = [Newline_id]
        scope_tks = truncate_section(
            scope_tks + ending, TruncateAt.Left, self.max_scope_tks
        )
        return scope_tks

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

    def _encode_unchanged_ref(
        self, path: ProjectPath, content: str
    ) -> Iterable[TokenSeq]:
        if (key := (path, content)) in self._value_cache:
            return self._value_cache[key]
        main_tks = encode_basic(f"#{str(path)}\n{content}")
        ref_chunks = (truncate_section(main_tks, TruncateAt.Right, self.max_ref_tks),)
        self._value_cache[key] = ref_chunks
        return ref_chunks

    def _encode_change(self, change: Change[str]) -> TokenSeq:
        if (key := _ObjId(id(change))) in self._id_cache:
            return self._id_cache[key]
        change_tks = change_to_tokens(change)
        self._id_cache[key] = change_tks
        return change_tks

    def _group_encode_changed_refs(
        self, changes: Sequence[ChangedSpan]
    ) -> Sequence[TokenSeq]:
        module2changes = groupby(changes, lambda c: c.path.module)
        all_chunks = list[TokenSeq]()
        for change_group in module2changes.values():
            change_group.sort(key=lambda c: c.line_range[0])
            file_tks = TokenSeq()
            # we'll add module as the chunk header, so we start within the module
            last_scope = change_group[0].parent_scopes[:1]
            for c in change_group:
                scope_diff = []
                for i, s in enumerate(c.parent_scopes):
                    if (
                        i >= len(last_scope)
                        or s.earlier().path != last_scope[i].earlier().path
                    ):
                        scope_diff.append(s)
                if scope_diff:
                    header_tks = self._encode_parent_scopes(scope_diff, 0)
                    file_tks.extend(header_tks)
                body_tks = self._encode_change(c.change)
                file_tks.extend(body_tks)
                file_tks.append(Newline_id)
                last_scope = c.parent_scopes

            mod_change = change_group[0].parent_scopes[:1]
            mod_chunks = break_into_chunks(
                file_tks,
                lambda i: self._encode_parent_scopes(mod_change, i),
                self.max_ref_tks,
                overlap=self.ref_chunk_overlap,
                max_return_chunks=self.max_chunks_per_ref,
            )
            all_chunks.extend(mod_chunks)
        return all_chunks


@dataclass
class JediUsageAnalysis:
    follow_imports: bool = True
    only_same_project_usages: bool = False

    def __post_init__(self):
        self.error_counts = dict[str, int]()
        self.tlogger: TimeLogger = TimeLogger()

    def get_module_usages(
        self, script: jedi.Script, proj_root: Path, silent: bool = False
    ):
        jmod: tree.Module = script._module_node
        line2usages = dict[int, set[PyDefinition]]()
        all_names = [
            name for k, names in jmod.get_used_names()._dict.items() for name in names
        ]
        all_names.sort(key=lambda x: x.start_pos)
        errors = self.error_counts
        resolve_cache = dict[_ObjId, set[PyDefinition]]()
        for name in tqdm(all_names, f"Analyzing {script.path}", disable=silent):
            name: tree.Name
            line = name.start_pos[0]
            usages = line2usages.setdefault(line, set())
            try:
                defs = fast_goto(
                    script,
                    name,
                    follow_imports=self.follow_imports,
                    follow_builtin_imports=False,
                )
                for d in defs:
                    key = _ObjId(id(d))
                    if (defs := resolve_cache.get(key)) is None:
                        defs = set(PyDefinition.from_signatures(d, proj_root))
                        resolve_cache[key] = defs
                    usages.update(defs)

            except (AttributeError, AssertionError) as e:
                text = repr(e)
                errors[text] = errors.setdefault(text, 0) + 1
            except ValueError as e:
                # if the message is "not enough values to unpack"
                if "not enough values to unpack (expected 2" in str(e):
                    errors[repr(e)] = errors.setdefault(str(e), 0) + 1
                else:
                    raise
        return LineUsageAnalysis(line2usages)


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
