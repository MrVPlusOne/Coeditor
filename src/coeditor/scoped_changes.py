import copy
import shutil
import sys
import time
import warnings
from abc import ABC, abstractmethod
from functools import cached_property

import jedi
import jedi.settings
import parso
from jedi.file_io import FileIO, FolderIO
from jedi.inference.context import ModuleContext
from jedi.inference.references import recurse_find_python_files
from parso.python import tree as ptree
from parso.tree import BaseNode, NodeOrLeaf

from ._utils import rec_iter_files
from .change import Added, Change, Deleted, Modified, get_named_changes
from .common import *
from .encoding import change_to_line_diffs, line_diffs_to_original_delta
from .git import CommitInfo

ScopeTree = ptree.Function | ptree.Class | ptree.Module
PyNode = ptree.PythonBaseNode | ptree.PythonNode


class LineRange(NamedTuple):
    start: int
    until: int

    def __contains__(self, l: int) -> bool:
        return self.start <= l < self.until

    def to_range(self) -> range:
        return range(self.start, self.until)


_tlogger = TimeLogger()


def line_range(start: int, end: int, can_be_empty: bool = False) -> LineRange:
    if not can_be_empty and start >= end:
        raise ValueError(f"Bad line range: {start=}, {end=}")
    return LineRange(start, end)


def _strip_empty_lines(s: str):
    s1 = s.lstrip("\n")
    s2 = s1.rstrip("\n")
    e_lines_left = len(s) - len(s1)
    e_lines_right = len(s1) - len(s2)
    return s2, e_lines_left, e_lines_right


@dataclass
class ChangeScope:
    """
    A change scope is a python module, non-hidden function, or a non-hidden class, or a python module.
        - functions and classes that are inside a parent function are considered hidden.
    """

    path: ProjectPath
    tree: ScopeTree
    spans: Sequence["StatementSpan"]
    subscopes: Mapping[str, Self]
    parent_scope: "ChangeScope | None"

    def __post_init__(self):
        # compute header
        if isinstance(self.tree, ptree.Module):
            header_code = f"# module: {self.path.module}"
            header_line_range = line_range(0, 0, can_be_empty=True)
        else:
            h_start, h_end = 0, 0
            tree = self.tree
            to_visit = list[NodeOrLeaf]()
            parent = not_none(tree.parent)
            while parent.type in ("decorated", "async_funcdef"):
                to_visit.insert(0, parent.children[0])
                parent = not_none(parent.parent)
            to_visit.extend(tree.children)
            visited = list[NodeOrLeaf]()
            for c in to_visit:
                if c.type == "suite":
                    break
                visited.append(c)
            header_code = "".join(cast(str, c.get_code()) for c in visited)
            header_code, e_left, e_right = _strip_empty_lines(header_code)
            h_start = not_none(visited[0].get_start_pos_of_prefix())[0] + e_left
            h_end = visited[-1].end_pos[0] + 1 - e_right
            assert_eq(count_lines(header_code), h_end - h_start)
            header_line_range = line_range(h_start, h_end)
            if self.spans and h_end > self.spans[0].line_range[0]:
                raise ValueError(
                    f"Header covers the fisrt span: {self.path=}, {h_start=}, {h_end=} "
                    f"{self.spans[0].line_range=}"
                )

        self.header_code: str = header_code + "\n"
        self.header_line_range: LineRange = header_line_range

    def ancestors(self) -> list[Self]:
        scope = self
        result = [scope]
        while scope := scope.parent_scope:
            result.append(scope)
        result.reverse()
        return result

    @cached_property
    def spans_code(self) -> str:
        return "\n".join(s.code for s in self.spans)

    @cached_property
    def all_code(self) -> str:
        return self.header_code + self.spans_code

    def search_span_by_line(self, line: int) -> "StatementSpan | None":
        # TODO: optimize this to avoid linear scan
        span = self._search_span(line)
        if span is not None:
            return span
        for s in self.subscopes.values():
            span = s.search_span_by_line(line)
            if span is not None:
                return span

    def _search(self, path: ElemPath, line: int) -> Self | "StatementSpan":
        scope = self._search_scope(path)
        if scope.header_line_range[0] <= line < scope.header_line_range[1]:
            return scope
        span = scope._search_span(line)
        return span or scope

    def _search_scope(self, path: ElemPath) -> Self:
        """Find the scope that can potentially contain the given path. Follow the
        path segments until no more subscopes are found."""
        segs = split_dots(path)
        scope = self
        for s in segs:
            if s in scope.subscopes:
                scope = scope.subscopes[s]
            else:
                break
        return scope

    def _search_span(self, line: int) -> "StatementSpan | None":
        for span in self.spans:
            if line in span.line_range:
                return span
        return None

    @staticmethod
    def from_tree(path: ProjectPath, tree: ScopeTree) -> "ChangeScope":
        spans = []
        subscopes = dict()
        scope = ChangeScope(path, tree, spans, subscopes, None)
        assert isinstance(tree, ScopeTree)
        is_func = isinstance(tree, ptree.Function)

        def mk_span(stmts):
            # remove leading newlines
            n_leading_newlines = 0
            for s in stmts:
                if s.type == ptree.Newline.type:
                    n_leading_newlines += 1
                else:
                    break
            if n_leading_newlines:
                stmts = stmts[n_leading_newlines:]
            if stmts:
                yield StatementSpan(len(spans), stmts, scope)

        current_stmts = []
        container = tree if isinstance(tree, ptree.Module) else tree.get_suite()
        if isinstance(container, BaseNode):
            content = container.children
        else:
            content = []
        for s in content:
            # we don't create inner scopes for function contents
            if is_func or _is_scope_statement(as_any(s)):
                current_stmts.append(s)
            else:
                if current_stmts:
                    spans.extend(mk_span(current_stmts))
                    current_stmts = []
        if current_stmts:
            spans.extend(mk_span(current_stmts))

        if is_func:
            # we don't create inner scopes for function contents
            if not spans:
                raise ValueError(f"Function with no spans: {path=}, {tree.get_code()=}")
            return scope
        for stree in tree._search_in_scope(ptree.Function.type, ptree.Class.type):
            stree: ptree.Function | ptree.Class
            name = cast(ptree.Name, stree.name).value
            spath = path.append(name)
            subscope = ChangeScope.from_tree(spath, stree)
            subscope.parent_scope = scope
            subscopes[name] = subscope
        return scope

    def __repr__(self):
        return (
            f"ChangeScope(path={self.path}, type={self.tree.type}, spans={self.spans})"
        )


_non_scope_stmt_types = {
    "decorated",
    "async_stmt",
    ptree.Class.type,
    ptree.Function.type,
}


def _is_scope_statement(stmt: PyNode) -> bool:
    """Will only return False for functions, classes, and import statments"""
    if stmt.type in _non_scope_stmt_types:
        return False
    if stmt.type == "simple_stmt" and stmt.children[0].type in ptree._IMPORTS:
        return False
    return True


@dataclass
class StatementSpan:
    """
    A statement span is a set of lines inside the same change scope. It is the basic unit of code changes handled by our model.
        - For a modified function, the span is the function itself.
        - For a modified module, the spans are the regions between the functions and classes plus
        the spans recursively generated.
        - For a modified class, the spans are the regions between methods plus
        the spans recursively generated.
    """

    nth_in_parent: int
    statements: Sequence[PyNode]
    scope: ChangeScope

    def __post_init__(self):
        assert self.statements
        origin_code = "".join(s.get_code() for s in self.statements)
        code, e_left, e_right = _strip_empty_lines(origin_code)
        start = not_none(self.statements[0].get_start_pos_of_prefix())[0] + e_left
        end = self.statements[-1].end_pos[0] + 1 - e_right

        self.code: str = code + "\n"
        try:
            self.line_range: LineRange = line_range(start, end)
        except ValueError:
            print_err(f"{e_right=}, {start=}, {end=}")
            print_err("Origin code:")
            print_err(origin_code)
            print_err("Stmts:")
            for s in self.statements:
                print_err(s)
            raise

    def __repr__(self):
        preview = self.code
        str_limit = 30
        if len(preview) > str_limit:
            preview = preview[:str_limit] + "..."
        return f"StatementSpan({self.line_range}, code={repr(preview)})"


@dataclass(frozen=True)
class ChangedSpan:
    "Represents the changes made to a statement span."
    change: Change[str]
    parent_scopes: Sequence[Change[ChangeScope]]
    line_range: LineRange

    def inverse(self) -> "ChangedSpan":
        return ChangedSpan(
            self.change.inverse(),
            [c.inverse() for c in self.parent_scopes],
            self.line_range,
        )

    @property
    def header_line_range(self) -> LineRange:
        parent_scope = self.parent_scopes[-1].earlier
        hrange = parent_scope.header_line_range
        return hrange

    @property
    def module(self) -> ModuleName:
        return self.parent_scopes[-1].earlier.path.module

    @property
    def scope(self) -> Change[ChangeScope]:
        return self.parent_scopes[-1]

    def _is_func_body(self) -> bool:
        return self.parent_scopes[-1].earlier.tree.type == ptree.Function.type

    def __repr__(self) -> str:
        return f"ChangeSpan(module={self.module}, range={self.line_range}, scope={self.scope.earlier.path.path}, type={self.change.as_char()})"


@dataclass
class JModule:
    "A light wrapper around a jedi module."
    mname: ModuleName
    tree: ptree.Module

    @cached_property
    def as_scope(self) -> ChangeScope:
        return ChangeScope.from_tree(ProjectPath(self.mname, ""), self.tree)

    @cached_property
    def imported_names(self):
        names = set[ptree.Name]()
        for stmt in self.tree.iter_imports():
            if isinstance(stmt, ptree.ImportFrom):
                for n in stmt.get_from_names():
                    assert isinstance(n, ptree.Name)
                    names.add(n)
            elif isinstance(stmt, ptree.ImportName):
                for n in stmt.get_defined_names():
                    assert isinstance(n, ptree.Name)
                    names.add(n)
        return names


@dataclass(frozen=True)
class JModuleChange:
    module_change: Change[JModule]
    changed: Sequence[ChangedSpan]

    def __repr__(self) -> str:
        return f"JModuleChange({self.changed})"

    def inverse(self) -> Self:
        "Create the inverse change."
        return JModuleChange(
            self.module_change.inverse(), [span.inverse() for span in self.changed]
        )

    @staticmethod
    def from_modules(
        module_change: Change[JModule],
        only_ast_changes: bool = True,
        return_unchanged: bool = False,
    ):
        "Compute the change spans from two versions of the same module."
        with _tlogger.timed("JModuleChange.from_modules"):
            changed = get_changed_spans(
                module_change.map(lambda m: m.as_scope),
                tuple(),
                only_ast_changes=only_ast_changes,
                return_unchanged=return_unchanged,
            )
            return JModuleChange(module_change, changed)


def get_python_files(project: Path) -> list[RelPath]:
    files = list[RelPath]()
    for f in recurse_find_python_files(FolderIO(str(project))):
        f: FileIO
        files.append(to_rel_path(Path(f.path).relative_to(project)))
    return files


DefaultIgnoreDirs = {".venv", ".mypy_cache", ".git", "venv", "build"}


@dataclass(frozen=True)
class EditTarget:
    lines: tuple[int, int]


@dataclass(frozen=True)
class JProjectChange:
    project_name: str
    changed: Mapping[ModuleName, JModuleChange]
    all_modules: Modified[Collection[JModule]]
    commit_info: "CommitInfo | None"

    def __repr__(self) -> str:
        commit = (
            f"commit={repr(self.commit_info.summary())}, " if self.commit_info else ""
        )
        return f"JProjectChange({commit}{self.changed})"


@dataclass
class ProjectState:
    project: jedi.Project
    scripts: Mapping[RelPath, jedi.Script]


TProb = TypeVar("TProb", covariant=True)
TEnc = TypeVar("TEnc", covariant=True)


class ProjectChangeProcessor(Generic[TProb], ABC):
    def pre_edit_analysis(
        self,
        pstate: ProjectState,
        modules: Mapping[RelPath, JModule],
        changes: Mapping[ModuleName, JModuleChange],
    ) -> Any:
        return None

    def post_edit_analysis(
        self,
        pstate: ProjectState,
        modules: Mapping[RelPath, JModule],
        changes: Mapping[ModuleName, JModuleChange],
    ) -> Any:
        return None

    @abstractmethod
    def process_change(
        self, pchange: "JProjectChange", pre_analysis: Any, post_analysis: Any
    ) -> Sequence[TProb]:
        ...

    def clear_stats(self):
        return None

    def append_stats(self, stats: dict[str, Any]) -> None:
        return None

    def set_training(self, is_training: bool) -> None:
        return None

    def use_unchanged(self) -> bool:
        return False


class NoProcessing(ProjectChangeProcessor[JProjectChange]):
    def process_change(
        self,
        pchange: JProjectChange,
        pre_analysis,
        post_analysis,
    ) -> Sequence[JProjectChange]:
        return [pchange]


def edits_from_commit_history(
    project_dir: Path,
    history: Sequence[CommitInfo],
    tempdir: Path,
    change_processor: ProjectChangeProcessor[TProb] = NoProcessing(),
    ignore_dirs=DefaultIgnoreDirs,
    silent: bool = False,
    time_limit: float | None = None,
) -> Sequence[TProb]:
    """Incrementally compute the edits to a project from the git history.
    Note that this will change the file states in the project directory, so
    you should make a copy of the project before calling this function.
    """
    tempdir = tempdir.resolve()
    if tempdir.exists():
        raise FileExistsError(f"Workdir '{tempdir}' already exists.")
    use_fast_parser = jedi.settings.fast_parser
    tempdir.mkdir(parents=True, exist_ok=False)
    try:
        run_command(
            ["cp", "-r", str(project_dir / ".git"), str(tempdir)],
            cwd=project_dir.parent,
        )

        return _edits_from_commit_history(
            tempdir,
            history,
            change_processor,
            ignore_dirs,
            silent,
            time_limit=time_limit,
        )
    finally:
        shutil.rmtree(tempdir)
        jedi.settings.fast_parser = use_fast_parser


def _deep_copy_subset_(dict: dict[T1, T2], keys: Collection[T1]) -> dict[T1, T2]:
    "This is more efficient than deepcopying each value individually if they share common data."
    keys = {k for k in keys if k in dict}
    to_copy = {k: dict[k] for k in keys}
    copies = copy.deepcopy(to_copy)
    for k in keys:
        dict[k] = copies[k]
    return dict


_Second = float


def parse_module_script(project: jedi.Project, path: Path):
    assert path.is_absolute(), f"Path is not absolute: {path=}"
    script = jedi.Script(path=path, project=project)
    mcontext = script._get_module_context()
    assert isinstance(mcontext, ModuleContext)
    mname = cast(str, mcontext.py__name__())
    if mname.startswith("src."):
        e = ValueError(f"Bad module name: {mname}")
        files = list(project.path.iterdir())
        print_err(f"project: {project.path}", file=sys.stderr)
        print_err(f"files in root: {files}", file=sys.stderr)
        raise e
    m = script._module_node
    assert isinstance(m, ptree.Module)
    # mname = PythonProject.rel_path_to_module_name(path.relative_to(proj.path))
    # m = parso.parse(path.read_text())
    jmod = JModule(mname, m)
    return jmod, script


def _edits_from_commit_history(
    project: Path,
    history: Sequence[CommitInfo],
    change_processor: ProjectChangeProcessor[TProb],
    ignore_dirs: set[str],
    silent: bool,
    time_limit: _Second | None,
) -> Sequence[TProb]:
    start_time = time.time()
    scripts = dict[RelPath, jedi.Script]()
    results = list[TProb]()

    def has_timeouted(step):
        if time_limit and (time.time() - start_time > time_limit):
            warnings.warn(
                f"_edits_from_commit_history timed out for {project}. ({time_limit=}) "
                f"Partial results ({step}/{len(history)-1}) will be returned."
            )
            return True
        else:
            return False

    def parse_module(path: Path):
        with _tlogger.timed("parse_module"):
            m, s = parse_module_script(proj, path)
            scripts[to_rel_path(path.relative_to(proj._path))] = s
            return m

    def checkout_commit(commit_hash: str):
        with _tlogger.timed("checkout"):
            subprocess.run(
                ["git", "checkout", "-f", commit_hash],
                cwd=project,
                capture_output=True,
                check=True,
            )

    # to ensure sure we are not accidentally overriding real code changes
    if list(project.iterdir()) != [project / ".git"]:
        raise FileExistsError(f"Directory '{project}' should contain only '.git'.")

    # checkout to the first commit
    commit_now = history[-1]
    checkout_commit(commit_now.hash)
    proj = jedi.Project(path=project, added_sys_path=[project / "src"])
    pstate = ProjectState(proj, scripts)

    # now we can get the first project state, although this not needed for now
    # but we'll use it later for pre-edit analysis
    init_srcs = [
        to_rel_path(f.relative_to(project))
        for f in rec_iter_files(project, dir_filter=lambda d: d.name not in ignore_dirs)
        if f.suffix == ".py"
    ]
    path2module = {
        f: parse_module(project / f)
        for f in tqdm(init_srcs, desc="building initial project", disable=silent)
    }

    def is_src(path_s: str) -> bool:
        path = Path(path_s)
        return path.suffix == ".py" and all(p not in ignore_dirs for p in path.parts)

    future_commits = list(reversed(history[:-1]))
    for step, commit_next in tqdm(
        list(enumerate(future_commits)),
        smoothing=0,
        desc="processing commits",
        disable=silent,
    ):
        if has_timeouted(step):
            return results
        # get changed files
        changed_files = run_command(
            [
                "git",
                "diff",
                "--no-renames",
                "--name-status",
                commit_now.hash,
                commit_next.hash,
            ],
            cwd=project,
        ).splitlines()

        path_changes = set[Change[str]]()

        for line in changed_files:
            segs = line.split("\t")
            if len(segs) == 2:
                tag, path = segs
                if not is_src(path):
                    continue
                if tag.endswith("A"):
                    path_changes.add(Added(path))
                elif tag.endswith("D"):
                    path_changes.add(Deleted(path))
                if tag.endswith("M"):
                    path_changes.add(Modified(path, path))
            elif len(segs) == 3:
                tag, path1, path2 = segs
                assert tag.startswith("R")
                if is_src(path1):
                    path_changes.add(Deleted(path1))
                if is_src(path2):
                    path_changes.add(Added(path2))

        # make deep copys of changed modules
        to_copy = {
            to_rel_path(Path(path_change.before))
            for path_change in path_changes
            if not isinstance(path_change, Added)
        }
        _deep_copy_subset_(path2module, to_copy)

        checkout_commit(commit_next.hash)

        new_path2module = path2module.copy()
        changed = dict[ModuleName, JModuleChange]()
        for path_change in path_changes:
            path = project / path_change.earlier
            rel_path = to_rel_path(path.relative_to(project))
            if not isinstance(path_change, Added) and rel_path not in new_path2module:
                warnings.warn(f"No module for file: {project/rel_path}")
                if isinstance(path_change, Deleted):
                    continue
                elif isinstance(path_change, Modified):
                    path_change = Added(path_change.after)
            match path_change:
                case Added():
                    mod = parse_module(path)
                    new_path2module[rel_path] = mod
                    changed[mod.mname] = JModuleChange.from_modules(Added(mod))
                case Deleted():
                    mod = new_path2module.pop(rel_path)
                    changed[mod.mname] = JModuleChange.from_modules(Deleted(mod))
                case Modified(path1, path2):
                    assert path1 == path2
                    mod_old = new_path2module.pop(rel_path)
                    new_path2module[rel_path] = mod_new = parse_module(path)
                    changed[mod_new.mname] = JModuleChange.from_modules(
                        Modified(mod_old, mod_new),
                        return_unchanged=change_processor.use_unchanged(),
                    )
            if has_timeouted(step):
                return results

        with _tlogger.timed("post_edit_analysis"):
            post_analysis = change_processor.post_edit_analysis(
                pstate,
                new_path2module,
                changed,
            )
        if has_timeouted(step):
            return results

        # now go backwards in time to perform pre-edit analysis
        checkout_commit(commit_now.hash)
        with _tlogger.timed("pre_edit_analysis"):
            pre_analysis = change_processor.pre_edit_analysis(
                pstate,
                path2module,
                changed,
            )
        checkout_commit(commit_next.hash)

        modules_mod = Modified(path2module.values(), new_path2module.values())
        pchange = JProjectChange(project.name, changed, modules_mod, commit_next)

        with _tlogger.timed("process_change"):
            processed = change_processor.process_change(
                pchange, pre_analysis, post_analysis
            )
            results.extend(processed)
        commit_now = commit_next
        path2module = new_path2module
    return results


def get_changed_spans(
    scope_change: Change[ChangeScope],
    parent_changes: tuple[Change[ChangeScope], ...] = (),
    only_ast_changes: bool = True,
    return_unchanged: bool = False,
) -> list[ChangedSpan]:
    """
    Extract the change spans from scope change.
        - We need a tree differencing algorithm that are robust to element movements.
        - To compute the changes to each statement region, we can compute the differences
        by concatenating all the regions before and after the edit
        (and hiding all the sub spans such as class methods), then map the changes
        to each line back to the original regions.

    ## Args:
    - `only_ast_changes`: if True, will skip the changes that are just caused by
    comments or formatting changes.
    - `return_unchanged`: if True, unchanged code spans will also be returned as
    ChangedSpan.
    """

    def get_modified_spans(
        old_scope: ChangeScope,
        new_scope: ChangeScope,
        parent_changes: Sequence[Change[ChangeScope]],
    ) -> Iterable[ChangedSpan]:
        if (
            not return_unchanged
            and only_ast_changes
            and code_equal(old_scope.spans_code, new_scope.spans_code)
        ):
            return
        diffs = change_to_line_diffs(
            Modified(old_scope.spans_code, new_scope.spans_code)
        )
        _, delta = line_diffs_to_original_delta(diffs)
        line = 0
        for span in old_scope.spans:
            code = span.code
            line_range = (line, line + count_lines(code))
            subdelta = delta.for_input_range(line_range).shifted(-line)
            if subdelta:
                new_code = subdelta.apply_to_input(code)
                change = Modified(code, new_code)
                yield ChangedSpan(
                    change,
                    parent_changes,
                    span.line_range,
                )
            elif return_unchanged:
                yield ChangedSpan(
                    Modified.from_unchanged(code),
                    parent_changes,
                    span.line_range,
                )
            line = line_range[1]

    def recurse(
        scope_change: Change[ChangeScope], parent_changes
    ) -> Iterable[ChangedSpan]:
        parent_changes = (*parent_changes, scope_change)
        match scope_change:
            case Modified(old_scope, new_scope):
                # compute statement differences
                yield from get_modified_spans(old_scope, new_scope, parent_changes)
                for sub_change in get_named_changes(
                    old_scope.subscopes, new_scope.subscopes
                ).values():
                    yield from recurse(sub_change, parent_changes)
            case Added(scope) | Deleted(scope):
                for span in scope.spans:
                    code_change = scope_change.new_value(span.code)
                    yield ChangedSpan(
                        code_change,
                        parent_changes,
                        span.line_range,
                    )
                for s in scope.subscopes.values():
                    s_change = scope_change.new_value(s)
                    yield from recurse(s_change, parent_changes)

    spans = list(recurse(scope_change, parent_changes))
    spans.sort(key=lambda s: s.line_range[0])
    return spans


def code_to_module(code: str) -> ptree.Module:
    return parso.parse(code)
