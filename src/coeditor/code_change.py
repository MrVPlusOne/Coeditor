from abc import abstractmethod
import copy
from functools import cached_property
from os import PathLike
import shutil
import sys
from coeditor.encoders import BasicTkQueryEdit
from coeditor.encoding import change_to_line_diffs, line_diffs_to_original_delta
from spot.static_analysis import ElemPath, ModuleName, ProjectPath, PythonProject
from .common import *
from .history import (
    Change,
    Added,
    Modified,
    Deleted,
    get_named_changes,
    CommitInfo,
    get_commit_history,
)
import jedi
import jedi.settings
from parso.python import tree as ptree
from parso.tree import NodeOrLeaf
import parso
from jedi.inference.references import recurse_find_python_files
from jedi.file_io import FileIO, FolderIO
from jedi.inference.context import ModuleContext

ScopeTree = ptree.Function | ptree.Class | ptree.Module
PyNode = ptree.PythonBaseNode | ptree.PythonNode
LineRange = NewType("LineRange", tuple[int, int])

_tlogger = TimeLogger()


def line_range(start: int, end: int, can_be_empty: bool = False) -> LineRange:
    if not can_be_empty and start >= end:
        raise ValueError(f"Bad line range: {start=}, {end=}")
    return LineRange((start, end))


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

    @cached_property
    def spans_code(self) -> str:
        return "\n".join(s.code for s in self.spans)

    @cached_property
    def all_code(self) -> str:
        return self.header_code + self.spans_code

    def _search(self, path: ElemPath, line: int) -> Self | "StatementSpan":
        scope = self._search_scope(path)
        if scope.header_line_range[0] <= line < scope.header_line_range[1]:
            return scope
        span = scope._search_span(line)
        return span or scope

    def _search_scope(self, path: ElemPath) -> Self:
        """Find the scope that can potentially contain the given path. Follow the
        path segments until no more subscopes are found."""
        segs = path.split(".")
        scope = self
        for s in segs:
            if s in scope.subscopes:
                scope = scope.subscopes[s]
            else:
                break
        return scope

    def _search_span(self, line: int) -> "StatementSpan | None":
        for span in self.spans:
            if span.line_range[0] <= line < span.line_range[1]:
                return span
        return None

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
            header_code, _, e_right = _strip_empty_lines(header_code)
            h_start = visited[0].start_pos[0]
            h_end = visited[-1].end_pos[0] + 1 - e_right
            assert_eq(count_lines(header_code) == h_end - h_start)
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

    @staticmethod
    def from_tree(path: ProjectPath, tree: ScopeTree) -> "ChangeScope":
        spans = []
        subscopes = dict()
        scope = ChangeScope(path, tree, spans, subscopes, None)
        assert isinstance(tree, ScopeTree)
        is_func = isinstance(tree, ptree.Function)

        current_stmts = []
        content = (
            tree.children
            if isinstance(tree, ptree.Module)
            else cast(ptree.PythonNode, tree.get_suite()).children
        )
        for s in content:
            # we don't create inner scopes for function contents
            if is_func or _is_scope_statement(as_any(s)):
                current_stmts.append(s)
            else:
                if current_stmts:
                    spans.append(StatementSpan(len(spans), current_stmts, scope))
                    current_stmts = []
        if current_stmts:
            spans.append(StatementSpan(len(spans), current_stmts, scope))

        if is_func:
            # we don't create inner scopes for function contents
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
        return f"ChangeScope(path={self.path}, type={self.tree.type})"


_non_scope_stmt_types = {"decorated", "async_stmt"}


def _is_scope_statement(stmt: PyNode) -> bool:
    match stmt:
        case ptree.PythonNode(type=node_type) if node_type not in _non_scope_stmt_types:
            return stmt.children[0].type not in ptree._IMPORTS
        case ptree.Flow():
            return True
        case _:
            return False


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
        # remove leading newlines
        n_leading_newlines = 0
        stmts = self.statements
        for s in stmts:
            if s.type == ptree.Newline.type:
                n_leading_newlines += 1
            else:
                break
        if n_leading_newlines:
            self.statements = stmts[n_leading_newlines:]

        origin_code = "".join(s.get_code() for s in self.statements)
        code, _, e_right = _strip_empty_lines(origin_code)
        start = self.statements[0].start_pos[0]
        end = self.statements[-1].end_pos[0] + 1 - e_right

        self.code: str = code + "\n"
        try:
            self.line_range: LineRange = line_range(start, end)
        except ValueError:
            print_err(f"{origin_code=}, {e_right=}, {start=}, {end=}")
            raise


@dataclass
class ChangedSpan:
    "Represents the changes made to a statement span."
    change: Change[str]
    parent_scopes: Sequence[Change[ChangeScope]]
    line_range: LineRange

    @property
    def header_line_range(self) -> LineRange:
        parent_scope = self.parent_scopes[-1].earlier()
        hrange = parent_scope.header_line_range
        return hrange

    @property
    def path(self) -> ProjectPath:
        return self.parent_scopes[-1].earlier().path

    def __repr__(self) -> str:
        return f"ChangeSpan(scope={self.path}, range={self.line_range}, type={self.change.as_char()})"


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


@dataclass
class JModuleChange:
    module_change: Change[JModule]
    changed: Mapping[ProjectPath, ChangedSpan]

    def __repr__(self) -> str:
        change_dict = {k.path: v.change.as_char() for k, v in self.changed.items()}
        return f"JModuleChange({change_dict})"

    @staticmethod
    def from_modules(module_change: Change[JModule]):
        "Compute the change spans from two versions of the same module."
        with _tlogger.timed("JModuleChange.from_modules"):
            changed = dict[ProjectPath, ChangedSpan]()
            for cspan in get_changed_spans(
                module_change.map(lambda m: m.as_scope), tuple()
            ):
                path = cspan.parent_scopes[-1].earlier().path
                changed[path] = cspan
            return JModuleChange(module_change, changed)


def get_python_files(project: Path):
    files = list[Path]()
    for f in recurse_find_python_files(FolderIO(str(project))):
        f: FileIO
        files.append(Path(f.path).relative_to(project))
    return files


DefaultIgnoreDirs = {".venv", ".mypy_cache", ".git", "venv", "build"}


@dataclass
class EditTarget:
    lines: tuple[int, int]


@dataclass
class JProjectChange:
    changed: Mapping[ModuleName, JModuleChange]
    all_modules: Modified[Collection[JModule]]
    commit_info: "CommitInfo | None"

    def __repr__(self) -> str:
        commit = (
            f"commit={repr(self.commit_info.summary())}, " if self.commit_info else ""
        )
        return f"JProjectChange({commit}{self.changed})"


TEnc = TypeVar("TEnc", covariant=True)


class ProjectChangeProcessor(Generic[TEnc]):
    def pre_edit_analysis(
        self,
        project: jedi.Project,
        modules: Mapping[RelPath, JModule],
        changes: Mapping[ModuleName, JModuleChange],
    ) -> Any:
        return None

    def post_edit_analysis(
        self,
        project: jedi.Project,
        modules: Mapping[RelPath, JModule],
        changes: Mapping[ModuleName, JModuleChange],
    ) -> Any:
        return None

    def encode_change(
        self, pchange: "JProjectChange", pre_analysis: Any, post_analysis: Any
    ) -> Iterable[TEnc]:
        ...


class NoProcessing(ProjectChangeProcessor[JProjectChange]):
    def encode_change(
        self,
        pchange: JProjectChange,
        pre_analysis,
        post_analysis,
    ) -> Iterable[JProjectChange]:
        yield pchange


def edits_from_commit_history(
    project_dir: Path,
    history: Sequence[CommitInfo],
    tempdir: Path,
    change_encoder: ProjectChangeProcessor[TEnc] = NoProcessing(),
    ignore_dirs=DefaultIgnoreDirs,
    silent: bool = False,
) -> Iterable[TEnc]:
    """Incrementally compute the edits to a project from the git history.
    Note that this will change the file states in the project directory, so
    you should make a copy of the project before calling this function.

    Note that this returns an iterator, and the file state cleaning up will
    only happen when the iterator is exhausted.
    """
    tempdir = tempdir.resolve()
    if tempdir.exists():
        raise FileExistsError(f"Workdir '{tempdir}' already exists.")
    tempdir.mkdir(parents=True, exist_ok=False)
    use_fast_parser = jedi.settings.fast_parser
    try:
        run_command(
            ["cp", "-r", str(project_dir / ".git"), str(tempdir)],
            cwd=project_dir.parent,
        )

        yield from _edits_from_commit_history(
            tempdir, history, change_encoder, ignore_dirs, silent
        )
    finally:
        run_command(["rm", "-rf", str(tempdir)], cwd=tempdir.parent)
        jedi.settings.fast_parser = use_fast_parser


def _edits_from_commit_history(
    project: Path,
    history: Sequence[CommitInfo],
    change_encoder: ProjectChangeProcessor[TEnc],
    ignore_dirs: set[str],
    silent: bool,
) -> Iterable[TEnc]:
    def parse_module(path: Path):
        with _tlogger.timed("parse_module"):
            s = jedi.Script(path=path, project=proj)
            mcontext = s._get_module_context()
            assert isinstance(mcontext, ModuleContext)
            mname = cast(str, mcontext.py__name__())
            if mname.startswith("src."):
                e = ValueError(f"Bad module name: {mname}")
                files = list(project.iterdir())
                print_err(f"project: {proj}", file=sys.stderr)
                print_err(f"files in root: {files}", file=sys.stderr)
                raise e
            m = copy.deepcopy(s._module_node)  # needed due to reusing
            assert isinstance(m, ptree.Module)
            # mname = PythonProject.rel_path_to_module_name(path.relative_to(proj.path))
            # m = parso.parse(path.read_text())
            return JModule(mname, m)

    def checkout_commit(commit_hash: str, force: bool = False):
        with _tlogger.timed("checkout"):
            subprocess.run(
                ["git", "checkout", "-f", commit_hash],
                cwd=project,
                capture_output=True,
                check=True,
            )

    # checkout to the first commit
    commit_now = history[-1]
    checkout_commit(commit_now.hash, force=True)
    proj = jedi.Project(path=project, added_sys_path=[project / "src"])

    # now we can get the first project state, although this not needed for now
    # but we'll use it later for pre-edit analysis
    path2module = {
        RelPath(f): parse_module(project / f)
        for f in tqdm(
            get_python_files(project), desc="building initial project", disable=silent
        )
    }

    def is_src(path_s: str) -> bool:
        path = Path(path_s)
        return path.suffix == ".py" and all(p not in ignore_dirs for p in path.parts)

    future_commits = list(reversed(history[:-1]))
    for commit_next in tqdm(
        future_commits, smoothing=0, desc="processing commits", disable=silent
    ):
        # get changed files
        changed_files = run_command(
            ["git", "diff", commit_now.hash, commit_next.hash, "--name-status"],
            cwd=project,
        ).splitlines()

        path_changes = list[Change[str]]()

        for line in changed_files:
            segs = line.split("\t")
            if len(segs) == 2:
                tag, path = segs
                if not is_src(path):
                    continue
                if tag.endswith("A"):
                    path_changes.append(Added(path))
                elif tag.endswith("D"):
                    path_changes.append(Deleted(path))
                if tag.endswith("M"):
                    path_changes.append(Modified(path, path))
            elif len(segs) == 3:
                tag, path1, path2 = segs
                assert tag.startswith("R")
                if not is_src(path1) or not is_src(path2):
                    continue
                path_changes.append(Deleted(path1))
                path_changes.append(Added(path2))

        checkout_commit(commit_next.hash)

        proj = jedi.Project(path=project, added_sys_path=[project / "src"])

        new_path2module = path2module.copy()
        changed = dict[ModuleName, JModuleChange]()
        for path_change in path_changes:
            path = project / path_change.earlier()
            rel_path = RelPath(path.relative_to(project))
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
                    mod_old = new_path2module[rel_path]
                    new_path2module[rel_path] = mod_new = parse_module(path)
                    changed[mod_new.mname] = JModuleChange.from_modules(
                        Modified(mod_old, mod_new)
                    )

        with _tlogger.timed("post_edit_analysis"):
            post_analysis = change_encoder.post_edit_analysis(
                proj,
                new_path2module,
                changed,
            )

        # now go backwards in time to perform pre-edit analysis
        checkout_commit(commit_now.hash)
        with _tlogger.timed("pre_edit_analysis"):
            pre_analysis = change_encoder.pre_edit_analysis(
                proj,
                path2module,
                changed,
            )
        checkout_commit(commit_next.hash)

        modules_mod = Modified(path2module.values(), new_path2module.values())
        pchange = JProjectChange(changed, modules_mod, commit_next)

        with _tlogger.timed("encode_change"):
            encs = change_encoder.encode_change(pchange, pre_analysis, post_analysis)
            yield from encs
        commit_now = commit_next
        path2module = new_path2module


def get_changed_spans(
    scope_change: Change[ChangeScope],
    parent_changes: tuple[Change[ChangeScope], ...] = (),
) -> list[ChangedSpan]:
    """
    Extract the change spans from scope change.
        - We need a tree differencing algorithm that are robust to element movements.
        - To compute the changes to each statement region, we can compute the differences
        by concatenating all the regions before and after the edit
        (and hiding all the sub spans such as class methods), then map the changes
        to each line back to the original regions.
    """

    def get_modified_spans(
        old_scope: ChangeScope,
        new_scope: ChangeScope,
        parent_changes: Sequence[Change[ChangeScope]],
    ) -> Iterable[ChangedSpan]:
        if code_equal(old_scope.spans_code, new_scope.spans_code):
            return
        diffs = change_to_line_diffs(
            Modified(old_scope.spans_code, new_scope.spans_code)
        )
        original, delta = line_diffs_to_original_delta(diffs)
        line = 0
        for span in old_scope.spans:
            code = span.code
            line_range = (line, line + len(code.split("\n")))
            if subdelta := delta.for_input_range(line_range):
                new_code = subdelta.apply_to_input(code)
                change = Modified(code, new_code)
                yield ChangedSpan(
                    change,
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
    m = jedi.Script(code)._module_node
    assert isinstance(m, ptree.Module)
    return m


def _search_in_scope(
    self: ptree.Scope, filter: Callable[[ptree.PythonBaseNode], bool]
) -> Iterable[ptree.PythonBaseNode]:
    def scan(children: Sequence[ptree.PythonBaseNode]):
        for element in children:
            if filter(element):
                yield element
            if element.type in ptree._FUNC_CONTAINERS:
                yield from scan(element.children)  # type: ignore

    return scan(self.children)  # type: ignore


def _to_decorated(tree: ptree.ClassOrFunc):
    decorated = not_none(tree.parent)
    if decorated.type == "async_funcdef":
        decorated = not_none(decorated.parent)

    if decorated.type == "decorated":
        return cast(ptree.PythonNode, decorated)
    else:
        return tree
