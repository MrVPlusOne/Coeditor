from abc import abstractmethod
import copy
from functools import cached_property
from os import PathLike
import shutil
import sys
from coeditor.encoders import BasicTkQueryEdit
from coeditor.encoding import change_to_line_diffs, line_diffs_to_original_delta
from spot.static_analysis import ModuleName, ProjectPath, PythonProject
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

ScopeTree = ptree.Module | ptree.Function | ptree.Class
PyNode = ptree.PythonBaseNode | ptree.PythonNode

_tlogger = TimeLogger()


@dataclass
class ChangeScope:
    """
    A change scope is a python module, non-hidden function, or a non-hidden class, or a python module.
        - functions and classes that are inside a parent function are considered hidden.
    """

    path: ProjectPath
    tree: ScopeTree
    spans: Sequence["StatementSpan"]
    subscopes: Mapping[ProjectPath, Self]
    parent_scope: Optional["ChangeScope"]

    @cached_property
    def spans_code(self) -> str:
        return "\n".join(s.code for s in self.spans)

    @cached_property
    def header_code(self) -> str:
        if self.parent_scope is None:
            return f"# module: {self.path.module}"
        # get the first non-empty line of self.tree
        first_line = "#" if isinstance(self.tree, ptree.Function) else ""
        for i, c in enumerate(self.tree.children):
            snippet = cast(str, c.get_code())
            if i == 0:
                # remove leading newlines
                snippet = snippet.lstrip("\n")
            assert isinstance(snippet, str)
            if count_lines(snippet) == 1:
                first_line += snippet
            else:
                first_line += snippet.splitlines()[0]
                break
        parent_header = self.parent_scope.header_code
        return f"{parent_header}\n{first_line}"

    @staticmethod
    def from_tree(path: ProjectPath, tree: ScopeTree) -> "ChangeScope":
        spans = []
        subscopes = dict()
        scope = ChangeScope(path, tree, spans, subscopes, parent_scope=None)
        if isinstance(tree, ptree.Function):
            span = StatementSpan([_to_decorated(tree)])
            spans.append(span)
        else:
            assert isinstance(tree, (ptree.Module, ptree.Class))
            current_stmts = []
            content = (
                tree.children
                if isinstance(tree, ptree.Module)
                else cast(ptree.PythonNode, tree.get_suite()).children
            )
            for s in content:
                if _is_scope_statement(as_any(s)):
                    current_stmts.append(s)
                else:
                    if current_stmts:
                        spans.append(StatementSpan(current_stmts))
                        current_stmts = []
            if current_stmts:
                spans.append(StatementSpan(current_stmts))

            for stree in tree._search_in_scope(ptree.Function.type, ptree.Class.type):
                stree: ptree.Function | ptree.Class
                spath = path.append(cast(ptree.Name, stree.name).value)
                subscope = ChangeScope.from_tree(spath, stree)
                subscope.parent_scope = scope
                subscopes[spath] = subscope
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

    statements: Sequence[PyNode]

    def __post_init__(self):
        assert self.statements
        code = "".join(s.get_code() for s in self.statements)
        n_lines_0 = count_lines(code)
        code = code.lstrip("\n")
        prefix_empty_lines = n_lines_0 - count_lines(code)
        start = self.statements[0].start_pos[0]
        start += prefix_empty_lines
        end = self.statements[-1].end_pos[0]

        self._prefix_empty_lines: int = prefix_empty_lines
        self.code: str = code
        self.line_range: tuple[int, int] = (start, end)


@dataclass
class ChangedSpan:
    "Represents the changes made to a statement span."
    change: Change[str]
    scope_change: Change[ChangeScope]
    line_range: tuple[int, int]
    old_statements: Sequence[PyNode]

    def __repr__(self):
        return f"ChangeSpan(scope={self.scope_change.earlier().path}, range={self.line_range}, type={self.change.as_char()})"


@dataclass
class JModule:
    "A light wrapper around a jedi module."
    mname: ModuleName
    tree: ptree.Module

    def _to_scope(self) -> ChangeScope:
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
            for cspan in get_changed_spans(module_change.map(lambda m: m._to_scope())):
                path = cspan.scope_change.earlier().path
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
        file_changes: Sequence[Change[str]],
    ) -> Any:
        return None

    def post_edit_analysis(
        self,
        project: jedi.Project,
        modules: Mapping[RelPath, JModule],
        file_changes: Sequence[Change[str]],
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

    # turn this off so we don't have to deep copy the Modules
    # jedi.settings.fast_parser = False
    # Update: Have to use deep copy for now due to a bug in jedi: https://github.com/davidhalter/jedi/issues/1888

    # checkout to the first commit
    commit_now = history[-1]
    with _tlogger.timed("checkout"):
        subprocess.run(
            ["git", "checkout", "-f", commit_now.hash],
            cwd=project,
            capture_output=True,
            check=True,
        )
    proj = jedi.Project(path=project, added_sys_path=[project / "src"])

    # now we can get the first project state, although this not needed for now
    # but we'll use it later for pre-edit analysis
    path2module = {
        RelPath(f): parse_module(project / f) for f in get_python_files(project)
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

        with _tlogger.timed("pre_edit_analysis"):
            pre_analysis = change_encoder.pre_edit_analysis(
                proj,
                path2module,
                path_changes,
            )

        # check out commit_next
        with _tlogger.timed("checkout"):
            subprocess.run(
                ["git", "checkout", commit_next.hash],
                cwd=project,
                capture_output=True,
                check=True,
            )
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
                path_changes,
            )

        pchange = JProjectChange(changed, commit_next)

        with _tlogger.timed("encode_change"):
            encs = change_encoder.encode_change(pchange, pre_analysis, post_analysis)
            yield from encs
        commit_now = commit_next
        path2module = new_path2module


def get_changed_spans(
    scope_change: Change[ChangeScope],
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
        old_scope: ChangeScope, new_scope: ChangeScope
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
                scope_change = Modified(old_scope, new_scope)
                yield ChangedSpan(
                    change, scope_change, span.line_range, span.statements
                )
            line = line_range[1]

    def recurse(scope_change) -> Iterable[ChangedSpan]:
        match scope_change:
            case Modified(old_scope, new_scope):
                # compute statement differences
                yield from get_modified_spans(old_scope, new_scope)
                for sub_change in get_named_changes(
                    old_scope.subscopes, new_scope.subscopes
                ).values():
                    yield from recurse(sub_change)
            case Added(scope) | Deleted(scope):
                for span in scope.spans:
                    code_change = scope_change.new_value(span.code)
                    yield ChangedSpan(
                        code_change, scope_change, span.line_range, span.statements
                    )
                for s in scope.subscopes.values():
                    s_change = scope_change.new_value(s)
                    yield from recurse(s_change)

    spans = list(recurse(scope_change))
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
