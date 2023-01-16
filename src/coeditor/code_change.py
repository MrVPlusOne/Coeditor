import copy
from functools import cached_property
from os import PathLike
import shutil
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

ScopeTree = ptree.Module | ptree.Function | ptree.Class

_tlogger = TimeLogger()


@dataclass
class ChangeScope:
    """
    A change scope is a python module, non-hidden function, or a non-hidden class, or a python module.
        - functions and classes that are inside a parent function are considered hidden.
    """

    path: ProjectPath
    tree: ScopeTree
    statements: Sequence[ptree.PythonBaseNode | ptree.PythonNode]
    subscopes: Mapping[ProjectPath, Self]

    @cached_property
    def statements_code(self) -> str:
        return "".join(s.get_code() for s in self.statements).strip("\n")

    @staticmethod
    def from_tree(path: ProjectPath, tree: ScopeTree) -> "ChangeScope":
        if isinstance(tree, ptree.Function):
            stmts = [_to_decorated(tree)]
            subscopes = dict()
        else:
            stmts = [
                s
                for s in tree._search_in_scope("simple_stmt")
                if s.children[0].type not in ptree._IMPORTS
            ]
            subscopes = dict()
            for stree in tree._search_in_scope(ptree.Function.type, ptree.Class.type):
                stree: ptree.Function | ptree.Class
                spath = path.append(cast(ptree.Name, stree.name).value)
                subscopes[spath] = ChangeScope.from_tree(spath, stree)
        return ChangeScope(path, tree, stmts, subscopes)

    def __repr__(self):
        return f"ChangeScope(path={self.path}, type={self.tree.type})"


@dataclass
class ChangeTarget:
    """
    A change target is a set of lines inside the same change scope. It is the basic unit of code changes handled by our model.
        - For a modified function, the target is the function itself.
        - For a modified module, the targets are the regions between the functions and classes plus
        the targets recursively generated.
        - For a modified class, the targets are the regions between methods plus
        the targets recursively generated.
    """

    scope: ChangeScope
    statements: Sequence[ptree.PythonBaseNode | ptree.PythonNode]

    @cached_property
    def code(self) -> str:
        return "".join(s.get_code() for s in self.statements)

    def __repr__(self):
        code_range = (self.statements[0].start_pos, self.statements[-1].end_pos)
        code = self.code
        if len(code) > 30:
            code = code[:30] + "..."
        return f"ChangeTarget(scope={self.scope.path}, range={code_range}, code={repr(code)})"


@dataclass
class JModule:
    "A light wrapper around a jedi module."
    mname: ModuleName
    tree: ptree.Module

    def _to_scope(self) -> ChangeScope:
        return ChangeScope.from_tree(ProjectPath(self.mname, ""), self.tree)


@dataclass
class JModuleChange:
    module_change: Change[JModule]
    changed: Mapping[ProjectPath, Change[ChangeTarget]]

    def __repr__(self) -> str:
        change_dict = {k.path: v.as_char() for k, v in self.changed.items()}
        return f"JModuleChange({change_dict})"

    @staticmethod
    def from_modules(module_change: Change[JModule]):
        "Compute the change targets from two versions of the same module."
        with _tlogger.timed("JModuleChange.from_modules"):
            changed = dict[ProjectPath, Change[ChangeTarget]]()
            for c in _get_change_targets(module_change.map(lambda m: m._to_scope())):
                path = c.get_any().scope.path
                changed[path] = c
            return JModuleChange(module_change, changed)


def get_python_src_map(project: Path) -> dict[ModuleName, Path]:
    src_map = dict[ModuleName, Path]()
    for f in recurse_find_python_files(FolderIO(str(project))):
        f: FileIO
        rel_path = Path(f.path).relative_to(project)
        mname = PythonProject.rel_path_to_module_name(rel_path)
        src_map[mname] = Path(f.path)
    return src_map


@dataclass
class JProjectChange:
    changed: Mapping[ModuleName, JModuleChange]
    commit_info: "CommitInfo | None"

    def __repr__(self) -> str:
        commit = (
            f"commit={repr(self.commit_info.summary())}, " if self.commit_info else ""
        )
        return f"JProjectChange({commit}{self.changed})"

    @staticmethod
    def edits_from_commit_history(
        project_dir: Path,
        history: Sequence[CommitInfo],
        tempdir: Path,
        ignore_dirs=PythonProject.DefaultIgnoreDirs,
        silent: bool = False,
    ) -> "list[JProjectChange]":
        """Incrementally compute the edits to a project from the git history.

        Returns:
            A list of edits to the project, from past to present.
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

            return _edits_from_commit_history(tempdir, history, ignore_dirs, silent)
        finally:
            run_command(["rm", "-rf", str(tempdir)], cwd=tempdir.parent)
            jedi.settings.fast_parser = use_fast_parser


def _edits_from_commit_history(
    project: Path,
    history: Sequence[CommitInfo],
    ignore_dirs=PythonProject.DefaultIgnoreDirs,
    silent: bool = False,
) -> "list[JProjectChange]":
    """Incrementally compute the edits to a project from the git history.
    Note that this will change the file states in the project directory, so
    you should make a copy of the project before calling this function.

    Returns:
        A list of edits to the project, from past to present.
    """

    def get_module_path(file_s: PathLike | str) -> ModuleName:
        path = Path(file_s)
        if path.is_absolute():
            path = path.relative_to(project)
        mname = PythonProject.rel_path_to_module_name(path)
        return mname

    def parse_module(path: Path):
        with _tlogger.timed("parse_module"):
            mname = get_module_path(path)
            proj.get_environment().version_info
            m = jedi.Script(path=path, project=proj)._module_node
            assert isinstance(m, ptree.Module)
            return JModule(mname, m)

    # turn this off so we don't have to deep copy the Modules
    jedi.settings.fast_parser = False

    # checkout to the first commit
    commit_now = history[-1]
    with _tlogger.timed("checkout"):
        subprocess.run(
            ["git", "checkout", "-f", commit_now.hash],
            cwd=project,
            capture_output=True,
            check=True,
        )
    proj = jedi.Project(path=project)

    # now we can get the first project state, although this not needed for now
    # but we'll use it later for pre-edit analysis
    src_map = get_python_src_map(project)
    modules = {m: parse_module(f) for m, f in src_map.items()}

    def is_src(path_s: str) -> bool:
        path = Path(path_s)
        return path.suffix == ".py" and all(p not in ignore_dirs for p in path.parts)

    future_commits = list(reversed(history[:-1]))
    results = list[JProjectChange]()
    for commit_next in tqdm(
        future_commits, smoothing=0, desc="processing commits", disable=silent
    ):
        # get changed files
        changed_files = run_command(
            ["git", "diff", commit_now.hash, commit_next.hash, "--name-status"],
            cwd=project,
        ).splitlines()
        # check out commit_next
        with _tlogger.timed("checkout"):
            subprocess.run(
                ["git", "checkout", commit_next.hash],
                cwd=project,
                capture_output=True,
                check=True,
            )
        proj = jedi.Project(path=project)

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

        changed = dict[ModuleName, JModuleChange]()
        for path_change in path_changes:
            path = path_change.get_any()
            mname = get_module_path(path)
            match path_change:
                case Added():
                    mod = parse_module(project / path)
                    modules[mod.mname] = mod
                    changed[mod.mname] = JModuleChange.from_modules(Added(mod))
                case Deleted():
                    mod = modules.pop(mname)
                    changed[mname] = JModuleChange.from_modules(Deleted(mod))
                case Modified(path1, path2):
                    assert path1 == path2
                    mod_old = modules[mname]
                    modules[mname] = mod_new = parse_module(project / path1)
                    changed[mname] = JModuleChange.from_modules(
                        Modified(mod_old, mod_new)
                    )

        pchange = JProjectChange(changed, commit_info=commit_next)
        commit_now = commit_next
        results.append(pchange)

    return results


def _get_change_targets(
    scope_change: Change[ChangeScope],
) -> Iterable[Change[ChangeTarget]]:
    """
    Extract the change targets from scope change.
        - We need a tree differencing algorithm that are robust to element movements.
        - To compute the changes to each statement region, we can compute the differences
        by concatenating all the regions before and after the edit
        (and hiding all the sub targets such as class methods), then map the changes
        to each line back to the original regions.
    """
    match scope_change:
        case Modified(old_scope, new_scope):
            # compute statement differences
            assert type(old_scope.statements) == type(new_scope.statements)
            if not code_equal(old_scope.statements_code, new_scope.statements_code):
                old_target = ChangeTarget(old_scope, old_scope.statements)
                new_target = ChangeTarget(new_scope, new_scope.statements)
                yield Modified(old_target, new_target)
            for sub_change in get_named_changes(
                old_scope.subscopes, new_scope.subscopes
            ).values():
                yield from _get_change_targets(sub_change)
        case Added(scope) | Deleted(scope):
            if scope.statements:
                target = ChangeTarget(scope, scope.statements)
                yield scope_change.new_value(target)
            for s in scope.subscopes.values():
                s_change = scope_change.new_value(s)
                yield from _get_change_targets(s_change)


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
