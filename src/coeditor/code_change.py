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
from parso.python import tree as ptree
from parso.tree import NodeOrLeaf
import parso
from jedi.inference.references import recurse_find_python_files
from jedi.file_io import FileIO, FolderIO

ScopeTree = ptree.Module | ptree.Function | ptree.Class


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

    def __post_init__(self):
        if not self.statements:
            raise ValueError("Change target must have at least one statement")

    def code(self) -> str:
        return "".join(s.get_code() for s in self.statements)

    def __repr__(self):
        code_range = (self.statements[0].start_pos, self.statements[-1].end_pos)
        code = self.code()
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

    @staticmethod
    def from_modules(module_change: Change[JModule]):
        "Compute the change targets from two versions of the same module."
        changed = dict[ProjectPath, Change[ChangeTarget]]()
        for c in _get_change_targets(module_change.map(lambda m: m._to_scope())):
            path = c.get_any().scope.path
            changed[path] = c
        return JModuleChange(module_change, changed)


def get_python_src_map(project: Path) -> dict[ModuleName, Path]:
    src_map = dict[ModuleName, Path]()
    for f in recurse_find_python_files(FolderIO(str(project))):
        f: FileIO
        mname = PythonProject.rel_path_to_module_name(Path(f.path))
        src_map[mname] = Path(f.path)
    return src_map


@dataclass
class JProjectChange:
    changed: Mapping[ModuleName, JModuleChange]
    commit_info: "CommitInfo | None"

    @staticmethod
    def edits_from_commit_history(
        project_dir: Path,
        history: Sequence[CommitInfo],
        workdir: Path,
        ignore_dirs=PythonProject.DefaultIgnoreDirs,
    ) -> "Iterable[JProjectChange]":
        """Incrementally compute the edits to a project from the git history.

        Returns:
            A list of edits to the project, from past to present.
        """

        src_map = dict[ModuleName, Path]()

        def get_module_path(file_s: str) -> ModuleName:
            path = Path(file_s)
            mname = PythonProject.rel_path_to_module_name(Path(path))
            src_map[mname] = workdir / path
            return mname

        # first copy into the workdir
        shutil.copytree(project_dir / ".git", workdir, dirs_exist_ok=False)
        project_dir = workdir

        # then checkout to the first commit
        commit_now = history[-1]
        run_command(["git", "checkout", "-f", commit_now.hash], cwd=workdir)

        # now we can get the first project state, although this not needed for now
        # but we'll use it later for pre-edit analysis
        src_map.update(get_python_src_map(workdir))
        scripts = {m: jedi.Script(path=f) for m, f in src_map.items()}

        def is_src(path_s: str) -> bool:
            path = Path(path_s)
            return path.suffix == ".py" and all(
                p not in ignore_dirs for p in path.parts
            )

        for commit_next in reversed(history):
            # get changed files
            changed_files = run_command(
                ["git", "diff", commit_now.hash, commit_next.hash, "--name-status"],
                cwd=workdir,
            ).splitlines()

            path_changes = list[Change[str]]()

            for line in changed_files:
                if not line:
                    continue
                if line[2] == " ":
                    tag = line[:2]
                    path = line[3:]
                    if not is_src(path):
                        continue
                    if tag.endswith("A"):
                        path_changes.append(Added(path))
                    elif tag.endswith("D"):
                        path_changes.append(Deleted(path))
                    if tag.endswith("M"):
                        path_changes.append(Modified(path, path))
                else:
                    tag, path1, path2 = line.split(" ")
                    assert tag.startswith("R")
                    if not is_src(path1) or not is_src(path2):
                        continue
                    path_changes.append(Deleted(path1))
                    path_changes.append(Added(path2))

            changed = dict[ModuleName, JModuleChange]()
            for path_change in changed_files:
                match path_change:
                    case Added(path):
                        mname = get_module_path(path)
                        scripts[mname] = script = jedi.Script(path=workdir / path)
                        mod = JModule(mname, script._module_node)
                        changed[mname] = JModuleChange.from_modules(Added(mod))
                    case Deleted(path):
                        mname = get_module_path(path)
                        script = scripts.pop(mname)
                        mod = JModule(mname, script._module_node)
                        changed[mname] = JModuleChange.from_modules(Deleted(mod))
                    case Modified(path1, path2):
                        assert path1 == path2
                        mname = get_module_path(path1)
                        mod_old = scripts.pop(mname)._module_node
                        scripts[mname] = script = jedi.Script(path=workdir / path1)
                        mod_new = script._module_node
                        changed[mname] = JModuleChange.from_modules(
                            Modified(mod_old, mod_new)
                        )

            pchange = JProjectChange(changed, commit_info=commit_next)
            commit_now = commit_next
            yield pchange


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
            if old_scope.statements != new_scope.statements:
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
