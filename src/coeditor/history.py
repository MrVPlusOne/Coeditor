# utils for computing editing history from git commits

from spot.static_analysis import (
    ElemPath,
    ModuleName,
    PythonElem,
    PythonModule,
    PythonProject,
    remove_comments,
)
from .common import *


@dataclass
class Added(Generic[T1]):
    after: T1


@dataclass
class Deleted(Generic[T1]):
    before: T1


@dataclass
class Modified(Generic[T1]):
    before: T1
    after: T1


Change = Added[T1] | Deleted[T1] | Modified[T1]


def empty_module(mname: ModuleName) -> PythonModule:
    return PythonModule.from_cst(cst.Module([]), mname)


def parse_cst_module(code: str) -> cst.Module:
    return remove_comments(cst.parse_module(code))


@dataclass
class ModuleEdit:
    before: PythonModule
    after: PythonModule
    added: dict[ElemPath, Added[PythonElem]]
    deleted: dict[ElemPath, Deleted[PythonElem]]
    modified: dict[ElemPath, Modified[PythonElem]]
    all_changes: dict[ElemPath, Change[PythonElem]]

    @property
    def is_empty(self) -> bool:
        return len(self.all_changes) == 0

    @staticmethod
    def from_modules(before: PythonModule, after: PythonModule):
        before_elems = {e.path.path: e for e in before.all_elements()}
        after_elems = {e.path.path: e for e in after.all_elements()}
        added = {
            path: Added(elem)
            for path, elem in after_elems.items()
            if path not in before_elems
        }
        deleted = {
            path: Deleted(elem)
            for path, elem in before_elems.items()
            if path not in after_elems
        }
        modified = {
            path: Modified(before_elems[path], after_elems[path])
            for path in before_elems.keys() & after_elems.keys()
            if before_elems[path].code != after_elems[path].code
        }
        changes = added | deleted | modified

        return ModuleEdit(before, after, added, deleted, modified, changes)


@dataclass
class ProjectEdit:
    """An edit to a project."""

    before: PythonProject
    after: PythonProject
    changes: dict[ModuleName, ModuleEdit]

    @staticmethod
    def from_code_changes(
        before: PythonProject,
        code_changes: Mapping[ModuleName, str | None],
        symlinks: Mapping[ModuleName, ModuleName] = {},
        src2module: Callable[[str], cst.Module | None] = parse_cst_module,
    ) -> "ProjectEdit":
        modules = before.modules
        changes = dict[ModuleName, ModuleEdit]()
        for mname, new_code in code_changes.items():
            if new_code is None:
                # got deleted
                mod_after = empty_module(mname)
            else:
                try:
                    if (m := src2module(new_code)) is None:
                        continue
                    mod_after = PythonModule.from_cst(m, mname)
                except cst.ParserSyntaxError:
                    continue

            mod_before = modules[mname] if mname in modules else empty_module(mname)
            mod_edit = ModuleEdit.from_modules(mod_before, mod_after)
            if new_code is None:
                modules.pop(mname, None)
            else:
                modules[mname] = mod_after
            if not mod_edit.is_empty:
                changes[mname] = mod_edit

        return ProjectEdit(
            before,
            # module2src_file is kept emtpy for now
            PythonProject(before.root_dir, modules, dict(symlinks), dict()),
            changes,
        )


def edits_from_history(
    project_dir: Path,
) -> list[ProjectEdit]:
    """Compute the edits to a project from git history.

    Args:
        project_dir: The directory of the project.

    Returns:
        A list of edits to the project, from past to present.
    """

    pass
