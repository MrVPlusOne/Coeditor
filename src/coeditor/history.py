# utils for computing editing history from git commits

import subprocess
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

    def __repr__(self):
        return f"ModuleEdit(added={list(self.added)}, deleted={list(self.deleted)}, modified={list(self.modified)})"

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
    description: str

    def __repr__(self):
        return f"ProjectEdit(msg={repr(self.description)}, changed={list(self.changes.keys())})"

    @staticmethod
    def from_code_changes(
        before: PythonProject,
        code_changes: Mapping[ModuleName, str | None],
        symlinks: Mapping[ModuleName, ModuleName] = {},
        src2module: Callable[[str], cst.Module | None] = parse_cst_module,
        description: str = "",
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
            description,
        )


def file_content_from_commit(
    project_dir: Path,
    commit: int,
    path: str,
) -> str:
    return subprocess.check_output(
        ["git", "show", f"HEAD~{commit}:{path}"],
        cwd=project_dir,
        text=True,
    )


def project_from_commit(
    project_dir: Path,
    commit: int,
    discard_bad_files: bool = True,
    file_filter: Callable[[Path], bool] = lambda p: True,
    ignore_dirs: set[str] = PythonProject.DefaultIgnoreDirs,
    src2module: Callable[[str], cst.Module | None] = parse_cst_module,
) -> PythonProject:
    """Get the project at a given commit.

    Args:
        project_dir: The directory of the project.
        commit: The commit time step relative to HEAD.

    Returns:
        The project at the given commit.
    """
    # get the list of files in the commit
    files_text = subprocess.check_output(
        ["git", "ls-tree", "-r", f"HEAD~{commit}", "--name-only"],
        cwd=project_dir,
        text=True,
    )

    modules = dict()
    src_map = dict[ModuleName, Path]()
    symlinks = dict()

    all_srcs = [
        f
        for f in (Path(l) for l in files_text.splitlines())
        if f.suffix == ".py"
        and all(p not in ignore_dirs for p in f.parts)
        and file_filter(f)
    ]

    for src in all_srcs:
        # FIXME
        # if src.is_symlink():
        #     continue
        src_text = file_content_from_commit(project_dir, commit, str(src))
        try:
            mod = src2module(src_text)
            if mod is None:
                continue
        except cst.ParserSyntaxError:
            if discard_bad_files:
                continue
            raise

        mod_name = PythonProject.rel_path_to_module_name(src)
        modules[mod_name] = PythonModule.from_cst(mod, mod_name)
        src_map[mod_name] = src

    # for src in all_srcs:
    #     if not src.is_symlink():
    #         continue
    #     mod_name = PythonProject.rel_path_to_module_name(src.relative_to(root))
    #     origin_name = PythonProject.rel_path_to_module_name(
    #         src.resolve().relative_to(root)
    #     )
    #     symlinks[mod_name] = origin_name

    proj = PythonProject(project_dir.resolve(), modules, symlinks, src_map)
    proj.verify_paths_unique()
    return proj


def _path_to_mod_name(path: str) -> ModuleName:
    return PythonProject.rel_path_to_module_name(Path(path))


def edits_from_git_history(
    project_dir: Path,
    start_commit: int,
    end_commit: int = 0,
    src2module: Callable[[str], cst.Module | None] = parse_cst_module,
    ignore_dirs=PythonProject.DefaultIgnoreDirs,
) -> list[ProjectEdit]:
    """Compute the edits to a project from git history.

    Args:
        project_dir: The directory of the project.
        start_commit: The start time step relative to HEAD.
        end_commit: The end time step relative to HEAD.

    Returns:
        A list of edits to the project, from past to present.
    """
    assert end_commit >= 0
    assert start_commit >= end_commit

    def is_src(path_s: str) -> bool:
        path = Path(path_s)
        return path.suffix == ".py" and all(p not in ignore_dirs for p in path.parts)

    project = project_from_commit(project_dir, start_commit, src2module=src2module)
    edits = []

    for t in range(start_commit - 1, end_commit - 1, -1):
        # get commit message
        commit_msg = subprocess.check_output(
            ["git", "show", f"HEAD~{t}", "-s", "--format=%s"],
            cwd=project_dir,
            text=True,
        )
        # get changed files
        changes_text = subprocess.check_output(
            ["git", "diff", f"HEAD~{t+1}", f"HEAD~{t}", "--name-status"],
            cwd=project_dir,
            text=True,
        )

        code_changes = dict[ModuleName, str | None]()
        for line in changes_text.splitlines():
            segs = line.split("\t")
            match segs:
                case ["D", path] if is_src(path):
                    code_changes[_path_to_mod_name(path)] = None
                case ["M" | "A", path] if is_src(path):
                    code_changes[_path_to_mod_name(path)] = file_content_from_commit(
                        project_dir, t, path
                    )
                case [tag, path1, path2] if (
                    tag.startswith("R") and is_src(path1) and is_src(path2)
                ):
                    code_changes[_path_to_mod_name(path2)] = file_content_from_commit(
                        project_dir, t, path2
                    )
                    code_changes[_path_to_mod_name(path1)] = None

        edit = ProjectEdit.from_code_changes(
            project, code_changes, description=commit_msg
        )
        edits.append(edit)
        project = edit.after

    return edits
