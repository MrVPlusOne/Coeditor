# utils for computing editing history from git commits

from spot.static_analysis import (
    ElemPath,
    ModuleAnlaysis,
    ModuleName,
    ProjectPath,
    PythonElem,
    PythonModule,
    PythonProject,
    UsageAnalysis,
    remove_comments,
)
from .common import *
from textwrap import indent
import sys


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


def show_change(
    change: Change[T1],
    name: str = "",
    show_elem: Callable[[T1], str] = lambda x: str(x),
) -> str:
    tab = "   "
    if isinstance(change, Added):
        return f"* Added: {name}\n{indent(show_elem(change.after), tab)}"
    elif isinstance(change, Deleted):
        return f"* Deleted: {name}\n{indent(show_elem(change.before), tab)}"
    elif isinstance(change, Modified):
        before = show_elem(change.before)
        after = show_elem(change.after)
        diff = show_string_diff(before, after)
        return f"* Modified: {name}\n{indent(diff, tab)}"


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
    commit_info: "CommitInfo | None"

    def all_elem_changes(self) -> Generator[Change[PythonElem], None, None]:
        for medit in self.changes.values():
            yield from medit.all_changes.values()

    def __repr__(self):
        if self.commit_info is None:
            return f"ProjectEdit(changed={self.changes})"
        else:
            return (
                f"ProjectEdit(msg={repr(self.commit_info.msg)}, changed={self.changes})"
            )

    @staticmethod
    def from_code_changes(
        before: PythonProject,
        code_changes: Mapping[ModuleName, str | None],
        symlinks: Mapping[ModuleName, ModuleName] = {},
        src2module: Callable[[str], cst.Module | None] = parse_cst_module,
        commit_info: "CommitInfo | None" = None,
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
            commit_info,
        )


def file_content_from_commit(
    project_dir: Path,
    commit: str,
    path: str,
) -> str:
    return run_command(
        ["git", "show", f"{commit}:{path}"],
        cwd=project_dir,
    )


def project_from_commit(
    project_dir: Path,
    commit: str,
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
    files_text = run_command(
        ["git", "ls-tree", "-r", commit, "--name-only"],
        cwd=project_dir,
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


@dataclass
class CommitInfo:
    hash: str
    parents: tuple[str, ...]
    msg: str


def get_commit_history(
    project_dir: Path,
    max_hisotry: int,
    commit_id: str = "HEAD",
) -> list[CommitInfo]:
    """Get the commit history of the project, start from the given `commit_id`,
    going backward in time.
    When a merge commit is encountered, the second parent (the branch that's
    being merged in) is used as the history.
    """
    commit_id = run_command(
        ["git", "rev-parse", commit_id],
        cwd=project_dir,
    ).strip()
    history = []
    for _ in range(max_hisotry):
        lines = run_command(
            ["git", "cat-file", "-p", commit_id],
            cwd=project_dir,
        ).splitlines()
        parents = []
        for line in lines[1:]:
            if line.startswith("parent "):
                parents.append(line.split(" ")[1])
            else:
                break
        commit_msg = run_command(
            ["git", "show", commit_id, "-s", "--format=%s"],
            cwd=project_dir,
        ).strip()
        history.append(CommitInfo(commit_id, tuple(parents), commit_msg))
        if not parents:
            break
        commit_id = parents[-1]
    return history


def edits_from_commit_history(
    project_dir: Path,
    history: Sequence[CommitInfo],
    src2module: Callable[[str], cst.Module | None] = parse_cst_module,
    ignore_dirs=PythonProject.DefaultIgnoreDirs,
    silent=False,
) -> list[ProjectEdit]:
    """Incrementally compute the edits to a project from the git history.

    Returns:
        A list of edits to the project, from past to present.
    """

    def is_src(path_s: str) -> bool:
        path = Path(path_s)
        return path.suffix == ".py" and all(p not in ignore_dirs for p in path.parts)

    commit_now = history[-1]
    with timed_action(
        f"Retriving initial project from commit {commit_now.hash}", silent=silent
    ):
        project = project_from_commit(
            project_dir, commit_now.hash, src2module=src2module
        )
    edits = []

    for commit_next in tqdm(
        list(reversed(history[:-1])),
        desc="Retriving commits",
        disable=silent,
    ):
        # get changed files
        changes_text = run_command(
            ["git", "diff", commit_now.hash, commit_next.hash, "--name-status"],
            cwd=project_dir,
        )

        code_changes = dict[ModuleName, str | None]()
        for line in changes_text.splitlines():
            segs = line.split("\t")
            match segs:
                case ["D", path] if is_src(path):
                    code_changes[_path_to_mod_name(path)] = None
                case ["M" | "A", path] if is_src(path):
                    code_changes[_path_to_mod_name(path)] = file_content_from_commit(
                        project_dir, commit_next.hash, path
                    )
                case [tag, path1, path2] if (
                    tag.startswith("R") and is_src(path1) and is_src(path2)
                ):
                    code_changes[_path_to_mod_name(path2)] = file_content_from_commit(
                        project_dir, commit_next.hash, path2
                    )
                    code_changes[_path_to_mod_name(path1)] = None

        edit = ProjectEdit.from_code_changes(
            project,
            code_changes,
            commit_info=commit_next,
        )
        edits.append(edit)
        project = edit.after
        commit_now = commit_next

    assert_eq(len(edits), len(history) - 1)
    return edits


@dataclass
class ContextualEdit:
    main_change: Modified[PythonElem]
    usee_changes: Sequence[Modified[PythonElem] | Deleted[PythonElem]]
    user_changes: Sequence[Modified[PythonElem] | Deleted[PythonElem]]

    def pprint(self, file=sys.stdout) -> None:
        print("=" * 10, "Main Change", "=" * 10, file=file)
        name = str(self.main_change.before.path)
        print(
            show_change(self.main_change, name, show_elem=lambda x: x.code), file=file
        )

        if self.usee_changes:
            print("=" * 10, "Usee Changes", "=" * 10, file=file)
            for c in self.usee_changes:
                print(
                    show_change(c, str(c.before.path), show_elem=lambda x: x.code),
                    file=file,
                )
                print("-" * 20, file=file)

        if self.user_changes:
            print("=" * 10, "User Changes", "=" * 10, file=file)
            for c in self.user_changes:
                print(
                    show_change(c, str(c.before.path), show_elem=lambda x: x.code),
                    file=file,
                )
                print("-" * 20, file=file)


@dataclass
class EditAnalysis:
    ctx_edits: Sequence[ContextualEdit]
    pedit: ProjectEdit


def analyze_edits(edits: Sequence[ProjectEdit], silent=False) -> list[EditAnalysis]:
    """Perform incremental edit analysis from a sequence of edits."""
    with timed_action("Performing intial module-level analysis...", silent=silent):
        module_analysis = {
            mname: ModuleAnlaysis(m) for mname, m in edits[0].before.modules.items()
        }

    analyzed = list[EditAnalysis]()
    for t, pedit in enumerate(tqdm(edits, desc="Analyzing edits", disable=silent)):
        before_project, after_project = pedit.before, pedit.after
        # pre-edit usage analysis
        uanalysis = UsageAnalysis(
            before_project,
            module_analysis,
            add_implicit_rel_imports=True,
            add_override_usages=True,
        )
        deletions = dict[ProjectPath, Deleted[PythonElem]]()
        modifications = dict[ProjectPath, Modified[PythonElem]]()
        for c in pedit.all_elem_changes():
            if isinstance(c, Deleted):
                deletions[c.before.path] = c
            elif isinstance(c, Modified):
                modifications[c.before.path] = c

        # create contextual edits
        ctx_edits = list[ContextualEdit]()
        ctx_changes = deletions | modifications
        for path, c in modifications.items():
            user_changes, usee_changes = [], []
            for u in uanalysis.user2used.get(path, []):
                if u.used != path and u.used in ctx_changes:
                    usee_changes.append(ctx_changes[u.used])
            for u in uanalysis.used2user.get(path, []):
                if u.user != path and u.user in ctx_changes:
                    user_changes.append(ctx_changes[u.user])
            ctx_edits.append(
                ContextualEdit(c, usee_changes=usee_changes, user_changes=user_changes)
            )

        analyzed.append(EditAnalysis(ctx_edits, pedit))

        if t < len(edits) - 1:
            # update module analysis, reuse when possible
            new_module_analysis = dict[ModuleName, ModuleAnlaysis]()
            for mname, new_m in after_project.modules.items():
                if (
                    old_m := before_project.modules.get(mname)
                ) is not None and old_m.code == new_m.code:
                    new_module_analysis[mname] = module_analysis[mname]
                else:
                    new_module_analysis[mname] = ModuleAnlaysis(new_m)
            module_analysis = new_module_analysis

    return analyzed
