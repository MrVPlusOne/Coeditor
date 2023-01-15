# utils for computing editing history from git commits

from spot.static_analysis import (
    ElemPath,
    ModuleAnalysis,
    ModuleName,
    ProjectPath,
    ProjectUsage,
    PythonElem,
    PythonFunction,
    PythonModule,
    PythonProject,
    PythonVariable,
    UsageAnalysis,
    remove_comments,
    CodePosition,
)
from spot.utils import show_expr
from .common import *
from textwrap import indent
import sys
import copy
import ast
from multiprocessing import current_process

E1 = TypeVar("E1", covariant=True)


class _ChangeBase:
    def show(self, name: str = "") -> str:
        return show_change(cast("Change", self), name=name)


@dataclass
class Added(Generic[E1], _ChangeBase):
    after: E1

    def map(self, f: Callable[[E1], T2]) -> "Added[T2]":
        return Added(f(self.after))

    def get_any(self) -> E1:
        return self.after

    @staticmethod
    def new_value(v: T1) -> "Added[T1]":
        return Added(v)

    @staticmethod
    def as_char():
        return "A"


@dataclass
class Deleted(Generic[E1], _ChangeBase):
    before: E1

    def map(self, f: Callable[[E1], T2]) -> "Deleted[T2]":
        return Deleted(f(self.before))

    def get_any(self) -> E1:
        return self.before

    @staticmethod
    def new_value(v: T1) -> "Deleted[T1]":
        return Deleted(v)

    @staticmethod
    def as_char():
        return "D"


@dataclass
class Modified(Generic[E1], _ChangeBase):
    before: E1
    after: E1

    def map(self, f: Callable[[E1], T2]) -> "Modified[T2]":
        return Modified(f(self.before), f(self.after))

    def get_any(self) -> E1:
        return self.before

    @staticmethod
    def as_char():
        return "M"


Change = Added[E1] | Deleted[E1] | Modified[E1]


def default_show_diff(
    before: Any | None, after: Any | None, max_ctx: int | None = 6
) -> str:
    def show_elem(elem: Any) -> str:
        match elem:
            case PythonVariable() | PythonFunction():
                return elem.code
            case _:
                return str(elem)

    def drop_last_line(s: str) -> str:
        return "\n".join(s.split("\n")[:-1])

    match before, after:
        case PythonFunction(tree=tree1) as before, PythonFunction(tree=tree2) as after:
            header1 = drop_last_line(
                show_expr(tree1.with_changes(body=cst.IndentedBlock([]))).strip()
            )
            header2 = drop_last_line(
                show_expr(tree2.with_changes(body=cst.IndentedBlock([]))).strip()
            )
            header_part = (
                header1
                if header1 == header2
                else show_string_diff(header1, header2, max_ctx=None)
            )
            body1 = show_expr(tree1.body)
            body2 = show_expr(tree2.body)
            if body1 == body2:
                body_lines = body1.splitlines()
                kept_lines = body_lines[:6]
                if (n_omit := len(body_lines) - len(kept_lines)) > 0:
                    kept_lines.append(f"... {n_omit} lines omitted ...")
                body_part = "\n".join(kept_lines)
            else:
                body_part = show_string_diff(body1, body2)
            return f"{header_part}\n{body_part}"
        case _:
            s1 = show_elem(before) if before is not None else ""
            s2 = show_elem(after) if after is not None else ""
            return show_string_diff(s1, s2, max_ctx=max_ctx)


def show_change(
    change: Change[T1],
    name: str = "",
    show_diff: Callable[[T1 | None, T1 | None], str] = default_show_diff,
) -> str:
    tab = "  "
    if isinstance(change, Added):
        return f"Added: {name}\n{indent(show_diff(None, change.after), tab)}"
    elif isinstance(change, Deleted):
        return f"Deleted: {name}\n{indent(show_diff(change.before, None), tab)}"
    elif isinstance(change, Modified):
        if change.before == change.after:
            return f"Unchanged: {name}"
        diff = show_diff(change.before, change.after)
        return f"Modified: {name}\n{indent(diff, tab)}"
    else:
        raise TypeError(f"Not a change type: {type(change)}")


def get_named_changes(
    old_map: Mapping[T1, T2], new_map: Mapping[T1, T2]
) -> Mapping[T1, Change[T2]]:
    "Compute the changes between two maps of named elements."
    old_names = set(old_map)
    new_names = set(new_map)
    deleted_names = old_names - new_names
    added_names = new_names - old_names
    modified_names = old_names & new_names
    changes = {}
    for name in deleted_names:
        changes[name] = Deleted(old_map[name])
    for name in added_names:
        changes[name] = Added(new_map[name])
    for name in modified_names:
        changes[name] = Modified(old_map[name], new_map[name])
    return changes


def empty_module(mname: ModuleName) -> PythonModule:
    return PythonModule.from_cst(cst.Module([]), mname, False)


def parse_cst_module(code: str) -> cst.Module:
    return remove_comments(cst.parse_module(code))


def to_modified_function(c: Change):
    if (
        isinstance(c, Modified)
        and isinstance(c.before, PythonFunction)
        and isinstance(c.after, PythonFunction)
    ):
        return cast(Modified[PythonFunction], c)
    else:
        return None


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

    def modified_functions(
        self,
        ast_must_change=True,
        body_must_change=True,
    ) -> dict[ElemPath, Modified[PythonFunction]]:
        def norm_code(f: PythonFunction) -> str:
            if body_must_change:
                code = f.header_body_code[1]
            else:
                code = f.code
            return normalize_code_by_ast(code)

        return {
            k: cast(Modified[PythonFunction], change)
            for k, change in self.modified.items()
            if isinstance(change.before, PythonFunction)
            and isinstance(change.after, PythonFunction)
            and (
                not ast_must_change
                or norm_code(change.before) != norm_code(change.after)
            )
        }

    def added_functions(self) -> dict[ElemPath, Added[PythonFunction]]:
        return {
            k: cast(Added[PythonFunction], change)
            for k, change in self.added.items()
            if isinstance(change.after, PythonFunction)
        }

    def sorted_elems(self, include_classes=True) -> list[ElemPath]:
        """Sort the elements from both before and after the edit into a single ordering."""

        def to_tuple(pos: CodePosition):
            return (pos.line, pos.column)

        after_pos = dict[ElemPath, tuple[int, int]]()
        if include_classes:
            for c in self.after.classes:
                path = c.path.path
                after_pos[path] = to_tuple(self.after.location_map[c.tree].start)
        for e in self.after.all_elements():
            path = e.path.path
            after_pos[path] = to_tuple(self.after.location_map[e.tree].start)

        before_elems = [e.path.path for e in self.before.all_elements()]
        to_prev_elem = {
            before_elems[i]: (before_elems[i - 1] if i > 0 else None)
            for i in range(len(before_elems))
        }
        for path in self.deleted:
            prev = to_prev_elem[path]
            if prev is None:
                after_pos[path] = (0, 0)
            else:
                after_pos[path] = after_pos[prev]
        sorted = list(after_pos)
        sorted.sort(key=lambda path: after_pos[path])
        return sorted

    @staticmethod
    def from_no_change(module: PythonModule) -> "ModuleEdit":
        return ModuleEdit(module, module, {}, {}, {}, {})

    @staticmethod
    def from_modules(before: PythonModule, after: PythonModule) -> "ModuleEdit":
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
            for path in (before_elems.keys() & after_elems.keys())
            if normalize_code_by_ast(before_elems[path].code)
            != normalize_code_by_ast(after_elems[path].code)
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

    def all_elem_changes(self) -> Iterable[Change[PythonElem]]:
        for medit in self.changes.values():
            yield from medit.all_changes.values()

    def modified_functions(
        self,
        ast_must_change=True,
        body_must_change=True,
    ) -> Iterable[Modified[PythonFunction]]:
        for medit in self.changes.values():
            yield from medit.modified_functions(
                ast_must_change=ast_must_change, body_must_change=body_must_change
            ).values()

    def __repr__(self):
        if self.commit_info is None:
            return f"ProjectEdit(changed={self.changes})"
        else:
            return (
                f"ProjectEdit(msg={repr(self.commit_info.msg)}, changed={self.changes})"
            )

    @staticmethod
    def from_module_changes(
        before: PythonProject,
        mod_changes: Mapping[ModuleName, PythonModule | None],
        symlinks: Mapping[ModuleName, ModuleName] = {},
        commit_info: "CommitInfo | None" = None,
    ) -> "ProjectEdit":
        modules = copy.copy(before.modules)
        changes = dict[ModuleName, ModuleEdit]()
        for mname, mod_after in mod_changes.items():
            deleting = mod_after is None
            if mod_after is None:
                mod_after = empty_module(mname)

            mod_before = modules[mname] if mname in modules else empty_module(mname)
            mod_edit = ModuleEdit.from_modules(mod_before, mod_after)
            if deleting:
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

    @staticmethod
    def from_code_changes(
        before: PythonProject,
        code_changes: Mapping[ModuleName, str | None],
        symlinks: Mapping[ModuleName, ModuleName] = {},
        remove_comments: bool = True,
        commit_info: "CommitInfo | None" = None,
    ) -> "ProjectEdit":
        mod_changes = dict[ModuleName, PythonModule | None]()
        for mname, new_code in code_changes.items():
            if new_code is None:
                mod_changes[mname] = None
            else:
                try:
                    m = cst.parse_module(new_code)
                    mod_changes[mname] = PythonModule.from_cst(
                        m, mname, remove_comments
                    )
                except (cst.ParserSyntaxError, cst.CSTValidationError):
                    continue
        return ProjectEdit.from_module_changes(
            before, mod_changes, symlinks, commit_info
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
    drop_comments: bool = True,
    max_workers: int | None = None,
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

    cst_modules = dict[str, cst.Module]()
    for src in all_srcs:
        # FIXME
        # if src.is_symlink():
        #     continue
        src_text = file_content_from_commit(project_dir, commit, str(src))
        try:
            mod = cst.parse_module(src_text)
        except (cst.ParserSyntaxError, cst.CSTValidationError):
            if discard_bad_files:
                continue
            raise

        mod_name = PythonProject.rel_path_to_module_name(src)
        cst_modules[mod_name] = mod
        src_map[mod_name] = src

    if (
        max_workers is None
        and len(cst_modules) > 3
        and current_process().name == "MainProcess"
    ):
        max_workers = None
    else:
        max_workers = 0

    py_modules = pmap(
        PythonModule.from_cst,
        list(cst_modules.values()),
        list(cst_modules.keys()),
        key_args={"drop_comments": drop_comments},
        desc="Creating Python Modules",
        max_workers=max_workers,
        tqdm_args={"disable": max_workers == 0},
    )
    modules = dict(zip(cst_modules.keys(), (py_modules)))

    # TODO: handle simlinks?
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
    max_hisotry: int | None = None,
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
    for _ in range(max_hisotry if max_hisotry else 100000):
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
    drop_comments: bool = True,
    ignore_dirs=PythonProject.DefaultIgnoreDirs,
) -> Iterator[ProjectEdit]:
    """Incrementally compute the edits to a project from the git history.

    Returns:
        A list of edits to the project, from past to present.
    """

    def is_src(path_s: str) -> bool:
        path = Path(path_s)
        return path.suffix == ".py" and all(p not in ignore_dirs for p in path.parts)

    commit_now = history[-1]
    project = project_from_commit(
        project_dir, commit_now.hash, drop_comments=drop_comments
    )

    for commit_next in reversed(history[:-1]):
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
            remove_comments=drop_comments,
        )
        project = edit.after
        commit_now = commit_next
        yield edit


def get_change_path(c: Change[PythonElem]) -> ProjectPath:
    match c:
        case Added(before) | Modified(before):
            return before.path
        case Deleted(after):
            return after.path


TAB = " " * 4


@dataclass
class ContextualEdit:
    main_change: Change[PythonElem]
    grouped_ctx_changes: dict[str, Sequence[Change[PythonElem]]]
    commit_info: CommitInfo | None
    updated_calls: list[tuple[ProjectPath, Modified[cst.Call]]]

    @property
    def path(self) -> ProjectPath:
        return get_change_path(self.main_change)

    def is_ctx_empty(self):
        return all(len(changes) == 0 for changes in self.grouped_ctx_changes.values())

    def pprint(self, file=sys.stdout) -> None:
        if self.commit_info is not None:
            print("Commit:", self.commit_info.hash, file=file)
        print("=" * 15, "Main Change", "=" * 15, file=file)
        print(show_change(self.main_change, str(self.path)), file=file)

        for group, changes in self.grouped_ctx_changes.items():
            if not changes:
                continue
            print(TAB, "=" * 10, group, "=" * 10, file=file)
            for c in changes:
                print(
                    indent(show_change(c, str(get_change_path(c))), TAB),
                    file=file,
                )
                print(TAB, "-" * 20, file=file)


@dataclass
class EditAnalysis:
    ctx_edits: Sequence[ContextualEdit]
    pedit: ProjectEdit


def is_signature_changed(c: Modified[PythonElem]):
    f1 = str(c.before.get_signature())
    f2 = str(c.after.get_signature())
    return f1 != f2


def analyze_edits(
    edits: Sequence[ProjectEdit],
    usees_in_ctx: bool = True,
    users_in_ctx: bool = True,
    post_usages_in_ctx: bool = True,
    record_type_usages: bool = False,
    silent=False,
) -> list[EditAnalysis]:
    """Perform incremental edit analysis from a sequence of edits."""

    timer = TimeLogger()

    if not edits:
        return []

    with timed_action(
        "Performing intial module-level analysis...", silent=silent
    ), timer.timed("ModuleAnlaysis/Initial"):
        module_analysis = {
            mname: ModuleAnalysis(m, record_type_usages=record_type_usages)
            for mname, m in edits[0].before.modules.items()
        }

    def analyze_project_(project: PythonProject) -> UsageAnalysis:
        # analyze changed modules, reusing when possible
        nonlocal module_analysis
        module_analysis = copy.copy(module_analysis)
        for mname, module in project.modules.items():
            if not (
                mname in module_analysis
                and module_analysis[mname].module.code == module.code
            ):
                with timer.timed("ModuleAnlaysis/Incremental"):
                    module_analysis[mname] = ModuleAnalysis(
                        module, record_type_usages=record_type_usages
                    )
        with timer.timed("UsageAnalysis"):
            return UsageAnalysis(
                project,
                module_analysis,
                add_implicit_rel_imports=True,
                add_override_usages=True,
                record_type_usages=record_type_usages,
            )

    def sort_usages(usages: Sequence[ProjectUsage]):
        "Make certain usages come first."
        return sorted(usages, key=lambda u: not u.is_certain)

    analyzed = list[EditAnalysis]()
    pre_analysis = analyze_project_(edits[0].before)

    for pedit in tqdm(edits, desc="Analyzing edits", disable=silent):
        ctx_changes = dict[ProjectPath, Change[PythonElem]]()
        for c in pedit.all_elem_changes():
            ctx_changes[get_change_path(c)] = c

        post_analysis = analyze_project_(pedit.after)

        refactorings = find_refactored_calls(pedit, pre_analysis, post_analysis)

        # create contextual edits
        ctx_edits = list[ContextualEdit]()
        for path, c in ctx_changes.items():
            change_groups = dict[str, list[ProjectPath]]()
            if usees_in_ctx:
                change_groups["usees"] = [
                    u.used for u in sort_usages(pre_analysis.user2used.get(path, []))
                ]
                if post_usages_in_ctx:
                    change_groups["post-usees"] = [
                        u.used
                        for u in sort_usages(post_analysis.user2used.get(path, []))
                    ]
            if users_in_ctx:
                change_groups["users"] = [
                    u.user for u in sort_usages(pre_analysis.used2user.get(path, []))
                ]
                if post_usages_in_ctx:
                    change_groups["post-users"] = [
                        u.user
                        for u in sort_usages(post_analysis.used2user.get(path, []))
                    ]
            grouped_ctx_changes = _select_change_ctx(path, ctx_changes, change_groups)
            ctx_edits.append(
                ContextualEdit(
                    c,
                    grouped_ctx_changes,
                    pedit.commit_info,
                    refactorings.get(path, []),
                )
            )
        analyzed.append(EditAnalysis(ctx_edits, pedit))
        pre_analysis = post_analysis

    if not silent:
        display(timer.as_dataframe())

    return analyzed


def _select_ast_calls(
    node: ast.AST, path: ProjectPath
) -> Generator[ast.Call, None, None]:
    """Return all call nodes with the mathcing function name in the AST."""
    segs = path.path.split(".")
    if segs[-1] == "__init__":
        f_name = segs[-2]
    else:
        f_name = segs[-1]
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id == f_name:
                yield n


def find_refactored_calls(
    pedit: ProjectEdit,
    pre_analysis: UsageAnalysis,
    post_analysis: UsageAnalysis,
) -> dict[ProjectPath, list[tuple[ProjectPath, Modified[cst.Call]]]]:
    """Analyze project changes and return a mapping from each function `f` to
    the refactored callsites within `f`."""

    changed_apis = set[ProjectPath]()
    for c in pedit.all_elem_changes():
        match c:
            case Modified(before=PythonFunction(), after=PythonFunction()):
                if is_signature_changed(c):
                    changed_apis.add(c.before.path)

    refactorings = dict[ProjectPath, list[tuple[ProjectPath, Modified[cst.Call]]]]()
    for c in pedit.modified_functions():
        path = c.before.path
        pre_usages = {
            u.used: u.callsite
            for u in pre_analysis.user2used.get(path, [])
            if u.callsite
        }
        pos_usages = {
            u.used: u.callsite
            for u in post_analysis.user2used.get(path, [])
            if u.callsite
        }
        call_changes = list[tuple[ProjectPath, Modified[cst.Call]]]()
        for k in changed_apis & pre_usages.keys() & pos_usages.keys():
            call_before = normalize_code_by_ast(show_expr(pre_usages[k]))
            call_after = normalize_code_by_ast(show_expr(pos_usages[k]))
            if call_before != call_after:
                call_changes.append((k, Modified(pre_usages[k], pos_usages[k])))
        refactorings[path] = call_changes
    return refactorings


class EditSelectors:
    @staticmethod
    def api_change_to_callsite(ce: ContextualEdit) -> ContextualEdit | None:
        """
        When the main change involves a signature change, try to predict
        the changes to the users.
        """
        if not isinstance(ce.main_change, Modified):
            return None
        api_path = ce.main_change.before.path
        changed_users = list[Modified[PythonElem]]()
        for c in ce.grouped_ctx_changes["users"]:
            if isinstance(c, Modified):
                if isinstance(c.before, PythonFunction):
                    before_calls = [
                        ast.unparse(f)
                        for f in _select_ast_calls(ast.parse(c.before.code), api_path)
                    ]
                    after_calls = [
                        ast.unparse(f)
                        for f in _select_ast_calls(ast.parse(c.after.code), api_path)
                    ]
                    involved = before_calls != after_calls
                else:
                    lines = show_change(c, str(get_change_path(c))).split("\n")
                    api_name = ce.main_change.before.name
                    involved = any(
                        l.strip().startswith("-") and api_name in l for l in lines
                    )
                if involved:
                    changed_users.append(c)
        if changed_users:
            return ContextualEdit(
                ce.main_change,
                {"users": changed_users},
                ce.commit_info,
                ce.updated_calls,
            )
        return None

    @staticmethod
    def usee_changes_to_user(ce: ContextualEdit) -> ContextualEdit | None:
        """
        Try to predict the main change from usee changes.
        """
        usees = ce.grouped_ctx_changes["usees"]
        post_usees = ce.grouped_ctx_changes["post-usees"]
        if usees or post_usees:
            return ContextualEdit(
                ce.main_change,
                {"usees": usees, "post_usees": post_usees},
                ce.commit_info,
                ce.updated_calls,
            )
        return None


def select_edits(
    analyzed_edits: list[EditAnalysis],
    select_edit: Callable[[ContextualEdit], ContextualEdit | None],
) -> tuple[list[ContextualEdit], list[ContextualEdit]]:
    selected = list[ContextualEdit]()
    all_edits = list[ContextualEdit]()
    for ae in analyzed_edits:
        for ce in ae.ctx_edits:
            all_edits.append(ce)
            if (ce := select_edit(ce)) is not None:
                selected.append(ce)

    return selected, all_edits


def _select_change_ctx(
    main_change_path: ProjectPath,
    ctx_changes: Mapping[ProjectPath, Change[PythonElem]],
    change_groups: Mapping[str, Sequence[ProjectPath]],
) -> dict[str, Sequence[Change[PythonElem]]]:
    counted = {main_change_path}
    change_ctx = dict()
    for group, paths in change_groups.items():
        new_changes = []
        for path in paths:
            if path not in counted and path in ctx_changes:
                new_changes.append(ctx_changes[path])
                counted.add(path)
        change_ctx[group] = new_changes
    return change_ctx
