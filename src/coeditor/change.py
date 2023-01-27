# utils for computing editing history from git commits

from abc import abstractmethod
from textwrap import indent

from .common import *

E1 = TypeVar("E1", covariant=True)


class _ChangeBase(Generic[E1]):
    def show(self, name: str = "") -> str:
        return show_change(cast("Change", self), name=name)

    @abstractmethod
    def earlier(self) -> E1:
        ...

    @abstractmethod
    def later(self) -> E1:
        ...


@dataclass(frozen=True)
class Added(_ChangeBase[E1]):
    after: E1

    def map(self, f: Callable[[E1], T2]) -> "Added[T2]":
        return Added(f(self.after))

    def earlier(self) -> E1:
        return self.after

    def later(self) -> E1:
        return self.after

    @staticmethod
    def new_value(v: T1) -> "Added[T1]":
        return Added(v)

    @staticmethod
    def as_char():
        return "A"


@dataclass(frozen=True)
class Deleted(_ChangeBase[E1]):
    before: E1

    def map(self, f: Callable[[E1], T2]) -> "Deleted[T2]":
        return Deleted(f(self.before))

    def earlier(self) -> E1:
        return self.before

    def later(self) -> E1:
        return self.before

    @staticmethod
    def new_value(v: T1) -> "Deleted[T1]":
        return Deleted(v)

    @staticmethod
    def as_char():
        return "D"


@dataclass(frozen=True)
class Modified(_ChangeBase[E1]):
    before: E1
    after: E1
    # Used for optimization. If False, `before`` may still equal to `after`.
    unchanged: bool = False

    def map(self, f: Callable[[E1], T2]) -> "Modified[T2]":
        if self.unchanged:
            return Modified.from_unchanged(f(self.before))
        else:
            return Modified(f(self.before), f(self.after))

    def earlier(self) -> E1:
        return self.before

    def later(self) -> E1:
        return self.after

    @staticmethod
    def as_char():
        return "M"

    @staticmethod
    def from_unchanged(v: T1) -> "Modified[T1]":
        return Modified(v, v, unchanged=True)

    def __repr__(self):
        if self.before == self.after:
            return f"Modified(before=after={repr(self.before)})"
        else:
            return f"Modified(before={repr(self.before)}, after={repr(self.after)})"


Change = Added[E1] | Deleted[E1] | Modified[E1]


def default_show_diff(
    before: Any | None, after: Any | None, max_ctx: int | None = 6
) -> str:
    before = str(before) if before is not None else ""
    after = str(after) if after is not None else ""

    return show_string_diff(before, after, max_ctx=max_ctx)


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


# def _select_ast_calls(
#     node: ast.AST, path: ProjectPath
# ) -> Generator[ast.Call, None, None]:
#     """Return all call nodes with the mathcing function name in the AST."""
#     segs = split_dots(path.path)
#     if segs[-1] == "__init__":
#         f_name = segs[-2]
#     else:
#         f_name = segs[-1]
#     for n in ast.walk(node):
#         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
#             if n.func.id == f_name:
#                 yield n


# def find_refactored_calls(
#     pedit: ProjectEdit,
#     pre_analysis: UsageAnalysis,
#     post_analysis: UsageAnalysis,
# ) -> dict[ProjectPath, list[tuple[ProjectPath, Modified[cst.Call]]]]:
#     """Analyze project changes and return a mapping from each function `f` to
#     the refactored callsites within `f`."""

#     changed_apis = set[ProjectPath]()
#     for c in pedit.all_elem_changes():
#         match c:
#             case Modified(before=PythonFunction(), after=PythonFunction()):
#                 if is_signature_changed(c):
#                     changed_apis.add(c.before.path)

#     refactorings = dict[ProjectPath, list[tuple[ProjectPath, Modified[cst.Call]]]]()
#     for c in pedit.modified_functions():
#         path = c.before.path
#         pre_usages = {
#             u.used: u.callsite
#             for u in pre_analysis.user2used.get(path, [])
#             if u.callsite
#         }
#         pos_usages = {
#             u.used: u.callsite
#             for u in post_analysis.user2used.get(path, [])
#             if u.callsite
#         }
#         call_changes = list[tuple[ProjectPath, Modified[cst.Call]]]()
#         for k in changed_apis & pre_usages.keys() & pos_usages.keys():
#             call_before = normalize_code_by_ast(show_expr(pre_usages[k]))
#             call_after = normalize_code_by_ast(show_expr(pos_usages[k]))
#             if call_before != call_after:
#                 call_changes.append((k, Modified(pre_usages[k], pos_usages[k])))
#         refactorings[path] = call_changes
#     return refactorings
