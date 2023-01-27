from coeditor.change import Added, Deleted, Modified
from spot.static_analysis import ModuleName
from .defs import *
from typing import *


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
        for stmt in self.iter_imports(self.tree):
            if isinstance(stmt, ptree.ImportFrom):
                for n in stmt.get_from_names():
                    assert isinstance(n, ptree.Name)
                    names.add(n)
            elif isinstance(stmt, ptree.ImportName):
                for n in stmt.get_defined_names():
                    assert isinstance(n, ptree.Name)
                    names.add(n)
        return names

    def iter_imports(self, tree):
        raise NotImplementedError


get_modified_spans = as_any(None)


def get_named_changes(*args):
    raise NotImplementedError


def recurse(scope_change: Change[ChangeScope], parent_changes) -> Iterable[ChangedSpan]:
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
