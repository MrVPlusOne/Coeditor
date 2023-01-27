# these code are for testing only

from functools import cached_property
from coeditor.common import *
from parso.python import tree as ptree
from coeditor.change import Change

from spot.static_analysis import ProjectPath

ScopeTree = ptree.Function | ptree.Class | ptree.Module
ChangedSpan = NewType("ChangedSpan", str)


@dataclass
class ChangeScope:
    """
    A change scope is a python module, non-hidden function, or a non-hidden class, or a python module.
        - functions and classes that are inside a parent function are considered hidden.
    """

    path: ProjectPath
    tree: ScopeTree
    spans: Sequence
    subscopes: Mapping[ProjectPath, Self]

    @cached_property
    def spans_code(self) -> str:
        return "\n".join(s.code for s in self.spans)

    @staticmethod
    def from_tree(path: ProjectPath, tree: ScopeTree) -> "ChangeScope":
        spans = []
        subscopes = dict()
        scope = ChangeScope(path, tree, spans, subscopes)
        assert isinstance(tree, ScopeTree)
        is_func = isinstance(tree, ptree.Function)

        current_stmts = []
        content = (
            tree.children
            if isinstance(tree, ptree.Module)
            else cast(ptree.PythonNode, tree.get_suite()).children
        )
        raise NotImplementedError
