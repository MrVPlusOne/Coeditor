import jedi
import pytest

from coeditor._utils import proj_root
from coeditor.c3problem import JediUsageAnalyzer, PyDefinition, PyFullName
from coeditor.common import *

testcase_root = Path(__file__).parent / "testcases"


def assert_has_usages(defs: Collection[PyDefinition], *full_names: str):
    nameset = list(d.full_name for d in defs)
    for name in full_names:
        if PyFullName(name) not in nameset:
            raise AssertionError(f"{name} not in {nameset}")


def assert_no_usages(defs: Collection[PyDefinition], *full_names: str):
    nameset = list(d.full_name for d in defs)
    for name in full_names:
        if PyFullName(name) in nameset:
            raise AssertionError(f"{name} should not be in {nameset}")


def test_anlayzing_defs():
    analyzer = JediUsageAnalyzer()
    project = jedi.Project(path=testcase_root, added_sys_path=[proj_root() / "src"])
    script = jedi.Script(path=testcase_root / "defs.py", project=project)
    analysis = analyzer.get_line_usages(script, range(0, 46), silent=True)

    if analyzer.error_counts:
        raise RuntimeError(f"Errors found: {analyzer.error_counts}")

    assert_has_usages(
        analysis.line2usages[10],
        "defs.ScopeTree",
        "parso.python.tree.Function",
        "parso.python.tree.Class",
        "parso.python.tree.Module",
    )

    assert_has_usages(
        analysis.line2usages[21],
        "defs.ChangeScope.path",
        "coeditor.common.ProjectPath",
    )

    with pytest.raises(AssertionError):
        # wait for jedi to be fixed.
        assert_has_usages(
            analysis.line2usages[22],
            "defs.ChangeScope.tree",
            "defs.ChangeScope",  # include parent usage as well
            "defs.ScopeTree",
        )

        assert_has_usages(
            analysis.line2usages[23],
            "defs.ChangeScope.spans",
            "defs.ChangeScope",
            "typing.Sequence",
        )

    assert_has_usages(
        analysis.line2usages[24],
        "typing.Mapping",
        "coeditor.common.ProjectPath",
    )

    assert_has_usages(
        analysis.line2usages[28],
        "defs.ChangeScope.spans",
    )

    assert_has_usages(
        analysis.line2usages[31],
        "coeditor.common.ProjectPath",
        "defs.ScopeTree",
        # "defs.ChangeScope",  # couldn't handle string annotations for now
    )

    assert_has_usages(
        analysis.line2usages[40],
        "parso.tree.BaseNode.__init__.children",
    )

    assert_has_usages(
        analysis.line2usages[42],
        "parso.python.tree.PythonNode",
        "parso.python.tree.Scope.get_suite",
        # "parso.python.tree.BaseNode.children",
    )


@pytest.mark.xfail(reason="Due to jedi bug")
def test_dataclass_signature():
    s = jedi.Script(
        dedent(
            """\
        from dataclasses import dataclass
        @dataclass
        class Foo:
            bar: int
        """
        )
    )

    defs = s.goto(3, 8)  # go to Foo directly
    assert len(defs) == 1
    n = defs[0]
    assert n._get_docstring_signature() == "Foo(bar: int)"

    defs = s.goto(4, 6)  # first go to bar
    print(f"{len(defs)=}")
    n = defs[0].parent()  # then go to parent
    assert n._get_docstring_signature() == "Foo(bar: int)"


def test_anlayzing_usages():
    analyzer = JediUsageAnalyzer()
    project = jedi.Project(path=testcase_root, added_sys_path=[proj_root() / "src"])
    script = jedi.Script(path=testcase_root / "usages.py", project=project)
    analysis = analyzer.get_line_usages(script, range(0, 63), silent=True)

    if analyzer.error_counts:
        raise RuntimeError(f"Errors found: {analyzer.error_counts}")

    assert_has_usages(
        analysis.line2usages[11],
        "usages.JModule.tree",
        "parso.python.tree.Module",
    )

    assert_has_usages(
        analysis.line2usages[13],
        "usages.JModule._to_scope",
        "defs.ChangeScope",
    )

    assert_has_usages(
        analysis.line2usages[14],
        "usages.JModule.mname",
        "usages.JModule.tree",
        "defs.ChangeScope",
        "defs.ChangeScope.from_tree",
        "coeditor.common.ProjectPath",
    )

    assert_has_usages(
        analysis.line2usages[19],
        "usages.JModule.iter_imports",
    )

    assert_has_usages(
        analysis.line2usages[21],
        # "parso.python.tree.ImportFrom.get_from_names",
    )

    assert_has_usages(
        analysis.line2usages[34],
        "coeditor._utils.as_any",
    )
