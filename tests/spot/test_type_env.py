import os
import shutil
from pathlib import Path

import pytest

from spot.static_analysis import FunctionSignature, mask_types
from spot.tokenized_src import PreprocessArgs
from spot.type_check import MypyResult, PythonType, remove_top_optional
from spot.type_env import (
    AnnotCat,
    AnnotPath,
    AnyAnnot,
    SelectAnnotations,
    TypeInfAction,
    annot_path,
    apply_annotations,
    collect_annots_info,
    collect_user_annotations,
    mypy_checker,
    normalize_type,
    parse_type_str,
    type_inf_env,
)
from spot.utils import (
    SpecialNames,
    as_any,
    assert_eq,
    cst,
    proj_root,
    read_file,
    write_file,
)

os.chdir(proj_root())


def test_annotation_collection():
    parsed = cst.parse_module(read_file("data/code/env_code_2.py"))
    annots = collect_annots_info(parsed)
    annot_paths = [(a.path, a.cat) for a in annots]
    correct_annot_paths: list[tuple[AnnotPath, AnnotCat]] = [
        (annot_path("fib", "n"), AnnotCat.FuncArg),
        (annot_path("fib", SpecialNames.Return), AnnotCat.FuncReturn),
        (annot_path("foo", "bar"), AnnotCat.FuncArg),
        (annot_path("foo", SpecialNames.Return), AnnotCat.FuncReturn),
        (annot_path("Bar", "z"), AnnotCat.ClassAtribute),
        (annot_path("Bar", "w"), AnnotCat.ClassAtribute),
        (annot_path("Bar", "__init__", SpecialNames.Return), AnnotCat.FuncReturn),
        (annot_path("Bar", "__init__", "x"), AnnotCat.FuncArg),
        (annot_path("Bar", "__init__", "self.x"), AnnotCat.ClassAtribute),
        (annot_path("Bar", "__init__", "self.y"), AnnotCat.ClassAtribute),
        (annot_path("Bar", "reset", "w0"), AnnotCat.FuncArg),
        (annot_path("Bar", "reset", SpecialNames.Return), AnnotCat.FuncReturn),
        (annot_path("Bar", "foo", "z"), AnnotCat.FuncArg),
        (annot_path("Bar", "foo", SpecialNames.Return), AnnotCat.FuncReturn),
        (annot_path("bar"), AnnotCat.GlobalVar),
    ]
    for pair in correct_annot_paths:
        assert pair in annot_paths
    for pair in annot_paths:
        assert pair in correct_annot_paths


def test_self_parameter_annotation():
    code = """
def foo(self: float, x: int) -> str:
    return "1"
"""
    parsed = cst.parse_module(code)
    _, types = collect_user_annotations(parsed)

    assert_eq(types, [PythonType.from_name("int"), PythonType.from_name("str")])
    n_segs = len(mask_types(parsed).code.split(SpecialNames.TypeMask))
    assert_eq(n_segs, len(types) + 1)

    sig = FunctionSignature.from_function(as_any(parsed.body[0]), False)
    assert len(sig.params) == len(types) - 1


parsed = cst.parse_module(read_file("data/code/bad_code_1.py"))


code_1_patch = {
    annot_path("fib", "n"): cst.Annotation(cst.Name("int")),
    annot_path("fib", SpecialNames.Return): cst.Annotation(cst.Name("int")),
    annot_path("t_add", SpecialNames.Return): cst.Annotation(cst.Name("str")),
    annot_path("bad_y"): AnyAnnot,
}


def test_annotation_applying():
    old_annots = collect_annots_info(parsed)
    old_map = {a.path: a.annot for a in old_annots if a.annot is not None}
    new_parsed = apply_annotations(parsed, code_1_patch)
    new_annots = collect_annots_info(new_parsed)
    new_map = {a.path: a.annot for a in new_annots if a.annot is not None}

    for k, v in code_1_patch.items():
        assert old_map[k].annotation.value != new_map[k].annotation.value  # type: ignore
        assert new_map[k].annotation.value == v.annotation.value  # type: ignore


def test_mypy_checker_1():
    with mypy_checker(Path("data/code"), wait_before_check=0.0) as checker:
        check_r = checker.recheck_project()
        assert isinstance(check_r, MypyResult)
        assert Path("data/code/bad_code_1.py").resolve() in check_r.error_dict
        assert Path("data/code/bad_code_2.py").resolve() in check_r.error_dict


def test_type_parsing():
    # test quoted types
    assert parse_type_str("'Foo[int]'") == parse_type_str("Foo[int]")
    assert parse_type_str('"Bar"') == parse_type_str("Bar")


def test_type_normalization():
    equiv_pairs: list[tuple[str, str]] = [
        ("list[int]", "List[int]"),
        ("dict[str, list]", "Dict[str, List]"),
        ("'Foo[int]'", "Foo[int]"),
        ("typing.Union[str, List]", "typing.Union[list, str]"),
        ("typing.Union[str, typing.Union[str, int]]", "str | int"),
        ("typing.Union[str, float, typing.Union[str, int]]", "str | int | float"),
        ("Union[str, float, None]", "Optional[Union[str, float]]"),
        ("str | None", "Optional[str]"),
        ("Any | None", "Optional"),
        ("List[Any]", "List"),
        ("Dict[Any, Any]", "Dict"),
    ]

    for a, b in equiv_pairs:
        ta = parse_type_str(a)
        tb = parse_type_str(b)
        assert normalize_type(ta) == normalize_type(tb)

    nonequiv_pairs: list[tuple[str, str]] = [
        ("Union[str, int]", "Union[str, list]"),
        ("typing.List[str]", "t.List[str]"),
        ("tuple[str, int]", "tuple[int, str]"),
        ("Dict[str, Any]", "Dict"),
    ]

    for a, b in nonequiv_pairs:
        ta = parse_type_str(a)
        tb = parse_type_str(b)
        assert normalize_type(ta) != normalize_type(tb)

    dict_ex = PythonType.from_str("Dict[Any, Any] | None")
    assert remove_top_optional(normalize_type(dict_ex)) == PythonType.from_name("Dict")

    dict2 = PythonType.from_str("Dict[str, Any]")
    assert normalize_type(dict2) == dict2
