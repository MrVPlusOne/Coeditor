from coeditor.history import *
from coeditor.encoding import _BaseTokenizer


def module_from_code(code: str, mname: ModuleName = "ex_module"):
    return PythonModule.from_cst(parse_cst_module(code), mname, True)


def test_module_edit_creation():
    code1 = dedent(
        """\
        def to_change():
            return 1
            
        def to_delete():
            return 2
        """
    )

    code2 = dedent(
        """\
        def to_change(x):
            return x

        class A:
            def added():
                return 3
        """
    )

    before = module_from_code(code1)
    after = module_from_code(code2)

    edit = ModuleEdit.from_modules(before, after)
    assert "to_delete" in edit.deleted.keys()
    assert "A.added" in edit.added.keys()
    assert "to_change" in edit.modified.keys()


def project_from_code(srcs: dict[ModuleName, str]) -> PythonProject:
    modules = [
        PythonModule.from_cst(parse_cst_module(code), mname, True)
        for mname, code in srcs.items()
    ]
    return PythonProject.from_modules(
        Path("test_project"),
        modules,
    )


def test_project_edit_creation1():
    code_before = {
        "code1": dedent(
            """\
            def to_change():
                return 1

            def no_change():
                return 2
            """
        ),
        "code2": dedent(
            """\
            # some comments    
            """
        ),
    }

    project_before = project_from_code(code_before)

    code_after = {
        "code1": dedent(
            """\
            def to_change(x):
                return x
            
            def no_change():
                # new comment
                return 2
            """
        ),
        "code2": dedent(
            """\
            # changed comments
            """
        ),
        "code3": dedent(
            """\
            def added():
                return 3
            """
        ),
    }

    pe = ProjectEdit.from_code_changes(
        project_before,
        code_changes=code_after,
    )

    # recompute in case the project is somehow modified
    project_before = project_from_code(code_before)
    project_after = project_from_code(code_after)

    for mname in project_before.modules:
        assert pe.before.modules[mname].code == project_before.modules[mname].code

    for mname in project_after.modules:
        assert pe.after.modules[mname].code == project_after.modules[mname].code

    assert pe.changes["code1"].modified.keys() == {"to_change"}
    assert "code2" not in pe.changes
    assert pe.changes["code3"].added.keys() == {"added"}


def assert_change_eq(actual: Modified[str], expected: Modified[str], name: str):
    if actual.before != expected.before:
        print(f"Failed for case: {name}")
        print("Expected before:\n", expected.before, "<EOF>")
        print("Reconstructed before:\n", actual.before, "<EOF>")
        raise ValueError(f"Failed for case: {name}")
    if actual.after != expected.after:
        print(f"Failed for case: {name}")
        print("Expected after:\n", expected.after, "<EOF>")
        print("Reconstructed after:\n", actual.after, "<EOF>")
        raise ValueError(f"Failed for case: {name}")


def assert_tks_eq(actual: TokenSeq, expected: TokenSeq, name: str):
    if actual != expected:
        print(f"Failed for case: {name}")
        print("Expected:\n", decode_tokens(expected), "<EOF>")
        print("Actual:\n", decode_tokens(actual), "<EOF>")
        raise ValueError(f"Failed for case: {name}")


def test_project_edit_creation2():
    code_before = {
        "code1": dedent(
            """\
            def x():
                return 1

            def y():
                return 2
            """
        ),
    }

    project_before = project_from_code(code_before)

    code_after = {
        "code1": dedent(
            """\
            import new_module
            
            def y():
                return 2
            

            def x():
                return 1
            """
        ),
    }

    project_after = project_from_code(code_after)

    pe = ProjectEdit.from_code_changes(
        project_before,
        code_changes=code_after,
    )

    for mname in project_after.modules:
        assert pe.after.modules[mname].code == project_after.modules[mname].code

    assert "code1" not in pe.changes


from coeditor.encoding import *


def test_encoding_decoding_identity():
    cases = {
        "empty": Modified("", ""),
        "generation": Modified("", "123"),
        "no change": Modified(
            dedent(
                """\
                def f1():
                    x = 1
                """
            ),
            dedent(
                """\
                def f1():
                    x = 1
                """
            ),
        ),
        "replace last": Modified(
            dedent(
                """\
                def f1():
                    x = 1"""
            ),
            dedent(
                """\
                def f1():
                    x = 2
                    return x * 2"""
            ),
        ),
        "no special tokens": Modified(
            dedent(
                """\
                def f1():
                    x = 1
                    y = 2
                    z = x + y
                    return z

                def f2():
                    f1()"""
            ),
            dedent(
                """\
                # new comment
                def f_new():
                    x = 1
                    if x > 0:
                        y = 2 * x
                    y *= 2
                    z = x + y
                    return z

                def f2():
                    f1()
                    return f_new() + a
                
                new_var = 0
                """
            ),
        ),
        "with special tokens": Modified(
            dedent(
                """\
                def f1():
                    x = "<add>"
                    y = "<del>\tx"
                    return x + y

                """
            ),
            dedent(
                """\
                # new comment 1
                # new comment 2
                def f1():
                    if newcond:
                        x = "<add>"
                    new_var = 5
                    y = "<del>"
                    return x + new_var + y
                """
            ),
        ),
        "super long": Modified(
            "\n".join(f"x = {i}" for i in range(0, 200)),
            "\n".join(f"x = {2* (i // 2)}" for i in range(0, 200)),
        ),
    }

    for name, c in cases.items():
        # print(show_change(c))
        c_tokens = change_to_tokens(c)
        print("c_tokens\n------\n", decode_tokens(c_tokens))
        c_rec = tokens_to_change(c_tokens)
        assert_change_eq(
            c_rec, c, "change_to_tokens |> tokens_to_change = identity: " + name
        )

        in_seq, out_seq = change_to_input_output(c)
        print("in_seq\n------\n", decode_tokens(in_seq))
        print("out_seq\n------\n", decode_tokens(out_seq))

        assert_tks_eq(
            in_seq,
            code_to_input(_BaseTokenizer.encode(c.before, add_special_tokens=False)),
            "change_to_input_output mathese code_to_input: " + name,
        )

        if len(c.before.split("\n")) < N_Extra_Ids:
            inlined = inline_output_tokens(in_seq, out_seq)
            assert inlined[-1] == Newline_id
            assert_tks_eq(
                inlined[:-1], change_to_tokens(c), "inline_output_tokens: " + name
            )
            c_rec2 = tokens_to_change(inlined[:-1])
            assert_change_eq(c_rec2, c, "tokens_to_change(inlined): " + name)
