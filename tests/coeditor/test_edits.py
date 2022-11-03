from coeditor.history import *


def module_from_code(code: str, mname: ModuleName = "ex_module"):
    return PythonModule.from_cst(parse_cst_module(code), mname)


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
        PythonModule.from_cst(parse_cst_module(code), mname)
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
    code_before = dedent(
        """\
        def f1():
            x = 1
            y = 2
            z = x + y
            return z

        def f2():
            f1()\
        """
    )

    code_after = dedent(
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
    )

    c = Modified(code_before, code_after)
    # print(show_change(c))
    c_rec = decode_change(encode_change(c))
    assert c_rec.before.strip("\n") == c.before.strip("\n")
    assert c_rec.after.strip("\n") == c.after.strip("\n")
