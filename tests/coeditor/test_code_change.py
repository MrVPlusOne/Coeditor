from textwrap import indent

import pytest

from coeditor.code_change import *
from coeditor.encoding import _BaseTokenizer


def test_change_scope():
    code1 = dedent(
        """\
        import os

        x = 1
        y = x + 1

        def f1():
            global x
            x *= 5
            return x

        if __name__ == "__main__":
            print(f1() + x)

        @annotated
        def f2():
            return 1

        @dataclass
        class A:
            attr1: int

            @staticmethod
            def method1():
                return 1

            class B:
                inner_attr1: int
        """
    )
    mod_tree = code_to_module(code1)
    scope = ChangeScope.from_tree(ProjectPath("code1", ""), mod_tree)
    global_spans = [
        dedent(
            """\
            x = 1
            y = x + 1
            """
        ),
        dedent(
            """\
            if __name__ == "__main__":
                print(f1() + x)
            """
        ),
    ]
    try:
        for i, code in enumerate(global_spans):
            assert_str_equal(scope.spans[i].code, code)
    except Exception:
        print_err(f"{scope.spans=}")
        raise

    f1_expect = dedent(
        """\
        global x
        x *= 5
        return x
        """
    )
    f1_code = scope.subscopes["f1"].spans_code
    assert_str_equal(f1_code, indent(f1_expect, " " * 4))

    f2_expect = dedent(
        """\
        @annotated
        def f2():
            return 1
        """
    )
    f2_code = scope.subscopes["f2"].all_code
    assert_str_equal(f2_code, f2_expect)

    attr1_expect = dedent(
        """\
        attr1: int
        """
    )
    attr1_code = scope.subscopes["A"].spans_code
    assert_str_equal(attr1_code, indent(attr1_expect, " " * 4))

    method1_expect = dedent(
        """\
        @staticmethod
        def method1():
            return 1
        """
    )
    method1_code = scope.subscopes["A"].subscopes["method1"].all_code
    assert_str_equal(method1_code, indent(method1_expect, " " * 4))

    inner_attr1_expect = dedent(
        """\
        class B:
            inner_attr1: int
        """
    )
    inner_class_code = scope.subscopes["A"].subscopes["B"].all_code
    assert_str_equal(inner_class_code, indent(inner_attr1_expect, " " * 4))


class TestChangedSpan:
    code1 = dedent(
        """\
        import os

        x = 1
        y = x + 1

        def f1():
            global x
            x *= 5
            return x

        if __name__ == "__main__":
            print(f1() + x)
        """
    )
    scope1 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code1))

    @staticmethod
    def check_changed_spans(
        changed_spans: Sequence[ChangedSpan], *expects: tuple[type, int]
    ):
        print(f"{changed_spans=}\nchanges={[cs.change for cs in changed_spans]}")
        assert_eq(
            len(changed_spans),
            len(expects),
        )
        for i, (change_type, n) in enumerate(expects):
            span = changed_spans[i]
            assert_eq(type(span.change), change_type)
            nl_change = span.change.map(count_lines)
            line_change = nl_change.later() - nl_change.earlier()
            assert_eq(line_change, n, extra_message=lambda: f"{i=}, {span.change=}")

    def test_same_size_update(self):
        code2 = dedent(
            """\
            import os

            x = 1
            y = x + 2

            def f1():
                global x
                x *= 5
                return x + 1

            if __name__ == "__main__":
                print(f1() + x + 1)
            """
        )

        scope2 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code2))
        self.check_changed_spans(
            get_changed_spans(Modified(self.scope1, scope2)),
            (Modified, 0),
            (Modified, 0),
            (Modified, 0),
        )

    def test_diff_size_update(self):
        code2 = dedent(
            """\
            import os

            x = 1
            y = x + 1
            z += 1

            def f1():
                global x
                x *= 5
                return x

            print(f1() + x)
            """
        )
        scope2 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code2))
        self.check_changed_spans(
            get_changed_spans(Modified(self.scope1, scope2)),
            (Modified, 1),
            (Modified, -1),
        )

    def test_fun_deletion(self):
        code2 = dedent(
            """\
            import os

            x = 2

            if __doc__ == "__main__":
                print(f1() + x)
                print("doc")
            """
        )
        scope2 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code2))
        self.check_changed_spans(
            get_changed_spans(Modified(self.scope1, scope2)),
            (Modified, -1),
            (Deleted, 0),
            (Modified, 1),
        )

    def test_fun_addition(self):
        code2 = dedent(
            """\
            import os

            x = 1
            @wrapped
            def new_f():
                pass
            y = x + 1

            def f1():
                global x
                x *= 5
                return x

            if __name__ == "__main__":
                print(f1() + x)
            """
        )
        scope2 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code2))
        self.check_changed_spans(
            get_changed_spans(Modified(self.scope1, scope2)),
            (Added, 0),
        )

    def test_class_addition(self):
        code1 = dedent(
            """\
            import os

            x = 1
            y = x + 1

            if __name__ == "__main__":
                print(f1() + x)
            """
        )

        code2 = dedent(
            """\
            import os

            x = 1
            y = x + 1

            @dataclass
            class Foo():
                "new class"
                x: int = 1
                y: int = 2

            if __name__ == "__main__":
                print(f1() + x)
            """
        )
        scope1 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code1))
        scope2 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code2))
        self.check_changed_spans(
            get_changed_spans(Modified(scope1, scope2)),
            (Added, 0),
        )

    def test_statement_move(self):
        code2 = dedent(
            """\
            import os

            x = 1

            def f1():
                global x
                x *= 5
                return x

            y = x + 1
            if __name__ == "__main__":
                print(f1() + x)
            """
        )
        scope2 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code2))
        self.check_changed_spans(
            get_changed_spans(Modified(self.scope1, scope2)),
        )

    def test_comments_change(self):
        # have to update code as well for the changes to count
        code2 = dedent(
            """\
            import os

            x = 1
            # belongs to f1

            def f1():
                "added doc string"
                global x
                x *= 5
                return x + 1

            # belongs to main
            if __name__ == "__main__":
                print(f1() + x + 1)  # belongs to print
            """
        )
        scope2 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code2))
        self.check_changed_spans(
            get_changed_spans(Modified(self.scope1, scope2)),
            (Modified, -1),
            (Modified, 1),
            (Modified, 1),
        )
