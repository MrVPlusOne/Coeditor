from coeditor.code_change import *
from coeditor.encoding import _BaseTokenizer
import pytest

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


def test_change_scope():
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
    for i, code in enumerate(global_spans):
        assert_str_equal(scope.spans[i].code, code)

    inner_code = dedent(
        """\
        def f1():
            global x
            x *= 5
            return x
        """
    )
    f1_code = scope.subscopes[ProjectPath("code1", "f1")].spans_code
    assert_str_equal(f1_code, inner_code)


class TestChangedSpan:
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
            assert_eq(line_change, n)

    def test_same_size_update(self):
        same_size_update = dedent(
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

        scope1 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code1))
        scope2 = ChangeScope.from_tree(
            ProjectPath("code1", ""), code_to_module(same_size_update)
        )
        self.check_changed_spans(
            get_changed_spans(Modified(scope1, scope2)),
            (Modified, 0),
            (Modified, 0),
            (Modified, 0),
        )

    def test_diff_size_update(self):
        diff_size_update = dedent(
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
        scope1 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code1))
        scope2 = ChangeScope.from_tree(
            ProjectPath("code1", ""), code_to_module(diff_size_update)
        )
        self.check_changed_spans(
            get_changed_spans(Modified(scope1, scope2)),
            (Modified, 1),
            (Modified, -1),
        )

    def test_fun_deletion(self):
        fun_deletion_update = dedent(
            """\
            import os
            
            x = 2
                
            if __doc__ == "__main__":
                print(f1() + x)
                print("doc")
            """
        )
        scope1 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code1))
        scope2 = ChangeScope.from_tree(
            ProjectPath("code1", ""), code_to_module(fun_deletion_update)
        )
        self.check_changed_spans(
            get_changed_spans(Modified(scope1, scope2)),
            (Modified, -1),
            (Deleted, 0),
            (Modified, 1),
        )

    def test_comments_change(self):
        # have to update code as well for the changes to count
        comment_update = dedent(
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
        scope1 = ChangeScope.from_tree(ProjectPath("code1", ""), code_to_module(code1))
        scope2 = ChangeScope.from_tree(
            ProjectPath("code1", ""), code_to_module(comment_update)
        )
        self.check_changed_spans(
            get_changed_spans(Modified(scope1, scope2)),
            (Modified, -1),
            (Modified, 3),
            (Modified, 1),
        )
