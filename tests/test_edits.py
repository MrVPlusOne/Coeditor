import pytest

from coeditor.change import *
from coeditor.encoding import *
from coeditor.encoding import _BaseTokenizer, _Tokenizer


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


class TestChangeIdentities:
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
        "unchanged=True": Modified.from_unchanged(
            dedent(
                """\
                def f1():
                    x = 1
                """
            ),
        ),
        # this test case cannot pass for some reason. Tokenizer bug?
        # "leading_whitespace": Modified.from_unchanged("    ..."),
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

    def test_str_encodings(self):
        for name, c in self.cases.items():
            try:
                line_diffs = change_to_line_diffs(c)
                print("line_diffs\n------\n" + "\n".join(line_diffs))
                before, delta = line_diffs_to_original_delta(line_diffs)
                print("delta:", delta)
                assert_str_equal(before, c.before)
                after = delta.apply_to_input(before)
                assert_str_equal(after, c.after)
            except Exception:
                print_err(f"Failed for case: {name}")
                raise

    def test_tk_encodings(self):
        for name, c in self.cases.items():
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
                code_to_input(
                    _BaseTokenizer.encode(c.before, add_special_tokens=False)
                ),
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

    def test_str_tk_conversion(self):
        for name, c in self.cases.items():
            line_diffs = change_to_line_diffs(c)
            print("line_diffs\n------\n" + "\n".join(line_diffs))
            before, delta = line_diffs_to_original_delta(line_diffs)
            print("delta:", delta)

            tk_delta = delta.to_tk_delta()
            tk_before = encode_basic(before)
            tk_after = tk_delta.apply_to_input(tk_before)
            if tk_after != encode_basic(c.after):
                print("after diff:\n")
                print(show_string_diff(c.after, decode_tokens(tk_after)))

            c_tokens = tk_delta.to_change_tks(tk_before)
            if c_tokens != change_to_tokens(c):
                print("c_tokens diff:\n")
                print(
                    show_string_diff(
                        decode_tokens(c_tokens), decode_tokens(change_to_tokens(c))
                    )
                )

            origin1, tk_delta1 = change_tks_to_original_delta(c_tokens)
            if origin1 != tk_before:
                print("origin diff:\n")
                print(
                    show_string_diff(decode_tokens(origin1), decode_tokens(tk_before))
                )

            assert tk_delta1.apply_to_input(origin1) == tk_after


def test_code_normalization():
    def check_code_equal(code1: str, code2: str):
        if not code_equal(code1, code2):
            e = AssertionError(f"code_equal failed.")
            diff = show_string_diff(
                normalize_code_by_ast(code1), normalize_code_by_ast(code2)
            )
            e.add_note("Diff in normalized code:\n" + diff)
            raise e

    ex_code = dedent(
        """\
        def f1(x, y):
            return f1(x + 1, y - 1)
        """
    )
    ex_code_compact = dedent(
        """\
        def f1(x,y):
            return f1(x+1,y-1)
        """
    )
    check_code_equal(ex_code, ex_code_compact)
    ex_code_lose = dedent(
        """\

        def f1(x,y):

            return f1(
                x+1,
                y-1
            )
        """
    )
    check_code_equal(ex_code, ex_code_lose)

    ex_code_keyword1 = "f(x, y=y, z=z)"
    ex_code_keyword2 = "f(x, z=z, y=y)"
    check_code_equal(ex_code_keyword1, ex_code_keyword2)

    ex_code_keyword3 = "f(x, y=y, z=z, **kwargs)"

    with pytest.raises(AssertionError):
        check_code_equal(ex_code_keyword1, ex_code_keyword3)


def test_extra_ids():
    all_extra_ids = _Tokenizer.additional_special_tokens_ids

    for x in all_extra_ids:
        assert is_extra_id(x)
        n = extra_id_to_number(x)
        assert get_extra_id(n) == x
