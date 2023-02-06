import pytest

from coeditor.change import *
from coeditor.encoding import *
from coeditor.encoding import _BaseTokenizer, _Tokenizer


def get_before(change: Change[str]) -> str:
    if isinstance(change, Modified):
        return change.before
    elif isinstance(change, Added):
        return ""
    elif isinstance(change, Deleted):
        return change.before
    else:
        raise ValueError(f"Unknown change type: {change}")


def get_after(change: Change[str]) -> str:
    if isinstance(change, Modified):
        return change.after
    elif isinstance(change, Added):
        return change.after
    elif isinstance(change, Deleted):
        return ""
    else:
        raise ValueError(f"Unknown change type: {change}")


def assert_change_eq(actual: Change[str], expected: Change[str], name: str):
    if get_before(actual) != get_before(expected):
        print_sections(
            ("Expected before", get_before(expected)),
            ("Reconstructed before", get_before(actual)),
        )
        raise ValueError(f"Failed for case: {name}")
    if get_after(actual) != get_after(expected):
        print_sections(
            ("Expected after", get_after(expected)),
            ("Reconstructed after", get_after(actual)),
        )
        raise ValueError(f"Failed for case: {name}")


def assert_tks_eq(actual: TokenSeq, expected: TokenSeq, name: str):
    if actual != expected:
        print_sections(
            ("Expected", decode_tokens(expected)),
            ("Reconstructed", decode_tokens(actual)),
        )
        raise ValueError(f"Failed for case: {name}")


class TestChangeIdentities:
    cases: dict[str, Change[str]] = {
        "empty": Modified("", ""),
        "generation": Modified("", "123"),
        "added": Added("a\nb\nc\n"),
        "deleted": Deleted("a\nb\nc\n"),
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
        "strings with newlines": Modified(
            dedent(
                """\
                If `True`, wraps the environments in an `AsyncVectorEnv` (which uses \n
                        `multiprocessing` to run the environments in parallel)  \n
                """
            ),
            dedent(
                """\
                If `True`, wraps the environments in an `AsyncVectorEnv` (which uses \n
                        `multiprocessing` to run the environments in parallel)  \n
                Added a line here.   \n
                and here.
                """
            ),
        ),
    }

    def test_str_encodings(self):
        for name, c in self.cases.items():
            try:
                line_diffs = change_to_line_diffs(c)
                print("line_diffs\n------\n" + "\n".join(line_diffs))
                before, delta = line_diffs_to_original_delta(line_diffs)
                print("before:")
                print(before)
                print("delta:", delta)
                assert_str_equal(before, get_before(c))
                after = delta.apply_to_input(before)
                assert_str_equal(after, get_after(c))
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
                code_to_input(encode_lines_join(get_before(c))),
                "change_to_input_output mathese code_to_input: " + name,
            )

            if len(splitlines(get_before(c))) < N_Extra_Ids:
                inlined = inline_output_tokens(in_seq, out_seq)
                if inlined:
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
            tk_before = encode_lines_join(before)
            tk_after = tk_delta.apply_to_input(tk_before)
            if tk_after != encode_lines_join(get_after(c)):
                print("after diff:\n")
                print(show_string_diff(get_after(c), decode_tokens(tk_after)))

            c_tokens = tk_delta.apply_to_change(tk_before)
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

    def test_apply_to_change(self):
        for name, c in self.cases.items():
            before, delta = StrDelta.from_change(c)
            tk_delta = delta.to_tk_delta()
            tk_before = encode_lines_join(before)
            tk_change = tk_delta.apply_to_change(tk_before)
            expect = change_to_tokens(c)
            if tk_change != expect:
                print_sections(
                    ("expect", decode_tokens(expect)),
                    ("tk_change", decode_tokens(tk_change)),
                )
                raise AssertionError(f"apply_to_change failed: {name}")

    def test_delta_decomposition(self):
        for name, c in self.cases.items():
            original, delta = TkDelta.from_change_tks(change_to_tokens(c))
            expect = delta.apply_to_input(original)
            keys = tuple(delta.keys())
            for _ in range(50):
                n_keys = int(len(keys) * random.random())
                sub_keys = random_subset(keys, n_keys)
                delta1, delta2 = delta.decompose_for_input(sub_keys)
                step1 = delta1.apply_to_input(original)
                step2 = delta2.apply_to_input(step1)
                if step2 != expect:
                    print_err(f"{sub_keys=}")
                    print_err("earlier", SEP)
                    print_err(c.earlier)
                    print_err("Original", SEP)
                    print_err(decode_tokens(original))
                    print_err("Expect", SEP)
                    print_err(decode_tokens(expect))
                    print_err("delta1", SEP)
                    print_err(delta1)
                    print_err("step1", SEP)
                    print_err(decode_tokens(step1))
                    print_err("delta2", SEP)
                    print_err(delta2)
                    print_err("step2", SEP)
                    print_err(decode_tokens(step2))
                    raise AssertionError("Failed for case: " + name)

    def test_get_new_target_lines(self):
        for name, c in self.cases.items():
            original, delta = TkDelta.from_change_tks(change_to_tokens(c))
            n_origin_lines = len(split_list(original, Newline_id))
            edit_lines = range(n_origin_lines + 1)
            keys = tuple(delta.keys())
            for _ in range(10):
                n_keys = int(len(keys) * random.random())
                sub_keys = random_subset(keys, n_keys)
                sub_keys.sort()
                delta1, delta2 = delta.decompose_for_change(sub_keys)
                new_edit_lines = delta1.get_new_target_lines(edit_lines)
                new_edit_set = set(new_edit_lines)
                for l in delta2.changed_lines():
                    if l not in new_edit_set and l != n_origin_lines:
                        print_err(f"{edit_lines=}")
                        print_err("original", SEP)
                        print_err(add_line_numbers(decode_tokens(original), start=0))
                        print_err(SEP)
                        print_err(f"{delta=}")
                        print_err(f"{sub_keys=}")
                        print_err(f"{delta1=}")
                        print_err("step1", SEP)
                        step1 = delta1.apply_to_change(original)
                        print_err(add_line_numbers(decode_tokens(step1), start=0))
                        print_err(SEP)
                        print_err(f"{new_edit_lines=}")
                        print_err(f"{delta2=}")
                        raise AssertionError(f"{l=} not in {new_edit_lines=}")


def test_edit_lines_transform():
    ex_code = dedent(
        """\
        a
        b
        c
        d
        e
        """
    )
    ex_delta = StrDelta(
        {
            1: ("+1",),
            2: ("+2",),
            3: ("-",),
            4: ("+d1", "+d2", "+d3"),
        }
    )
    after_expect = dedent(
        """\
         a
        +1
         b
        +2
         c
        -d
        +d1
        +d2
        +d3
         e
        """
    )

    tk_delta = ex_delta.to_tk_delta()
    all_lines = range(6)
    new_target_lines = tk_delta.get_new_target_lines(all_lines)
    expect = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10)
    assert_eq(new_target_lines, expect)

    later_lines = range(3, 6)
    new_target_lines = tk_delta.get_new_target_lines(later_lines)
    # only the last 5 lines should be edited
    expect = (6, 7, 8, 9, 10)
    assert_eq(new_target_lines, expect)


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
