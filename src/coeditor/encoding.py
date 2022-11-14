# utils to encode and decode code changes into CodeT5 format.

import difflib
import copy
import spot.utils
from spot.data import output_ids_as_seqs
from .common import *
from .history import *

"""
Only use this when we want to avoid encoding <add> and <del> as special tokens.
"""
_BaseTokenizer = spot.utils.DefaultTokenizer

Add = "<add>"
Del = "<del>"

"""
`_BaseTokenizer` extended with <add> and <del> tokens. 
Note that you should avoid using _Tokenizer.encode(text) directly as it
will incorrectly eat the spaces around <add> and <del>.
Use `encode_change` instead.
"""
_Tokenizer = copy.deepcopy(_BaseTokenizer)
_Tokenizer.add_tokens([Add, Del])


def get_tk_id(token: str) -> int:
    "Convert a token str into the corresponding integer index."
    seq = _Tokenizer.encode(token, add_special_tokens=False)
    assert len(seq) == 1
    id = seq[0]
    assert_eq(_Tokenizer.decode([id], add_special_tokens=False), token)
    return id


Add_id = get_tk_id(Add)
Del_id = get_tk_id(Del)
Newline_id = get_tk_id("\n")
BOS_id = get_tk_id("<s>")
EOS_id = get_tk_id("</s>")

N_Extra_Ids = 100


_min_extra_id = _Tokenizer.additional_special_tokens_ids[0]
_max_extra_id = _Tokenizer.additional_special_tokens_ids[-1]
assert _min_extra_id < _max_extra_id


def is_extra_id(tk: int) -> bool:
    return _min_extra_id <= tk <= _max_extra_id


def get_extra_id(i: int) -> int:
    assert 0 <= i < N_Extra_Ids
    return _Tokenizer.additional_special_tokens_ids[N_Extra_Ids - 1 - i]


def decode_tokens(tokens: TokenSeq) -> str:
    return _Tokenizer.decode(tokens, add_special_tokens=False)


def change_to_tokens(change: Change[str]) -> TokenSeq:
    "Encode a change as a token sequence."
    c = change.to_modified("")
    diffs = list(
        difflib.unified_diff(
            splitlines(c.before),
            splitlines(c.after),
            n=100000,  # don't really have a limit
            lineterm="",
        )
    )[3:]
    rearrange_diffs_(diffs)
    if not diffs:
        # as a special case, `unified_diff` would return an empty when there is no change.
        diffs = [" " + l for l in splitlines(c.before)]
    return encode_diffs(diffs)


def tokens_to_change(tokens: TokenSeq) -> Modified[str]:
    "Decode a token sequence into a change."
    tk_lines = split_list(tokens, Newline_id, keep_sep=True)

    before_lines = list[str]()
    after_lines = list[str]()
    for tk_line in tk_lines:
        if tk_line and tk_line[0] == Add_id:
            after_lines.append(_Tokenizer.decode(tk_line[1:]))
        elif tk_line and tk_line[0] == Del_id:
            before_lines.append(_Tokenizer.decode(tk_line[1:]))
        else:
            line = _Tokenizer.decode(tk_line)
            before_lines.append(line)
            after_lines.append(line)

    return Modified(before="".join(before_lines), after="".join(after_lines))


def code_to_input(code: str) -> TokenSeq:
    """
    Prepare the code to be edited into the input format and tokenize it.

    In this format, there will be an extra_id token at the beginning of each line.
    Ther will always be an extra_id leading an empty line at the end to allow appending.
    """
    lines = splitlines(code)
    if len(lines) >= N_Extra_Ids:
        raise ValueError(f"Input has more than {N_Extra_Ids} lines: len = {len(lines)}")
    output = TokenSeq()
    for i, line in enumerate(lines):
        output.append(get_extra_id(i))
        output.extend(_BaseTokenizer.encode(line, add_special_tokens=False))

    return output


def change_to_input_output(change: Modified[str]) -> tuple[TokenSeq, TokenSeq]:
    """
    Encode the change as a pair of input and output token sequences.
    If we inline the output tokens into the input tokens, we should
    get back the token sequence corresponding to the given change.
    """
    tks = change_to_tokens(change)
    tk_lines = split_list(tks, Newline_id, keep_sep=True)

    input_lines: list[TokenSeq] = []
    out_buff = TokenSeq()
    output_segs: list[TokenSeq] = []

    for tk_line in tk_lines:
        if tk_line and tk_line[0] == Add_id:
            out_buff.extend(tk_line)
        elif tk_line and tk_line[0] == Del_id:
            input_lines.append(tk_line[1:])
            out_buff.append(Del_id)
            output_segs.append(out_buff)
            out_buff = TokenSeq()
        else:
            input_lines.append(tk_line)
            output_segs.append(out_buff)
            out_buff = TokenSeq()

    if out_buff:
        output_segs[-1].extend(out_buff)

    assert_eq(len(input_lines), len(output_segs))

    for i in range(0, min(N_Extra_Ids, len(input_lines))):
        input_lines[i] = [get_extra_id(i)] + input_lines[i]
        output_segs[i] = [get_extra_id(i)] + output_segs[i]

    input = join_list(input_lines, None)
    output = join_list(output_segs, None)
    return input, output


def inline_output_tokens(
    input: TokenSeq, output: TokenSeq, leave_unpredicted=True
) -> TokenSeq:
    """Inline CodeT5's output tokens into its input tokens."""
    out_map = output_ids_as_seqs(output)
    combined = TokenSeq()
    for tk in input:
        if is_extra_id(tk):
            if tk in out_map:
                combined.extend(out_map[tk])
            elif leave_unpredicted:
                combined.append(tk)
        else:
            combined.append(tk)
    return combined


def rearrange_diffs_(diffs: list[str]) -> None:
    """
    Rearrange the diffs (in-place) so that additions appear before deletions
    whenever possible.
    This order should be easier for the model to decode.
    """
    i = 0
    while True:
        if i >= len(diffs):
            return
        # find the next deletion
        while not diffs[i].startswith("-"):
            i += 1
            if i >= len(diffs):
                return
        del_start = i
        while i < len(diffs) and diffs[i].startswith("-"):
            i += 1
            if i >= len(diffs):
                return
        del_end = i

        if not diffs[i].startswith("+"):
            # no additions to swap with this deletion block
            continue
        # find the end of the current addition block
        add_start = i
        while i < len(diffs) and diffs[i].startswith("+"):
            i += 1
        add_end = i

        # swap the two blocks
        add_block = diffs[add_start:add_end]
        del_block = diffs[del_start:del_end]

        diffs[del_start : del_start + len(add_block)] = add_block
        diffs[add_end - len(del_block) : add_end] = del_block


def encode_diffs(diffs: list[str]) -> TokenSeq:
    """
    A helper function to encode the diff lines (with '+', '-', or ' ' prefixes)
    into a token sequence with the special <add> and <del> tokens.
    """
    tokens = TokenSeq()
    for i, diff in enumerate(diffs):
        if diff.startswith("+"):
            tokens.append(Add_id)
        elif diff.startswith("-"):
            tokens.append(Del_id)
        else:
            assert diff.startswith(" ")
        tokens.extend(_BaseTokenizer.encode(diff[1:], add_special_tokens=False))
    return tokens


@dataclass
class WindowArgs:
    max_window_size: int
    left_ctx_ratio: float = 0.5

    @staticmethod
    def Default() -> "WindowArgs":
        return WindowArgs(4096, 0.5)


@dataclass
class TokenizedEdit:
    path: ProjectPath
    input_tks: TokenSeq
    output_tks: TokenSeq

    def print(self) -> None:
        print("-" * 20, f"Training Example: {self.path}", "-" * 20)
        print("Input:")
        print(indent(decode_tokens(self.input_tks), " " * 4))
        print("Output:")
        print(indent(decode_tokens(self.output_tks), " " * 4))

    def truncate_ctx(
        self,
        args: WindowArgs,
    ) -> "TokenizedEdit":
        return TokenizedEdit(
            self.path, truncate_ctx(self.input_tks, args), self.output_tks
        )


def truncate_ctx(
    input_tks: TokenSeq,
    args: WindowArgs,
) -> TokenSeq:
    """
    Truncate the input to make it fit within the max window size.
    The cutoff is centered around the <extra_id> tokens.
    """
    extra_id_poses = [i for i, tk in enumerate(input_tks) if is_extra_id(tk)]
    assert extra_id_poses
    assert 0 <= args.left_ctx_ratio <= 1
    main_left = extra_id_poses[0]
    main_right = min(extra_id_poses[-1], main_left + args.max_window_size)
    main_size = main_right - main_left + 1
    assert main_size >= 0

    left_size = int((args.max_window_size - main_size) * args.left_ctx_ratio)
    right_size = args.max_window_size - main_size - left_size

    right_ctx_end = min(len(input_tks), main_right + right_size + 1)
    right_size = right_ctx_end - 1 - main_right
    assert right_size >= 0

    # if right_size doesn't use up all the space, we can expand the left context
    left_size = args.max_window_size - main_size - right_size
    left_ctx_start = max(0, extra_id_poses[0] - left_size)
    assert left_size >= 0

    new_input = input_tks[left_ctx_start:right_ctx_end]
    if left_ctx_start > 0:
        new_input[0] = BOS_id
    if right_ctx_end < len(input_tks):
        new_input[-1] = EOS_id
    assert len(new_input) <= args.max_window_size
    return new_input


@dataclass
class FileLevelEditEncoder:
    def encode_edit(
        self,
        pedit: ProjectEdit,
    ) -> list[TokenizedEdit]:
        examples = list[TokenizedEdit]()
        for mname, mchanges in pedit.changes.items():
            modifications = dict[ProjectPath, Modified[PythonElem]]()
            for c in mchanges.all_changes.values():
                if isinstance(c, Modified):
                    modifications[c.before.path] = c
            if not modifications:
                continue
            mod_before = pedit.before.modules[mname]
            code_before = mod_before.code.split("\n")
            mod_after = pedit.after.modules[mname]
            code_after = mod_after.code.split("\n")

            for path, c in modifications.items():
                after_range = mod_after.location_map[c.after.tree]
                code_above_after = "\n".join(code_after[: after_range.start.line - 1])
                code_below_after = "\n".join(code_after[after_range.end.line :])

                before_range = mod_before.location_map[c.before.tree]
                code_above_before = "\n".join(
                    code_before[: before_range.start.line - 1]
                )
                code_below_before = "\n".join(code_before[before_range.end.line :])

                above_change = Modified(code_above_before, code_above_after)
                below_change = Modified(code_below_before, code_below_after)

                ex = TokenizedEdit(path, [], [])
                ex.input_tks.extend(change_to_tokens(above_change))
                input, output = change_to_input_output(c.map(lambda e: e.code))
                ex.input_tks.append(Newline_id)
                ex.input_tks.extend(input)
                ex.input_tks.append(Newline_id)
                ex.output_tks = output
                ex.input_tks.extend(change_to_tokens(below_change))
                examples.append(ex)
        return examples
