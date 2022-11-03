# utils to encode and decode code changes into CodeT5 format.

import difflib
from coeditor.history import Change, Modified, Added, Deleted, PythonElem
from .common import *
import copy
import spot.utils

"""
Note that you should avoid using _Tokenizer.encode directly as it
will incorrectly eat the spaces around <add> and <del>.
Use `encode_change` instead.
"""
_Tokenizer = copy.deepcopy(spot.utils.DefaultTokenizer)


class Encoding:
    Add = "<add>"
    Del = "<del>"

    _Tokenizer.add_tokens([Add, Del])

    @staticmethod
    def get_tk_id(token: str) -> int:
        seq = _Tokenizer.encode(token, add_special_tokens=False)
        assert len(seq) == 1
        id = seq[0]
        assert_eq(_Tokenizer.decode([id], add_special_tokens=False), token)
        return id

    Add_id = get_tk_id(Add)
    Del_id = get_tk_id(Del)
    Newline_id = get_tk_id("\n")


for id in [Encoding.Add_id, Encoding.Del_id, Encoding.Newline_id]:
    assert isinstance(id, int), f"Invalid token: {id}"

CodeChange = Change[PythonElem]


def encode_change(change: Change[str]) -> TokenSeq:
    before, after = "", ""
    if isinstance(change, (Deleted, Modified)):
        before = change.before
    if isinstance(change, (Added, Modified)):
        after = change.after
    diffs = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            n=100000,  # don't really have a limit
            lineterm="",
        )
    )[3:]
    rearrage_diffs_(diffs)
    return encode_diffs(diffs)


def decode_change(tokens: TokenSeq) -> Modified[str]:
    tk_lines = list[TokenSeq]()
    for tk in tokens:
        if tk == Encoding.Newline_id:
            tk_lines.append([])
        else:
            if not tk_lines:
                tk_lines.append(TokenSeq())
            tk_lines[-1].append(tk)

    before_lines = list[str]()
    after_lines = list[str]()
    for tk_line in tk_lines:
        if tk_line and tk_line[0] == Encoding.Add_id:
            after_lines.append(_Tokenizer.decode(tk_line[1:]))
        elif tk_line and tk_line[0] == Encoding.Del_id:
            before_lines.append(_Tokenizer.decode(tk_line[1:]))
        else:
            line = _Tokenizer.decode(tk_line)
            before_lines.append(line)
            after_lines.append(line)

    return Modified(before="\n".join(before_lines), after="\n".join(after_lines))


def decode_change_as_string(tokens: TokenSeq) -> str:
    """Directly decode the token sequence using the underlying tokenizer."""
    return _Tokenizer.decode(tokens)


def rearrage_diffs_(diffs: list[str]) -> None:
    """
    Rearrange the diffs (in-place) so that additions appear before deletions
    whenever possible.
    This order should be easier for the model to decode.
    """
    i = 0
    while True:
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

        if i >= len(diffs):
            return


def encode_diffs(diffs: list[str]) -> TokenSeq:
    tokens = TokenSeq()
    for diff in diffs:
        if diff.startswith("+"):
            tokens.append(Encoding.Add_id)
        elif diff.startswith("-"):
            tokens.append(Encoding.Del_id)
        else:
            assert diff.startswith(" ")
        tokens.extend(_Tokenizer.encode(diff[1:], add_special_tokens=False))
        tokens.append(Encoding.Newline_id)
    return tokens
