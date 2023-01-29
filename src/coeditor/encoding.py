# utils to encode and decode code changes into CodeT5 format.

import copy
import random
from abc import ABC, abstractmethod
from textwrap import indent

from nltk.translate.bleu_score import sentence_bleu
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from .change import Added, Change, Deleted, Modified, show_change
from .common import *

TokenizerType = RobertaTokenizer


def _turn_off_tokenizer_warning(tokenizer: TokenizerType):
    tokenizer.deprecation_warnings[
        "sequence-length-is-longer-than-the-specified-maximum"
    ] = True


"""
Only use this when we want to avoid encoding <add> and <del> as special tokens.
"""
_BaseTokenizer = cast(
    TokenizerType, TokenizerType.from_pretrained("Salesforce/codet5-base")
)
_turn_off_tokenizer_warning(_BaseTokenizer)

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
PAD_id = get_tk_id("<pad>")

N_Extra_Ids = 100


_min_extra_id = _Tokenizer.additional_special_tokens_ids[0]
_max_extra_id = _Tokenizer.additional_special_tokens_ids[-1]
assert _min_extra_id < _max_extra_id


def is_extra_id(tk: int) -> bool:
    return _min_extra_id <= tk <= _max_extra_id


def get_extra_id(i: int) -> Token:
    assert 0 <= i < N_Extra_Ids
    return _min_extra_id + (N_Extra_Ids - 1 - i)


def extra_id_to_number(tk: Token) -> int:
    assert is_extra_id(tk)
    return _max_extra_id - tk


def decode_tokens(tokens: TokenSeq, prettify: bool = False) -> str:
    text = _Tokenizer.decode(tokens, add_special_tokens=False)
    if prettify:
        text = text.replace("<extra_id_", "<mask_")
    return text


def encode_basic(text: str, add_special_tokens=False) -> TokenSeq:
    "Encode a string into a token sequence using the base tokenizer."
    return _BaseTokenizer.encode(text, add_special_tokens=add_special_tokens)


def output_ids_as_seqs(output_ids: Iterable[Token]) -> dict[Token, TokenSeq]:
    """Parse the CodeT5 model's output as a series of key-value pairs.
    <pad>, <mask>, or <s> or </s> tokens are filtered out."""
    buff = TokenSeq()
    key = None
    seqs = dict[Token, TokenSeq]()

    for tk in output_ids:
        if tk <= 0 or tk == BOS_id or tk == EOS_id:
            continue  # pad, mask token, or sequence token
        if _min_extra_id <= tk <= _max_extra_id:
            if key is not None:
                seqs[key] = buff
            buff = TokenSeq()
            key = tk
        else:
            buff.append(tk)
    if key is not None:
        seqs[key] = buff
    return seqs


def change_to_line_diffs(change: Change[str]) -> list[str]:
    "Encode a change as a token sequence."
    match change:
        case Modified(before, after):
            if change.unchanged:
                diffs = []
            else:
                diffs = compute_line_diffs(splitlines(before), splitlines(after))
            # rearrange_diffs_(diffs)
            if not diffs:
                # as a special case, `unified_diff` would return an empty when there is no change.
                diffs = [" " + l for l in splitlines(before)]
        case Added(after):
            diffs = ["+" + l for l in splitlines(after)]
        case Deleted(before):
            diffs = ["-" + l for l in splitlines(before)]
        case _:
            raise ValueError(f"Invalid change type: {change}.")
    return diffs


@dataclass(frozen=True)
class StrDelta:
    """Stores the line deltas for each line. A line delta is a list of added lines
    (starting with a '+') followed by optionally a `-` line
    (for deleting the current line)."""

    _deltas: Mapping[int, tuple[str, ...]]

    def apply_to_input(self, input: str):
        lines = splitlines(input)
        new_lines = list[str]()
        for i, line in enumerate(lines):
            deleted = False
            if delta := self._deltas.get(i):
                for action in delta:
                    if action[0] == "+":
                        new_lines.append(action[1:])
                    elif action[0] == "-":
                        deleted = True
            if not deleted:
                new_lines.append(line)
        if delta := self._deltas.get(len(lines)):
            for action in delta:
                if action[0] == "+":
                    new_lines.append(action[1:])
        return "\n".join(new_lines)

    def __repr__(self):
        line_diffs = "\n".join(f"  {l}: {a}" for l, a in self._deltas.items())
        return f"StrDelta(\n{line_diffs}\n)"

    def for_input_range(self, line_range: tuple[int, int]) -> Self:
        """Compute the delta for the given line range."""
        a, b = line_range
        new_delta = {k: v for k, v in self._deltas.items() if a <= k < b}
        return StrDelta(new_delta)

    def shifted(self, shift_lines: int) -> Self:
        return StrDelta({k + shift_lines: v for k, v in self._deltas.items()})

    def __bool__(self) -> bool:
        return self.num_changes() > 0

    def to_tk_delta(self) -> "TkDelta":
        deltas = dict[int, tuple[TokenSeq, ...]]()
        for k, line_delta in self._deltas.items():
            line_tk_delta = list[TokenSeq]()
            for action in line_delta:
                if action[0] == "+":
                    line_tk_delta.append([Add_id] + encode_basic(action[1:]))
                elif action[0] == "-":
                    line_tk_delta.append([Del_id])
                else:
                    raise ValueError(f"Invalid action: {action}")
            deltas[k] = tuple(line_tk_delta)
        return TkDelta(deltas)

    def num_changes(self) -> int:
        "Return the number of changed lines in the delta."
        return sum(len(a) for a in self._deltas.values())


def line_diffs_to_original_delta(diffs: list[str]) -> tuple[str, StrDelta]:
    input_lines: list[str] = []
    line_delta: list[str] = []
    deltas = dict[int, tuple[str, ...]]()

    for diff_line in diffs:
        assert diff_line
        assert not diff_line.endswith("\n"), f"bad diff line: {repr(diff_line)}"
        if diff_line[0] == "+":
            line_delta.append(diff_line)
        elif diff_line[0] == "-":
            line_delta.append("-")
            deltas[len(input_lines)] = tuple(line_delta)
            input_lines.append(diff_line[1:])
            line_delta = []
        else:
            assert diff_line[0] == " ", f"unexpected diff_line: {repr(diff_line)}"
            if line_delta:
                deltas[len(input_lines)] = tuple(line_delta)
                line_delta = []
            input_lines.append(diff_line[1:])
    if line_delta:
        deltas[len(input_lines)] = tuple(line_delta)

    str_delta = StrDelta(deltas)
    input = "\n".join(input_lines)
    return input, str_delta


# (line, action id)
DeltaKey = NewType("DeltaKey", tuple[int, int])


@dataclass(frozen=True)
class TkDelta:
    """The Tokenized version of :class:`StrDelta`."""

    _deltas: Mapping[int, tuple[TokenSeq, ...]]

    def changed_lines(self) -> Collection[int]:
        return self._deltas.keys()

    def keys(self) -> Iterable[DeltaKey]:
        for k, _ in self.items():
            yield k

    def items(self) -> Iterable[tuple[DeltaKey, TokenSeq]]:
        for l, acts in self._deltas.items():
            for i, act in enumerate(acts):
                yield DeltaKey((l, i)), act

    def __getitem__(self, key: DeltaKey) -> TokenSeq:
        line, i = key
        return self._deltas[line][i]

    def apply_to_input(self, input: TokenSeq):
        lines = split_list(input, Newline_id)
        new_lines = list[TokenSeq]()
        for i, line in enumerate(lines):
            deleted = False
            if delta := self._deltas.get(i):
                for action in delta:
                    if action[0] == Add_id:
                        new_lines.append(action[1:])
                    elif action[0] == Del_id:
                        deleted = True
            if not deleted:
                new_lines.append(line)
        if delta := self._deltas.get(len(lines)):
            for action in delta:
                if action[0] == Add_id:
                    new_lines.append(action[1:])
        return join_list(new_lines, Newline_id)

    def get_line_change(self, line: int) -> tuple[TokenSeq, ...]:
        return self._deltas.get(line, ())

    def apply_to_change(self, change: TokenSeq) -> TokenSeq:
        lines = split_list(change, Newline_id)

        new_lines = list[TokenSeq]()
        for i, line in enumerate(lines):
            deleted = False
            if delta := self._deltas.get(i):
                for action in delta:
                    if action[0] == Add_id:
                        new_lines.append(action)
                    elif action[0] == Del_id:
                        deleted = True
            if deleted:
                new_lines.append([Del_id] + line)
            else:
                new_lines.append(line)
        if delta := self._deltas.get(len(lines)):
            for action in delta:
                if action[0] == Add_id:
                    new_lines.append(action)
        return join_list(new_lines, Newline_id)

    def __repr__(self):
        line_diffs = "\n".join(
            f"  {k}: {tuple(map(decode_tokens, a))}" for k, a in self._deltas.items()
        )
        return f"TkDelta(\n{line_diffs}\n)"

    def for_input_range(self, line_range: tuple[int, int]) -> Self:
        """Compute the delta for the given line range."""
        a, b = line_range
        new_delta = {k: v for k, v in self._deltas.items() if a <= k < b}
        return TkDelta(new_delta)

    def shifted(self, shift_lines: int) -> Self:
        return TkDelta({k + shift_lines: v for k, v in self._deltas.items()})

    def decompose_for_input(
        self, first_keys: Collection[DeltaKey]
    ) -> tuple[Self, Self]:
        """
        Decompose the delta into two deltas such that applying them sequentially
        using `apply_to_input` is equivalent to applying the original delta.
        """
        key_set = set(first_keys)
        acts1 = dict[int, list[TokenSeq]]()
        acts2 = dict[int, list[TokenSeq]]()
        l_shift = 0
        for l, acts in self._deltas.items():
            for i, act in enumerate(acts):
                key = DeltaKey((l, i))
                if key in key_set:
                    acts1.setdefault(l, []).append(act)
                    l_shift += 1 if act[0] == Add_id else -1
                else:
                    acts2.setdefault(l + l_shift, []).append(act)
        delta1 = TkDelta({k: tuple(v) for k, v in acts1.items()})
        delta2 = TkDelta({k: tuple(v) for k, v in acts2.items()})
        return delta1, delta2

    def decompose_for_change(
        self, first_keys: Collection[DeltaKey]
    ) -> tuple[Self, Self]:
        """
        Decompose the delta into two deltas such that applying them sequentially
        using `apply_to_change` is equivalent to applying the original delta.
        """

        key_set = set(first_keys)
        acts1 = dict[int, list[TokenSeq]]()
        acts2 = dict[int, list[TokenSeq]]()
        l_shift = 0
        for l, acts in self._deltas.items():
            for i, act in enumerate(acts):
                key = DeltaKey((l, i))
                if key in key_set:
                    acts1.setdefault(l, []).append(act)
                    if act[0] == Add_id:
                        l_shift += 1
                    else:
                        assert act[0] == Del_id
                        # the additions cannot be applied to a <del> line
                        if prev_acts := acts2.pop(l + l_shift, None):
                            acts2[l + l_shift + 1] = prev_acts
                else:
                    acts2.setdefault(l + l_shift, []).append(act)
        delta1 = TkDelta({k: tuple(v) for k, v in acts1.items()})
        delta2 = TkDelta({k: tuple(v) for k, v in acts2.items()})
        return delta1, delta2

    def change_groups(self) -> Sequence[Sequence[DeltaKey]]:
        """Group individual changes into logical groups using heuristics.
        Currently, this only groups a <del> immediately followed by an <add>,
        as well as contiguous <del> blocks."""

        def is_key_type(key_id: int, type: Token):
            if key_id >= len(keys):
                return False
            return self[keys[key_id]][0] == type

        def is_next(key1: int, key2: int):
            if key2 >= len(keys):
                return False
            l1 = keys[key1][0]
            l2 = keys[key2][0]
            return l1 == l2 or (l2 == l1 + 1)

        groups = list[tuple[DeltaKey, ...]]()
        keys = tuple(self.keys())
        i = 0
        while i < len(keys):
            # case 1: <del> immediately followed by <add>
            if (
                is_next(i, i + 1)
                and is_key_type(i, Del_id)
                and is_key_type(i + 1, Add_id)
            ):
                groups.append((keys[i], keys[i + 1]))
                i += 2
                continue
            # case 2: contiguous <del> blocks
            if is_key_type(i, Del_id):
                del_block = [keys[i]]
                i += 1
                while (
                    i < len(keys)
                    and is_next(i - 1, i)
                    and is_key_type(i, Del_id)
                    and not is_key_type(i + 1, Add_id)
                ):
                    del_block.append(keys[i])
                    i += 1
                if del_block:
                    groups.append(tuple(del_block))
                    continue
            # case 3: single action
            groups.append((keys[i],))
            i += 1
        assert_eq(join_list(groups), list(keys))
        return groups

    def get_new_target_lines(self, lines: Sequence[int]) -> Sequence[int]:
        """Given a list of lines to edit, return the corresponding new lines to edit
        after applying this delta."""
        if not lines:
            return tuple()
        last_line = lines[-1]
        line_set = set(lines)
        new_edit_lines = list[int]()
        offset = 0
        for l in range(last_line + 1):
            deleted = False
            for act in self.get_line_change(l):
                if act[0] == Add_id:
                    if l in line_set:
                        new_edit_lines.append(l + offset)
                    offset += 1
                elif act[0] == Del_id:
                    deleted = True
            if not deleted and l in line_set:
                new_edit_lines.append(l + offset)
        return tuple(new_edit_lines)

    @staticmethod
    def from_output_tks(tks: TokenSeq) -> "TkDelta":
        ad_tks = (Add_id, Del_id)

        def seg_to_tuple(seg: TokenSeq) -> tuple[TokenSeq]:
            result = list[TokenSeq]()
            ptr = 0
            for i, x in enumerate(seg):
                if i > 0 and x in ad_tks:
                    if seg[ptr] in ad_tks:
                        result.append(seg[ptr:i])
                    ptr = i
            if ptr < len(seg) and seg[ptr] in ad_tks:
                result.append(seg[ptr:])
            return tuple(result)

        segs = output_ids_as_seqs(tks)
        deltas = {
            extra_id_to_number(k): seg_to_tuple(seg) for k, seg in segs.items() if seg
        }
        return TkDelta(deltas)

    def __bool__(self) -> bool:
        return self.num_changes() > 0

    def to_str_delta(self) -> StrDelta:
        deltas = dict[int, tuple[str, ...]]()
        for k, line_delta in self._deltas.items():
            line_str_delta = list[str]()
            for action in line_delta:
                if action[0] == Add_id:
                    line_str_delta.append(f"+{decode_tokens(action[1:])}")
                elif action[0] == Del_id:
                    line_str_delta.append("-")
                else:
                    raise ValueError(f"Invalid action: {decode_tokens(action)}")
            deltas[k] = tuple(line_str_delta)
        return StrDelta(deltas)

    def num_changes(self) -> int:
        "Return the number of changed lines in the delta."
        return StrDelta.num_changes(cast(StrDelta, self))

    @staticmethod
    def from_change_tks(change_tks: TokenSeq) -> tuple[TokenSeq, "TkDelta"]:
        "Return the original input and the delta."
        return change_tks_to_original_delta(change_tks)

    @staticmethod
    def empty() -> "TkDelta":
        return TkDelta({})


def change_tks_to_original_delta(change: TokenSeq) -> tuple[TokenSeq, TkDelta]:
    diffs = split_list(change, Newline_id)
    input_lines: list[TokenSeq] = []
    line_delta: list[TokenSeq] = []
    deltas = dict[int, tuple[TokenSeq, ...]]()

    for diff_line in diffs:
        if diff_line and diff_line[0] == Add_id:
            line_delta.append(diff_line)
        elif diff_line and diff_line[0] == Del_id:
            line_delta.append([Del_id])
            deltas[len(input_lines)] = tuple(line_delta)
            del diff_line[:1]
            input_lines.append(diff_line)
            line_delta = []
        else:
            if line_delta:
                deltas[len(input_lines)] = tuple(line_delta)
                line_delta = []
            input_lines.append(diff_line)
    if line_delta:
        deltas[len(input_lines)] = tuple(line_delta)

    str_delta = TkDelta(deltas)
    input = join_list(input_lines, Newline_id)
    return input, str_delta


def change_to_tokens(change: Change[str]) -> TokenSeq:
    match change:
        case Modified(before=before, after=after, unchanged=unchanged):
            if unchanged or before == after:
                return encode_basic(before)
            else:
                diffs = change_to_line_diffs(change)
                return encode_diffs(diffs)
        case Added() | Deleted():
            lines = split_list(encode_basic(change.earlier), Newline_id)
            tk = Add_id if isinstance(change, Added) else Del_id
            return join_list(([tk] + line for line in lines), Newline_id)
        case _:
            raise AssertionError(f"Not a change type: {change}")


def tokens_to_change(tokens: TokenSeq) -> Modified[str]:
    "Decode a token sequence into a change."
    tk_lines = split_list(tokens, Newline_id)

    before_lines = list[TokenSeq]()
    after_lines = list[TokenSeq]()
    for tk_line in tk_lines:
        if tk_line and tk_line[0] == Add_id:
            after_lines.append(tk_line[1:])
        elif tk_line and tk_line[0] == Del_id:
            before_lines.append(tk_line[1:])
        else:
            before_lines.append(tk_line)
            after_lines.append(tk_line)
    before_code = decode_tokens(join_list(before_lines, Newline_id))
    after_code = decode_tokens(join_list(after_lines, Newline_id))

    return Modified(before_code, after_code)


def code_to_input(code_tks: TokenSeq) -> TokenSeq:
    """
    Convert the original code into model input by inserting <extra_id> tokens.

    In this format, there will be an <extra_id> token at the beginning of each line.
    An additional <extra_id> will be added at the end to allow appending.
    """
    tk_lines = split_list(code_tks, Newline_id)
    tk_lines.append([])
    input_seq = TokenSeq()
    for i, line in enumerate(tk_lines):
        if i < N_Extra_Ids:
            input_seq.append(get_extra_id(i))
        input_seq.extend(line)
        if i < len(tk_lines) - 1:
            input_seq.append(Newline_id)

    return input_seq


def check_output_tokens(tks: TokenSeq) -> bool:
    """Check if a token sequence is a valid output of CodeT5."""
    for i, tk in enumerate(tks):
        if tk == Del_id:
            # a <del> token cannot be followed by normal code
            if i + 1 < len(tks) and not is_extra_id(tks[i + 1]):
                return False
    return True


def change_to_input_output(change: Change[str]) -> tuple[TokenSeq, TokenSeq]:
    """
    Encode the change as a pair of input and output token sequences.
    If we inline the output tokens into the input tokens and drop the
    last newline token, we should get back the token sequence corresponding
    to the given change.

    Note that en extra newline is added to the input to allow appending, as was done
    in `code_to_input`.
    """
    tks = change_to_tokens(change)
    return change_tks_to_input_output(tks)


def change_tks_to_input_output(tks: TokenSeq) -> tuple[TokenSeq, TokenSeq]:
    "See `change_to_input_output`."
    tk_lines = split_list(tks, Newline_id)

    input_lines: list[TokenSeq] = []
    out_buff = TokenSeq()
    output_segs: list[TokenSeq] = []

    for i, tk_line in enumerate(tk_lines):
        if tk_line and tk_line[0] == Add_id:
            out_buff.extend(tk_line)
            out_buff.append(Newline_id)
        elif tk_line and tk_line[0] == Del_id:
            input_lines.append(tk_line[1:])
            out_buff.append(Del_id)
            output_segs.append(out_buff)
            out_buff = TokenSeq()
        else:
            input_lines.append(tk_line)
            output_segs.append(out_buff)
            out_buff = TokenSeq()
    input_lines.append(TokenSeq())
    output_segs.append(out_buff)

    assert_eq(len(input_lines), len(output_segs))

    output_segs = output_segs[:N_Extra_Ids]
    for i in range(0, len(output_segs)):
        input_lines[i] = [get_extra_id(i)] + input_lines[i]
        output_segs[i] = [get_extra_id(i)] + output_segs[i]

    input = join_list(input_lines, Newline_id)
    output = join_list(output_segs, None)
    if not check_output_tokens(output):
        str_segs = [decode_tokens(tks) for tks in output_segs]
        change = tokens_to_change(tks)
        msg = f"Invalid output tokens.\n Output segs: {str_segs}\n Change: {show_change(change)}"
        raise ValueError(msg)
    return input, output


def inline_output_tokens(
    input: TokenSeq, output: TokenSeq, leave_unpredicted=False
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


def encode_diffs(diffs: Sequence[str]) -> TokenSeq:
    """
    A helper function to encode the diff lines (with '+', '-', or ' ' prefixes)
    into a token sequence with the special <add> and <del> tokens.
    """
    prefixes = list[TokenSeq]()
    code_lines = list[str]()
    for i, diff in enumerate(diffs):
        if diff.startswith("+"):
            prefixes.append([Add_id])
        elif diff.startswith("-"):
            prefixes.append([Del_id])
        else:
            assert diff.startswith(" ")
            prefixes.append([])
        code_lines.append(diff[1:])
    code_tks = _BaseTokenizer.encode("\n".join(code_lines), add_special_tokens=False)
    code_lines = split_list(code_tks, Newline_id)
    for i, line in enumerate(code_lines):
        if prefixes[i]:
            code_lines[i] = prefixes[i] + line
    return join_list(code_lines, Newline_id)


def extract_edit_change(input_tks: TokenSeq, output_tks: TokenSeq) -> Modified[str]:
    inlined = inline_output_tokens(input_tks, output_tks)
    return tokens_to_change(inlined)


class TruncateAt(enum.Enum):
    Left = 0
    Right = 1

    def reversed(self) -> Self:
        if self == TruncateAt.Left:
            return TruncateAt.Right
        else:
            return TruncateAt.Left


def break_into_chunks(
    tks: TokenSeq,
    header_f: Callable[[int], TokenSeq],
    chunk_size: int,
    overlap: int,
    right_to_left: bool = False,
    add_bos: bool = True,
    max_return_chunks: int | None = None,
) -> list[TokenSeq]:
    """
    Break the token sequence into chunks with max size `chunk_size`.

    Arguments:
    - `tks` (TokenSeq): a sequence of tokens to be broken into chunks
    - `header_f` (Callable[[int], TokenSeq]): a function that takes in an
    int (representing the chunk number) and returns a sequence of tokens to be used as
    the header for that chunk
    - `chunk_size` (int): the maximum size for each chunk
    - `overlap` (int): the amount of overlap between consecutive chunks
    - `right_to_left` (bool, optional, default=False): a flag indicating whether the
    chunks should be created by going from the right to left
    - `add_bos` (bool, optional, default=True): a flag indicating whether the beginning
    and end of each chunk should be marked with special tokens (BOS and EOS)
    - `max_return_chunks` (int, optional, default=None): the maximum number of chunks
    to return. If None, all chunks will be returned.
    """
    chunks = list[TokenSeq]()
    i = 0
    L = len(tks)
    while i < L:
        chunk_id = len(chunks)
        header = header_f(chunk_id)
        this_overlap = overlap if i > 0 else 0
        progress = chunk_size - len(header) - this_overlap
        assert progress > 0, f"Not making progress: {progress = }"
        body = TokenSeq()
        if right_to_left:
            end = L - (i - this_overlap)
            start = max(0, end - progress)
        else:
            start = i - this_overlap
            end = min(L, start + progress)
        body.extend(tks[start:end])
        if add_bos and i > 0:
            if right_to_left:
                body[-1] = EOS_id
            else:
                body[0] = BOS_id
        if add_bos and i + progress < L - 1:
            if right_to_left:
                body[0] = BOS_id
            else:
                body[-1] = EOS_id
        chunk = header + body
        assert len(chunk) <= chunk_size
        chunks.append(chunk)
        if max_return_chunks is not None and len(chunks) >= max_return_chunks:
            break
        i += progress
    return chunks


def truncate_section(
    sec: TokenSeq,
    direction: TruncateAt,
    limit: int,
    add_bos: bool = True,
    inplace: bool = False,
) -> TokenSeq:
    if len(sec) <= limit:
        return sec

    if direction.value == TruncateAt.Left.value:
        if inplace:
            del sec[:-limit]
        else:
            sec = sec[-limit:]
        if add_bos and sec:
            sec[0] = BOS_id
    else:
        assert_eq(direction.value, TruncateAt.Right.value)
        if inplace:
            del sec[limit:]
        else:
            sec = sec[:limit]
        if add_bos and sec:
            sec[-1] = EOS_id
    return sec


def truncate_sections(
    total_limit: int,
    *sections: tuple[TokenSeq, TruncateAt],
    add_bos: bool,
    inplace: bool = False,
) -> tuple[TokenSeq, ...]:
    """Truncate a list of token sequences to fit within a total length limit.
    Earlier sections have priority over later sections.
    """

    # first, reserve equal space to each section
    section_lens = [total_limit // len(sections) for _ in sections]
    remaining = total_limit
    for i, (tks, _) in enumerate(sections):
        l = min(len(tks), section_lens[i])
        remaining -= l
        section_lens[i] = l
    assert remaining >= 0

    # for the unused space, assign to ealier sections when possible
    for i, (tks, _) in enumerate(sections):
        if remaining <= 0:
            break
        inc = min(remaining, len(tks) - section_lens[i])
        section_lens[i] += inc
        remaining -= inc

    return tuple(
        truncate_section(tks, truncate_dir, section_lens[i], add_bos, inplace=inplace)
        for i, (tks, truncate_dir) in enumerate(sections)
    )


class TokenizedEdit(ABC):
    input_tks: TokenSeq
    output_tks: TokenSeq
    main_tks: TokenSeq
    path: ProjectPath
    change_type: Change[None]

    @abstractmethod
    def all_ctxs(self) -> dict[str, TokenSeq]:
        pass

    def meta_data_lines(self) -> list[str]:
        return [f"path: {str(self.path)}"]

    def stats(self) -> Mapping[str, int | float]:
        return {
            "input_tks": len(self.input_tks),
            "output_tks": len(self.output_tks),
            "main_tks": len(self.main_tks),
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={str(self.path)}, type={type(self.change_type).__name__}, len(input_tks)={len(self.input_tks)}, len(output_tks)={len(self.output_tks)})"

    def show(self, pred_tks: TokenSeq | None = None) -> str:
        def show_label(i: int):
            return f" <{i}>" if i <= 9 else f"<{i}>"

        def show_content(tks: TokenSeq):
            if tks and tks[0] == Add_id:
                return "+ " + decode_tokens(tks[1:])
            elif tks and tks[0] == Del_id:
                return "- " + decode_tokens(tks[1:])
            else:
                return "  " + decode_tokens(tks)

        def show_extra_tokens(tks: TokenSeq, main_tk_lines: dict[Token, TokenSeq]):
            segs = output_ids_as_seqs(tks)
            lines = []
            for k, seg in segs.items():
                if not seg:
                    continue  # skip empty lines
                if seg[-1] == Del_id:
                    # show the deleted line
                    origin_line = split_list(main_tk_lines.get(k, []), Newline_id)[0]
                    origin_line.append(Newline_id)
                    seg = seg + origin_line
                label = show_label(id_map.get(k, -1))
                lines.append(f"{label}:{indent(decode_tokens(seg), ' ' * 4).lstrip()}")
            return "".join(lines)

        def show_ctx(ctx_tks: TokenSeq):
            lines = split_list(ctx_tks, Newline_id)
            return "\n".join("  " + show_content(l) for l in lines)

        main_segs = output_ids_as_seqs(self.main_tks)
        id_map = {k: i for i, k in enumerate(main_segs)}
        main_lines = list[str]()
        for line_tks in split_list(self.main_tks, Newline_id):
            if line_tks and is_extra_id(line_tks[0]):
                prefix = show_label(id_map.get(line_tks[0], -1))
                line = prefix + show_content(line_tks[1:])
            else:
                line = "    " + show_content(line_tks)
            main_lines.append(line)

        pred_lines = (
            ["========Prediction========", f"{show_extra_tokens(pred_tks, main_segs)}"]
            if pred_tks
            else []
        )
        outputs = [
            "-" * 80,
            *self.meta_data_lines(),
            "========Ground Truth========",
            show_extra_tokens(self.output_tks, main_segs),
            *pred_lines,
            "========Main Code========",
            "\n".join(main_lines),
        ] + [
            f"==========={name}===========\n" + show_ctx(tks)
            for name, tks in self.all_ctxs().items()
        ]
        return "\n".join(outputs)

    import warnings

    # turn off redundant BLEU warnings
    warnings.simplefilter(
        "ignore",
        category=UserWarning,
        lineno=552,
    )

    def is_repetitive_edit(self, blue_threshold=0.8) -> bool:
        """Check if all additions in the output_tokens can be matched to
        an addition in the input_tokens with a BLEU score above the threshold."""

        def get_changes(tks, key_tk: Token):
            if tks and tks[0] == key_tk:
                s = decode_tokens(tks[1:])
                s.strip()
                return encode_basic(s)
            else:
                return []

        ctx_lines = split_list(self.input_tks, Newline_id)
        main_lines = output_ids_as_seqs(self.input_tks)
        ctx_addtions = [tks for l in ctx_lines if (tks := get_changes(l, Add_id))]
        ctx_deletions = [tks for l in ctx_lines if (tks := get_changes(l, Del_id))]

        def has_match(line, line_key: Token):
            if line:
                if line[0] == Add_id:
                    added = line[1:]
                    return any(
                        as_any(sentence_bleu([ref], added)) > blue_threshold
                        for ref in ctx_addtions
                    )
                elif line == [Del_id]:
                    if line_key not in main_lines:
                        print(f"Key {decode_tokens([line_key])} not found.")
                        print("Main tokens:")
                        print(decode_tokens(self.main_tks))
                    deleted = main_lines[line_key]
                    return any(
                        as_any(sentence_bleu([ref], deleted)) > blue_threshold
                        for ref in ctx_deletions
                    )
                else:
                    raise ValueError(f"Unexpected line: {decode_tokens(line)}")
            else:
                return True

        out_segs = output_ids_as_seqs(self.output_tks)
        if all(not s for s in out_segs.values()):
            return False
        for k, seg in out_segs.items():
            for line in split_list(seg, Newline_id):
                if not has_match(line, k):
                    return False
        return True

    def is_small_edit(self, max_changes: int = 5) -> bool:
        n_changes = sum(tk == Add_id or tk == Del_id for tk in self.output_tks)
        return n_changes <= max_changes

    def check_extra_ids(self) -> None:
        main_keys = {k for k in self.main_tks if is_extra_id(k)}
        out_keys = {k for k in self.output_tks if is_extra_id(k)}
        assert out_keys.issubset(
            main_keys
        ), f"Output keys not in main keys: {out_keys - main_keys}"


TEdit = TypeVar("TEdit", bound=TokenizedEdit)


def compress_change_tks(tks: TokenSeq, max_ctx: int):
    lines = split_list(tks, sep=Newline_id)
    to_keep = [False for _ in lines]
    # mark which lines to keep
    for i, line in enumerate(lines):
        if line and (line[0] == Add_id or line[0] == Del_id):
            for j in range(max(0, i - max_ctx), min(len(lines), i + max_ctx + 1)):
                to_keep[j] = True
    new_lines = list[TokenSeq]()
    i = 0
    OMIT = encode_basic("...")
    while i < len(lines):
        if to_keep[i]:
            new_lines.append(lines[i])
            i += 1
        else:
            j = i + 1
            while j < len(lines) and not to_keep[j]:
                j += 1
            new_lines.append(OMIT)
            i = j
    return join_list(new_lines, sep=Newline_id)


__ordered_extra_ids = [get_extra_id(i) for i in range(100)]
__random_extra_ids = [get_extra_id(i) for i in range(100)]


def random_extra_id_map() -> dict[Token, Token]:
    """Uniformly randomly map extra_ids to other extra_ids (1-to-1). This can be
    used to improve the training such that every extra_id appears with the same frequency."""
    random.shuffle(__random_extra_ids)
    return dict(zip(__ordered_extra_ids, __random_extra_ids))


def truncate_output_tks(in_tks: TokenSeq, out_tks: TokenSeq) -> TokenSeq:
    in_keys = {tk: None for tk in in_tks if is_extra_id(tk)}
    out_segs = output_ids_as_seqs(out_tks)
    if in_keys.keys() == out_segs.keys():
        return out_tks
    else:
        out = TokenSeq()
        for k in in_keys:
            if k in out_segs:
                out.append(k)
                out.extend(out_segs[k])
        return out


def change_tks_to_query_context(change_tks: TokenSeq, respect_lines: int):
    lines = split_list(change_tks, Newline_id)
    spliter = 0
    result_lines = 0
    for i, l in enumerate(lines):
        if l and l[0] == Del_id:
            pass
        else:
            result_lines += 1
        if result_lines <= respect_lines:
            spliter = i + 1

    context = join_list(lines[:spliter], Newline_id)
    query = change_tks_to_input_output(join_list(lines[spliter:], Newline_id))
    return query, context


def apply_output_tks_to_change(
    change_tks: TokenSeq,
    respect_lines: int,
    out_tks: TokenSeq,
) -> Modified[str]:
    (input_tks, _), context = change_tks_to_query_context(change_tks, respect_lines)
    change_tks = (
        context
        + [Newline_id]
        + inline_output_tokens(input_tks, out_tks, leave_unpredicted=False)
    )
    return tokens_to_change(change_tks)
