# utils to encode and decode code changes into CodeT5 format.

from abc import ABC, abstractmethod
import asyncio
from dataclasses import field
import difflib
import copy
import random

from textwrap import indent
from spot.static_analysis import (
    ProjectPath,
    PythonElem,
    PythonFunction,
    show_element,
)
import spot.utils
from spot.data import output_ids_as_seqs
from .common import *
from .history import (
    Change,
    Modified,
    Added,
    Deleted,
    ModuleEdit,
    ProjectEdit,
    analyze_edits,
    get_change_path,
    show_change,
)
from nltk.translate.bleu_score import sentence_bleu

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

def extra_id_to_number(tk: int) -> int:
    assert is_extra_id(tk)
    return _max_extra_id - tk

def decode_tokens(tokens: TokenSeq) -> str:
    return _Tokenizer.decode(tokens, add_special_tokens=False)


def encode_basic(text: str, add_special_tokens=False) -> TokenSeq:
    "Encode a string into a token sequence using the base tokenizer."
    return _BaseTokenizer.encode(text, add_special_tokens=add_special_tokens)


def change_to_tokens(change: Change[str]) -> TokenSeq:
    "Encode a change as a token sequence."
    match change:
        case Modified(before, after):
            diffs = list(
                difflib.unified_diff(
                    splitlines(before),
                    splitlines(after),
                    n=100000,  # don't really have a limit
                    lineterm="",
                )
            )[3:]
            rearrange_diffs_(diffs)
            if not diffs:
                # as a special case, `unified_diff` would return an empty when there is no change.
                diffs = [" " + l for l in splitlines(before)]
        case Added(after):
            diffs = ["+" + l for l in splitlines(after)]
        case Deleted(before):
            diffs = ["-" + l for l in splitlines(before)]
        case _:
            raise ValueError(f"Invalid change type: {change}.")
    return encode_diffs(diffs)


def tokens_to_change(tokens: TokenSeq) -> Modified[str]:
    "Decode a token sequence into a change."
    tk_lines = split_list(tokens, Newline_id)

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

    return Modified(before="\n".join(before_lines), after="\n".join(after_lines))


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


# TODO: change this to output a sequence of pairs
# also take in a list of chunk start lines
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
        if i < len(diffs) - 1:
            tokens.append(Newline_id)
    return tokens


def extract_edit_change(input_tks: TokenSeq, output_tks: TokenSeq) -> Modified[str]:
    inlined = inline_output_tokens(input_tks, output_tks)
    return tokens_to_change(inlined)


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

    def show(self) -> str:
        return self.show_prediction(None)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={str(self.path)}, type={type(self.change_type).__name__}, len(input_tks)={len(self.input_tks)}, len(output_tks)={len(self.output_tks)})"

    def show_prediction(self, pred_tks: TokenSeq | None = None) -> str:
        def show_label(i: int):
            return f" <{i}>" if i <= 9 else f"<{i}>"

        def show_extra_tokens(tks: TokenSeq, main_tk_lines: dict[Token, TokenSeq]):
            segs = output_ids_as_seqs(tks)
            lines = []
            for k, seg in segs.items():
                if not seg:
                    continue  # skip empty lines
                if seg[-1] == Del_id:
                    # show the delted line
                    origin_line = main_tk_lines.get(k, [])
                    seg = seg + origin_line
                label = show_label(id_map.get(k, -1))
                lines.append(f"{label}:{indent(decode_tokens(seg), ' ' * 4).lstrip()}")
            return "".join(lines)

        main_segs = output_ids_as_seqs(self.main_tks)
        id_map = {k: i for i, k in enumerate(main_segs)}
        main_lines = list[str]()
        for line_tks in split_list(self.main_tks, Newline_id):
            if line_tks and is_extra_id(line_tks[0]):
                line = show_label(id_map.get(line_tks[0], -1)) + decode_tokens(
                    line_tks[1:]
                )
            else:
                line = decode_tokens(line_tks)
            main_lines.append(line)

        pred_lines = (
            ["========Prediction========", f"{show_extra_tokens(pred_tks, main_segs)}"]
            if pred_tks
            else []
        )
        outputs = [
            *self.meta_data_lines(),
            "========Ground Truth========",
            show_extra_tokens(self.output_tks, main_segs),
            *pred_lines,
            "========Main Code========",
            "\n".join(main_lines),
        ] + [
            f"==========={name}===========\n" + decode_tokens(tks)
            for name, tks in self.all_ctxs().items()
        ]
        return "\n".join(outputs)

    def inline_changes(self, lines: int) -> "TokenizedEdit | None":
        """Inline the first `lines` lines of the output changes into the main code.
        Will return None if the remaining changes to be predicted are empty.
        """
        out_dict = output_ids_as_seqs(self.output_tks)
        to_inline = TokenSeq()
        to_predict = TokenSeq()
        for i, k in enumerate(out_dict.keys()):
            if i < lines:
                to_inline.append(k)
                to_inline.extend(out_dict[k])
            else:
                to_predict.append(k)
                to_predict.extend(out_dict[k])
        if not to_predict:
            # the remaining changes are empty
            return None
        main_tks = inline_output_tokens(
            self.main_tks, to_inline, leave_unpredicted=True
        )
        edit = copy.copy(self)
        edit.main_tks = main_tks
        edit.output_tks = to_predict
        return edit

    def inline_signature_changes(self) -> "TokenizedEdit | None":
        """If this edit is applied on a function, inline all the changes
        appeared in the function signature."""
        if EOS_id in self.main_tks:
            return None
        change = extract_edit_change(self.main_tks, self.output_tks)
        try:
            mod = cst.parse_module(dedent(change.before))
        except Exception:
            print("Failed to parse the code:\n" + change.before)
            raise
        match mod.body:
            case [cst.FunctionDef() as f]:
                f = f.with_changes(body=cst.IndentedBlock([]))
                f_code = show_expr(f)
                header_lines = len(f_code.split("\n")) - 1
                return self.inline_changes(lines=header_lines)
        return None

    def prefix_from_signature(self) -> TokenSeq:
        """Get a the prefix of the output_ids corresponding to the function
        signature changes. This can be used to constrain decoding."""
        if EOS_id in self.main_tks:
            return TokenSeq()
        change = extract_edit_change(self.main_tks, self.output_tks)
        try:
            mod = cst.parse_module(dedent(change.before))
        except Exception:
            print("Failed to parse the code:\n" + change.before)
            return TokenSeq()
        match mod.body:
            case [cst.FunctionDef() as f]:
                f = f.with_changes(body=cst.IndentedBlock([]))
                f_code = show_expr(f)
                header_lines = len(f_code.split("\n")) - 1
                out_dict = output_ids_as_seqs(self.output_tks)
                prefix_tks = TokenSeq()
                for i, k in enumerate(out_dict.keys()):
                    if i < header_lines:
                        prefix_tks.append(k)
                        prefix_tks.extend(out_dict[k])
                return prefix_tks
        return TokenSeq()

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


class TruncateAt(enum.Enum):
    Left = 0
    Right = 1


def break_into_chunks(
    tks: TokenSeq, chunk_size: int, overlap: int, add_bos: bool
) -> list[TokenSeq]:
    # break the token sequence into chunks of size chunk_size
    chunks = list[TokenSeq]()
    for i in range(0, len(tks), chunk_size - overlap):
        chunk = tks[i : i + chunk_size]
        if add_bos:
            if i > 0:
                chunk[0] = BOS_id
            if i + chunk_size < len(tks):
                chunk[-1] = EOS_id
        chunks.append(chunk)
    return chunks


def truncate_section(
    sec: TokenSeq, direction: TruncateAt, limit: int, add_bos: bool
) -> TokenSeq:
    if len(sec) <= limit:
        return sec

    if direction == TruncateAt.Left:
        sec = sec[-limit:]
        if add_bos and sec:
            sec[0] = BOS_id
    else:
        assert_eq(direction, TruncateAt.Right)
        sec = sec[:limit]
        if add_bos and sec:
            sec[-1] = EOS_id
    return sec


def truncate_sections(
    total_limit: int,
    *sections: tuple[TokenSeq, TruncateAt],
    add_bos: bool,
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
        truncate_section(tks, truncate_dir, section_lens[i], add_bos)
        for i, (tks, truncate_dir) in enumerate(sections)
    )


@dataclass
class FileBasedTokenizedEdit(TokenizedEdit):
    main_tks: TokenSeq
    left_tks: TokenSeq
    right_tks: TokenSeq
    output_tks: TokenSeq
    path: ProjectPath
    change_type: Change[None]
    add_truncate_bos: bool

    @property
    def input_tks(self) -> TokenSeq:
        return join_list([self.left_tks, self.main_tks, self.right_tks], sep=Newline_id)

    def all_ctxs(self) -> dict[str, TokenSeq]:
        return {
            "left context": self.left_tks,
            "right context": self.right_tks,
        }


# MainPrompt = encode_basic("\n# EDIT:\n")
MainPrompt = TokenSeq()

TEdit = TypeVar("TEdit", bound=TokenizedEdit)


class EditEncoder(Generic[T1], ABC):
    # If True, will only add BOS and EOS tokens to the truncated sections.
    add_truncate_bos: bool = True

    @abstractmethod
    def encode_pedit(
        self,
        pedit: ProjectEdit,
        include_additions: bool = False,
    ) -> Iterable[T1]:
        pass

    def maybe_wrap_bos(self, tks: TokenSeq) -> TokenSeq:
        "Wrap the tokens with BOS and EOS tokens if `Add_Truncation_BOS` is False."
        if self.add_truncate_bos:
            return tks
        else:
            return [BOS_id] + tks + [EOS_id]

    def maybe_wrap_bos_code(self, code: str) -> str:
        "Wrap the tokens with BOS and EOS tokens if `Add_Truncation_BOS` is False."
        if self.add_truncate_bos:
            return code
        else:
            return f"<s>{code}</s>"


@dataclass
class FileBasedEditEncoder(EditEncoder[FileBasedTokenizedEdit]):
    n_max_tks: int = 4000
    add_truncate_bos: bool = True

    def encode_pedit(
        self,
        pedit: ProjectEdit,
        include_additions: bool = False,
    ) -> Iterable[FileBasedTokenizedEdit]:
        for me in pedit.changes.values():
            yield from self.encode_medit(me)

    def encode_medit(
        self,
        medit: ModuleEdit,
    ) -> Iterable[FileBasedTokenizedEdit]:
        modifications = medit.modified_functions(ast_must_change=True)
        if not modifications:
            return
        mod_before = medit.before
        code_before = self.maybe_wrap_bos_code(mod_before.code).split("\n")
        mod_after = medit.after
        code_after = self.maybe_wrap_bos_code(mod_after.code).split("\n")
        mod_name = mod_after.name
        ctx_lines = self.n_max_tks // 2

        for path, c in modifications.items():
            after_range = mod_after.location_map[c.after.tree]
            after_start = after_range.start.line - 1
            after_end = after_range.end.line
            before_range = mod_before.location_map[c.before.tree]
            before_start = before_range.start.line - 1
            before_end = before_range.end.line

            code_main_before = "\n".join(code_before[before_start:before_end])
            code_main_after = "\n".join(code_after[after_start:after_end])

            code_above_after = "\n".join(
                code_after[max(0, after_start - ctx_lines) : after_start]
            )
            code_below_after = "\n".join(
                code_after[after_end : min(len(code_after), after_end + ctx_lines)]
            )

            code_above_before = "\n".join(
                code_before[max(0, before_start - ctx_lines) : before_start]
            )
            code_below_before = "\n".join(
                code_before[before_end : min(len(code_before), before_end + ctx_lines)]
            )

            above_change = Modified(code_above_before, code_above_after)
            below_change = Modified(code_below_before, code_below_after)

            above_tks = change_to_tokens(above_change)
            below_tks = change_to_tokens(below_change)
            input, output = change_to_input_output(
                Modified(code_main_before.strip("\n"), code_main_after.strip("\n"))
            )

            main_tks, above_tks, below_tks = truncate_sections(
                self.n_max_tks - len(MainPrompt) - 1,
                (input, TruncateAt.Right),
                (above_tks, TruncateAt.Left),
                (below_tks, TruncateAt.Right),
                add_bos=self.add_truncate_bos,
            )
            above_tks.extend(MainPrompt)
            output_tks = truncate_output_tks(main_tks, output)
            if not output_tks:
                # can happen if input too long
                continue

            edit = FileBasedTokenizedEdit(
                main_tks=main_tks,
                output_tks=output_tks,
                left_tks=above_tks,
                right_tks=below_tks,
                path=ProjectPath(mod_name, path),
                change_type=c.map(lambda _: None),
                add_truncate_bos=self.add_truncate_bos,
            )
            yield edit


@dataclass
class CstBasedTokenizedEdit(TokenizedEdit):
    main_tks: TokenSeq
    left_tks: TokenSeq
    right_tks: TokenSeq
    output_tks: TokenSeq
    path: ProjectPath
    change_type: Change[None]
    elems: set[ProjectPath]

    @property
    def input_tks(self) -> TokenSeq:
        return join_list([self.left_tks, self.main_tks, self.right_tks], sep=Newline_id)

    def all_ctxs(self) -> dict[str, TokenSeq]:
        return {
            "left context": self.left_tks,
            "right context": self.right_tks,
        }

    def truncate_ctx_(self, length: int):
        ctx_len = length - len(self.main_tks)
        current_ctx = len(self.left_tks) + len(self.right_tks)
        assert current_ctx > ctx_len, "Can't truncate to a larger length"
        ratio = ctx_len / current_ctx
        n_left = int(len(self.left_tks) * ratio)
        n_right = ctx_len - n_left
        self.left_tks = self.left_tks[-n_left:]
        if self.left_tks:
            self.left_tks[0] = BOS_id
        self.right_tks = self.right_tks[:n_right]
        if self.right_tks:
            self.right_tks[-1] = EOS_id


@dataclass
class CstBasedEditEncoder(EditEncoder[CstBasedTokenizedEdit]):
    n_max_tks: int = 4000
    collapse_unchanged: bool = True
    add_truncate_bos: bool = True

    def encode_pedit(
        self,
        pedit: ProjectEdit,
        include_additions: bool = False,
    ) -> Iterable[CstBasedTokenizedEdit]:
        def get_selected(
            elems: Iterable[ProjectPath], elem_lens: Iterable[int], selection_len: int
        ):
            for e, e_len in zip(elems, elem_lens):
                yield e
                selection_len -= e_len
                if selection_len <= 0:
                    break
            pass

        ctx_encoder = CtxEncoder(pedit, self.collapse_unchanged)
        for mname, medit in pedit.changes.items():
            mod_fs = medit.modified_functions(ast_must_change=True)
            if include_additions:
                mod_fs |= medit.added_functions()
            if not mod_fs:
                continue

            sorted_elems = [
                ProjectPath(mname, p) for p in medit.sorted_elems(include_classes=True)
            ]

            for i, path in enumerate(sorted_elems):
                if (c := mod_fs.get(path.path)) is None:
                    continue
                body_change = c.map(lambda x: x.header_body_code[1])
                if (
                    isinstance(body_change, Modified)
                    and count_lines(body_change.before) > 99
                ):
                    # skip functions that are too long
                    continue
                main_tks, output_tks = change_to_input_output(body_change)
                output_tks = truncate_section(
                    output_tks, TruncateAt.Right, self.n_max_tks, self.add_truncate_bos
                )
                if not output_tks:
                    print("Body change:\n", body_change)
                    print("Main change:\n", c.map(lambda x: x.code))
                    raise RuntimeError("No output tokens")
                left_etks = [
                    ctx_encoder.encode_ctx_element(p) for p in sorted_elems[:i]
                ]
                right_etks = [
                    ctx_encoder.encode_ctx_element(p) for p in sorted_elems[i + 1 :]
                ]
                header_tks = change_to_tokens(c.map(lambda x: x.header_body_code[0]))
                main_tks = header_tks + [Newline_id] + main_tks
                left_ctx = join_list(left_etks, sep=Newline_id)
                right_ctx = join_list(right_etks, sep=Newline_id)
                if not self.add_truncate_bos:
                    left_ctx = [BOS_id] + left_ctx
                    right_ctx.append(EOS_id)
                main_tks, left_tks, right_tks = truncate_sections(
                    self.n_max_tks - len(MainPrompt) - 1,
                    (main_tks, TruncateAt.Right),
                    (left_ctx, TruncateAt.Left),
                    (right_ctx, TruncateAt.Right),
                    add_bos=self.add_truncate_bos,
                )
                output_tks = truncate_output_tks(main_tks, output_tks)
                if not output_tks:
                    # can happen if input too long
                    continue

                selected = {path}
                for e in get_selected(
                    reversed(sorted_elems[:i]),
                    reversed([len(e) for e in left_etks]),
                    len(left_tks),
                ):
                    selected.add(e)

                for e in get_selected(
                    sorted_elems[i + 1 :],
                    [len(e) for e in right_etks],
                    len(right_tks),
                ):
                    selected.add(e)

                left_tks.extend(MainPrompt)
                ex = CstBasedTokenizedEdit(
                    main_tks=main_tks,
                    left_tks=left_tks,
                    right_tks=right_tks,
                    output_tks=output_tks,
                    path=path,
                    change_type=c.map(lambda _: None),
                    elems=selected,
                )
                yield ex


@dataclass
class AnalysisBasedTokenizedEdit(TokenizedEdit):
    main_tks: TokenSeq
    left_tks: TokenSeq
    right_tks: TokenSeq
    extra_tks: TokenSeq
    output_tks: TokenSeq
    path: ProjectPath
    change_type: Change[None]
    elems: set[ProjectPath]
    updated_calls: list[tuple[ProjectPath, Modified[cst.Call]]]

    def meta_data_lines(self) -> list[str]:
        updated_calls = [f"Call updated: {str(p)}" for p, _ in self.updated_calls]
        return [f"path: {str(self.path)}", *updated_calls]

    @property
    def input_tks(self) -> TokenSeq:
        return join_list(
            [self.extra_tks, self.left_tks, self.main_tks, self.right_tks],
            sep=Newline_id,
        )

    def all_ctxs(self) -> dict[str, TokenSeq]:
        return {
            "extra context": self.extra_tks,
            "left context": self.left_tks,
            "right context": self.right_tks,
        }


@dataclass
class AnalysisBasedEditEncoder(EditEncoder[AnalysisBasedTokenizedEdit]):
    n_max_tks: int = 4000
    extra_ctx_names: Sequence[str] = ("usees",)
    collapse_unchanged: bool = True
    record_type_usages: bool = False
    add_truncate_bos: bool = True

    # currently not used
    CtxSepTokens = encode_basic("\n# Usees ends\n")

    def encode_pedit(
        self,
        pedit: ProjectEdit,
        include_additions: bool = False,
    ):
        raise NotImplementedError("Use `encode_pedits` instead.")

    def encode_pedits(
        self,
        pedits: Sequence[ProjectEdit],
        include_additions: bool = False,
    ) -> Iterable[AnalysisBasedTokenizedEdit]:
        analyses = analyze_edits(
            pedits, record_type_usages=self.record_type_usages, silent=True
        )
        # display(UsageAnalysis.TLogger.as_dataframe())
        cst_encoder = CstBasedEditEncoder(
            n_max_tks=self.n_max_tks,
            collapse_unchanged=self.collapse_unchanged,
            add_truncate_bos=self.add_truncate_bos,
        )
        for analysis in analyses:
            pedit = analysis.pedit
            ctx_encoder = CtxEncoder(pedit, self.collapse_unchanged)
            path_to_cxt_edit = {e.path: e for e in analysis.ctx_edits}
            tk_edits = list(
                cst_encoder.encode_pedit(pedit, include_additions=include_additions)
            )
            for edit in tk_edits:
                ctx_edit = path_to_cxt_edit[edit.path]
                ctx_changes = [
                    c
                    for group in self.extra_ctx_names
                    for c in ctx_edit.grouped_ctx_changes[group]
                    if get_change_path(c) not in edit.elems
                ]
                if ctx_changes:
                    extra_ctx_tks = self.maybe_wrap_bos(
                        ctx_encoder.encode_ctx_changes(ctx_changes)
                    )
                else:
                    extra_ctx_tks = TokenSeq()

                main_tks, extra_tks, left_tks, right_tks = truncate_sections(
                    self.n_max_tks,
                    (edit.main_tks, TruncateAt.Right),
                    (extra_ctx_tks, TruncateAt.Left),
                    (edit.left_tks, TruncateAt.Left),
                    (edit.right_tks, TruncateAt.Right),
                    add_bos=self.add_truncate_bos,
                )

                yield AnalysisBasedTokenizedEdit(
                    main_tks=main_tks,
                    left_tks=left_tks,
                    right_tks=right_tks,
                    extra_tks=extra_tks,
                    output_tks=truncate_output_tks(main_tks, edit.output_tks),
                    path=edit.path,
                    change_type=edit.change_type,
                    elems={get_change_path(c) for c in ctx_changes} | edit.elems,
                    updated_calls=ctx_edit.updated_calls,
                )


@dataclass
class CtxEncoder:
    pedit: ProjectEdit
    collapse_unchanged: bool
    indent_in_class: bool = True
    cache: dict[ProjectPath, TokenSeq] = field(default_factory=dict)

    def encode_ctx_element(self, ppath: ProjectPath) -> TokenSeq:
        "Encode a single element in the context. Results are cached."
        if ppath in self.cache:
            return self.cache[ppath]
        pedit = self.pedit
        can_indent = self.indent_in_class
        maybe_dedent = (lambda x: x) if can_indent else dedent
        if (medit := pedit.changes.get(ppath.module)) is None:
            medit = ModuleEdit.from_no_change(pedit.after.modules[ppath.module])
        module_after = medit.after
        path = ppath.path

        if path in medit.all_changes:
            mod = medit.all_changes[path]
            elem = mod.before if isinstance(mod, Deleted) else mod.after
            if (
                self.collapse_unchanged
                and isinstance(mod, Deleted)
                and isinstance(mod.before, PythonFunction)
            ):
                # as a special case, we also collapose the body of deleted functions
                f_code = show_element(
                    collapse_code(mod.before.tree), can_indent and elem.in_class
                )
                elem_tks = change_to_tokens(Deleted(f_code))
            else:
                elem_tks = change_to_tokens(mod.map(lambda e: maybe_dedent(e.code)))
        elif path in module_after.elems_dict:
            elem = module_after.elems_dict[path]
            if self.collapse_unchanged and isinstance(elem, PythonFunction):
                tree = collapse_code(elem.tree)
                code = show_element(tree, can_indent and elem.in_class)
            else:
                code = maybe_dedent(elem.code)
            elem_tks = encode_basic(code)
        else:
            # FIXME: inner classes are pulled out in this implementation
            if (elem := module_after.classes_dict.get(path)) is None:
                elem = pedit.before.modules[ppath.module].classes_dict[path]
            cls = elem.tree.with_changes(body=cst.IndentedBlock([]))
            # drop the `pass` part
            cls_lines = show_element(cls, indent=False).split("\n")[:-1]
            header_tks = encode_basic("\n".join(cls_lines))
            elem_tks = [Newline_id] + header_tks
        if isinstance(elem, PythonFunction):
            elem_tks.append(Newline_id)

        self.cache[ppath] = elem_tks
        return elem_tks

    def encode_ctx_changes(self, changes: Sequence[Change[PythonElem]]):
        # group changes by parents
        parent2change = spot.utils.groupby(changes, lambda c: get_change_path(c).pop())
        sorted_paths = list[ProjectPath]()
        for parent, changes in parent2change.items():
            if parent.path:
                sorted_paths.append(parent)
            for c in changes:
                sorted_paths.append(get_change_path(c))
        return join_list(
            (self.encode_ctx_element(p) for p in sorted_paths),
            sep=Newline_id,
        )


def collapse_code(tree: cst.CSTNode) -> cst.CSTNode:
    class Transformer(cst.CSTTransformer):
        OMIT = cst.SimpleStatementSuite([cst.Expr(cst.Ellipsis())])

        def visit_FunctionDef(self, node) -> Optional[bool]:
            return False

        def leave_FunctionDef(
            self, original_node: "cst.FunctionDef", updated_node: "cst.FunctionDef"
        ):
            return updated_node.with_changes(body=self.OMIT)

    out = tree.visit(Transformer())
    assert isinstance(out, cst.CSTNode)
    return out


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
