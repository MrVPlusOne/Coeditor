# utils to encode and decode code changes into CodeT5 format.

import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import field
import difflib
import copy
import random

from functools import cache
from spot.static_analysis import StubGenerator, show_element
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


def encode_basic(text: str):
    "Encode a string into a token sequence using the base tokenizer."
    return _BaseTokenizer.encode(text, add_special_tokens=False)


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


def change_to_input_output(change: Modified[str]) -> tuple[TokenSeq, TokenSeq]:
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


@dataclass
class WindowArgs:
    max_window_size: int
    left_ctx_ratio: float = 0.5

    def truncate_ctx(
        self,
        input_tks: TokenSeq,
    ) -> TokenSeq:
        """
        Truncate the input to make it fit within the max window size.
        The cutoff is centered around the <extra_id> tokens.
        """
        extra_id_poses = [i for i, tk in enumerate(input_tks) if is_extra_id(tk)]
        assert extra_id_poses
        assert 0 <= self.left_ctx_ratio <= 1
        main_left = extra_id_poses[0]
        main_end = min(extra_id_poses[-1] + 1, main_left + self.max_window_size)
        main_size = main_end - main_left
        assert main_size >= 0

        if main_size > self.max_window_size:
            # the main part is too long, truncate it
            left_ctx_start = main_left
            right_ctx_end = left_ctx_start + self.max_window_size
        else:
            left_size = int((self.max_window_size - main_size) * self.left_ctx_ratio)
            right_size = self.max_window_size - main_size - left_size
            assert right_size >= 0, f"right_size: {right_size}"

            right_ctx_end = min(len(input_tks), main_end + right_size)
            right_size = right_ctx_end - main_end
            assert right_size >= 0, f"right_size: {right_size}"

            # if right_size doesn't use up all the space, we can expand the left context
            left_size = self.max_window_size - main_size - right_size
            left_ctx_start = max(0, extra_id_poses[0] - left_size)
            assert left_size >= 0

        new_input = input_tks[left_ctx_start:right_ctx_end]
        if left_ctx_start > 0:
            new_input[0] = BOS_id
        if right_ctx_end < len(input_tks):
            new_input[-1] = EOS_id
        assert len(new_input) <= self.max_window_size
        return new_input

    @staticmethod
    def Default() -> "WindowArgs":
        return WindowArgs(4096, 0.5)


@dataclass
class TokenizedEdit:
    input_tks: TokenSeq
    output_tks: TokenSeq
    path: ProjectPath

    def truncate_ctx(
        self,
        args: WindowArgs,
    ) -> "TokenizedEdit":
        return TokenizedEdit(
            args.truncate_ctx(self.input_tks),
            self.output_tks,
            path=self.path,
        )

    def as_change(self, main_code_only: bool = False) -> Modified[str]:
        input_tks = self.get_main_tks() if main_code_only else self.input_tks
        inlined = inline_output_tokens(input_tks, self.output_tks)
        return tokens_to_change(inlined)

    def get_main_tks(self) -> TokenSeq:
        lines = split_list(self.input_tks, Newline_id)
        main_lines = list[TokenSeq]()
        for line in lines:
            if line and is_extra_id(line[0]):
                main_lines.append(line)
        return join_list(main_lines, Newline_id)

    def show(self, ctx_tks: int = 2048) -> str:
        return self.show_prediction(None, ctx_tks)

    def show_prediction(self, pred_tks: TokenSeq | None = None, ctx_tks: int = 1000):
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
                lines.append(
                    f"{show_label(id_map[k])}:{indent(decode_tokens(seg), ' ' * 4).lstrip()}"
                )
            return "\n".join(lines)

        extra_id_poses = [i for i, tk in enumerate(self.input_tks) if is_extra_id(tk)]
        l, r = extra_id_poses[0], extra_id_poses[-1]
        left_ctx = self.input_tks[max(0, l - ctx_tks) : l]
        right_ctx = self.input_tks[r + 1 : min(len(self.input_tks), r + ctx_tks)]
        main_tks = self.input_tks[l : r + 1]

        main_lines = output_ids_as_seqs(main_tks)
        id_map = {k: i for i, k in enumerate(main_lines)}
        new_main_tks = []
        for tk in main_tks:
            if is_extra_id(tk):
                new_main_tks.extend(encode_basic(show_label(id_map[tk])))
            else:
                new_main_tks.append(tk)

        pred_lines = (
            ["========Prediction========", f"{show_extra_tokens(pred_tks, main_lines)}"]
            if pred_tks
            else []
        )
        outputs = [
            f"========Left Context========",
            decode_tokens(left_ctx),
            "========Ground Truth========",
            show_extra_tokens(self.output_tks, main_lines),
            *pred_lines,
            "========Main Code========",
            decode_tokens(new_main_tks),
            "========Right Context========",
            decode_tokens(right_ctx),
        ]
        return "\n".join(outputs)


@dataclass
class FileBasedEditEncoder:
    window: WindowArgs

    def encode_pedit(
        self,
        pedit: ProjectEdit,
    ) -> Iterable[TokenizedEdit]:
        for me in pedit.changes.values():
            yield from self.encode_medit(me)

    def encode_medit(
        self,
        medit: ModuleEdit,
    ) -> Iterable[TokenizedEdit]:
        modifications = medit.modified_functions()
        if not modifications:
            return
        mod_before = medit.before
        code_before = mod_before.code.split("\n")
        mod_after = medit.after
        code_after = mod_after.code.split("\n")
        mod_name = mod_after.name
        ctx_lines = self.window.max_window_size // 2

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
                Modified(code_main_before, code_main_after)
            )

            ex = TokenizedEdit([], [], path=ProjectPath(mod_name, path))
            ex.input_tks.extend(above_tks)
            ex.input_tks.append(Newline_id)
            ex.input_tks.extend(input)
            ex.input_tks.append(Newline_id)
            ex.output_tks = output
            ex.input_tks.extend(below_tks)
            ex = ex.truncate_ctx(self.window)
            yield ex


@dataclass
class CstBasedEditEncoder:
    window: WindowArgs
    collapse_unchanged: bool = True

    def encode_pedit(
        self,
        pedit: ProjectEdit,
    ) -> Iterable[TokenizedEdit]:
        prompt = encode_basic("\n# EDIT:\n")
        ctx_encoder = CtxEncoder(pedit, self.collapse_unchanged)
        for mname, medit in pedit.changes.items():
            mod_fs = medit.modified_functions()
            if not mod_fs:
                continue

            sorted_elems = [
                ProjectPath(mname, p) for p in medit.sorted_elems(include_classes=True)
            ]

            for i, path in enumerate(sorted_elems):
                if (c := mod_fs.get(path.path)) is None:
                    continue
                main_change = c.map(lambda x: x.code)
                main_tks, output_tks = change_to_input_output(main_change)
                left_ctx = join_list(
                    (ctx_encoder.encode_ctx_element(p) for p in sorted_elems[:i]),
                    sep=Newline_id,
                )
                right_ctx = join_list(
                    (ctx_encoder.encode_ctx_element(p) for p in sorted_elems[i + 1 :]),
                    sep=Newline_id,
                )
                input_tks = left_ctx + prompt + main_tks + [Newline_id] + right_ctx
                ex = TokenizedEdit(input_tks, output_tks, path=path)
                ex = ex.truncate_ctx(self.window)
                yield ex


@dataclass
class AnalysisBasedEditEncoder:
    window: WindowArgs
    extra_ctx_size: int
    extra_ctx_names: Sequence[str] = ("usees",)
    collapse_unchanged: bool = True

    CtxSepTokens = encode_basic("\n# Usees ends\n")

    def encode_pedits(
        self,
        pedits: Sequence[ProjectEdit],
    ) -> Iterable[TokenizedEdit]:
        anlyses = analyze_edits(pedits, silent=True)
        # display(UsageAnalysis.TLogger.as_dataframe())
        cst_encoder = CstBasedEditEncoder(self.window, self.collapse_unchanged)
        for analysis in anlyses:
            pedit = analysis.pedit
            ctx_encoder = CtxEncoder(pedit, self.collapse_unchanged)
            path_to_cxt_edit = {e.path: e for e in analysis.ctx_edits}
            tk_edits = list(cst_encoder.encode_pedit(pedit))
            for edit in tk_edits:
                module = edit.path.module
                ctx_edit = path_to_cxt_edit[edit.path]
                ctx_changes = [
                    c
                    for group in self.extra_ctx_names
                    for c in ctx_edit.grouped_ctx_changes[group]
                    if get_change_path(c).module != module
                ]
                if ctx_changes:
                    extra_ctx_tks = ctx_encoder.encode_ctx_changes(ctx_changes)
                    max_ctx_size = max(
                        self.extra_ctx_size,
                        self.window.max_window_size - len(edit.input_tks),
                    )
                    if len(extra_ctx_tks) > max_ctx_size:
                        extra_ctx_tks = extra_ctx_tks[:max_ctx_size]
                        extra_ctx_tks[-1] = EOS_id
                    inner_window = copy.deepcopy(self.window)
                    inner_window.max_window_size -= len(extra_ctx_tks)
                    edit = edit.truncate_ctx(inner_window)
                    edit.input_tks = extra_ctx_tks + self.CtxSepTokens + edit.input_tks
                yield edit


@dataclass
class CtxEncoder:
    pedit: ProjectEdit
    collapse_unchanged: bool
    cache: dict[ProjectPath, TokenSeq] = field(default_factory=dict)

    def encode_ctx_element(self, ppath: ProjectPath) -> TokenSeq:
        "Encode a single element in the context. Results are cached."
        if ppath in self.cache:
            return self.cache[ppath]
        pedit = self.pedit
        if (medit := pedit.changes.get(ppath.module)) is None:
            medit = ModuleEdit.from_no_change(pedit.after.modules[ppath.module])
        module_after = medit.after
        path = ppath.path

        if path in medit.all_changes:
            mod = medit.all_changes[path]
            elem = mod.before if isinstance(mod, Deleted) else mod.after
            elem_tks = change_to_tokens(mod.map(lambda e: e.code))
        elif path in module_after.elems_dict:
            elem = module_after.elems_dict[path]
            if self.collapse_unchanged and isinstance(elem, PythonFunction):
                tree = collapse_code(elem.tree)
                code = show_element(tree, elem.in_class)
            else:
                code = elem.code
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
