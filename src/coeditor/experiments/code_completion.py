import torch

from coeditor.c3problem import C3Problem, C3ToCodeCompletion, SrcInfo
from coeditor.change import Change, Modified
from coeditor.common import *
from coeditor.encoding import (
    Add_id,
    BOS_id,
    Del_id,
    EOS_id,
    Newline_id,
    TkDelta,
    TruncateAt,
    _Tokenizer,
    change_tks_to_original_delta,
    change_to_tokens,
    decode_tokens,
    encode_lines_join,
    encode_single_line,
    get_extra_id,
    inline_output_tokens,
    output_ids_as_seqs,
    tk_splitlines,
    truncate_sections,
)
from coeditor.model import CodeT5Model
from coeditor.scoped_changes import (
    ChangedSpan,
    ChangeScope,
    JProjectChange,
    ProjectChangeProcessor,
)


@dataclass(frozen=True)
class FIMProblem:
    "A fill-in-the-middle problem."
    left_ctx: Sequence[TokenSeq]
    right_ctx: Sequence[TokenSeq]
    middle_tks: TokenSeq
    src_info: SrcInfo
    max_ctx_tks: int

    def to_codet5_format(self) -> tuple[TokenSeq, TokenSeq]:
        left, right = truncate_sections(
            self.max_ctx_tks - 1,
            (join_list(self.left_ctx, Newline_id), TruncateAt.Left),
            (join_list(self.right_ctx, Newline_id), TruncateAt.Right),
            add_bos=True,
        )
        input = left + [Newline_id, get_extra_id(0), Newline_id] + right
        output = [BOS_id, get_extra_id(0)] + self.middle_tks + [EOS_id]
        return input, output


@dataclass
class C3CompletionGenerator(ProjectChangeProcessor[FIMProblem]):
    VERSION = "1.0"
    max_ctx_tks = 2048
    min_target_size: int = C3ToCodeCompletion.min_target_size
    # change spans with more than this many lines will be ignored
    max_span_lines: int = 500
    # change spans with more than this many characters will be ignored
    max_span_chars: int = 6000

    def __post_init__(self):
        self._sampler = C3ToCodeCompletion(self.min_target_size)

    def use_unchanged(self) -> bool:
        return True

    def process_change(
        self,
        pchange: JProjectChange,
        pre_analysis: None,
        post_analysis: None,
    ) -> Sequence[FIMProblem]:
        probs = list[FIMProblem]()
        src_info: SrcInfo = {
            "project": pchange.project_name,
            "commit": pchange.commit_info,
        }
        for m, mchange in pchange.changed.items():
            all_spans = list(mchange.changed)
            all_spans.sort(key=lambda s: s.line_range)
            old_spans, new_spans = self._get_old_new_spans(all_spans)
            for i, span in enumerate(all_spans):
                if not self.should_mk_problem(
                    span,
                    func_only=not self.is_training,
                    max_chars=self.max_span_chars,
                    max_lines=self.max_span_lines,
                ) or code_equal(span.change.earlier, span.change.later):
                    # only keep non-trivial modifications
                    continue
                origin, delta = change_tks_to_original_delta(
                    change_to_tokens(span.change)
                )
                sampled = self._sampler.extract_completion(origin, delta)
                if sampled is None:
                    continue
                new_origin, new_delta = sampled
                left, middle, right = self._split_change(new_origin, new_delta)
                above_spans = [left] if left else []
                # add previous spans until total size exceeds max_ctx_tks
                above_sum = len(left)
                for span in reversed(new_spans[: 2 * i + 1]):
                    if above_sum + len(span) >= self.max_ctx_tks:
                        break
                    above_sum += len(span)
                    above_spans.append(span)
                above_spans.reverse()
                below_spans = [right] if right else []
                below_sum = len(right)
                for span in old_spans[2 * i + 2 :]:
                    # take until below sum exceeds max_ctx_tks
                    if below_sum + len(span) >= self.max_ctx_tks:
                        break
                    below_sum += len(span)
                    below_spans.append(span)
                probs.append(
                    FIMProblem(
                        above_spans, below_spans, middle, src_info, self.max_ctx_tks
                    )
                )
        return probs

    def _get_old_new_spans(
        self, spans: Sequence[ChangedSpan]
    ) -> tuple[list[TokenSeq], list[TokenSeq]]:
        old_spans = list[TokenSeq]()
        new_spans = list[TokenSeq]()
        last_scope = []
        for span in spans:
            scope_diff = list[Change[ChangeScope]]()
            for i, s in enumerate(span.parent_scopes):
                if i >= len(last_scope) or s.later.path != last_scope[i].later.path:
                    scope_diff.append(s)
            old_header = "\n".join(
                s.earlier.header_code.strip("\n") for s in scope_diff
            )
            new_header = "\n".join(s.later.header_code.strip("\n") for s in scope_diff)
            old_spans.append(encode_lines_join(old_header))
            old_spans.append(encode_lines_join(span.change.earlier))
            new_spans.append(encode_lines_join(new_header))
            new_spans.append(encode_lines_join(span.change.later))
            last_scope = span.parent_scopes
        return old_spans, new_spans

    def _split_change(
        self, origin: TokenSeq, delta: TkDelta
    ) -> tuple[TokenSeq, TokenSeq, TokenSeq]:
        assert_eq(delta.num_changes(), 1)
        lines = tk_splitlines(origin)
        key, action = list(delta.items())[0]
        assert_eq(action[0], Add_id, lambda: "delta must be a single addition")
        target_line = key[0]
        left = join_list(
            (
                r
                for l in lines[:target_line]
                if (r := _change_line_to_result(l)) is not None
            ),
            Newline_id,
        )
        right = join_list(lines[target_line:], Newline_id)
        middle = action[1:]
        return left, middle, right


def _change_line_to_result(line: TokenSeq) -> TokenSeq | None:
    if not line:
        return []
    if line[0] == Add_id:
        return line[1:]
    elif line[0] == Del_id:
        return None
    else:
        return line


@dataclass
class InfillResult:
    text: str  # the completed document (with infills inserted)
    infills: Sequence[str]  # the list of infills generated


@dataclass
class CodeT5Wrapper:
    model: CodeT5Model

    def infill(self, parts: Sequence[str], max_to_generate: int = 128) -> InfillResult:
        input = self.input_from_parts(parts)
        out_seq = self.infill_tks(input, max_to_generate)
        print(f"{decode_tokens(out_seq)=}")
        infill_seqs = list(output_ids_as_seqs(out_seq).values())
        infills = [decode_tokens(s) for s in infill_seqs[: len(parts) - 1]]
        print(f"{infills=}")

        inlined = inline_output_tokens(input, out_seq, leave_unpredicted=True)
        text = decode_tokens(inlined)
        return InfillResult(text, infills)

    def input_from_parts(self, parts: Sequence[str]) -> TokenSeq:
        segs = list[str]()
        for i, s in enumerate(parts):
            segs.append(s)
            if i < len(parts) - 1:
                segs.append(f"<extra_id_{i}>")
        return _Tokenizer.encode("".join(segs))

    def infill_tks(self, input: TokenSeq, max_to_generate: int) -> TokenSeq:
        print(f"{decode_tokens(input)=}")
        device = self.model.device
        input_ids = torch.LongTensor([input]).to(device)
        with torch.autocast("cuda"):
            output = self.model.generate(input_ids, max_length=max_to_generate)
        assert isinstance(output, torch.Tensor)
        return output[0].tolist()

    @staticmethod
    def from_pretrained(model_name: str = "Salesforce/codet5-base"):
        model = CodeT5Model.from_pretrained(model_name)
        assert isinstance(model, CodeT5Model)
        return CodeT5Wrapper(model)
