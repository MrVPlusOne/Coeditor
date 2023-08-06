import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from coeditor.c3problem import (
    C3ProblemGenerator,
    C3ToCodeCompletion,
    CompletionKind,
    SrcInfo,
    TkC3Problem,
)
from coeditor.change import Change
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
    get_extra_id,
    output_ids_as_seqs,
    tk_splitlines,
    truncate_sections,
)
from coeditor.model import C3DataLoader, CodeT5Model, RetrievalEditorModel
from coeditor.scoped_changes import (
    ChangedSpan,
    ChangeScope,
    JProjectChange,
    ProjectChangeProcessor,
)

CodeT5TKN = _Tokenizer


@dataclass(frozen=True)
class FIMProblem:
    "A fill-in-the-middle problem."
    left_ctx: Sequence[str]
    right_ctx: Sequence[str]
    middle: str
    src_info: SrcInfo
    max_ctx_tks: int
    kind: CompletionKind
    path: ProjectPath

    def uid(self) -> tuple[ProjectPath, str]:
        return self.path, not_none(self.src_info["commit"]).hash

    def get_contexts(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tks_limit: int = 2040,
        max_output: int = 256,
    ) -> tuple[str, str]:
        """Get the left and right contexts, truncated to the given token limit.
        This is mostly for visualization as the FIM model handles context truncation
        internally (in a more efficient way)."""
        left_tks: TokenSeq = tokenizer.encode(
            "\n".join(self.left_ctx) + "\n", add_special_tokens=False
        )
        right_tks: TokenSeq = tokenizer.encode(
            "\n" + "".join(self.right_ctx), add_special_tokens=False
        )
        left_tks, right_tks = truncate_sections(
            tks_limit - max_output,
            (left_tks, TruncateAt.Left),
            (right_tks, TruncateAt.Right),
            add_bos=False,
        )
        left = tokenizer.decode(left_tks, clean_up_tokenization_spaces=False)
        right = tokenizer.decode(right_tks, clean_up_tokenization_spaces=False)
        return left, right


@dataclass
class C3CompletionGenerator(ProjectChangeProcessor[FIMProblem]):
    """
    Extract fiil-in-the-middle problems from code changes.

    ## Arguments
    - `addition_only`: whether to only extract from problems where the last change is
    a pure additon (rather than a replacement). `addition_only` problems are easier
    for code completion models since they don't see any code that get deleted.

    ## Change log
    - version 1.1: Limit context str length to `10 * max_ctx_tks`.
    - version 1.2: Add `path` attribute to `FIMProblem`.
    """

    VERSION = "1.2"
    max_ctx_tks: int = 2048
    min_target_size: int = C3ToCodeCompletion.min_target_size
    use_additions: bool = C3ToCodeCompletion.use_additions
    use_modifications: bool = C3ToCodeCompletion.use_modifications
    generator: C3ProblemGenerator = field(default_factory=C3ProblemGenerator)

    def __post_init__(self):
        self._sampler = C3ToCodeCompletion(
            self.min_target_size,
            use_additions=self.use_additions,
            use_modifications=self.use_modifications,
        )

    def use_unchanged(self) -> bool:
        return True

    def post_edit_analysis(self, *args, **kwargs) -> list[ModuleName]:
        return self.generator.post_edit_analysis(*args, **kwargs)

    def process_change(
        self,
        pchange: JProjectChange,
        pre_analysis: None,
        post_analysis: Sequence[ModuleName],
    ) -> Sequence[FIMProblem]:
        probs = list[FIMProblem]()
        src_info: SrcInfo = {
            "project": pchange.project_name,
            "commit": pchange.commit_info,
        }
        for m in post_analysis:
            if (mchange := pchange.changed.get(m)) is None:
                continue
            all_spans = list(mchange.changed)
            all_spans.sort(key=lambda s: s.line_range)
            old_spans, new_spans = self._get_old_new_spans(all_spans)
            for i, span in enumerate(all_spans):
                if not self.should_mk_problem(
                    span,
                    func_only=not self.is_training,
                    max_chars=self.generator.max_span_chars,
                    max_lines=self.generator.max_span_lines,
                ) or code_equal(span.change.earlier, span.change.later):
                    # only keep non-trivial modifications
                    continue
                origin, delta = change_tks_to_original_delta(
                    change_to_tokens(span.change)
                )
                sampled = self._sampler.extract_completion(origin, delta)
                if sampled is None:
                    continue
                new_origin, new_delta, kind = sampled
                left, middle, right = self._split_change(new_origin, new_delta)
                above_spans = [left] if left else []
                # add previous spans until total size exceeds max_ctx_tks
                above_sum = len(left)
                for s in reversed(new_spans[: 2 * i + 1]):
                    if above_sum + len(s) >= self.max_ctx_tks * 6:
                        break
                    above_sum += len(s)
                    above_spans.append(s)
                above_spans.reverse()
                below_spans = [right] if right else []
                below_sum = len(right)
                for s in old_spans[2 * i + 2 :]:
                    # take until below sum exceeds max_ctx_tks
                    if below_sum + len(s) >= self.max_ctx_tks * 6:
                        break
                    below_sum += len(s)
                    below_spans.append(s)
                probs.append(
                    FIMProblem(
                        above_spans,
                        below_spans,
                        middle,
                        src_info,
                        self.max_ctx_tks,
                        kind,
                        path=span.scope.later.path,
                    )
                )
        return probs

    def _get_old_new_spans(
        self, spans: Sequence[ChangedSpan]
    ) -> tuple[list[str], list[str]]:
        old_spans = list[str]()
        new_spans = list[str]()
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
            old_spans.append(old_header)
            old_spans.append(span.change.earlier)
            new_spans.append(new_header)
            new_spans.append(span.change.later)
            last_scope = span.parent_scopes
        return old_spans, new_spans

    def _split_change(self, origin: TokenSeq, delta: TkDelta) -> tuple[str, str, str]:
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
        return tuple(decode_tokens(x) for x in (left, middle, right))


def _change_line_to_result(line: TokenSeq) -> TokenSeq | None:
    if not line:
        return []
    if line[0] == Add_id:
        return line[1:]
    elif line[0] == Del_id:
        return None
    else:
        return line


class FIMModel(ABC):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    @abstractmethod
    def infill(self, left: str, right: str, max_output: int) -> str:
        ...


@dataclass
class CodeT5Wrapper(FIMModel):
    model: CodeT5Model
    tks_limit: int = 2048
    tokenizer = CodeT5TKN

    def infill(self, left: str, right: str, max_output: int = 128) -> str:
        tkn = self.tokenizer
        device = self.model.device
        left_tks: TokenSeq = tkn.encode(left, add_special_tokens=False)
        right_tks: TokenSeq = tkn.encode(right, add_special_tokens=False)
        left_tks, right_tks = truncate_sections(
            self.tks_limit - max_output - 8,
            (left_tks, TruncateAt.Left),
            (right_tks, TruncateAt.Right),
            add_bos=False,
        )
        input_ids = join_list(
            [[BOS_id], left_tks, [get_extra_id(0)], right_tks, [EOS_id]]
        )
        input_ids = torch.LongTensor([input_ids]).to(device)
        output_ids = self.model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_length=max_output,
        )
        assert isinstance(output_ids, torch.Tensor)
        output_ids = output_ids[0].tolist()
        infill_ids = output_ids_as_seqs(output_ids)[get_extra_id(0)]
        return decode_tokens(infill_ids)

    @staticmethod
    def from_pretrained(model_name: str = "Salesforce/codet5-base"):
        model = CodeT5Model.from_pretrained(model_name)
        assert isinstance(model, CodeT5Model)
        return CodeT5Wrapper(model)


_infill_prefix = _Tokenizer.encode("<s><extra_id_0><add>", add_special_tokens=False)


def infill_with_coeditor(
    coeditor: RetrievalEditorModel, tk_prob: TkC3Problem, max_length: int = 128
) -> TokenSeq:
    """Run the Coeditor model on the (inifilling version) C3 Problem, return the
    model output."""

    device = coeditor.device
    batch = C3DataLoader.pack_batch([tk_prob])
    # the prefix is always an addition
    prefix_allowed_tokens_fn = RetrievalEditorModel._prefix_constraint([_infill_prefix])
    input_ids = torch.LongTensor(batch["input_ids"]).to(device)
    output_ids = coeditor.generate(
        input_ids=input_ids,
        references=batch["references"],
        query_ref_list=batch["query_ref_list"],
        do_sample=False,
        max_length=max_length,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )
    assert isinstance(output_ids, torch.Tensor)
    output_ids = output_ids[0].tolist()
    return output_ids
