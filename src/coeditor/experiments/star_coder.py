import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM

from coeditor.common import *
from coeditor.encoding import TruncateAt, truncate_sections
from coeditor.experiments.code_completion import FIMModel

SantaCoderModelType = GPTBigCodeForCausalLM
SantaCoderTokenizerType = GPT2TokenizerFast


@dataclass
class StarCoderWrapper(FIMModel):
    model: SantaCoderModelType
    tokenizer: SantaCoderTokenizerType
    tks_limit: int = 1024 * 8

    def __post_init__(self):
        vocab = self.tokenizer.vocab
        self.endoftext = vocab["<|endoftext|>"]
        self.fim_prefix = vocab["<fim_prefix>"]
        self.fim_middle = vocab["<fim_middle>"]
        self.fim_suffix = vocab["<fim_suffix>"]

    def infill(self, left: str, right: str, max_output: int) -> str:
        tkn = self.tokenizer
        device = self.model.device
        left_tks: TokenSeq = tkn.encode(left, add_special_tokens=False)
        right_tks: TokenSeq = tkn.encode(right, add_special_tokens=False)
        left_tks, right_tks = truncate_sections(
            self.tks_limit - max_output - 4,
            (left_tks, TruncateAt.Left),
            (right_tks, TruncateAt.Right),
            add_bos=False,
        )

        input_ids = join_list(
            [
                [self.fim_prefix],
                left_tks,
                [self.fim_suffix],
                right_tks,
                [self.fim_middle],
            ]
        )
        total_length = len(input_ids) + max_output
        if total_length > self.tks_limit:
            warnings.warn(
                f"Total length {total_length} exceeds the limit of {self.tks_limit}."
            )
        input_ids = torch.tensor([input_ids], device=device)
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=None,
            do_sample=False,
            max_length=total_length,
            eos_token_id=self.endoftext,
            pad_token_id=self.endoftext,
        )
        assert isinstance(output, torch.Tensor)
        output_ids = output[0].tolist()
        output_ids = output_ids[input_ids.size(1) :]
        if output_ids[-1] == self.endoftext:
            output_ids = output_ids[:-1]
        completion: str = tkn.decode(output_ids, clean_up_tokenization_spaces=False)

        return completion

    @staticmethod
    def from_pretrained(
        model_name: str = "bigcode/starcoderbase-7b", half_precision: bool = True
    ):
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        assert isinstance(model, SantaCoderModelType)
        assert isinstance(tokenizer, SantaCoderTokenizerType)
        if half_precision:
            model = model.half()
        return StarCoderWrapper(model, tokenizer)
