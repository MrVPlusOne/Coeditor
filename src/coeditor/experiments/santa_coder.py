import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

from coeditor.common import *
from coeditor.encoding import TruncateAt, truncate_sections
from coeditor.experiments.code_completion import FIMModel

SantaCoderModelType = GPT2PreTrainedModel
SantaCoderTokenizerType = GPT2TokenizerFast


@dataclass
class SantaCoderWrapper(FIMModel):
    model: SantaCoderModelType
    tokenizer: SantaCoderTokenizerType
    tks_limit: int = 2048

    def __post_init__(self):
        added = self.tokenizer.get_added_vocab()
        self.endoftext = self.tokenizer.encode("<|endoftext|>")[0]
        self.fim_prefix = added["<fim-prefix>"]
        self.fim_middle = added["<fim-middle>"]
        self.fim_suffix = added["<fim-suffix>"]

    def infill(self, left: str, right: str, max_length: int) -> str:
        tkn = self.tokenizer
        device = self.model.device
        left_tks: TokenSeq = tkn.encode(left, add_special_tokens=False)
        right_tks: TokenSeq = tkn.encode(right, add_special_tokens=False)
        left_tks, right_tks = truncate_sections(
            self.tks_limit - max_length - 4,
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
        total_length = len(input_ids) + max_length
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
    def from_pretrained(model_name: str = "bigcode/santacoder"):
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        assert isinstance(model, SantaCoderModelType)
        assert isinstance(tokenizer, SantaCoderTokenizerType)
        return SantaCoderWrapper(model, tokenizer)
