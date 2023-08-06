import json

import tokenizers
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.xglm.modeling_xglm import XGLMForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from coeditor.common import *
from coeditor.encoding import TruncateAt, truncate_sections
from coeditor.experiments.code_completion import FIMModel

InCoderModelType = XGLMForCausalLM
InCoderTokenizerType = PreTrainedTokenizerFast

# signals the start of a document
BOS = "<|endoftext|>"
# signals the end of a generated infill
EOM = "<|endofmask|>"


def make_sentinel(i) -> str:
    # signals (1) a location to insert an infill and (2) the start of the infill generation
    return f"<|mask:{i}|>"


@dataclass
class InCoderWrapper(FIMModel):
    model: InCoderModelType
    tokenizer: InCoderTokenizerType
    tks_limit: int = 2048

    def __post_init__(self):
        self.bos_ids = self.tokenizer.encode(BOS, add_special_tokens=False)
        self.mask0_ids = self.tokenizer.encode(
            make_sentinel(0), add_special_tokens=False
        )
        self.mask1_ids = self.tokenizer.encode(
            make_sentinel(1), add_special_tokens=False
        )

    def infill(self, left: str, right: str, max_output: int) -> str:
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
            [
                self.bos_ids,
                left_tks,
                self.mask0_ids,
                right_tks,
                self.mask1_ids,
                self.mask0_ids,
            ]
        )
        total_length = len(input_ids) + max_output
        if total_length > self.tks_limit:
            warnings.warn(
                f"Total length too large: {total_length=} (> {self.tks_limit})"
            )
        input_ids = torch.LongTensor([input_ids]).to(device)
        output = self.model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_length=total_length,
        )
        assert isinstance(output, Tensor)
        output_ids = output[0].tolist()
        output_ids = output_ids[input_ids.size(1) :]
        completion: str = tkn.decode(output_ids, clean_up_tokenization_spaces=False)

        if EOM not in completion:
            completion += EOM
        completion = completion[: completion.index(EOM) + len(EOM)]
        infilled = completion[: -len(EOM)]
        return infilled

    def infill_multi(
        self,
        parts: Sequence[str],
        max_to_generate: int = 128,
        temperature: float = 0.2,
        extra_sentinel: bool = True,
        max_retries: int = 1,
        VERBOSE: bool = False,
    ):
        retries_attempted = 0
        done = False
        prompt = text = ""
        infills = []

        while (not done) and (retries_attempted < max_retries):
            retries_attempted += 1

            if VERBOSE:
                print(f"retry {retries_attempted}")

            ## (1) build the prompt
            if len(parts) == 1:
                prompt = parts[0]
            else:
                prompt = ""
                # encode parts separated by sentinel
                for sentinel_ix, part in enumerate(parts):
                    prompt += part
                    if extra_sentinel or (sentinel_ix < len(parts) - 1):
                        prompt += make_sentinel(sentinel_ix)

            infills = list[str]()

            done = True

            ## (2) generate infills
            for sentinel_ix, part in enumerate(parts[:-1]):
                prompt += make_sentinel(sentinel_ix)
                # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
                completion = self.generate(prompt, max_to_generate, temperature)
                completion = completion[len(prompt) :]
                if EOM not in completion:
                    if VERBOSE:
                        print(f"warning: {EOM} not found")
                    completion += EOM
                    done = False
                completion = completion[: completion.index(EOM) + len(EOM)]
                infilled = completion[: -len(EOM)]
                infills.append(infilled)
                prompt += completion

        return infills

    def generate(
        self, input: str, max_to_generate: int = 128, temperature: float = 0.2
    ):
        """
        Do standard left-to-right completion of the prefix `input` by sampling from the model
        """
        tkn = self.tokenizer
        device = self.model.device
        input_ids: Tensor = tkn(input, return_tensors="pt").input_ids.to(device)
        max_length = max_to_generate + input_ids.flatten().size(0)
        if max_length > 2048:
            print(
                "warning: max_length {} is greater than the context window {}".format(
                    max_length, 2048
                )
            )
        output = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=0.95,
            temperature=temperature,
            max_length=max_length,
        )
        assert isinstance(output, Tensor)
        # pass clean_up_tokenization_spaces=False to avoid removing spaces before punctuation, e.g. "from ." -> "from."
        detok_hypo_str = tkn.decode(
            output.flatten(), clean_up_tokenization_spaces=False
        )
        if detok_hypo_str.startswith(BOS):
            detok_hypo_str = detok_hypo_str[len(BOS) :]
        return detok_hypo_str

    @staticmethod
    def from_pretrained(
        model_name: str = "facebook/incoder-1B", half_precision: bool = True
    ):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert isinstance(model, InCoderModelType)
        assert isinstance(tokenizer, InCoderTokenizerType)
        if half_precision:
            model = model.half()
        return InCoderWrapper(model, tokenizer)
