import json

import tokenizers
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.xglm.modeling_xglm import XGLMForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from coeditor.common import *

from .code_completion import InfillResult

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
class InCoderWrapper:
    model: InCoderModelType
    tokenizer: InCoderTokenizerType

    def infill(
        self,
        parts: Sequence[str],
        max_to_generate: int = 128,
        temperature: float = 0.2,
        extra_sentinel: bool = True,
        max_retries: int = 1,
        VERBOSE: bool = False,
    ):
        """
        Generate infills to complete a partial document, e.g.
        [A C E] -> [A B C D E], where B and D are infills that have been generated.
        parts: List[str]. list of parts of the document. One string will be
                inserted in between each element, i.e. infilling N-1 locations for a list
                of length N.
        max_to_generate: int. maximum number of tokens to generate. Keep in mind
                that the model context size is 2048.
        temperature: float. temperature parameter for sampling.
        extra_sentinel: bool. we recommend setting this to True, as it makes it
                easier for the model to end generated infills. See the footnote in
                section 2.2 of our paper for details.
        max_retries: int. if > 1, use rejection sampling to keep sampling infills until
                all infills sample a completion token.
        returns a dictionary containing the following:
            text:  str, the completed document (with infills inserted)
            parts:  List[str], length N. Same as passed to the method
            infills:  List[str], length N-1. The list of infills generated
            retries_attempted:  number of retries used (if max_retries > 1)
        """
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

            infills = []
            complete = []

            done = True

            ## (2) generate infills
            for sentinel_ix, part in enumerate(parts[:-1]):
                complete.append(part)
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
                complete.append(infilled)
                prompt += completion
            complete.append(parts[-1])
            text = "".join(complete)

        return InfillResult(
            text,
            infills,
        )

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
        model_name: str = "facebook/incoder-1B", half_precision: bool = False
    ):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        assert isinstance(model, InCoderModelType)
        assert isinstance(tokenizer, InCoderTokenizerType)
        if half_precision:
            model = model.half()
        return InCoderWrapper(model, tokenizer)
