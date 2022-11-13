from .common import *
from .encoding import _Tokenizer, Encoding
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

CodeT5Model = T5ForConditionalGeneration


class CoeditorModel:
    codet5: CodeT5Model
