import copy
from .common import *
from transformers.models.t5.modeling_t5 import (
    T5Config,
    T5Stack,
    T5ForConditionalGeneration,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    T5PreTrainedModel,
)
import torch
from torch import FloatTensor, LongTensor, Tensor
from torch import nn
from coeditor.encoding import BOS_id, EOS_id


class RetrievalEditorModel(T5PreTrainedModel):
    """
    A CodeT5 model that takes in multiple reference code snippets and a
    query code snippet with multiple masked spans and perdicts the maksed spans.

    While the computational cost of a normal CodeT5 encoder increases quadratically,
    this model only increases linearly with the number of reference code snippets.
    """

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        pad_id = self.config.pad_token_id
        assert isinstance(pad_id, int)
        self.pad_id = pad_id

        # Model parallel
        # self.model_parallel = False
        # self.device_map = None

    def encode_token_seqs(self, references: list[TokenSeq]) -> LongTensor:
        max_len = max(len(ref) for ref in references)
        rows = []
        for ref in references:
            row = ref + [self.pad_id] * (max_len - len(ref))
            rows.append(row)
        out = LongTensor(rows).to(self.device)
        return cast(LongTensor, out)

    def forward(
        self,
        # encoder args
        input_ids: LongTensor | None = None,  # queries
        references: LongTensor | None = None,
        ref_masks: list[list[int]] | None = None,
        labels: LongTensor | None = None,
        # decoder args
        encoder_outputs: "RetrivalEncoderOutputs | None" = None,
        decoder_input_ids: LongTensor | None = None,
        decoder_inputs_embeds: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        past_key_values=None,
        use_cache=None,
        # not used args below
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Seq2SeqLMOutput:
        """
        Shapes
        - input_ids: (num_queries, seq_len,)
        - references: (num_refs, ref_len)
        - ref_masks: for each query, a list of reference indices. If none,
        assume all references are accessible to all queries.
        """
        if labels is not None:
            assert_eq(labels.dim(), 2)

        if encoder_outputs is None:
            assert input_ids is not None
            encoder = self.get_encoder()
            encoder_outputs = encoder.forward(input_ids, references, ref_masks)

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = cast(LongTensor, self._shift_right(labels))

        decoder_outputs = self.decoder.forward(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_outputs.hidden_state_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        assert isinstance(decoder_outputs, BaseModelOutputWithPastAndCrossAttentions)

        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=cast(Any, encoder_outputs.last_hidden_state),
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        references,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "references": references,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            # "attention_mask": attention_mask,
            # "head_mask": head_mask,
            # "decoder_head_mask": decoder_head_mask,
            # "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return RetrivalEncoder(self.encoder, self.pad_id)

    def get_decoder(self):
        return self.decoder

    @staticmethod
    def from_code_t5(size: Literal["small", "base", "large"]):
        model = RetrievalEditorModel.from_pretrained(f"Salesforce/codet5-{size}")
        assert isinstance(model, RetrievalEditorModel)
        return model


@dataclass
class RetrivalEncoderOutputs:
    last_hidden_state: Tensor
    hidden_state_mask: Tensor


@dataclass
class RetrivalEncoder:
    encoder: T5Stack
    pad_id: int

    def __call__(self, *args: Any, **kwds: Any) -> RetrivalEncoderOutputs:
        return self.forward(*args, **kwds)

    def forward(
        self,
        input_ids: LongTensor,
        references: LongTensor | None = None,
        ref_masks: list[list[int]] | None = None,
        # not used arguments below:
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> RetrivalEncoderOutputs:
        """
        Shapes
        - input_ids: (num_queries, seq_len,)
        - references: (num_refs, ref_len)
        - ref_masks: for each query, a list of reference indices. If none,
        assume all references are accessible to all queries.
        """
        if references is None:
            references = cast(LongTensor, LongTensor([[self.pad_id]]).to(input_ids.device))

        assert_eq(input_ids.dim(), 2)
        assert_eq(references.dim(), 2)

        n_queries = input_ids.size(0)
        query_attention_mask = cast(LongTensor, input_ids.ne(self.pad_id).long())
        query_lens = query_attention_mask.sum(dim=1)

        n_refs = references.size(0)
        ref_attention_mask = cast(LongTensor, references.ne(self.pad_id).long())
        ref_lens = ref_attention_mask.sum(dim=1)

        ref_outputs = self.encode_references(
            references, attention_mask=ref_attention_mask
        )
        # (n_refs, ref_len, model_dim)
        ref_states = ref_outputs.last_hidden_state

        # todo: make the refs available for query's attention
        query_outputs = self.encode_references(
            input_ids, attention_mask=query_attention_mask
        )
        # (n_queries, query_len, model_dim)
        query_states = query_outputs.last_hidden_state

        ref_list = [ref_states[i][: ref_lens[i]] for i in range(n_refs)]
        query_list = [query_states[i][: query_lens[i]] for i in range(n_queries)]

        if ref_masks is None:
            ref_masks = [list(range(n_refs)) for _ in range(n_queries)]

        state_rows = []
        for i, query in enumerate(query_list):
            row = [ref_list[j] for j in ref_masks[i]]
            row.append(query)
            # each row of shape (seq_len, model_dim)
            state_rows.append(torch.cat(row, dim=0))

        # (n_queries, seq_len, model_dim)
        hidden_states = nn.utils.rnn.pad_sequence(state_rows, batch_first=False)
        seq_len = hidden_states.size(1)
        state_mask = [
            [1] * row.size(0) + [0] * (seq_len - row.size(0)) for row in state_rows
        ]
        state_mask = torch.BoolTensor(state_mask).to(input_ids.device)
        return RetrivalEncoderOutputs(
            last_hidden_state=hidden_states, hidden_state_mask=state_mask
        )

    def encode_references(
        self,
        input_ids: LongTensor,
        attention_mask: LongTensor | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """input_ids: shape (num_refs, seq_len)"""
        out = self.encoder.forward(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        assert isinstance(out, BaseModelOutputWithPastAndCrossAttentions)
        return out
