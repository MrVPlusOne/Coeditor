import copy

from coeditor.dataset import TokenizedEditDataset
from coeditor.encoders import BasicTkQueryEdit
from coeditor.history import Modified
from coeditor.model import (
    DecodingArgs,
    EvalArgs,
    TrainingArgs,
    compute_loss_metrics,
    wrap_bos,
)
from spot.static_analysis import ProjectPath
from spot.utils import cprint, groupby, scalar_stats

from .common import *
from transformers.models.t5.modeling_t5 import (
    T5Config,
    T5Attention,
    T5LayerNorm,
    T5Block,
    T5Stack,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5LayerFF,
    T5ForConditionalGeneration,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    T5PreTrainedModel,
)
import torch
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from torch import nn
from coeditor.encoding import (
    Add_id,
    BOS_id,
    Del_id,
    EOS_id,
    decode_tokens,
    encode_basic,
    get_tk_id,
    random_extra_id_map,
    _Tokenizer,
)
import transformers
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    EvalPrediction,
    BatchEncoding,
)
from transformers.trainer import EvalLoopOutput
from datasets.arrow_dataset import Dataset


PAD_id = 0
CheckNaN: bool = False


def check_nan(name: str, x: Tensor, inputs: dict):
    if CheckNaN and torch.isnan(x).any():
        print(f"NaN found in {name}")
        print("Found value:", x)
        for k, v in inputs.items():
            print(k, "=", v)
        raise Exception(f"NaN found in {name}")


class RetrievalEditorModel(T5PreTrainedModel):
    is_parallelizable = False
    supports_gradient_checkpointing = False

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

        self.query_attened_ref = True

    def train_on_data(
        self,
        training_name: str,
        train_data: TokenizedEditDataset,
        eval_data: TokenizedEditDataset,
        train_args: "TrainingArgs",
        batch_args: "BatchArgs",
    ) -> None:
        train_dir = get_model_dir(trained=False) / training_name

        train_edits = train_data.all_edits()
        eval_edits = eval_data.all_edits()
        assert len(train_edits) > 0, "No training edits provided."

        train_lodader = edits_to_dataloader(
            train_edits, args=batch_args, shuffle=True, desc="Training Epoch"
        )
        eval_batch_args = copy.deepcopy(batch_args)
        eval_batch_args.max_queries *= 2
        eval_batch_args.min_queires *= 2
        eval_batch_args.shuffle_extra_ids = False
        eval_batch_args.max_ref_dropout = 0.0
        eval_loader = edits_to_dataloader(
            eval_edits,
            args=eval_batch_args,
            shuffle=False,
            desc="Eval Epoch",
        )

        model = self

        class DynamicTrainer(Seq2SeqTrainer):
            def get_train_dataloader(self):
                return train_lodader

            def get_eval_dataloader(self, eval_dataset):
                return eval_loader

            def evaluation_loop(
                self,
                dataloader,
                description: str,
                prediction_loss_only: Optional[bool] = None,
                ignore_keys: Optional[List[str]] = None,
                metric_key_prefix: str = "eval",
            ) -> EvalLoopOutput:
                metrics = model.eval_loss_on_loader(dataloader)
                n_samples = metrics["loss_per_ex"].weight
                metrics = {
                    f"{metric_key_prefix}_{k}": v.mean() for k, v in metrics.items()
                }
                return EvalLoopOutput(
                    predictions=tuple(),
                    label_ids=tuple(),
                    metrics=metrics,
                    num_samples=n_samples,
                )

        epoch_steps = len(train_lodader)
        print("Number of training batches (estimate):")
        eval_interval = max(10, epoch_steps // 4)
        trainer_args = Seq2SeqTrainingArguments(
            output_dir=str(train_dir),
            overwrite_output_dir=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_interval,
            logging_steps=eval_interval,
            save_steps=eval_interval,
            num_train_epochs=train_args.max_train_epochs,
            save_total_limit=3,
            # prediction_loss_only=True,
            learning_rate=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
            metric_for_best_model="loss_per_tk",
            greater_is_better=False,
            fp16=True,
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to=["wandb"],
            disable_tqdm=True,
        )

        trainer = DynamicTrainer(
            self,
            trainer_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()
        save_dir = get_model_dir(trained=True) / training_name
        self.save(save_dir)
        print("Model saved to:", save_dir)

    def eval_loss_on_data(
        self, data: TokenizedEditDataset, batch_args: "BatchArgs"
    ) -> dict[str, WeightedSum]:
        batch_args = copy.deepcopy(batch_args)
        batch_args.max_ref_dropout = 0.0
        batch_args.shuffle_extra_ids = False
        eval_loader = edits_to_dataloader(
            data.all_edits(),
            args=batch_args,
            shuffle=False,
            desc="Eval Epoch",
        )
        return self.eval_loss_on_loader(eval_loader)

    @torch.no_grad()
    @torch.autocast("cuda")
    def eval_loss_on_loader(self, dataloader):
        core = self
        previous = core.training
        core.eval()
        metrics = dict[str, WeightedSum]()
        for batch in tqdm(dataloader, desc="evaluate loss", unit="batch"):
            batch["input_ids"] = batch["input_ids"].to(core.device)
            batch["labels"] = batch["labels"].to(core.device)
            outputs = core.forward(**batch)
            assert isinstance(outputs, Seq2SeqLMOutput)
            if CheckNaN:
                if outputs.logits.isnan().any():
                    print("loss:", not_none(outputs.loss).item())
                    print("batch:", batch)
                    raise ValueError("NaN in logits")
            for k, v in compute_loss_metrics(outputs.logits, batch["labels"]).items():
                v = v + metrics.get(k, WeightedSum(0.0, 0))
                metrics[k] = v
        core.train(mode=previous)

        return metrics

    def save(self, save_dir: Path, *args, **kwargs):
        super().save_pretrained(save_dir, *args, **kwargs)
        extra_args = {
            "query_attend_ref": self.query_attened_ref,
        }
        pickle_dump(save_dir / "extra_args.pkl", extra_args)

    @staticmethod
    def load(save_dir: Path) -> "RetrievalEditorModel":
        model = RetrievalEditorModel.from_pretrained(save_dir)
        assert isinstance(model, RetrievalEditorModel)
        if (save_dir / "extra_args.pkl").exists():
            extra_args = pickle_load(save_dir / "extra_args.pkl")
            model.query_attened_ref = extra_args["query_attend_ref"]
        else:
            warnings.warn("No extra args found, using default model setting.")
        return model

    def encode_token_seqs(
        self, references: Sequence[TokenSeq] | Sequence[str], pad_id=None
    ) -> LongTensor:
        references = [
            encode_basic(ref) if isinstance(ref, str) else ref for ref in references
        ]
        out = pad_token_seqs(references, pad_id=pad_id)
        out = out.to(self.device)
        return cast(LongTensor, out)

    def forward(
        self,
        # encoder args
        input_ids: LongTensor | None = None,  # queries
        references: Sequence[TokenSeq] | None = None,
        query_ref_list: Sequence[Sequence[int]] | None = None,
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
        - input_ids: (n_queries, query_len)
        - labels: (n_queries, label_len)
        """
        if labels is not None:
            assert_eq(labels.dim(), 2)

        try:
            if encoder_outputs is None:
                assert input_ids is not None
                encoder = self.get_encoder()
                encoder_outputs = encoder.forward(input_ids, references, query_ref_list)

            if labels is not None and decoder_input_ids is None:
                # get decoder inputs from shifting lm labels to the right
                assert_eq(labels.dtype, torch.long)
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
            assert isinstance(
                decoder_outputs, BaseModelOutputWithPastAndCrossAttentions
            )

            sequence_output = decoder_outputs[0]
            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim**-0.5)

            lm_logits = self.lm_head(sequence_output)
        except torch.cuda.OutOfMemoryError:  # type: ignore
            total_ref_len = sum(len(x) for x in references) if references else 0
            n_references = len(references) if references else 0
            if input_ids is not None:
                print(f"{input_ids.shape = }")
            if labels is not None:
                print(f"{labels.shape = }")
            print(f"{n_references = }, {total_ref_len = }")
            raise

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
        encoder_outputs=None,
        past=None,
        use_cache=None,
        **kwargs,
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
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
        return RetrivalEncoder(self.encoder, query_attened_ref=self.query_attened_ref)

    def get_decoder(self):
        return self.decoder

    def _reorder_cache(self, past, beam_idx):
        return T5ForConditionalGeneration._reorder_cache(
            cast(Any, None), past, beam_idx
        )

    @staticmethod
    def from_code_t5(
        size: Literal["small", "base", "large"],
        query_attened_ref: bool = True,
        reuse_embed: bool = False,
    ) -> "RetrievalEditorModel":
        model = RetrievalEditorModel.from_pretrained(f"Salesforce/codet5-{size}")
        assert isinstance(model, RetrievalEditorModel)
        embed_layer = model.resize_token_embeddings(len(_Tokenizer))
        if reuse_embed:
            w_map = {Add_id: get_tk_id("+"), Del_id: get_tk_id("-")}
            for k, v in w_map.items():
                embed_layer.weight.data[k] = embed_layer.weight[v]
        model.query_attened_ref = query_attened_ref
        model.config.vocab_size = len(_Tokenizer)
        return model


@dataclass
class RetrivalEncoderOutputs(transformers.utils.ModelOutput):
    last_hidden_state: Tensor
    hidden_state_mask: Tensor | None = None


@dataclass
class RetrivalEncoder:
    encoder: T5Stack
    query_attened_ref: bool

    def __call__(self, *args: Any, **kwds: Any) -> RetrivalEncoderOutputs:
        return self.forward(*args, **kwds)

    def forward(
        self,
        input_ids: LongTensor,
        references: Sequence[TokenSeq] | None = None,
        query_ref_list: Sequence[Sequence[int]] | None = None,
        # not used arguments below:
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> RetrivalEncoderOutputs:
        """
        Shapes
        - input_ids: (n_queries, seq_len)
        - references: (num_refs, ref_len)
        - ref_masks: for each query, a list of reference indices. If none,
        assume all references are accessible to all queries.
        """
        if references is None:
            references = []

        assert_eq(input_ids.dim(), 2)
        assert_eq(input_ids.dtype, torch.long)
        device = self.encoder.device

        n_queries = input_ids.size(0)
        n_refs = len(references)

        if query_ref_list is None:
            query_ref_list = [list(range(n_refs)) for _ in range(n_queries)]

        def split_outputs(
            lens: Sequence[int], out: BaseModelOutputWithPastAndCrossAttentions
        ) -> Iterable[BaseModelOutputWithPastAndCrossAttentions]:
            for i, l in enumerate(lens):
                hidden_states = tuple(
                    s[i : i + 1, :l] for s in not_none(out.hidden_states)
                )
                yield BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden_states[-1],  # type: ignore
                    hidden_states=hidden_states,  # type: ignore
                )

        ref_outputs = batched_map(
            references,
            group_key=lambda ref: self._round_length_group(len(ref)),
            f=lambda refs: split_outputs(
                [len(x) for x in refs],
                self.encoder.forward(
                    pad_token_seqs(refs).to(device),
                    output_hidden_states=True,
                    return_dict=True,
                ),
            ),
        )

        def encode_queries(query_ids: Sequence[int]) -> Iterable[Tensor]:
            queries = [cast(LongTensor, input_ids[q, : q_lens[q]]) for q in query_ids]
            assert query_ref_list is not None
            query_refs = [query_ref_list[q] for q in query_ids]
            q_tensor, q_mask = stack_pad_tensors(queries)
            assert_eq(q_tensor.dim(), 2)

            if self.query_attened_ref:
                enc = self.encode_query_complex(
                    query_ids=cast(LongTensor, q_tensor),
                    query_attention_mask=q_mask,
                    ref_outputs=ref_outputs,
                    query_ref_list=query_refs,
                )
            else:
                enc = self.encode_query_simple(
                    query_ids=cast(LongTensor, q_tensor),
                    query_attention_mask=q_mask,
                    ref_outputs=ref_outputs,
                    query_ref_list=query_refs,
                )
            last_hidden_state, hidden_state_mask = enc
            for i, _ in enumerate(queries):
                yield last_hidden_state[i, hidden_state_mask[i]]

        q_lens = input_ids.ne(PAD_id).sum(dim=1).tolist()

        last_hidden_states = batched_map(
            range(n_queries),
            group_key=lambda q: self._round_length_group(q_lens[q]),
            f=encode_queries,
        )
        last_hidden_state, hidden_state_mask = stack_pad_tensors(last_hidden_states)

        return RetrivalEncoderOutputs(
            last_hidden_state=last_hidden_state, hidden_state_mask=hidden_state_mask
        )

    @staticmethod
    def _round_length_group(x: int) -> int:
        if x <= 64:
            return 64
        return 2 ** math.ceil(math.log(x, 2))

    def encode_references(
        self,
        input_ids: LongTensor,
        attention_mask: Tensor | None = None,
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
        check_nan(
            "last_hidden_state",
            out.last_hidden_state,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        return out

    def encode_query_simple(
        self,
        query_ids: LongTensor,
        query_attention_mask: Tensor,
        ref_outputs: Sequence[BaseModelOutputWithPastAndCrossAttentions],
        query_ref_list: Sequence[Sequence[int]],
    ) -> tuple[Tensor, Tensor]:
        n_queries = len(query_ref_list)

        ref_state_list = [cast(Tensor, ro.last_hidden_state)[0] for ro in ref_outputs]

        query_outputs = self.encode_references(
            query_ids, attention_mask=query_attention_mask
        )
        query_states = query_outputs.last_hidden_state
        query_state_list = [
            query_states[i, query_attention_mask[i]] for i in range(n_queries)
        ]

        qref_rows = []
        for q in range(n_queries):
            qrefs = [ref_state_list[r] for r in query_ref_list[q]]
            qrefs.append(query_state_list[q])
            row_tensor = torch.cat(qrefs, dim=0)
            assert row_tensor.ndim == 2  # (sum(ref_lens) + query_len, model_dim)
            qref_rows.append(row_tensor)
        qref_states, qref_masks = stack_pad_tensors(qref_rows)
        assert_eq(qref_states.ndim == 3)
        assert_eq(qref_states.size(0), n_queries)

        return qref_states, qref_masks

    def encode_query_complex(
        self,
        query_ids: LongTensor,
        query_attention_mask: BoolTensor,
        ref_outputs: Sequence[BaseModelOutputWithPastAndCrossAttentions],
        query_ref_list: Sequence[Sequence[int]],
    ) -> tuple[Tensor, Tensor]:
        assert (
            query_ids[:, 0].ne(PAD_id).all()
        ), "queries must be padded only at the end."
        n_queries = len(query_ref_list)
        assert_eq(n_queries, query_ids.size(0))
        device = self.encoder.device
        model_d = self.encoder.config.d_model

        qref_hidden_states = list[Tensor]()
        qref_attention_masks = list[BoolTensor]()
        n_stacks = len(self.encoder.block) + 1
        if ref_outputs:
            assert_eq(n_stacks, len(not_none(ref_outputs[0].hidden_states)))
        for s in range(n_stacks):
            ref_state_list = [
                cast(Tuple[Tensor, ...], ref_states.hidden_states)[s][0]
                for ref_states in ref_outputs
            ]

            qref_rows = []
            for q in range(n_queries):
                qrefs = [ref_state_list[r] for r in query_ref_list[q]]
                if not qrefs:
                    qref_rows.append(torch.empty(0, model_d).to(device))
                else:
                    qref_rows.append(torch.cat(qrefs, dim=0))
            # qrefs are padded at the end
            qref_states, qref_masks = stack_pad_tensors(
                qref_rows
            )  # (n_queries, sum(ref_lens), model_dim)
            qref_hidden_states.append(qref_states)
            qref_attention_masks.append(qref_masks)

        query_outputs = encode_query_stack(
            stack=self.encoder,
            input_ids=query_ids,
            ref_hidden_states=tuple(qref_hidden_states),
            ref_attention_mask=qref_attention_masks[0],
        )

        # concat last hidden states
        query_states = query_outputs.last_hidden_state
        ref_states = qref_hidden_states[-1]
        ref_mask = qref_attention_masks[-1]

        combine_rows = []
        for q in range(n_queries):
            query_s = query_states[q, query_attention_mask[q]]
            ref_s = ref_states[q, ref_mask[q]]
            combine_rows.append(torch.cat([ref_s, query_s], dim=0))
        return stack_pad_tensors(combine_rows)


def stack_pad_tensors(xs: Sequence[Tensor]) -> tuple[Tensor, BoolTensor]:
    """Pad a list of tensors at the end. Return the padded tensor and a mask."""
    padded = nn.utils.rnn.pad_sequence(list(xs), batch_first=True)
    n_batch, n_len = padded.shape[:2]
    mask = cast(BoolTensor, padded.new_zeros(n_batch, n_len, dtype=torch.bool))
    for i, x in enumerate(xs):
        mask[i, : x.shape[0]] = True
    return padded, mask


def t5_cross_attention(
    layer: T5LayerSelfAttention,
    hidden_states,
    key_value_states,
    position_bias=None,
    output_attentions=False,
) -> tuple[Tensor, ...]:
    """Use a self attention layer as a cross attention layer.
    Note that you should encode any attention mask directly into position_bias.
    """
    normed_hidden_states = layer.layer_norm(hidden_states)
    normed_key_value_states = layer.layer_norm(key_value_states)
    attention_output = layer.SelfAttention.forward(
        normed_hidden_states,
        key_value_states=normed_key_value_states,
        position_bias=position_bias,
        output_attentions=output_attentions,
        # layer_head_mask=layer_head_mask,
        # past_key_value=past_key_value,
        # use_cache=use_cache,
        # query_length=query_length,
    )
    hidden_states = hidden_states + layer.dropout(attention_output[0])
    outputs = (hidden_states,) + attention_output[1:]
    return cast(tuple[Tensor, ...], outputs)


def encode_query_block(
    block: T5Block,
    query_hidden_states: Tensor,  # (n_queries, query_len, model_dim)
    ref_hidden_states: Tensor,  # (n_queries, ref_len, model_dim)
    position_bias: Tensor,
    output_attentions: bool = False,
) -> tuple[Tensor, ...]:
    """Run a T5Block to encode the query. Instead of using self-attention, this uses
    a hybrid attention where the query is allowed to attend to both itself and the references.
    """

    layer0 = block.layer[0]
    assert isinstance(layer0, T5LayerSelfAttention)
    key_value_states = torch.cat([ref_hidden_states, query_hidden_states], dim=1)
    hybrid_attention_outputs = t5_cross_attention(
        layer0,
        query_hidden_states,
        key_value_states=key_value_states,
        position_bias=position_bias,
        output_attentions=output_attentions,
    )
    hidden_states = hybrid_attention_outputs[0]

    check_nan(
        "hybrid_attention_outputs[0]",
        hidden_states,
        {
            "query_hidden_states": query_hidden_states,
            "ref_hidden_states": ref_hidden_states,
        },
    )

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    # Apply Feed Forward layer
    ff_layer = block.layer[-1]
    assert isinstance(ff_layer, T5LayerFF)
    hidden_states: Tensor = ff_layer.forward(hidden_states)

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    return (hidden_states, *hybrid_attention_outputs[1:])


def encode_query_stack(
    stack: T5Stack,
    input_ids: LongTensor,  # (n_queries, query_len)
    ref_hidden_states: tuple[Tensor, ...],  # tuples of (n_queries, ref_len, model_dim)
    ref_attention_mask: BoolTensor | None = None,  # (n_queries, ref_len)
    RefDistance: int = 1000,  # the added distance between the query and references
) -> BaseModelOutputWithPastAndCrossAttentions:
    """Run a T5Stack to encode the query. Instead of using self-attention, this uses
    a hybrid attention where the query is allowed to attend to both itself and the references.
    The relative distances between the query and the references are treated as infinity.
    """
    assert not stack.is_decoder
    device = input_ids.device

    assert input_ids[:, 0].ne(PAD_id).all(), "input_ids must be padded at only the end."

    input_shape = input_ids.size()
    batch_size, query_len = input_shape
    _, ref_len, model_dim = ref_hidden_states[0].size()

    # Masking query will cause numerical issues. We don't need to mask it anyway.
    input_attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    query_mask = input_ids.ne(PAD_id)
    if ref_attention_mask is None:
        ref_attention_mask = cast(
            BoolTensor, torch.ones(batch_size, ref_len, dtype=torch.bool).to(device)
        )
        assert_eq(input_ids.size(0), ref_attention_mask.size(0))

    assert_eq(ref_attention_mask.ndim, 2)
    assert_eq(query_mask.ndim, 2)
    assert_eq(ref_attention_mask.size(0), query_mask.size(0))
    # combine input and ref attention masks
    attention_mask = input_attention_mask.unsqueeze(2) * torch.cat(
        [ref_attention_mask, query_mask], dim=1
    ).unsqueeze(1)

    assert_eq(tuple(attention_mask.shape), (batch_size, query_len, query_len + ref_len))

    extended_attention_mask = stack.get_extended_attention_mask(
        attention_mask, input_shape
    )

    attention_layer = cast(T5Block, stack.block[0]).layer[0].SelfAttention
    assert isinstance(attention_layer, T5Attention)

    n_queries = input_ids.size(0)
    ref_lens = ref_attention_mask.sum(dim=1)[:, None]  # (n_queries, 1)
    # relative pos needs to be of shape (n_quries, query_len, ref_len + query_len)
    # ref_pos = torch.arange(ref_len, device=device, dtype=torch.long)[
    #     None, :
    # ]  # (1, ref_len)
    ref_pos = (torch.arange(ref_len, device=device, dtype=torch.long) - RefDistance)[
        None, :
    ]  # (1, ref_len)
    ref_pos = ref_pos + torch.zeros(
        n_queries, 1, device=device, dtype=torch.long
    )  # (n_queries, ref_len)
    query_pos = (
        torch.arange(query_len, device=device, dtype=torch.long)[None, :] + ref_lens
    )  # (n_queries, query_len)
    key_pos = torch.cat([ref_pos, query_pos], dim=1)  # (n_queries, ref_len + query_len)
    relative_pos = (
        key_pos[:, None, :] - query_pos[:, :, None]
    )  # (n_queries, query_len, ref_len + query_len)
    position_bias = compute_bias(attention_layer, relative_pos)
    check_nan("position_bias", position_bias, {})
    check_nan("extended_attention_mask", extended_attention_mask, {})
    position_bias = extended_attention_mask + position_bias
    check_nan("position_bias_after", position_bias, {})

    assert stack.embed_tokens is not None
    inputs_embeds = stack.embed_tokens(input_ids)
    hidden_states = stack.dropout(inputs_embeds)

    for i, block in enumerate(stack.block):
        # Model parallel
        assert isinstance(block, T5Block)
        ref_states = ref_hidden_states[i]
        layer_outputs = encode_query_block(
            block,
            hidden_states,
            ref_states,
            position_bias=position_bias,
            output_attentions=False,
        )

        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
        layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

        check_nan(
            "hidden_states",
            layer_outputs[0],
            {"i": i, "input_hidden_states": hidden_states, "ref_states": ref_states},
        )
        hidden_states, present_key_value_state = layer_outputs[:2]
        assert isinstance(hidden_states, Tensor)

    hidden_states = stack.final_layer_norm(hidden_states)
    hidden_states = stack.dropout(hidden_states)

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        # past_key_values=present_key_value_states,
        # hidden_states=all_hidden_states,
        # attentions=all_attentions,
        # cross_attentions=all_cross_attentions,
    )


def compute_bias(
    self: T5Attention,
    relative_pos: Tensor,
) -> Tensor:
    """Compute binned relative position bias from `relative_pos` of
    the shape `(n_queries, query_length, key_length)`"""
    relative_position_bucket = self._relative_position_bucket(
        relative_pos,  # shape (query_length, key_len)
        bidirectional=(not self.is_decoder),
        num_buckets=self.relative_attention_num_buckets,
        max_distance=self.relative_attention_max_distance,
    )
    values = self.relative_attention_bias(
        relative_position_bucket
    )  # shape (n_qureis, query_len, key_len, n_heads)
    values = values.permute(
        [0, 3, 1, 2]
    )  # shape (n_queries, n_heads, query_len, key_len)
    return values


def retrieval_cost_model(ref_size: int, query_size: int, dec_size: int) -> float:
    a = 1 / 128
    return (
        a * (ref_size + query_size) * (query_size + dec_size)
        + 2 * ref_size
        + query_size
        + 2 * dec_size
    )


@dataclass
class BatchArgs:
    max_output_tks: int = 256
    max_query_tks: int = 512
    min_queires: int = 1
    max_queries: int = 8
    max_ref_tks: int = 50 * 256
    max_ref_dropout: float = 0.3
    shuffle_extra_ids: bool = True
    use_only_modified: bool = True

    def cost_limit(self) -> float:
        return self.min_queires * retrieval_cost_model(
            self.max_ref_tks, 512, self.max_output_tks
        )


def edits_to_batches(
    edit_groups: Sequence[Sequence[BasicTkQueryEdit]],
    args: BatchArgs,
) -> list[dict]:
    def process_edit(e: BasicTkQueryEdit):
        labels = e.output_tks

        labels = wrap_bos(labels)

        if len(labels) > args.max_output_tks:
            labels = labels[: args.max_output_tks]

        input_ids = e.input_tks

        if args.shuffle_extra_ids and random.random() < 0.5:
            id_map = random_extra_id_map()
            input_ids = [id_map.get(tk, tk) for tk in input_ids]
            labels = [id_map.get(tk, tk) for tk in labels]

        return input_ids, labels

    cost_limit = args.cost_limit()
    warned_batch_size = False

    batches = list[dict]()
    bsizes = list[int]()
    for edit_group in edit_groups:
        pedit = edit_group[0].tk_pedit

        processed = [process_edit(x) for x in edit_group]
        input_tks_list = [x[0] for x in processed]
        output_tks_list = [x[1] for x in processed]

        queries_left = list(range(len(edit_group)))

        while queries_left:
            # down-sample references if needed
            references = [
                (path, seg)
                for path, segs in pedit.tk_references.items()
                for seg in segs
            ]
            n_ref = round(
                len(references) * (1 - args.max_ref_dropout * random.random())
            )
            ref_left = list(references)
            random.shuffle(ref_left)
            ref_size_sum = 0
            ref_selected = list[tuple[ProjectPath, TokenSeq]]()
            while ref_left and len(ref_selected) < n_ref:
                ref = ref_left.pop()
                if ref_size_sum + len(ref[1]) <= args.max_ref_tks:
                    ref_selected.append(ref)
                    ref_size_sum += len(ref[1])
                else:
                    break

            # find the maximal batch size
            max_bsize = min(len(queries_left), args.max_queries)
            bsize = 0
            for bsize in range(1, max_bsize + 1):
                queries = queries_left[:bsize]
                query_size = max(len(input_tks_list[i]) for i in queries)
                out_size = max(len(output_tks_list[i]) for i in queries)
                cost = bsize * retrieval_cost_model(
                    ref_size_sum,
                    query_size,
                    out_size,
                )
                if cost > cost_limit:
                    bsize -= 1
                    break
            if bsize == 0:
                if not warned_batch_size:
                    warned_batch_size = True
                    warnings.warn(
                        "Batch query limit is too small. Will use a query size of 1."
                    )
                bsize = 1

            queries = queries_left[:bsize]
            assert queries, "No queries in batch!"
            queries_left = queries_left[bsize:]
            input_ids = [input_tks_list[qid] for qid in queries]
            query_ref_list = [
                [
                    i
                    for i, (p, _) in enumerate(ref_selected)
                    if p != edit_group[qid].path
                ]
                for qid in queries
            ]
            labels = [output_tks_list[qid] for qid in queries]
            batch = {
                "input_ids": input_ids,
                "references": [tks for (_, tks) in ref_selected],
                "query_ref_list": query_ref_list,
                "labels": labels,
            }
            batches.append(batch)
            bsizes.append(bsize)

    batch_stats = {k: f"{v:.1f}" for k, v in scalar_stats(bsizes).items()}
    cprint("blue", f"num batches: {len(batches)}")
    cprint("blue", f"Batch stats: {batch_stats}")

    return batches


@dataclass
class _BatchSampler:
    edit_groups: list[list[BasicTkQueryEdit]]
    data_args: BatchArgs
    shuffle: bool
    desc: str
    tqdm_args: dict | None = None

    def __post_init__(self):
        if self.shuffle:
            random.shuffle(self.edit_groups)
        self._len_est = self.estimate_n_batches()
        self.epochs = 0

    def __len__(self) -> int:
        return self._len_est

    def estimate_n_batches(self) -> int:
        batches = edits_to_batches(self.edit_groups, self.data_args)
        return len(batches)

    def __iter__(self) -> Iterable[Mapping]:
        if self.shuffle:
            for es in self.edit_groups:
                random.shuffle(es)
        batches = edits_to_batches(self.edit_groups, self.data_args)
        if self.shuffle:
            random.shuffle(batches)

        tqdm_args = self.tqdm_args or {"smoothing": 0.0}
        for b in tqdm(batches, desc=self.desc + f" {self.epochs}", **tqdm_args):
            input_ids = pad_token_seqs(b["input_ids"])
            labels = pad_token_seqs(b["labels"], pad_id=-100)
            yield {
                "input_ids": input_ids,
                "references": b["references"],
                "query_ref_list": b["query_ref_list"],
                "labels": labels,
            }
        self.epochs += 1


def edits_to_dataloader(
    edits: Sequence[BasicTkQueryEdit],
    args: BatchArgs,
    desc: str,
    shuffle: bool = False,
):
    # if args.use_only_modified:
    #     edits = [e for e in edits if isinstance(e.change_type, Modified)]
    assert edits
    edit_groups = list(groupby(edits, lambda e: id(e.tk_pedit)).values())
    assert edit_groups
    return _BatchSampler(edit_groups, args, shuffle=shuffle, desc=desc)


def pad_token_seqs(seqs: Sequence[TokenSeq], pad_id=None) -> LongTensor:
    max_len = max(len(ref) for ref in seqs)
    if pad_id is None:
        pad_id = PAD_id
    rows = []
    for row in seqs:
        row = row + [pad_id] * (max_len - len(row))
        rows.append(row)
    return LongTensor(rows)
