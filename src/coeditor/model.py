import copy
import dataclasses
import logging
import shutil
from textwrap import indent

import torch
import transformers
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, nn
from transformers import (
    AutoConfig,
    EarlyStoppingCallback,
    SchedulerType,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from transformers.models.t5.modeling_t5 import (
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    T5Attention,
    T5Block,
    T5Config,
    T5ForConditionalGeneration,
    T5LayerFF,
    T5LayerSelfAttention,
    T5PreTrainedModel,
    T5Stack,
)
from transformers.trainer import EvalLoopOutput

from coeditor._utils import cprint, groupby, scalar_stats
from coeditor.c3problem import (
    C3Problem,
    C3ProblemTokenizer,
    C3ProblemTransform,
    TkC3Problem,
)
from coeditor.change import Modified
from coeditor.encoding import (
    Add_id,
    BOS_id,
    Del_id,
    EOS_id,
    Newline_id,
    PAD_id,
    TkDelta,
    _Tokenizer,
    apply_output_tks_to_change,
    change_tks_to_original_delta,
    change_to_tokens,
    decode_tokens,
    encode_lines_join,
    get_tk_id,
    is_extra_id,
    output_ids_as_seqs,
    random_extra_id_map,
    tokens_to_change,
)
from coeditor.tk_array import TkArray

from .common import *

CheckNaN: bool = False
CodeT5Model = T5ForConditionalGeneration


@dataclass
class DecodingArgs:
    max_output_tks: int = 512
    do_sample: bool = False
    top_p: float = 0.9
    num_beams: Optional[int] = 1
    length_penalty: float = 0.0
    marginalize_samples: int = 1

    def to_model_args(self) -> dict:
        return {
            "max_length": self.max_output_tks,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
        }


@dataclass
class TrainingArgs:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_train_epochs: int = 3
    reinit_weights: bool = False
    quicktest: bool = False
    lr_scheduler_type: SchedulerType = SchedulerType.LINEAR


class ModelPrediction(TypedDict):
    input_ids: TokenSeq
    output_ids: TokenSeq
    labels: TokenSeq


def compute_loss_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> Mapping[str, WeightedSum]:
    logits = logits.permute(0, 2, 1)  # shape: (batch, vocab, seq_len)
    nlogp = torch.nn.functional.cross_entropy(
        logits,
        labels,
        reduction="none",
        ignore_index=-100,
    )  # shape: (batch, seq_len)
    loss = nlogp.sum().item()
    ex_prob = torch.exp(-nlogp.sum(dim=1)).sum().item()
    bsize = logits.size(0)
    label_tks = (labels != -100).sum().item()
    return {
        "loss": WeightedSum(loss / label_tks, 1),
        "loss_per_tk": WeightedSum(loss, label_tks),
        "loss_per_ex": WeightedSum(loss, bsize),
        "prob_per_ex": WeightedSum(ex_prob, bsize),
    }


def wrap_bos(x: TokenSeq) -> TokenSeq:
    if x:
        assert x[0] != BOS_id
    return [BOS_id] + x + [EOS_id]


def drop_empty_labels(x: TokenSeq) -> TokenSeq:
    """Drop the <extra_id>s that are not followed by a code token."""
    new_seq = TokenSeq()
    for k, v in output_ids_as_seqs(x).items():
        if v:
            new_seq.append(k)
            new_seq.extend(v)
    return new_seq


def remove_pad_ids(ids: TokenSeq) -> TokenSeq:
    return [tk for tk in ids if tk != PAD_id and tk >= 0]


def check_nan(name: str, x: Tensor, inputs: dict):
    if CheckNaN and torch.isnan(x).any():
        print(f"NaN found in {name}")
        print("Found value:", x)
        for k, v in inputs.items():
            print(k, "=", v)
        raise Exception(f"NaN found in {name}")


class PredictedChange(NamedTuple):
    change: Modified[str]
    out_tks: TokenSeq
    score: float
    n_samples: int


class RetrievalModelPrediction(TypedDict):
    input_ids: TokenSeq
    output_ids: TokenSeq
    labels: TokenSeq
    references: list[TokenSeq]


@dataclass
class RetrievalDecodingResult:
    eval_args: dict
    problems: Sequence[C3Problem]
    predictions: Sequence[RetrievalModelPrediction]

    def __post_init__(self):
        assert_eq(len(self.problems), len(self.predictions))

    def exact_match_accuracy(self) -> tuple[CountedSum, dict[int, bool]]:
        ex2correct = dict[int, bool]()
        bad_probs = list[C3Problem]()
        for i, mp in enumerate(self.predictions):
            prob = self.problems[i]
            original = prob.span.original.tolist()
            pred_delta = TkDelta.from_output_tks(prob.edit_lines, mp["output_ids"])
            label_delta = TkDelta.from_output_tks(prob.edit_lines, mp["labels"])
            if not prob.edit_lines:
                bad_probs.append(prob)
                continue
            line_shift = prob.edit_lines[0]
            pred_change = pred_delta.shifted(line_shift).apply_to_change(original)
            label_change = label_delta.shifted(line_shift).apply_to_change(original)
            pred_code = tokens_to_change(pred_change).after
            label_code = tokens_to_change(label_change).after
            ex2correct[i] = code_equal(pred_code, label_code)
        correct_count = CountedSum(sum(ex2correct.values()), len(ex2correct))
        if bad_probs:
            cprint("yellow", "Number of problems with no edits:", len(bad_probs))
            for prob in bad_probs[:5]:
                print(prob.summary())
        return correct_count, ex2correct

    def save_examples_to_dir(self, out_dir: Path, ex2correct: dict[int, bool]) -> None:
        shutil.rmtree(out_dir, ignore_errors=True)
        (out_dir / "correct").mkdir(parents=True, exist_ok=True)
        (out_dir / "incorrect").mkdir(parents=True, exist_ok=True)

        all_probs = dict[int, C3Problem]()
        for ex_id, correct in tqdm(ex2correct.items(), desc="saving examples"):
            ex = self.predictions[ex_id]
            prob = self.problems[ex_id]
            compare_str = self.show_prediction(prob, ex)
            out_file = (
                out_dir / ("correct" if correct else "incorrect") / f"ex-{ex_id}.txt"
            )
            out_file.write_text(compare_str)
            all_probs[ex_id] = prob
        pickle_dump(out_dir / "ex_probs.pkl", all_probs)

    @classmethod
    def show_prediction(cls, prob: C3Problem, pred: RetrievalModelPrediction) -> str:
        span = prob.span
        tk_prob = TkC3Problem(
            input=TkArray.new(pred["input_ids"]),
            output=TkArray.new(pred["labels"]),
            path=span.headers[-1].path,
            change_type=prob.change_type,
            named_references=[
                (f"reference-{i}", TkArray.new(ref))
                for i, ref in enumerate(pred["references"])
            ],
            project=prob.src_info["project"],
            commit=prob.src_info["commit"],
        )
        return tk_prob.show(pred["output_ids"])


class AttentionMode(enum.Enum):
    basic = enum.auto()
    query2ref = enum.auto()
    bidirectional = enum.auto()


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

        amode = getattr(config, "attention_mode", AttentionMode.bidirectional.name)
        self.attention_mode = AttentionMode[amode]
        self.tlogger = TimeLogger()

    @property
    def attention_mode(self):
        return self._attention_mode

    @attention_mode.setter
    def attention_mode(self, mode: AttentionMode):
        self._attention_mode = mode
        self.config.attention_mode = mode.name

    def train_on_data(
        self,
        training_name: str,
        train_loader: "C3DataLoader",
        eval_loader: "C3DataLoader",
        train_args: "TrainingArgs",
    ) -> None:
        train_dir = get_model_dir(trained=False) / training_name
        eval_loader.tqdm_args = {"disable": True}

        model = self
        # model = torch.compile(self.to("cuda"))  # pytorch doesn't support python 3.11 yet.

        class DynamicTrainer(Seq2SeqTrainer):
            def get_train_dataloader(self):
                return train_loader

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
                metrics = model.eval_loss_on_loader(as_any(dataloader))
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

        epoch_steps = len(train_loader)
        cprint("blue", "Number of training batches (estimate):", epoch_steps)
        trainer_args = Seq2SeqTrainingArguments(
            output_dir=str(train_dir),
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=max(1, min(1000, epoch_steps // 10)),
            num_train_epochs=train_args.max_train_epochs,
            save_total_limit=2,
            lr_scheduler_type=train_args.lr_scheduler_type,
            learning_rate=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
            metric_for_best_model="loss_per_tk",
            greater_is_better=False,
            fp16=True,
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to=["wandb"],
            disable_tqdm=True,
            # torchdynamo="inductor",  # use compiled model
        )

        trainer = DynamicTrainer(
            self,
            trainer_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )

        trainer.train()
        save_dir = get_model_dir(trained=True) / training_name
        self.save(save_dir)
        print("Model saved to:", save_dir)

    @torch.no_grad()
    @torch.autocast("cuda")
    def eval_loss_on_loader(self, dataloader: "C3DataLoader"):
        core = self
        previous = core.training
        core.eval()
        metrics = dict[str, WeightedSum]()
        for batch in dataloader.__iter__():
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

    @torch.no_grad()
    @torch.autocast("cuda")
    def predict_on_data(
        self,
        eval_problems: Sequence[C3Problem],
        tokenizer: C3ProblemTokenizer,
        batch_args: "BatchArgs",
        dec_args: DecodingArgs,
    ):
        if batch_args.shuffle_extra_ids:
            warnings.warn(
                "Shuffling extra ids during eval can lead to incorrect results."
            )

        eval_loader = C3DataLoader(
            eval_problems,
            None,
            tokenizer,
            batch_args,
            shuffle=False,
            desc="predict_on_data",
        )

        gen_args = dec_args.to_model_args()
        batch_elems = list[RetrievalModelPrediction]()
        for batch in eval_loader:  # type: ignore
            out_tks = self.generate(
                batch["input_ids"].to(self.device),
                references=batch["references"],
                query_ref_list=batch["query_ref_list"],
                **gen_args,
            ).tolist()  # type: ignore
            input_ids = batch["input_ids"].tolist()
            labels = batch["labels"].tolist()
            query_ref_list = batch["query_ref_list"]
            for i in range(len(input_ids)):
                all_refs = batch["references"]
                references = [all_refs[j] for j in query_ref_list[i]]
                e = RetrievalModelPrediction(
                    input_ids=remove_pad_ids(input_ids[i]),
                    output_ids=remove_pad_ids(out_tks[i]),
                    labels=labels[i],
                    references=references,
                )
                batch_elems.append(e)
        return RetrievalDecodingResult(
            eval_args={"batch_args": batch_args, "dec_args": dec_args},
            problems=eval_problems,
            predictions=batch_elems,
        )

    @torch.autocast("cuda")
    def predict_on_batch(
        self,
        batch: dict,
        originals: Sequence[TokenSeq],
        dec_args: DecodingArgs,
        n_solutions: int = 1,
    ) -> list[list[PredictedChange]]:
        """
        Returns nested list of shape `(batch_size, n_solutions)`.
        """
        timed = self.tlogger.timed

        def marginalize_preds(
            preds: Sequence[Modified[str]],
            out_tks: Sequence[TokenSeq],
            weights: Sequence[float],
            scores: Sequence[float],
        ) -> list[PredictedChange]:
            """For sampling techniques, all sample should have equal weights 1/N. For
            search-based techniques, the `weights` should equal to the solutions' probabilities."""
            assert preds
            groups = groupby(
                range(len(preds)),
                keyfunc=lambda i: normalize_code_by_ast(preds[i].after),
            )
            groups = list(groups.values())
            for group in groups:
                # within each group, sort by score
                group.sort(key=lambda i: scores[i], reverse=True)
            groups.sort(
                key=lambda g: (sum(weights[i] for i in g), scores[g[0]]), reverse=True
            )
            return [
                PredictedChange(
                    preds[g[0]], out_tks[g[0]], sum(weights[i] for i in g), len(g)
                )
                for g in groups
            ]

        use_sampling = dec_args.marginalize_samples > 1
        if use_sampling:
            assert_eq(dec_args.do_sample, True)
            assert_eq(dec_args.num_beams, 1)
            N = dec_args.marginalize_samples
        else:
            N = dec_args.num_beams or 1
        gen_args = dec_args.to_model_args()
        input_ids = batch["input_ids"]
        if not isinstance(input_ids, torch.LongTensor):
            input_ids = torch.LongTensor(input_ids)
        with timed("model.generate"), tqdm(total=dec_args.max_output_tks) as pbar:
            gen_out = self.generate(
                input_ids.to(self.device),
                references=batch["references"],
                query_ref_list=batch["query_ref_list"],
                num_return_sequences=N,
                return_dict_in_generate=True,
                output_scores=True,
                **gen_args,
                tqdm=pbar,
            )
        assert not isinstance(gen_out, torch.LongTensor)
        out_tks = gen_out["sequences"]
        if isinstance(out_tks, torch.Tensor):
            out_tks = out_tks.tolist()
        out_tks = [remove_pad_ids(x) for x in out_tks]
        assert isinstance(out_tks, list)
        logging.debug("Max out length:", max(len(x) for x in out_tks))
        assert_eq(len(out_tks), len(originals) * N)
        originals = join_list([[x] * N for x in originals])
        if (pred_scores := gen_out.get("sequences_scores", None)) is None:
            pred_scores = [0.0] * len(out_tks)
        if use_sampling:
            pred_weights = [1.0 / N] * len(out_tks)
        else:
            pred_weights = [math.exp(x) for x in pred_scores]
        with timed("assemble changes"):
            pred_changes = list[Modified[str]]()
            for change_tks, out in zip(originals, out_tks):
                pred = apply_output_tks_to_change(change_tks, 0, out)
                pred_changes.append(pred)
        assert_eq(len(pred_changes), len(out_tks), len(pred_scores))

        solutions = list[list[PredictedChange]]()
        for i in range(0, len(pred_changes), N):
            sols = marginalize_preds(
                pred_changes[i : i + N],
                out_tks[i : i + N],
                pred_weights[i : i + N],
                pred_scores[i : i + N],
            )
            solutions.append(sols[:n_solutions])
        return solutions

    def save(self, save_dir: Path, *args, **kwargs):
        super().save_pretrained(save_dir, *args, **kwargs)

    @staticmethod
    def load(save_dir: Path | str) -> "RetrievalEditorModel":
        model = RetrievalEditorModel.from_pretrained(save_dir)
        assert isinstance(model, RetrievalEditorModel)
        # for loading model in legacy format
        if isinstance(save_dir, Path) and (save_dir / "extra_args.pkl").exists():
            extra_args = pickle_load(save_dir / "extra_args.pkl")
            model.attention_mode = extra_args.get(
                "attention_mode", AttentionMode.query2ref
            )
        return model

    def encode_token_seqs(
        self, references: Sequence[TokenSeq] | Sequence[str], pad_id=None
    ) -> LongTensor:
        references = [
            encode_lines_join(ref) if isinstance(ref, str) else ref
            for ref in references
        ]
        out = pad_token_seqs(references, pad_id=pad_id)
        out = out.to(self.device)
        return cast(LongTensor, out)

    def profile_run(self, repeats: int = 10, max_refs: int = 10):
        rand = random.Random(42)
        for i in tqdm(range(repeats), "test run"):
            input_ids = 5 * torch.ones(
                1, rand.randint(64, 512), dtype=torch.long, device=self.device
            )
            n_refs = rand.randint(max_refs // 2, max_refs)
            references = [[5] * rand.randint(64, 512) for _ in range(n_refs)]
            labels = 5 * torch.ones(1, 128, dtype=torch.long, device=self.device)
            with torch.autocast("cuda"):
                self.forward(as_any(input_ids), references, labels=as_any(labels))

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
        tqdm=None,
    ) -> Seq2SeqLMOutput:
        """
        Shapes
        - input_ids: (n_queries, query_len)
        - labels: (n_queries, label_len)
        """
        if labels is not None:
            assert_eq(labels.dim(), 2)

        def run_decoder(enc_results: Sequence[dict]):
            if len(enc_results) == 1:
                decoder_input_ids = enc_results[0]["decoder_input_ids"].unsqueeze(0)
                encoder_hidden_states = enc_results[0][
                    "encoder_hidden_states"
                ].unsqueeze(0)
                decoder_attention_mask = None
                encoder_attention_mask = None
            else:
                decoder_input_ids, decoder_attention_mask = stack_pad_tensors(
                    [x["decoder_input_ids"] for x in enc_results]
                )
                encoder_hidden_states, encoder_attention_mask = stack_pad_tensors(
                    [x["encoder_hidden_states"] for x in enc_results]
                )

            decoder_outputs = self.decoder.forward(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
            )
            assert isinstance(
                decoder_outputs, BaseModelOutputWithPastAndCrossAttentions
            )
            n = len(enc_results)
            return [decoder_outputs.last_hidden_state[i] for i in range(n)]

        def decode_group(enc_r: dict):
            assert isinstance(enc_r, dict)
            s1 = _round_length_group(enc_r["decoder_input_ids"].size(0))
            s2 = _round_length_group(enc_r["encoder_hidden_states"].size(0))
            return (s1, s2)

        try:
            if encoder_outputs is None:
                assert input_ids is not None
                encoder = self.get_encoder()
                with self.tlogger.timed("encoder.forward"):
                    encoder_outputs = encoder.forward(
                        input_ids, references, query_ref_list
                    )

            if labels is not None and decoder_input_ids is None:
                # get decoder inputs from shifting lm labels to the right
                assert_eq(labels.dtype, torch.long)

                last_hidden = encoder_outputs.last_hidden_state
                last_mask = not_none(encoder_outputs.hidden_state_mask)
                last_states = [
                    {
                        "encoder_hidden_states": last_hidden[i][last_mask[i]],
                        "decoder_input_ids": self._shift_right(
                            labels[i : i + 1]
                        ).squeeze(0),
                    }
                    for i in range(last_hidden.size(0))
                ]
                with self.tlogger.timed("decoder.forward"):
                    dec_hidden_states = batched_map(
                        last_states,
                        group_key=decode_group,
                        f=run_decoder,
                    )
                    decoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
                        cast(FloatTensor, stack_pad_tensors(dec_hidden_states)[0])
                    )
            else:
                # use simple batching for decoding
                with self.tlogger.timed("decoder.forward"):
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
        return RetrivalEncoder(
            self.encoder,
            attention_mode=self.attention_mode,
        )

    def get_decoder(self):
        return self.decoder

    def _reorder_cache(self, past, beam_idx):
        if past is None:
            warnings.warn(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)
                    ),
                )

            # assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past

    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor=None,
        stopping_criteria=None,
        logits_warper=None,
        # max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ):
        """An optimized sample implementation that does not waste computation
        on finished sequences."""
        # init values
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        if logits_warper is None:
            logits_warper = LogitsProcessorList()
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        device = self.device

        # keep track of which sequences are already finished
        unfinished_ids = torch.LongTensor(range(input_ids.shape[0])).to(device)
        sequences = input_ids.int().tolist()
        sequences_scores = [0.0 for _ in range(input_ids.shape[0])]
        # TODO: reduce cost using particle weights

        # auto-regressive generation
        t = 0
        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.forward(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_token_logits = cast(FloatTensor, outputs.logits[:, -1, :])

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            assert_eq(next_tokens.ndim, 1)
            for i, id in enumerate(unfinished_ids.tolist()):
                sequences_scores[id] += math.log(probs[i, next_tokens[i]].item())
                sequences[id].append(next_tokens[i].item())

            next_subset = next_tokens != eos_token_id
            subset_ids = torch.arange(len(unfinished_ids), device=device)[next_subset]
            unfinished_ids = unfinished_ids[next_subset]

            # update generated ids, model inputs, and length for next step
            input_ids = cast(
                torch.LongTensor,
                next_tokens[next_subset].unsqueeze(-1),
            )
            assert_eq(input_ids.ndim, 2)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(
                    model_kwargs["past"], subset_ids
                )

            if (pbar := model_kwargs.get("tqdm")) is not None:
                pbar = cast(tqdm, pbar)
                pbar.set_postfix({"unfinished": len(unfinished_ids)})
                pbar.update()
            # stop when each sentence is finished, or if we exceed the maximum length
            fake_input = torch.LongTensor([[1] * t]).to(device)
            t += 1
            if len(unfinished_ids) == 0 or stopping_criteria(fake_input, None):  # type: ignore
                break

        if return_dict_in_generate:
            return {"sequences": sequences, "sequences_scores": sequences_scores}
        else:
            return sequences

    @staticmethod
    def from_code_t5(
        size: Literal["small", "base", "large"],
        attention_mode: AttentionMode = AttentionMode.bidirectional,
        reuse_embed: bool = False,
        reinit_weights: bool = False,
    ) -> "RetrievalEditorModel":
        model_path = f"Salesforce/codet5-{size}"
        if reinit_weights:
            config = AutoConfig.from_pretrained(model_path)
            model = RetrievalEditorModel(config)
        else:
            model = RetrievalEditorModel.from_pretrained(model_path)
        assert isinstance(model, RetrievalEditorModel)
        embed_layer = model.resize_token_embeddings(len(_Tokenizer))
        if reuse_embed:
            w_map = {Add_id: get_tk_id("+"), Del_id: get_tk_id("-")}
            for k, v in w_map.items():
                embed_layer.weight.data[k] = embed_layer.weight[v]
        model.attention_mode = attention_mode
        model.config.vocab_size = len(_Tokenizer)
        return model


@dataclass
class RetrivalEncoderOutputs(transformers.utils.ModelOutput):
    last_hidden_state: Tensor
    hidden_state_mask: Tensor | None = None


@dataclass
class RetrivalEncoder:
    encoder: T5Stack
    attention_mode: AttentionMode

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
        tqdm=None,
    ) -> RetrivalEncoderOutputs:
        """
        Shapes
        - input_ids: (n_queries, seq_len)
        - references: (num_refs, ref_len)
        - ref_masks: for each query, a list of reference indices. If none,
        assume all references are accessible to all queries.
        """

        def to_long_tensor(data):
            return cast(
                LongTensor,
                torch.tensor(data, dtype=torch.long).to(device),
            )

        if references is None:
            references = []

        assert_eq(input_ids.dim(), 2)
        assert_eq(input_ids.dtype, torch.long)
        device = self.encoder.device

        n_queries = input_ids.size(0)
        q_lens = input_ids.ne(PAD_id).sum(dim=1).tolist()
        n_refs = len(references)

        if query_ref_list is None:
            query_ref_list = [list(range(n_refs)) for _ in range(n_queries)]

        if self.attention_mode.value == AttentionMode.bidirectional.value:
            # use bidirectional implementation
            queries = [cast(LongTensor, input_ids[i, :l]) for i, l in enumerate(q_lens)]
            refs = [
                [to_long_tensor(references[rid]) for rid in rids]
                for rids in query_ref_list
            ]
            hidden_rows = self.bidirectional_forward(queries, refs)
            last_hidden_state, hidden_state_mask = stack_pad_tensors(hidden_rows)
            return RetrivalEncoderOutputs(
                last_hidden_state=last_hidden_state, hidden_state_mask=hidden_state_mask
            )

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
            group_key=lambda ref: _round_length_group(len(ref)),
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

            if self.attention_mode.value == AttentionMode.query2ref.value:
                enc = self.encode_query_uni_directional(
                    query_ids=cast(LongTensor, q_tensor),
                    query_attention_mask=q_mask,
                    ref_outputs=ref_outputs,
                    query_ref_list=query_refs,
                )
            else:
                assert_eq(self.attention_mode.value, AttentionMode.basic.value)
                enc = self.encode_query_basic(
                    query_ids=cast(LongTensor, q_tensor),
                    query_attention_mask=q_mask,
                    ref_outputs=ref_outputs,
                    query_ref_list=query_refs,
                )
            last_hidden_state, hidden_state_mask = enc
            for i, _ in enumerate(queries):
                yield last_hidden_state[i, hidden_state_mask[i]]

        def query_group_key(q: int) -> tuple[int, int]:
            q_len = q_lens[q]
            ref_len = sum(
                len(not_none(references)[r]) for r in not_none(query_ref_list)[q]
            )
            return _round_length_group(q_len), _round_length_group(ref_len)

        last_hidden_states = batched_map(
            range(n_queries),
            group_key=query_group_key,
            f=encode_queries,
        )
        last_hidden_state, hidden_state_mask = stack_pad_tensors(last_hidden_states)

        return RetrivalEncoderOutputs(
            last_hidden_state=last_hidden_state, hidden_state_mask=hidden_state_mask
        )

    def _init_embed(self, input_ids: LongTensor) -> Tensor:
        stack = self.encoder
        assert stack.embed_tokens is not None
        inputs_embeds = stack.embed_tokens(input_ids)
        hidden_states = stack.dropout(inputs_embeds)
        return hidden_states

    def bidirectional_forward(
        self,
        queries: Sequence[LongTensor],
        references: Sequence[Sequence[LongTensor]],
    ) -> Sequence[Tensor]:
        """Each query and ref is allowed to attend to itself using self-attention,
        and additionally, the query is allowed to attend to all references and vice versa.

        Return the hidden states for each query and references.
        """
        assert_eq(len(queries), len(references))
        stack = self.encoder
        cache = dict()
        hidden_states = list[Tensor]()

        for refs, query in zip(references, queries):
            input_ids = torch.cat([*refs, query], dim=0)
            h = self._init_embed(cast(LongTensor, input_ids)).unsqueeze(0)
            block_sizes = [x.size(0) for x in [*refs, query]]
            for i, block in enumerate(stack.block):
                assert isinstance(block, T5Block)
                h = query_ref_layer_batched(block, h, block_sizes, cache)
            h = stack.dropout(stack.final_layer_norm(h)).squeeze(0)
            hidden_states.append(h)

        return hidden_states

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

    def encode_query_basic(
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

    def encode_query_uni_directional(
        self,
        query_ids: LongTensor,
        query_attention_mask: BoolTensor,
        ref_outputs: Sequence[BaseModelOutputWithPastAndCrossAttentions],
        query_ref_list: Sequence[Sequence[int]],
    ) -> tuple[Tensor, Tensor]:
        # assert (
        #     query_ids[:, 0].ne(PAD_id).all()
        # ), "queries must be padded only at the end."
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

        query_outputs = _encode_query_stack(
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
    Basically a self attention layer with layer_norm, dropout, and skip connection.
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


def t5_attention(
    self: T5Attention,
    hidden_states,
    key_value_states,
    position_bias=None,
):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    def shape(states):
        """projection"""
        return states.view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    # (batch_size, n_heads, seq_length, dim_per_head)
    query_states = shape(self.q(hidden_states))

    # get key/value states
    key_states = shape(self.k(key_value_states))
    value_states = shape(self.v(key_value_states))

    # compute scores
    scores = torch.matmul(
        query_states, key_states.transpose(3, 2)
    )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

    scores += position_bias
    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        scores
    )  # (batch_size, n_heads, seq_length, key_length)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )  # (batch_size, n_heads, seq_length, key_length)

    attn_output = unshape(
        torch.matmul(attn_weights, value_states)
    )  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    return attn_output


def t5_sparse_attention(
    self: T5Attention,
    hidden_states: Tensor,
    block_lens: Sequence[int],
    get_bias: Callable[[int, int, bool], Tensor],
):
    """
    A block-sparse self-attention layer that allows all blocks to attend to the
    last block (in additon to themselves).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]
    if not block_lens:
        raise ValueError("block_lens must not be empty")
    block_lens = list(block_lens)

    blocks = list[tuple[int, int]]()
    start = 0
    for block_len in block_lens:
        end = start + block_len
        blocks.append((start, end))
        start = end
    assert_eq(start, seq_length)

    def shape(states: Tensor):
        """projection"""
        return states.view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)

    def unshape(states: Tensor):
        """reshape from (batch_size, n_heads, seq_length, dim_per_head) to (batch_size, seq_length, dim)"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    # (batch_size, n_heads, seq_length, dim_per_head)
    query_states = shape(self.q(hidden_states))
    key_states = shape(self.k(hidden_states))
    value_states = shape(self.v(hidden_states))

    queries = list(torch.split_with_sizes(query_states, block_lens, dim=2))
    keys = list(torch.split_with_sizes(key_states, block_lens, dim=2))
    values = list(torch.split_with_sizes(value_states, block_lens, dim=2))

    g_start, g_end = blocks[-1]
    # add everything except the last block as special blocks
    queries.append(query_states[:, :, :g_start, :])
    keys.append(key_states[:, :, :g_start, :])
    values.append(value_states[:, :, :g_start, :])
    N = len(blocks)

    attn_outputs = []

    # compute for ref blocks
    for i, self_len in enumerate(block_lens):
        is_global = i == len(blocks) - 1
        j = N if is_global else N - 1

        query = queries[i]
        key = keys[i]
        value = values[i]

        other_key = keys[j]
        other_value = values[j]
        other_len = other_key.size(2)

        # compute scores
        self_scores = torch.matmul(query, key.transpose(3, 2))
        other_scores = torch.matmul(query, other_key.transpose(3, 2))
        # (batch_size, n_heads, query_len, query_len + global_len)
        scores = torch.cat([other_scores, self_scores], dim=-1)
        pos_bias = get_bias(self_len, other_len, is_global)
        scores = scores + pos_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        other_weights = attn_weights[:, :, :, :other_len]
        self_weights = attn_weights[:, :, :, other_len:]
        self_output = torch.matmul(self_weights, value)
        other_output = torch.matmul(other_weights, other_value)
        # (batch_size, n_heads, seq_length, dim_per_head)
        out = self_output + other_output
        assert_eq(out.size(2), self_len)
        attn_outputs.append(out)

    # (batch_size, seq_length, dim)
    attn_output = unshape(torch.cat(attn_outputs, dim=2))
    assert_eq(attn_output.ndim, 3)
    assert_eq(attn_output.size(1), seq_length)

    attn_output = self.o(attn_output)
    assert isinstance(attn_output, Tensor)
    assert_eq(attn_output.shape, hidden_states.shape)
    return attn_output


def _encode_query_block(
    block: T5Block,
    query_hidden_states: Tensor,  # (n_queries, query_len, model_dim)
    ref_hidden_states: Tensor,  # (n_queries, ref_len, model_dim)
    position_bias: Tensor,
    output_attentions: bool = False,
) -> Tensor:
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

    # Apply Feed Forward layer
    ff_layer = block.layer[-1]
    assert isinstance(ff_layer, T5LayerFF)
    return _run_t5_ff(ff_layer, hidden_states)


def _run_t5_ff(ff_layer: T5LayerFF, x: Tensor):
    # clamp inf values to enable fp16 training
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp_value = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp_value, max=clamp_value)

    # Apply Feed Forward layer
    x = ff_layer.forward(x)

    # clamp inf values to enable fp16 training
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp_value = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x


def _get_extended_attention_mask(
    attention_mask: Tensor, input_shape: tuple[int, ...], dtype
) -> Tensor:
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(
        dtype=dtype
    )  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


def _get_position_bias(
    attention_layer: T5Attention,
    batch_size: int,
    query_len: int,
    ref_len: int,
    device,
    query_mask: BoolTensor | None = None,
    ref_mask: BoolTensor | None = None,
    RefDistance: int = 10000,
):
    n_queries = batch_size
    if query_mask is None:
        query_mask = cast(
            BoolTensor,
            torch.ones(n_queries, query_len, device=device, dtype=torch.bool),
        )
    if ref_mask is None:
        ref_mask = cast(
            BoolTensor, torch.ones(n_queries, ref_len, device=device, dtype=torch.bool)
        )

    input_shape = (n_queries, query_len)
    # Masking query will cause numerical issues. We don't need to mask it anyway.
    # input_attention_mask = torch.ones(*input_shape, dtype=torch.bool, device=device)

    assert_eq(ref_mask.ndim, 2)
    assert_eq(query_mask.ndim, 2)
    assert_eq(ref_mask.size(0), query_mask.size(0))
    # combine input and ref attention masks
    attention_mask = torch.cat([ref_mask, query_mask], dim=1).unsqueeze(1)
    assert_eq(tuple(attention_mask.shape), (n_queries, 1, query_len + ref_len))

    ref_lens = ref_mask.sum(dim=1)[:, None]  # (n_queries, 1)
    # relative pos needs to be of shape (n_quries, query_len, ref_len + query_len)
    ref_pos = (torch.arange(ref_len, device=device, dtype=torch.long) - RefDistance)[
        None, :
    ]  # (1, ref_len)
    ref_pos = ref_pos.expand(n_queries, -1)
    query_pos = (
        torch.arange(query_len, device=device, dtype=torch.long)[None, :] + ref_lens
    )  # (n_queries, query_len)
    key_pos = torch.cat([ref_pos, query_pos], dim=1)  # (n_queries, ref_len + query_len)
    relative_pos = (
        key_pos[:, None, :] - query_pos[:, :, None]
    )  # (n_queries, query_len, ref_len + query_len)
    position_bias = compute_bias(attention_layer, relative_pos)
    check_nan("position_bias", position_bias, {})
    extended_attention_mask = _get_extended_attention_mask(
        attention_mask, cast(tuple, input_shape), position_bias.dtype
    )
    check_nan("extended_attention_mask", extended_attention_mask, {})
    position_bias = extended_attention_mask + position_bias
    check_nan("position_bias_after", position_bias, {})
    return position_bias


def query_ref_layer_batched(
    block: T5Block,
    hidden_states: Tensor,
    block_sizes: Sequence[int],
    cache: dict,
    RefDistance: int = 10000,
):
    """Compute the new hidden states using query-ref bidirectional sparse attention.
    The last block is the query."""
    st_layer = block.layer[0]
    assert isinstance(st_layer, T5LayerSelfAttention)
    ff_layer = block.layer[-1]
    assert isinstance(ff_layer, T5LayerFF)

    assert_eq(hidden_states.ndim, 3)

    device = hidden_states.device
    input_shape = hidden_states.shape

    def get_bias(query_len: int, ref_len: int, is_global: bool):
        bias = cache.get(("pbias", query_len, ref_len, is_global))
        if bias is None:
            bias = _get_position_bias(
                st_layer.SelfAttention,
                1,
                query_len,
                ref_len,
                device,
                RefDistance=RefDistance if is_global else -RefDistance,
            )
            cache[("pbias", query_len, ref_len, is_global)] = bias
        return bias

    normed_hidden_states = st_layer.layer_norm(hidden_states)
    atten_out = t5_sparse_attention(
        st_layer.SelfAttention,
        normed_hidden_states,
        block_sizes,
        get_bias,
    )
    hidden_states = hidden_states + st_layer.dropout(atten_out)
    hidden_states = _run_t5_ff(ff_layer, hidden_states)
    assert_eq(hidden_states.shape, input_shape)

    return hidden_states


def _encode_query_stack(
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
    assert input_ids[:, 0].ne(PAD_id).all(), "input_ids must be padded at only the end."

    assert stack.embed_tokens is not None
    inputs_embeds = stack.embed_tokens(input_ids)
    hidden_states = stack.dropout(inputs_embeds)

    attention_layer = cast(T5Block, stack.block[0]).layer[0].SelfAttention
    assert isinstance(attention_layer, T5Attention)

    position_bias = _get_position_bias(
        attention_layer,
        inputs_embeds.size(0),
        inputs_embeds.size(1),
        ref_hidden_states[0].size(1),
        inputs_embeds.device,
        query_mask=cast(BoolTensor, input_ids.ne(PAD_id)),
        ref_mask=ref_attention_mask,
        RefDistance=RefDistance,
    )

    for i, block in enumerate(stack.block):
        assert isinstance(block, T5Block)
        ref_states = ref_hidden_states[i]
        hidden_states = _encode_query_block(
            block,
            hidden_states,
            ref_states,
            position_bias=position_bias,
            output_attentions=False,
        )

        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
        # layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

        check_nan(
            "hidden_states",
            not_none(hidden_states),
            {"i": i, "input_hidden_states": hidden_states, "ref_states": ref_states},
        )
        # hidden_states, present_key_value_state = layer_outputs[:2]
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


def retrieval_cost_model(ref_size: int, query_size: int, output_size: int) -> float:
    a = 1 / 256
    return (
        a * (ref_size + query_size) * (query_size + output_size)
        + ref_size
        + query_size
        + 2 * output_size
    )


@dataclass
class BatchArgs:
    min_queries: int = 1
    max_queries: int = 8
    shuffle_extra_ids: bool = True

    @classmethod
    def train_default(cls) -> Self:
        return cls()

    @classmethod
    def eval_default(cls) -> Self:
        return BatchArgs(
            max_queries=32,
            shuffle_extra_ids=False,
        )

    @classmethod
    def service_default(cls) -> Self:
        args = BatchArgs.eval_default()
        return args


@dataclass
class C3DataLoader:
    all_probs: Sequence[C3Problem]
    transform: C3ProblemTransform | None
    tokenizer: C3ProblemTokenizer
    batch_args: BatchArgs
    shuffle: bool
    desc: str
    tqdm_args: dict | None = None
    chunk_size: int = 1000
    workers: int = DefaultWorkers

    def __post_init__(self):
        n_batches, batch_stats = self.estimate_batch_stats()
        self._len_est = n_batches
        self._batch_stast = batch_stats
        self.epochs = 0

    def __len__(self) -> int:
        return self._len_est

    def _to_tokenized(self, probs: Sequence[C3Problem]) -> Iterable[TkC3Problem]:
        probs = list(probs)
        if self.transform is not None:
            # we can afford to store all transformed problems beforehand
            probs = join_list(
                pmap(
                    self.transform.transform,
                    probs,
                    chunksize=500,
                    max_workers=self.workers,
                )
            )
        if self.shuffle:
            # we need to shuffle after the transform to help serialization
            # this also mixes the problems better
            random.shuffle(probs)
        for i in range(0, len(probs), self.chunk_size):
            # we can only afford to tokenize the problems on-the-fly
            group = probs[i : i + self.chunk_size]
            yield from pmap(
                self.tokenizer.tokenize_problem,
                group,
                tqdm_args={"disable": True},
                max_workers=self.workers,
            )

    def estimate_batch_stats(self):
        factor = 10
        n = max(1, len(self.all_probs) // factor)
        subset = random_subset(self.all_probs, n, rng=42)
        batches = self._problems_to_batches(self._to_tokenized(subset))
        bsizes = list[int]()
        for b in tqdm(batches, desc="estimate_batch_stats", smoothing=0.0):
            bsizes.append(len(b["input_ids"]))
        batch_stats = {k: f"{v:.1f}" for k, v in scalar_stats(bsizes).items()}
        # better to have a smaller estimate to avoid triggering data regeneration
        size_est = max(1, int(len(self.all_probs) / n * len(bsizes) * 0.99))
        return size_est, batch_stats

    def __iter__(self) -> Iterable[dict]:
        batches = self._problems_to_batches(self._to_tokenized(self.all_probs))
        tqdm_args = self.tqdm_args or {"smoothing": 0.0}
        for b in tqdm(
            batches,
            total=self._len_est,
            desc=self.desc + f" (epoch={self.epochs})",
            **tqdm_args,
        ):
            input_ids = pad_token_seqs(b["input_ids"])
            labels = pad_token_seqs(b["labels"], pad_id=-100)
            yield {
                "input_ids": input_ids,
                "references": b["references"],
                "query_ref_list": b["query_ref_list"],
                "labels": labels,
            }
        self.epochs += 1

    def _post_process(self, e: TkC3Problem):
        max_output_tks = self.tokenizer.max_output_tks
        shuffle_extra_ids = self.batch_args.shuffle_extra_ids
        labels = e.output_tks
        labels = wrap_bos(labels)

        if len(labels) > max_output_tks:
            labels = labels[:max_output_tks]

        input_ids = e.input_tks

        if shuffle_extra_ids and random.random() < 0.5:
            id_map = random_extra_id_map()
            input_ids = [id_map.get(tk, tk) for tk in input_ids]
            labels = [id_map.get(tk, tk) for tk in labels]

        return input_ids, labels

    def _cost_limit(self) -> float:
        min_queries = self.batch_args.min_queries
        tkn = self.tokenizer
        return min_queries * retrieval_cost_model(
            tkn.max_ref_tks_sum, tkn.max_query_tks, tkn.max_output_tks
        )

    @staticmethod
    def pack_batch(probs: Sequence[TkC3Problem]):
        assert probs, "Empty batch found"
        input_ids = [x.input_tks for x in probs]
        labels = [x.output_tks for x in probs]
        refs = [[y.tolist() for y in x.references] for x in probs]
        id2ref = {id(ref): ref for row in refs for ref in row}
        references = [id2ref[x] for x in id2ref]
        id2order = {x: i for i, x in enumerate(id2ref)}
        query_ref_list = [[id2order[id(ref)] for ref in row] for row in refs]
        return {
            "input_ids": input_ids,
            "references": references,
            "query_ref_list": query_ref_list,
            "labels": labels,
        }

    def _problems_to_batches(self, problems: Iterable[TkC3Problem]) -> Iterable[dict]:

        tkn = self.tokenizer
        cost_limit = self._cost_limit()
        warned_batch_size = False
        # sample references for each query
        current_batch = list[TkC3Problem]()
        current_cost = 0
        for tk_prob in problems:
            all_refs = [x[1] for x in tk_prob.named_references]
            ref_size_sum = sum(len(ref) for ref in all_refs)
            assert ref_size_sum <= tkn.max_ref_tks_sum, f"{ref_size_sum=}"
            input_tks, output_tks = self._post_process(tk_prob)
            cost = retrieval_cost_model(
                ref_size=ref_size_sum,
                query_size=len(input_tks),
                output_size=len(output_tks),
            )
            tk_prob = dataclasses.replace(
                tk_prob, input_tks=input_tks, output_tks=output_tks
            )
            if cost > cost_limit and not warned_batch_size:
                warned_batch_size = True
                warnings.warn("Batch cost limit is too small.")
            if (not current_batch) or (
                cost + current_cost <= cost_limit
                and len(current_batch) < self.batch_args.max_queries
            ):
                current_batch.append(tk_prob)
                current_cost += cost
            else:
                yield self.pack_batch(current_batch)
                current_batch = [tk_prob]
                current_cost = cost
        if current_batch:
            yield self.pack_batch(current_batch)


def pad_token_seqs(seqs: Sequence[TokenSeq], pad_id=None) -> LongTensor:
    assert seqs, "Cannot pad empty sequence."
    max_len = max(len(ref) for ref in seqs)
    if pad_id is None:
        pad_id = PAD_id
    rows = []
    for row in seqs:
        row = row + [pad_id] * (max_len - len(row))
        rows.append(row)
    return LongTensor(rows)


def _round_length_group(x: int) -> int:
    return math.ceil(x / 64) * 64
