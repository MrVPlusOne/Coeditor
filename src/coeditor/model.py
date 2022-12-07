import copy
from dataclasses import field
import shutil
import torch

from coeditor.dataset import TokenizedEditDataset
from coeditor.encoding import (
    TokenizedEdit,
    _Tokenizer,
    extract_edit_change,
    get_extra_id,
    is_extra_id,
    BOS_id,
    EOS_id,
    random_extra_id_map,
)
from spot.data import output_ids_as_seqs
from spot.model import dynamic_dataloader, DataLoader
from spot.static_analysis import ProjectPath
from .common import *
from transformers.models.t5.modeling_t5 import (
    T5ForConditionalGeneration,
    Seq2SeqLMOutput,
)
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets.arrow_dataset import Dataset

CodeT5Model = T5ForConditionalGeneration


@dataclass
class DecodingArgs:
    base_tokens: int = 128
    tokens_per_line: int = 16
    max_output_tks: int = 512
    do_sample: bool = False
    top_p: float = 0.9
    num_beams: Optional[int] = 1
    length_penalty: float = 1.0


@dataclass
class TrainingArgs:
    max_batch_tokens: int
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    quicktest: bool = False


@dataclass
class EvalArgs:
    max_batch_tokens: int


@dataclass
class CoeditorModel:
    """
    args:
    - skip_unchanged: whether to skip <extra_id>s that are not followed by a
    change in the output sequence.
    """

    codet5: CodeT5Model
    data_args: "DataTransformArgs"

    @torch.autocast("cuda")
    def predict(
        self, input_tks: TokenSeq, decode_args: DecodingArgs | None = None
    ) -> TokenSeq:
        if decode_args is None:
            decode_args = DecodingArgs()
        x = torch.tensor([input_tks]).to(self.codet5.device)
        n_lines = sum(is_extra_id(tk) for tk in input_tks)
        max_length = decode_args.base_tokens + decode_args.tokens_per_line * n_lines
        max_length = min(max_length, decode_args.max_output_tks)
        output = self.codet5.generate(
            x,
            max_length,
            do_sample=decode_args.do_sample,
            top_p=decode_args.top_p,
            num_beams=decode_args.num_beams,
            length_penalty=decode_args.length_penalty,
        )[0]
        return output.tolist()

    def predict_on_batch(
        self,
        batch: dict,
        decode_args: DecodingArgs,
    ) -> list[TokenSeq]:
        x = batch["input_ids"].to(self.codet5.device)
        output = self.codet5.generate(
            x,
            max_length=decode_args.max_output_tks,
            do_sample=decode_args.do_sample,
            top_p=decode_args.top_p,
            num_beams=decode_args.num_beams,
            length_penalty=decode_args.length_penalty,
        )
        return [y.tolist() for y in output]

    @torch.autocast("cuda")
    def predict_on_loader(
        self,
        eval_loader: DataLoader,
        dec_args: DecodingArgs,
    ) -> dict[int, TokenSeq]:
        self.codet5.eval()
        predictions = dict[int, TokenSeq]()
        for batch in tqdm(eval_loader, desc="decoding", unit="batch"):
            ex_ids: list[int] = batch["ex_ids"].tolist()
            preds = self.predict_on_batch(batch, decode_args=dec_args)
            for ex_id, pred in zip(ex_ids, preds):
                predictions[ex_id] = pred
        return predictions

    def predict_on_data(
        self,
        eval_data: TokenizedEditDataset,
        eval_args: "EvalArgs",
        dec_args: DecodingArgs,
    ) -> "DatasetDecodingResult":
        eval_edits = eval_data.all_edits()
        dataset = edits_to_dataset(
            eval_edits,
            self.data_args,
            add_ex_id=True,
        )
        data_collator = DataCollatorForSeq2Seq(_Tokenizer)
        loader = dynamic_dataloader(
            dataset, eval_args.max_batch_tokens, data_collator, shuffle=True
        )
        pred_dict = self.predict_on_loader(loader, dec_args)
        pred_seq = [pred_dict[i] for i in range(len(eval_edits))]

        return DatasetDecodingResult(
            eval_args,
            dec_args,
            edits=eval_edits,
            input_ids=dataset["input_ids"],
            labels=dataset["labels"],
            predictions=pred_seq,
        )

    def save_pretrained(self, path: Path):
        pickle_dump(path / "data_transform_args.pkl", self.data_args)
        self.codet5.save_pretrained(path)

    def to(self, device):
        self.codet5.to(device)

    def train_on_data(
        self,
        training_name: str,
        train_data: TokenizedEditDataset,
        eval_data: TokenizedEditDataset,
        train_args: "TrainingArgs",
        eval_args: "EvalArgs",
    ) -> None:
        train_coeditor_model(
            self, training_name, train_data, eval_data, train_args, eval_args
        )

    def eval_loss_on_data(
        self,
        eval_data: TokenizedEditDataset,
        eval_args: "EvalArgs",
    ):
        eval_edits = eval_data.all_edits()
        eval_loader = edits_to_dataloader(
            eval_edits,
            eval_args.max_batch_tokens,
            self.data_args,
            shuffle=True,
        )
        return eval_label_likelihood(self, eval_loader)

    @staticmethod
    def load_pretrained(path: Path):
        codet5 = CodeT5Model.from_pretrained(path)
        assert isinstance(codet5, CodeT5Model)
        if (path / "data_transform_args.pkl").exists():
            dargs = pickle_load(path / "data_transform_args.pkl")
        else:
            dargs = DataTransformArgs()
        return CoeditorModel(codet5, data_args=dargs)

    @staticmethod
    def from_code_t5(data_args: "DataTransformArgs", use_small_model=False):
        path = (
            "Salesforce/codet5-small" if use_small_model else "Salesforce/codet5-base"
        )
        codet5 = CodeT5Model.from_pretrained(path)
        assert isinstance(codet5, CodeT5Model)
        codet5.resize_token_embeddings(len(_Tokenizer))
        return CoeditorModel(codet5, data_args=data_args)


@dataclass
class DatasetDecodingResult:
    eval_args: "EvalArgs"
    dec_args: DecodingArgs
    edits: list[TokenizedEdit]
    input_ids: list[TokenSeq]
    labels: list[TokenSeq]
    predictions: list[TokenSeq]

    def __post_init__(self):
        assert_eq(len(self.input_ids), len(self.predictions), len(self.labels))

    def subset(self, ids: Sequence[int]):
        return DatasetDecodingResult(
            self.eval_args,
            self.dec_args,
            [self.edits[i] for i in ids],
            [self.input_ids[i] for i in ids],
            [self.labels[i] for i in ids],
            [self.predictions[i] for i in ids],
        )

    def exact_match_accuracy(self) -> WeightedSum[int, int]:
        exact_match = WeightedSum(0, 0)
        for x, y, pred in zip(self.input_ids, self.labels, self.predictions):
            true_code = extract_edit_change(x, y).after
            pred_code = extract_edit_change(x, pred).after
            is_correct = normalize_code_by_ast(true_code) == normalize_code_by_ast(
                pred_code
            )
            exact_match += WeightedSum(int(is_correct), 1)
        return exact_match

    def save_examples_to_dir(
        self, out_dir: Path, ex_ids: Sequence[int], ctx_tks: int = 2000
    ):
        shutil.rmtree(out_dir, ignore_errors=True)
        (out_dir / "correct").mkdir(parents=True, exist_ok=True)
        (out_dir / "incorrect").mkdir(parents=True, exist_ok=True)

        for ex_id in tqdm(ex_ids, desc="saving examples"):
            pred_tks = self.predictions[ex_id]
            true_code = extract_edit_change(
                self.input_ids[ex_id], self.labels[ex_id]
            ).after
            pred_code = extract_edit_change(self.input_ids[ex_id], pred_tks).after
            is_correct = normalize_code_by_ast(true_code) == normalize_code_by_ast(
                pred_code
            )
            id_map = {
                k: get_extra_id(i)
                for i, k in enumerate(output_ids_as_seqs(self.input_ids[ex_id]))
            }
            pred_tks = [id_map.get(t, t) for t in pred_tks]
            compare_str = self.edits[ex_id].show_prediction(pred_tks, ctx_tks=ctx_tks)
            out_file = (
                out_dir / ("correct" if is_correct else "incorrect") / f"ex-{ex_id}.txt"
            )
            out_file.write_text(compare_str)


def train_coeditor_model(
    model: CoeditorModel,
    training_name: str,
    train_data: TokenizedEditDataset,
    eval_data: TokenizedEditDataset,
    train_args: TrainingArgs,
    eval_args: EvalArgs,
):
    train_dir = get_model_dir(trained=False) / training_name

    train_edits = train_data.all_edits()
    eval_edits = eval_data.all_edits()
    if train_args.quicktest:
        train_edits = train_edits[:10]
        eval_edits = eval_edits[:2]

    train_lodader = edits_to_dataloader(
        train_edits,
        train_args.max_batch_tokens,
        args=model.data_args,
        shuffle=True,
    )
    eval_loader = edits_to_dataloader(
        eval_edits,
        eval_args.max_batch_tokens,
        args=model.data_args,
        shuffle=True,
    )

    class DynamicTrainer(Seq2SeqTrainer):
        def get_train_dataloader(self):
            return train_lodader

        def get_eval_dataloader(self, eval_dataset):
            return eval_loader

    eval_interval = 6 * len(eval_loader)
    trainer_args = Seq2SeqTrainingArguments(
        output_dir=str(train_dir),
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_interval,
        logging_steps=eval_interval,
        save_steps=eval_interval,
        save_total_limit=3,
        prediction_loss_only=True,
        learning_rate=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
        num_train_epochs=4,
        fp16=True,
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to=["wandb"],
    )

    trainer = DynamicTrainer(
        model.codet5,
        trainer_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    save_dir = get_model_dir(trained=True) / training_name
    model.save_pretrained(save_dir)
    print("Model saved to:", save_dir)


@torch.no_grad()
@torch.autocast("cuda")
def eval_label_likelihood(
    model: CoeditorModel,
    eval_loader: DataLoader,
):
    core = model.codet5
    core.eval()
    loss_per_ex = WeightedSum(0.0, 0)
    loss_per_tk = WeightedSum(0.0, 0)
    prob_per_ex = WeightedSum(0.0, 0)
    for batch in tqdm(eval_loader, desc="evaluate loss", unit="batch"):
        input_ids = batch["input_ids"].to(core.device)
        labels = batch["labels"].to(core.device)
        attention_mask = batch["attention_mask"].to(core.device)
        outputs = core.forward(input_ids, labels=labels, attention_mask=attention_mask)
        assert isinstance(outputs, Seq2SeqLMOutput)
        logits = outputs.logits.permute(0, 2, 1)  # shape: (batch, vocab, seq_len)
        logp = torch.nn.functional.cross_entropy(
            logits,
            labels,
            reduction="none",
            ignore_index=-100,
        )  # shape: (batch, seq_len)
        loss = logp.sum().item()
        ex_prob = torch.exp(-logp.sum(dim=1)).sum().item()
        bsize = input_ids.shape[0]
        label_tks = (labels != -100).sum().item()
        loss_per_ex += WeightedSum(loss, bsize)
        loss_per_tk += WeightedSum(loss, label_tks)
        prob_per_ex += WeightedSum(ex_prob, bsize)

    return {
        "loss_per_ex": loss_per_ex,
        "loss_per_tk": loss_per_tk,
        "prob_per_ex": prob_per_ex,
    }


def code_tk_loss(logits: torch.FloatTensor, labels: torch.LongTensor):
    # This computation depends on the label sequence length, which may not be
    # suitable for comparing different encoding schemes.
    special_tks = {
        _Tokenizer.pad_token_id,
        _Tokenizer.eos_token_id,
        _Tokenizer.bos_token_id,
        -100,
    }

    def is_code_token(tk):
        return not is_extra_id(tk) and tk not in special_tks

    cpu_labels = cast(torch.LongTensor, labels.cpu())
    rows = logits.size(0)
    n_tokens = 0
    total_loss = 0.0
    for i in range(rows):
        selected = [is_code_token(tk) for tk in cpu_labels[i].tolist()]
        ce = torch.nn.functional.cross_entropy(
            logits[i, selected], labels[i, selected], reduction="sum"
        )
        n_tokens += sum(selected)
        total_loss += float(ce.item())

    return total_loss, n_tokens


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


@dataclass
class DataTransformArgs:
    skip_unchanged: bool = False
    shuffle_extra_ids: bool = False
    max_label_tks: int = 512


def edits_to_dataset(
    edits: Sequence[TokenizedEdit],
    args: DataTransformArgs,
    add_ex_id: bool = False,
) -> Dataset:
    def process_edit(e: TokenizedEdit):
        labels = e.output_tks
        if args.skip_unchanged:
            labels = drop_empty_labels(labels)
        labels = wrap_bos(labels)
        if len(labels) > args.max_label_tks:
            labels = labels[: args.max_label_tks]

        input_ids = e.input_tks

        if args.shuffle_extra_ids:
            id_map = random_extra_id_map()
            input_ids = [id_map.get(tk, tk) for tk in input_ids]
            labels = [id_map.get(tk, tk) for tk in labels]

        return input_ids, labels

    processed = [process_edit(e) for e in edits]
    d: dict[str, Any] = {
        "input_ids": [x[0] for x in processed],
        "labels": [x[1] for x in processed],
    }
    if add_ex_id:
        d["ex_ids"] = list(range(len(edits)))
    return Dataset.from_dict(d)


def edits_to_dataloader(
    edits: Sequence[TokenizedEdit],
    max_batch_tokens: int,
    args: DataTransformArgs,
    add_ex_id: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    dataset = edits_to_dataset(edits, args, add_ex_id)
    data_collator = DataCollatorForSeq2Seq(_Tokenizer)
    return dynamic_dataloader(dataset, max_batch_tokens, data_collator, shuffle=shuffle)
