from dataclasses import field
import torch

import wandb
from coeditor.dataset import TokenizedEditDataset
from coeditor.encoding import TokenizedEdit, WindowArgs, _Tokenizer, is_extra_id
from spot.model import dynamic_dataloader
from .common import *
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets.arrow_dataset import Dataset

CodeT5Model = T5ForConditionalGeneration


@dataclass
class DecodingArgs:
    base_tokens: int = 128
    tokens_per_line: int = 16
    do_sample: bool = False
    top_p: float = 0.9
    num_beams: Optional[int] = 8
    length_penalty: float = 1.0


@dataclass
class CoeditorModel:
    codet5: CodeT5Model

    def predict(
        self, input_tks: TokenSeq, decode_args: DecodingArgs | None = None
    ) -> TokenSeq:
        if decode_args is None:
            decode_args = DecodingArgs()
        x = torch.tensor([input_tks]).to(self.codet5.device)
        n_lines = sum(is_extra_id(tk) for tk in input_tks)
        max_length = decode_args.base_tokens + decode_args.tokens_per_line * n_lines
        output = self.codet5.generate(
            x,
            max_length,
            do_sample=decode_args.do_sample,
            top_p=decode_args.top_p,
            num_beams=decode_args.num_beams,
            length_penalty=decode_args.length_penalty,
        )[0]
        return output.tolist()

    def save_pretrained(self, path: Path):
        self.codet5.save_pretrained(path)

    def to(self, device):
        self.codet5.to(device)

    @staticmethod
    def load_pretrained(path: Path):
        codet5 = CodeT5Model.from_pretrained(path)
        assert isinstance(codet5, CodeT5Model)
        return CoeditorModel(codet5)

    @staticmethod
    def from_code_t5(use_small_model=False):
        path = (
            "Salesforce/codet5-small" if use_small_model else "Salesforce/codet5-base"
        )
        codet5 = CodeT5Model.from_pretrained(path)
        assert isinstance(codet5, CodeT5Model)
        codet5.resize_token_embeddings(len(_Tokenizer))
        return CoeditorModel(codet5)


@dataclass
class TrainingArgs:
    train_max_batch_tokens: int
    eval_max_batch_tokens: int
    train_window: WindowArgs
    eval_window: WindowArgs
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    quicktest: bool = False


def train_coeditor_model(
    model: CoeditorModel,
    training_name: str,
    train_data: TokenizedEditDataset,
    eval_data: TokenizedEditDataset,
    args: TrainingArgs,
):
    train_dir = get_model_dir(trained=False) / training_name

    wandb.init(dir=train_dir, project="Coeditor", name=training_name)

    trainer_args = Seq2SeqTrainingArguments(
        output_dir=str(train_dir),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=400,
        prediction_loss_only=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=1,
        fp16=True,
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to=["wandb"],
    )

    data_collator = DataCollatorForSeq2Seq(_Tokenizer)

    train_edits = list(train_data.all_edits())
    eval_edits = list(eval_data.all_edits())
    if args.quicktest:
        train_edits = train_edits[:10]
        eval_edits = eval_edits[:10]
    train_dataset = edits_to_dataset(
        [e.truncate_ctx(args.train_window) for e in train_edits]
    )
    eval_dataset = edits_to_dataset(
        [e.truncate_ctx(args.eval_window) for e in eval_edits]
    )
    train_lodader = dynamic_dataloader(
        train_dataset, args.train_max_batch_tokens, data_collator, shuffle=True
    )
    eval_loader = dynamic_dataloader(
        eval_dataset, args.eval_max_batch_tokens, data_collator, shuffle=False
    )

    class DynamicTrainer(Seq2SeqTrainer):
        def get_train_dataloader(self):
            return train_lodader

        def get_eval_dataloader(self, eval_dataset):
            return eval_loader

    trainer = DynamicTrainer(
        model.codet5,
        trainer_args,
    )

    trainer.train()
    save_dir = get_model_dir(trained=True) / training_name
    model.save_pretrained(save_dir)
    print("Model saved to:", save_dir)
    return model


def edits_to_dataset(edits: Sequence[TokenizedEdit]) -> Dataset:
    return Dataset.from_dict(
        {
            "input_ids": [e.input_tks for e in edits],
            "labels": [e.output_tks for e in edits],
        }
    )
