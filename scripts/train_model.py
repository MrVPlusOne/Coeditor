import copy
import os
import shutil
import warnings

import wandb
from prepare_data import make_or_load_datasets

from coeditor._utils import cprint, run_long_task
from coeditor.c3problem import C3Problem
from coeditor.common import *
from coeditor.dataset import C3EditEncoder
from coeditor.model import (
    BatchArgs,
    C3DataLoader,
    DecodingArgs,
    RetrievalEditorModel,
    TrainingArgs,
)


def train_model(
    dataset_name="medium",
    model_variant="-sig-analysis-post_usees",
    encoder: C3EditEncoder = C3EditEncoder(),
    batch_args=BatchArgs.train_default(),
    eval_batch_args=BatchArgs.eval_default(),
    train_args=TrainingArgs(),
    recreate_data: bool = False,
    eval_only: bool = False,
):
    # model_variant = "-file"
    model_name = f"coeditor-{dataset_name}"
    model_name += model_variant

    dec_args = DecodingArgs()
    if train_args.quicktest:
        model_name = "quicktest-" + model_name

    if not eval_only:
        check_save_dir(model_name)

    datasets = make_or_load_datasets(
        dataset_name, encoder.change_processor, recreate_data=recreate_data
    )

    config_dict = {
        k: get_modified_args(v)
        for k, v in {
            "edit_tokenizer": encoder.edit_tokenizer.get_args(),
            "batch_args": batch_args,
            "train_args": train_args,
            "dec_args": dec_args,
        }.items()
    }

    project = "Coeditor" if not train_args.quicktest else "Coeditor-quicktest"
    if eval_only:
        project = "eval-" + project
    wandb.init(dir="..", project=project, name=model_name, config=config_dict)

    if train_args.quicktest:
        print("Using fewer data for quick test.")
        n_quick_exs = 20
        datasets = {name: data[:n_quick_exs] for name, data in datasets.items()}

    if not eval_only:
        model = RetrievalEditorModel.from_code_t5(
            "base", reuse_embed=True, reinit_weights=train_args.reinit_weights
        )
    else:
        model = RetrievalEditorModel.load(get_model_dir() / model_name)

    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        warnings.warn(
            "CUDA_VISIBLE_DEVICES not set, using 0. Note that "
            "the Huggingface Trainer will use all visible GPUs for training."
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def transform_data(data: Sequence[C3Problem]) -> list[C3Problem]:
        transformed = pmap(encoder.problem_tranformer.transform, data, chunksize=1000)
        return join_list(transformed)

    datasets = {split: transform_data(data) for split, data in datasets.items()}

    train_tkn = encoder.edit_tokenizer
    eval_tkn = copy.deepcopy(train_tkn)
    eval_tkn.max_ref_tks_sum *= 2
    eval_loader = C3DataLoader(
        datasets["valid"], eval_tkn, eval_batch_args, shuffle=False, desc="eval"
    )

    if not eval_only:
        train_loader = C3DataLoader(
            datasets["train"], train_tkn, batch_args, shuffle=True, desc="training"
        )

        with timed_action("Warm-up Training"):
            warmup_bargs = copy.deepcopy(batch_args)
            warmup_bargs.min_queries *= 4
            warmup_bargs.max_queries *= 2

            warm_up_data = random_subset(datasets["train"], len(datasets["train"]) // 4)
            warmup_tkn = copy.deepcopy(train_tkn)
            warmup_tkn.max_ref_tks_sum //= 3
            warmup_loader = C3DataLoader(
                warm_up_data,
                warmup_tkn,
                warmup_bargs,
                shuffle=True,
                desc="warm-up training",
            )

            warmup_targs = copy.deepcopy(train_args)
            warmup_targs.learning_rate *= 4
            warmup_targs.max_train_epochs = 1
            model.train_on_data(model_name, warmup_loader, eval_loader, warmup_targs)
        with timed_action("Fine-tune Training"):
            model.train_on_data(model_name, train_loader, eval_loader, train_args)

    model.to("cuda")
    with timed_action("Loss Evaluation"):
        eval_result = model.eval_loss_on_loader(eval_loader)
        eval_dict = {f"test/{k}": v.average() for k, v in eval_result.items()}
        wandb.log(eval_dict)

    max_saved_samples = 300

    with timed_action("Accuracy Evaluation"):
        dec_result = model.predict_on_data(
            datasets["test"], eval_tkn, eval_batch_args, dec_args
        )
        pickle_dump(get_model_dir() / model_name / "dec_result.pkl", dec_result)
        exact_acc, exact_correct_map = dec_result.exact_match_accuracy()
        wandb.log({"test/exact-acc": exact_acc.average()})

        out_dir = get_model_dir() / model_name / "exact_match_samples"
        dec_result.save_examples_to_dir(
            out_dir, random_subset(exact_correct_map, max_saved_samples)
        )
        cprint("blue", "Exact-match samples saved to:", out_dir)

    return model


def check_save_dir(model_name: str) -> None:
    training_dir = get_model_dir(False) / model_name
    trained_dir = get_model_dir(True) / model_name
    if training_dir.exists():
        print(f"Training directory already exists:", training_dir)
        answer = input("Remove and retrain? (y/n):")
        if answer.lower().strip() == "y":
            shutil.rmtree(training_dir)
            return
        else:
            print("Training aborted.")
            exit(1)
    if trained_dir.exists():
        print(f"Saved model already exists:", trained_dir)
        answer = input("Model will be overriden at the end. Continue? (y/n):")
        if answer.lower().strip() != "y":
            print("Training aborted.")
            exit(1)


if __name__ == "__main__":
    os.chdir(proj_root())
    with run_long_task("train_model.py"):
        train_model(
            dataset_name="xl",
            model_variant="-c3-v1.3",
            train_args=TrainingArgs(
                max_train_epochs=1,
                quicktest=False,
            ),
            encoder=C3EditEncoder(),
            recreate_data=False,
            eval_only=False,
        )
