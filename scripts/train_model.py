import copy
import os
import shutil
import warnings

import wandb
from prepare_data import make_or_load_dataset

from coeditor._utils import cprint, run_long_task
from coeditor.c3problem import C3ProblemChangeDropout
from coeditor.common import *
from coeditor.dataset import C3CombinedEncoder, C3ProblemDataset
from coeditor.model import (
    BatchArgs,
    C3DataLoader,
    DecodingArgs,
    RetrievalEditorModel,
    TrainingArgs,
)


def train_model(
    model_name: str,
    dataset_name: str,
    encoder: C3CombinedEncoder = C3CombinedEncoder(),
    batch_args=BatchArgs.train_default(),
    eval_batch_args=BatchArgs.eval_default(),
    train_args=TrainingArgs(),
    recreate_data: bool = False,
    resumed_from: Path | None = None,
    eval_only: bool = False,
    quicktest: bool = False,
):
    assert dataset_name in model_name, "Model name should contain dataset name."

    dec_args = DecodingArgs()
    if quicktest:
        model_name = "quicktest-" + model_name

    if not eval_only:
        check_save_dir(model_name)

    # problems will be transformed and saved for valid and test but not train.
    datasets = make_or_load_dataset(
        dataset_name,
        encoder.change_processor,
        encoder.problem_tranform,
        remake_problems=recreate_data,
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

    project = "Coeditor" if not quicktest else "Coeditor-quicktest"
    if eval_only:
        project = "eval-" + project
    wandb.init(dir="..", project=project, name=model_name, config=config_dict)

    if quicktest:
        print("Using fewer data for quick test.")
        n_quick_exs = 20
        datasets = C3ProblemDataset(
            train=datasets["train"][:n_quick_exs],
            valid=datasets["valid"][:n_quick_exs],
            test=datasets["test"][:n_quick_exs],
        )

    if resumed_from is None:
        model = RetrievalEditorModel.from_code_t5("base", reuse_embed=True)
    else:
        model = RetrievalEditorModel.load(resumed_from)

    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        warnings.warn(
            "CUDA_VISIBLE_DEVICES not set, using 0. Note that "
            "the Huggingface Trainer will use all visible GPUs for training."
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_tkn = encoder.edit_tokenizer
    eval_tkn = copy.deepcopy(train_tkn)
    eval_tkn.max_query_tks *= 2
    eval_tkn.max_output_tks *= 2
    eval_tkn.max_ref_tks_sum *= 2

    eval_loader = C3DataLoader(
        datasets["valid"], None, eval_tkn, eval_batch_args, shuffle=False, desc="eval"
    )

    if not eval_only and resumed_from is None:
        with timed_action("Warm-up Training"):
            warmup_bargs = copy.deepcopy(batch_args)
            warmup_bargs.min_queries *= 4
            warmup_bargs.max_queries *= 2

            warm_up_data = random_subset(
                datasets["train"], len(datasets["train"]) // 4, rng=42
            )
            warmup_tkn = copy.copy(train_tkn)
            warmup_tkn.max_ref_tks_sum //= 3
            warmup_loader = C3DataLoader(
                warm_up_data,
                encoder.problem_tranform,
                warmup_tkn,
                warmup_bargs,
                shuffle=True,
                desc="warm-up training",
            )
            print("Warmup batch stats:")
            pprint(warmup_loader.get_batch_stats())

            warmup_targs = copy.deepcopy(train_args)
            warmup_targs.learning_rate *= 4
            warmup_targs.max_train_epochs = 1
            model.train_on_data(model_name, warmup_loader, eval_loader, warmup_targs)

    if not eval_only:
        with timed_action("Fine-tune Training"):
            # we attach the problem transform to the dataloader to generate data on-the-fly
            train_loader = C3DataLoader(
                datasets["train"],
                encoder.problem_tranform,
                train_tkn,
                batch_args,
                shuffle=True,
                desc="training",
            )
            print("Fine-tune batch stats:")
            pprint(train_loader.get_batch_stats())
            model.train_on_data(model_name, train_loader, eval_loader, train_args)

    model.to("cuda")
    with timed_action("Loss Evaluation"):
        eval_result = model.eval_loss_on_loader(eval_loader)
        eval_dict = {f"test/{k}": v.average() for k, v in eval_result.items()}
        wandb.log(eval_dict)

    with timed_action("Accuracy Evaluation"):
        out_dir = get_model_dir() / model_name / "exact_match_samples"
        exact_acc = model.eval_on_data(
            datasets["test"],
            eval_tkn,
            eval_batch_args,
            dec_args,
            out_dir,
            probs_to_save=300,
        )
        print("Exact-match accuracy:", exact_acc)
        wandb.log({"test/exact-acc": exact_acc.average()})
        cprint("blue", "Exact-match samples saved to:", out_dir)

    return model


def check_save_dir(model_name: str) -> None:
    "Prompt user to remove existing training directory or abort."
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
            model_name="coeditor-xl-c3-dropout-v1.6-resumed",
            dataset_name="xl",
            train_args=TrainingArgs(
                max_train_epochs=1,
            ),
            encoder=C3CombinedEncoder(
                problem_tranform=C3ProblemChangeDropout(),
            ),
            resumed_from=(
                get_model_dir(False) / "coeditor-xl-c3-dropout-v1.6/checkpoint-125000"
            ),
            eval_only=False,
            recreate_data=False,
            quicktest=False,
        )
