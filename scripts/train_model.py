import copy
import multiprocessing
import os
import shutil
import warnings

import wandb

from coeditor._utils import cprint, run_long_task
from coeditor.c3problem import C3ProblemChangeInlining, C3ToCodeCompletion, TkC3Problem
from coeditor.common import *
from coeditor.dataset import (
    C3CombinedEncoder,
    C3ProblemDataset,
    make_or_load_dataset,
    make_or_load_transformed_dataset,
)
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
    dec_args = DecodingArgs()
    if quicktest:
        model_name = "quicktest-" + model_name

    if not eval_only:
        check_save_dir(model_name)

    # problems will be transformed and saved for valid and test but not train.
    datasets = make_or_load_dataset(
        dataset_name,
        encoder.change_processor,
        remake_problems=recreate_data,
        workers=multiprocessing.cpu_count(),
    )

    with timed_action("Making or loading transformed C3 problems for eval"):
        # it's important to cache these due to randomness in the transformations
        eval_probs = make_or_load_transformed_dataset(
            dataset_name,
            datasets,
            encoder,
            remake_problems=recreate_data,
            workers=multiprocessing.cpu_count(),
        )

    # limit the number of examples for faster testing
    datasets["valid"] = random_subset(eval_probs["valid"], 10000, rng=42)
    datasets["test"] = random_subset(eval_probs["test"], 10000, rng=42)

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

    valid_loader = C3DataLoader(
        datasets["valid"], None, eval_tkn, eval_batch_args, shuffle=False, desc="eval"
    )

    if not eval_only:
        # follow a 3-stage training pipeline
        scales = [4, 2, 1]
        for stage, scale in enumerate(scales):
            s_bargs = copy.deepcopy(batch_args)
            s_bargs.min_queries *= scale
            s_tkn = copy.copy(train_tkn)
            s_tkn.max_ref_tks_sum //= scale
            s_probs = [
                x
                for x in datasets["train"]
                if sum(c.change_size() for c in x.relevant_changes)
                <= s_tkn.max_ref_tks_sum
            ]
            s_probs = random_subset(s_probs, len(s_probs) // 4 * scale)
            s_loader = C3DataLoader(
                s_probs,
                encoder.problem_tranform,
                s_tkn,
                batch_args,
                filter=_not_truncated,
                shuffle=True,
                desc=f"stage {stage} training",
            )
            s_targs = copy.deepcopy(train_args)
            s_targs.learning_rate *= scale
            s_targs.max_train_epochs = 1
            with timed_action(f"stage {stage} training"):
                model.train_on_data(model_name, s_loader, valid_loader, s_targs)

    model.to("cuda")
    test_loader = C3DataLoader(
        datasets["test"], None, eval_tkn, eval_batch_args, shuffle=False, desc="test"
    )
    print(f"{len(test_loader)}")
    print(f"{len(test_loader.all_probs)}")
    with timed_action("Loss Evaluation"):
        eval_result = model.eval_loss_on_loader(test_loader)
        eval_dict = {f"test/{k}": v.average() for k, v in eval_result.items()}
        wandb.log(eval_dict)

    with timed_action("Accuracy Evaluation"):
        out_dir = get_model_dir() / model_name / "exact_match_samples"
        exact_acc = model.eval_on_data(
            datasets["test"],
            test_loader,
            dec_args,
            out_dir,
            probs_to_save=300,
        )
        print("Exact-match accuracy:", exact_acc)
        wandb.log({"test/exact-acc": exact_acc.average()})
        cprint("blue", "Exact-match samples saved to:", out_dir)

    return model


def _not_truncated(p: TkC3Problem) -> bool:
    return not p.truncated


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


def eval_code_completion():
    train_model(
        model_name="coeditor-xl-c3-completion-v1.6-resumed",
        dataset_name="tiny",
        encoder=C3CombinedEncoder(
            problem_tranform=C3ToCodeCompletion(),
        ),
        resumed_from=(get_model_dir(True) / "coeditor-xl-c3-dropout-v1.6-resumed"),
        eval_only=True,
    )


def train_new_model():
    train_model(
        model_name="coeditor-perm2k-c3-multi-v1.7.1",
        dataset_name="perm2k",
        train_args=TrainingArgs(
            max_train_epochs=1,
        ),
        encoder=C3CombinedEncoder(
            problem_tranform=C3ProblemChangeInlining(),
        ),
        recreate_data=False,
        quicktest=False,
    )


if __name__ == "__main__":
    os.chdir(proj_root())

    with run_long_task("train_model.py"):
        train_new_model()
