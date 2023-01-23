import copy
import os
import warnings
from coeditor.dataset import C3EditEncoder
from coeditor.model import DecodingArgs
from spot.utils import run_long_task
from train_model import check_save_dir, TokenizedEditDataset

import wandb
from coeditor.common import *
from coeditor.encoders import QueryRefEditEncoder
from coeditor.retrieval_model import (
    AttentionMode,
    RetrievalEditorModel,
    BatchArgs,
    TrainingArgs,
    SchedulerType,
)
from prepare_data import make_or_load_datasets


def train_model(
    dataset_name="medium",
    model_variant="-sig-analysis-post_usees",
    encoder: C3EditEncoder = C3EditEncoder(),
    batch_args=BatchArgs.train_default(),
    test_batch_args=BatchArgs.eval_default(),
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

    datasets = make_or_load_datasets(dataset_name, encoder, recreate_data=recreate_data)

    config_dict = {
        k: get_modified_args(v)
        for k, v in {
            "data_args": batch_args,
            "train_args": train_args,
            "dec_args": dec_args,
        }.items()
    }

    project = "Coeditor" if not train_args.quicktest else "Coeditor-quicktest"
    wandb.init(dir="..", project=project, name=model_name, config=config_dict)

    if train_args.quicktest:
        print("Using fewer data for quick test.")
        n_quick_exs = 20
        for name, dataset in datasets.items():
            datasets[name] = TokenizedEditDataset.from_edits(
                dataset.all_edits()[:n_quick_exs]
            )

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

    if not eval_only:
        with timed_action("Warm-up Training"):
            warmup_bargs = copy.deepcopy(batch_args)
            warmup_bargs.max_total_ref_tks //= 4
            warmup_bargs.min_queires *= 4
            warmup_bargs.max_queries *= 2

            warmup_targs = copy.deepcopy(train_args)
            warmup_targs.learning_rate *= 4
            warmup_targs.max_train_epochs = 1
            all_edits = datasets["train"].all_edits()
            warmup_edits = random_subset(all_edits, len(all_edits) // 4)
            model.train_on_data(
                model_name,
                TokenizedEditDataset.from_edits(warmup_edits),
                datasets["valid"],
                warmup_targs,
                batch_args=warmup_bargs,
                eval_batch_args=test_batch_args,
            )
        with timed_action("Fine-tune Training"):
            model.train_on_data(
                model_name,
                datasets["train"],
                datasets["valid"],
                train_args,
                batch_args=batch_args,
                eval_batch_args=test_batch_args,
            )

    model.to("cuda")
    with timed_action("Loss Evaluation"):
        eval_result = model.eval_loss_on_data(datasets["test"], test_batch_args)
        eval_dict = {f"test/{k}": v.average() for k, v in eval_result.items()}
        wandb.log(eval_dict)

    max_saved_samples = 300

    with timed_action("Accuracy Evaluation"):
        dec_result = model.predict_on_data(datasets["test"], test_batch_args, dec_args)
        pickle_dump(get_model_dir() / model_name / "dec_result.pkl", dec_result)
        exact_acc, exact_correct_map = dec_result.exact_match_accuracy()
        wandb.log({"test/exact-acc": exact_acc.average()})

        out_dir = get_model_dir() / model_name / "exact_match_samples"
        dec_result.save_examples_to_dir(
            out_dir, random_subset(exact_correct_map, max_saved_samples)
        )
        print("Exact-match samples saved to:", out_dir)

    return model


if __name__ == "__main__":
    os.chdir(proj_root())
    with run_long_task("train_retrieval_model.py"):
        train_model(
            dataset_name="xl",
            model_variant="-c3-v1",
            train_args=TrainingArgs(
                max_train_epochs=1,
                quicktest=False,
            ),
            encoder=C3EditEncoder(),
            recreate_data=False,
            eval_only=False,
        )
