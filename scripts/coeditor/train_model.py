import os
import random
from typing import *

import wandb
from coeditor.common import *
from coeditor.dataset import TokenizedEditDataset
from coeditor.encoding import AnalysisBasedEditEncoder, CstBasedEditEncoder, EditEncoder
from coeditor.model import *
from prepare_data import make_or_load_datasets
from spot.model import input_cost_model
from spot.utils import run_long_task


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


def train_model(
    dataset_name="medium",
    model_variant="-sig-analysis-post_usees",
    encoder: EditEncoder = AnalysisBasedEditEncoder(
        extra_ctx_names=("usees", "post-usees")
    ),
    drop_comments: bool = True,
    data_args=DataTransformArgs(shuffle_extra_ids=True),
    max_batch_tokens: int = 4100,
    recreate_data: bool = False,
    quicktest: bool = False,
):
    # model_variant = "-file"
    model_name = f"coeditor-{dataset_name}"
    model_name += model_variant

    train_args = TrainingArgs(
        max_batch_cost=input_cost_model(max_batch_tokens, data_args.max_label_tks),
        quicktest=quicktest,
    )
    valid_args = EvalArgs(
        max_batch_cost=2 * input_cost_model(max_batch_tokens, data_args.max_label_tks)
    )
    test_args = EvalArgs(
        max_batch_cost=2 * input_cost_model(max_batch_tokens, data_args.max_label_tks)
    )
    dec_args = DecodingArgs()
    if train_args.quicktest:
        model_name = "quicktest-" + model_name

    check_save_dir(model_name)

    datasets = make_or_load_datasets(
        dataset_name,
        encoder,
        drop_comments=drop_comments,
        recreate_data=recreate_data,
    )

    config_dict = {
        k: get_modified_args(v)
        for k, v in {
            "data_args": data_args,
            "train_args": train_args,
            "valid_args": valid_args,
            "test_args": test_args,
            "dec_args": dec_args,
        }.items()
    }

    project = "Coeditor" if not train_args.quicktest else "Coeditor-quicktest"
    wandb.init(dir="..", project=project, name=model_name, config=config_dict)

    if train_args.quicktest:
        print("Using fewer data for quick test.")
        for name, dataset in datasets.items():
            datasets[name] = TokenizedEditDataset.from_edits(dataset.all_edits()[:10])

    model = CoeditorModel.from_code_t5(data_args, reuse_embed=True)

    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        warnings.warn(
            "CUDA_VISIBLE_DEVICES not set, using 0. Note that "
            "the Huggingface Trainer will use all visible GPUs for training."
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with timed_action("Training"):
        model.train_on_data(
            model_name, datasets["train"], datasets["valid"], train_args, valid_args
        )

    with timed_action("Loss Evaluation"):
        eval_result = model.eval_loss_on_data(datasets["test"], test_args)
        eval_dict = {f"test/{k}": v.average() for k, v in eval_result.items()}
        wandb.log(eval_dict)

    max_saved_samples = 200

    with timed_action("Accuracy Evaluating"):
        dec_result = model.predict_on_data(datasets["test"], test_args, dec_args)
        pickle_dump(get_model_dir() / model_name / "dec_result.pkl", dec_result)
        exact_acc, exact_correct_map = dec_result.exact_match_accuracy()
        wandb.log({"test/exact-acc": exact_acc.average()})

        out_dir = get_model_dir() / model_name / "exact_match_samples"
        dec_result.save_examples_to_dir(
            out_dir, random_subset(exact_correct_map, max_saved_samples)
        )
        print("Exact-match samples saved to:", out_dir)

        if isinstance(encoder, AnalysisBasedEditEncoder):
            call_acc, call_correct_map = dec_result.call_update_accuracy()
            wandb.log({"test/call-update-acc": call_acc.average()})
            out_dir = get_model_dir() / model_name / "call_update_samples"
            dec_result.save_examples_to_dir(
                out_dir, random_subset(call_correct_map, max_saved_samples)
            )
            print("Call-update samples saved to:", out_dir)
    return model


if __name__ == "__main__":
    os.chdir(proj_root())
    with run_long_task("train_model.py"):
        train_model(
            dataset_name="large",
            model_variant="-analysis-post_usees-reuse",
            # encoder=CstBasedEditEncoder(),
            encoder=AnalysisBasedEditEncoder(
                extra_ctx_names=("usees", "post-usees"), add_truncate_bos=False
            ),
            recreate_data=False,
            quicktest=False,
        )
