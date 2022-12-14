import copy
import os
import warnings
from train_model import check_save_dir, TokenizedEditDataset

import wandb
from coeditor.common import *
from coeditor.encoders import BasicQueryEditEncoder
from coeditor.retrieval_model import RetrievalEditorModel, BatchArgs, TrainingArgs
from prepare_data import make_or_load_datasets


def train_model(
    dataset_name="medium",
    model_variant="-sig-analysis-post_usees",
    encoder: BasicQueryEditEncoder = BasicQueryEditEncoder(),
    batch_args=BatchArgs(),
    train_args=TrainingArgs(),
    recreate_data: bool = False,
):
    # model_variant = "-file"
    model_name = f"coeditor-{dataset_name}"
    model_name += model_variant

    # dec_args = DecodingArgs()
    if train_args.quicktest:
        model_name = "quicktest-" + model_name

    check_save_dir(model_name)

    datasets = make_or_load_datasets(
        dataset_name,
        encoder,
        recreate_data=recreate_data,
        predict_added_in_training=not batch_args.use_only_modified,
    )

    config_dict = {
        k: get_modified_args(v)
        for k, v in {
            "data_args": batch_args,
            "train_args": train_args,
            # "dec_args": dec_args,
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

    model = RetrievalEditorModel.from_code_t5("base", reuse_embed=True)
    model.query_attened_ref = True

    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        warnings.warn(
            "CUDA_VISIBLE_DEVICES not set, using 0. Note that "
            "the Huggingface Trainer will use all visible GPUs for training."
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with timed_action("Training"):
        model.train_on_data(
            model_name,
            datasets["train"],
            datasets["valid"],
            train_args,
            batch_args=batch_args,
        )

    with timed_action("Loss Evaluating"):
        test_batch_args = copy.deepcopy(batch_args)
        test_batch_args.min_queires *= 2
        test_batch_args.max_queries *= 2
        eval_result = model.eval_loss_on_data(datasets["test"], test_batch_args)
        eval_dict = {f"test/{k}": v.average() for k, v in eval_result.items()}
        wandb.log(eval_dict)

    # max_saved_samples = 200

    # with timed_action("Accuracy Evaluating"):
    #     dec_result = model.predict_on_data(datasets["test"], test_args, dec_args)
    #     pickle_dump(get_model_dir() / model_name / "dec_result.pkl", dec_result)
    #     exact_acc, exact_correct_map = dec_result.exact_match_accuracy()
    #     wandb.log({"test/exact-acc": exact_acc.average()})

    #     out_dir = get_model_dir() / model_name / "exact_match_samples"
    #     dec_result.save_examples_to_dir(
    #         out_dir, random_subset(exact_correct_map, max_saved_samples)
    #     )
    #     print("Exact-match samples saved to:", out_dir)

    #     if isinstance(encoder, AnalysisBasedEditEncoder):
    #         call_acc, call_correct_map = dec_result.call_update_accuracy()
    #         wandb.log({"test/call-update-acc": call_acc.average()})
    #         out_dir = get_model_dir() / model_name / "call_update_samples"
    #         dec_result.save_examples_to_dir(
    #             out_dir, random_subset(call_correct_map, max_saved_samples)
    #         )
    #         print("Call-update samples saved to:", out_dir)
    return model


if __name__ == "__main__":
    os.chdir(proj_root())
    train_model(
        dataset_name="large",
        model_variant="-query-basic",
        batch_args=BatchArgs(),
        train_args=TrainingArgs(max_train_epochs=4, quicktest=False),
        encoder=BasicQueryEditEncoder(),
        recreate_data=False,
    )
