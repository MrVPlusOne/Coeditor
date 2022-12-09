from coeditor.common import *
from coeditor.dataset import *
from coeditor.encoding import (
    FileBasedEditEncoder,
    CstBasedEditEncoder,
    AnalysisBasedEditEncoder,
)
from spot.utils import pretty_print_dict


def make_or_load_datasets(
    dataset_name: str,
    encoder: EditEncoder[TEdit],
    predict_added_in_training: bool = True,
    recreate_data: bool = False,
) -> dict[str, TokenizedEditDataset[TEdit]]:
    tag = "" if predict_added_in_training else "no_added-"
    save_dir = get_dataset_dir(dataset_name) / (tag + repr_modified_args(encoder))

    if recreate_data or not save_dir.exists():
        if dataset_name == "SPOT":
            datasets = {
                "test": dataset_from_projects(
                    [proj_root()], encoder, [predict_added_in_training]
                )
            }
        else:
            datasets = datasets_from_repos(
                get_dataset_dir(dataset_name) / "repos",
                encoder,
                predict_added_in_training=predict_added_in_training,
            )
        with timed_action("Saving datasets to disk"):
            save_datasets(datasets, save_dir)
        print("Tokenized dataset saved to:", save_dir)
        print("Dataset stats:")
        for group, dataset in datasets.items():
            print("=" * 20, group, "=" * 20)
            pretty_print_dict(dataset.overall_stats())
    else:
        with timed_action("Loading datasets from disk"):
            datasets = load_datasets(save_dir)

    return datasets


if __name__ == "__main__":
    os.chdir(proj_root())

    # dataset_name = "SPOT"
    dataset_name = "large"
    encoders = [
        CstBasedEditEncoder(),
        AnalysisBasedEditEncoder(extra_ctx_names=("usees", "post-usees")),
    ]
    for encoder in encoders:
        with timed_action(f"Preparing dataset with encoder {encoder}"):
            make_or_load_datasets(
                dataset_name,
                encoder,
                predict_added_in_training=True,
            )
