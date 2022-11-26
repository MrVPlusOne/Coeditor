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
    recreate_data: bool = False,
) -> dict[str, TokenizedEditDataset]:
    window = WindowArgs(4096)
    # encoder = FileBasedEditEncoder(window)
    # encoder = CstBasedEditEncoder(window)
    encoder = AnalysisBasedEditEncoder(
        window=window, extra_ctx_size=1000, extra_ctx_names=("usees", "post-usees")
    )
    save_dir = get_dataset_dir(dataset_name) / repr_modified_args(encoder)

    if recreate_data or not save_dir.exists():
        if dataset_name == "SPOT":
            datasets = {"test": dataset_from_projects([proj_root()], encoder)}
        else:
            datasets = datasets_from_repos(
                get_dataset_dir(dataset_name) / "repos", encoder
            )
        save_datasets(datasets, save_dir)
        print("Tokenized dataset saved to:", save_dir)
        print("Dataset stats:")
        for group, dataset in datasets.items():
            print("=" * 20, group, "=" * 20)
            pretty_print_dict(dataset.overall_stats())
    else:
        datasets = load_datasets(save_dir)

    return datasets


if __name__ == "__main__":
    os.chdir(proj_root())

    # dataset_name = "SPOT"
    dataset_name = "medium"
    datasets = make_or_load_datasets(dataset_name, recreate_data=True)
