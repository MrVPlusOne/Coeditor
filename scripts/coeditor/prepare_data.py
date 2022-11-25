from coeditor.common import *
from coeditor.dataset import *
from coeditor.encoding import (
    FileBasedEditEncoder,
    CstBasedEditEncoder,
    AnalysisBasedEditEncoder,
)
from spot.utils import pretty_print_dict

os.chdir(proj_root())

recreate_data = True

dataset_name = "medium"
# dataset_name = "SPOT"

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
        datasets = datasets_from_repos(get_dataset_dir(dataset_name) / "repos", encoder)
    save_datasets(datasets, save_dir)
    print("Tokenized dataset saved to:", save_dir)
    print("Dataset stats:")
    for group, dataset in datasets.items():
        print("=" * 20, group, "=" * 20)
        pretty_print_dict(dataset.overall_stats())
else:
    with timed_action(f"Load dataset from: {save_dir}"):
        datasets = load_datasets(save_dir)

display(datasets)
