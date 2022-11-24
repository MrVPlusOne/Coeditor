from coeditor.common import *
from coeditor.dataset import *
from coeditor.encoding import (
    FileBasedEditEncoder,
    CstBasedEditEncoder,
    AnalysisBasedEditEncoder,
)
from spot.utils import pretty_print_dict

os.chdir(proj_root())

# dataset_name = "medium"
dataset_name = "SPOT"

window = WindowArgs(4096)
# encoder = FileBasedEditEncoder(window)
# encoder = CstBasedEditEncoder(window)
encoder = AnalysisBasedEditEncoder(
    window=window, extra_ctx_min_size=1000, extra_ctx_names=("usees", "post-usees")
)
save_dir = get_dataset_dir(dataset_name) / repr_modified_args(encoder)

if dataset_name == "SPOT":
    datasets = {"test": dataset_from_projects([proj_root()], encoder)}
else:
    datasets = datasets_from_repos(get_dataset_dir(dataset_name) / "repos", encoder)

display(datasets)

save_datasets(datasets, save_dir)
print("Tokenized dataset saved to:", save_dir)

for group, dataset in datasets.items():
    dataset = pickle_load(save_dir / f"{group}.pkl")
    print("=" * 20, group, "=" * 20)
    pretty_print_dict(dataset.overall_stats())
