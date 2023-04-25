"""Evaluate Coeditor's exact match performance in a single-round editing setting.

This script generates the data for the ablation study.
"""

import os

from coeditor.c3problem import C3ProblemGenerator, C3ProblemTokenizer
from coeditor.common import *
from coeditor.dataset import make_or_load_dataset
from coeditor.model import BatchArgs, C3DataLoader, DecodingArgs, RetrievalEditorModel

os.chdir(proj_root())

dataset_name = "perm2k"
model_device = "cuda"

model_names = {
    "No Ablation": "coeditor-perm2k-c3-multi-v1.7.3",
    "No Diffs": "coeditor-perm2k-c3-multi-no_change-v1.7.3",
    "No Defs": "coeditor-perm2k-c3-multi-no_defs-v1.7.2",
    "Small Context": "coeditor-perm2k-c3-multi-2048-v1.7.2",
}

testset = make_or_load_dataset(
    dataset_name,
    C3ProblemGenerator(),
    splits=("test",),
    time_limit_per_commit=40,
)["test"]
# testset = random_subset(testset, 50, rng=42)
print(f"{len(testset)}")

accs = dict[str, float]()
for name, full_name in model_names.items():
    model = RetrievalEditorModel.load(get_model_dir() / full_name)
    model.to(model_device)

    with timed_action(f"Evaluating {name}"):
        out_dir = get_model_dir() / full_name / "exact_match_samples"
        eval_tkn = C3ProblemTokenizer.for_eval()
        if name == "Small Context":
            eval_tkn.max_ref_tks_sum = 2048
        eval_batch_args = BatchArgs.eval_default()
        test_loader = C3DataLoader(
            testset, None, eval_tkn, eval_batch_args, shuffle=False, desc="test"
        )
        exact_acc = model.eval_on_data(
            testset,
            test_loader,
            DecodingArgs(),
            out_dir,
            probs_to_save=300,
        )
        print("Exact-match accuracy:", exact_acc)
        cprint("blue", "Exact-match samples saved to:", out_dir)
        accs[name] = exact_acc.average()

    model.to("cpu")
    del model

pretty_print_dict(accs)
