"""Evaluate Coeditor's exact match performance in a single-round editing setting.

This script generates the results for the ablation studies.
"""

import os

import numpy as np

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


# we load the older dataset format since the models above were trained on it.
testset = pickle_load(
    get_dataset_dir("perm2k") / "processed" / "valid-C3ProblemGenerator(VERSION=2.9)"
)

# # uncomment below to load with the newest dataset format
# testset = make_or_load_dataset(
#     dataset_name,
#     C3ProblemGenerator(),
#     splits=("valid",),
#     time_limit_per_commit=40,
# )["valid"]

# testset = random_subset(testset, 50, rng=42)
print(f"{len(testset)=}")

accs = dict[str, dict]()
results = dict[str, list[bool]]()
for name, full_name in model_names.items():
    if "checkpoint" in full_name:
        model_path = get_model_dir(False) / full_name
    else:
        model_path = get_model_dir() / full_name
    model = RetrievalEditorModel.load(model_path)
    model.to(model_device)

    out_dir = get_model_dir() / full_name / "exact_match_samples"
    eval_tkn = C3ProblemTokenizer.for_eval()
    if name == "Small Context":
        eval_tkn.max_ref_tks_sum = 2048
    eval_batch_args = BatchArgs.eval_default()

    with timed_action(f"Evaluating {name}"):
        test_loader = C3DataLoader(
            testset, None, eval_tkn, eval_batch_args, shuffle=False, desc="evaluating"
        )
        correctness = model.eval_on_data(
            testset,
            test_loader,
            DecodingArgs(),
            out_dir,
            probs_to_save=300,
        )
        results[name] = correctness
        exact_acc = float(np.mean(correctness))
        lb, ub = bootstrap_sample(list(map(float, correctness)))
        print("Exact-match accuracy:", exact_acc)
        print(f"95% CI: [{lb:.4f}, {ub:.4f}]")
        cprint("blue", "Exact-match samples saved to:", out_dir)
        accs[name] = {"mean": exact_acc, "lb": lb, "ub": ub}

    model.to("cpu")
    del model

pretty_print_dict(accs)
pickle_dump(Path("output/single_round_eval-accs.pkl"), accs)
pickle_dump(Path("output/single_round_eval-results.pkl"), results)

baseline_perf = results["No Ablation"]
for name in ["No Diffs", "No Defs", "Small Context"]:
    this_perf = results[name]
    pvalue = bootstrap_compare(this_perf, baseline_perf)
    print(f"(vs. No Ablation) {name} p-value: {pvalue:.4f}")
