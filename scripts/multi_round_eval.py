"""Evaluate Coeditor's performance in a multi-round editing setting."""

import os

from coeditor.c3problem import C3ProblemGenerator, C3ProblemTokenizer
from coeditor.common import *
from coeditor.dataset import make_or_load_dataset
from coeditor.model import (
    DecodingArgs,
    MultiRoundEvaluator,
    MultiRoundStrategy,
    RetrievalEditorModel,
)

os.chdir(proj_root())

dataset_name = "perm2k"
N_test = 5000  # number of test examples to evaluate
# NOTE: You can change the `model_name`` below to a `Path` to load a local model.
model_name = "MrVPlusOne/coeditor-perm2k-base-v1.7.3"
model_device = "cuda"

# %%
testset = make_or_load_dataset(
    dataset_name,
    C3ProblemGenerator(),
    splits=("test",),
    time_limit_per_commit=40,
)["test"]

print(f"{len(testset)}")
subset = random_subset(testset, N_test, rng=42)
print(f"{len(subset)=}")


# %%
tokenizer = C3ProblemTokenizer.for_eval()
dec_args = DecodingArgs(do_sample=False, num_beams=1)
model = RetrievalEditorModel.load(model_name)
model.to(model_device)

strategies: list[MultiRoundStrategy] = ["pick_first", "most_uncertain"]
for strategy in strategies:
    evaluator = MultiRoundEvaluator(model, tokenizer, dec_args, strategy=strategy)
    metric_stats = [
        evaluator.multi_round_edit_gain(ex, print_steps=False)
        for ex in tqdm(subset, smoothing=0.0)
    ]

    print("=" * 100)
    print("Prompting strategy:", strategy)
    target_file = (
        proj_root() / f"output/multi_round_eval/{model_name}/{strategy}-{N_test}.pkl"
    )
    pickle_dump(target_file, metric_stats)
    for cm in evaluator.cost_models:
        cm_name = cm.name
        print(SEP)
        print("Cost model:", cm_name)
        stats = [s[cm_name] for s in metric_stats]

        keys = ["label_edit_gain", "first_edit_gain", "total_edit_gain", "rounds"]
        mean_stats = {k: scalar_stats([getattr(s, k) for s in stats]) for k in keys}
        pretty_print_dict(mean_stats)

        print(f"For all edits (n={len(stats)}):")
        label_sum = sum(s.label_edit_gain for s in stats)
        single_sum = sum(s.first_edit_gain for s in stats)
        multi_sum = sum(s.total_edit_gain for s in stats)
        print(f"Single-round Gain ratio: {single_sum / label_sum:.2%}")
        print(f"Multi-round Gain ratio: {multi_sum / label_sum:.2%}")
