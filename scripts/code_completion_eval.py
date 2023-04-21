"""This script compares the performance of Coeditor against code completion models
on FIM problems extracted from code changes."""

import os
import shutil

import torch
from numpy import mean

from coeditor.c3problem import (
    C3ProblemGenerator,
    C3ProblemTokenizer,
    C3ToCodeCompletion,
    CompletionKind,
)
from coeditor.common import *
from coeditor.dataset import make_or_load_dataset
from coeditor.encoding import inline_output_tokens, tokens_to_change
from coeditor.experiments.code_completion import (
    C3CompletionGenerator,
    CodeT5Wrapper,
    FIMModel,
    infill_with_coeditor,
)
from coeditor.experiments.in_coder import InCoderWrapper
from coeditor.experiments.santa_coder import SantaCoderWrapper
from coeditor.model import RetrievalEditorModel

os.chdir(proj_root())

dataset_name = "perm2k"
device = "cuda"
N_test = 5000
use_additions = True
use_modifications = True

# first, load the test data, in FIM format
fim_probs = make_or_load_dataset(
    dataset_name,
    C3CompletionGenerator(
        use_additions=use_additions, use_modifications=use_modifications
    ),
    splits=("test",),
    time_limit_per_commit=20,
    remake_problems=False,
)["test"]
print(f"{len(fim_probs) = }")

# and in C3 format
c3_probs = make_or_load_dataset(
    dataset_name,
    C3ProblemGenerator(),
    splits=("test",),
    time_limit_per_commit=40,
)["test"]
transform = C3ToCodeCompletion(
    use_additions=use_additions, use_modifications=use_modifications
)
c3_probs = join_list(transform.transform(p) for p in c3_probs)
print(f"{len(c3_probs) = }")

# down-sample problems
fim_probs = random_subset(fim_probs, N_test, rng=42)
c3_probs = random_subset(c3_probs, N_test, rng=42)

sample_ids = set(random_subset(range(len(fim_probs)), 100, rng=73))
sample_dir = proj_root() / "output" / "code_completion_eval"
if sample_dir.exists():
    shutil.rmtree(sample_dir)

ModelName = str
accuracies = dict[ModelName, dict[str, float]]()


def get_accs(results: dict[CompletionKind, list[bool]]) -> dict[str, float]:
    """Get the accuracy of the model on additions, modifications, and all problems."""
    return {
        "add": float(mean(results["add"])),
        "mod": float(mean(results["mod"])),
        "all": float(mean(results["add"] + results["mod"])),
    }


# %% Evaluate the performance of Coeditor
coeditor = RetrievalEditorModel.load("MrVPlusOne/coeditor-perm2k-base-v1.7.3")
coeditor.half()
coeditor.to("cuda")
tknizer = C3ProblemTokenizer.for_eval()
coeditor_results: dict[CompletionKind, list[bool]] = {"add": [], "mod": []}
add_results = list[bool]()
replace_results = list[bool]()
for i, prob in tqdm(list(enumerate(c3_probs)), smoothing=0, desc="Evaluating Coeditor"):
    tk_prob = tknizer.tokenize_problem(prob)
    output = infill_with_coeditor(coeditor, tk_prob)
    pred_code = tokens_to_change(inline_output_tokens(tk_prob.main_tks, output)).after
    label_code = tokens_to_change(
        inline_output_tokens(tk_prob.main_tks, tk_prob.output_tks)
    ).after
    correct = code_equal(pred_code, label_code)
    if "add" in prob.transformations:
        kind = "add"
    else:
        assert "mod" in prob.transformations
        kind = "mod"
    coeditor_results[kind].append(correct)

    if i in sample_ids:
        ex_dir = sample_dir / f"ex{i}"
        ex_dir.mkdir(parents=True, exist_ok=True)
        (ex_dir / "Coeditor-base.txt").write_text(tk_prob.show(output))

accuracies["Coeditor-base"] = get_accs(coeditor_results)
print("Coeditor-base accuracy:")
pretty_print_dict(accuracies["Coeditor-base"])
coeditor.to("cpu")

# %% Evaluate the performance of FIM models
santa_coder = SantaCoderWrapper.from_pretrained()
incoder = InCoderWrapper.from_pretrained("facebook/incoder-1B", half_precision=True)
incoder6B = InCoderWrapper.from_pretrained("facebook/incoder-6B", half_precision=True)

# CodeT5 is trained on very short spans, which makes it unsuitable for this
# task without fine-tuning.
# codet5 = CodeT5Wrapper.from_pretrained("Salesforce/codet5-large")
# codet5.model.half()


fim_models: dict[str, FIMModel] = {
    "SantaCoder": santa_coder,
    "InCoder-1B": incoder,
    "InCoder-6B": incoder6B,
    # "CodeT5-large": codet5,
}
for name, model in fim_models.items():
    with run_long_task(f"Evaluating {name}"):
        model.model.to(device)

        results: dict[CompletionKind, list[bool]] = {"add": [], "mod": []}
        for i, prob in tqdm(
            list(enumerate(fim_probs)), smoothing=0, desc=f"Evaluating {name}"
        ):
            left_ctx = "\n".join(prob.left_ctx) + "\n"
            right_ctx = "\n" + "\n".join(prob.right_ctx)
            with torch.no_grad():
                pred = model.infill(left_ctx, right_ctx, max_length=128)
            if pred:
                pred = pred.split("\n")[0]  # only keep the first predicted line
            left_part = prob.left_ctx[-1] + "\n" if prob.left_ctx else ""
            right_part = "\n" + prob.right_ctx[0] if prob.right_ctx else ""
            pred_code = left_part + pred + right_part
            label_code = left_part + prob.middle + right_part
            correct = code_equal(pred_code, label_code)
            results[prob.kind].append(correct)
            if i in sample_ids:
                ex_dir = sample_dir / f"ex{i}"
                ex_dir.mkdir(parents=True, exist_ok=True)
                pred_str = (
                    f"prediction:\n{pred}\n{SEP}\nlabel:\n{prob.middle}\n"
                    f"{SEP}\nleft context:\n{left_ctx}\n{SEP}\n"
                    f"right context:\n{right_ctx}"
                )
                (ex_dir / f"{name}.txt").write_text(pred_str)

        accuracies[name] = acc = get_accs(results)
        print(f"{name} accuracy:")
        pretty_print_dict(acc)
        model.model.to("cpu")


print(SEP)
print(f"Summary ({use_additions=}, {use_modifications=}):")
for model, acc in accuracies.items():
    print("Model: " + model)
    for group, acc in acc.items():
        print(f"\t{group}: {acc:.2%}")
