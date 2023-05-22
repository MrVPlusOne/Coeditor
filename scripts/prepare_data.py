"""
This script preprocesses the repos into the PyCommits format introduced in the paper.
You can safely skip this step since it will automatically be run when you
train a new model (and with the corresponding encoder parameters).

The raw repos will be loaded from `get_dataset_dir(dataset_name) / "repos"`, and the
processed results will be saved to `get_dataset_dir(dataset_name) / "processed"`
and `get_dataset_dir(dataset_name) / "transformed"`.
"""

from coeditor._utils import run_long_task
from coeditor.c3problem import C3ProblemChangeInlining, C3ProblemGenerator
from coeditor.common import *
from coeditor.dataset import *

if __name__ == "__main__":
    os.chdir(proj_root())

    dataset_name = "perm2k"
    encoder = C3CombinedEncoder(
        problem_tranform=C3ProblemChangeInlining(
            max_inline_ratio=0.6, allow_empty_problems=True
        ),
    )
    with run_long_task(
        f"Preparing dataset {dataset_name} with encoder {encoder.change_processor}"
    ):
        problems = make_or_load_dataset(
            dataset_name,
            encoder.change_processor,
            ("valid", "test", "train"),
            remake_problems=False,
        )

        transformed = make_or_load_transformed_dataset(
            dataset_name,
            problems,
            encoder,
        )

    tokenizer = C3ProblemTokenizer()
    for name, probs in transformed.items():
        probs = cast(Sequence[C3Problem], probs)
        print("=" * 40, name, "=" * 40)
        stats = tokenizer.compute_stats(probs)
        pretty_print_dict(stats)
