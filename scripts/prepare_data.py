from coeditor._utils import run_long_task
from coeditor.c3problem import C3ProblemChangeInlining, C3ProblemGenerator
from coeditor.common import *
from coeditor.dataset import *

if __name__ == "__main__":
    os.chdir(proj_root())

    dataset_name = "tiny"
    encoder = C3CombinedEncoder(
        problem_tranform=C3ProblemChangeInlining(
            max_inline_ratio=1.0, allow_empty_problems=True
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
