from coeditor._utils import run_long_task
from coeditor.c3problem import C3ProblemChangeInlining, C3ProblemGenerator
from coeditor.common import *
from coeditor.dataset import *

if __name__ == "__main__":
    os.chdir(proj_root())

    dataset_name = "perm2K"
    generator = C3ProblemGenerator()
    transform = C3ProblemChangeInlining()
    with run_long_task(f"Preparing dataset {dataset_name} with encoder {generator}"):
        problems = make_or_load_dataset(
            dataset_name, generator, transform, remake_problems=False
        )

    tokenizer = C3ProblemTokenizer()
    for name, probs in problems.items():
        probs = cast(Sequence[C3Problem], probs)
        print("=" * 40, name, "=" * 40)
        stats = tokenizer.compute_stats(probs)
        pretty_print_dict(stats)
