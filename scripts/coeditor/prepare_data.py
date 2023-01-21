from coeditor.common import *
from coeditor.ctx_change_encoder import (
    C3Problem,
    C3ProblemGenerator,
    C3ProblemTokenizer,
    TkC3Problem,
)
from coeditor.dataset import *


if __name__ == "__main__":
    os.chdir(proj_root())

    # dataset_name = "SPOT"
    dataset_name = "small"
    encoders = [
        C3EditEncoder(
            C3ProblemGenerator(),
            C3ProblemTokenizer(),
        )
    ]
    for encoder in encoders:
        with timed_action(f"Preparing dataset with encoder {encoder}"):
            make_or_load_datasets(dataset_name, encoder)
