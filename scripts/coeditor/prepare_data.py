from coeditor.common import *
from coeditor.ctx_change_encoder import (
    C3Problem,
    C3ProblemGenerator,
    C3ProblemTokenizer,
    TkC3Problem,
)
from coeditor.dataset import *
from spot.utils import run_long_task


if __name__ == "__main__":
    os.chdir(proj_root())

    dataset_name = "medium"
    encoders = [
        C3EditEncoder(
            C3ProblemGenerator(),
            C3ProblemTokenizer(),
        )
    ]
    for encoder in encoders:
        with run_long_task(f"Preparing dataset {dataset_name} with encoder {encoder}"):
            make_or_load_datasets(dataset_name, encoder, recreate_data=True)
