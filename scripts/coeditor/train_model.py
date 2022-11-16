import os
from typing import *
from coeditor.common import *
from coeditor.encoding import WindowArgs
from coeditor.model import (
    CodeT5Model,
    CoeditorModel,
    train_coeditor_model,
    TrainingArgs,
)

os.chdir(proj_root())

data_name = "small"
train_args = TrainingArgs(
    train_max_batch_tokens=4096,
    train_window=WindowArgs(2048),
    eval_max_batch_tokens=4096 * 2,
    eval_window=WindowArgs(4096),
    quicktest=False,
)

train_name = f"{data_name}-quicktest" if train_args.quicktest else data_name

data_dir = get_dataset_dir(data_name) / "tokenized-file_based"
datasets = {name: pickle_load(data_dir / f"{name}.pkl") for name in ["train", "test"]}

model = CoeditorModel.from_code_t5()

train_coeditor_model(model, train_name, datasets["train"], datasets["test"], train_args)
