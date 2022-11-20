import os
from typing import *

import wandb
from coeditor.common import *
from coeditor.encoding import WindowArgs
from coeditor.model import (
    CoeditorModel,
    TrainingArgs,
    EvalArgs,
)

os.chdir(proj_root())

data_name = "small"
train_args = TrainingArgs(
    max_batch_tokens=4096,
    window=WindowArgs(2048),
    quicktest=False,
)
eval_args = EvalArgs(
    max_batch_tokens=4096 * 2,
    window=WindowArgs(4096),
)

train_name = f"{data_name}-quicktest" if train_args.quicktest else data_name

data_dir = get_dataset_dir(data_name) / "tokenized-file_based"
datasets = {name: pickle_load(data_dir / f"{name}.pkl") for name in ["train", "test"]}

model = CoeditorModel.from_code_t5()

wandb.init(dir="../wandb", project="Coeditor", name=train_name)

model.train_on_data(
    train_name, datasets["train"], datasets["test"], train_args, eval_args
)

eval_result = model.eval_on_data(datasets["test"], eval_args)
wandb.log({f"test/{k}": v for k, v in eval_result.items()})
