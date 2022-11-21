import os
from typing import *

import wandb
from coeditor.common import *
from coeditor.dataset import TokenizedEditDataset
from coeditor.encoding import WindowArgs
from coeditor.model import (
    CoeditorModel,
    TrainingArgs,
    EvalArgs,
)

os.chdir(proj_root())

data_name = "small"
skip_unchanged = True
train_args = TrainingArgs(
    max_batch_tokens=4096,
    window=WindowArgs(2048),
    quicktest=False,
)
eval_args = EvalArgs(
    max_batch_tokens=4096 * 2,
    window=WindowArgs(4096),
)

model_name = f"coeditor-{data_name}"
if skip_unchanged:
    model_name += "-skip"
if train_args.quicktest:
    model_name = "quicktest-" + model_name

data_dir = get_dataset_dir(data_name) / "tokenized-file_based"
datasets = {name: pickle_load(data_dir / f"{name}.pkl") for name in ["train", "test"]}
train_data: TokenizedEditDataset = datasets["train"]
eval_data: TokenizedEditDataset = datasets["test"]
if train_args.quicktest:
    train_data = TokenizedEditDataset.from_edits(list(train_data.all_edits())[:10])
    eval_data = TokenizedEditDataset.from_edits(list(eval_data.all_edits())[:10])

model = CoeditorModel.from_code_t5()
model.skip_unchanged = True

project = "Coeditor" if not train_args.quicktest else "Coeditor-quicktest"
wandb.init(dir="../wandb", project=project, name=model_name)

model.train_on_data(model_name, train_data, eval_data, train_args, eval_args)

eval_result = model.eval_on_data(eval_data, eval_args)
eval_dict = {f"test/{k}": v.average() for k, v in eval_result.items()}
wandb.log(eval_dict)
