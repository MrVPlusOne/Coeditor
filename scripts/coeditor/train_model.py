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

data_name = "medium"
skip_unchanged = False
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
with timed_action("Loading datasets"):
    datasets: dict[str, TokenizedEditDataset] = {
        name: pickle_load(data_dir / f"{name}.pkl")
        for name in ["train", "valid", "test"]
    }
if train_args.quicktest:
    for name, dataset in datasets.items():
        datasets[name] = TokenizedEditDataset.from_edits(list(dataset.all_edits())[:10])

model = CoeditorModel.from_code_t5()
model.skip_unchanged = skip_unchanged

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    warnings.warn(
        "CUDA_VISIBLE_DEVICES not set, using 0. Note that "
        "the Huggingface Trainer will use all visible GPUs for training."
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

project = "Coeditor" if not train_args.quicktest else "Coeditor-quicktest"
wandb.init(dir="../wandb", project=project, name=model_name)

with timed_action("Training"):
    model.train_on_data(
        model_name, datasets["train"], datasets["valid"], train_args, eval_args
    )

with timed_action("Evaluating"):
    eval_result = model.eval_on_data(datasets["test"], eval_args)

eval_dict = {f"test/{k}": v.average() for k, v in eval_result.items()}
wandb.log(eval_dict)
