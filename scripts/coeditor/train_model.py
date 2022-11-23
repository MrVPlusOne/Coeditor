import os
from typing import *

import wandb
from coeditor.common import *
from coeditor.dataset import TokenizedEditDataset
from coeditor.encoding import WindowArgs
from coeditor.model import *

os.chdir(proj_root())


def check_save_dir(model_name: str):
    to_check = [get_model_dir(b) / model_name for b in [True, False]]
    exists = [path for path in to_check if path.exists()]
    if exists:
        for path in exists:
            print(f"Path already exists:", path)
        answer = input("Continue training? (y/n):")
        if answer.lower().strip() != "y":
            print("Training aborted.")
            exit(1)


data_name = "medium"
data_dir = get_dataset_dir(data_name) / "tokenized-file_collapsed"
model_variant = "-collapse-shuffle"
data_args = DataTransformArgs(
    shuffle_extra_ids=True,
)
train_args = TrainingArgs(
    max_batch_tokens=4096,
    window=WindowArgs(2048),
    quicktest=False,
)
valid_args = EvalArgs(
    max_batch_tokens=4096 * 2,
    window=WindowArgs(2048),
)
test_args = EvalArgs(
    max_batch_tokens=4096 * 2,
    window=WindowArgs(4096),
)

model_name = f"coeditor-{data_name}"
model_name += model_variant
if train_args.quicktest:
    model_name = "quicktest-" + model_name

if not train_args.quicktest:
    check_save_dir(model_name)

config_dict = {
    k: get_modified_args(v)
    for k, v in {
        "data": data_args,
        "train": train_args,
        "valid": valid_args,
        "test": test_args,
    }.items()
}

project = "Coeditor" if not train_args.quicktest else "Coeditor-quicktest"
wandb.init(dir="..", project=project, name=model_name, config=config_dict)

with timed_action(f"Loading datasets: {data_dir}"):
    datasets: dict[str, TokenizedEditDataset] = {
        name: pickle_load(data_dir / f"{name}.pkl")
        for name in ["train", "valid", "test"]
    }
if train_args.quicktest:
    for name, dataset in datasets.items():
        datasets[name] = TokenizedEditDataset.from_edits(list(dataset.all_edits())[:10])

model = CoeditorModel.from_code_t5(data_args)

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    warnings.warn(
        "CUDA_VISIBLE_DEVICES not set, using 0. Note that "
        "the Huggingface Trainer will use all visible GPUs for training."
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


with timed_action("Training"):
    model.train_on_data(
        model_name, datasets["train"], datasets["valid"], train_args, valid_args
    )

with timed_action("Evaluating"):
    eval_result = model.eval_on_data(datasets["test"], test_args)

eval_dict = {f"test/{k}": v.average() for k, v in eval_result.items()}
wandb.log(eval_dict)
