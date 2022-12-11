# %%

import os
import asyncio
from typing import *

from spot.utils import not_none, proj_root, get_dataroot

os.chdir(proj_root())

datadir = get_dataroot()

# %%
# experiment configurations

from spot.data import (
    get_tk_dataset_name,
    load_tokenized_srcsets,
    TypeCheckSettings,
)
from spot.model import CtxArgs, DecodingArgs, ModelSPOT, ModelWrapper
from spot.train import TrainingConfig, TypeCheckArgs
from termcolor import colored

use_type_checker = False

config = TrainingConfig(
    quicktest=False,
    ctx_size=2048,
    left_margin=1024,
    right_margin=1023,
    modifications="no_type_checker",
)
gpu_id = 0
TypeCheckSettings.temp_path = f"DAgger-{gpu_id}"

if config.quicktest:
    print(colored("quicktest: True", "red"))

project_name = "test-SPOT" if config.quicktest else "SPOT"
train_ctx_args = config.train_ctx_args()
tc_args = TypeCheckArgs(check_in_isolation=config.check_in_isolation)

datasets_name = get_tk_dataset_name(
    drop_comments=config.drop_comments,
)

model_name = "DAgger-model--" + config.as_name()

tk_dataset = load_tokenized_srcsets(
    datadir,
    datasets_name,
    data_reduction=config.data_reduction,
    quicktest=config.quicktest,
)


# %%
# initialize the model
from spot.model import load_model_spot, DefaultTokenizer
from spot.model import ModelWrapper
from spot.dagger import DAggerModel
import torch

train_dec_args = DecodingArgs(
    max_batch_cost=8 * config.ctx_size,
    ctx_args=config.dec_ctx_args(),
    do_sample=True,  # use necleus sampling during training
    top_p=0.9,
)

model = load_model_spot("Salesforce/codet5-base")
wrapper = ModelWrapper(model, DefaultTokenizer, train_dec_args)
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
wrapper.to(device)
dmodel = DAggerModel(wrapper, use_type_checker=use_type_checker)


# %%
# pre-train evaluation
# from spot.utils import pretty_print_dict

# eval_r = asyncio.run(dmodel.eval_on_data(tk_dataset["test"][0:50]))
# pretty_print_dict(eval_r.accuracies)


# %%
# train the model
from spot.dagger import DAggerArgs
from spot.utils import run_long_task
import wandb
import shutil

ckpt_dir = datadir / f"checkpoints/running/{model_name}"

with run_long_task("DAgger training"):
    wandb.init(
        project=project_name,
        name=model_name,
        config=config.as_dict(),
        dir=str(datadir),
    )

    dargs = DAggerArgs(
        ckpt_dir,
        grad_accum_steps=config.grad_accum_labels,
        replay_buffer_size=1000,
    )

    finished = False
    try:
        asyncio.run(
            dmodel.train_on_data(
                tk_dataset["train"],
                dargs,
                log_fn=lambda t, x: wandb.log({"train/step": t, **x}),
            )
        )
        finished = True
    finally:
        save_tpye = "saved" if finished else "saved-emergent"
        save_path = datadir / f"checkpoints/{save_tpye}/{model_name}"
        print(colored(f"Saving trained model to: {save_path}", "blue"))
        shutil.rmtree(save_path, ignore_errors=True)
        wrapper.save_pretrained(save_path)

# %%
# post-train full evaluation
from spot.utils import pretty_print_dict, pretty_show_dict, PickleCache
from spot.visualization import string_to_html

test_dec_args = DecodingArgs(
    max_batch_cost=8 * config.ctx_size,
    ctx_args=CtxArgs(
        ctx_size=4096,
        left_margin=2048,
        right_margin=1023,
    ),
    do_sample=False,
    num_beams=8,
)
dmodel.wrapper.args = test_dec_args

eval_cache = PickleCache(save_path / "eval_cache")  # type: ignore

eval_r = eval_cache.cached(
    "eval_test", lambda: asyncio.run(dmodel.eval_on_data(tk_dataset["test"]))
)
pretty_print_dict(eval_r.accuracies)


def wandb_string(s: str):
    return wandb.Html(string_to_html(s))


wandb.log({"test/accuracies": wandb_string(pretty_show_dict(eval_r.accuracies))})

# %%
# compute valid set performance
import re
from spot.utils import not_none

validset = tk_dataset["valid"][0:-1:3]
# dmodel.wrapper.args = train_dec_args

with run_long_task("DAgger evaluating (valid set)"):
    for model_path in ckpt_dir.glob("step=*"):
        print(colored(f"Evaluating model checkpoint: {model_path}", "blue"))
        m = not_none(re.match("step=(.+)", model_path.name)).groups()[0]
        step = int(m)
        wrapper = ModelWrapper.from_pretrained(model_path)
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        wrapper.to(device)
        dmodel = DAggerModel(wrapper)
        eval_r = asyncio.run(dmodel.eval_on_data(validset))
        wandb.log(
            {
                "valid/full_acc": eval_r.accuracies["full_acc"].acc,
                "train/step": step,
            }
        )
