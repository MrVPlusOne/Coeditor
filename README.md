# Coeditor: Leveraging Repo-level Diffs for Code Auto-editing

Coeditor is a transformer model that auto-edits your code based on your recent code changes. This repo includes the server code for the [Coeditor VSCode extension](https://marketplace.visualstudio.com/items?itemName=JiayiWei.vscode-coeditor) and the scripts for data processing, model training, and evaluation. The ideas behind Coeditor are presented in the ICLR Spotlight paper, [Coeditor: Leveraging Repo-level Diffs for Code Auto-editing](https://openreview.net/forum?id=ALVwQjZRS8), by Jiayi Wei, Greg Durrett, and Isil Dillig.

Watch the [Coeditor demo](https://www.youtube.com/watch?v=hjZE__jslzs) on Youtube.

## Installation

### Method 1: with Poetry (recommended)

This project uses [poetry](https://python-poetry.org) to manage the package dependencies. Poetry records all dependencies in the `pyproject.toml` file and manages the (project-specific) virtual environment for you.

You can install poetry via the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry completions bash >> ~/.bash_completion
```

To install all dependencies required by Coeditor, make sure you have python 3.11 installed, then, run the following at the project root:

```bash
poetry install
```

You can then spawn a shell within the project's virtual environment via `poetry shell`.

### Method 2: using requirements.txt

Alternatively, you can also install all dependencies using the exported [`requirements.txt`](requirements.txt) file.

```bash
pip3 install -r requirements.txt
```

## Usages

**Note**: All scripts below should be run within the poetry shell (or the virtual environment in which you installed all the dependencies).

### Use the VSCode extension server✨

Run [`python scripts/start_server.py`](scripts/start_server.py) to start the Coeditor VSCode extension server. This will download the pre-trained Coeditor model from Huggingface (if not already) and start the extension service at port 5042.

### Run Coeditor inside a notebook
- As an alternative to using the VSCode extension, you can directly run Coeditor inside [this notebook](notebooks/run_api.ipynb) by specifying a target file and line nubmer.

### Run unit tests

You can run all unit tests via `poetry run pytest` (or just `pytest` if you run inside the poetry shell).

### Download the PyCommits dataset

1. (Optional) Configure the directories. Create the file `config/coeditor.json` and use the following template to specify where you want to store the dataset and the trained models:

```json
{
    "datasets_root": "/path/to/datasets/directory",
    "models_root": "/path/to/models/direcotry"
}
```

2. Run the cells in [notebooks/download_data.ipynb](notebooks/download_data.ipynb) to clone the repos from GitHub. Note that we use the GitHub search API to search for repos with permissive licenses, so the results may change over time even though the query remains the same.

3. (Optional) Run [scripts/prepare_data.py](scripts/prepare_data.py) to preprocess the repos into the PyCommits format introduced in the paper. You can safely skip this step since it will automatically be run when you train a new model (and with the corresponding encoder parameters).

### Train a new model

Use the [scripts/train_model.py](scripts/train_model.py) script to train a new model from scratch. By default, this script trains a model under our default settings, but you can uncomment the corresponding function calls at the bottom of the script to train a model following one of the ablation settings in the paper.

**Note**: Only training with a single GPU is tested. You can set the GPU to use via the `CUDA_VISIBLE_DEVICES` environment variable.

### Evaluate pre-trained models

- **Comparison with Code Completion Approaches**: Run [scripts/code_completion_eval.py](scripts/code_completion_eval.py) to obtain the results reported in section 4.1 of the paper.
- **Multi-round editing**: Run [scripts/multi_round_eval.py](scripts/multi_round_eval.py) to obtain the results reported in section 4.2 of the paper.
- **Ablation Studies**: Run [scripts/single_round_eval.py](scripts/single_round_eval.py) to obtain the results reported in section 4.3 of the paper.


## Citation
Please cite our paper as:
```
@inproceedings{
wei2024coeditor,
title={Coeditor: Leveraging Repo-level Diffs for Code Auto-editing},
author={Jiayi Wei and Greg Durrett and Isil Dillig},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=ALVwQjZRS8}
}
```
