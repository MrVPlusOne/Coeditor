# Coeditor: Project-level Contextual Code Change Prediction


## Installation

This project uses [pipenv](https://pipenv.pypa.io/en/latest/) to manage the package dependencies. Pipenv tracks the exact package versions and manages the (project-specific) virtual environment for you. To install all dependencies, make sure you have pipenv and python 3.11 installed, then, run the following at the project root:
```bash
pipenv --python <path-to-your-python-3.11>  # create a new environment for this project
pipenv sync --dev
```

To add new dependences into the virtual environment, you can either add them via `pipenv install ..` (using `pipenv`) or `pipenv run pip install ..` (using `pip` from within the virtual environment). If your pytorch installation is not working properly, you might need to install it via the `pip` approach rather than `pipenv`.

We also provide a generated Python `requirements.txt` file in case you cannot use `pipenv` for some reasone.

## Usage
All `.py` scripts below can be run via `pipenv run python <script-name.py>`. For `.ipynb` notebooks, make sure you select the pipenv environment as the kernel. All commands listed here should be run from the project root.

- Run all unit tests: `pipenv run pytest`.
- Train a new model: [`scripts/train_model.py`](scripts/train_model.py).
    - You can optionally specify which GPUs should be used by setting the `CUDA_VISIBLE_DEVICES` environment variable. For example, to use GPUs 0 and 1, run `CUDA_VISIBLE_DEVICES=0,1 pipenv run python scripts/train_model.py`.
- download training data: [`scripts/download_data.ipynb`](scripts/download_data.ipynb).
- TODO: add other relevant scripts.


## Development
Please see [DevGuide.md](DevGuide.md).
