# Coeditor: Project-level Contextual Code Change Prediction


## Installation

This project uses [pipenv](https://pipenv.pypa.io/en/latest/) to manage the package dependencies. Pipenv tracks the exact package versions and manages the (project-specific) virtual environment for you. To install all dependencies, make sure you have pipenv installed, then, at the project root, run:
```bash
pipenv sync
```

To add new dependences into the virtual environment, you can manually install them via `pipenv install ..` (using `pipenv`) or `pipenv run pip install ..` (using `pip` from within the virtual environment). Note that pipenv records pacakage requirements in the `Pipfile` and also manages additional environment variables in the `.env` file.

We also provide a generated Python `requirements.txt` file in case you cannot use `pipenv` for some reasone.

## Usage
All commands listed here assuming are run from the project root.

- run tests: `pipenv run pytest/tests`
- train a new model: `pipenv run python scripts/train_model.py`.
    - You can optionally specify which GPUs should be used by setting the `CUDA_VISIBLE_DEVICES` environment variable. For example, to use GPUs 0 and 1, run `CUDA_VISIBLE_DEVICES=0,1 pipenv run python scripts/train_model.py`.
- download training data: See [`scripts/download_data.ipynb`](scripts/download_data.ipynb).



## Development
Please see [DevGuide.md](DevGuide.md).
