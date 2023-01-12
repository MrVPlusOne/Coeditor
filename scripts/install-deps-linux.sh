#! /bin/bash
conda create -n coeditor python=3.11
python_loc=$(conda info --envs | grep "coeditor" | awk '{print $2}')
"$python_loc/bin/pip" install pipenv --user
pipenv --python="$python_loc/bin/python"
pipenv sync