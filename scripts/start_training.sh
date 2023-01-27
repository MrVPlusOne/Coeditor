#!/bin/bash

# Control the GPUs used by huggingface Trainer here.
CUDA_VISIBLE_DEVICES="0" pipenv run python scripts/coeditor/train_model.py
