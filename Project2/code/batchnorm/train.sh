#!/bin/bash

## Switch conda environment
eval "$(conda shell.bash hook)"
conda activate dl

CUDA_VISIBLE_DEVICES=0 python main.py
