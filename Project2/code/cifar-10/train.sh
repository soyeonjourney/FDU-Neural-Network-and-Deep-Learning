#!/bin/bash

## Switch conda environment
eval "$(conda shell.bash hook)"
conda activate dl

## Toy Case
CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNet18 --batch-size=128 --lr=0.1 --max-epoch=10

## Network Architecture
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNet18 --batch-size=128 --lr=0.1 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --model=PreActResNet18 --batch-size=128 --lr=0.1 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --model=WideResNet28x10 --batch-size=128 --lr=0.1 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNeXt29_32x4d --batch-size=128 --lr=0.1 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --model=DenseNet121 --batch-size=128 --lr=0.1 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --model=DPN26 --batch-size=128 --lr=0.1 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --model=DLA --batch-size=128 --lr=0.1 --max-epoch=200

## Ablation Study
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNet50 --batch-size=128 --lr=0.1 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNeXt29_2x32d --batch-size=128 --lr=0.1 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNeXt29_2x64d --batch-size=128 --lr=0.1 --max-epoch=200
## CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNeXt29_8x64d --batch-size=64 --lr=0.1 --max-epoch=200  # Looooooong time to train, just skip it
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNeXt50_2x40d --batch-size=128 --lr=0.1 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNeXt50_8x14d --batch-size=128 --lr=0.1 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNeXt50_32x4d --batch-size=128 --lr=0.1 --max-epoch=200

## Loss Function
# CUDA_VISIBLE_DEVICES=0 python main_criterion.py --max-epoch=50

## Activation Function
# CUDA_VISIBLE_DEVICES=0 python main_activation.py --max-epoch=50

## Optimizer
# CUDA_VISIBLE_DEVICES=0 python main_optimizer.py --max-epoch=50

## Cutout
# CUDA_VISIBLE_DEVICES=0 python main.py --model=WideResNet28x10 --batch-size=128 --lr=0.1 --max-epoch=200 --use-cutout
