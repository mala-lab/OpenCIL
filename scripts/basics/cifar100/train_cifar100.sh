#!/bin/bash
# sh scripts/basics/cifar100/train_cifar100.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/preprocessors/base_preprocessor.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/train/baseline.yml
