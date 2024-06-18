#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

METHOD=$1
NUM_CPTASK=$2

python main.py \
--config configs/datasets/imagenet/imagenet_cil.yml \
configs/datasets/imagenet/imagenet_cil_ood.yml \
configs/networks/incremental_resnet_18_imagenet.yml \
configs/pipelines/ood_after_cil_finetune/finetune_regmix.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/ebo_cil_finetune.yml \
--cilmethod $METHOD \
--increment $NUM_CPTASK \