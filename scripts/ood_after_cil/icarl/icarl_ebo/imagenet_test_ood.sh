#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

NUM_CPTASK=$1

python main.py \
--config configs/datasets/imagenet/imagenet_cil.yml \
configs/datasets/imagenet/imagenet_cil_ood.yml \
configs/networks/incremental_resnet_18_imagenet.yml \
configs/pipelines/ood_after_cil/icarl.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/ebo_cil.yml \
--increment $NUM_CPTASK \