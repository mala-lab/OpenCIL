#!/bin/bash
# sh scripts/ood/conf_branch/cifar100_train_conf_branch.sh

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

METHOD=$1
NUM_CPTASK=$2

python main.py \
--config configs/datasets/cifar100/cifar100_cil.yml \
configs/networks/incremental_resnet_32.yml \
configs/pipelines/finetune/finetune_augmix.yml \
configs/preprocessors/augmix_preprocessor.yml \
--cilmethod $METHOD \
--increment $NUM_CPTASK \
