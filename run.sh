#!/bin/bash

num_task=20 #10 5


device=0
for cil in bic foster icarl wa
do
    export CUDA_VISIBLE_DEVICES=${device}
    bash scripts/cil_only/${cil}/cifar100_train_cil.sh ${num_task} &
    let device=device+1 
done
wait

device=0
for ood in logitnorm t2fnorm augmix regmix vos npos ber 
do
    export CUDA_VISIBLE_DEVICES=${device}
    for cil in bic foster icarl wa
    do
        bash scripts/cil_only_finetune/${ood}/cifar100_train_cil.sh ${cil} ${num_task} &
    done
    let device=device+1 
done
for ood in klm
do
    export CUDA_VISIBLE_DEVICES=${device}
    for cil in bic foster icarl wa
    do
        bash scripts/ood_after_cil/${cil}/${cil}_${ood}/cifar100_test_ood.sh ${num_task} &
    done
done
wait

device=0
for ood in baseood ebo gen maxlogit nnguide odin react relation
do
    export CUDA_VISIBLE_DEVICES=${device}
    for cil in bic foster icarl wa
    do
        bash scripts/ood_after_cil/${cil}/${cil}_${ood}/cifar100_test_ood.sh ${num_task} &
    done
    let device=device+1 
done
wait


device=0
for ood in logitnorm t2fnorm augmix regmix vos npos ber 
do
    export CUDA_VISIBLE_DEVICES=${device}
    for cil in bic foster icarl wa
    do
        bash scripts/ood_after_cil_finetune/${ood}/finetune_ebo/cifar100_test_ood.sh ${cil} ${num_task} &
    done
    let device=device+1 
done


