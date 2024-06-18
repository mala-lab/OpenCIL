# OpenCIL: Benchmarking Out-of-Distribution Detection in Class-Incremental Learning

## Description  
OpenCIL is the first benchmark platform expressly tailored both post-hoc-based and fine-tuning-based out-of-distribution (OOD) detection methods for different types of class incremental learning (CIL) models,  evaluating the capability of CIL models in rejecting diverse OOD samples.
OpenCIL offers a flexible algorithmic template that allows users to effortlessly implement their unique designs within the platform.
Moreover, it provides a new evaluation protocol to fairly and systematically compare diverse OOD detection methods among different incremental steps, and comprehensively evaluate 60 baselines that are composed by 15 OOD detectors and 4 CIL models.
We further propose a new baseline called BER that can effectively mitigate the common issues in the 60 baselines.

## Setup

### Environment
```
## Installing base package from opencil
conda env create -f environment.yaml
conda activate opencil

## Installing additional package
pip install seaborn
```
## Dataset
### Download and Prepare data
Download full dataset at the following [GDrive](https://drive.google.com/drive/u/1/folders/15pMDnQWNyMnBb_9i2uKgOX1Q2f5sOmSN) link. Then extract it into folder data which have the following structure:

```

├── data
│   └── benchmark_imglist
│       ├── cifar100
│       ├── imagenet
│   └── images_classic
│       ├── cifar10
│       ├── cifar100
│       ├── mnist
│       ├── places365
│       ├── svhn
│       ├── texture
│       ├── tin
│   └── images_largescale
│       ├── imagenet_1k
│           ├── train
│           ├── val
│       ├── imagenet_o
│       ├── inaturalist
│       ├── openimage_o
│       ├── species_sub
```

Note that train.zip and val.zip uploaded at largescale_data in GDrive should be stored inside imagenet_1k folder as shown above.
### Dataset explanation
<b>1. ID dataset for training class incremental learning model </b>

Training an class incremental learning approach on two main dataset: small dataset-cifar100 and large scale dataset-imagenet1k. 

Training such model requires splitting the original dataset into multiple tasks, Data samples from each task form a disjoint set with data samples from other tasks. Please refer to problem formulation of CIL for further information.

At each task, a collection of data samples from current task along with an exemplar set of samples from previous tasks are fed into the model for training. Below are information about different kinds of setting varying the number of classes that we have for each task to train CIL model on each dataset:
- cifar100: 100 classes in total with 3 main settings:
    - 5 tasks: 20 classes per each task
    - 10 tasks: 10 classes per each task
    - 20 tasks: 5 classes per each task
    - exemplar size for different task settings: 2,000
- imagenet1k: 1000 classes in total with 3 main settings:
    - 5 tasks: 200 classes per each task
    - 10 tasks: 100 classes per each task
    - 20 tasks: 500 classes per each task
    - exemplar size for different task settings: 20,000

These data setting for training incremental learning model will be automatically set up once you run the training script. For more information about how data is set up, please refer to main code of loading data ```opencil/datasets/data_manager_pycil.py``` which is called via ```get_datamanager``` inside ```opencil/datasets/utils.py```.

<b>2. dataset for training fine-tuning-based OOD detection method (optional) </b>

After training an class incremental learning model, we can freeze the feature extractor of CIL model and finetune an extra classifier to reject OOD samples which maintaining incremental classification accuracy.

Furthermore, we only can use the current task ID data to train this extra classifier, so the data split is the same as above.

<b>3. OOD dataset for testing CIL model </b>
After training an class incremental learning model, we should test its performance on ood detection score using other ood dataset.

Experiments should be conducted on both small scale and large scale dataset to demonstrate model's robustness:

- Small scale dataset→ Training CIL model on Cifar100: 100 classes and then perform testing OOD function on the following OOD dataset:
    - nearood: cifar10, tin.
    - farood: mnist, svhn, texture, places365.
- Large scale dataset→ Training CIL model on Imagenet1K: 1000 classes and then perform testing on OOD function on the following OOD dataset:
    - nearood: species, inaturalist, openimageo, imageneto.
    - farood: texture, mnist.

## Weights and preliminary results
To be updated

Weights and preliminary results are stored inside the same folder. 

## Usage
Scripts are mainly used in this repository. They are defined in ```scripts``` and there are four main types of scripts:
### Training a CIL method
Scripts for training any supported CIL methods are mostly defined in 
```scripts/cil_only/$CIL_METHOD$```. We have benchmarked 4 most common class incremental learning methods in total which are:

- iCaRL: Incremental Classifier and Representation Learning. [Paper](https://arxiv.org/abs/1611.07725)
- BiC: Large Scale Incremental Learning. [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf)
- FOSTER: Feature Boosting and Compression for Class-Incremental Learning. [Paper](https://arxiv.org/abs/2204.04662)
- WA: Maintaining Discrimination and Fairness in Class Incremental Learning. [Paper](https://arxiv.org/pdf/1911.07053.pdf)

These scripts could be used to train incremental learning model on two main dataset: cifar100 or imagenet1k. You might want to add more CIL methods following exactly the same configuration as aforementioned CIL methods.

#### How to run:
An example of training CIL model using iCaRL method with 10 tasks on cifar100 dataset is shown below:
```
bash scripts/cil_only/icarl/cifar100_train_cil.sh 10
```
### Testing OOD scoring function on pretrained CIL model
Scripts for testing ood performance of incremental learning model which has been trained before are mostly stored in ```scripts/ood_after_cil```.

The folder name of these scripts would be the combination of an CIL method that you want to test and an OOD scoring function that you want to use. 

For example, testing CIL model trained with iCaRL approach on cifar100 dataset using ebo function would result in the scripts at ```scripts/ood_after_cil/icarl/icarl_ebo/cifar100_test_ood.sh```. 

We have 9 most common OOD scoring function being implemented in this repository which are:
- MSP: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.
 [Paper](https://arxiv.org/abs/1610.02136). Note that the name of this ood scoring function is flexibly inside the code which could be ```msp``` or ```baseood```.
- ODIN: Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks.
 [Paper](https://arxiv.org/abs/1706.02690)
- EBO: Energy-based Out-of-distribution Detection.
 [Paper](https://arxiv.org/abs/2010.03759)
- KLM: Scaling Out-of-Distribution Detection for Real-World Settings. [Paper](https://proceedings.mlr.press/v162/hendrycks22a/hendrycks22a.pdf)

#### How to run
An example of testing iCaRL CIL model trained on CIFAR100 using MSP scoring function with 10 tasks is shown below
```
bash scripts/ood_after_cil/icarl/icarl_baseood/cifar100_test_ood.sh 10
```


### Fine-tuning-based OOD detection methods on pretrained CIL model
Scripts for finetuning any supported OOD detection methods on pretrained CIL methods are mostly defined in 
```scripts/cil_only_finetune/$OOD_METHOD$```. We have benchmarked 6 most common class incremental learning methods in total which are:

- MSP: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.
 [Paper](https://arxiv.org/abs/1610.02136). Note that the name of this ood scoring function is flexibly inside the code which could be ```msp``` or ```baseood```.
- ODIN: Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks.
 [Paper](https://arxiv.org/abs/1706.02690)
- EBO: Energy-based Out-of-distribution Detection.
 [Paper](https://arxiv.org/abs/2010.03759)
- KLM: Scaling Out-of-Distribution Detection for Real-World Settings. [Paper](https://proceedings.mlr.press/v162/hendrycks22a/hendrycks22a.pdf)

and including our proposed baseling **BER**. You can choose one of these 4 supported CIL models as pretrained CIL methods.

#### How to run
An example of finetuning OOD detection method REGMIX on CIL model iCaRL with 10 tasks on cifar100 dataset is shown below:
```
bash scripts/cil_only_finetune/regmix/cifar100_train_cil.sh icarl 10
```

Note that the pretrained CIL model should be prepared first.



### Testing OOD scoring function on finetuned CIL model
Scripts for testing ood performance of finetuned CIL model are mostly stored in ```scripts/ood_after_cil_finetune/$OOD_METHOD$```.


#### How to run
An example of testing the OOD detection method REGMIX finetuned CIL model of iCaRL on CIFAR100 with 10 tasks is shown below
```
bash scripts/ood_after_cil_finetune/regmix/cifar100_test_ood.sh icarl 10
```

Note that the finetuned CIL model should be prepared first.
