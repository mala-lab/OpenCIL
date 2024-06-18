import torch
import pdb
import random
from numpy import load
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from opencil.preprocessors.test_preprocessor import TestStandardPreProcessor
from opencil.preprocessors.utils import get_preprocessor
from opencil.utils.config import Config

from .feature_dataset import FeatDataset
from .data_manager_pycil import DataManager
from .imglist_dataset import ImglistDataset
from .incremental_dataset import IncrementalDataset, FragmentedOODDataset


def get_dataloader(config: Config):
    # prepare a dataloader dictionary
    dataset_config = config.dataset
    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        preprocessor = get_preprocessor(config, split)
        # weak augmentation for data_aux
        data_aux_preprocessor = TestStandardPreProcessor(config)
        CustomDataset = eval(split_config.dataset_class)
        dataset = CustomDataset(name=dataset_config.name + '_' + split,
                                imglist_pth=split_config.imglist_pth,
                                data_dir=split_config.data_dir,
                                num_classes=dataset_config.num_classes,
                                preprocessor=preprocessor,
                                data_aux_preprocessor=data_aux_preprocessor)
        sampler = None
        if dataset_config.num_gpus * dataset_config.num_machines > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            split_config.shuffle = False

        dataloader = DataLoader(dataset,
                                batch_size=split_config.batch_size,
                                shuffle=split_config.shuffle,
                                num_workers=dataset_config.num_workers,
                                sampler=sampler)

        dataloader_dict[split] = dataloader
    return dataloader_dict


def get_ood_dataloader(config: Config):
    # specify custom dataset class
    ood_config = config.ood_dataset
    CustomDataset = eval(ood_config.dataset_class)
    dataloader_dict = {}
    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config, split)
        data_aux_preprocessor = TestStandardPreProcessor(config)
        if split == 'val':
            # validation set
            dataset = CustomDataset(
                name=ood_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=ood_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            dataloader = DataLoader(dataset,
                                    batch_size=ood_config.batch_size,
                                    shuffle=ood_config.shuffle,
                                    num_workers=ood_config.num_workers)
            dataloader_dict[split] = dataloader
        else:
            # dataloaders for csid, nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config.datasets:
                dataset_config = split_config[dataset_name]
                dataset = CustomDataset(
                    name=ood_config.name + '_' + split,
                    imglist_pth=dataset_config.imglist_pth,
                    data_dir=dataset_config.data_dir,
                    num_classes=ood_config.num_classes,
                    preprocessor=preprocessor,
                    data_aux_preprocessor=data_aux_preprocessor)
                dataloader = DataLoader(dataset,
                                        batch_size=ood_config.batch_size,
                                        shuffle=ood_config.shuffle,
                                        num_workers=ood_config.num_workers)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict


def get_feature_dataloader(dataset_config: Config):
    # load in the cached feature
    loaded_data = load(dataset_config.feat_path, allow_pickle=True)
    total_feat = torch.from_numpy(loaded_data['feat_list'])
    del loaded_data
    # reshape the vector to fit in to the network
    total_feat.unsqueeze_(-1).unsqueeze_(-1)
    # let's see what we got here should be something like:
    # torch.Size([total_num, channel_size, 1, 1])
    print('Loaded feature size: {}'.format(total_feat.shape))

    split_config = dataset_config['train']

    dataset = FeatDataset(feat=total_feat)
    dataloader = DataLoader(dataset,
                            batch_size=split_config.batch_size,
                            shuffle=split_config.shuffle,
                            num_workers=dataset_config.num_workers)

    return dataloader

# PyCIL format
def get_data_manager(config: Config):
    
    return DataManager(config)


# CIL part
def get_cil_dataloader(config: Config):
    # get configuration for current dataset
    dataset_config = config.dataset
    all_dataloader_task = []

    # get configuration for cil setting
    num_tasks = config.num_tasks
    total_num_classes = config.dataset.num_classes
    start_class_idx = 0

    preprocessor_dict = {}

    print('------------------ Preloading Incremental Dataset --------------------------', flush=True)
    for tid in tqdm(range(num_tasks)):
        dataloader_task_dict = {}
        for split in dataset_config.split_names:
            split_config = dataset_config[split]
            preprocessor = get_preprocessor(config, split)
            preprocessor_dict[split] = preprocessor
            # weak augmentation for data_aux
            data_aux_preprocessor = TestStandardPreProcessor(config)
            CustomDataset = eval(split_config.dataset_class)

            if tid < num_tasks - 1:
                num_classes_per_task = int(total_num_classes/num_tasks)
            else:
                num_classes_per_task = total_num_classes - tid * int(total_num_classes/num_tasks)
            
            dataset = CustomDataset(name=dataset_config.name + '_' + split,
                                    imglist_pth=split_config.imglist_pth,
                                    data_dir=split_config.data_dir,
                                    start_class_idx=start_class_idx,
                                    num_classes=num_classes_per_task,
                                    current_task_id=tid,
                                    preprocessor=preprocessor,
                                    data_aux_preprocessor=data_aux_preprocessor)

            sampler = None
            if dataset_config.num_gpus * dataset_config.num_machines > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                split_config.shuffle = False

            dataloader = DataLoader(dataset,
                                    batch_size=split_config.batch_size,
                                    shuffle=split_config.shuffle,
                                    num_workers=dataset_config.num_workers,
                                    sampler=sampler)
            

            dataloader_task_dict[split] = dataloader
        # update start class idx
        start_class_idx += num_classes_per_task
        all_dataloader_task.append(dataloader_task_dict)
    return all_dataloader_task, preprocessor_dict


def get_cil_ood_dataloader(config: Config, num_tasks, task_id, type):

    # specify custom dataset class
    ood_config = config.ood_dataset
    CustomDataset = eval(ood_config.dataset_class)
    dataloader_dict = {}
    stat_ood_num_samples = {}

    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config, split)
        data_aux_preprocessor = TestStandardPreProcessor(config)
        if split == 'val':
            ValDataset = eval(split_config.dataset_class)

            # validation set
            dataset = ValDataset(
                name=ood_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=ood_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            dataloader = DataLoader(dataset,
                                    batch_size=ood_config.batch_size,
                                    shuffle=ood_config.shuffle,
                                    num_workers=ood_config.num_workers)
            dataloader_dict[split] = dataloader
        else:
            # dataloaders for csid, nearood, farood
            sub_dataloader_dict = {}
            stat_dict = {}
            for dataset_name in split_config.datasets:
                dataset_config = split_config[dataset_name]


                with open(dataset_config.imglist_pth) as imgfile:
                    all_imglist = imgfile.readlines()

                num_ood_samples = len(all_imglist)
                num_ood_samples_per_task = int(num_ood_samples/num_tasks)
                
                # if type == 'all': # all previous ood up to current task
                #     if task_id < num_tasks - 1:
                #         next_imglist_ood = all_imglist[:(task_id+1)*num_ood_samples_per_task]
                #     else:
                #         next_imglist_ood = all_imglist
                # elif type == 'old': # ood of old task
                #     next_imglist_ood = all_imglist[:task_id*num_ood_samples_per_task]
                # else: # ood for present task
                #     next_imglist_ood = all_imglist[task_id*num_ood_samples_per_task:(task_id+1)*num_ood_samples_per_task]
                if type == 'accumulating':
                    next_imglist_ood = all_imglist[:(task_id+1)*num_ood_samples_per_task]
                elif type == 'nth':
                    next_imglist_ood = all_imglist[task_id*num_ood_samples_per_task:(task_id+1)*num_ood_samples_per_task]
                elif type == 'all_with_100_random_samples':
                    next_imglist_ood = random.sample(all_imglist, 100)
                
                dataset = CustomDataset(
                    name=ood_config.name + '_' + split + '_task' + str(task_id),
                    imglist=next_imglist_ood,
                    data_dir=dataset_config.data_dir,
                    num_classes=ood_config.num_classes,
                    preprocessor=preprocessor,
                    data_aux_preprocessor=data_aux_preprocessor)
                
                # append dataloader
                dataloader = DataLoader(dataset,
                                        batch_size=ood_config.batch_size,
                                        shuffle=ood_config.shuffle,
                                        num_workers=ood_config.num_workers)
                sub_dataloader_dict[dataset_name] = dataloader
                stat_dict[dataset_name] = len(next_imglist_ood)
            dataloader_dict[split] = sub_dataloader_dict
            stat_ood_num_samples[split] = stat_dict

    return dataloader_dict, stat_ood_num_samples


def get_concat_dataloader(config: Config, growing_id_samples_dict):

    dataset_config = config.dataset
    dataloader_dict = {}
    for split in growing_id_samples_dict:
        split_config = dataset_config[split]
        concat_data = ConcatDataset(growing_id_samples_dict[split])
        dataloader = DataLoader(concat_data,
                                batch_size=split_config.batch_size,
                                shuffle=split_config.shuffle,
                                num_workers=dataset_config.num_workers,
                                sampler=None)
        dataloader_dict[split] = dataloader

    return dataloader_dict


