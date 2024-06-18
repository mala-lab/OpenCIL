import logging
import numpy as np
import io
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets, transforms


from opencil.preprocessors.utils import get_preprocessor


class DataManager(object):
    def __init__(self, config):#increment is num class each task
        self.config = config
        self.dataset_config = config.dataset
        self.shuffle = self.dataset_config.shuffle_order
        self.init_cls = self.dataset_config.init_cls_tfs_thf
        self.seed = self.config.seed
        self.dataset_name = self.dataset_config.name
        

        if self.init_cls == 0:
            # tfs setting, init first task as the other task
            self.init_cls = self.config.increment
        
        self._setup_data(self.dataset_name, self.shuffle, self.seed)
        assert self.init_cls <= len(self._class_order), "No enough classes."
        self._increments = [self.init_cls]
        while sum(self._increments) + int(self.config.increment) < len(self._class_order):
            self._increments.append(int(self.config.increment))
        
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)


    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)
    
    def get_accumulate_tasksize(self,task):
        return sum(self._increments[:task+1])

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None, ood_eval=None):
        # get corresponding source data
        # if source == "train":
        #     x, y = self._train_data, self._train_targets
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "val":
            x, y = self._val_data, self._val_targets
        elif source == "test":
            if ood_eval:
                # for ood evaluation after cil phase
                x, y = self._test_data, self._test_targets
            else: 
                # for accuracy evaluation or finetuning cil model
                x, y = self._valtest_data, self._valtest_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))
        
        # get corresponding transform
        
        trsf = get_preprocessor(self.config, mode) # mode = "train" or "test"
        
        data, targets = [], []
        
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
        
        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf)
        else:
            
            return DummyDataset(data, targets, trsf)
        
    def get_divided_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None, ood_eval=None):
        # get corresponding source data
        # if source == "train":
        #     x, y = self._train_data, self._train_targets
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "val":
            x, y = self._val_data, self._val_targets
        elif source == "test":
            if ood_eval:
                # for ood evaluation after cil phase
                x, y = self._test_data, self._test_targets
            else: 
                # for accuracy evaluation or finetuning cil model
                x, y = self._valtest_data, self._valtest_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))
        
        # get corresponding transform
        
        trsf = get_preprocessor(self.config, mode) # mode = "train" or "test"
        
        train_data, train_targets = [], []
        val_data, val_targets = [], []
        
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            train_data.append(class_data)
            train_targets.append(class_targets)
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            val_data.append(appendent_data)
            val_targets.append(appendent_targets)
        
        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        if ret_data:
            return train_data, train_targets, val_data, val_targets, DummyDataset(train_data, train_targets, trsf), DummyDataset(val_data, val_targets, trsf)
        else:
            return DummyDataset(train_data, train_targets, trsf), DummyDataset(val_data, val_targets, trsf)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._valtest_data, self._valtest_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        # get corresponding transform
        trsf = get_preprocessor(self.config, mode) # mode = "train" or "test"

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf), \
        DummyDataset(val_data, val_targets, trsf)


    def _setup_data(self, dataset_name, shuffle_order, seed):
        # idata = _get_idata(dataset_name)
        print("========Preparing dataset==============")

        train_imglist_pth, train_dir = self.dataset_config.train.imglist_pth, self.dataset_config.train.data_dir
        val_imglist_pth, val_dir = self.dataset_config.val.imglist_pth, self.dataset_config.val.data_dir
        test_imglist_pth, test_dir = self.dataset_config.test.imglist_pth, self.dataset_config.test.data_dir

        self._train_data, self._train_targets = read_img_list_pth(train_imglist_pth, train_dir)
        self._val_data, self._val_targets = read_img_list_pth(val_imglist_pth, val_dir)
        self._test_data, self._test_targets = read_img_list_pth(test_imglist_pth, test_dir)
        
        self._valtest_data = np.concatenate((self._test_data, self._val_data), axis=0)
        self._valtest_targets = np.concatenate((self._test_targets, self._val_targets), axis=0)

        # Transforms
        # self._train_trsf = idata.train_trsf
        # self._test_trsf = idata.test_trsf
        # self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle_order:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._val_targets = _map_new_class_index(self._val_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        self._valtest_targets = _map_new_class_index(self._valtest_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        # with open(path, 'rb') as f:
        #     content = f.read()
        # filebytes = content
        # buff = io.BytesIO(filebytes)
        # image = Image.open(buff).convert('RGB')

        with open(path, "rb") as f:
            image = Image.open(f).convert('RGB')
        

        image = self.trsf(image)
        label = self.labels[idx]

        return image, label
    
def read_img_list_pth(img_list_pth, data_dir):
    with open(img_list_pth) as imgfile:
        all_imglist = imgfile.readlines()

    data = []
    targets = []

    for line in all_imglist:
        class_idx = int(line.split(' ')[1])
        img_path = line.split(' ')[0]
        img_path = os.path.join(data_dir, img_path)

        data.append(img_path)
        targets.append(class_idx)

    data = np.array(data)
    targets = np.array(targets)
    return data, targets


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


# def _get_idata(dataset_name):
#     name = dataset_name.lower()
#     if name == "cifar10":
#         return iCIFAR10()
#     elif name == "cifar100":
#         return iCIFAR100()
#     elif name == "imagenet1000":
#         return iImageNet1000()
#     elif name == "imagenet100":
#         return iImageNet100()
#     else:
#         raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)

    with open(path, 'rb') as f:
        content = f.read()
    filebytes = content
    buff = io.BytesIO(filebytes)

    image = Image.open(buff).convert('RGB')

    return image


# def accimage_loader(path):
#     """
#     Ref:
#     https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
#     accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
#     accimage is available on conda-forge.
#     """
#     import accimage

#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


# def default_loader(path):
#     """
#     Ref:
#     https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
#     """
#     from torchvision import get_image_backend

#     if get_image_backend() == "accimage":
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)

def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar100_cil":
        return iCIFAR100()
    elif name == "imagenet1000_cil":
        return iImageNet1000()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()


