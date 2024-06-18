import ast
import io
import logging
import os

import torch
from PIL import Image, ImageFile

from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)

# Each instance represent for dataset for a particular task
class IncrementalDataset(BaseDataset):
    def __init__(self,
                 name,
                 imglist_pth,
                 data_dir,
                 start_class_idx,
                 num_classes, # num class for each task
                 current_task_id,
                 preprocessor,
                 data_aux_preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(IncrementalDataset, self).__init__(**kwargs)

        self.name = name
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.current_task_id = current_task_id
        self.preprocessor = preprocessor
        self.transform = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        
        with open(imglist_pth) as imgfile:
            all_imglist = imgfile.readlines()

        # calibrate imglist so that it only includes list of image paths for current task
        ## define range of class id
        # start_class_idx = num_classes * (current_task_id)
        end_class_idx = start_class_idx + num_classes - 1 #inclusive

        self.images = []
        self.labels = []
        all_existing_class = []

        for line in all_imglist:
            class_idx = int(line.split(' ')[1])

            if class_idx not in all_existing_class:
                all_existing_class.append(class_idx)

            if class_idx >= start_class_idx and class_idx <= end_class_idx:
                self.images.append(line)
                self.labels.append(class_idx)
        
        self.class_indices = list(range(len(all_existing_class)))
                
    
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')

    def __len__(self):
        if self.maxlen is None:
            return len(self.images)
        else:
            return min(len(self.images), self.maxlen)

    def getitem(self, index):
        line = self.images[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        # some preprocessor methods require setup
        self.preprocessor.setup(**kwargs)
        try:
            if not self.dummy_read:
                with open(path, 'rb') as f:
                    content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                image = Image.open(buff).convert('RGB')
                sample['data'] = self.transform(image)
                sample['data_aux'] = self.transform_aux_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                # calibrate to make class id fit
                # sample['label'] = int(extra_str) - self.num_classes * self.current_task_id

                # normal idx
                sample['label'] = int(extra_str)
            # # Generate Soft Label
            # soft_label = torch.Tensor(self.num_classes)
            # if sample['label'] < 0:
            #     soft_label.fill_(1.0 / self.num_classes)
            # else:
            #     soft_label.fill_(0)
            #     soft_label[sample['label']] = 1
            # sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample['data'], sample['label']
    

class FragmentedOODDataset(BaseDataset): # only used for evaluation of cil ood
    def __init__(self,
                 name,
                 imglist,
                 data_dir,
                 num_classes, #equally dividing samples across task
                 preprocessor,
                 data_aux_preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(FragmentedOODDataset, self).__init__(**kwargs)

        self.name = name
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.transform = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        
        self.imglist = imglist


        self.images = []
        self.labels = []
        all_existing_class = []
    
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        # some preprocessor methods require setup
        self.preprocessor.setup(**kwargs)
        try:
            if not self.dummy_read:
                with open(path, 'rb') as f:
                    content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                image = Image.open(buff).convert('RGB')
                sample['data'] = self.transform(image)
                sample['data_aux'] = self.transform_aux_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                # calibrate to make class id fit
                # sample['label'] = int(extra_str) - self.num_classes * self.current_task_id

                # normal idx
                sample['label'] = int(extra_str)
            # # Generate Soft Label
            # soft_label = torch.Tensor(self.num_classes)
            # if sample['label'] < 0:
            #     soft_label.fill_(1.0 / self.num_classes)
            # else:
            #     soft_label.fill_(0)
            #     soft_label[sample['label']] = 1
            # sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample['data'], sample['label']
