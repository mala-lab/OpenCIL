from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin_min
import scipy
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor, BaseCILPostprocessor

class KLMatchingPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = self.config.dataset.num_classes

    def kl(self, p, q):
        return scipy.stats.entropy(p, q)

    def setup(self, net_list, id_loader_dict, ood_loader_dict):
        [_, net] = net_list
        net.eval()

        print('Extracting id validation softmax posterior distributions')
        all_softmax = []
        preds = []
        with torch.no_grad():
            for data, label in tqdm(id_loader_dict['val'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = data.cuda()
                logits = net(data)
                all_softmax.append(F.softmax(logits, 1).cpu())
                preds.append(logits.argmax(1).cpu())

        all_softmax = torch.cat(all_softmax)
        preds = torch.cat(preds)

        self.mean_softmax_val = []
        for i in tqdm(range(self.num_classes)):
            # if there are no validation samples
            # for this category
            if torch.sum(preds.eq(i).float()) == 0:
                temp = np.zeros((self.num_classes, ))
                temp[i] = 1
                self.mean_softmax_val.append(temp)
            else:
                self.mean_softmax_val.append(
                    all_softmax[preds.eq(i)].mean(0).numpy())

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits = net(data)
        preds = logits.argmax(1)
        softmax = F.softmax(logits, 1).cpu().numpy()
        scores = -pairwise_distances_argmin_min(
            softmax, np.array(self.mean_softmax_val), metric=self.kl)[1]
        return preds, torch.from_numpy(scores)


class KLMatchingCILPostprocessor(BaseCILPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = self.config.dataset.num_classes

    def kl(self, p, q):
        return scipy.stats.entropy(p, q)

    def setup(self, net_list, id_loader_dict, ood_loader_dict):
        [_, net] = net_list
        net.eval()

        print('Extracting id validation softmax posterior distributions')
        all_softmax = []
        preds = []
        with torch.no_grad():
            for data, label in tqdm(id_loader_dict['val'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = data.cuda()
                logits = net(data)['logits']
                all_softmax.append(F.softmax(logits, 1).cpu())
                preds.append(logits.argmax(1).cpu())

        all_softmax = torch.cat(all_softmax)
        preds = torch.cat(preds)

        num_classes = logits.shape[1] # should be the number of class for current task

        self.mean_softmax_val = []
        for i in tqdm(range(num_classes)): # should be the number of class for current task
            # if there are no validation samples
            # for this category
            if torch.sum(preds.eq(i).float()) == 0:
                temp = np.zeros((num_classes, )) # should be the number of class for current task
                temp[i] = 1
                self.mean_softmax_val.append(temp)
            else:
                self.mean_softmax_val.append(
                    all_softmax[preds.eq(i)].mean(0).numpy())

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits = net(data)['logits']
        preds = logits.argmax(1)
        softmax = F.softmax(logits, 1).cpu().numpy()

        scores = -pairwise_distances_argmin_min(
            softmax, np.array(self.mean_softmax_val), metric=self.kl)[1]
        return preds, torch.from_numpy(scores)
    

class KLMatchingCILFinetunePostprocessor(BaseCILPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = self.config.dataset.num_classes

    def kl(self, p, q):
        return scipy.stats.entropy(p, q)

    def setup(self, net_list, id_loader_dict, ood_loader_dict):
        [_, net] = net_list
        net.eval()

        print('Extracting id validation softmax posterior distributions')
        all_softmax = []
        preds = []
        with torch.no_grad():
            for data, label in tqdm(id_loader_dict['val'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = data.cuda()
                logits = net(data)['aux_logits']
                all_softmax.append(F.softmax(logits, 1).cpu())
                preds.append(logits.argmax(1).cpu())

        all_softmax = torch.cat(all_softmax)
        preds = torch.cat(preds)

        num_classes = logits.shape[1] # should be the number of class for current task

        self.mean_softmax_val = []
        for i in tqdm(range(num_classes)): # should be the number of class for current task
            # if there are no validation samples
            # for this category
            if torch.sum(preds.eq(i).float()) == 0:
                temp = np.zeros((num_classes, )) # should be the number of class for current task
                temp[i] = 1
                self.mean_softmax_val.append(temp)
            else:
                self.mean_softmax_val.append(
                    all_softmax[preds.eq(i)].mean(0).numpy())

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits = net(data)['aux_logits']
        preds = logits.argmax(1)
        softmax = F.softmax(logits, 1).cpu().numpy()

        scores = -pairwise_distances_argmin_min(
            softmax, np.array(self.mean_softmax_val), metric=self.kl)[1]
        return preds, torch.from_numpy(scores)
