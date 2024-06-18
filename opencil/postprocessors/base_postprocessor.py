from typing import Any

import numpy as np
import pdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self, net: nn.Module, data_loader: DataLoader):
        pred_list, conf_list, label_list = [], [], []
        # for data, label in data_loader:
        #     data = data.cuda()
        #     label = label.cuda()
        
        for data, label in data_loader:
            data = data.cuda()
            label = label.cuda()
            pred, conf = self.postprocess(net, data)
            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, label_list
    

class BaseCILPostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)['logits']

        # output = torch.cat(outputs, dim=1)

        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        # return pred, conf, score
        return pred, conf

    # def inference(self, net: nn.Module, data_loader: DataLoader, output=False):
    def inference(self, net: nn.Module, data_loader: DataLoader):
        pred_list, conf_list, label_list = [], [], []
        # score_list = []
        for data, label in data_loader:
            data = data.to(self.config.device)
            label = label.to(self.config.device)
            # pred, conf, _ = self.postprocess(net, data, label)
            pred, conf = self.postprocess(net, data)
            # pred, conf, score = self.postprocess(net, data)


            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

                # score_list.append(score[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        # score_list = np.array(score_list)        

        # if output:
        #     return pred_list, score_list, label_list
        
        return pred_list, conf_list, label_list
    

class BaseCILFinetunePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)['aux_logits']

        # output = torch.cat(outputs, dim=1)

        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self, net: nn.Module, data_loader: DataLoader):
        pred_list, conf_list, label_list = [], [], []
        for data, label in data_loader:
            data = data.to(self.config.device)
            label = label.to(self.config.device)
            # pred, conf, _ = self.postprocess(net, data, label)
            pred, conf = self.postprocess(net, data)


            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, label_list
