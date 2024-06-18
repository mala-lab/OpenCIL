from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin_min
import scipy
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor, BaseCILPostprocessor


class ReActPostprocessor(BaseCILPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def setup(self, net_list, id_loader_dict, ood_loader_dict):
        [_, net] = net_list        
        net.eval()

        print('Extracting id validation feature posterior distributions')
        activation_log = []
        with torch.no_grad():
            for data, label in tqdm(id_loader_dict['val'],
                                desc='Eval: ',
                                position=0,
                                leave=True):
                data = data.cuda()
                # data = data.float()

                feature = net(data, return_feature=True)["features"]
                activation_log.append(feature.data.cpu().numpy())

        self.activation_log = np.concatenate(activation_log, axis=0)

        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)
        

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        
        output = net(data, feature_operate = "ReAct", feature_operate_parameter = self.threshold)['logits']
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        energyconf = torch.logsumexp(output, dim=1)

        return pred, energyconf
