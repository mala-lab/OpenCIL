from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin_min
import scipy
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor, BaseCILPostprocessor


class GENPostprocessor(BaseCILPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.gamma = self.args.gamma
        self.M = self.args.M
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):

        output = net(data)['logits']
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.generalized_entropy(score, self.gamma, self.M)

        return pred, conf

    def generalized_entropy(self, softmax_id_val, gamma=0.1, M=100):
        probs = softmax_id_val
        probs_sorted = torch.sort(probs, dim=1)[0][:, -M:]
        scores = torch.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma),
                           dim=1)

        return -scores
