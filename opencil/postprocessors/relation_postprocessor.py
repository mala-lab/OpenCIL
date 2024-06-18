from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin_min
import scipy
from tqdm import tqdm
from copy import deepcopy
from scipy.special import logsumexp
from math import ceil

from .base_postprocessor import BasePostprocessor, BaseCILPostprocessor

def normalize(feat, nc=50000):
    with torch.no_grad():
        split = ceil(len(feat) / nc)
        for i in range(split):
            feat_ = feat[i * nc:(i + 1) * nc]
            feat[i * nc:(i + 1) *
                 nc] = feat_ / torch.sqrt((feat_**2).sum(-1) + 1e-10).reshape(-1, 1)

    return feat


def kernel(feat, feat_t, prob, prob_t, split=2):
    """Kernel function (assume feature is normalized)
    """
    size = ceil(len(feat_t) / split)
    rel_full = []
    for i in range(split):
        feat_t_ = feat_t[i * size:(i + 1) * size]
        prob_t_ = prob_t[i * size:(i + 1) * size]

        with torch.no_grad():
            dot = torch.matmul(feat, feat_t_.transpose(1, 0))
            dot = torch.clamp(dot, min=0.)

            sim = torch.matmul(prob, prob_t_.transpose(1, 0))
            rel = dot * sim

        rel_full.append(rel)

    rel_full = torch.cat(rel_full, dim=-1)
    return rel_full


def get_relation(feat, feat_t, prob, prob_t, pow=1, chunk=50, thres=0.03):
    """Get relation values (top-k and summation)
    
    Args:
        feat (torch.Tensor [N,D]): features of the source data
        feat_t (torch.Tensor [N',D]): features of the target data
        prob (torch.Tensor [N,C]): probabilty vectors of the source data
        prob_t (torch.Tensor [N',C]): probabilty vectors of the target data
        pow (int): Temperature of kernel function
        chunk (int): batch size of kernel calculation (trade off between memory and speed)
        thres (float): cut off value for small relation graph edges. Defaults to 0.03.

    Returns:
        graph: statistics of relation graph
    """

    n = feat.shape[0]
    n_chunk = ceil(n / chunk)

    score = []
    for i in range(n_chunk):
        feat_ = feat[i * chunk:(i + 1) * chunk]
        prob_ = prob[i * chunk:(i + 1) * chunk]

        rel = kernel(feat_, feat_t, prob_, prob_t)

        mask = (rel.abs() > thres)
        rel_mask = mask * rel
        edge_sum = (rel_mask.sign() * (rel_mask.abs()**pow)).sum(-1)

        score.append(edge_sum.cpu())

    score = torch.cat(score, dim=0)

    return score


class RelationPostprocessor(BaseCILPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        super(RelationPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.pow = self.args.pow
        self.feature_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def setup(self, net_list, id_loader_dict, ood_loader_dict):
        [_, net] = net_list        
        net.eval()
        print('Extracting id validation neibor prior distributions')



        feature_log = []
        prob_log = []
        with torch.no_grad():
            for data, labels in tqdm(id_loader_dict['val'], desc='Setup: ', position=0, leave=True):
                data = data.cuda()

                out = net(data, return_feature=True)
                logit, feature = out["logits"], out["features"]
                prob = torch.softmax(logit, dim=1)
                feature_log.append(normalize(feature))
                prob_log.append(prob)

        self.feat_train = torch.cat(feature_log, axis=0)
        self.prob_train = torch.cat(prob_log, axis=0)

        

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
    
        out = net(data, return_feature=True)
        output, feature = out["logits"], out["features"]

        feature = normalize(feature)
        prob = torch.softmax(output, dim=1)

        score = get_relation(feature, self.feat_train, prob, self.prob_train, pow=self.pow)

        _, pred = torch.max(prob, dim=1)

        return pred, score
