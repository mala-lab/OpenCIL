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

from .base_postprocessor import BasePostprocessor, BaseCILPostprocessor

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


def knn_score(bankfeas, queryfeas, k=100, min=False):

    bankfeas = deepcopy(np.array(bankfeas))
    queryfeas = deepcopy(np.array(queryfeas))

    index = faiss.IndexFlatIP(bankfeas.shape[-1])
    index.add(bankfeas)
    D, _ = index.search(queryfeas, k)
    if min:
        scores = np.array(D.min(axis=1))
    else:
        scores = np.array(D.mean(axis=1))
    return scores


class NNGuidePostprocessor(BaseCILPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.alpha = self.args.alpha
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def setup(self, net_list, id_loader_dict, ood_loader_dict):
        [_, net] = net_list        
        net.eval()
        print('Extracting id validation neibor prior distributions')


        bank_feas = []
        bank_logits = []
        with torch.no_grad():
            for data, labels in tqdm(id_loader_dict['train'],
                                desc='Setup: ',
                                position=0,
                                leave=True):
                data = data.cuda()

                out = net(data, return_feature=True)
                logit, feature = out["logits"], out["features"]
                bank_feas.append(normalizer(feature.data.cpu().numpy()))
                bank_logits.append(logit.data.cpu().numpy())
                if len(bank_feas
                        ) * id_loader_dict['train'].batch_size > int(
                            len(id_loader_dict['train'].dataset) *
                            self.alpha):
                    break

        bank_feas = np.concatenate(bank_feas, axis=0)
        bank_confs = logsumexp(np.concatenate(bank_logits, axis=0),
                                axis=-1)
        self.bank_guide = bank_feas * bank_confs[:, None]

        

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):

        out = net(data, return_feature=True)
        logit, feature = out["logits"], out["features"]
        feas_norm = normalizer(feature.data.cpu().numpy())
        energy = logsumexp(logit.data.cpu().numpy(), axis=-1)

        conf = knn_score(self.bank_guide, feas_norm, k=self.K)
        score = conf * energy

        _, pred = torch.max(torch.softmax(logit, dim=1), dim=1)
        return pred, torch.from_numpy(score)
