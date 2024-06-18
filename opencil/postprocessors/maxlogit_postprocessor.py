from typing import Any

import torch
import torch.nn as nn
import pdb

from .base_postprocessor import BasePostprocessor, BaseCILPostprocessor

class MaxLogitCILPostprocessor(BaseCILPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)['logits']  
        conf, pred = torch.max(output, dim=1)
        return pred, conf