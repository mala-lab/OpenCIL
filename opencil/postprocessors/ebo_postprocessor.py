from typing import Any

import torch
import torch.nn as nn
import pdb

from .base_postprocessor import BasePostprocessor, BaseCILPostprocessor

class EBOPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.temperature = self.args.temperature
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.temperature * torch.logsumexp(output / self.temperature,
                                                  dim=1)
        return pred, conf
    
    def set_hyperparam(self,  hyperparam:list):
        self.temperature =hyperparam[0] 
    
    def get_hyperparam(self):
        return self.temperature
    

class EBOCILPostprocessor(BaseCILPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.temperature = self.args.temperature
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)['logits'] 

        # output = torch.cat(outputs, dim=1)
        
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.temperature * torch.logsumexp(output / self.temperature,
                                                  dim=1)
        return pred, conf
    
    def set_hyperparam(self,  hyperparam:list):
        self.temperature =hyperparam[0] 
    
    def get_hyperparam(self):
        return self.temperature
    

class EBOCILFinetunePostprocessor(BaseCILPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.temperature = self.args.temperature
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)['aux_logits'] 
        # output = net(data, feature_operate="T2FNorm_test", feature_operate_parameter=0.04)["aux_logits"]

        # output = torch.cat(outputs, dim=1)
        
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.temperature * torch.logsumexp(output / self.temperature,
                                                  dim=1)
        return pred, conf
    
    def set_hyperparam(self,  hyperparam:list):
        self.temperature =hyperparam[0] 
    
    def get_hyperparam(self):
        return self.temperature
    