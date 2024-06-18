from types import MethodType

import mmcv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from mmcls.apis import init_model

import opencil.utils.comm as comm

from .resnet18_32x32 import ResNet18_32x32
from .resnet18_64x64 import ResNet18_64x64
from .resnet18_224x224 import ResNet18_224x224
from .resnet18_256x256 import ResNet18_256x256
from .resnet32 import ResNet32, resnet32
from .resnet50 import ResNet50
from .wrn import WideResNet
from .incremental_net import LLL_Net


def get_cil_network(init_model):
    il_network = LLL_Net(init_model, remove_existing_head=True) # truly important, it produces powerful feature vectors
    return il_network


def get_network(network_config, num_initial_classes=None):

    num_classes = network_config.num_classes

    if network_config.name == 'resnet18_32x32':
        net = ResNet18_32x32(num_classes=num_classes)
    
    elif network_config.name == 'resnet32':
        if num_initial_classes is not None:
            # allocate specialized treatment network for cil
            net = resnet32(num_classes=num_initial_classes)
        else:
            net = resnet32(num_classes=num_classes)
    
    elif network_config.name == 'incremental_resnet18_32x32':
        net = ResNet18_32x32(num_classes=num_initial_classes)

    elif network_config.name == 'resnet18_256x256':
        net = ResNet18_256x256(num_classes=num_classes)

    elif network_config.name == 'resnet18_64x64':
        net = ResNet18_64x64(num_classes=num_classes)

    elif network_config.name == 'resnet18_224x224':
        net = ResNet18_224x224(num_classes=num_classes)

    elif network_config.name == 'resnet50':
        net = ResNet50(num_classes=num_classes)

    elif network_config.name == 'wrn':
        net = WideResNet(depth=28,
                         widen_factor=10,
                         dropRate=0.0,
                         num_classes=num_classes)

    elif network_config.name == 'vit':
        cfg = mmcv.Config.fromfile(network_config.model)
        net = init_model(cfg, network_config.checkpoint, 0)
        net.get_fc = MethodType(
            lambda self: (self.head.layers.head.weight.cpu().numpy(),
                          self.head.layers.head.bias.cpu().numpy()), net)


    if network_config.pretrained:
        if type(net) is dict:
            for subnet, checkpoint in zip(net.values(),
                                          network_config.checkpoint):
                if checkpoint is not None:
                    if checkpoint != 'none':
                        subnet.load_state_dict(torch.load(checkpoint),
                                               strict=False)
        elif network_config.name == 'bit' and not network_config.normal_load:
            net.load_from(np.load(network_config.checkpoint))
        elif network_config.name == 'vit':
            pass
        else:
            try:
                net.load_state_dict(torch.load(network_config.checkpoint),
                                    strict=False)
            except RuntimeError:
                # sometimes fc should not be loaded
                loaded_pth = torch.load(network_config.checkpoint)
                loaded_pth.pop('fc.weight')
                loaded_pth.pop('fc.bias')
                net.load_state_dict(loaded_pth, strict=False)
        print('Model Loading {} Completed!'.format(network_config.name))
    if network_config.num_gpus > 1:
        if type(net) is dict:
            for key, subnet in zip(net.keys(), net.values()):
                net[key] = torch.nn.parallel.DistributedDataParallel(
                    subnet,
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=True)
        else:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(),
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=True)

    if network_config.num_gpus > 0:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()
        torch.cuda.manual_seed(1)
        np.random.seed(1)
    cudnn.benchmark = True
    return net
