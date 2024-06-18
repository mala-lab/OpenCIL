import copy
import logging
import torch
import pdb
from torch import nn
import torch.nn.functional as F
from .resnet32_pycil import resnet32
from .resnet18_224x224_pycil import resnet18_224x224
from .resnet18_cosine_pycil import resnet18 as cosine_resnet18
from .linears_pycil import SimpleLinear, SplitCosineLinear, CosineLinear


def get_convnet(config, pretrained=False):
    backbone = config.network.backbone.name
    if backbone == "resnet32":
        return resnet32()
    elif backbone == "resnet18_224x224":
        return resnet18_224x224(config=config)
    elif backbone == "cosine_resnet18":
        return cosine_resnet18(config=config)
    else:
        raise NotImplementedError("Unknown type {}".format(backbone))

class BaseNet(nn.Module):
    def __init__(self, config, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(config, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet.forward_cil(x)["features"]

    def forward(self, x, return_feature=False, return_feature_list=False):
        x = self.convnet.forward_cil(x, return_feature, return_feature_list)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        if return_feature_list:
            out["feature_list"] = x["feature_list"]
        if return_feature:
            out["features"] = x["features"]
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class IncrementalNet(BaseNet):
    def __init__(self, config, pretrained, gradcam=False):
        super().__init__(config, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x, return_feature=False, return_feature_list=False, feature_operate=None, feature_operate_parameter=None):
        x = self.convnet.forward_cil(x, return_feature, return_feature_list)

        if feature_operate == "ReAct":
            x["features"] = x["features"].clip(max=feature_operate_parameter)
        else:
            pass
            
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        if return_feature_list:
            out["feature_list"] = x["feature_list"]
        if return_feature:
            out["features"] = x["features"]
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, config, pretrained, bias_correction=False):
        super().__init__(config, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x, return_feature=False, return_feature_list=False, feature_operate=None, feature_operate_parameter=None):
        x = self.convnet.forward_cil(x, return_feature, return_feature_list)


        if feature_operate == "ReAct":
            x["features"] = x["features"].clip(max=feature_operate_parameter)
        else:
            pass



        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        if return_feature_list:
            out["feature_list"] = x["feature_list"]

        if return_feature:
            out["features"] = x["features"]

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class FOSTERNet(nn.Module):
    def __init__(self, config, pretrained):
        super(FOSTERNet, self).__init__()
        self.convnet_type = config.network.backbone.name
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.backbone = self.convnet_type

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet.forward_cil(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x, return_feature=False, return_feature_list=False, feature_operate=None, feature_operate_parameter=None):
        all_out = [convnet.forward_cil(x, return_feature, return_feature_list) for convnet in self.convnets]
        
        features = [out_s["features"] for out_s in all_out]
        features = torch.cat(features, 1)

        if feature_operate == "ReAct":
            features = features.clip(max=feature_operate_parameter)
        else:
            pass

        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        if return_feature_list: # constructing feature list
            feature_list = [out_s["feature_list"] for out_s in all_out] # each element is a list of feature

            # concatenate these features
            num_feature = len(feature_list[0])

            concatenate_feature_list = []
            for i in range(num_feature):
                feature_i = []
                for j in range(len(feature_list)):
                    feature_i.append(feature_list[j][i])
                feature_i = torch.cat(feature_i, 1)
                concatenate_feature_list.append(feature_i)
            out["feature_list"] = concatenate_feature_list
        if return_feature:
            out["features"] = features
        return out

    def update_fc(self, nb_classes, config):
        self.convnets.append(get_convnet(config))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma




class FinetuneNet_icarl(nn.Module):
    def __init__(self, config, pretrained):
        super(FinetuneNet_icarl, self).__init__()
        self.convnet = get_convnet(config, pretrained)
        self.out_dim = None
        self.fc = None
        self.aux_fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet.forward_cil(x)["features"]

    def forward(self, x, return_feature=False, return_feature_list=False, feature_operate=None, feature_operate_parameter=None):
        x = self.convnet.forward_cil(x, return_feature, return_feature_list)


        if feature_operate == "T2FNorm_train":
            x["features"] = F.normalize(x["features"], dim=-1) / feature_operate_parameter
        elif feature_operate == "T2FNorm_test":
            x["features"] = x["features"] / feature_operate_parameter
        else:
            pass

        out = self.fc(x["features"])
        aux_out = self.aux_fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)
        out.update(aux_logits = aux_out["logits"])
        

        if return_feature_list:
            out["feature_list"] = x["feature_list"]
        if return_feature:
            out["features"] = x["features"]
        return out

    def update_fc(self, nb_classes, config):
        self.fc = self.generate_fc(self.feature_dim, nb_classes)
        self.aux_fc = self.generate_fc(self.feature_dim, nb_classes)
    
    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

class FinetuneNet_foster(nn.Module):
    def __init__(self, config, pretrained):
        super(FinetuneNet_foster, self).__init__()
        self.convnets = nn.ModuleList()
        self.out_dim = None
        self.fc = None
        self.aux_fc = None

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet.forward_cil(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x, return_feature=False, return_feature_list=False, feature_operate=None, feature_operate_parameter=None):
        all_out = [convnet.forward_cil(x, return_feature, return_feature_list) for convnet in self.convnets]

        features = [out_s["features"] for out_s in all_out]
        features = torch.cat(features, 1)


        if feature_operate == "T2FNorm_train":
            features = F.normalize(features, dim=-1) / feature_operate_parameter
        elif feature_operate == "T2FNorm_test":
            features = features / feature_operate_parameter
        else:
            pass

        out = self.fc(features)
        aux_out = self.aux_fc(features)
        
        out.update(aux_logits = aux_out["logits"])
        out.update({"features": features})
        out.update({"eval_logits": out["logits"]})

        return out

    def update_fc(self, nb_classes, config):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(config))
        elif len(self.convnets) == 1:
            self.convnets.append(get_convnet(config))
        else:
            pass
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim

        self.fc = self.generate_fc(self.feature_dim, nb_classes)
        self.aux_fc = self.generate_fc(self.feature_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

class FinetuneNet_bic(BaseNet):
    def __init__(self, config, pretrained, bias_correction=True):
        super().__init__(config, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x, return_feature=False, return_feature_list=False, feature_operate=None, feature_operate_parameter=None):
        x = self.convnet.forward_cil(x, return_feature, return_feature_list)

        if feature_operate == "T2FNorm_train":
            x["features"] = F.normalize(x["features"], dim=-1) / feature_operate_parameter
        elif feature_operate == "T2FNorm_test":
            x["features"] = x["features"] / feature_operate_parameter
        else:
            pass


        out = self.fc(x["features"])
        aux_out = self.aux_fc(x["features"])

        if self.bias_correction:
            logits = out["logits"]
            aux_logits = aux_out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
                aux_logits = layer(
                    aux_logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits
            aux_out["logits"] = aux_logits

        out.update(x)
        out.update(aux_logits = aux_out["logits"])

        if return_feature_list:
            out["feature_list"] = x["feature_list"]

        if return_feature:
            out["features"] = x["features"]

        return out

    def update_fc(self, nb_classes, config):
        self.fc = self.generate_fc(self.feature_dim, nb_classes)
        self.aux_fc = self.generate_fc(self.feature_dim, nb_classes)

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

class FinetuneNet_wa(BaseNet):
    def __init__(self, config, pretrained, gradcam=False):
        super().__init__(config, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

        self.aux_fc = None

    def update_fc(self, nb_classes, config):
        self.fc = self.generate_fc(self.feature_dim, nb_classes)
        self.aux_fc = self.generate_fc(self.feature_dim, nb_classes)

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x, return_feature=False, return_feature_list=False, feature_operate=None, feature_operate_parameter=None):
        x = self.convnet.forward_cil(x, return_feature, return_feature_list)

        if feature_operate == "T2FNorm_train":
            x["features"] = F.normalize(x["features"], dim=-1) / feature_operate_parameter
        elif feature_operate == "T2FNorm_test":
            x["features"] = x["features"] / feature_operate_parameter
        else:
            pass

            
        out = self.fc(x["features"])
        aux_out = self.aux_fc(x["features"])
        out.update(x)
        out.update(aux_logits = aux_out["logits"])
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        if return_feature_list:
            out["feature_list"] = x["feature_list"]
        if return_feature:
            out["features"] = x["features"]
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )