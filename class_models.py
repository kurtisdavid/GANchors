import torch
import torch.nn as nn
from torchvision.models import resnet18
from collections import OrderedDict
import copy

# based off of https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py
#   only removed prints, so that we can extract output shape
def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).to(device) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return summary

class celebAModel(nn.Module):

    def __init__(self, input_shape, device='cpu'):
        super(celebAModel,self).__init__()
        self.device = device
        self.input_shape = input_shape
        model = resnet18(pretrained=True).to(self.device)
        children = list(model.children())
        tmp = nn.Sequential(*(children[:-2]))
        out_shape = self.extract_output(copy.deepcopy(tmp))
        self.feature_extract = nn.Sequential(*(copy.deepcopy(tmp)),
                                       nn.AvgPool2d(kernel_size = (out_shape[2],out_shape[3]), stride = 1))
        self.layers = nn.Sequential(
            nn.Linear(in_features = out_shape[1], out_features = 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features = 64, out_features = 2)
        )
        del tmp

    def forward(self, X):
        X = self.feature_extract(X).view(X.shape[0],-1)
        return self.layers(X)

    def extract_output(self, model):
        summary_ = summary(model, self.input_shape, device=self.device)
        return summary_[list(summary_)[-1]]['output_shape']
