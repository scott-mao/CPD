import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorly as tl
from tensorly.decomposition import parafac, tucker, partial_tucker

def cp_decomp(layer, rank):
    W = layer.weight.data

    last, first, vertical, horizontal = parafac(W, rank=rank, init='random')
    
    pointwise_s_to_r_layer = nn.Conv2d(in_channels=first.shape[0],
                                       out_channels=first.shape[1],
                                       kernel_size=1,
                                       padding=0,
                                       bias=False)

    depthwise_r_to_r_layer = nn.Conv2d(in_channels=rank,
                                       out_channels=rank,
                                       kernel_size=vertical.shape[0],
                                       stride=layer.stride,
                                       padding=layer.padding,
                                       dilation=layer.dilation,
                                       groups=rank,
                                       bias=False)
                                       
    pointwise_r_to_t_layer = nn.Conv2d(in_channels=last.shape[1],
                                       out_channels=last.shape[0],
                                       kernel_size=1,
                                       padding=0,
                                       bias=True)
    
    if layer.bias is not None:
        pointwise_r_to_t_layer.bias.data = layer.bias.data

    sr = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    rt = last.unsqueeze_(-1).unsqueeze_(-1)
    rr = torch.stack([vertical.narrow(1, i, 1) @ torch.t(horizontal).narrow(0, i, 1) for i in range(rank)]).unsqueeze_(1)

    pointwise_s_to_r_layer.weight.data = sr 
    pointwise_r_to_t_layer.weight.data = rt
    depthwise_r_to_r_layer.weight.data = rr

    new_layers = [pointwise_s_to_r_layer,
                  depthwise_r_to_r_layer, pointwise_r_to_t_layer]
    return new_layers


def tucker_decomp(layer, rank):
    W = layer.weight.data

    # TODO: find how to init when SVD already computed
    # http://tensorly.org/stable/_modules/tensorly/decomposition/_tucker.html
    core, [last, first] = partial_tucker(W, modes=[0,1], rank=rank, init='svd')

    first_layer = nn.Conv2d(in_channels=first.shape[0],
                                       out_channels=first.shape[1],
                                       kernel_size=1,
                                       padding=0,
                                       bias=False)

    core_layer = nn.Conv2d(in_channels=core.shape[1],
                                       out_channels=core.shape[0],
                                       kernel_size=layer.kernel_size,
                                       stride=layer.stride,
                                       padding=layer.padding,
                                       dilation=layer.dilation,
                                       bias=False)

    last_layer = nn.Conv2d(in_channels=last.shape[1],
                                       out_channels=last.shape[0],
                                       kernel_size=1,
                                       padding=0,
                                       bias=True)
    
    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    fk = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    lk = last.unsqueeze_(-1).unsqueeze_(-1)

    first_layer.weight.data = fk
    last_layer.weight.data = lk
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return new_layers


