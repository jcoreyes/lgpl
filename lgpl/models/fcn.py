import torch
import torch.nn as nn
import torch.nn.functional as F
from lgpl.models.weight_init import xavier_init
from lgpl.utils.torch_utils import gpu_enabled
import numpy as np
import torch.nn.init as init

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class FCN(nn.Module):
    def __init__(self, input_dim, output_dim, filter_sizes, upfilter_sizes,
                 hidden_act=nn.ReLU, output_act=None, mlp=None):
        super(FCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        net = nn.Sequential()
        prev_size = input_dim

        for i, filter_size in enumerate(filter_sizes):
            net.add_module(name='conv %d' %i,
                           module=nn.Conv2d(prev_size[0],
                                            filter_size[0],
                                            filter_size[1],
                           padding=(filter_size[1]-1)//2))
            net.add_module(name='conv relu %d' %i, module=hidden_act())
            prev_size = filter_size
            if i < 3:
                net.add_module(name='dropout fcn filter %d' % i, module=nn.Dropout(p=0.2))

        for i, upfilter_size in enumerate(upfilter_sizes):
            net.add_module(name='upfilter %d' %i,
                           module=nn.ConvTranspose2d(prev_size[0],
                                  upfilter_size[0],
                                  upfilter_size[1],
                           padding=(upfilter_size[1]-1)//2))
            if i < len(upfilter_size) - 1:
                net.add_module(name='upfilter relu %d' % i, module=hidden_act())
            prev_size = upfilter_size

        if output_act is not None:
            net.add_module(name='output_act', module=output_act(dim=1))

        if mlp is not None:
            net.add_module('flatten', Flatten())
            net.add_module('mlp', mlp)

        self.network = net

        self._initialize_weights()

        if gpu_enabled():
            self.cuda()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weight.data)

            if isinstance(m, nn.ConvTranspose2d):
                init.xavier_normal(m.weight.data)


    def forward(self, x):
        return self.network(x)

