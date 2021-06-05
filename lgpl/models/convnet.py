import torch
import torch.nn as nn
import torch.nn.functional as F
from lgpl.models.weight_init import xavier_init
from lgpl.utils.torch_utils import gpu_enabled
import numpy as np
import torch.nn.init as init
from functools import reduce
from operator import mul

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Convnet(nn.Module):
    def __init__(self, input_dim, output_dim, filter_sizes, flat_dim, hidden_sizes=(32, 32),
                 hidden_act=nn.ReLU, output_act=None, batchnorm=False):
        super(Convnet, self).__init__()
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
            if batchnorm:
                net.add_module(name='bn %d' %i, module=nn.BatchNorm2d(filter_size[0]))
            net.add_module(name='conv act %d' %i, module=hidden_act())
            prev_size = filter_size


        net.add_module('flatten', Flatten())
        # because conv layers preserve input dim
        #prev_size = prev_size[0] * reduce(mul, input_dim[1:])
        #prev_size = 242
        prev_size = flat_dim
        for i, hidden_size in enumerate(hidden_sizes):
            net.add_module(name='linear %d' % i, module=nn.Linear(prev_size, hidden_size))
            net.add_module(name='act %d' % i, module=hidden_act())
            prev_size = hidden_size

        net.add_module(name='finallayer', module=nn.Linear(hidden_sizes[-1], output_dim))


        if output_act is not None:
            net.add_module(name='output_act', module=output_act())

        self.network = net

        self._initialize_weights()

        if gpu_enabled():
            self.cuda()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)

            if isinstance(m, nn.ConvTranspose2d):
                init.xavier_normal_(m.weight.data)

            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)


    def forward(self, x):
        return self.network(x)

class TemporalConvEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, filter_sizes, flat_size, hidden_sizes=(32, 32),
                 hidden_act=nn.ReLU, output_act=None, pool=False, batchnorm_input=False):
        super(TemporalConvEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        net = nn.Sequential()
        prev_size = input_dim

        if batchnorm_input:
            net.add_module('bn_input', nn.BatchNorm1d(prev_size[0]))


        for i, filter_size in enumerate(filter_sizes):
            net.add_module(name='conv %d' %i,
                           module=nn.Conv1d(prev_size[0],
                                            filter_size[0],
                                            filter_size[1],
                                            padding=0,
                                            stride=filter_size[1] // 2) # fixed at half kernel size for now
                           )
            if pool:
                net.add_module('pool %d' %i, module=nn.MaxPool1d(2)) # fixed at 2 kernel size for now
            net.add_module(name='conv act %d' %i, module=hidden_act())
            prev_size = filter_size
        net.add_module('flatten', Flatten())
        prev_size = flat_size
        for i, hidden_size in enumerate(hidden_sizes):
            net.add_module(name='linear %d' % i, module=nn.Linear(prev_size, hidden_size))
            net.add_module(name='act %d' % i, module=hidden_act())
            prev_size = hidden_size

        net.add_module(name='finallayer', module=nn.Linear(hidden_sizes[-1], output_dim))


        if output_act is not None:
            net.add_module(name='output_act', module=output_act(dim=1))

        self.network = net


        self.apply(xavier_init)


    def reset(self, bs):
        pass

    def forward(self, x):
        # x is (bs, seq_len, input_dim)
        # reshape to (bs, input_dim, seq_len)

        return self.network(x.transpose(1, 2))

class Sent1dConvEncoder(nn.Module):
    def __init__(self, conv_network, embeddings):
        super(Sent1dConvEncoder, self).__init__()
        self.conv_network = conv_network
        self.embeddings = embeddings

    def forward(self, x):
        # x is (bs, seq_len) of idxs
        x = self.embeddings(x.long())
        x = self.conv_network(x)

        return x


class Temporal2DConvEncoder(nn.Module):
    def __init__(self, temporal_encoder, mlp_network, conv_network, extra_obs_network, hidden_dim, conv_input_shape):
        super(Temporal2DConvEncoder, self).__init__()
        self.temporal_encoder = temporal_encoder
        self.mlp_network = mlp_network
        self.h_size = 1
        if self.rnn_network.bidirectional:
            self.h_size += 1
        self.h_size *= self.rnn_network.num_layers
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.conv_network = conv_network
        self.extra_obs_network = extra_obs_network
        self.conv_input_shape = conv_input_shape
        self.obs_len = int(np.prod(conv_input_shape))

        #self.apply(xavier_init)

    def forward(self, x, lens=None):
        bs, seq_len, dim = x.shape

        obs = x[:, :, :self.obs_len].float() / 10
        extra_obs = x[:, :, self.obs_len:].float()

        x1 = self.conv_network(obs.view(-1, self.obs_len).view(-1, *self.conv_input_shape)).view(bs, seq_len, -1)
        x1 = x1.permute(1, 0, 2)

        x2 = self.extra_obs_network(extra_obs.view(-1, extra_obs.shape[-1])).view(bs, seq_len, -1)
        x2 = x2.permute(1, 0, 2)
        x = torch.cat([x1, x2], -1)


        return self.temporal_encoder(x)

