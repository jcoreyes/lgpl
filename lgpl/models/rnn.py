import torch
import torch.nn as nn
from lgpl.models.weight_init import xavier_init, orthogonal_init
from lgpl.utils.torch_utils import Variable, from_numpy, FloatTensor
import numpy as np

class RNN(nn.Module):
    def __init__(self, rnn_network, hidden_dim):
        super(RNN, self).__init__()
        self.rnn_network = rnn_network
        self.h_size = 1
        if self.rnn_network.bidirectional:
            self.h_size += 1
        self.h_size *= self.rnn_network.num_layers
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.apply(xavier_init)
        self.rnn_network.apply(orthogonal_init)

    def init_hidden(self, bs):
        self.hidden = (from_numpy(np.zeros(self.h_size, bs, self.hidden_dim)),
                       from_numpy(np.zeros(self.h_size, bs, self.hidden_dim)))

    def set_state(self, state):
        self.hidden = torch.split(state, self.h_size, 0)

    def get_state(self):
        return torch.cat(self.hidden, 0)

    def forward(self, x):
        # x is (seq_len, bs, input_dim)
        out, self.hidden = self.rnn_network(x, self.hidden)
        return out

class RNNEncoder(nn.Module):
    def __init__(self, rnn_network, mlp_network, hidden_dim):
        super(RNNEncoder, self).__init__()
        self.rnn_network = rnn_network
        self.mlp_network = mlp_network
        self.h_size = 1
        if self.rnn_network.bidirectional:
            self.h_size += 1
        self.h_size *= self.rnn_network.num_layers
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.apply(xavier_init)
        self.rnn_network.apply(orthogonal_init)

    def reset(self, bs):
        self.init_hidden(bs)

    def init_hidden(self, bs):
        self.hidden = (from_numpy(np.zeros((self.h_size, bs, self.hidden_dim))),
                       from_numpy(np.zeros((self.h_size, bs, self.hidden_dim))))

    def set_state(self, state):
        self.hidden = torch.split(state, self.h_size, 0)

    def get_state(self):
        return torch.cat(self.hidden, 0)

    def forward(self, x, lens=None):
        # x is (bs, seq_len, input_dim)
        if lens is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, list(lens), batch_first=False)
        #import pdb;
        #pdb.set_trace()
        #out, self.hidden = self.rnn_network(x, self.hidden)
        x = x.permute(1, 0, 2)
        out, _ = self.rnn_network(x, self.hidden)

        #import pdb; pdb.set_trace()
        if lens is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=False)
            lens = lens.long()
            out = out[lens-1, torch.arange(end=out.shape[1]), :]
            #out = out[-1, :, :]
        else:
            out = out[-1, :, :]

        return self.mlp_network(out)

class RNNConvEncoder(nn.Module):
    def __init__(self, rnn_network, mlp_network, conv_network, extra_obs_network, hidden_dim, conv_input_shape):
        super(RNNConvEncoder, self).__init__()
        self.rnn_network = rnn_network
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
        self.rnn_network.apply(orthogonal_init)

    def reset(self, bs):
        self.init_hidden(bs)

    def init_hidden(self, bs):
        self.hidden = (from_numpy(np.zeros((self.h_size, bs, self.hidden_dim))),
                       from_numpy(np.zeros((self.h_size, bs, self.hidden_dim))))

    def set_state(self, state):
        self.hidden = torch.split(state, self.h_size, 0)

    def get_state(self):
        return torch.cat(self.hidden, 0)

    def forward(self, x, lens=None):
        bs, seq_len, dim = x.shape

        #import pdb; pdb.set_trace()
        obs = x[:, :, :self.obs_len].float() / 10
        extra_obs = x[:, :, self.obs_len:].float()

        x1 = self.conv_network(obs.view(-1, self.obs_len).view(-1, *self.conv_input_shape)).view(bs, seq_len, -1)
        x1 = x1.permute(1, 0, 2)

        x2 = self.extra_obs_network(extra_obs.view(-1, extra_obs.shape[-1])).view(bs, seq_len, -1)
        x2 = x2.permute(1, 0, 2)
        x = torch.cat([x1, x2], -1)
        self.init_hidden(bs)
        out, _ = self.rnn_network(x, self.hidden)


        if self.rnn_network.bidirectional:
            # https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
            # might want to take out[-1, :, :self.hidden_dim] and out[0, :, self.hidden_dim:]
            out = self.mlp_network(out.mean(0))
        else:
            out = self.mlp_network(out[-1, :, :])
        return out

class RNNSentEncoder(nn.Module):
    def __init__(self, rnn_network, mlp_network, embeddings, hidden_dim, mean=False):
        super(RNNSentEncoder, self).__init__()
        self.rnn_network = rnn_network
        self.mlp_network = mlp_network
        self.h_size = 1
        if self.rnn_network.bidirectional:
            self.h_size += 1
        self.h_size *= self.rnn_network.num_layers
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.embeddings = embeddings
        self.mean = mean

        self.apply(xavier_init)
        self.rnn_network.apply(orthogonal_init)

    def reset(self, bs):
        self.init_hidden(bs)

    def init_hidden(self, bs):
        self.hidden = (from_numpy(np.zeros((self.h_size, bs, self.hidden_dim))),
                       from_numpy(np.zeros((self.h_size, bs, self.hidden_dim))))

    def set_state(self, state):
        self.hidden = torch.split(state, self.h_size, 0)

    def get_state(self):
        return torch.cat(self.hidden, 0)

    def forward(self, x, lens=None):
        # x is (bs, seq_len) of word indices
        bs = x.shape[0]
        x = self.embeddings(x.long())
        x = x.permute(1, 0, 2)
        self.init_hidden(bs)
        out, _ = self.rnn_network(x, self.hidden)

        if self.mean:
            return out.mean(0)

        if self.rnn_network.bidirectional:
            out = self.mlp_network(out.mean(0))
        else:
            out = self.mlp_network(out[-1, :, :])

        return out

class MeanEncoder(nn.Module):
    def __init__(self, rnn_network, mlp_network, encoder, hidden_dim, type='mean'):
        super().__init__()
        self.rnn_network = rnn_network
        self.mlp_network = mlp_network
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.h_size = 1
        if self.rnn_network.bidirectional:
            self.h_size += 1
        self.type = type

    def init_hidden(self, bs):
        self.hidden = (from_numpy(np.zeros((self.h_size, bs, self.hidden_dim))),
                       from_numpy(np.zeros((self.h_size, bs, self.hidden_dim))))

    def forward(self, x, lens):
        # x is (bs, max_len, dim)
        # lens is (bs,) and specifies length
        #return self.mlp_network(x[:, 0, :])

        x_shape = x.shape
        bs = x_shape[0]
        if self.type == 'mean':

            x = self.encoder(x.view(-1, x_shape[-1])).view((bs, x_shape[1], -1))

            for i in range(x_shape[1]):
                x[lens<i+1, i, :] = 0
            out = x.sum(1) / lens.view(bs, 1).float()
        elif self.type == 'last':
            #x = self.encoder(x.view(-1, x_shape[-1])).view((bs, x_shape[1], -1))
            out = self.encoder(x[torch.arange(end=bs), lens.long()-1, :])
            #x = x.permute(1, 0, 2)
            # x = x.unsqueeze(1)
            # self.init_hidden(bs)
            # out, _ = self.rnn_network(x, self.hidden)
            # out = out[0, :, :]
            # out = self.mlp_network(out)


        else:
            x = self.encoder(x.view(-1, x_shape[-1])).view((bs, x_shape[1], -1))
            x = x.permute(1, 0, 2)
            self.init_hidden(bs)
            out, _ = self.rnn_network(x, self.hidden)
            out = out[lens.long() - 1, torch.arange(end=out.shape[1]), :]
            out = self.mlp_network(out)

        return out

class MeanEncoder2(nn.Module):
    def __init__(self, encoder, prev_traj_encoder, correction_encoder):
        super().__init__()
        self.encoder = encoder
        self.prev_traj_encoder = prev_traj_encoder
        self.correction_encoder = correction_encoder


    def forward(self, prev_trajs, corrections, lens):
        # x is (bs, max_len, dim)
        # lens is (bs,) and specifies length
        #return self.mlp_network(x[:, 0, :])

        x_shape = corrections.shape
        bs = x_shape[0]
        x = corrections
        x = self.correction_encoder(x.view(-1, x_shape[-1])).view(bs, x_shape[1], -1)

        # x2 = prev_trajs
        # x2_shape = x2.shape
        # x2 = self.prev_traj_encoder(prev_trajs.view(-1, x2_shape[-1]))
        #
        #
        # x3 = self.encoder(torch.cat([x, x2], -1)).view(bs, x_shape[1], -1)


        for i in range(x_shape[1]):
            x[lens<i+1, i, :] = 0
        #if bs > 1 and lens.max() != lens.min():
        #    import ipdb;
        #    ipdb.set_trace()
        out = x.sum(1) / lens.view(bs, 1).float()

        return out

class CorrectionTrajEncoder(nn.Module):
    def __init__(self, encoder, traj_encoder, correction_encoder, obs_dim, obs_net=None, obs_shape=None):
        super().__init__()
        self.encoder = encoder
        self.traj_encoder = traj_encoder
        self.correction_encoder = correction_encoder
        self.obs_dim = obs_dim
        self.obs_net = obs_net
        self.obs_shape = obs_shape
        self.obs_len = np.prod(self.obs_shape)

    def forward(self, prev_trajs, corrections, lens):
        # prev_trajs is (bs, max_corrections, obs_dim * traj_len)
        bs, n_c, c_dim = corrections.shape

        x1 = self.correction_encoder(corrections.view(-1, c_dim))

        if self.obs_net is not None:
            x2 = self.obs_net(prev_trajs.view(-1, self.obs_dim)[:, :self.obs_len].view(-1, *self.obs_shape))
            x2 = self.traj_encoder(x2.view(bs*n_c, -1, x2.shape[-1]))
        else:
            x2 = self.traj_encoder(prev_trajs.view(bs*n_c, -1, self.obs_dim))

        x = torch.cat([x1, x2], -1)
        x = self.encoder(x).view(bs, n_c, -1)

        for i in range(n_c):
            x[lens<i+1, i, :] = 0
        #if bs > 1 and lens.max() != lens.min():
        #    import ipdb;
        #    ipdb.set_trace()
        out = x.sum(1) / lens.view(bs, 1).float()

        return out