import torch.nn as nn
import torch
import numpy as np
from lgpl.math.distributions.normal import Normal
from lgpl.policies.policy import Policy
from lgpl.utils.torch_utils import np_to_var
from lgpl.models.weight_init import xavier_init

class GaussianMLPPolicy(Policy):
    def __init__(self, mean_network, log_var_network,
                 min_var=1e-6, obs_filter=None, init=xavier_init):
        super().__init__()
        self.mean_network = mean_network
        self.log_var_network = log_var_network

        self.obs_filter = obs_filter
        self.min_log_var = np_to_var(np.log(np.array([min_var])).astype(np.float32))

        self.apply(init)
        # # self.apply(weights_init_mlp)
        # if scale_final:
        #     if hasattr(self.mean_network, 'network'):
        #         self.mean_network.network.finallayer.weight.data.mul_(0.01)


    def forward(self, x):
        if type(x) is np.ndarray:
            x = np_to_var(x)
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        mean, log_var = self.mean_network(x), self.log_var_network(x)
        log_var = torch.max(self.min_log_var, log_var)
        dist = Normal(mean=mean, log_var=log_var)
        return dist

class GaussianRNNPolicy(nn.Module):

    def __init__(self, rnn_module, mlp_policy):
        super(GaussianRNNPolicy, self).__init__()
        self.rnn_module = rnn_module
        self.mpl_policy = mlp_policy



    def forward(self, rnn_input, mlp_input):
        # rnn_input is (seq_len, bs, input_dim)
        rnn_output = self.rnn_module.forward(rnn_input)
        combined_input = torch.cat([rnn_output, mlp_input], 1)
        return self.mpl_policy.foward(combined_input)

class GaussianBaselinePolicy(Policy):
    def __init__(self, obs_shape,
                 mean_network, log_var_network,
                 output_dim, min_var=1e-6):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)


        self.mean_network = mean_network
        self.log_var_network = log_var_network
        self.output_dim = output_dim # Used for sampler
        self.min_log_var = np_to_var(np.log(np.array([min_var])).astype(np.float32))

    def reset(self, shape):
        self.cache_inputs = []

    def forward(self, x, corrections):

        # X will be flat and contain extra_dim (if obj picked up)
        # prev_traj is just last state for now
        bs = x.shape[0]

        #obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))

        out = torch.cat([x, corrections], -1)
        mean, log_var = self.mean_network(out), self.log_var_network(out)
        log_var = torch.max(self.min_log_var, log_var)
        dist = Normal(mean=mean, log_var=log_var)
        return dist

class GaussianBaselinePolicy2(Policy):
    def __init__(self, obs_shape,
                 mean_network, log_var_network,
                 output_dim, min_var=1e-6):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)


        self.mean_network = mean_network
        self.log_var_network = log_var_network
        self.output_dim = output_dim # Used for sampler
        self.min_log_var = np_to_var(np.log(np.array([min_var])).astype(np.float32))

    def reset(self, shape):
        self.cache_inputs = []

    def forward(self, x, corrections):

        # X will be flat and contain extra_dim (if obj picked up)
        # prev_traj is just last state for now
        bs = x.shape[0]

        #obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))

        out = torch.cat([x, corrections], -1)
        mean, log_var = self.mean_network(out), self.log_var_network(out)
        log_var = torch.max(self.min_log_var, log_var)
        dist = Normal(mean=mean, log_var=log_var)
        return dist

class GaussianTrajCorrectionPolicy(Policy):
    def __init__(self, obs_shape, obs_network, prev_traj_encoder,
                 mean_network, log_var_network, correction_encoder, hl_goal_encoder, traj_correction_encoder,
                 output_dim, obs_filter=None, min_var=1e-6):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)

        self.obs_network = obs_network
        self.correction_encoder = correction_encoder
        self.mean_network = mean_network
        self.log_var_network = log_var_network
        self.output_dim = output_dim # Used for sampler
        self.prev_traj_encoder = prev_traj_encoder
        self.hl_goal_encoder = hl_goal_encoder
        self.traj_correction_encoder = traj_correction_encoder
        self.obs_filter = obs_filter
        self.min_log_var = np_to_var(np.log(np.array([min_var])).astype(np.float32))

    def reset(self, shape):
        self.cache_inputs = []

    def forward(self, x, corrections, correction_lens, prev_trajs_lst, prev_traj_lens, hl_goal,
                prev_traj_enc=None, hl_goal_enc=None, correction_enc=None):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        # X will be flat and contain extra_dim (if obj picked up)
        # prev_traj is just last state for now
        bs = x.shape[0]

        #obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))
        obs = x

        if obs.dtype == np.float32:
            obs = np_to_var(obs)

        x1 = self.obs_network(obs)
        x2 = self.hl_goal_encoder(hl_goal) if len(self.cache_inputs) == 0 else self.cache_inputs[0]

        x3 = self.prev_traj_encoder(prev_trajs_lst) if len(self.cache_inputs) == 0 else self.cache_inputs[1]

        x4 = self.correction_encoder(corrections, correction_lens) if len(self.cache_inputs) == 0 else self.cache_inputs[2]

        self.cache_inputs = [x2, x3, x4]

        x5 = self.traj_correction_encoder(torch.cat([x3, x4], -1))
        out = torch.cat([x1, x2, x5], -1)

        mean, log_var = self.mean_network(out), self.log_var_network(out)
        log_var = torch.max(self.min_log_var, log_var)
        dist = Normal(mean=mean, log_var=log_var)
        return dist


class GaussianTrajCorrectionPolicy3(Policy):
    def __init__(self, obs_shape, obs_network, prev_traj_encoder,
                 mean_network, log_var_network, correction_encoder, hl_goal_encoder, traj_correction_encoder,
                 output_dim, obs_filter=None, min_var=1e-6):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)

        self.obs_network = obs_network
        self.correction_encoder = correction_encoder
        self.mean_network = mean_network
        self.log_var_network = log_var_network
        self.output_dim = output_dim # Used for sampler
        self.prev_traj_encoder = prev_traj_encoder
        self.hl_goal_encoder = hl_goal_encoder
        self.traj_correction_encoder = traj_correction_encoder
        self.obs_filter = obs_filter
        self.min_log_var = np_to_var(np.log(np.array([min_var])).astype(np.float32))

    def reset(self, shape):
        self.cache_inputs = []

    def forward(self, x, corrections, correction_lens, prev_trajs_lst, prev_traj_lens, hl_goal,
                prev_traj_enc=None, hl_goal_enc=None, correction_enc=None):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        # X will be flat and contain extra_dim (if obj picked up)
        # prev_traj is just last state for now
        bs = x.shape[0]

        #obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))
        obs = x

        if obs.dtype == np.float32:
            obs = np_to_var(obs)

        x1 = self.obs_network(obs)
        x2 = self.hl_goal_encoder(hl_goal) if len(self.cache_inputs) == 0 else self.cache_inputs[0]

        x3 = self.traj_correction_encoder(prev_trajs_lst, corrections, correction_lens) if len(self.cache_inputs) == 0 else self.cache_inputs[1]

        self.cache_inputs = [x2, x3]

        out = torch.cat([x1, x2, x3], -1)

        mean, log_var = self.mean_network(out), self.log_var_network(out)
        log_var = torch.max(self.min_log_var, log_var)
        dist = Normal(mean=mean, log_var=log_var)
        return dist