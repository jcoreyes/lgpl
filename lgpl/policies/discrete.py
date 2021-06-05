import torch.nn as nn
from lgpl.math.distributions.categorical import Categorical as irlCategorical
from torch.distributions import Categorical
from lgpl.policies.policy import Policy
import torch
import numpy as np
from lgpl.utils.torch_utils import np_to_var


class CategoricalNetwork(Policy):
    def __init__(self, prob_network, output_dim, obs_filter=None):
        super().__init__()
        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        self.modules = [self.prob_network]
        self.obs_filter = obs_filter

    def reset(self, shape):
        pass

    def forward(self, x):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        prob = self.prob_network(x)
        dist = irlCategorical(prob)
        return dist

class CategoricalMinigetNetwork(Policy):
    def __init__(self, obs_network, extra_obs_network, prob_network, output_dim, obs_len, obs_shape):
        super().__init__()
        self.obs_network = obs_network
        self.extra_obs_network = extra_obs_network
        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        self.obs_len = obs_len
        self.obs_shape = obs_shape
        #self.modules = [self.prob_network]
        #self.obs_filter = obs_filter

    def reset(self, shape):
        pass

    def forward(self, x):
        bs = x.shape[0]
        obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))
        extra_obs = x[:, self.obs_len:]

        x1 = self.obs_network(obs)
        x2 = self.extra_obs_network(extra_obs)
        prob = self.prob_network(torch.cat([x1, x2], -1))
        dist = irlCategorical(prob)
        return dist

class CategoricalLanguageNetwork(Policy):
    def __init__(self, state_network, language_network, prob_network,
                 output_dim, obs_filter=None):
        super().__init__()
        self.state_network = state_network
        self.language_network = language_network
        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        self.modules = [state_network, language_network, prob_network]
        self.obs_filter = obs_filter

    def reset(self, shape):
        pass

    def forward(self, x, language):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        state_vec = self.state_network(x)
        language_vec = self.language_network(x)
        prob = self.prob_network(torch.cat([state_vec, language_vec], -1))
        dist = Categorical(prob)
        return dist

class CategoricalSubgoalNetwork(Policy):
    def __init__(self, obs_shape, extra_dim, obs_network, extra_obs_network, language_network,
                 prob_network, output_dim, obs_filter=None):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)
        self.extra_dim = extra_dim
        self.obs_network = obs_network
        self.extra_obs_network = extra_obs_network
        self.language_network = language_network
        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        #self.modules = [self.prob_network]
        self.obs_filter = obs_filter

    def reset(self, shape):
        pass

    def forward(self, x, prev_traj, language):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        # X will be flat and contain extra_dim (if obj picked up)
        # prev_traj is just last state for now
        bs = x.shape[0]

        obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))
        extra_obs = x[:, self.obs_len:]

        prev_traj = prev_traj.reshape((bs, *self.obs_shape))


        if obs.dtype == np.float32:
            obs = np_to_var(obs)
        if extra_obs.dtype == np.float32:
            extra_obs = np_to_var(extra_obs)
        combined_obs = torch.cat([obs, prev_traj], 1)

        #combined_obs = np_to_var(np.zeros((1, 10, 13, 13), dtype=np.float32))
        #extra_obs = np_to_var(np.zeros((1, 10), dtype=np.float32))
        #import pdb; pdb.set_trace()
        x1 = self.obs_network(combined_obs / 10)
        x2 = self.extra_obs_network(extra_obs)

        language_vec = self.language_network(language)


        prob = self.prob_network(torch.cat([x1, x2, language_vec], -1))
        dist = Categorical(prob)
        return dist

class CategoricalBaselineNetwork(Policy):
    def __init__(self, obs_shape, extra_dim, obs_network, extra_obs_network,
                 prob_network, hl_goal_encoder, sent_encoder, output_dim, obs_filter=None):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)
        self.extra_dim = extra_dim
        self.obs_network = obs_network
        self.extra_obs_network = extra_obs_network

        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        self.hl_goal_encoder = hl_goal_encoder
        self.sent_encoder = sent_encoder
        #self.modules = [self.prob_network]
        self.obs_filter = obs_filter

    def reset(self, shape):
        pass
    # x, hl_goal, corrections, prev_traj, lens
    def forward(self,  x, hl_goal, corrections, prev_traj, lens):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        # X will be flat and contain extra_dim (if obj picked up)
        # prev_traj is just last state for now
        bs = x.shape[0]

        obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))
        extra_obs = x[:, self.obs_len:]


        if obs.dtype == np.float32:
            obs = np_to_var(obs)
        if extra_obs.dtype == np.float32:
            extra_obs = np_to_var(extra_obs)

        x1 = self.obs_network(obs)
        x2 = self.extra_obs_network(extra_obs)

        #import pdb; pdb.set_trace()
        x3 = self.sent_encoder(hl_goal)

        #bs, c_len, c_dim = corrections.shape

        prob = self.prob_network(torch.cat([x1, x2, x3], -1))
        #dist = Categorical(prob)
        dist = irlCategorical(prob)
        return dist

class CategoricalBaselineMinigridMisraNetwork(Policy):
    def __init__(self, obs_shape, extra_dim, obs_network, extra_obs_network,
                 prob_network, hl_goal_encoder, sent_encoder, output_dim, obs_filter=None):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)
        self.extra_dim = extra_dim
        self.obs_network = obs_network
        self.extra_obs_network = extra_obs_network

        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        self.hl_goal_encoder = hl_goal_encoder
        self.sent_encoder = sent_encoder
        #self.modules = [self.prob_network]
        self.obs_filter = obs_filter
        self.K = 2

    def reset(self, shape):
        pass
    # x, hl_goal, corrections, prev_traj, lens
    def forward(self,  x, hl_goal, corrections, prev_traj, lens):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        # X will be flat and contain extra_dim (if obj picked up)
        # prev_traj is just last state for now
        bs = x.shape[0]

        prev_action = x[:, -self.output_dim:]
        x = x[:, :-self.output_dim].reshape((bs, self.K, -1))
        obs = x[:, :, :self.obs_len].reshape((bs, self.obs_shape[0] * self.K, self.obs_shape[1], self.obs_shape[2]))
        extra_obs = x[:, :, self.obs_len:].reshape((bs, -1))


        if obs.dtype == np.float32:
            obs = np_to_var(obs)
        if extra_obs.dtype == np.float32:
            extra_obs = np_to_var(extra_obs)

        x1 = self.obs_network(obs)

        x2 = self.extra_obs_network(torch.cat([extra_obs, prev_action], -1))
        x3 = self.sent_encoder(hl_goal)

        prob = self.prob_network(torch.cat([x1, x2, x3], -1))
        #dist = Categorical(prob)
        dist = Categorical(prob)
        return dist

class CategoricalCorrectionNetwork(Policy):
    def __init__(self, obs_shape, extra_dim, obs_network, extra_obs_network, language_network,
                 prob_network, context_encoder, hl_goal_encoder, output_dim, obs_filter=None):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)
        self.extra_dim = extra_dim
        self.obs_network = obs_network
        self.extra_obs_network = extra_obs_network
        self.language_network = language_network
        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        self.context_encoder = context_encoder
        self.hl_goal_encoder = hl_goal_encoder
        #self.modules = [self.prob_network]
        self.obs_filter = obs_filter

    def reset(self, shape):
        pass

    def forward(self, x, corrections, correction_lens, hl_goal):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        # X will be flat and contain extra_dim (if obj picked up)
        # prev_traj is just last state for now
        bs = x.shape[0]

        obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))
        extra_obs = x[:, self.obs_len:]

        #prev_traj = prev_traj.reshape((bs, *self.obs_shape))


        if obs.dtype == np.float32:
            obs = np_to_var(obs)
        if extra_obs.dtype == np.float32:
            extra_obs = np_to_var(extra_obs)
        #combined_obs = torch.cat([obs, prev_traj], 1)

        #combined_obs = np_to_var(np.zeros((1, 10, 13, 13), dtype=np.float32))
        #extra_obs = np_to_var(np.zeros((1, 10), dtype=np.float32))
        #import pdb; pdb.set_trace()
        x1 = self.obs_network(obs / 10)
        x2 = self.extra_obs_network(extra_obs)
        x3 = self.hl_goal_encoder(hl_goal)

        bs, c_len, c_dim = corrections.shape

        #corrections_vec = self.language_network(corrections.view((-1, c_dim))).view((bs, c_len, -1)).permute(1, 0, 2)
        corrections_vec = corrections.permute(1, 0, 2)

        self.context_encoder.init_hidden(bs)
        context_vec = self.context_encoder(corrections_vec, correction_lens.data)

        prob = self.prob_network(torch.cat([x1, x2, x3, context_vec], -1))
        dist = Categorical(prob)
        return dist

class CategoricalTrajCorrectionNetwork(Policy):
    def __init__(self, obs_shape, extra_dim, obs_network, extra_obs_network, prev_traj_encoder,
                 prob_network, correction_encoder, hl_goal_encoder, traj_correction_encoder,
                 output_dim, obs_filter=None):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)
        self.extra_dim = extra_dim
        self.obs_network = obs_network
        self.extra_obs_network = extra_obs_network
        self.correction_encoder = correction_encoder
        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        self.prev_traj_encoder = prev_traj_encoder
        self.hl_goal_encoder = hl_goal_encoder
        self.traj_correction_encoder = traj_correction_encoder
        self.obs_filter = obs_filter

    def reset(self, shape):
        self.cache_inputs = []

    def forward(self, x, corrections, correction_lens, prev_trajs_lst, prev_traj_lens, hl_goal,
                prev_traj_enc=None, hl_goal_enc=None, correction_enc=None):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        # X will be flat and contain extra_dim (if obj picked up)
        # prev_traj is just last state for now
        bs = x.shape[0]

        obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))
        extra_obs = x[:, self.obs_len:]




        if obs.dtype == np.float32:
            obs = np_to_var(obs)
        if extra_obs.dtype == np.float32:
            extra_obs = np_to_var(extra_obs)

        x1 = self.obs_network(obs)
        x2 = self.extra_obs_network(extra_obs)
        x3 = self.hl_goal_encoder(hl_goal) if len(self.cache_inputs) == 0 else self.cache_inputs[0]
        x4 = self.prev_traj_encoder(prev_trajs_lst) if len(self.cache_inputs) == 0 else self.cache_inputs[1]

        # x4 = [self.prev_traj_encoder(prev_trajs) for prev_trajs in prev_trajs_lst]
        # for i in range(1, correction_lens.max()+1):
        #     x4[correction_lens>i, :] = 0
        # x4 = sum(x4) / correction_lens

        #x5 = self.correction_encoder(corrections) if correction_enc is None else correction_enc

        #x5 = [self.correction_encoder(prev_trajs) for prev_trajs in prev_trajs_lst]
        # c_shape = corrections.shape
        # x5 = self.correction_encoder(corrections.view(-1, c_shape[-1])).view((bs, c_shape[1], -1))
        # for i in range(c_shape[1]):
        #     x5[correction_lens<i+1, i, :] = 0
        # x5 = x5.sum(1) / correction_lens.view(bs, 1).float()
        x5 = self.correction_encoder(corrections, correction_lens) if len(self.cache_inputs) == 0 else self.cache_inputs[2]
        self.cache_inputs = [x3, x4, x5]

        x6 = self.traj_correction_encoder(torch.cat([x4, x5], -1))

        prob = self.prob_network(torch.cat([x1, x2, x3, x6], -1))
        dist = Categorical(prob)
        return dist

class CategoricalTrajCorrectionAblationNetwork(Policy):
    def __init__(self, obs_shape, extra_dim, obs_network, extra_obs_network, prev_traj_encoder,
                 prob_network, correction_encoder, hl_goal_encoder, traj_correction_encoder,
                 output_dim, obs_filter=None, ablation=None):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)
        self.extra_dim = extra_dim
        self.obs_network = obs_network
        self.extra_obs_network = extra_obs_network
        self.correction_encoder = correction_encoder
        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        self.prev_traj_encoder = prev_traj_encoder
        self.hl_goal_encoder = hl_goal_encoder
        self.traj_correction_encoder = traj_correction_encoder
        self.obs_filter = obs_filter
        self.ablation = ablation

    def reset(self, shape):
        self.cache_inputs = []

    def forward(self, x, corrections, correction_lens, prev_trajs_lst, prev_traj_lens, hl_goal,
                prev_traj_enc=None, hl_goal_enc=None, correction_enc=None):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        # X will be flat and contain extra_dim (if obj picked up)
        # prev_traj is just last state for now
        bs = x.shape[0]

        obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))
        extra_obs = x[:, self.obs_len:]




        if obs.dtype == np.float32:
            obs = np_to_var(obs)
        if extra_obs.dtype == np.float32:
            extra_obs = np_to_var(extra_obs)

        x1 = self.obs_network(obs)
        x2 = self.extra_obs_network(extra_obs)


        if self.ablation == 'ablation1_noprev_traj':
            x3 = self.hl_goal_encoder(hl_goal) if len(self.cache_inputs) == 0 else self.cache_inputs[0]
            x6 = self.correction_encoder(corrections, correction_lens) if len(self.cache_inputs) == 0 else self.cache_inputs[1]
            self.cache_inputs = [x3, x6]
            prob = self.prob_network(torch.cat([x1, x2, x3, x6], -1))
        elif self.ablation == 'ablation2_no_hlg':
            x4 = self.prev_traj_encoder(prev_trajs_lst) if len(self.cache_inputs) == 0 else self.cache_inputs[0]
            x5 = self.correction_encoder(corrections, correction_lens) if len(self.cache_inputs) == 0 else self.cache_inputs[1]
            self.cache_inputs = [x4, x5]
            x6 = self.traj_correction_encoder(torch.cat([x4, x5], -1))
            prob = self.prob_network(torch.cat([x1, x2, x6], -1))
        elif self.ablation == 'ablation3_only_prevc':
            x3 = self.hl_goal_encoder(hl_goal) if len(self.cache_inputs) == 0 else self.cache_inputs[0]
            x4 = self.prev_traj_encoder(prev_trajs_lst) if len(self.cache_inputs) == 0 else self.cache_inputs[1]
            x5 = self.correction_encoder(corrections, correction_lens) if len(self.cache_inputs) == 0 else self.cache_inputs[2]
            self.cache_inputs = [x3, x4, x5]

            x6 = self.traj_correction_encoder(torch.cat([x4, x5], -1))
            prob = self.prob_network(torch.cat([x1, x2, x3, x6], -1))
        else:
            x3 = self.hl_goal_encoder(hl_goal) if len(self.cache_inputs) == 0 else self.cache_inputs[0]
            x4 = self.prev_traj_encoder(prev_trajs_lst) if len(self.cache_inputs) == 0 else self.cache_inputs[1]
            x5 = self.correction_encoder(corrections, correction_lens) if len(self.cache_inputs) == 0 else self.cache_inputs[2]
            self.cache_inputs = [x3, x4, x5]

            x6 = self.traj_correction_encoder(torch.cat([x4, x5], -1))
            prob = self.prob_network(torch.cat([x1, x2, x3, x6], -1))

        dist = Categorical(prob)
        return dist

class CategoricalCorrectionBaselineNetwork(Policy):
    def __init__(self, obs_shape, extra_dim, obs_network, extra_obs_network, language_network,
                 prob_network, output_dim, obs_filter=None):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)
        self.extra_dim = extra_dim
        self.obs_network = obs_network
        self.extra_obs_network = extra_obs_network
        self.language_network = language_network
        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        #self.modules = [self.prob_network]
        self.obs_filter = obs_filter

    def reset(self, shape):
        pass

    def forward(self, x, language):
        if self.obs_filter is not None:
            x.data = self.obs_filter(x.data)
        # X will be flat and contain extra_dim (if obj picked up)
        bs = x.shape[0]

        obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))
        extra_obs = x[:, self.obs_len:]

        if obs.dtype == np.float32:
            obs = np_to_var(obs)
        if extra_obs.dtype == np.float32:
            extra_obs = np_to_var(extra_obs)


        #combined_obs = np_to_var(np.zeros((1, 10, 13, 13), dtype=np.float32))
        #extra_obs = np_to_var(np.zeros((1, 10), dtype=np.float32))
        #import pdb; pdb.set_trace()
        x1 = self.obs_network(obs)
        x2 = self.extra_obs_network(extra_obs)

        language_vec = self.language_network(language)


        prob = self.prob_network(torch.cat([x1, x2, language_vec], -1))
        dist = Categorical(prob)
        return dist

class CategorgicalPusherBaselinePolicy(Policy):
    def __init__(self, obs_shape, prob_network, output_dim, hlg_encoder):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)
        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        self.hlg_encoder = hlg_encoder

    def reset(self, shape):
        self.cache_inputs = []

    def forward(self, x, hl_goal, corrections, prev_traj, lens):
        x1 = self.hlg_encoder(hl_goal)
        out = torch.cat([x, x1], -1)
        prob = self.prob_network(out)
        #dist = Categorical(prob)
        dist = irlCategorical(prob)
        return dist

class CategorgicalPusherBaselineMisraPolicy(Policy):
    def __init__(self, obs_shape, prob_network, output_dim, hlg_encoder):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)
        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        self.hlg_encoder = hlg_encoder

    def reset(self, shape):
        self.cache_inputs = []

    def forward(self, x, hl_goal, corrections, prev_traj, lens):
        x1 = self.hlg_encoder(hl_goal)
        out = torch.cat([x, x1], -1)
        prob = self.prob_network(out)
        #dist = Categorical(prob)
        dist = Categorical(prob)
        return dist

class CategorgicalCorrectionPusherPolicy(Policy):
    def __init__(self, obs_shape, prob_network, output_dim,
                 correction_traj_encoder, hlg_encoder):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)
        self.prob_network = prob_network
        self.output_dim = output_dim # Used for sampler
        self.correction_traj_encoder = correction_traj_encoder
        self.hlg_encoder = hlg_encoder


    def reset(self, shape):
        self.cache_inputs = []

    def forward(self, x, hl_goal, corrections, prev_traj, lens):
        x1 = self.hlg_encoder(hl_goal)
        x2 = self.correction_traj_encoder(prev_traj, corrections, lens)
        out = torch.cat([x, x1, x2], -1)
        prob = self.prob_network(out)
        dist = Categorical(prob)
        return dist

class CategorgicalMinigridBaselinePolicy(Policy):
    def __init__(self, obs_shape, prob_network, output_dim, hlg_encoder, obs_net, extra_obs_net):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)
        self.prob_network = prob_network
        self.obs_net = obs_net
        self.extra_obs_net = extra_obs_net
        self.output_dim = output_dim # Used for sampler
        self.hlg_encoder = hlg_encoder

    def reset(self, shape):
        self.cache_inputs = []

    def forward(self, x, hl_goal, corrections, prev_traj, lens):
        x1 = self.hlg_encoder(hl_goal)

        bs = x.shape[0]

        obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))
        extra_obs = x[:, self.obs_len:]
        x = torch.cat([self.obs_net(obs), self.extra_obs_net(extra_obs)], -1)

        out = torch.cat([x, x1], -1)
        prob = self.prob_network(out)
        dist = irlCategorical(prob)
        return dist

class CategorgicalCorrectionMinigridPolicy(Policy):
    def __init__(self, obs_shape, prob_network, output_dim,
                 correction_traj_encoder, hlg_encoder, obs_net, extra_obs_net):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)
        self.prob_network = prob_network
        self.obs_net = obs_net
        self.extra_obs_net = extra_obs_net
        self.output_dim = output_dim # Used for sampler
        self.correction_traj_encoder = correction_traj_encoder
        self.hlg_encoder = hlg_encoder

    def reset(self, shape):
        self.cache_inputs = []

    def forward(self, x, hl_goal, corrections, prev_traj, lens):
        x1 = self.hlg_encoder(hl_goal)
        x2 = self.correction_traj_encoder(prev_traj, corrections, lens)

        bs = x.shape[0]

        obs = x[:, :self.obs_len].reshape((bs, *self.obs_shape))
        extra_obs = x[:, self.obs_len:]
        x = torch.cat([self.obs_net(obs), self.extra_obs_net(extra_obs)], -1)

        out = torch.cat([x, x1, x2], -1)
        prob = self.prob_network(out)
        dist = irlCategorical(prob)
        return dist


class CategoricalPusherTrajCorrectionPolicy(Policy):
    def __init__(self, obs_shape, obs_network, prev_traj_encoder,
                 prob_network, correction_encoder, hl_goal_encoder, traj_correction_encoder,
                 output_dim, obs_filter=None, min_var=1e-6):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_len = np.prod(obs_shape)

        self.obs_network = obs_network
        self.correction_encoder = correction_encoder
        self.prob_network = prob_network
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
        #x2 = self.hl_goal_encoder(hl_goal) if len(self.cache_inputs) == 0 else self.cache_inputs[0]

        x3 = self.traj_correction_encoder(prev_trajs_lst, corrections, correction_lens) if len(self.cache_inputs) == 0 else self.cache_inputs[0]

        self.cache_inputs = [x3]

        out = torch.cat([x1, x3], -1)

        prob = self.prob_network(out)
        dist = Categorical(prob)
        return dist