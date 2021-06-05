import numpy as np
import torch
from lgpl.math.distributions.distribution import Distribution
# from torch.autograd import Variable
from lgpl.utils.torch_utils import Variable, get_numpy, np_to_var, FloatTensor

from torch.distributions import Categorical as  pyCategorical

EPS = 1e-8

class Categorical(Distribution):
    def __init__(self, prob):
        self.probs = prob
        self.dim = self.probs.size()[-1]
        self.bs = prob.size()[0]
        self._mle = None

    def log_likelihood(self, x):
        if x.size()[-1] != self.probs.size()[-1]:
            prob = torch.gather(self.probs, -1, x).squeeze()
            return torch.log(prob + EPS)
        else:
            return torch.log(torch.sum(self.probs * x, -1) + EPS)

    def entropy(self):
        return -(self.probs * torch.log(self.probs + EPS)).sum(-1)

    def log_likelihood_ratio(self, x, new_dist):
        ll_new = new_dist.log_likelihood(x)
        ll_old = self.log_likelihood(x)
        return torch.exp(ll_new - ll_old)

    def kl(self, dist):
        # Compute D_KL(self || dist)
        return torch.sum(dist.probs * (torch.log(dist.probs + EPS) - torch.log(self.probs + EPS)), -1)

    def sample(self, deterministic=False):
        if deterministic:
            return self.probs.max(-1)[1].unsqueeze(-1)
        else:
            return pyCategorical(self.probs).sample().unsqueeze(-1)

    def mode(self):
        return self.probs.max(-1)[1].unsqueeze(-1)

    def compute_mle(self):
        onehot = np.zeros(self.probs.size())
        onehot[np.arange(0, self.bs), get_numpy(torch.max(self.probs, -1)[1]).astype(np.int32)] = 1
        return np_to_var(onehot)

    def combine(self, dist_lst, func=torch.stack, axis=0):
        self.probs = func([dist.probs for dist in dist_lst], axis)
        return self

    def detach(self):
        return Categorical(self.probs.detach())

    def reshape(self, new_shape):
        self.probs = self.probs.view(*new_shape)
        return self

    @property
    def mle(self):
        if self._mle is None:
            self._mle = self.compute_mle()
        return self._mle

class RecurrentCategorical(Categorical):
    def __init__(self, prob, path_len, mle):
        self.probs = prob
        self.path_len = path_len
        self.dim = self.probs.size()[-1]
        self.bs = prob.size()[0]
        self._mle = mle

        self.probs_3d = self.probs.view(self.bs, path_len, -1)

    def log_likelihood(self, x):
        #import pdb; pdb.set_trace()
        if x.size()[-1] != self.probs.size()[-1]:
            # prob = torch.gather(self.probs, -1, x).squeeze()
            # return torch.log(prob + EPS)

            return torch.log(torch.sum(self.probs * x.view(self.probs.size()), -1) + EPS).sum(-1)
        else:
            return torch.log(torch.sum(self.probs * x, -1) + EPS)

    def sample(self, deterministic=False):
        if deterministic:
            return self.prob_3d.max(-1)[1].unsqueeze(-1)
        else:
            cat_size = self.probs_3d.size()[-1]
            onehot = np.zeros((self.bs * self.path_len, cat_size))
            idx = torch.multinomial(self.probs.view(-1, cat_size), 1)
            onehot[np.arange(self.bs * self.path_len), get_numpy(idx.squeeze())] = 1
            return np_to_var(onehot.reshape(self.probs_3d.size()))

class RecurrentMultiCategorical(Categorical):
    def __init__(self, cat_sizes, path_len, mle=None, **kwargs):
        super().__init__(**kwargs)
        self.cat_sizes = cat_sizes
        self.path_len = path_len

        # Probs will be (bs, path_len * sum(cat_sizes))
        self.probs_3d = self.probs.view(self.bs, path_len, -1)
        self._mle = mle

    def log_likelihood(self, x):
        # x is (bs, path_len * sum(cat_sizes)) one hots
        bs = x.size()[0]
        x_3d = x.view(bs, self.path_len, sum(self.cat_sizes))
        count = 0
        total_ll = Variable(torch.zeros(1))
        for cat_size in self.cat_sizes:
            prob = self.probs_3d[:, :, count: count + cat_size]
            onehot = x_3d[:, :, count: count + cat_size]
            ll = torch.log(torch.sum(prob * onehot, -1) + EPS)
            total_ll += ll.sum()
            count += cat_size
        return total_ll / bs
# torch.log(sd_traj.probs_3d[0, 0, idx[0, 0, 0]] + 1e-8) + torch.log(sd_traj.probs_3d[0, 0, idx[0, 0, 1] + 30] + 1e-8)

    def log_likelihood_full(self, x):
        # x is (bs, path_len, sum(cat_sizes)) one hots
        # probs = self.probs_3d * x
        #
        # return torch.log(torch.sum(self.probs_3d * x, -1) + EPS)
        bs = x.size()[0]
        x_3d = x.view(bs, self.path_len, sum(self.cat_sizes))
        count = 0
        total_ll = Variable(torch.zeros(bs, self.path_len))
        for cat_size in self.cat_sizes:
            prob = self.probs_3d[:, :, count: count + cat_size]
            onehot = x_3d[:, :, count: count + cat_size]
            #ll = torch.log(torch.sum(prob * onehot, -1) + EPS)
            #total_ll += ll
            #ll = torch.sum(prob * onehot, -1) + EPS
            _, p_idx = prob.max(-1)
            _, x_idx = onehot.max(-1)
            #import pdb; pdb.set_trace()
            ll = - torch.pow((p_idx - x_idx).float(), 2)

            total_ll += ll
            count += cat_size
        return total_ll

    def compute_mle(self):
        # Return (bs, path_len,  len(cat_sizes)) of idxs for each cat
        count = 0

        arg_maxs = []
        for cat_size in self.cat_sizes:
            cat = self.probs_3d[:, :, count: count + cat_size]
            count += cat_size
            _, argmax = torch.max(cat, -1)
            arg_maxs.append(argmax)

        return torch.stack(arg_maxs, -1)
