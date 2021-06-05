import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from lgpl.models.weight_init import xavier_init
from lgpl.utils.torch_utils import Variable, from_numpy, gpu_enabled, get_numpy, np_to_var

class Baseline:
    """
    Value function
    """

    def fit(self, obs, returns):
        pass

    def forward(self, obs):
        pass

class ZeroBaseline(Baseline):
    def forward(self, obs_np):
        #import pdb; pdb.set_trace()
        bs, obs_dim = obs_np.size()
        return Variable(torch.zeros(bs))


    def predict(self, obs_np):
        bs, path_len, obs_dim = obs_np.shape
        return Variable(torch.zeros(bs, path_len))

class LinearFeatureBaseline(Baseline):
    def __init__(self, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff


    def _features(self, obs):
        o = np.clip(obs, -10, 10)
        l = o.shape[0]
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)


    def fit(self, obs_np, returns_np):
        bs, path_len, obs_dim = obs_np.shape
        obs = obs_np.reshape(-1, obs_dim)
        returns = returns_np.reshape(-1)

        featmat = self._features(obs)
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def predict(self, obs_np):
        bs, path_len, obs_dim = obs_np.shape
        obs = obs_np.reshape(-1, obs_dim)
        if self._coeffs is None:
            return Variable(torch.zeros((bs, path_len)))
        returns = self._features(obs).dot(self._coeffs).reshape((-1, path_len))
        return np_to_var(returns)

    def forward(self, obs):
        return self.predict(get_numpy(obs))

class NNBaseline(Baseline, torch.nn.Module):
    def __init__(self, network, batch_size=128, max_epochs=20, optimizer=optim.Adam):
        super().__init__()
        self.network = network
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.optimizer = optimizer(self.network.parameters())
        # if gpu_enabled():
        #     self.network.cuda()

    def fit(self, obs_np, returns_np):
        self.network.apply(xavier_init)
        bs, path_len, obs_dim = obs_np.shape

        obs = from_numpy(obs_np.reshape(-1, obs_dim).astype(np.float32))
        returns = from_numpy(returns_np.reshape(-1).astype(np.float32))

        dataloader = DataLoader(TensorDataset(obs, returns), batch_size=self.batch_size,
                                 shuffle=True)
        for epoch in range(self.max_epochs):
            for x, y in dataloader:
                self.optimizer.zero_grad()
                x = Variable(x)
                y = Variable(y).float().view(-1, 1)
                loss = (self.network(x) - y).pow(2).mean()
                loss.backward()
                self.optimizer.step()
        print('loss %f' % get_numpy(loss)[0])

    def forward(self, obs):
        return self.network(obs)

    def predict(self, obs_np):
        bs, path_len, obs_dim = obs_np.shape

        obs = np_to_var(obs_np.reshape(-1, obs_dim).astype(np.float32))

        return self.network(obs).view(-1, path_len)