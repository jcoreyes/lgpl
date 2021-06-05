import numpy as np
import torch

from exps.data_utils import gridp_to_corr, traj2d_to_gridp
from exps.data_utils import traj_to_gridp
from lgpl.envs.pusher.env_utils import gen_correction as gen_correction_pusher
from lgpl.envs.gym_maze.env_utils import gen_correction as gen_correction_maze
from lgpl.datasets.torch_dataset import MultiTensorDataset
from torch.utils.data.dataloader import DataLoader
from lgpl.utils.torch_utils import get_numpy, from_numpy, np_to_var, gpu_enabled, Variable

import gym

class CorrectionDataset:
    def __init__(self):
        pass

class PusherCorrectionDataset(CorrectionDataset):
    def __init__(self, env_configs, envs, optimal_trajs, random_trajs, expert_policies,
                 obs_dim, action_dim, path_len, correction_dim,
                 buffer_size=int(5e6), batch_size=128):
        """
        :param env_configs: list where each elem contains block and goal id
        :param optimal_trajs: list where each element is [(s0, a0), (s1, a1), ...]
        :param obs_dim: tuple
        :param action_dim: tuple
        :param path_len:
        :param correction_dim:
        :param buffer_size:
        """
        super().__init__()
        self.env_configs = env_configs
        self.optimal_trajs = optimal_trajs
        self.expert_policies = expert_policies
        if gpu_enabled():
            [x.cuda() for x in self.expert_policies]
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.path_len = path_len
        self.envs = envs
        self.batch_size = batch_size

        self.states = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.traj_idx = np.zeros(buffer_size, dtype=np.int64)
        self.trajs = np.zeros((buffer_size//path_len, path_len, obs_dim), dtype=np.float32)
        self.corrections = np.zeros((buffer_size, correction_dim), dtype=np.float32)
        self.random_trajs = [traj[:, :self.obs_dim] for traj in random_trajs]

        self.size = 0
        self.n_trajs = 0
        self._dataloader = None

        self.random_trajs_corrections = []
        for i, traj in enumerate(random_trajs):
            self.random_trajs_corrections.append(self.add(traj, i))

    @staticmethod
    def get_policy_input(tensors, **kwargs):
        traj_encoder = kwargs.get('traj_encoder', None)
        if traj_encoder is None:
            raise ValueError('Must pass in keyword argument \'traj_encoder\'.')
        traj, obs, correction, action = tensors
        traj_encoder.reset(traj.shape[0])
        traj_encoding = traj_encoder.forward(Variable(traj))
        policy_input = torch.cat([Variable(obs), traj_encoding, Variable(correction)], -1)
        return policy_input, action

    def generate_correction(self, traj, env_config_idx):
        goal = self.env_configs[env_config_idx]
        optimal_traj = self.optimal_trajs[env_config_idx]
        return gen_correction_pusher(traj, optimal_traj, goal)

    def add(self, traj, env_config_idx, use_expert_pol=False):

        correction = self.generate_correction(traj, env_config_idx)

        traj_obs = traj[:, :self.obs_dim]

        if use_expert_pol:
            traj_actions = get_numpy(self.expert_policies[env_config_idx].forward(np_to_var(traj_obs)).mle)
        else:
            traj_actions = self.optimal_trajs[env_config_idx][:, self.obs_dim:]
        self.trajs[self.n_trajs] = traj_obs
        self.corrections[self.n_trajs] = correction.flatten()

        for i in range(traj.shape[0]):
            #self.states[self.size] = traj_obs[i]
            self.states[self.size] = self.optimal_trajs[env_config_idx][i, :self.obs_dim]
            self.actions[self.size] = traj_actions[i]
            self.traj_idx[self.size] = self.n_trajs
            self.size += 1

        self.n_trajs += 1

        self._dataloader = None

        if self.size >= self.buffer_size:
            raise OverflowError

        return correction


    def get_tensors(self):
        tensors = [self.trajs[self.traj_idx[:self.size]], self.states[:self.size],
                   self.corrections[self.traj_idx[:self.size]], self.actions[:self.size]]
        return [from_numpy(tensor) for tensor in tensors]

    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = DataLoader(MultiTensorDataset(self.get_tensors()),
                                          batch_size=self.batch_size, shuffle=True)

        return self._dataloader


class MazeCorrectionDataset(CorrectionDataset):
    def __init__(self, maze_ids, mazes, optimal_trajs, random_trajs,
                 obs_dim, action_dim, path_len, correction_dim, map_dim,
                 buffer_size=int(5e6), batch_size=128):
        """
        :param maze_ids: list of maze ids in same order as trajs
        :param mazes: list of mazes (numpy arrays)
        :param optimal_trajs: list where each element is [(s0, a0), (s1, a1), ...]
        :param random_trajs: list where each element is [(s0, a0), (s1, a1), ...]
        :param obs_dim: tuple
        :param action_dim: tuple
        :param path_len:
        :param correction_dim:
        :param map_dim:
        :param buffer_size:
        """
        super().__init__()
        self.maze_ids = maze_ids
        self.optimal_trajs = optimal_trajs
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.map_dim = (map_dim,) if type(map_dim) is int else tuple(map_dim)
        self.path_len = path_len

        self.batch_size = batch_size

        self.states = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.traj_idx = np.zeros(buffer_size, dtype=np.int64)
        self.trajs = np.zeros((buffer_size//path_len, path_len, obs_dim), dtype=np.float32)
        self.corrections = np.zeros((buffer_size, correction_dim), dtype=np.float32)
        self.maze_idx = np.zeros(buffer_size, dtype=np.int64)
        self.mazes = np.stack(mazes).astype(np.float32)
        self.random_trajs = [traj[:, :self.obs_dim] for traj in random_trajs]

        self.size = 0
        self.n_trajs = 0
        self._dataloader = None

        self.random_trajs_corrections = []
        for traj, maze_id in zip(random_trajs, maze_ids):
            self.random_trajs_corrections.append(self.add(traj, maze_id))

    @staticmethod
    def get_policy_input(tensors, **kwargs):
        traj_encoder = kwargs.get('traj_encoder', None)
        maze_encoder = kwargs.get('maze_encoder', None)
        if traj_encoder is None or maze_encoder is None:
            raise ValueError('Must pass in both keyword arguments \'traj_encoder\' and \'maze_encoder\'.')
        traj, obs, correction, action, maze = tensors
        traj_encoder.reset(traj.shape[0])
        traj_encoding = traj_encoder.forward(Variable(traj))
        maze_encoding = maze_encoder.forward(Variable(maze))
        policy_input = torch.cat([Variable(obs), traj_encoding, Variable(correction), maze_encoding], -1)
        return policy_input, action

    def maze_id_to_idx(self, maze_id):
        try:
            return self.maze_ids.index(maze_id)
        except ValueError:
            raise ValueError('Maze %d not in dataset initialization' % maze_id)

    def generate_correction(self, traj, maze_id):
        maze_idx = self.maze_id_to_idx(maze_id)
        optimal_traj = self.optimal_trajs[maze_idx]
        return gridp_to_corr(traj_to_gridp(optimal_traj), traj_to_gridp(traj), self.map_dim)[0]

    def add(self, traj, maze_id):
        maze_idx = self.maze_id_to_idx(maze_id)
        correction = self.generate_correction(traj, maze_id)

        traj_obs = traj[:, :self.obs_dim]
        traj_actions = self.optimal_trajs[maze_idx][:, self.obs_dim:].astype(np.uint8)
        self.trajs[self.n_trajs] = traj_obs
        self.corrections[self.n_trajs] = correction.flatten()

        for i in range(self.optimal_trajs[maze_idx].shape[0]):
            #self.states[self.size] = traj_obs[i]
            self.states[self.size] = self.optimal_trajs[maze_idx][i, :self.obs_dim]
            self.actions[self.size][traj_actions[i]] = 1
            self.traj_idx[self.size] = self.n_trajs
            self.maze_idx[self.size] = maze_idx
            self.size += 1

        self.n_trajs += 1

        self._dataloader = None

        if self.size >= self.buffer_size:
            raise OverflowError

        return correction


    def get_tensors(self):
        tensors = [self.trajs[self.traj_idx[:self.size]], self.states[:self.size],
                   self.corrections[self.traj_idx[:self.size]], self.actions[:self.size],
                   self.mazes[self.maze_idx[:self.size]]]
        return [from_numpy(tensor) for tensor in tensors]

    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = DataLoader(MultiTensorDataset(self.get_tensors()), batch_size=self.batch_size, shuffle=True)
        return self._dataloader

class MazeCorrection2DDataset(CorrectionDataset):
    def __init__(self, maze_ids, mazes, optimal_trajs, random_trajs,
                 obs_dim, action_dim, path_len, map_dim,
                 buffer_size=int(5e6), batch_size=128):
        """
        :param maze_ids: list of maze ids in same order as trajs
        :param mazes: list of mazes (numpy arrays)
        :param optimal_trajs: list where each element is [(s0, a0), (s1, a1), ...]
        :param random_trajs: list where each element is [(s0, a0), (s1, a1), ...]
        :param obs_dim: tuple
        :param action_dim: tuple
        :param path_len:
        :param correction_dim:
        :param map_dim:
        :param buffer_size:
        """
        super().__init__()
        self.maze_ids = maze_ids
        self.optimal_trajs = optimal_trajs
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.map_dim = (map_dim,) if type(map_dim) is int else tuple(map_dim)
        self.path_len = path_len

        self.batch_size = batch_size

        self.states = np.zeros((buffer_size, *map_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.traj_idx = np.zeros(buffer_size, dtype=np.int64)
        self.trajs = np.zeros((buffer_size//path_len, path_len, *map_dim), dtype=np.float32)
        self.corrections = np.zeros((buffer_size, *map_dim), dtype=np.float32)
        self.maze_idx = np.zeros(buffer_size, dtype=np.int64)
        self.mazes = np.stack(mazes).astype(np.float32)
        self.random_trajs = random_trajs

        self.size = 0
        self.n_trajs = 0
        self._dataloader = None

        self.random_trajs_corrections = []
        for traj, maze_id in zip(random_trajs, maze_ids):
            self.random_trajs_corrections.append(self.add(traj, maze_id))

    @staticmethod
    def get_policy_input(tensors, **kwargs):
        traj, obs, correction, action, maze = tensors
        # import pdb; pdb.set_trace()
        policy_input = torch.cat([Variable(obs).unsqueeze(1),
                                  Variable(correction).unsqueeze(1),
                                  Variable(maze)], 1)
        return policy_input, action

    def traj_to_action(self, traj):
        def obs_to_action(obs, next_obs):
            x = obs.argmax() // self.map_dim[0]
            y = obs.argmax() % self.map_dim[0]
            x_next = next_obs.argmax() // self.map_dim[0]
            y_next = next_obs.argmax() % self.map_dim[0]
            action = np.zeros(4)
            if y_next - y == -1:
                action[0] = 1
            elif y_next - y == 1:
                action[1] = 1
            elif x_next - x == 1:
                action[2] = 1
            elif x_next - x == -1:
                action[3] = 1
            else:
                raise ValueError("illegal action between observations")
            return action
        actions = np.zeros((traj.shape[0], 4))
        for i in range(traj.shape[0]-1):
            actions[i] = obs_to_action(traj[i], traj[i+1])
        # set fixed arbitrary last action
        actions[-1][0] = 1
        return actions

    def maze_id_to_idx(self, maze_id):
        try:
            return self.maze_ids.index(maze_id)
        except ValueError:
            raise ValueError('Maze %d not in dataset initialization' % maze_id)

    def generate_correction(self, traj, maze_id):
        maze_idx = self.maze_id_to_idx(maze_id)
        optimal_traj = self.optimal_trajs[maze_idx]
        # change traj to gridps
        return gridp_to_corr(traj2d_to_gridp(optimal_traj), traj2d_to_gridp(traj), self.map_dim)[0]

    def add(self, traj, maze_id):
        maze_idx = self.maze_id_to_idx(maze_id)
        correction = self.generate_correction(traj, maze_id)

        traj_actions = self.traj_to_action(self.optimal_trajs[maze_idx])
        self.trajs[self.n_trajs] = traj
        self.corrections[self.n_trajs] = correction

        for i in range(self.optimal_trajs[maze_idx].shape[0]):
            #self.states[self.size] = traj_obs[i]
            self.states[self.size] = self.optimal_trajs[maze_idx][i, :self.obs_dim]
            self.actions[self.size] = traj_actions[i]
            self.traj_idx[self.size] = self.n_trajs
            self.maze_idx[self.size] = maze_idx
            self.size += 1

        self.n_trajs += 1

        self._dataloader = None

        if self.size >= self.buffer_size:
            raise OverflowError

        return correction


    def get_tensors(self):
        tensors = [self.trajs[self.traj_idx[:self.size]], self.states[:self.size],
                   self.corrections[self.traj_idx[:self.size]], self.actions[:self.size],
                   self.mazes[self.maze_idx[:self.size]]]
        return [from_numpy(tensor) for tensor in tensors]

    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = DataLoader(MultiTensorDataset(self.get_tensors()), batch_size=self.batch_size, shuffle=True)
        return self._dataloader