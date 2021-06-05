
import numpy as np
from lgpl.datasets.torch_dataset import MultiTensorDataset
from lgpl.samplers.rollout import rollout
from lgpl.samplers.vectorized_sampler import VectorizedSampler, VectorizedSampler2
from torch.utils.data.dataloader import DataLoader
from lgpl.utils.torch_utils import get_numpy, from_numpy, np_to_var, gpu_enabled
import torch

class LGPLDatasetMiniGrid:
    def __init__(self, env_configs, envs, expert_pols,
                 obs_dim, action_dim, extra_dim, path_len, correction_dim, hl_goal_dim, max_corrections=5,
                 buffer_size=int(1e6), batch_size=128, traj_subsample=5, add_expert_data=False):
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
        self.expert_pols = expert_pols
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.extra_dim = extra_dim
        self.correction_dim = correction_dim
        self.hl_goal_dim = hl_goal_dim
        self.path_len = path_len
        self.envs = envs
        self.max_corrections = max_corrections

        self.states = np.zeros((buffer_size, obs_dim+extra_dim), dtype=np.uint8)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.traj_idx = np.zeros(buffer_size, dtype=np.int64)
        self.traj_sublen = path_len // traj_subsample
        self.trajs = np.zeros((buffer_size, max_corrections, (obs_dim + extra_dim) * self.traj_sublen), dtype=np.uint8)
        self.traj_subsample = traj_subsample
        self.traj_len = np.zeros((buffer_size))
        # self.corrections = np.zeros((buffer_size, max_corrections, correction_dim), dtype=np.int64)
        self.corrections = np.zeros((buffer_size, max_corrections, correction_dim), dtype=np.float32)
        self.correction_lens = np.zeros((buffer_size), dtype=np.long)
        self.hl_goals = np.zeros((buffer_size, hl_goal_dim), dtype=np.float32)

        self.size = 0
        self.idx = 0
        self.n_trajs = 0
        self._dataloader = None

        self.expert_pols = expert_pols
        self.expert_trajs = []
        finished = []

        print("Loading in expert policies and testing performance")
        for i, (env, expert_pol) in enumerate(zip(envs, expert_pols)):
            expert_traj = rollout(expert_pol, env, path_len, deterministic=False, finish_early=True)
            self.expert_trajs.append(expert_traj)
            finished_flag = sum([x['has_finished'] for x in expert_traj['infos']]) > 0 and len(expert_traj['infos'][0]['curr_subgoals']) == 5
            if finished_flag:
                finished.append(i)
            print(env.all_subgoals, 'Expert Finished: ', finished_flag)
        finished = list(range(len(envs))) # TODO for now keep all
        print(len(finished) / float(len(envs)))

        def slice(lst, idx):
            return [lst[i] for i in idx]
        self.envs = slice(self.envs, finished)
        self.expert_trajs = slice(self.expert_trajs, finished)
        self.expert_pols = slice(self.expert_pols, finished)

        self.sampler = VectorizedSampler2(env=self.envs[0], policy=None, env_name=None, n_envs=len(self.envs), envs=self.envs)

        if add_expert_data:
            prev_traj = np.zeros_like(self.trajs[0])
            correction = np.zeros_like(self.corrections[0])
            correction_lens = np.ones(1)

            for i, expert_traj in enumerate(self.expert_trajs):
                traj_obs = expert_traj['obs']
                self.add(traj_obs, expert_traj['actions'], correction, correction_lens, prev_traj, self.envs[i].full_info)


    def add(self, obs_lst, actions, corrections, correction_len, prev_traj, hl_goal):

        for i in range(len(obs_lst)):
            self.states[self.idx, :] = obs_lst[i]
            self.actions[self.idx, :] = actions[i]
            self.hl_goals[self.idx, :] = hl_goal
            self.traj_idx[self.idx] = self.n_trajs
            self.corrections[self.idx, ...] = corrections
            self.correction_lens[self.idx] = correction_len
            self.trajs[self.idx, :] = prev_traj
            self.idx += 1
            self.idx = self.idx % self.buffer_size
            self.size += 1

        self.size = min(self.size, self.buffer_size)
        self._dataloader = None


    def get_tensors(self):
        tensors = [self.states[:self.size], self.actions[:self.size], self.corrections[:self.size],
                   self.correction_lens[:self.size],
                   self.trajs[:self.size], self.hl_goals[:self.size]]
        return [torch.from_numpy(tensor) for tensor in tensors]

    def get_policy_input(self, tensors):
        if gpu_enabled():
            return [x.cuda() for x in tensors]
        return tensors


    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = DataLoader(MultiTensorDataset(self.get_tensors()),
                                          shuffle=True, batch_size=self.batch_size)
        return self._dataloader
