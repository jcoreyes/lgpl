
import numpy as np
from lgpl.datasets.torch_dataset import MultiTensorDataset
from lgpl.samplers.rollout import rollout
from lgpl.samplers.vectorized_sampler import VectorizedSampler, VectorizedSampler2
from torch.utils.data.dataloader import DataLoader
from lgpl.utils.torch_utils import get_numpy, from_numpy, np_to_var
from torch.autograd import Variable
import torch
from lgpl.pytorch_rl.utils import collect_traj
import gym


class TrajCorrectionDatasetMiniGrid:
    def __init__(self, env_configs, envs, expert_pols,
                 obs_dim, action_dim, extra_dim, path_len, correction_dim, hl_goal_dim, max_corrections=5,
                 buffer_size=int(1e6), batch_size=128, traj_subsample=5):
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
        #self.optimal_trajs = optimal_trajs
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
        self.trajs = np.zeros((buffer_size//path_len, path_len//traj_subsample, obs_dim+extra_dim), dtype=np.uint8)
        self.traj_subsample = traj_subsample
        self.traj_len = np.zeros((buffer_size))
        self.corrections = np.zeros((buffer_size, max_corrections, correction_dim), dtype=np.int64)
        self.correction_lens = np.zeros((buffer_size), dtype=np.long)
        self.hl_goals = np.zeros((buffer_size, hl_goal_dim), dtype=np.int64)

        self.size = 0
        self.idx = 0
        self.n_trajs = 0
        self._dataloader = None

        self.expert_pols = expert_pols
        self.expert_trajs = []
        finished = []

        for i, (env, expert_pol) in enumerate(zip(envs, expert_pols)):
            # expert_traj = collect_traj(expert_pol, env, [env.observation_space.shape[0]],
            #                           path_len=path_len, deterministic=False, render=False,
            #                            finish_early=False)
            expert_traj = rollout(expert_pol, env, path_len, deterministic=False, finish_early=True)
            self.expert_trajs.append(expert_traj)
            if sum([x['has_finished'] for x in expert_traj['infos']]) > 0 and len(expert_traj['infos'][0]['curr_subgoals']) == 5:
                finished.append(i)
            print(env.all_subgoals)

        print(len(finished) / float(len(envs)))

        def slice(lst, idx):
            return [lst[i] for i in idx]
        self.envs = slice(self.envs, finished)
        self.expert_trajs = slice(self.expert_trajs, finished)
        self.expert_pols = slice(self.expert_pols, finished)
        self.subgoal_encodings = []
        self.first_subgoals = []

        for env, expert_traj in zip(self.envs, self.expert_trajs):
            prev_state = env.reset()[:self.obs_dim]
            subgoals = expert_traj['infos'][0]['curr_subgoals']
            subgoals_en = [env.subgoal_encode(s).flatten() for s in subgoals]
            self.subgoal_encodings.append(subgoals_en)
            self.first_subgoals.append(subgoals_en[0])
            # for i, (obs, action, info) in enumerate(zip(expert_traj['obs'], expert_traj['action_probs'], expert_traj['infos'])):
            #     subgoal_len = len(expert_traj['infos'][i]['curr_subgoals'])
            #     # if on first subgoal then add all goal sequences
            #     if subgoal_len == len(subgoals):
            #         for s_i in range(1, subgoal_len+1):
            #             self.add([obs], [action], [subgoals_en[:s_i]], prev_state, env.hl_goal)
            #     else:
            #         goal_idx = len(subgoals) - subgoal_len
            #         for s_i in range(goal_idx, len(subgoals)):
            #             self.add([obs], [action], [subgoals_en[:s_i+1]], prev_state, env.hl_goal)

        self.sampler = VectorizedSampler2(env=self.envs[0], policy=None, env_name=None, n_envs=len(self.envs), envs=self.envs)


    def generate_correction(self, traj, prev_traj, env_idx, prev_corrections, prev_correction_len,
                            cuda=False, add_to_data=True,
                            deterministic=True):
        expert_traj = self.expert_trajs[env_idx]
        expert_pol = self.expert_pols[env_idx]

        #prev_traj = np.stack(traj['obs'], 0)

        env = self.envs[env_idx]
        # for i, (expert_info, bad_info) in enumerate(zip(expert_traj['infos'], traj['infos'])):
        #     # TODO for now break if subgoals increase beyond initial
        #     # if len(bad_info['curr_subgoals']) > len(traj['infos'][0]['curr_subgoals']):
        #     #     break
        #     bad_obs.append(traj['obs'][i].astype(np.float32))
        bad_obs = traj['obs']

        #
        subgoal_lens = [len(x['curr_subgoals']) for x in traj['infos']]
        min_len_idx = np.argmin(subgoal_lens)
        correction = env.sent_to_idx(traj['infos'][min_len_idx]['curr_subgoals'][0])

        obs = np_to_var(np.stack(bad_obs).astype(np.float32))
        with torch.no_grad():
            dist = expert_pol.forward(obs)

        action_probs = get_numpy(dist.prob)

        if add_to_data:
            self.add(bad_obs, action_probs, prev_corrections, prev_correction_len, prev_traj, env.hl_goal)
        prev_corrections[prev_correction_len, :] = correction

        return prev_corrections

    def add(self, obs_lst, actions, corrections, correction_len, prev_traj, hl_goal):

        
        self.trajs[self.n_trajs, :, :] = 0
        prev_traj = prev_traj[::self.traj_subsample, :]
        self.trajs[self.n_trajs, :prev_traj.shape[0], :] = prev_traj
        self.traj_len[self.n_trajs] = prev_traj.shape[0]


        for i in range(len(obs_lst)):
            self.states[self.idx, :] = obs_lst[i]
            self.actions[self.idx, :] = actions[i]
            self.hl_goals[self.idx, :] = hl_goal
            self.traj_idx[self.idx] = self.n_trajs

            #for c in range(len(corrections)):
            self.corrections[self.idx, ...] = corrections
            self.correction_lens[self.idx] = correction_len
            self.idx += 1
            self.idx = self.idx % self.buffer_size
            self.size += 1

        self.size = min(self.size, self.buffer_size)

        self.n_trajs += 1
        self.n_trajs = self.n_trajs % (self.buffer_size // self.path_len)
        # # TODO Handle overflow
        # if self.size >= self.buffer_size:
        #     raise OverflowError

        self._dataloader = None


    def get_tensors(self):
        tensors = [self.states[:self.size], self.actions[:self.size], self.corrections[:self.size],
                   self.correction_lens[:self.size],
                   self.trajs[self.traj_idx[:self.size]], self.hl_goals[:self.size]]
        return [from_numpy(tensor) for tensor in tensors]

    def get_policy_input(self, tensors):
        # sort input by correction len to work with variable length rnn
        #lens, perm_idx = tensors[3].sort(0, descending=True)
        #return [tensor[perm_idx] for tensor in tensors]
        return tensors


    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = DataLoader(MultiTensorDataset(self.get_tensors()),
                                          shuffle=True, batch_size=self.batch_size)
        return self._dataloader
