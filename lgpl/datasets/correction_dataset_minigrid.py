
import numpy as np
from lgpl.datasets.torch_dataset import MultiTensorDataset
from torch.utils.data.dataloader import DataLoader
from lgpl.utils.torch_utils import get_numpy, from_numpy, np_to_var
from torch.autograd import Variable
import torch
from lgpl.pytorch_rl.utils import collect_traj
import gym


class CorrectionDatasetMiniGrid:
    def __init__(self, env_configs, envs, expert_pols,
                 obs_dim, action_dim, extra_dim, path_len, correction_dim, hl_goal_dim,
                 buffer_size=int(5e6), batch_size=128, max_corrections=5):
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

        self.states = np.zeros((buffer_size, obs_dim+extra_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.traj_idx = np.zeros(buffer_size, dtype=np.int64)
        self.trajs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.corrections = np.zeros((buffer_size, max_corrections, correction_dim), dtype=np.float32)
        self.correction_len = np.zeros((buffer_size), dtype=np.long)
        self.hl_goals = np.zeros((buffer_size, hl_goal_dim), dtype=np.float32)

        self.size = 0
        self.idx = 0
        self.n_trajs = 0
        self._dataloader = None

        self.expert_pols = expert_pols
        self.expert_trajs = []
        finished = []
        for i, (env, expert_pol) in enumerate(zip(envs, expert_pols)):
            expert_traj = collect_traj(expert_pol, env, [env.observation_space.shape[0]],
                                      path_len=path_len, deterministic=False, render=False,
                                       finish_early=True)
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
            for i, (obs, action, info) in enumerate(zip(expert_traj['obs'], expert_traj['action_probs'], expert_traj['infos'])):
                subgoal_len = len(expert_traj['infos'][i]['curr_subgoals'])
                # if on first subgoal then add all goal sequences
                if subgoal_len == len(subgoals):
                    for s_i in range(1, subgoal_len+1):
                        self.add([obs], [action], [subgoals_en[:s_i]], prev_state, env.hl_goal)
                else:
                    goal_idx = len(subgoals) - subgoal_len
                    for s_i in range(goal_idx, len(subgoals)):
                        self.add([obs], [action], [subgoals_en[:s_i+1]], prev_state, env.hl_goal)

                # subgoal_idx = len(subgoals) - subgoal_len
                # # TODO Fix
                # for s_i in range(0, subgoal_len):
                #     subgoal_idx = len(subgoals) - s_i
                #     self.add([obs], [action], subgoals_en[:subgoal_idx+1], prev_state)


    def generate_correction(self, traj, env_idx, cuda=False, add_to_data=True,
                            deterministic=True):
        expert_traj = self.expert_trajs[env_idx]
        expert_pol = self.expert_pols[env_idx]

        bad_traj_last_state = traj['obs'][0].astype(np.float32)[:self.obs_dim]

        bad_obs = []
        subgoals_lst = []
        env = self.envs[env_idx]
        for i, (info1, info2) in enumerate(zip(expert_traj['infos'], traj['infos'])):
            # TODO for now break if subgoals change
            if len(info2['curr_subgoals']) > len(self.subgoal_encodings[env_idx]):
                break
            bad_obs.append(traj['obs'][i].astype(np.float32))
            goal_idx = len(self.subgoal_encodings[env_idx]) - len(info2['curr_subgoals'])
            subgoals_lst.append(self.subgoal_encodings[env_idx][:goal_idx+1])

        obs = torch.from_numpy(np.stack(bad_obs)).float()
        states = torch.zeros(len(bad_obs), expert_pol.state_size)
        masks = torch.ones(len(bad_obs), 1)
        with torch.no_grad():
            value, actions, action_log_prob, states, dist = expert_pol.act(
                Variable(obs.cuda() if cuda else obs),
                Variable(states.cuda() if cuda else states),
                Variable(masks.cuda() if cuda else masks),
                deterministic=deterministic
            )

        action_probs = get_numpy(dist.probs)

        if add_to_data:
            self.add(bad_obs, action_probs, subgoals_lst, bad_traj_last_state, env.hl_goal)


    def add(self, obs_lst, actions, corrections_lst, prev_traj, hl_goal):

        #import pdb; pdb.set_trace()
        #assert len(obs_lst) == len(actions)
        #assert len(actions) == len(correction)
        for i in range(len(obs_lst)):
            self.states[self.idx, :] = obs_lst[i]
            self.actions[self.idx, :] = actions[i]
            self.trajs[self.idx, :] = prev_traj
            self.hl_goals[self.idx, :] = hl_goal

            corrections = corrections_lst[i]
            for j, c in enumerate(corrections):
                self.corrections[self.idx, j, :] = c
            self.correction_len[self.idx] = len(corrections)
            if len(corrections) == 0:
                import pdb; pdb.set_trace()

            self.idx += 1
            self.idx = self.idx % self.buffer_size
            self.size += 1

        self.size = min(self.size, self.buffer_size)

        # # TODO Handle overflow
        # if self.size >= self.buffer_size:
        #     raise OverflowError

        self._dataloader = None



        #return correction


    def get_tensors(self):
        tensors = [self.states[:self.size], self.actions[:self.size], self.corrections[:self.size],
                   self.correction_len[:self.size], self.hl_goals[:self.size]]
        return [from_numpy(tensor) for tensor in tensors]

    def get_policy_input(self, tensors):
        # sort input by correction len to work with variable length rnn
        lens, perm_idx = tensors[3].sort(0, descending=True)
        return [tensor[perm_idx] for tensor in tensors]
        #return tensors


    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = DataLoader(MultiTensorDataset(self.get_tensors()),
                                          shuffle=True, batch_size=self.batch_size)
        return self._dataloader