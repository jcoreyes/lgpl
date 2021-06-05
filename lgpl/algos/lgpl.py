"""
Supervised learning
"""
from lgpl.utils.torch_utils import Variable, get_numpy, from_numpy, np_to_var, gpu_enabled
import torch
import lgpl.utils.logger as logger
from collections import OrderedDict
from lgpl.samplers.rollout import rollout
import numpy as np
EPS = 1e-8

class LGPL:

    def __init__(self, policy, optimizer, correction_horizon=3, loss_type='cross_entropy',
                 custom_loss_fn=None, logger_output='sl.csv', dagger_interval=5, use_fullinfo=False, entropy_bonus=0):
        self.policy = policy
        self.optimizer = optimizer
        self.correction_horizon = correction_horizon

        self.loss_type = loss_type
        self.custom_loss_fn = custom_loss_fn
        self.logger_output = logger_output
        self.dagger_interval = dagger_interval
        self.use_fullinfo = use_fullinfo
        self.entropy_bonus = entropy_bonus
        if gpu_enabled():
            self.policy.cuda()

    def train_epoch(self, dataset, test=False):
        self.policy.eval() if test else self.policy.train()
        stats = OrderedDict([('loss', 0),
                             ('accuracy', 0),
                             ('entropy', 0),
                             ])
        for tensors in dataset.dataloader:
            self.optimizer.zero_grad()
            states, actions, corrections, correction_lens, prev_traj, hl_goal = dataset.get_policy_input(tensors)
            self.policy.reset(states.shape[0])
            y_hat = self.policy.forward(states.float(), hl_goal, corrections, prev_traj.float(), correction_lens)
            loss = self.loss(Variable(actions), y_hat).mean(0)
            if self.entropy_bonus > 0:
                loss -= self.entropy_bonus * y_hat.entropy().mean()

            if not test:
                loss.backward()
                self.optimizer.step()

            stats['loss'] += float(get_numpy(loss))
            stats['entropy'] += get_numpy(y_hat.entropy().mean())
            stats['accuracy'] += get_numpy((actions.argmax(1) == y_hat.probs.argmax(1)).float().mean())

        for k, v in stats.items():
            stats[k] = v / max(len(dataset.dataloader), 1)

        stats['size'] = dataset.size
        stats['n_envs'] = len(dataset.envs)
        prefix = 'test' if test else 'train'
        stats = OrderedDict({'%s %s' % (prefix, k): v for k,v in stats.items()})
        return stats

    def run_corrections(self, dataset, dataset_name, num_corrections=1,
                         add_to_data=False, render=False, env_idx=0):
        self.policy.eval()
        n_envs = len(dataset.envs)
        corrections = np.zeros((n_envs, dataset.max_corrections, dataset.correction_dim))
        prev_trajs = np.zeros((n_envs, dataset.max_corrections, dataset.trajs.shape[-1]))
        correction_lens = np.ones(n_envs, dtype=np.int64)
        if self.use_fullinfo:
            hl_goals = np.stack([env.full_info for env in dataset.envs], 0)
        else:
            hl_goals = np.stack([env.hl_goal for env in dataset.envs], 0)

        has_finished = np.zeros((n_envs, num_corrections))
        completion = np.zeros((n_envs, num_corrections))

        for c in range(num_corrections):
            trajs = dataset.sampler.rollout(self.policy, dataset.path_len,
                                            add_inputs=[hl_goals, corrections, prev_trajs, correction_lens],
                                            render=render, env_idx=env_idx)

            for i, traj_obs in enumerate(trajs['obs']):
                finished = trajs['infos'][-1][i]['has_finished']
                completion_frac = trajs['infos'][-1][i]['completion']
                # stop correcting and adding to data if finished
                if finished or has_finished[i].sum() > 0:
                    has_finished[i, c:] = 1
                    completion[i, c:] = 1
                    continue
                completion[i ,c] = completion_frac
                if add_to_data:
                    with torch.no_grad():
                        if dataset.path_len == 350:
                            dist = dataset.expert_pols[i].forward(np_to_var(traj_obs[:, :28]))
                        else:
                            dist = dataset.expert_pols[i].forward(np_to_var(traj_obs[:, :255]))
                    # filter out if obs or actions are too big
                    actions = get_numpy(dist.probs)
                    dataset.add(traj_obs, actions, corrections[i], correction_lens[i], prev_trajs[i], hl_goals[i])

                if c < num_corrections - 1:
                    correction = dataset.envs[i].corrections.gen_corr(traj_obs, trajs['infos'][-1][i])
                    corrections[i, correction_lens[i], :] = correction
                    prev_trajs[i, correction_lens[i], :] = traj_obs[::dataset.traj_subsample, :].flatten()
                    if render and env_idx == i:
                        print(dataset.envs[env_idx].corrections.idx_to_sent(correction))
                correction_lens[i] += 1

        has_finished = has_finished.mean(0)
        completion = completion.mean(0)
        stats = ([('Mean Finished CI %d' % i, has_finished[i]) for i in range(num_corrections)] +
                 [('Completion CI %d' % i, completion[i]) for i in range(num_corrections)])

        stats = OrderedDict([(dataset_name + ' ' + k, v) for k, v in stats])
        return stats

    def train(self, train_dataset, iters=100, test_dataset=None, render=False, add_to_data=True):
        for train_itr in range(iters):
            with logger.prefix("SL Train Iter %d | " % train_itr):

                if train_itr % self.dagger_interval == 0:
                    train_stats = self.run_corrections(train_dataset, 'train',
                                                        num_corrections=train_dataset.max_corrections,
                                                        add_to_data=add_to_data)
                    test_stats = self.run_corrections(test_dataset, 'test',
                                                       num_corrections=test_dataset.max_corrections,
                                                       add_to_data=add_to_data) if test_dataset is not None else {}
                    c_stats = OrderedDict(list(train_stats.items()) + list(test_stats.items()))

                print('Training epoch')
                train_loss = self.train_epoch(train_dataset)
                test_loss = self.train_epoch(test_dataset, test=True) if test_dataset is not None else {}

                stats = OrderedDict(list(train_loss.items()) + list(test_loss.items()))

                if render and train_itr % 10 == 0:
                   self.render(train_dataset, 0, 1)

                if train_itr % 10 == 0 and logger.get_snapshot_dir() is not None:
                    self.save(logger.get_snapshot_dir() + '/snapshots', train_itr)

                stats = OrderedDict(list(stats.items()) + list(c_stats.items()))
                logger.print_diagnostics(stats)
                logger.write_tabular(stats, self.logger_output)


    def render(self, dataset, env_idx, num_corrections):
        self.policy.eval()
        env = dataset.envs[env_idx]
        n_envs = 1
        corrections = np.zeros((n_envs, dataset.max_corrections, dataset.correction_dim))
        prev_trajs = np.zeros((n_envs, dataset.max_corrections, dataset.obs_dim * dataset.traj_sublen))
        correction_lens = np.ones(n_envs, dtype=np.int64)
        for c in range(num_corrections):
            traj = rollout(self.policy, env, dataset.path_len,
                           add_inputs=[env.hl_goal, corrections, prev_trajs, correction_lens],
                           deterministic=False, plot=True)
            traj_obs = np.stack(traj['obs'])
            corrections[0, correction_lens[0], :] = env.corrections.gen_corr2(traj_obs)
            prev_trajs[0, correction_lens[0], :] = traj_obs[::dataset.traj_subsample, :].flatten()

    def save(self, snapshot_dir, itr):
        import os
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        torch.save(self.policy.state_dict(), snapshot_dir + '/policy_itr%d.pkl' %itr)

    def load(self, snapshot_dir, itr):
        self.policy.load_state_dict(torch.load(snapshot_dir + '/policy.pkl'))


    def loss(self, y, y_hat):
        if self.custom_loss_fn is not None:
            return self.custom_loss_fn(y, y_hat)
        if self.loss_type == 'mse':
            return self.mse_loss(y, y_hat.mean)
        if self.loss_type == 'll':
            return -y_hat.log_likelihood(y)
        elif self.loss_type == 'kl':
            return self.kl_loss(y, y_hat)

    def mse_loss(self, y, y_hat):
        """
          :param y: (bs, output_dim)
          :param y_hat: (bs, output_dim)
          :return: (bs)
          """
        se = (y - y_hat).pow(2)
        for i in range(len(se.size()) - 1):
            se = se.mean(-1)
        return se

    def kl_loss(self, y, y_hat):
        kl = (y * (torch.log(y + 1e-8) - torch.log(y_hat.probs + 1e-8))).sum(1)
        return kl