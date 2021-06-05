"""
Supervised learning
"""
from lgpl.models.convnet import Convnet
from lgpl.models.mlp import MLP
from lgpl.policies.discrete import CategoricalMinigetNetwork
from lgpl.utils.torch_utils import Variable, get_numpy, from_numpy, np_to_var, gpu_enabled
import torch
import lgpl.utils.logger as logger
from collections import OrderedDict
from lgpl.samplers.rollout import rollout_correction
import numpy as np
EPS = 1e-8

class BaselineMinigrid:

    def __init__(self, policy, optimizer, correction_horizon=3, loss_type='cross_entropy',
                 custom_loss_fn=None, logger_output='sl.csv', dagger_interval=5, stochastic_policy=True,
                 random_action_p=0, stochastic_test=False):
        self.policy = policy
        self.optimizer = optimizer
        self.correction_horizon = correction_horizon

        self.loss_type = loss_type
        self.custom_loss_fn = custom_loss_fn
        self.logger_output = logger_output
        self.dagger_interval = dagger_interval
        self.stochastic_policy = stochastic_policy
        self.random_action_p = random_action_p
        self.stochastic_test = stochastic_test
        if gpu_enabled():
            self.policy.cuda()

    def train_epoch(self, dataset, test=False):
        self.policy.eval() if test else self.policy.train()
        stats = OrderedDict([('loss', 0),
                             ('accuracy', 0)
                             ])
        for tensors in dataset.dataloader:
            self.optimizer.zero_grad()
            states, actions, correction, correction_len, hl_goal = dataset.get_policy_input(tensors)
            y_hat = self.policy.forward(states.float(), correction, correction_len, hl_goal)
            loss = self.loss(Variable(actions), y_hat).mean(0)
            if not test:
                loss.backward()
                self.optimizer.step()

            stats['accuracy'] += get_numpy((actions.argmax(1) == y_hat.probs.argmax(1)).float().mean())
            stats['loss'] += float(get_numpy(loss))

        stats['loss'] /= len(dataset.dataloader)
        stats['accuracy'] /= len(dataset.dataloader)
        stats['size'] = dataset.size
        prefix = 'test' if test else 'train'
        stats = OrderedDict({'%s %s' % (prefix, k): v for k,v in stats.items()})
        return stats

    def generate_initial_data(self, dataset):

        for i, env in enumerate(dataset.envs):
            print('Generating data for env ', i)
            context = [np.zeros((1, dataset.correction_dim), dtype=np.float32)]
            traj = rollout_correction(self.policy, env, dataset.path_len,
                           add_inputs=[context])
            dataset.generate_correction(traj, i, cuda=True, add_to_data=True)


    def test_corrections(self, dataset, dataset_name, num_corrections=1,
                         add_to_data=False):
        self.policy.eval()
        returns = np.zeros((num_corrections))
        has_finished = np.zeros((num_corrections))
        completion = np.zeros((num_corrections))
        subgoals_left = np.zeros((num_corrections))
        context = np.zeros((dataset.max_corrections, dataset.correction_dim))
        for i, env in enumerate(dataset.envs):
            context[:] = 0
            #corrections = [dataset.first_subgoals[i]]
            prev_state = env.reset()[:env.obs_len]
            for c in range(num_corrections):
                context[c, :] = dataset.subgoal_encodings[i][c]
                traj = rollout_correction(self.policy, env, dataset.path_len,
                               add_inputs=[np.expand_dims(context, 0),
                                           np.array([c+1]), np.expand_dims(dataset.get_hl_goal(i), 0)],
                               deterministic=not self.stochastic_policy or (not self.stochastic_test and dataset_name=='test'),
                                          random_action_p=self.random_action_p)
                if add_to_data:
                    dataset.generate_correction(traj, i, cuda=True, add_to_data=add_to_data, deterministic=True)

                traj_return = np.sum(traj['rewards'])
                returns[c] += traj_return
                if sum([x['has_finished'] for x in traj['infos']]) > 0:
                    has_finished[c] += 1
                completion[c] += 1 - (traj['infos'][-1]['subgoals_left'] / float(len(traj['infos'][-1]['all_subgoals'])))
                subgoals_left[c] += traj['infos'][-1]['subgoals_left']
        returns /= len(dataset.envs)
        has_finished /= len(dataset.envs)
        completion /= len(dataset.envs)
        subgoals_left /= len(dataset.envs)

        stats = OrderedDict([('Mean Return CI %d' % i, returns[i]) for i in range(num_corrections)] +
                            [('Mean Finished CI %d' % i, has_finished[i]) for i in range(num_corrections)] +
                            [('Completion CI %d' % i, completion[i]) for i in range(num_corrections)] +
                            [('Mean Subgoals Left CI %d' % i, subgoals_left[i]) for i in range(num_corrections)]
                            )
        stats = OrderedDict([(dataset_name + ' ' + k, v) for k, v in stats.items()])

        return stats

    def train(self, train_dataset, iters=100, test_dataset=None):

        #self.generate_initial_data(train_dataset)
        #if test_dataset is not None:
        #    self.generate_initial_data(test_dataset)


        for train_itr in range(iters):
            with logger.prefix("SL Train Iter %d | " % train_itr):

                train_loss = self.train_epoch(train_dataset)
                test_loss = self.train_epoch(test_dataset, test=True) if test_dataset is not None else {}

                stats = OrderedDict(list(train_loss.items()) + list(test_loss.items()))


                if train_itr % self.dagger_interval == 0:
                    train_stats = self.test_corrections(train_dataset, 'train', num_corrections=1,
                                                        add_to_data=True if train_itr > 10 else False)
                    test_stats = self.test_corrections(test_dataset, 'test', num_corrections=1,
                                                       add_to_data=False) if test_dataset is not None else {}
                    c_stats = OrderedDict(list(train_stats.items()) + list(test_stats.items()))

                #if train_itr % 10 == 0:
                #    self.render(train_dataset, 0, 1)

                if train_itr % 10 == 0 and logger.get_snapshot_dir() is not None:
                    self.save(logger.get_snapshot_dir() + '/snapshots', train_itr)

                stats = OrderedDict(list(stats.items()) + list(c_stats.items()))
                logger.print_diagnostics(stats)
                logger.write_tabular(stats, self.logger_output)


    def render(self, dataset, env_idx, num_corrections):
        env = dataset.envs[env_idx]
        correction = dataset.first_subgoals[env_idx]
        prev_state = env.reset()[:env.obs_len]
        for c in range(num_corrections):
            traj = rollout_correction(self.policy, env, dataset.path_len,
                           add_inputs=[prev_state, np.expand_dims(correction, 0)],
                           deterministic=True, plot=True)

    def save(self, snapshot_dir, itr):
        import os
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        torch.save(self.policy.state_dict(), snapshot_dir + '/policy.pkl')

    def load(self, snapshot_dir, itr):
        self.policy.load_state_dict(torch.load(snapshot_dir + '/policy.pkl'))


    def loss(self, y, y_hat):
        if self.custom_loss_fn is not None:
            return self.custom_loss_fn(y, y_hat)
        if self.loss_type == 'mse':
            return self.mse_loss(y, y_hat.mean)
        if self.loss_type == 'll':
            return -y_hat.log_likelihood(y)
        elif self.loss_type == 'cross_entropy':
            return self.cross_entropy_loss(y, y_hat)
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

    def cross_entropy_loss(self, y, y_hat):
        ll = -torch.log((y * y_hat.probs).sum(1) + EPS)
        #import pdb; pdb.set_trace()
        # dealing with categorical y_hat and onehot variable y
        # ll = -torch.log((y * y_hat.prob).sum(1) + EPS)
        # handle multidim outputs
        # if len(ll.shape) > 1:
        #     for i in range(len(ll.size()) - 1):
        #         ll = ll.mean(-1)
        return ll
    
    def kl_loss(self, y, y_hat):
        kl = (y * (torch.log(y + 1e-8) - torch.log(y_hat.probs + 1e-8))).sum(1)
        return kl

def load_expert_pol(path=None):
    def make_policy():
        multid_obs = (5, 7, 7)
        obs_dim = np.prod(multid_obs)
        action_dim = 6

        obs_enc_dim = 256
        extra_obs_enc_dim = 32
        extra_obs_dim = 10
        obs_network = Convnet((multid_obs[0], *multid_obs[1:]), obs_enc_dim, filter_sizes=((16, 2), (16, 2)),
                              output_act=torch.nn.ReLU, hidden_sizes=(256, 256), flat_dim=400)
        extra_obs_network = MLP(extra_obs_dim, extra_obs_enc_dim, final_act=torch.nn.ReLU)

        policy = CategoricalMinigetNetwork(obs_network=obs_network,
                                           extra_obs_network=extra_obs_network,
                                           prob_network=MLP(obs_enc_dim + extra_obs_enc_dim, action_dim,
                                                            hidden_sizes=(128,),
                                                            final_act=torch.nn.Softmax),
                                           output_dim=action_dim,
                                           obs_len=obs_dim,
                                           obs_shape=multid_obs)
        return policy

    policy = make_policy()
    if path is not None:
        policy.load_state_dict(torch.load(path))
    return policy
