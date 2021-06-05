import argparse
import os
import pickle
from os import getcwd
from os.path import join
import gym
import numpy as np
import torch
import torch.nn as nn
from doodad.easy_sweep.hyper_sweep import Sweeper
from lgpl.algos.lgpl import LGPL
from lgpl.datasets.baseline_dataset_pusher2 import BaselineDatasetPusher2
from lgpl.launchers.launcher_util import run_experiment
from lgpl.models.mlp import MLP
from lgpl.policies.discrete import CategorgicalCorrectionPusherPolicy, CategorgicalPusherBaselinePolicy
from sklearn.model_selection import train_test_split

import torch.optim as optim
from lgpl.utils.logger_utils import get_repo_dir
from lgpl.envs.pusher.env_utils import load_expert_pols

def run_exp(vv):
    import lgpl.envs.pusher.pusher_env3
    expert_dir = join(get_repo_dir(), 'data/s3/pusher/pusher3v4-expert1//')
    env_names, expert_policies = load_expert_pols(expert_dir, vv['n_envs'])
    train_idx, test_idx = train_test_split(np.arange(len(env_names)), train_size=0.90, shuffle=True)

    max_corrections = vv['max_corrections']

    path_len = 350
    traj_subsample = vv['traj_subsample']

    env = gym.make(env_names[0])

    hlg_enc_dim = vv['hlg_encoder']['output_dim']
    obs_dim = env.observation_space.shape[0]
    action_dim = 4 #env.action_space.shape[0]
    hl_goal_dim = env.full_info_dim
    correction_dim = 3

    hlg_encoder = MLP(hl_goal_dim, **vv['hlg_encoder'])

    prob_network = MLP(obs_dim + hlg_enc_dim, action_dim, final_act=nn.Softmax,
                       **vv['prob_network'])

    policy = CategorgicalPusherBaselinePolicy(obs_dim, prob_network, action_dim, hlg_encoder)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    all_envs = [gym.make(env_name) for env_name in env_names]

    def slice(lst, idx):
        return [lst[i] for i in idx]
    train_dataset = BaselineDatasetPusher2(slice(env_names, train_idx),
                                  slice(all_envs, train_idx),
                                  slice(expert_policies, train_idx),
                 obs_dim, action_dim, path_len, correction_dim, hl_goal_dim,
                                  batch_size=512, traj_subsample=traj_subsample, max_corrections=max_corrections)
    test_dataset = None
    if len(test_idx) > 0:
        test_dataset = BaselineDatasetPusher2(slice(env_names, test_idx),
                                      slice(all_envs, test_idx),
                                      slice(expert_policies, test_idx),
                     obs_dim, action_dim, path_len, correction_dim, hl_goal_dim,
                                     batch_size=512, traj_subsample=traj_subsample, max_corrections=max_corrections)

    algo = LGPL(policy, optimizer, **vv['algo_args'], use_fullinfo=True)

    if vv['load_dir'] is not None:
        algo.load(vv['load_dir'], 0)
        algo.test(test_dataset, test_dataset=test_dataset,
                  max_corrections=max_corrections-1)
    else:
        algo.train(train_dataset, test_dataset=test_dataset, iters=101, render=False)

parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir', default='tmp')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--mode', default='local')
parser.add_argument('--load_dir', default=None)
parser.set_defaults(gpu=True)
command_args = vars(parser.parse_args())



params = {
    'seed': [222, 333, 444, 555],
    'no_vis': [False],
    'render': [True],
    'cuda': [True],
    'log_interval': [1],
    'algo_args': [dict(loss_type='kl', dagger_interval=5)],
    'n_envs': [1000],
    'max_corrections': [1],
    'traj_subsample': [50],
    'prob_network': [dict(hidden_sizes=(256, 256, 256),batchnorm=True)],
    'hlg_encoder':        [dict(hidden_sizes=(16,), output_dim=16, )],
    'embedding_dim': [16],
    'method': ['fullinfo_pusher']

}

exp_id = 0

for sweep_param in Sweeper(params, 1):
    exp_dir = 'pusher_fullinfo_baselineexp1'
    base_log_dir = getcwd() + '/data/exps/pusher/%s/' % (exp_dir)
    instance_type = 'c4.xlarge'
    variant = {**command_args, **sweep_param}
    # Override args with sweep params if overlap
    for k, v in sweep_param.items():
        variant[k] = v

    variant['log_dir'] = base_log_dir

    run_experiment(
        run_exp,
        exp_id=exp_id,
        instance_type=instance_type,
        use_gpu=variant['gpu'],
        mode='local',
        seed=variant['seed'],
        prepend_date_to_exp_prefix=False,
        exp_prefix=exp_dir,
        base_log_dir=base_log_dir,
        variant=variant,
        spot_price=0.06,
        region='us-east-2'
    )
    exp_id += 1


