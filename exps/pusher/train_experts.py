import argparse
import os
import pickle
from os import getcwd

import gym
import numpy as np
import torch
import torch.optim as optim
from doodad.easy_sweep.hyper_sweep import Sweeper
from lgpl.algos.ppo import PPO
from lgpl.launchers.launcher_util import run_experiment
from lgpl.models.mlp import MLP
from lgpl.models.baselines import NNBaseline
from lgpl.policies.discrete import CategoricalNetwork
import torch.nn as nn
# Construct maze env

def run_exp(vv):

    import lgpl.envs.pusher.pusher_env3
    pusher_id = vv['pusher_id']
    env_id = 'PusherEnv2v%d-v%d' % (pusher_id, vv['env_version'])
    env = gym.make(env_id)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    path_len = 350
    # Make policy
    policy = CategoricalNetwork(prob_network=MLP(obs_dim, action_dim, final_act=nn.Softmax), output_dim=action_dim)

    baseline = NNBaseline(MLP(obs_dim, 1, hidden_sizes=(64, 64)))
    n_envs = 20
    algo = PPO(env, env_id, policy, baseline, obs_dim=obs_dim, action_dim=action_dim,
               optimizer=optim.Adam(list(policy.parameters()) + list(baseline.parameters()), lr=1e-3), max_path_length=path_len,
               batch_size=path_len*n_envs, plot=False, n_itr=500, save_step=2, use_gae=False, terminate_early=True,
               save_last=True, ppo_batch_size=128, entropy_bonus=0.001)

    algo.train()



parser = argparse.ArgumentParser()

variant_group = parser.add_argument_group('variant')
variant_group.add_argument('--exp_dir', default='tmp')
variant_group.add_argument('--gpu', action='store_true')
variant_group.add_argument('--mode', default='local')
variant_group.add_argument('--exp_id_start', default=0)
variant_group.add_argument('--exp_id_stride', default=1)
parser.set_defaults(gpu=False)
v_command_args = parser.parse_args()
command_args = {k.dest:vars(v_command_args)[k.dest] for k in variant_group._group_actions}


params = {
    'seed': [111],
    'env_version': [4],
    'pusher_id': range(int(command_args['exp_id_start']), 1000, int(command_args['exp_id_stride']))
}



for args in Sweeper(params, 1):
    exp_dir = command_args['exp_dir']
    base_log_dir = getcwd() + '/data/exps/pusher/%s/' % (exp_dir)
    use_gpu = command_args['gpu']
    instance_type = 'c4.xlarge'
    run_experiment(
        run_exp,
        exp_id=args['pusher_id'],
        instance_type=instance_type,
        use_gpu=use_gpu,
        mode=command_args['mode'],
        seed=args['seed'],
        prepend_date_to_exp_prefix=False,
        exp_prefix=exp_dir,
        base_log_dir=base_log_dir,
        variant={**args, **command_args},
        spot_price=0.08,
        region='us-west-1',
        sync_interval=600,
    )
    break # Remove if training all envs