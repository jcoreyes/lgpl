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
from lgpl.utils.torch_utils import from_numpy, set_gpu_mode
from lgpl.envs.gym_minigrid.env_utils import load_expert_pol

def run_exp(vv):
    import lgpl.envs.gym_minigrid.envs
    set_gpu_mode(vv['gpu'])
    env_id = 'MiniGrid-SixRoomAbsPickPlaceEnv%d-v5' % vv['env_id']
    env = gym.make(env_id)
    multid_obs = env.multid_observation_space.shape
    obs_dim = np.prod(multid_obs)
    action_dim = env.action_space.n

    path_len = 100
    extra_obs_dim = 10
    # Make policy
    baseline = NNBaseline(MLP(obs_dim + extra_obs_dim, 1))

    policy = load_expert_pol()
    algo = PPO(env, env_id, policy, baseline, obs_dim=obs_dim, action_dim=action_dim,
               optimizer=optim.Adam(list(policy.parameters()) + list(baseline.parameters()), lr=1e-3), max_path_length=path_len,
               batch_size=path_len*20, plot=False, n_itr=200, save_step=5, use_gae=False, terminate_early=True, save_last=True,
               entropy_bonus=0.001)

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

env_ids = range(int(command_args['exp_id_start']), 3240, int(command_args['exp_id_stride']))

params = {
    'seed': [111],
    'env_id': env_ids
}

for args in Sweeper(params, 1):
    exp_dir = command_args['exp_dir']
    base_log_dir = getcwd() + '/data/exps/%s/' % (exp_dir)
    use_gpu = command_args['gpu']
    instance_type = 'c4.xlarge'
    exp_id = args['env_id']
    run_experiment(
        run_exp,
        exp_id=exp_id,
        instance_type=instance_type,
        use_gpu=use_gpu,
        mode=command_args['mode'],
        seed=args['seed'],
        prepend_date_to_exp_prefix=False,
        exp_prefix=exp_dir,
        base_log_dir=base_log_dir,
        variant={**args, **command_args},
        spot_price=0.06,
        region='us-east-2'
    )
    break # Remove if training all envs

