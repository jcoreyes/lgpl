import argparse
from os import getcwd
from os.path import join
import gym
import numpy as np
import torch
import torch.nn as nn
from doodad.easy_sweep.hyper_sweep import Sweeper
from lgpl.algos.lgpl import LGPL
from lgpl.datasets.lgpl_dataset_minigrid import LGPLDatasetMiniGrid
from lgpl.launchers.launcher_util import run_experiment
from lgpl.models.convnet import Convnet, Sent1dConvEncoder, TemporalConvEncoder
from lgpl.models.mlp import MLP
from lgpl.policies.discrete import CategorgicalMinigridBaselinePolicy
from sklearn.model_selection import train_test_split
from lgpl.utils.torch_utils import set_gpu_mode

import torch.optim as optim
from lgpl.utils.logger_utils import get_repo_dir
from lgpl.envs.gym_minigrid.env_utils import load_expert_pols

def run_exp(vv):
    import lgpl.envs.gym_minigrid.envs
    expert_dir = join(get_repo_dir(), 'data/exps/pickplace9/pickplace9/')
    env_names, expert_policies = load_expert_pols(expert_dir, vv['n_envs'])
    train_idx, test_idx, env_ids = train_test_split(np.arange(len(env_names)), train_size=0.90, shuffle=True)

    max_corrections = vv['max_corrections']
    extra_obs_dim = 10

    path_len = 100
    traj_subsample = vv['traj_subsample']

    env = gym.make(env_names[0])
    correction_dim = env.correction_dim
    multid_obs = env.multid_observation_space.shape
    obs_dim = np.prod(multid_obs)
    action_dim = env.action_space.n
    hl_goal_dim = env.full_info_dim

    embedding_dim = vv['embedding_dim']
    vocab_size = env.vocab_size
    obs_enc_dim = vv['obs_net']['output_dim']
    hlg_enc_dim = vv['hlg_encoder']['output_dim']
    extra_obs_enc_dim = vv['extra_obs_net']['output_dim']


    obs_network = Convnet(multid_obs, output_act=nn.ReLU, **vv['obs_net'])
    extra_obs_network = MLP(extra_obs_dim, final_act=nn.ReLU, **vv['extra_obs_net'])

    hlg_encoder = MLP(hl_goal_dim, **vv['hlg_encoder'], final_act=nn.ReLU)

    prob_network = MLP(obs_enc_dim + extra_obs_enc_dim + hlg_enc_dim, action_dim, final_act=nn.Softmax,
                       **vv['prob_network'])

    policy = CategorgicalMinigridBaselinePolicy(multid_obs, prob_network, action_dim,
                                               hlg_encoder, obs_network, extra_obs_network)


    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    all_envs = [gym.make(env_name) for env_name in env_names]

    def slice(lst, idx):
        return [lst[i] for i in idx]
    train_dataset = LGPLDatasetMiniGrid(slice(env_names, train_idx),
                                  slice(all_envs, train_idx),
                                  slice(expert_policies, train_idx),
                 obs_dim, action_dim, extra_obs_dim, path_len, correction_dim, hl_goal_dim,
                                  batch_size=512, traj_subsample=traj_subsample, max_corrections=max_corrections)
    test_dataset = None
    if len(test_idx) > 0:
        test_dataset = LGPLDatasetMiniGrid(slice(env_names, test_idx),
                                      slice(all_envs, test_idx),
                                      slice(expert_policies, test_idx),
                     obs_dim, action_dim, extra_obs_dim, path_len, correction_dim, hl_goal_dim,
                                     batch_size=512, traj_subsample=traj_subsample, max_corrections=max_corrections)

    algo = LGPL(policy, optimizer, loss_type=vv['loss_type'], use_fullinfo=True)

    algo.train(train_dataset, test_dataset=test_dataset, iters=51)

parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir', default='tmp')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--mode', default='local')
parser.add_argument('--load_dir', default=None)
parser.set_defaults(gpu=True)
command_args = vars(parser.parse_args())



params = {
    'seed': [111, 222, 333, 444, 555],
    'no_vis': [False],
    'render': [True],
    'cuda': [True],
    'log_interval': [1],
    'loss_type': ['kl'],
    'n_envs': [2000],
    'max_corrections': [1],
    'traj_subsample': [25],
    'embedding_dim': [16],
    'obs_net':      [dict(filter_sizes=((8, 2), (8, 2)), hidden_sizes=(128, 128), flat_dim=200, output_dim=128,
                     batchnorm=True)],
    'hlg_encoder':        [dict(hidden_sizes=(64, 64), output_dim=64)],
    'extra_obs_net':           [dict(hidden_sizes=(16, ), output_dim=16)],
    'prob_network':            [dict(hidden_sizes=(128, 128),batchnorm=True)],

    'ablation': [None],
    'method': ['fullinfo']

}

exp_id = 0

for sweep_param in Sweeper(params, 1):
    exp_dir = command_args['exp_dir']
    base_log_dir = getcwd() + '/data/exps/minigrid/%s/' % (exp_dir)
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



