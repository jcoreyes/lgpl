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
from lgpl.models.convnet import Sent1dConvEncoder, TemporalConvEncoder
from lgpl.models.mlp import MLP
from lgpl.models.rnn import CorrectionTrajEncoder
from lgpl.policies.discrete import CategorgicalCorrectionPusherPolicy
from sklearn.model_selection import train_test_split

import torch.optim as optim
from lgpl.utils.logger_utils import get_repo_dir
from lgpl.envs.pusher.env_utils import load_expert_pols

def run_exp(vv):
    import lgpl.envs.pusher.pusher_env3
    expert_dir = join(get_repo_dir(), 'data/exps/pusher/pusher3v4-expert1/')
    env_names, expert_policies = load_expert_pols(expert_dir, vv['n_envs'])
    train_idx, test_idx = train_test_split(np.arange(len(env_names)), train_size=0.9, shuffle=True)

    max_corrections = vv['max_corrections']

    path_len = 350
    traj_subsample = vv['traj_subsample']

    env = gym.make(env_names[0])

    traj_enc_dim = vv['prev_traj_encoder']['output_dim']
    correction_enc_dim = vv['correction_encoder']['output_dim']
    correction_traj_enc_dim = vv['correction_traj_encoder']['output_dim']
    hlg_enc_dim = vv['hlg_encoder']['output_dim']
    embedding_dim = vv['embedding_dim']
    vocab_size = env.vocab_size
    obs_dim = env.observation_space.shape[0]
    action_dim = 4 #env.action_space.shape[0]
    hl_goal_dim = env.hl_goal_dim
    correction_dim = 3
    embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    correction_encoder = Sent1dConvEncoder(conv_network=TemporalConvEncoder(input_dim=(embedding_dim,), **vv['correction_encoder']),
                                           embeddings=embeddings)
    hlg_encoder = Sent1dConvEncoder(conv_network=TemporalConvEncoder(input_dim=(embedding_dim,), **vv['hlg_encoder']),
                                    embeddings=embeddings)
    traj_encoder = TemporalConvEncoder(input_dim=(obs_dim, ), **vv['prev_traj_encoder'])
    correction_traj_encoder = CorrectionTrajEncoder(encoder=MLP(input_dim=correction_enc_dim + traj_enc_dim,
                                                                **vv['correction_traj_encoder']),
                                                    correction_encoder=correction_encoder,
                                                    traj_encoder=traj_encoder, obs_dim=obs_dim)

    prob_network = MLP(obs_dim + hlg_enc_dim + correction_traj_enc_dim, action_dim, final_act=nn.Softmax,
                       **vv['prob_network'])

    policy = CategorgicalCorrectionPusherPolicy(obs_dim, prob_network, action_dim,
                                              correction_traj_encoder, hlg_encoder)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=vv['weight_decay'])

    all_envs = [gym.make(env_name) for env_name in env_names]

    def slice(lst, idx):
        return [lst[i] for i in idx]
    train_dataset = BaselineDatasetPusher2(slice(env_names, train_idx),
                                  slice(all_envs, train_idx),
                                  slice(expert_policies, train_idx),
                 obs_dim, action_dim, path_len, correction_dim, hl_goal_dim,
                                  batch_size=512, traj_subsample=traj_subsample, max_corrections=max_corrections,
                                           buffer_size=int(vv['buffer_size']))
    test_dataset = None
    if len(test_idx) > 0:
        test_dataset = BaselineDatasetPusher2(slice(env_names, test_idx),
                                      slice(all_envs, test_idx),
                                      slice(expert_policies, test_idx),
                     obs_dim, action_dim, path_len, correction_dim, hl_goal_dim,
                                     batch_size=512, traj_subsample=traj_subsample, max_corrections=max_corrections+5)

    algo = LGPL(policy, optimizer, **vv['algo_args'])

    if vv['load_dir'] is not None:
        algo.load(vv['load_dir'], 0)
        algo.run_corrections(test_dataset, 'test', max_corrections, render=True)
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
    'seed': [222],
    'no_vis': [False],
    'render': [True],
    'cuda': [True],
    'log_interval': [1],
    'algo_args': [dict(loss_type='kl', dagger_interval=5)],
    'n_envs': [1000],
    'max_corrections': [6],
    'weight_decay': [1e-7],
    'traj_subsample': [70],
    'buffer_size': [3e6],
    'prob_network': [dict(hidden_sizes=(256, 256, 256),batchnorm=True)],
    'prev_traj_encoder':  [dict(hidden_sizes=(16,), output_dim=8, filter_sizes=((8, 2),), flat_size=16,
                               pool=True, batchnorm_input=True)],
    'correction_encoder': [dict(hidden_sizes=(16,), output_dim=16, filter_sizes=((8, 2),), flat_size=16,)],
    'hlg_encoder':        [dict(hidden_sizes=(16,), output_dim=16, filter_sizes=((8, 2),), flat_size=16,)],
    'correction_traj_encoder': [dict(hidden_sizes=(32, ), output_dim=32)],
    'embedding_dim': [16],
    'method': ['lgpl']

}

exp_id = 0

for sweep_param in Sweeper(params, 1):
    exp_dir = 'pusher_lglp_exp1'
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


