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
from lgpl.models.rnn import CorrectionTrajEncoder
from lgpl.policies.discrete import CategorgicalCorrectionMinigridPolicy
from sklearn.model_selection import train_test_split
from lgpl.utils.torch_utils import set_gpu_mode

import torch.optim as optim
from lgpl.utils.logger_utils import get_repo_dir
from lgpl.envs.gym_minigrid.env_utils import load_expert_pols


def slice(lst, idx):
    return [lst[i] for i in idx]

def run_exp(vv):
    import lgpl.envs.gym_minigrid.envs
    expert_dir = join(get_repo_dir(), 'data/exps/minigrid/pickplace9/pickplace9/')
    env_names, expert_policies, env_ids = load_expert_pols(expert_dir, vv['n_envs'], env_version=5)
    train_idx, test_idx = train_test_split(np.arange(len(env_names)), train_size=0.9, shuffle=True)

    max_corrections = vv['max_corrections']
    extra_obs_dim = 10

    path_len = 100
    traj_subsample = vv['traj_subsample']

    env = gym.make(env_names[0])
    correction_dim = env.correction_dim
    multid_obs = env.multid_observation_space.shape
    obs_dim = np.prod(multid_obs)
    action_dim = env.action_space.n
    hl_goal_dim = env.hl_goal_dim

    embedding_dim = vv['embedding_dim']
    vocab_size = env.vocab_size
    obs_enc_dim = vv['obs_net']['output_dim']
    hlg_enc_dim = vv['hlg_encoder']['output_dim']
    extra_obs_enc_dim = vv['extra_obs_net']['output_dim']
    traj_enc_dim = vv['prev_traj_encoder']['output_dim']
    correction_enc_dim = vv['correction_encoder']['output_dim']
    correction_traj_enc_dim = vv['correction_traj_encoder']['output_dim']


    obs_network = Convnet(multid_obs, output_act=nn.ReLU, **vv['obs_net'])
    extra_obs_network = MLP(extra_obs_dim, final_act=nn.ReLU, **vv['extra_obs_net'])

    embeddings = nn.Embedding(vocab_size, embedding_dim)
    correction_encoder = Sent1dConvEncoder(conv_network=TemporalConvEncoder(input_dim=(embedding_dim,), **vv['correction_encoder']),
                                           embeddings=embeddings)
    hlg_encoder = Sent1dConvEncoder(conv_network=TemporalConvEncoder(input_dim=(embedding_dim,), **vv['hlg_encoder']),
                                    embeddings=embeddings)

    traj_obs_net = Convnet(multid_obs, output_act=nn.ReLU, **vv['traj_obs_net'])
    traj_encoder = TemporalConvEncoder(input_dim=(vv['traj_obs_net']['output_dim'], ), **vv['prev_traj_encoder'])

    correction_traj_encoder = CorrectionTrajEncoder(encoder=MLP(input_dim=correction_enc_dim + traj_enc_dim,
                                                                **vv['correction_traj_encoder']),
                                                    correction_encoder=correction_encoder,
                                                    traj_encoder=traj_encoder, obs_dim=obs_dim+extra_obs_dim,
                                                    obs_net=traj_obs_net, obs_shape=multid_obs)

    prob_network = MLP(obs_enc_dim + extra_obs_enc_dim + hlg_enc_dim + correction_traj_enc_dim, action_dim, final_act=nn.Softmax,
                       **vv['prob_network'])

    policy = CategorgicalCorrectionMinigridPolicy(multid_obs, prob_network, action_dim,
                                              correction_traj_encoder, hlg_encoder, obs_network, extra_obs_network)


    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    all_envs = [gym.make(env_name) for env_name in env_names]


    train_dataset = LGPLDatasetMiniGrid(slice(env_names, train_idx),
                                  slice(all_envs, train_idx),
                                  slice(expert_policies, train_idx),
                 obs_dim, action_dim, extra_obs_dim, path_len, correction_dim, hl_goal_dim,
                                  batch_size=512, traj_subsample=traj_subsample, max_corrections=max_corrections,
                                        buffer_size=int(vv['buffer_size']))
    test_dataset = None
    if len(test_idx) > 0:
        test_dataset = LGPLDatasetMiniGrid(slice(env_names, test_idx),
                                      slice(all_envs, test_idx),
                                      slice(expert_policies, test_idx),
                     obs_dim, action_dim, extra_obs_dim, path_len, correction_dim, hl_goal_dim,
                                     batch_size=512, traj_subsample=traj_subsample, max_corrections=max_corrections,
                                           buffer_size=int(1e5))

    algo = LGPL(policy, optimizer, loss_type=vv['loss_type'])

    algo.train(train_dataset, test_dataset=test_dataset, iters=202)

parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir', default='tmp')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--mode', default='local')
parser.add_argument('--load_dir', default=None)
parser.set_defaults(gpu=True)
command_args = vars(parser.parse_args())



params = {
    'seed': [333],
    'no_vis': [False],
    'render': [True],
    'cuda': [True],
    'log_interval': [1],
    'loss_type': ['kl'],
    'n_envs': [2500],
    'max_corrections': [6],
    'traj_subsample': [25],
    'embedding_dim': [16],
    'buffer_size': [2e6],
    'obs_net':      [dict(filter_sizes=((8, 2), (8, 2)), hidden_sizes=(32,), flat_dim=200, output_dim=32,
                     batchnorm=True)],
    'traj_obs_net': [dict(filter_sizes=((4, 2), (4, 2)), hidden_sizes=(16, ), flat_dim=100, output_dim=16,
                     batchnorm=True)],
    'prev_traj_encoder':  [dict(hidden_sizes=(16,), output_dim=2, filter_sizes=((8, 2),), flat_size=8, pool=True)],
    'correction_encoder': [dict(hidden_sizes=(16,), output_dim=32, filter_sizes=((8, 2),), flat_size=16, )],
    'hlg_encoder':        [dict(hidden_sizes=(16,), output_dim=4, filter_sizes=((8, 2),), flat_size=24, )],
    'correction_traj_encoder': [dict(hidden_sizes=(32, ), output_dim=32)],
    'extra_obs_net':           [dict(hidden_sizes=(16, ), output_dim=16)],
    'prob_network':            [dict(hidden_sizes=(128, 128),batchnorm=True)],

    'ablation': [None],
    'method': ['lgpl']

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


