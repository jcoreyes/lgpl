import os
import numpy as np
import torch
import gym
from glob import glob
from lgpl.policies.discrete import CategoricalNetwork
from lgpl.models.mlp import MLP
from lgpl.envs.env_utils import register_env
from lgpl.samplers.rollout import rollout
from lgpl.utils.torch_utils import set_gpu_mode
from gym.envs.registration import register

set_gpu_mode(False)
data_dir = os.path.expanduser('~') + '/levine/irl/data/'
traj_dir = os.path.join(data_dir, 'trajectories')
if not os.path.isdir(traj_dir):
    os.mkdir(traj_dir)
env_name = 'Maze-v0'
EXP_NAME = 'train-vpg-q'
# number of policy iterations
NUM_ITER = 50
# how often policies were sampled
MULT_POLICY = 2
# time horizon for trajectories collected from each policy
PATH_LENGTH = 1000
# end-inclusive size of sliding window to find slope to decide when to save trajectories
SLOPE_WINDOW = 3
# counting how many mazes were solved at all
success_maze = 0
fail_maze = 0
# which mazes were solved
success_mazes = []

try:
    for ii in range(1, 1000):
        print('Starting maze %03d' % ii)
        maze_dir = os.path.join(traj_dir, 'maze_%03d' % ii)
        if not os.path.isdir(maze_dir):
            os.mkdir(maze_dir)

        maze_name = "maze2d_%03d.npy" % ii
        maze_path = os.path.join(data_dir, 'mazes', maze_name)

        env_name = 'Maze%d-v0' % ii
        register(id=env_name, entry_point='lgpl.envs.gym_maze.maze_env:MazeEnv',
                 kwargs=dict(maze_file=maze_path, onehot_version=True))
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        policy = CategoricalNetwork(MLP(obs_dim, action_dim, final_act=torch.nn.Softmax, hidden_sizes=(128, 128)),
                                    action_dim)
        try:
            exp_dir = glob('**/%s/**/*%03d--s-111' % (EXP_NAME, ii), recursive=True)[0]
        except:
            continue

        snapshot_dir = os.path.join(exp_dir, 'snapshots')
        if not os.path.isdir(snapshot_dir):
            continue

        rwds = []
        slopes = [0 for _ in range(SLOPE_WINDOW - 1)]
        trajs = {}

        for jj in range(NUM_ITER):
            with open(os.path.join(snapshot_dir, 'policy_%d.pkl' % (jj * MULT_POLICY)), 'rb') as policy_file:
                # load in policy info
                policy.load_state_dict(torch.load(policy_file))

                # take best of 20 nondeterministic rollouts
                best_ro = rollout(policy, env, PATH_LENGTH, deterministic=False)
                best_sample_return = sum(best_ro['rewards'])
                for _ in range(19):
                    ro = rollout(policy, env, PATH_LENGTH, deterministic=False)
                    returns = sum(ro['rewards'])
                    if best_sample_return < returns:
                        best_sample_returns = returns
                        best_ro = ro

                # append return and slope
                rwds.append(best_sample_return)
                if len(rwds) >= SLOPE_WINDOW:
                    slopes.append(rwds[-1] - rwds[-SLOPE_WINDOW])

                # put trajectory in numpy array
                obs_arr = np.array(best_ro['obs'])
                rewards_arr = np.array(best_ro['rewards']).reshape((-1, 1))
                actions_arr = np.array([int(a) for a in best_ro['actions']]).reshape((-1, 1))
                ro_np = np.hstack((obs_arr, rewards_arr, actions_arr))
                trajs[jj * MULT_POLICY] = ro_np

        slopes = np.array(slopes)
        rwds = np.array(rwds)

        best_iter = rwds.argmax()
        best_rwd = rwds[best_iter]
        try:
            start_idx = np.where(rwds > 0.1*best_rwd)[0][0]
        except:
            # best reward is negative
            fail_maze += 1
            print('Failed maze %d' % ii)
            continue
        end_idx = np.where(rwds > 0.9*best_rwd)[0][0] + 1
        success_maze += 1
        success_mazes.append(ii)
        for traj in range(start_idx, end_idx):
            np.save(os.path.join(maze_dir, 'maze_%03d_iter_%02d_traj' % (ii, traj * MULT_POLICY)),
                    trajs[traj * MULT_POLICY])
except Exception as e:
    print(e)
    import pdb; pdb.set_trace()
import pdb; pdb.set_trace()
print('Solved %d mazes, failed %d mazes' % (success_maze, fail_maze))
