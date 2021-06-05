from gym.envs import register
from gym.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from gym import utils
import pickle
from lgpl.envs.env_utils import get_asset_xml
from lgpl.utils.logger_utils import get_repo_dir
from lgpl.utils.torch_utils import get_numpy
from collections import OrderedDict
from itertools import combinations, chain
import gym

NUM_TASKS = 1000

GOAL_THRESH = 0.03

TOP_LEFT = 0.6, 0
WIDTH = 0.3
HEIGHT = 0.6
N_BLOCKS = 5

# BLOCKS = np.array([
#     0.68, 0.43,  # obst block 1 purple
#     1.1, 0.35,  # obst block 2 gray
#     0.82, -0.06,
#     1.05, -0.404,
#     0.667, -0.687,
# ]).reshape((5, 2))
GOAL_BLOCK_COLORS = ['red', 'cyan', 'yellow']
BLOCK_COLORS = ['purple', 'black', 'white', 'blue', 'green']
# all_goals = save_goal_samples(num_tasks)
# import IPython
# IPython.embed()

dirs_ord = OrderedDict([('up', np.array([0, 0.3])),
					('left', np.array([-0.3, 0])),
					('down', np.array([0, -0.3])),
					('right', np.array([0.3, 0]))])
dirs_ord_list = list(dirs_ord.keys())
dir_mtx = np.stack(dirs_ord.values())
dirs = dict(dirs_ord)

dirs2_ord = OrderedDict([('up', np.array([0, 0.3])),
					('left', np.array([-0.3, 0])),
					('down', np.array([0, -0.3])),
					('right', np.array([0.3, 0])),
                    ('upright', np.array([0.3, 0.3])),
                    ('upleft', np.array([-0.3, 0.3])),
                    ('downright', np.array([0.3, -0.3])),
                    ('downleft', np.array([-0.3, -0.3]))])
dir2_mtx = np.stack(dirs2_ord.values())
dirs2_ord_list = list(dirs2_ord.keys())

def generate_env(BLOCKS=None, debug=False):
    if BLOCKS is None:
        BLOCKS = np.zeros((N_BLOCKS, 2))
        for i in range(N_BLOCKS):
            while True:
                pos = np.random.uniform((0.6, -0.8), (1.1, 0.8))
                if i == 0 or np.linalg.norm(BLOCKS[:i, :] - pos, axis=-1).min() > 0.3:
                    BLOCKS[i, :] = pos
                    break

    def random_between(b1xy, b2xy):
        b1x, b1y = b1xy
        b2x, b2y = b2xy
        target = None
        for i in range(10000):
            target_cand = np.random.uniform((min(b1x, b2x),min(b1y, b2y)), (max(b1x, b2x), max(b1y, b2y)))
            if np.linalg.norm(BLOCKS - target_cand, axis=-1).min() > 0.1 and\
                         target_cand[0] > 0.5 and target_cand[0] < 1.3 and target_cand[1] < 1.1 and target_cand[1] > -1.1:
                target = target_cand
                break
        return target

    hlgs = ['between', 'cardinal', 'close']
    dirs = {'up': np.array([0, 0.3]),
            'left': np.array([-0.3, 0]),
            'down': np.array([0, -0.3]),
            'right': np.array([0.3, 0])}

    hlg = np.random.choice(hlgs)
    if hlg == 'between':
        b1, b2 = np.random.choice(N_BLOCKS, 2, replace=False)

        target = random_between(BLOCKS[b1], BLOCKS[b1])
        sent = '%s %s %s' %(hlg, BLOCK_COLORS[b1], BLOCK_COLORS[b2])
    elif hlg == 'cardinal':
        dir = list(dirs.keys())[np.random.choice(len(dirs))]
        b1 = np.random.choice(N_BLOCKS)
        b2xy = BLOCKS[b1, :] + dirs[dir]
        target = random_between(BLOCKS[b1, :], b2xy)
        sent = '%s of %s' % (dir, BLOCK_COLORS[b1])
        if debug:
            print(hlg, BLOCK_COLORS[int(b1)], dir, target)
    elif hlg == 'close':
        b1 = np.random.choice(N_BLOCKS)
        angle = np.random.uniform(0, 2*np.pi)
        b2xy = BLOCKS[b1, :] + 0.3 * np.array([np.cos(angle), np.sin(angle)])
        target = random_between(BLOCKS[b1, :], b2xy)
        sent = 'close to %s' % BLOCK_COLORS[b1]
        if debug:
            print(hlg, BLOCK_COLORS[int(b1)], angle * 180 / np.pi)

    block_choice = np.random.choice(3)
    hlg = dict(choice=block_choice, hlg=sent, goal_pos=target, block_pos=BLOCKS)

    if debug:
        env = PusherEnv2(**hlg)
        env.reset()
        for i in range(1000000):
            env.step(np.zeros(3))
            env.render()



    return hlg if target is not None else None

class Corrections2:
    def __init__(self, choice, goal, goal_pos, initial_goal_block_pos, initial_goal_block_poss, obst_blocks,
                 correction_type=3):

        self.choice = choice
        self.vocab = ['start', 'between', 'up', 'left', 'down', 'right', 'close', 'between', 'to', 'of', 'towards', 'alittle', 'alot',
                      'move', 'touch', 'the', 'upleft', 'upright', 'downleft', 'downright', 'push'] + BLOCK_COLORS + GOAL_BLOCK_COLORS
        self.word_to_idx = {k:i for i, k in enumerate(self.vocab)}
        self.correction_dim = 3
        self.vocab_size = len(self.vocab)
        self.goal = goal
        self.goal_pos = goal_pos
        self.initial_goal_block_pos = initial_goal_block_pos
        self.initial_goal_block_poss = initial_goal_block_poss.reshape((3, 2))
        self.obst_blocks = obst_blocks
        self.obs_goal_bx = 3 + 2 * choice
        self.correction_type = correction_type
        self.closest_blocks_to_goal = np.linalg.norm(self.goal_pos - self.obst_blocks, axis=1).argsort()

    def sent_to_idx(self, sent):
        return np.array([self.word_to_idx[i] for i in sent.split(' ')])

    def idx_to_sent(self, idx):
        return ' '.join([self.vocab[i] for i in idx])

    def sent_to_onehot(self, sent):
        onehot = np.eye(self.vocab_size)
        return np.concatenate([onehot[x] for x in self.sent_to_idx(sent)])

    def fixed_blockdir_from_goal(self, obs, choice, closest_block_idx):
        # target is left of green block or up of red block
        # get locations of 5 fixed blocks
        fixed_pos = obs[9:19].reshape((-1, 2))
        block_idx = np.linalg.norm(fixed_pos - self.goal_pos, axis=1).argsort()[closest_block_idx]
        v1 = self.goal_pos - fixed_pos[block_idx, :]
        angles = np.abs(np.array([self.py_ang(v1, v2) for v2 in dir2_mtx]))

        # return correction such as 'left of green'
        return '%s of %s' % (dirs2_ord_list[angles.argmin()], BLOCK_COLORS[block_idx])

    def block_from_target(self, obs, choice):
        goal_pos = self.goal[2:4]
        goal_block_pos = obs[3 + 2 * choice:5 + 2 * choice]
        fixed_pos = obs[9:19].reshape((-1, 2))
        # dist from fixed to goal block
        dist_to_goal_blk = np.linalg.norm(fixed_pos - goal_block_pos, axis=1)
        dist_to_goal = np.linalg.norm(fixed_pos - goal_pos, axis=1)
        # pick block with lowest sum of dists
        block_num = (dist_to_goal + dist_to_goal_blk).argmin()
        return 'move towards %s' % BLOCK_COLORS[block_num]

    def py_ang(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'    """
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)

    def rel_dir_to_goal(self, obs, choice):
        # Find relative direction of goal block to target ie move to left
        goal_block_pos = obs[3 + 2 * choice:5 + 2 * choice]
        dist = np.linalg.norm(goal_block_pos - self.goal_pos)
        v1 = self.goal_pos - goal_block_pos
        angles = np.abs(np.array([self.py_ang(v1, v2) for v2 in dir2_mtx]))
        return 'move to %s' % dirs2_ord_list[angles.argmin()]

    def rel_dir_dist_to_goal(self, obs, choice):
        # Find relative direction of goal block to target ie move to left
        goal_block_pos = obs[3 + 2 * choice:5 + 2 * choice]
        dist = np.linalg.norm(goal_block_pos - self.goal_pos)
        v1 = self.goal_pos - goal_block_pos
        angles = np.abs(np.array([self.py_ang(v1, v2) for v2 in dir2_mtx]))
        if dist < 0.3:
            mod = 'alittle'
        else:
            mod = 'alot'
        return 'push %s %s' % (mod, dirs2_ord_list[angles.argmin()])


    def closest_block_dir(self, obs, choice):
        # Find closest block to goal pos and returns relative direction of that
        fixed_pos = obs[9:19].reshape((-1, 2))
        block_num = np.linalg.norm(fixed_pos - self.goal_pos, axis=1).argmin()
        closest_pos = self.goal_pos - fixed_pos[block_num, :]
        dirs_blk = closest_pos @ dir_mtx

        return '%s of %s' %(dirs_ord_list[dirs_blk.argmax()], BLOCK_COLORS[block_num])


    def gen_corr(self, traj_obs, traj_info):
        goal_block_pos = traj_obs[-1, self.obs_goal_bx:self.obs_goal_bx+2]
        goal_block_poss = traj_obs[-1, 3:3+6].reshape((3, 2))

        if np.linalg.norm(goal_block_pos - self.initial_goal_block_pos) < 0.03 or \
                np.linalg.norm(goal_block_poss - self.initial_goal_block_poss, axis=-1).argmax() != self.choice:
            correction_sent = 'touch the %s' % GOAL_BLOCK_COLORS[self.choice]
        else:
            sent1 = self.rel_dir_dist_to_goal(traj_obs[-1], self.choice)
            sent2 = self.fixed_blockdir_from_goal(traj_obs[-1], self.choice, 0)
            sent3 = self.fixed_blockdir_from_goal(traj_obs[-1], self.choice, 1)
            sent4 = self.block_from_target(traj_obs[-1], self.choice)
            correction_sents = [sent1, sent2, sent3, sent4]
            correction_sent = np.random.choice(correction_sents, p=[0.5, 0.2, 0.2, 0.1])
        correction = self.sent_to_idx(correction_sent)
        return correction

class PusherEnv3(MujocoEnv, utils.EzPickle):

    FILE = 'pusher_env2.xml'

    def __init__(self, choice=0, goal_pos=(1, 1), hlg=None, block_pos=None):
        self.choice = choice
        self.obst_blocks = block_pos.copy()
        self.obst_blocks[:, 0] -= np.arange(0, 5) * 0.2

        self.goal = np.array([ 1., 3.,
                               0.568, -0.272, # goal pos 1
                               0., 0., 0, # gripper x y angle
                               0.3, 0.2, # block pos 1 red
                               0.3, 0., # block pos 2 cyan
                               0.3, -0.2, # block pos 3 yellow
                               ] + self.obst_blocks.flatten().tolist())
        self.goal[2:4] = goal_pos
        self.goal_pos = goal_pos

        self.initial_goal_block_pos = self.goal[7+self.choice*2:7+self.choice*2+2]
        self.initial_goal_block_poss = self.goal[7:7+6]
        self.initial_block_dist = np.linalg.norm(goal_pos - self.initial_goal_block_pos)
        self.corrections = Corrections2(self.choice, self.goal, goal_pos, self.initial_goal_block_pos,
                                        self.initial_goal_block_poss, block_pos)

        self.close_block = np.random.choice(self.corrections.closest_blocks_to_goal[:2])
        self.hlg = '%s close %s' % (GOAL_BLOCK_COLORS[self.choice], BLOCK_COLORS[self.close_block])

        self.hl_goal = self.corrections.sent_to_idx(self.hlg)
        self.hl_goal_dim = self.hl_goal.size
        self.correction_dim = self.corrections.correction_dim
        self.vocab_size = self.corrections.vocab_size
        self.full_info = np.concatenate([self.goal_pos, np.eye(3)[self.choice]])
        self.full_info_dim = self.full_info.size

        MujocoEnv.__init__(self, get_asset_xml('pusher_env2.xml'), 1)
        self.action_space = gym.spaces.Discrete(4)

    def reset_model(self, reset_args=None):
        goal_pos = self.goal[2:4]

        body_pos = self.model.body_pos.copy()

        poss = self.goal[13:].reshape((-1, 2))
        for i in range(poss.shape[0]):
            body_pos[-6+i][:2] = poss[i]

        body_pos[-1][:2] = goal_pos

        self.model.body_pos[:] = body_pos

        curr_qvel = np.zeros_like(self.sim.data.qvel)

        curr_qpos = self.goal[4:13].copy()

        # add random noise  to gripper
        curr_qpos[:3] += np.random.uniform(low=-0.05, high=0.05, size=3)
        curr_qvel[:3] += np.random.randn(3) * 0.1

        #import pdb; pdb.set_trace()
        self.set_state(curr_qpos, curr_qvel)
        self.sim.forward()
        # TODO Might need this.Commented out in porting from rllab mujoco env to gym mujoco env
        #self.current_com = self.model.data.com_subtree[0]
        #self.dcom = np.zeros_like(self.current_com)

        return self.get_current_obs()

    def get_current_obs(self):

        return np.concatenate([
            self.sim.data.qpos.flat[:3],
            self.sim.data.geom_xpos[-9:-1, :2].flat,
            self.sim.data.qvel.flat,
        ]).reshape(-1)

    def reward(self, next_obs, action):
        # PLACE_RWD = 10  # one time reward for solving block. also negative penalty for unsolving block
        # OFFSET_RWD = 15 # positive offset so that rewards aren't negative
        # EPS_CLOSE = 0.1 # distance to count as solved
        #

        cbx = 3 + 2 * self.choice

        # TODO: Maybe need to change angle here
        curr_gripper_pos = self.sim.data.site_xpos[0, :2]
        #curr_block_pos = np.array([next_obs[curr_block_xidx], next_obs[curr_block_yidx]]).T
        curr_block_pos = next_obs[cbx:cbx+2]
        dist_to_block = np.linalg.norm(curr_gripper_pos - curr_block_pos)
        block_dist = np.linalg.norm(self.goal_pos - curr_block_pos)

        ctrl_cost = 0 #np.sum(np.square(action))
        # return -(0.01 * ctrl_cost + 0.02 * dist_to_block + 0.1 * block_dist), block_dist, ctrl_cost
        return -(0.15 * ctrl_cost + dist_to_block + 5 * block_dist), block_dist, ctrl_cost

    def step(self, action):

        if self.action_space is None or action.size == 3:
            action = 0

        ctrl = 1.3
        if action == 0:
            a = np.array([0.0, ctrl, 0.0])
        elif action == 1:
            a = np.array([0.0, -ctrl, 0.0])
        elif action == 2:
            a = np.array([ctrl, 0.0, 0.0])
        elif action == 3:
            a = np.array([-ctrl, 0.0, 0.0])
        else:
            raise Exception
        #import pdb; pdb.set_trace(
        self.do_simulation(a, self.frame_skip)

        next_obs = self.get_current_obs()
        reward, block_dist, ctrl_cost = self.reward(next_obs, action)
        has_finished = block_dist < GOAL_THRESH

        completion = 1 - (min(block_dist, self.initial_block_dist) / self.initial_block_dist)


        done = False
        info = {'has_finished': has_finished, 'completion': completion,
                'goal_pos': self.goal, 'choice': self.choice, 'ctrl_cost': ctrl_cost}
        return next_obs, reward, done, info

    def log_diagnostics(self, sd, itr):
        blockchoice = self.choice
        goal_poss = self.goal_pos

        cbx = 3 + 2 * blockchoice
        sd_obs = sd['obs']
        if len(sd['obs'].shape) == 2:
            sd_obs = np.expand_dims(sd_obs, 0)
        initial_block_poss = sd_obs[:, 0, cbx:cbx+2]
        curr_block_poss = sd_obs[:, -1, cbx:cbx+2]
        initial_block_dist = np.linalg.norm(goal_poss - initial_block_poss, axis=-1)
        block_dist = np.linalg.norm(goal_poss - curr_block_poss, axis=-1)
        #ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(get_numpy(sd['actions'])), -1).mean()
        return OrderedDict(
            [('Avg Initial Block Dist', initial_block_dist.mean()),
             ('Avg Final Block Dist', block_dist.mean()),
             ('Mean_Finished', (block_dist < GOAL_THRESH).mean()),
             #('Ctrl_Cost', sd['infos'][-1])
             ])
         #   ('Ctrl Cost', ctrl_cost)

def register_envs(envs_files, v):

    sent_env_pairs = []
    for env_file in envs_files:
        sent_env_pairs += pickle.load(open(get_repo_dir() + env_file, 'rb'))
    env_idx = 0
    env_names = []
    for env_kwargs in sent_env_pairs:
        env_name = 'PusherEnv2v%d-v%d' % (env_idx, v)
        register(env_name,
                 entry_point='lgpl.envs.pusher.pusher_env3:PusherEnv3',
                 kwargs=env_kwargs)
        env_names.append(env_name)
        env_idx += 1


def generate_envs1():
    sent_env_pairs = []
    while len(sent_env_pairs) < 1000:
        env_args = generate_env(debug=False)
        if env_args is not None:
            sent_env_pairs.append(env_args)
        print(len(sent_env_pairs))
    save_file = 'env_data/pusher/pusher_envs1.pkl'
    pickle.dump(sent_env_pairs, open(save_file, 'wb'))

def generate_envs2(save=False):
    sent_env_pairs = []
    block_pos = generate_env(debug=False)['block_pos']
    while len(sent_env_pairs) < 1000:
        env_args = generate_env(BLOCKS=block_pos, debug=False)
        if env_args is not None:
            sent_env_pairs.append(env_args)
        print(len(sent_env_pairs))
    if save:
        save_file = 'env_data/pusher/pusher_envs3.pkl'
        pickle.dump(sent_env_pairs, open(save_file, 'wb'))


register_envs(['/env_data/pusher/pusher_envs3.pkl'], 4)
if __name__ == "__main__":
    #generate_envs2()
    pass

