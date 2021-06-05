from gym.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from gym import utils
import pickle
from lgpl.envs.env_utils import get_asset_xml
from lgpl.utils.logger_utils import get_repo_dir
from lgpl.utils.torch_utils import get_numpy
from collections import OrderedDict
from itertools import combinations, chain

NUM_TASKS = 1000
NUM_GOALS = 2
NUM_BLOCKS = 6
NUM_MOV_BLOCKS = 4

TOP_LEFT = 0.6, 0
WIDTH = 0.3
HEIGHT = 0.6

def save_goal_samples(num_tasks, filename, num_goals):
    all_goals = []
    blockarr = np.array(range(5))
    for _ in range(num_tasks):
        blockpositions = np.zeros((13, 1))
        blockpositions[0:3] = 0
        positions_sofar = []
        np.random.shuffle(blockarr)
        for i in range(5):
            xpos = np.random.uniform(.35, .65)
            ypos = np.random.uniform(-.5 + 0.2*i, -.3 + 0.2*i)
            blocknum = blockarr[i]
            blockpositions[3+2*blocknum] = -0.2*(blocknum + 1) + xpos
            blockpositions[4+2*blocknum] = ypos
            curr_pos = np.array([xpos, ypos])

        for idxs in combinations(range(5), num_goals):
            blockchoice = [blockarr[i] for i in idxs]
            goals = []
            for i in range(num_goals):
                goal_xpos = np.random.uniform(0, 1)
                goal_ypos = np.random.uniform(-.5 + .2*i, -.3 + .2*i)
                goals.append(np.array([goal_xpos, goal_ypos]))
            curr_goal = np.concatenate([blockchoice, *goals, blockpositions[:,0]])
            all_goals.append(curr_goal)
    all_goals = np.asarray(all_goals)
    np.random.shuffle(all_goals)
    pickle.dump(np.asarray(all_goals), open(filename, "wb"))
    return np.asarray(all_goals)

def save_goal_samples_rect(filename, num_tasks):
    all_goals = []
    blockarr = np.arange(NUM_MOV_BLOCKS)
    for _ in range(num_tasks):
        # rearrange the movable blocks
        np.random.shuffle(blockarr)

        # goals are sampled from region with top left corner (0.4, -0.3), width 0.5, and height 0.6
        goals = []
        for goal in range(NUM_GOALS):
            goals.append(np.random.rand(2) * np.array((0.5, 0.6)) + np.array((0.4, -0.3)))

        ref_objs = []
        for ref_obj in range(NUM_BLOCKS - NUM_MOV_BLOCKS):
            ref_objs.append(np.random.rand(2) * np.array((0.5, 0.6)) + np.array((0.4, -0.3)))

        blockpositions = np.zeros((3+2*NUM_BLOCKS, 1))
        blockpositions[0:3] = 0
        for idx, ref_obj in enumerate(ref_objs):
            blockpositions[3+2*(NUM_MOV_BLOCKS+idx)] = ref_obj[0]
            blockpositions[4+2*(NUM_MOV_BLOCKS+idx)] = ref_obj[1]
        combos = list(combinations(range(NUM_MOV_BLOCKS), NUM_GOALS))
        np.random.shuffle(combos)
        for idxs in combos:
            for idx, block in enumerate(blockarr):
                xpos = 0.15
                ypos = 0.2 * idx
                blockpositions[3+2*block] = xpos
                blockpositions[4+2*block] = ypos
            blockchoice = [blockarr[i] for i in idxs]
            all_goals.append(np.concatenate([blockchoice, np.concatenate(goals), blockpositions[:,0]]))
            break
    all_goals = np.asarray(all_goals)
    pickle.dump(all_goals, open(filename, "wb"))
    return all_goals

# save_goal_samples(NUM_TASKS, '/home/suvansh/levine/irl/data/pusher/pusher_trainSet_%dTasks_%dGoals__4.pkl' % (NUM_TASKS, NUM_GOALS), NUM_GOALS)
# save_goal_samples_rect('/home/suvansh/levine/irl/data/pusher/pusher_trainSet_rect_%d_by_%d_at_%d_%d.pkl' % (WIDTH*100, HEIGHT*100, TOP_LEFT[0]*100, TOP_LEFT[1]*100), TOP_LEFT, WIDTH, HEIGHT)
# save_goal_samples_rect('/home/suvansh/levine/irl/data/pusher/pusher_trainSet_rect_%d_tasks__2.pkl' % NUM_TASKS, NUM_TASKS)

# all_goals = save_goal_samples(num_tasks)
# import IPython
# IPython.embed()
class PusherEnvRandomized(MujocoEnv, utils.EzPickle):

    FILE = 'pusher_env.xml'

    def __init__(self, choice=0):
        self.choice = choice
        # all_goals = pickle.load(open(get_repo_dir() + '/data/pusher/pusher_trainSet_120Tasks_3Goals__4.pkl', "rb"))
        # all_goals = pickle.load(open(get_repo_dir() + '/data/pusher/pusher_trainSet_rect_%d_by_%d_at_%d_%d.pkl' % (WIDTH*100, HEIGHT*100, TOP_LEFT[0]*100, TOP_LEFT[1]*100), "rb"))
        all_goals = pickle.load(open(get_repo_dir() + '/data/pusher/pusher_trainSet_rect_%d_tasks__2.pkl' % NUM_TASKS, "rb"))
        self.all_goals = all_goals
        self.goal = self.all_goals[self.choice]
        self.solved = np.zeros(NUM_GOALS)
        MujocoEnv.__init__(self, get_asset_xml('pusher_env.xml'), 1)

    @staticmethod
    def make_env(blockchoice, goals, blockpositions):
        """
        :param blockchoice: list of distinct numbers 0-4 representing the blocks to be moved
        :param goals: list of len(BLOCKCHOICE) np arrays, each of length 2, specifying xy coordinates of goals
        :param blockpositions: gripper xy and angle, and block positions
        :return:
        """
        env = PusherEnvRandomized()
        env.goal = np.concatenate([blockchoice, *goals, blockpositions[:,0]])

    def reset_model(self, reset_args=None):
        blockchoice = self.goal[0]
        goal_poss = [self.goal[NUM_GOALS+2*i:NUM_GOALS+2+2*i] for i in range(NUM_GOALS)]
        blockpositions = self.goal[3*NUM_GOALS:]

        body_pos = self.model.body_pos.copy()
        for idx, goal_pos in enumerate(goal_poss):
            body_pos[-NUM_GOALS+idx][0] = goal_pos[0]
            body_pos[-NUM_GOALS+idx][1] = goal_pos[1]

        self.model.body_pos[:] = body_pos
        curr_qpos = blockpositions

        curr_qvel = np.zeros_like(self.sim.data.qvel)
        self.set_state(curr_qpos, curr_qvel)


        self.sim.forward()
        # TODO Might need this.Commented out in porting from rllab mujoco env to gym mujoco env
        #self.current_com = self.model.data.com_subtree[0]
        #self.dcom = np.zeros_like(self.current_com)

        return self.get_current_obs()

    def get_current_obs(self):

        return np.concatenate([
            self.sim.data.qpos.flat[:3],
            self.sim.data.geom_xpos[-8:-3, :2].flat,
            self.sim.data.qvel.flat,
        ]).reshape(-1)

    # def reward(self, next_obs, action):
    #     PLACE_RWD = 0.25  # at every timestep, per solved block
    #     FUTURE_MULT = 1.5
    #     FUTURE_WEIGHT = 1
    #     OFFSET_RWD = 10 # positive offset so that rewards aren't negative
    #
    #
    #     blockchoice = self.goal[:NUM_GOALS]
    #     curr_block_xidx = (3 + 2 * blockchoice).astype(int)
    #     curr_block_yidx = (4 + 2 * blockchoice).astype(int)
    #     # TODO: Maybe need to change angle here
    #     curr_gripper_pos = self.sim.data.site_xpos[0, :2]
    #     curr_block_pos = np.array([next_obs[curr_block_xidx], next_obs[curr_block_yidx]]).T
    #     goal_pos = np.stack([self.goal[NUM_GOALS + 2 * i:NUM_GOALS + 2 * i + 2] for i in range(NUM_GOALS)])
    #     dist_to_blocks = np.linalg.norm(curr_gripper_pos - curr_block_pos, axis=1)
    #     block_dists = np.linalg.norm(goal_pos - curr_block_pos, axis=1)
    #
    #     rwds = []
    #     i = 0
    #     for i in range(block_dists.shape[0]):
    #         if block_dists[i] < 0.2:
    #             rwds.append(PLACE_RWD)
    #             FUTURE_WEIGHT *= FUTURE_MULT
    #             if i == block_dists.shape[0] - 1:
    #                 # all solved!
    #                 FUTURE_WEIGHT = 0
    #         else:
    #             break
    #
    #     dist_to_block = dist_to_blocks[i]
    #     block_dist = block_dists[i]
    #     ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
    #     reward = FUTURE_WEIGHT * (OFFSET_RWD - dist_to_block - 5 * block_dist) - ctrl_cost + sum(rwds)

    def reward(self, next_obs, action):
        PLACE_RWD = 10  # one time reward for solving block. also negative penalty for unsolving block
        OFFSET_RWD = 15 # positive offset so that rewards aren't negative
        EPS_CLOSE = 0.1 # distance to count as solved

        blockchoice = self.goal[:NUM_GOALS]
        curr_block_xidx = (3 + 2 * blockchoice).astype(int)
        curr_block_yidx = (4 + 2 * blockchoice).astype(int)
        # TODO: Maybe need to change angle here
        curr_gripper_pos = self.sim.data.site_xpos[0, :2]
        curr_block_pos = np.array([next_obs[curr_block_xidx], next_obs[curr_block_yidx]]).T
        goal_pos = np.stack([self.goal[NUM_GOALS + 2 * i:NUM_GOALS + 2 * i + 2] for i in range(NUM_GOALS)])
        dist_to_blocks = np.linalg.norm(curr_gripper_pos - curr_block_pos, axis=1)
        block_dists = np.linalg.norm(goal_pos - curr_block_pos, axis=1)

        next_solved = (block_dists < EPS_CLOSE).astype(int)
        rwd = PLACE_RWD * np.sum(next_solved - self.solved)
        self.solved = next_solved
        i = 0
        for i in range(block_dists.shape[0]):
            if block_dists[i] > EPS_CLOSE:
                break

        rwds = np.zeros_like(self.solved).astype(float)
        j = i
        while j < len(rwds):
            rwds[j] = 2 ** (i-j)
            j += 1
        dist_to_block_rwd = dist_to_blocks * rwds
        block_dist_rwd = block_dists * rwds
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        return OFFSET_RWD - np.sum(dist_to_block_rwd) - 5 * np.sum(block_dist_rwd) - ctrl_cost + rwd

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        next_obs = self.get_current_obs()
        reward = self.reward(next_obs, action)
        done = False
        return next_obs, reward, done, {}

    def log_diagnostics(self, sd, itr):
        blockchoice = self.goal[:NUM_GOALS]
        goal_poss = [self.goal[NUM_GOALS + 2 * i:NUM_GOALS + 2 + 2 * i] for i in range(NUM_GOALS)]

        curr_block_xidx = [int(3 + 2*bc) for bc in blockchoice]
        sd_obs = sd['obs']
        if len(sd['obs'].shape) == 2:
            sd_obs = np.expand_dims(sd_obs, 0)
        initial_block_poss = [sd_obs[:, 0, cbx:cbx+2] for cbx in curr_block_xidx]
        curr_block_poss = [sd_obs[:, -1, cbx:cbx+2] for cbx in curr_block_xidx]
        initial_block_dists = [np.linalg.norm(goal_pos - initial_block_pos, axis=-1)
                               for goal_pos, initial_block_pos in zip(goal_poss, initial_block_poss)]
        block_dists = [np.linalg.norm(goal_pos - curr_block_pos, axis=-1)
                       for goal_pos, curr_block_pos in zip(goal_poss, curr_block_poss)]

        #ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(get_numpy(sd['actions'])), -1).mean()
        return OrderedDict(chain(*[
            [('Avg Initial Block %d Dist' % idx, initial_block_dist.mean()),
             ('Avg Final Block %d Dist' % idx, block_dist.mean()),
             ('Goal %d' % idx, goal_poss[idx]),
             ('Block %d' % idx, blockchoice[idx])]
            for idx, (initial_block_dist, block_dist) in enumerate(zip(initial_block_dists, block_dists))
         #   ('Ctrl Cost', ctrl_cost)
        ]))


# if __name__ == "__main__":
#     env = PusherEnvRandomized()
#     env.reset()
#     env.step(np.zeros(3))
#     env.render()
#     import pdb; pdb.set_trace()

