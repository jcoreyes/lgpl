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
import sys
import gym
from lgpl.utils.getch import getKey
from lgpl.envs.pusher.pusher_env2 import Corrections2

if __name__ == "__main__":
    import lgpl.envs.pusher.pusher_env3
    env_id = np.random.choice(range(1000))
    env_name = 'PusherEnv2v%d-v4' % env_id

    env = gym.make(env_name)
    #env.corrections = Corrections2(env.choice, env.goal, env.goal_pos, env.initial_goal_block_pos)
    env.reset()
    env_return = 0
    while True:
        key = getKey()

        if key == 'w':
            a = 0
        elif key == 'a':
            a = 2
        elif key  == 's':
            a = 1
        elif key  == 'd':
            a = 3
        elif key  == 'q':
            break
        # elif keyName == 'CTRL':
        #     action = env.actions.done

        obs, reward, done, info = env.step(a)
        env_return += reward
        print('reward=%.3f, return=%.3f' % (reward, env_return))

        correction = env.corrections.gen_corr(np.reshape(obs, (1, -1)))
        correction_sent = env.corrections.idx_to_sent(correction)
        hlg_sent = env.hlg
        print(hlg_sent, correction_sent)
        env.render()




