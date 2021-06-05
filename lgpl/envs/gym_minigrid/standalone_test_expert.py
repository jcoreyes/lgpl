#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy
import gym
import time
from optparse import OptionParser
import torch
from os.path import join
import os

import lgpl.envs.gym_minigrid
from lgpl.utils.logger_utils import get_repo_dir
from lgpl.utils.torch_utils import np_to_var, Variable, get_numpy


def main(env=None):
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGridAbs-MultiRoom-N6-v0'
    )
    (options, args) = parser.parse_args()

    expert_dir = join(get_repo_dir(), 'data/exps/pickplace3/pickplace3/')
    exp_dirs = [join(expert_dir, x) for x in reversed(os.listdir(expert_dir))][:10]
    env_names = []
    expert_policies = []
    for exp_dir in exp_dirs:
        policy_file = os.listdir(join(exp_dir, 'ppo'))[0]
        env_names.append(policy_file.split('.')[0])
        expert_pol = torch.load(join(exp_dir, 'ppo', policy_file))[0]
        expert_pol.cuda()
        expert_policies.append(expert_pol)
    expert_pol = expert_policies[0]
    # Load the gym environment

    env = gym.make(env_names[0])

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()
    global env_return
    env_return = 0
    # Create a window to render into
    renderer = env.render('human')
    def keyDownCb(keyName):
        global env_return
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0
        if 'Abs' in options.env_name:
            if keyName == 'LEFT':
                action = env.actions.west
            elif keyName == 'RIGHT':
                action = env.actions.east
            elif keyName == 'UP':
                action = env.actions.north
            elif keyName == 'DOWN':
                action = env.actions.south
            elif keyName == 'PAGE_UP':
                action = env.actions.pickup
            elif keyName == 'PAGE_DOWN':
                action = env.actions.drop
            else:
                print("unknown key %s" % keyName)
                return
        else:
            if keyName == 'LEFT':
                action = env.actions.left
            elif keyName == 'RIGHT':
                action = env.actions.right
            elif keyName == 'UP':
                action = env.actions.forward
            elif keyName == 'SPACE':
                action = env.actions.toggle
            elif keyName == 'PAGE_UP':
                action = env.actions.pickup
            elif keyName == 'PAGE_DOWN':
                action = env.actions.drop
            else:
                print("unknown key %s" % keyName)
                return
        # elif keyName == 'CTRL':
        #     action = env.actions.done



        obs, reward, done, info = env.step(action)
        states = torch.zeros((1, expert_pol.state_size))
        masks = torch.ones((1, 1))
        with torch.no_grad():
            value, action, action_log_prob, states = expert_pol.act(
                np_to_var(obs).unsqueeze(0),
                Variable(states),
                Variable(masks),
                deterministic=True
            )

        print(list(env.actions)[int(get_numpy(action)[0])])
        print(info['curr_subgoals'])
        env_return += reward
        print('step=%s, reward=%.3f, return=%.3f' % (env.step_count, reward, env_return))

        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
