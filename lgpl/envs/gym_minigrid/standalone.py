#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy
import gym
import time
from optparse import OptionParser
from lgpl.envs.gym_minigrid.envs.six_room_abs_pickplace_fullobs_relcorr import *

import lgpl.envs.gym_minigrid

def main(env=None):
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    if env is None:
        env = gym.make(options.env_name)

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
