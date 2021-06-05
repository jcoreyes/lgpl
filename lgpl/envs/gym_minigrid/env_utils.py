import numpy as np
from lgpl.models.convnet import Convnet
from lgpl.models.mlp import MLP
from lgpl.policies.discrete import CategoricalMinigetNetwork
from lgpl.utils.torch_utils import gpu_enabled
import torch.nn as nn
import torch

def load_expert_pol(path=None):
    def make_policy():
        multid_obs = (5, 7, 7)
        obs_dim = np.prod(multid_obs)
        action_dim = 6

        obs_enc_dim = 256
        extra_obs_enc_dim = 32
        extra_obs_dim = 10
        obs_network = Convnet((multid_obs[0], *multid_obs[1:]), obs_enc_dim, filter_sizes=((16, 2), (16, 2)),
                              output_act=torch.nn.ReLU, hidden_sizes=(256, 256), flat_dim=400)
        extra_obs_network = MLP(extra_obs_dim, extra_obs_enc_dim, final_act=torch.nn.ReLU)

        policy = CategoricalMinigetNetwork(obs_network=obs_network,
                                           extra_obs_network=extra_obs_network,
                                           prob_network=MLP(obs_enc_dim + extra_obs_enc_dim, action_dim,
                                                            hidden_sizes=(128,),
                                                            final_act=torch.nn.Softmax),
                                           output_dim=action_dim,
                                           obs_len=obs_dim,
                                           obs_shape=multid_obs)
        return policy

    policy = make_policy()
    if path is not None:
        policy.load_state_dict(torch.load(path))
    return policy

def load_expert_pols(expert_dir, n_envs, env_version=5):
    from os.path import join
    import os
    import json
    import traceback
    exp_dirs = [join(expert_dir, x) for x in reversed(os.listdir(expert_dir))]
    env_names = []
    expert_policies = []
    env_ids = []
    for exp_dir in exp_dirs:
        env_id = json.load(open(join(exp_dir, 'variant.json')))['env_id']
        if not os.path.exists(join(exp_dir, 'snapshots')):
            print(exp_dir)
            continue
        policy_file = join(exp_dir, 'snapshots', os.listdir(join(exp_dir, 'snapshots'))[0])
        try:
            expert_pol = load_expert_pol(policy_file)
            if gpu_enabled():
                expert_pol.cuda()
            expert_policies.append(expert_pol)
        except:
            print(exp_dir)
            traceback.print_exc()
            continue
        env_name ='MiniGrid-SixRoomAbsPickPlaceEnv%d-v%d' % (env_id, env_version)
        env_names.append(env_name)
        env_ids.append(env_id)
    if n_envs < len(env_names):
        env_names = env_names[:n_envs]
        env_ids = env_ids[:n_envs]
        expert_policies = expert_policies[:n_envs]

    return env_names, expert_policies, env_ids

def get_subgoals(env):
    """
    :param env: MiniGrid SixRoomEnv
    :return: list of natural language subgoals
    (e.g. ["goto red door", "pickup green triangle", "goto red door",
            "goto yellow door", "goto blue goal"])
    """
    # assert all(hasattr(env, attr) for attr in ['agent_pos', 'true_obj_tuple', 'true_goal_pos', 'true_object',
    #                                            'true_goal']), \
    #     'env must be SixRoomEnv or SixRoomAbsoluteEnv'

    subgoals = []

    agent_room = env.get_room(env.agent_pos)
    obj_room = env.get_room(env.true_object.cur_pos)
    goal_room = env.get_room(env.true_goal_pos)

    if not env.done_cond(env.agent_pos):
        if env.carrying != env.true_object:
            # agent isn't carrying the true object. agent should go to it
            if env.carrying:
                # agent is holding a non true object. agent should drop it
                subgoals.append("drop %s object" % env.carrying.color)
            if agent_room != obj_room:
                if agent_room:
                    # agent is in a room that doesn't contain the goal obj. agent should go to its room's exit
                    subgoals.append("goto %s door" % agent_room.color)
                    if obj_room:
                        # obj is in a room. agent should go to that door
                        subgoals.append("goto %s door" % obj_room.color)
                else:
                    # agent is in the hallway. obj is in a room. agent should go to that room.
                    subgoals.append("goto %s door" % obj_room.color)
            # handle sending agent to object.
            subgoals.append("pickup %s %s" % (env.true_obj_tuple[1], env.true_obj_tuple[0]))

            if obj_room != goal_room:
                # after completing previous subgoals, agent will be in obj room. should go to goal room.
                if obj_room:
                    # agent should leave obj room.
                    subgoals.append("goto %s door" % obj_room.color)
                # handle sending agent to goal room
                subgoals.append("goto %s door" % goal_room.color)
        else:
            # agent is carrying the true object. agent should go to goal
            if agent_room != goal_room:
                # agent is not in goal room. should go there.
                if agent_room:
                    # agent is in a room. should go to door.
                    subgoals.append("goto %s door" % agent_room.color)
                # handle sending agent to goal room
                subgoals.append("goto %s door" % goal_room.color)
        # handle sending agent to goal
        subgoals.append("goto %s goal" % env.true_goal.color)

    return subgoals


def get_dir(from_, to_, env):
    assert from_ != to_, 'no direction within hallway or room'
    if from_ is None:
        # going from hallway to room
        dir = 'east' if to_.doorPos[0] > env.grid_size // 2 else 'west'
    elif to_ is None:
        # going from room to hallway
        # prioritize east-west over north-south since that's how doors to exit are aligned
        dir = 'east' if from_.doorPos[0] < env.grid_size // 2 else 'west'
    else:
        # going from room to room
        if from_.doorPos[0] != to_.doorPos[0]:
            dir = 'east' if from_.doorPos[0] < to_.doorPos[0] else 'west'
        else:
            # note position index increases southward
            dir = 'north' if from_.doorPos[1] > to_.doorPos[1] else 'south'
    return dir


def get_rel_subgoals(env):
    """
    :param env: MiniGrid SixRoomEnv
    :return: list of natural language subgoals, using relative specification of room
    (e.g. ["goto north room", "pickup green triangle", "goto east room", "goto blue goal"])
    """
    assert all(hasattr(env, attr) for attr in ['agent_pos', 'true_obj_tuple', 'true_goal_pos', 'true_object',
                                               'true_goal']), \
        'env must be in SixRoomEnv or SixRoomAbsoluteEnv family'

    subgoals = []

    agent_room = env.get_room(env.agent_pos)
    obj_room = env.get_room(env.true_object.cur_pos)
    goal_room = env.get_room(env.true_goal_pos)
    # because subgoals need to be relative to previous subgoals, track cur_room in planned path
    cur_room = agent_room

    if not env.done_cond(env.agent_pos):
        if env.carrying != env.true_object:
            # agent isn't carrying the true object. agent should go to it
            if env.carrying:
                # agent is holding a non true object. agent should drop it
                subgoals.append("drop %s object" % env.carrying.color)
            if agent_room != obj_room:
                # agent should go to obj room
                subgoals.append("goto %s room" % get_dir(agent_room, obj_room, env))
            # handle sending agent to object.
            subgoals.append("pickup %s %s" % (env.true_obj_tuple[1], env.true_obj_tuple[0]))

            if obj_room != goal_room:
                # agent should go to goal room
                subgoals.append("goto %s room" % get_dir(obj_room, goal_room, env))
        else:
            # agent is carrying the true object. agent should go to goal
            if agent_room != goal_room:
                # agent should go to goal room
                subgoals.append("goto %s room" % get_dir(agent_room, goal_room, env))
        # handle sending agent to goal
        subgoals.append("goto %s goal" % env.true_goal.color)

    return subgoals