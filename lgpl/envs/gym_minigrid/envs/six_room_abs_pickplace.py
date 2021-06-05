# Adapted from lockedroom.py
from collections import OrderedDict

import numpy as np
from gym import spaces


from lgpl.envs.gym_minigrid.minigrid_absolute import *
from lgpl.envs.gym_minigrid.register import register
from lgpl.utils.logger_utils import get_repo_dir
from lgpl.envs.gym_minigrid.env_utils import get_rel_subgoals
import pickle

VERBS = ['start_token', 'pickup', 'goto', 'drop', 'enter', 'exit', 'carry']
DETS = list(COLOR_NAMES) + ['the']
NOUNS = ['ball', 'triangle', 'object', 'room', 'goal', 'square', 'door']
DIRS = ['west', 'east', 'north', 'south']

WORD_TO_IDX = {k:v for v, k in enumerate(VERBS + DETS + NOUNS)}
IDX_TO_WORD = {v:k for k, v in WORD_TO_IDX.items()}
VOCAB_SIZE = len(WORD_TO_IDX)
SENT_LEN = 3

class Room:
    def __init__(self,
        top,
        size,
        doorPos
    ):
        self.top = top
        self.size = size
        self.doorPos = doorPos
        self.color = None
        self.locked = False

    def rand_pos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._rand_pos(
            topX + 1, topX + sizeX - 1,
            topY + 1, topY + sizeY - 1
        )

    def rand_pos_nowall(self, env):
        # Don't have it touch walls
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._rand_pos(
            topX + 2, topX + sizeX - 2,
            topY + 2, topY + sizeY - 2
        )

class Corrections:
    def __init__(self, all_subgoals):
        self.all_subgoals = all_subgoals
        self.max_subgoals = len(all_subgoals)

    def sent_to_idx(self, sent):
        return np.array([WORD_TO_IDX[word] for word in sent.split(' ')])

    def gen_corr(self, traj_obs, traj_info):
        min_subgoals_left = traj_info['subgoals_left']
        correction = self.all_subgoals[self.max_subgoals - min_subgoals_left]

        return self.sent_to_idx(correction)

class SixRoomAbsolutePickPlaceEnv(MiniGridAbsoluteEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self, true_object, true_goal, objects=None, goals=None, door_config=None, reward='dense', seed=1,
            debug=False, random_start=True, obs_type='normal'
    ):
        """
        Creates a new environment
        :param objects: a list of tuples of types, colors, and coordinate
                tuples for decoy objects e.g. [('ball', 'green', (2,3)), ('triangle', 'red', (1,1)].
        :param goals: a list of coordinate tuples for decoy goals in order
                correspondin to OBJECTS param e.g. [((0,1), 'red'), ((2,0), 'blue')]. Must have
                same length as OBJECTS.
        :param true_object: a tuple for the object the agent must move e.g. ('ball', 'blue', (2,4))
        :param true_goal: a tuple corresponding to the goal of the true object and color ((4, 14), 'purple')
        :param reward: 'sparse' or 'dense', specifies reward function
        :param seed: for randomization
        """
        size = 19
        objects = objects or []
        goals = goals or []

        assert len(goals) <= 6, 'must have at most 6 object-goal pairs'
        assert len(objects) == len(goals), 'must have as many objects as goals'
        self.objects = objects
        self.goals = goals
        self.true_obj_tuple = (*true_object[:2], np.array(true_object[2]))
        self.true_goal_tuple = true_goal

        self.true_goal_pos = np.array(true_goal[0])
        self.reward_type = reward
        self.door_config = door_config or list(COLOR_NAMES)
        self.correction_dim = SENT_LEN
        self.random_start = random_start
        # self.hl_goal = np.concatenate([self.word_to_onhot(true_object[0]),
        #                                self.word_to_onhot(true_object[1]),
        #                                self.word_to_onhot(true_goal[-1])])
        self.hl_goal = np.array([WORD_TO_IDX[true_object[0]],
                                 WORD_TO_IDX[true_object[1]],
                                 WORD_TO_IDX[true_goal[-1]],
                                 WORD_TO_IDX['goal']
                                 ]).astype(np.int64)
        self.hl_goal_dim = self.hl_goal.shape[0]
        self.vocab_size = VOCAB_SIZE
        self.obs_type = obs_type

        self.K = 2
        self.prev_obs = np.zeros((self.K, 255))
        super().__init__(grid_size=size, max_steps=100, seed=seed)

        self.all_goals_en = np.array([WORD_TO_IDX[x] for x in ' '.join(self.all_subgoals).split(' ')])
        colors_onehot = np.eye(len(COLOR_NAMES))
        obj_onehot = np.eye(3)
        shape_to_idx = dict(ball=0, triangle=1, square=2)
        color_to_idx = {c:i for i, c in enumerate(COLOR_NAMES)}
        # self.full_info = np.concatenate([obj_onehot[shape_to_idx[true_object[0]]],
        #                          colors_onehot[color_to_idx[true_object[1]]],
        #                          colors_onehot[color_to_idx[true_goal[-1]]],
        #                          colors_onehot[color_to_idx[self.all_subgoals[0].split(' ')[1]]],
        #                          colors_onehot[color_to_idx[self.all_subgoals[3].split(' ')[1]]]
        #                          ])
        self.full_info = self.all_goals_en

        self.full_info_dim = self.full_info.size
        self.corrections = Corrections(self.all_subgoals)


    def done_cond(self, pos):
        """
        Check if environment is done given the agent's position
        """
        return False

    def finished(self, pos):
        return np.alltrue(self.true_object.cur_pos == self.true_goal_pos)

    def word_to_onhot(self, word):
        onehot = np.zeros(len(WORD_TO_IDX))
        onehot[WORD_TO_IDX[word]] = 1
        return onehot

    def sent_to_idx(self, sent):
        return np.array([WORD_TO_IDX[word] for word in sent.split(' ')])

    def idx_to_sent(self, idx):
        return ' '.join([IDX_TO_WORD[i] for i in idx])

    def _gen_obs(self):
        obs = super().gen_obs()
        #subgoals = get_subgoals(self)
        #subgoal_encoding = self.subgoal_encode(subgoals[0]) if subgoals else 0
        return np.concatenate([obs.swapaxes(0, 2).flatten(),
                               np.repeat(np.array([self.carry_flag == True]), 10)]).astype(np.float32)

    def _gen_obs_misra(self):
        obs = super().gen_obs()
        # obs is (W, H, C) so swap to (C, W, H)
        flat_obs = np.concatenate([obs.swapaxes(0, 2).flatten(),
                               np.repeat(np.array([self.carry_flag == True]), 10)]).astype(np.float32)
        self.prev_obs[1:self.K, :] = self.prev_obs[:self.K-1, :]
        self.prev_obs[0, :] = flat_obs
        return np.concatenate([self.prev_obs.flatten(), self.prev_action])

    def gen_obs(self):
        if self.obs_type == 'normal':
            return self._gen_obs()
        else:
            return self._gen_obs_misra()

    def _reward(self):
        # NOTE: assumes true object and true goal are both in a room, not necessarily same

        if self.reward_type == 'sparse':
            rwd = 1 if self.done_cond(self.agent_pos) else 0
        elif self.reward_type == 'dense':

            def subgoal_to_dist(subgoal):
                verb, color, obj = subgoal.split()
                dist_func = lambda x: np.square(x).sum()
                if verb == 'drop':
                    return dist_func(self.true_object.cur_pos - self.true_goal.cur_pos)
                elif verb == 'goto' or verb == 'enter' or verb == 'exit' or verb == 'carry':
                    if obj == 'room' or obj == 'door':
                        room = list(filter(lambda room: room.color == color, self.rooms))
                        if len(room) != 1:
                            raise RuntimeError('there must be exactly one room with color %s' % color)
                        room = room[0]
                        return dist_func(np.array(room.doorPos) - self.agent_pos)
                        # sub_done = dist == 0
                    elif obj == 'goal':
                        assert self.true_goal.color == color, \
                            'subgoal cannot include goal that is not the true goal of the environment'
                        return dist_func(self.true_goal.cur_pos - self.agent_pos)

                    else:
                        raise RuntimeError('invalid subgoal %s' % subgoal)
                elif verb == 'pickup':
                    assert self.true_obj_tuple[1] == color, \
                        'subgoal cannot include object that is not the true object of the environment'
                    return dist_func(self.true_object.cur_pos - self.agent_pos)
                else:
                    raise RuntimeError('invaliPickpd subgoal %s' % subgoal)

            subgoals = get_subgoals(self)
            self.curr_subgoals = subgoals
            rwd = 0
            if self.finished(self.agent_pos):
                rwd = 10
                self.has_finished = True
                self.min_subgoals_left = 0
            # one time rwd for completing subgoal
            elif len(subgoals) < self.min_subgoals_left:
                rwd += 100
                self.min_subgoals_left = len(subgoals)
            # negative reward for increasing subgoals
            elif len(subgoals) > self.len_subgoals:
                rwd -= 100
                rwd -= 0.01 * subgoal_to_dist(subgoals[0])
            else:
                rwd -= 0.01 * subgoal_to_dist(subgoals[0])

            self.len_subgoals = len(subgoals)

        else:
            raise ValueError('reward type must be sparse or dense')

        return rwd

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = GridAbsolute(width, height)

        # Generate the surrounding walls
        for i in range(0, width):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, height-1, Wall())
        for j in range(0, height):
            self.grid.set(0, j, Wall())
            self.grid.set(width-1, j, Wall())

        # Hallway walls
        lWallIdx = width // 2 - 2
        rWallIdx = width // 2 + 2
        for j in range(0, height):
            self.grid.set(lWallIdx, j, Wall())
            self.grid.set(rWallIdx, j, Wall())

        self.rooms = []
        self.room_cache = {}

        # Room splitting walls
        for n in range(0, 3):
            j = n * (height // 3)
            for i in range(0, lWallIdx):
                self.grid.set(i, j, Wall())
            for i in range(rWallIdx, width):
                self.grid.set(i, j, Wall())

            roomW = lWallIdx + 1
            roomH = height // 3 + 1
            self.rooms.append(Room(
                (0, j),
                (roomW, roomH),
                (lWallIdx, j + 3)
            ))
            self.rooms.append(Room(
                (rWallIdx, j),
                (roomW, roomH),
                (rWallIdx, j + 3)
            ))

        # Assign the door colors
        for color, room in zip(self.door_config, self.rooms):
            room.color = color
            if room.locked:
                self.grid.set(*room.doorPos, LockedDoor(color))
            else:
                self.grid.set(*room.doorPos, Door(color))

        true_goal_color = self.true_goal_tuple[1]
        # Place goals and objects
        for (type, color, obj_coords), (goal_coords, goal_color) in zip(self.objects, self.goals):
            assert type in ['triangle', 'square', 'ball']
            assert not self.grid.get(*obj_coords), 'objects cannot be placed where walls or doors are'
            assert not self.grid.get(*goal_coords), 'goals cannot be placed where walls or doors are'
            obj = TYPE_TO_CLASS_ABS[type](color)
            goal = Goal(goal_color)
            self.grid.set(*obj_coords, obj)
            self.grid.set(*goal_coords, goal)
            obj.cur_pos = np.array(obj_coords)
            goal.cur_pos = np.array(goal_coords)

        used_objs = [obj[:2] for obj in self.objects]

        type, color, obj_coords = self.true_obj_tuple
        assert self.true_obj_tuple[:2] not in used_objs, \
            'true goal must be distinct from other objects in type-color combination'
        assert not self.grid.get(*obj_coords), 'true object cannot be placed where other objects are'
        assert not self.grid.get(*self.true_goal_pos), 'true goal cannot be placed where other objects are'
        self.true_object = TYPE_TO_CLASS_ABS[type](color)
        self.true_goal = Goal(true_goal_color)
        self.grid.set(*obj_coords, self.true_object)
        self.grid.set(*self.true_goal_pos, self.true_goal)
        self.true_object.cur_pos = np.array(obj_coords)
        self.true_goal.cur_pos = np.array(self.true_goal_pos)

        # Randomize the player start position

        self.random_start = False
        if self.random_start:
            while True:
                pos = np.random.randint(0, self.grid_size, size=2)
                if self.grid.get(*pos) is None:
                    self.place_agent(top=(pos[0], pos[1]), size=(1, 1))
                    break
        else:
            self.place_agent(
                top=(lWallIdx+1, 12),
                size=(1, 1)
            )

    def reset(self):
        self.prev_obs[:] = 0
        obs = super().reset()
        self.all_subgoals = get_subgoals(self)
        self.min_subgoals_left = len(self.all_subgoals)
        self.len_subgoals = len(self.all_subgoals)
        self.has_finished = False
        self.curr_subgoals = self.all_subgoals

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['has_finished'] = self.has_finished
        info['subgoals_left'] = self.min_subgoals_left
        info['all_subgoals'] = self.all_subgoals
        info['curr_subgoals'] = self.curr_subgoals
        info['subgoals_completed'] = len(self.all_subgoals) - self.min_subgoals_left
        info['hl_goal'] = self.hl_goal
        info['completion'] = 1 - (self.min_subgoals_left / float(len(self.all_subgoals)))
        return obs, reward, done, info
    
    def log_diagnostics(self, sd, itr):
        # sd infos will be [path_len][batch_idx]
        # used with lgpl.algos.batch_polopt
        if len(sd['infos']) == 1:
            info = sd['infos'][0]
        else:
            info = sd['infos']
        stats = OrderedDict([('Mean_Finished', np.mean([x['has_finished'] for x in info[-1]])),
                             ('Completion', np.mean([1-(x['subgoals_left']/5.0) for x in info[-1]]))
                            ])
        return stats
        
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
                subgoals.append("drop the object")
            if agent_room != obj_room:
                if agent_room:
                    # agent is in a room that doesn't contain the goal obj. agent should go to its room's exit
                    subgoals.append("exit %s room" % agent_room.color)
                    if obj_room:
                        # obj is in a room. agent should go to that door
                        subgoals.append("enter %s room" % obj_room.color)
                else:
                    # agent is in the hallway. obj is in a room. agent should go to that room.
                    subgoals.append("enter %s room" % obj_room.color)
            # handle sending agent to object.
            subgoals.append("pickup %s %s" % (env.true_obj_tuple[1], env.true_obj_tuple[0]))

            if obj_room != goal_room:
                # after completing previous subgoals, agent will be in obj room. should go to goal room.
                if obj_room:
                    # agent should leave obj room.
                    subgoals.append("exit %s room" % obj_room.color)
                # handle sending agent to goal room
                subgoals.append("carry %s door" % goal_room.color)
        else:
            # agent is carrying the true object. agent should go to goal
            if agent_room != goal_room:
                # agent is not in goal room. should go there.
                if agent_room:
                    # agent is in a room. should go to door.
                    subgoals.append("exit %s room" % agent_room.color)
                # handle sending agent to goal room
                subgoals.append("carry %s door" % goal_room.color)
        # handle sending agent to goal
        subgoals.append("goto %s goal" % env.true_goal.color)

    return subgoals

def register_envs(envs_files, v):

    sent_env_pairs = []
    for env_file in envs_files:
        sent_env_pairs += pickle.load(open(get_repo_dir() + env_file, 'rb'))
    env_idx = 0
    env_names = []
    for sent, env_kwargs in sent_env_pairs:
        env_name = 'MiniGrid-SixRoomAbsPickPlaceEnv%d-v%d' % (env_idx, v)
        register(env_name,
                 entry_point='lgpl.envs.gym_minigrid.envs:SixRoomAbsolutePickPlaceEnv',
                 kwargs=env_kwargs)
        env_names.append(env_name)
        env_idx += 1

register_envs(['/env_data/minigrid/pickplace_envs5.pkl'], 5)


