import random
from lgpl.envs.gym_minigrid.envs.six_room_env import SixRoomEnv
from lgpl.envs.gym_minigrid.minigrid import *

# dummy env for purposes of querying room positions
env = SixRoomEnv(('triangle', 'red', (15, 3)), (3, 7)) # [('ball', 'green', (2, 2))], [(2, 3)])

def rand_room_pos(env, set_exclude=None, max_iter=1000):
    """
    Return a random position in a random room from ENV, excluding
    from SET_EXCLUDE if provided. Updates the set with returned position.
    Will try MAX_ITER times.
    """
    get_pos = lambda env: env._rand_elem(env.rooms).rand_pos_nowall(env)
    pos = get_pos(env)
    i = 0
    if set_exclude is not None:
        while pos in set_exclude:
            pos = get_pos(env)
            i += 1
            if i >= max_iter:
                raise ValueError('no possible position in env\'s rooms')
        set_exclude.add(pos)
    return pos




def rand_pos_inroom(env, room, set_exclude=None, max_iter=1000):
    """
    Return a random position in a random room from ENV, excluding
    from SET_EXCLUDE if provided. Updates the set with returned position.
    Will try MAX_ITER times.
    """
    color_to_room = {r.color: r for r in env.rooms}
    def get_pos(env):
        return color_to_room[room].rand_pos_nowall(env)
    pos = get_pos(env)
    i = 0
    if set_exclude is not None:
        while pos in set_exclude:
            pos = get_pos(env)
            i += 1
            if i >= max_iter:
                raise ValueError('no possible position in env\'s rooms')
        set_exclude.add(pos)
    return pos