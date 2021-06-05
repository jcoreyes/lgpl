import nltk
from gym.envs import register
from nltk.parse.generate import generate
from lgpl.language.sentence_to_env import *
import pickle

class Grammar:
    def __init__(self):
        COLORS = ['"green"', '"red"', '"blue"', '"purple"', '"yellow"', '"grey"']
        OBJS = ['"triangle"', '"square"', '"ball"']
        DIRS = ['"left"', '"right"', '"above"', '"below"']
        VERBS = ['"pickup"', '"drop"']
        PREPS = ['"in"']
        self.grammar = nltk.CFG.fromstring("""
        S -> V NP | V NP PP
        NP -> Det N
        PP -> P LNP
        LNP -> Det Loc
        LNP -> RelDet Loc
        Loc -> "room"
        P ->""" + '| '.join(PREPS) + """
        V ->""" + '| '.join(VERBS) + """
        Det ->""" + '| '.join(COLORS) + """
        RelDet ->""" + '| '.join(DIRS) + """
        N ->""" + '| '.join(OBJS) + """
        """)
        print(self.grammar)
        self.sentences = [' '.join(s) for s in generate(self.grammar, n=1000)]

        print(self.sentences)
        print('%d different sentences possible.' % len(self.sentences))

class PickupGrammar:
    def __init__(self):
        COLORS = ['"green"', '"red"', '"blue"', '"purple"', '"yellow"', '"grey"']
        OBJS = ['"triangle"', '"square"', '"ball"']
        VERBS = ['"pickup"']
        self.grammar = nltk.CFG.fromstring("""
        S -> V NP
        NP -> Det N
        V ->""" + '| '.join(VERBS) + """
        Det ->""" + '| '.join(COLORS) + """
        N ->""" + '| '.join(OBJS) + """
        """)
        print(self.grammar)
        self.sentences = [' '.join(s) for s in generate(self.grammar, n=1000)]

        print(self.sentences)
        print('%d different sentences possible.' % len(self.sentences))

    def sent_to_env(self, sentence, rtype='sparse', env_seed=1):
        verb, true_color, true_type = sentence.split()
        objs = []
        goals = []
        positions = set()
        allowed_types = ['square', 'ball', 'triangle']
        while len(objs) < 3:
            type = random.choice(allowed_types)
            color = random.choice(list(COLOR_TO_IDX.keys()))
            if true_type == type and true_color == color:
                continue
            objs.append((type, color, rand_room_pos(env, positions)))
            goals.append(rand_room_pos(env, positions))

        true_obj = (true_type, true_color, rand_room_pos(env, positions))
        true_goal = rand_room_pos(env, positions)

        kwargs = {'true_object': true_obj,
                  'true_goal': true_goal,
                  'objects': objs,
                  'goals': goals}
        return kwargs

class PickPlaceGrammar:
    def __init__(self):
        COLORS = ['"green"', '"red"', '"blue"', '"purple"', '"yellow"', '"grey"']
        OBJS = ['"triangle"', '"square"', '"ball"']
        VERBS = ['"place"']
        self.grammar = nltk.CFG.fromstring("""
        S -> V NP
        NP -> Det N Det
        V ->""" + '| '.join(VERBS) + """
        Det ->""" + '| '.join(COLORS) + """
        N ->""" + '| '.join(OBJS) + """
        """)
        print(self.grammar)
        self.sentences = [' '.join(s) for s in generate(self.grammar, n=1000)]

        print(self.sentences)
        print('%d different sentences possible.' % len(self.sentences))

    def sent_to_env(self, sentence, rtype='sparse', env_seed=1):
        verb, true_color, true_type, true_goal_color = sentence.split()
        objs = []
        goals = []
        positions = set()
        allowed_types = ['square', 'ball', 'triangle']
        while len(objs) < 3:
            type = random.choice(allowed_types)
            color = random.choice(list(COLOR_TO_IDX.keys()))
            goal_color = random.choice(list(COLOR_TO_IDX.keys()))
            if (true_type == type and true_color == color) or goal_color == true_goal_color:
                continue
            objs.append((type, color, rand_room_pos(env, positions)))
            goals.append((rand_room_pos(env, positions), goal_color))

        true_obj = (true_type, true_color, rand_room_pos(env, positions))
        true_goal = (rand_room_pos(env, positions), true_goal_color)

        kwargs = {'true_object': true_obj,
                  'true_goal': true_goal,
                  'objects': objs,
                  'goals': goals}
        return kwargs

class PickPlaceGrammar2:
    def __init__(self):
        COLORS = ['"green"', '"red"', '"blue"', '"purple"', '"yellow"', '"grey"']
        OBJS = ['"triangle"', '"square"', '"ball"']
        VERBS = ['"place"']

        # verb
        # object-phrase (green triangle)
        # in-phrase (green room)
        # goal phrase (yellow goal) location-phrase (red room)
        # place green triangle green room yellow goal red room
        self.grammar = nltk.CFG.fromstring("""
        S -> V OP IP GP LP
        IP -> Det R
        OP -> Det N
        GP -> Det G
        LP -> Det R
        V ->""" + '| '.join(VERBS) + """
        Det ->""" + '| '.join(COLORS) + """
        N ->""" + '| '.join(OBJS) + """
        G -> 'goal'
        R -> 'room'
        """)
        print(self.grammar)
        
        self.sentences = []
        # filter out envs where obj room and goal room are same
        for s in generate(self.grammar, n=int(1e6)):
            if s[3] == s[-2]:
                continue
            else:
                self.sentences.append(s)
        print(len(self.sentences))

    def sent_to_env(self, sentence, rtype='sparse', env_seed=1):
        #verb, true_color, true_type, true_goal_color = sentence.split()
        true_color, true_type, true_obj_room = sentence[1:4]
        true_goal_color = sentence[5]
        true_goal_room = sentence[-2]

        objs = []
        goals = []
        positions = set()
        allowed_types = ['square', 'ball', 'triangle']
        while len(objs) < 3:
            type = random.choice(allowed_types)
            color = random.choice(list(COLOR_TO_IDX.keys()))
            goal_color = random.choice(list(COLOR_TO_IDX.keys()))
            if (true_type == type and true_color == color) or goal_color == true_goal_color:
                continue
            objs.append((type, color, rand_room_pos(env, positions)))
            goals.append((rand_room_pos(env, positions), goal_color))

        true_obj = (true_type, true_color, rand_pos_inroom(env, true_obj_room, positions))
        true_goal = (rand_pos_inroom(env, true_goal_room, positions), true_goal_color)

        kwargs = {'true_object': true_obj,
                  'true_goal': true_goal,
                  'objects': objs,
                  'goals': goals}
        return kwargs


class PickPlaceGrammar3:
    # Randomize color rooms
    def __init__(self):
        COLORS = ['"green"', '"red"', '"blue"', '"purple"', '"yellow"', '"grey"']
        OBJS = ['"triangle"', '"square"', '"ball"']
        VERBS = ['"place"']

        # verb
        # object-phrase (green triangle)
        # in-phrase (green room)
        # goal phrase (yellow goal) location-phrase (red room)
        # place green triangle green room yellow goal red room
        self.grammar = nltk.CFG.fromstring("""
        S -> V OP IP GP LP
        IP -> Det R
        OP -> Det N
        GP -> Det G
        LP -> Det R
        V ->""" + '| '.join(VERBS) + """
        Det ->""" + '| '.join(COLORS) + """
        N ->""" + '| '.join(OBJS) + """
        G -> 'goal'
        R -> 'room'
        """)
        print(self.grammar)

        self.sentences = []
        # filter out envs where obj room and goal room are same
        for s in generate(self.grammar, n=int(1e6)):
            if s[3] == s[-2]:
                continue
            else:
                self.sentences.append(s)
        print(len(self.sentences))
        # self.sentences = [' '.join(s) for s in generate(self.grammar, n=int(1e6))]
        # filter out where IP and LP are same
        # self.sentences = [s for in sentences if ]
        # print(self.sentences)
        # print('%d different sentences possible.' % len(self.sentences))

    def sent_to_env(self, sentence, rtype='sparse', env_seed=1):
        # verb, true_color, true_type, true_goal_color = sentence.split()
        true_color, true_type, true_obj_room = sentence[1:4]
        true_goal_color = sentence[5]
        true_goal_room = sentence[-2]

        objs = []
        goals = []
        positions = set()
        allowed_types = ['square', 'ball', 'triangle']
        while len(objs) < 3:
            type = random.choice(allowed_types)
            color = random.choice(list(COLOR_TO_IDX.keys()))
            goal_color = random.choice(list(COLOR_TO_IDX.keys()))
            if (true_type == type and true_color == color) or goal_color == true_goal_color:
                continue
            objs.append((type, color, rand_room_pos(env, positions)))
            goals.append((rand_room_pos(env, positions), goal_color))

        true_obj = (true_type, true_color, rand_pos_inroom(env, true_obj_room, positions))
        true_goal = (rand_pos_inroom(env, true_goal_room, positions), true_goal_color)

        door_config = np.arange(len(COLORS))
        np.random.shuffle(door_config)
        door_config = [list(COLOR_TO_IDX.keys())[x] for x in door_config]

        kwargs = {'true_object': true_obj,
                  'true_goal': true_goal,
                  'objects': objs,
                  'door_config': door_config,
                  'goals': goals}
        return kwargs

def envs_from_grammar(grammar, save_file=None, render=False):

    sent_env_pairs = []
    env_idx = 0
    for sentence in grammar.sentences:

        env_kwargs = grammar.sent_to_env(sentence)
        sent_env_pairs.append((sentence, env_kwargs))
        if render and env_idx == 10:
            env_name = 'MiniGrid-SixRoomAbsPickPlaceEnv%d-v6' % env_idx
            register(env_name,
                     entry_point='lgpl.envs.gym_minigrid.envs:SixRoomAbsolutePickPlaceEnv',
                     kwargs=env_kwargs)
            env = gym.make(env_name)
            env.reset()
            #print(sentence)
            env.render()
            import pdb; pdb.set_trace()
        env_idx += 1

    if save_file is not None:
        pickle.dump(sent_env_pairs, open(save_file, 'wb'))

if __name__ == '__main__':
    save_file = 'env_data/minigrid/pickplace_envs2.pkl'
    grammar = PickPlaceGrammar2()
    envs_from_grammar(grammar, save_file=save_file, render=False)