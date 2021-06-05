import torch
from lgpl.utils.torch_utils import np_to_var, get_numpy, FloatTensor
import numpy as np

def rollout(policy, env, max_path_length,
            add_cat_input=None, add_inputs=None, plot=False, deterministic=False, finish_early=False):
    obs = env.reset()
    sd = dict(obs=[], rewards=[], actions=[], action_dist_lst=[], infos=[], action_probs=[])
    for s in range(max_path_length):
        x = np_to_var(np.expand_dims(obs, 0))

        if add_inputs is not None:
            action_dist = policy.forward(x, *[np_to_var(i) for i in add_inputs])
        else:
            action_dist = policy.forward(x)

        if not deterministic:
            action = action_dist.sample()
        else:
            action = action_dist.mode()
            #action = action_dist.sample(deterministic=deterministic)

        #action = np_to_var(np.ones((1, 1)))

        next_obs, reward, done, info = env.step(get_numpy(action.squeeze()))


        sd['obs'].append(obs)
        sd['rewards'].append(reward)
        sd['actions'].append(get_numpy(action))
        #sd['action_dist_lst'].append(action_dist)
        if hasattr(action_dist, 'prob'):
            sd['action_probs'].append(get_numpy(action_dist.prob))
        sd['infos'].append(info)

        obs = next_obs
        #print(s)
        if plot:
            env.render()

        if finish_early and info['has_finished']:
            break
    # if plot:
    #     env.close()

    return sd

def rollout_subgoal(policy, env, max_path_length,
            add_cat_input=None, add_inputs=None, plot=False, deterministic=False, baseline=False):
    obs = env.reset()
    sd = dict(obs=[], rewards=[], actions=[], action_dist_lst=[], infos=[])
    # changed index 1 to -1 for compatibility with baselines as well since cur_subgoal is last add_input in both cases
    cur_subgoal = add_inputs[-1]

    for s in range(max_path_length):
        x = np.expand_dims(obs, 0)
        #import pdb; pdb.set_trace()
        add_inputs[-1] = cur_subgoal
        if add_inputs is not None:
            action_dist = policy.forward(x, *[np_to_var(i) for i in add_inputs])
        else:
            action_dist = policy.forward(x)

        if deterministic:
            action = action_dist.sample()
        else:
            action = action_dist.mode()

        #action = np_to_var(np.ones((1, 1)))

        next_obs, reward, done, info = env.step(get_numpy(action.squeeze()))


        sd['obs'].append(obs)
        sd['rewards'].append(reward)
        sd['actions'].append(action.data)
        sd['action_dist_lst'].append(action_dist)
        sd['infos'].append(info)
        if not baseline:
            cur_subgoal = np.expand_dims(env.subgoal_encode(info['curr_subgoals'][0]).flatten(), 0)

        obs = next_obs
        #print(s)
        if plot:
            env.render()

        if info['has_finished']:
            break

    return sd

def rollout_correction(policy, env, max_path_length,
            add_cat_input=None, add_inputs=None, plot=False, deterministic=False, finish_early=True,
                       random_action_p=0):
    obs = env.reset()
    sd = dict(obs=[], rewards=[], actions=[], action_dist_lst=[], infos=[])

    #cur_subgoal = add_inputs[1]

    for s in range(max_path_length):
        x = np.expand_dims(obs, 0)
        #add_inputs[1] = cur_subgoal
        if add_inputs is not None:
            action_dist = policy.forward(x, *[np_to_var(i) for i in add_inputs])
        else:
            action_dist = policy.forward(x)

        if random_action_p > 0:
            action = action_dist.probs.argmax()
            if np.random.rand(1) <= random_action_p:
                action = torch.randint(0, action_dist.probs.shape[-1], (1, ))
        elif deterministic:
            action = action_dist.probs.argmax()
        else:
            #action = action_dist.mode()
            #import pdb; pdb.set_trace()
            #action = action_dist.sample(deterministic=deterministic)
            action = action_dist.sample()

        #action = np_to_var(np.ones((1, 1)))

        next_obs, reward, done, info = env.step(get_numpy(action.squeeze()))


        sd['obs'].append(obs)
        sd['rewards'].append(reward)
        sd['actions'].append(action.data)
        sd['action_dist_lst'].append(action_dist)
        sd['infos'].append(info)
        #cur_subgoal = np.expand_dims(env.subgoal_encode(info['curr_subgoals'][0]).flatten(), 0)

        obs = next_obs
        #print(s)
        if plot:
            env.render()

        if info['has_finished'] and finish_early:
            break

    return sd
