
from lgpl.models.mlp import MLP
import torch
from lgpl.policies.discrete import CategoricalNetwork
from lgpl.utils.torch_utils import gpu_enabled

def load_expert_pol(path=None):
    def make_policy():
        obs_dim = 28
        # action_dim = 3
        action_dim = 4
        mlp_policy_input_dim = obs_dim
        # Make policy
        # policy = GaussianMLPPolicy(mean_network=MLP(mlp_policy_input_dim, action_dim, hidden_sizes=(32, 32)),
        #                            log_var_network=Parameter(mlp_policy_input_dim, action_dim, init=np.log(1.0)),
        #                            )
        policy = CategoricalNetwork(prob_network=MLP(obs_dim, action_dim, final_act=torch.nn.Softmax), output_dim=action_dim)

        return policy

    policy = make_policy()
    if path is not None:
        policy.load_state_dict(torch.load(path))
    return policy

def load_expert_pols(expert_dir, n_envs):
    from os.path import join
    import os
    import json
    import traceback

    exp_dirs = [join(expert_dir, x) for x in reversed(os.listdir(expert_dir))]
    exp_dirs.sort()
    env_names = []
    expert_policies = []

    for exp_dir in exp_dirs:
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
        env_id = json.load(open(join(exp_dir, 'variant.json')))['pusher_id']
        env_name ='PusherEnv2v%d-v4' % env_id
        env_names.append(env_name)
    if n_envs < len(env_names):
        env_names = env_names[:n_envs]
        expert_policies = expert_policies[:n_envs]
    return env_names, expert_policies
