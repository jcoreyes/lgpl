import numpy as np
from os.path import join
from lgpl.utils.logger_utils import get_repo_dir
import pickle
import os
import json
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def plot_rl(env_name='minigrid', expert_name='exps/pickplace9/pickplace9', env_prefix='Env', env_ver='-v5',
            json_id='env_id'):
    env_names = pickle.load(open(join(get_repo_dir(), 'env_data/%s/test.pkl' %env_name), 'rb'))
    env_ids = set([int(x.split(env_prefix)[-1].split(env_ver)[0]) for x in env_names])
    expert_dir = join(get_repo_dir(), 'data/%s' % expert_name)
    exp_dirs = [join(expert_dir, x) for x in reversed(os.listdir(expert_dir))]

    completion = np.zeros(200)
    n_exps = 0
    for exp_dir in exp_dirs:
        env_id = json.load(open(join(exp_dir, 'variant.json')))[json_id]
        if env_id not in env_ids:
            continue
        data = np.genfromtxt(join(exp_dir, 'progress.csv'), dtype=float, delimiter=',', names=True)
        c_rate = data['Completion'][:200]
        if c_rate.size < 200:
            c_rate = np.concatenate([c_rate, np.ones(200 - c_rate.size)])

        completion += c_rate
        n_exps += 1
    completion /= n_exps

    return completion

def plot_rl_pusher(env_name='minigrid', expert_name='exps/pickplace9/pickplace9', env_prefix='Env', env_ver='-v5',
            json_id='env_id'):
    env_names = pickle.load(open(join(get_repo_dir(), 'env_data/%s/test.pkl' %env_name), 'rb'))
    env_ids = set([int(x.split(env_prefix)[-1].split(env_ver)[0]) for x in env_names])
    expert_dir = join(get_repo_dir(), 'data/%s' % expert_name)
    exp_dirs = [join(expert_dir, x) for x in reversed(os.listdir(expert_dir))]


    completion = np.zeros(300)

    n_exps = 0
    for exp_dir in exp_dirs:
        if not os.path.exists(join(exp_dir, 'variant.json')):
            continue
        env_id = json.load(open(join(exp_dir, 'variant.json')))[json_id]

        if env_id not in env_ids:
            continue
        try:
            data = np.genfromtxt(join(exp_dir, 'progress.csv'), dtype=float, delimiter=',', names=True)
        except:
            continue

        if 'finetune' in expert_name:
            c_rate = data['Completion']
        else:
            c_rate = 1 - data['Avg_Final_Block_Dist'] / data['Avg_Initial_Block_Dist']

        c_rate[c_rate<0] = 0
        c_rate = c_rate[:300]
        if c_rate.size < 300:
            c_rate = np.concatenate([c_rate, np.ones(300 - c_rate.size)])

        completion += c_rate
        n_exps += 1
    completion /= n_exps

    return completion


def plot_lgpl(dir_name):
    expert_dir = join(get_repo_dir(), 'data/exps/%s' % dir_name)
    exp_dirs = [join(expert_dir, x) for x in os.listdir(expert_dir)]

    c_rates = None
    for exp_dir in exp_dirs:
        data = np.genfromtxt(join(exp_dir, 'sl.csv'), dtype=float, delimiter=',', names=True)
        data = data[::5]
        #for i in range(5):
        if c_rates is None:
            c_rates = np.zeros(data.size)
        c_rates += data['test_Completion_CI_%d' %5]
    c_rates /= len(exp_dirs)
    #plt.plot(np.arange(5), c_rates)
    #plt.show()
    return c_rates

def plot_lgpl2(dir_name):
    expert_dir = join(get_repo_dir(), 'data/exps/%s' % dir_name)
    exp_dirs = [join(expert_dir, x) for x in os.listdir(expert_dir)]

    c_rates = np.zeros(6)
    for exp_dir in exp_dirs:
        data = np.genfromtxt(join(exp_dir, 'sl.csv'), dtype=float, delimiter=',', names=True)

        for i in range(6):
            c_rates[i] += data['test_Completion_CI_%d' %i].max
    c_rates /= len(exp_dirs)
    #plt.plot(np.arange(5), c_rates)
    #plt.show()
    return c_rates


def plot_minigrid():
    rl = plot_rl()
    # rl_finetune = plot_rl(expert_name='exps/minigrid/finetune_rl_exp1/finetune-rl-exp1')
    rl_finetune = plot_rl(expert_name='exps/minigrid/minigrid_final/rl_finetune_instruction_exp13/rl-finetune-instruction-exp13')
    lgpl = plot_lgpl('minigrid/minigrid_final/pickplace9exp5/pickplace9exp5')
    lgpl_reward = plot_lgpl('minigrid/lgpl_reward_completion_exp1/lgpl-reward-completion-exp1')


    rl = rl[:150]
    rl_finetune = rl_finetune[:150]
    plt.figure(1)

    plt.plot(np.arange(lgpl.size)*5, lgpl, label='GPL')
    max_x = rl.size * 20
    plt.plot(np.arange(lgpl.size*5, max_x), np.repeat(np.array([lgpl.max()]), max_x-lgpl.size*5), linestyle='dashed',
             color='tab:blue')


    plt.plot(np.arange(lgpl_reward.size)*5, lgpl_reward, label='GPR')
    plt.plot(np.arange(lgpl_reward.size*5, max_x), np.repeat(np.array([lgpl_reward.max()]), max_x-lgpl_reward.size*5), linestyle='dashed',
             color='tab:green')


    #plt.plot(np.arange(rl.size)*20, rl, label='RL')
    #plt.plot(np.arange(rl_finetune.size)*20, rl_finetune, label='Pretraining with RL Finetuning')
    plt.legend(loc='center right', prop={'size': 18})
    plt.xlabel('Meta-test complexity (trajectories per task)', fontsize=16)
    plt.ylabel('Completion Rate', fontsize=16)
    plt.title('Multi-Room Object Manipulation', fontsize=16)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('minigrid_curve5.png')
    #plt.show()

def plot_pusher():
    rl = plot_rl_pusher(env_name='pusher', expert_name='s3/pusher/pusher3v4-expert1', env_ver='-v4',
               env_prefix='Env2v', json_id='pusher_id')
    # rl_finetune = plot_rl_pusher(env_name='pusher', expert_name='exps/pusher/finetune_rl_exp5/finetune-rl-exp5', env_ver='-v4',
    #              env_prefix='Env2v', json_id='pusher_id')
    rl_finetune = plot_rl_pusher(env_name='pusher', expert_name='exps/pusher/rl_finetune_exp1/rl-finetune-exp1', env_ver='-v4',
                 env_prefix='Env2v', json_id='pusher_id')
    lgpl = plot_lgpl('pusher/pusher_final/pusher_lglp_exp1/pusher-lglp-exp1')
    lgpl_reward = plot_lgpl('pusher/pusher_lglp_reward_exp1/pusher-lglp-reward-exp1')


    rl = rl[:150]
    rl_finetune = rl_finetune[:150]
    plt.figure(0)

    plt.plot(np.arange(lgpl.size)*5, lgpl, label='GPL')
    max_x = rl.size * 20
    plt.plot(np.arange(lgpl.size*5, max_x), np.repeat(np.array([lgpl.max()]), max_x-lgpl.size*5), linestyle='dashed',
             color='tab:blue')


    plt.plot(np.arange(lgpl_reward.size)*5, lgpl_reward, label='GPR')
    plt.plot(np.arange(lgpl_reward.size*5, max_x), np.repeat(np.array([lgpl_reward.max()]), max_x-lgpl_reward.size*5), linestyle='dashed',
             color='tab:green')

    
    #plt.plot(np.arange(rl.size)*20, rl, label='RL')
    #plt.plot(np.arange(rl_finetune.size)*20, rl_finetune, label='Pretraining with RL Finetuning')
    plt.legend(loc='center right', prop={'size': 16})
    plt.xlabel('Meta-test complexity (trajectories per task)', fontsize=16)
    plt.ylabel('Completion Rate', fontsize=18)
    plt.title('Robotic Object Relocation', fontsize=16)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('pusher_curve5.png')
   # plt.show()

if __name__ == '__main__':
    plot_pusher()
    #plt.rcParams.update({'font.size': 60})
    plot_minigrid()
