import numpy as np
from itertools import cycle
from toolz import curry
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

def bandit_to_color(i):
    c = cycle('rbgmcy')
    for _ in range(i):
        next(c)
    return next(c)      
    
def exemplos_beta():
    alphas = [0, 2, 2, 16, 32, 130]
    betas = [0, 0, 3, 9, 50, 80]
    cores = ['r','b', 'g', 'c', 'm', 'y']
    
    fig = plt.figure(tight_layout=True, figsize=(9,4))
    gs = gridspec.GridSpec(2, 3)
    
    for i in range(6):
        ax = fig.add_subplot(gs[int(i/3),i%3])
        x = np.linspace(0,1)
        plt.xlim(-0.1, 1.1)
        plt.plot(x, beta.pdf(x, 1+alphas[i], 1+betas[i]), lw=3, alpha=1, c = cores[i])
        
        if i!=0:
            ax.plot([alphas[i]/(alphas[i]+betas[i]), alphas[i]/(alphas[i]+betas[i])],[0,beta.pdf(alphas[i]/(alphas[i]+betas[i]), 1+alphas[i], 1+betas[i])], c=cores[i],alpha = 0.7)
        
        plt.title('Sucessos: '+str(alphas[i])+' | Falhas: '+str(betas[i]))
        

def evaluate_policys(policy_list, bandit_list, labels, time, regret=True):
    reward_curve_values, regret_curve_values = [], []
    for policy in tqdm(policy_list):
        choices, rewards = policy.play(bandit_list, time)
        reward_curve_values.append(reward_curve(rewards))
        if regret:
            regret_curve_values.append(regret_curve(choices, bandit_list))
    
    if regret:
        fig, ax = plt.subplots(ncols=2, figsize=(6,3))
    else:
        fig, ax = plt.subplots(figsize=(6,3))
        ax = [ax]

    for reward_curve_value, label in zip(reward_curve_values, labels):
        ax[0].plot(range(len(reward_curve_value)),reward_curve_value, label=label)
    ax[0].set_xlabel('Rodada')
    ax[0].set_ylabel('Recompensa até a rodada')
    
    if regret:
        for regret_curve_value, label in zip(regret_curve_values, labels):
            ax[1].plot(range(len(regret_curve_value)),regret_curve_value, label=label)
        ax[1].set_xlabel('Rodada')
        ax[1].set_ylabel('Regret até a rodada')

    plt.legend()
    plt.show()

from itertools import accumulate

def regret_curve(choices, bandit_list):
    best_bandit_theta = np.max([bandido.theta for bandido in bandit_list])
    return list(accumulate(best_bandit_theta - bandit_list[choice].theta for choice in choices))

def reward_curve(rewards):
    return list(accumulate(rewards))