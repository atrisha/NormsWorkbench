'''
Created on 10 Mar 2023

@author: Atrisha
'''

from normative_information_design.normative_information_design_single_context.norm_recommendation_information_design import parallel_env
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils
import itertools
import mdptoolbox, mdptoolbox.example


def generate_transition_matrix():
    reward_matrix =  np.zeros(shape=(9,9))
    transition_map = dict()
    for action in np.arange(0.1,1,.1):
        print(action)
        if action not in transition_map:
            transition_map[action] = dict()
        actlist_t, actlist_r = [],[]
        for run_iter in np.arange(100):
            
            env = parallel_env(render_mode='human',attr_dict={'true_state':{'n1':0.55},'stewarding_flag':False})
            ''' Check that every norm context has at least one agent '''
            if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
                raise Exception()
            
            number_of_iterations = 50000
            env.NUM_ITERS = number_of_iterations
            for state in np.arange(0.1,1,.1): 
                if state==0.6 and action==0.4:
                    f=1
                env.reset()
                env.common_prior = utils.est_beta_from_mu_sigma(state, 0.2)
                env.prior_baseline = env.common_prior
                env.normal_constr_w = 0.3
                env.common_proportion_prior = env.common_prior
                env.prior_baseline = env.common_prior
                env.prior_prop_baseline = env.common_prior
            
                if abs(action-env.common_prior_mean) <= env.normal_constr_w:
                    valid_distr = True
                else:
                    valid_distr = False
                posterior_mean = env.generate_posteriors(action)
                if valid_distr:
                    population_actions = {agent.id:agent.act(env,run_type='self-ref',baseline=False) for agent in env.possible_agents}
                    observations, reward, terminations, truncations, infos = env.step(population_actions,0,baseline=False)
                else:
                    observations, reward, terminations, truncations, infos = env.common_prior, -1, {agent.id:False for agent in env.possible_agents}, {agent.id:False for agent in env.possible_agents}, {agent.id:{} for agent in env.possible_agents}
                #print(round(action,1),round(state,1),round(observations[0]/sum(observations),1))
                #print(round(action,1),round(state,1),round(reward,1))
                next_state = round(observations[0]/sum(observations),1)
                a_idx, s_idx, s_prime_idx = int(round(action,1)*10)-1, int(round(state,1)*10)-1, int(round(next_state,1)*10)-1
                if (s_idx,s_prime_idx) not in transition_map[action]:
                    transition_map[action][(s_idx,s_prime_idx)] = 1
                else:
                    transition_map[action][(s_idx,s_prime_idx)] += 1
                reward_matrix[s_idx,a_idx] += round(reward,1)
                #print('----')
    reward_matrix = reward_matrix/100    
    transition_matrix = np.zeros(shape=(9,9,9))
    for act,s_s_data in transition_map.items():
        a_idx = int(round(act,1)*10)-1
        for s_s_prime,ct in s_s_data.items():
            transition_matrix[a_idx,s_s_prime[0],s_s_prime[1]] = ct
    for a_idx in np.arange(transition_matrix.shape[0]):
        s_data = transition_matrix[a_idx]
        s_data_sum = np.sum(s_data,axis=1)
        s_data_prob = s_data / s_data_sum[:,None]
        transition_matrix[a_idx] = s_data_prob
    return transition_matrix, reward_matrix

P, R = generate_transition_matrix()

fh = mdptoolbox.mdp.FiniteHorizon(P, R, 0.5, 100)
#fh = mdptoolbox.mdp.QLearning(P, R, 0.9)
fh.run()
#print([np.round((x+1)/10,1) for x in list(fh.policy)])
print(fh.policy[:99])