'''
Created on 10 Mar 2023

@author: Atrisha
'''

from normative_information_design.all_networks import *
from normative_information_design.normative_information_design_single_context.norm_recommendation_information_design import StewardAgent, parallel_env
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils
import itertools
import mdptoolbox, mdptoolbox.example


def generate_transition_matrix():
    reward_matrix =  np.zeros(shape=(90,90))
    transition_map = dict()
    for action in np.arange(0.1,1,.01):
        print(action)
        if action not in transition_map:
            transition_map[action] = dict()
        actlist_t, actlist_r = [],[]
        for run_iter in np.arange(10):
            
            env = parallel_env(render_mode='human',attr_dict={'true_state':{'n1':0.6},'stewarding_flag':True,'target_op':0.3})
            ''' Check that every norm context has at least one agent '''
            if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
                raise Exception()
            
            number_of_iterations = 50000
            env.NUM_ITERS = number_of_iterations
            for state in np.arange(0.1,1,.01): 
                if state==0.9 and action==0.6:
                    f=1
                env.reset()
                env.common_prior = utils.est_beta_from_mu_sigma(state, 0.2)
                #print(state,env.common_prior[0]/np.sum(env.common_prior))
                env.prior_baseline = env.common_prior
                env.normal_constr_w = 0.3
            
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
                next_state = round(observations[0]/sum(observations),2)
                a_idx, s_idx, s_prime_idx = utils.approx_index(np.arange(0.1,1,.01),action,0.01), utils.approx_index(np.arange(0.1,1,.01),state,0.01), utils.approx_index(np.arange(0.1,1,.01),next_state,0.01)
                if (s_idx,s_prime_idx) not in transition_map[action]:
                    transition_map[action][(s_idx,s_prime_idx)] = 1
                else:
                    transition_map[action][(s_idx,s_prime_idx)] += 1
                reward_matrix[s_idx,a_idx] += round(reward,1)
                #print('----')
    reward_matrix = reward_matrix/100    
    transition_matrix = np.repeat(np.diag(np.ones(90))[np.newaxis,:,:],90,axis=0)
    for act,s_s_data in transition_map.items():
        a_idx = utils.approx_index(np.arange(0.1,1,.01),act,0.01)
        for s_s_prime,ct in s_s_data.items():
            transition_matrix[a_idx,s_s_prime[0],s_s_prime[1]] = ct
    for a_idx in np.arange(transition_matrix.shape[0]):
        s_data = transition_matrix[a_idx]
        s_data_sum = np.sum(s_data,axis=1)
        s_data_prob = s_data / s_data_sum[:,None]
        s_data_prob[np.isnan(s_data_prob)] = 0
        transition_matrix[a_idx] = s_data_prob
        check_stochastic = np.sum(transition_matrix[a_idx], axis = 1)
        
    return transition_matrix, reward_matrix

P, R = generate_transition_matrix()
discount,horizon = 0.5,99
fh = mdptoolbox.mdp.QLearning(P, R, discount)
#fh = mdptoolbox.mdp.QLearning(P, R, 0.9)
fh.run()
#print([round((x+1)/10,1) for x in list(fh.policy)])
print(P)
print(R)
print([(xs,np.arange(0.1,1,.01)[xi]) for xs,xi in zip(np.arange(0.1,1,.01),list(fh.policy))])