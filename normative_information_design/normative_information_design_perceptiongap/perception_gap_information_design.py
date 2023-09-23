'''
Created on 15 Jan 2023

@author: Atrisha
'''

import functools

import gymnasium
from gymnasium.spaces import Discrete, Box
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
import constants
import utils
import matplotlib.pyplot as plt
import copy
from multiprocessing import Process
import os
from conda.common._logic import TRUE
import csv
from pathlib import Path
from scipy.special import softmax
import re
from collections import Counter
from sympy.stats import P, E, variance, Beta, Normal
from sympy import simplify
from scipy.stats import beta, norm, halfnorm, pearsonr
import seaborn as sns
import pandas as pd
import torch
from sympy.solvers import solve
from sympy import Symbol
from scipy import optimize
import math
from math import isnan
from tabulate import tabulate
import scipy




        

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "sim_s1"}
    
    def generate_opinions(self,distr_shape,distr_params):
        if distr_shape == 'uniform':
            ops = [o for o in np.random.uniform(low=0.5,high=1,size=int(self.num_players*distr_params['mean_op_degree']))]
            ops.extend([o for o in np.random.uniform(low=0,high=0.5,size=int(self.num_players*(1-distr_params['mean_op_degree'])))])
            np.random.shuffle(ops)
            if len(ops) < self.num_players:
                ops.extend([np.random.random()])
            return ops
        elif distr_shape == 'gaussian':
            ops = np.random.normal(distr_params['mean_op_degree'], distr_params['SD'], self.num_players)
            ops = np.clip(ops, 0, 1)
            np.random.shuffle(ops)
            if len(ops) < self.num_players:
                ops.extend([np.random.random()])
            return ops
        elif distr_shape == 'U':
            mu1, std1 = distr_params['mean_op_degree_apr'], distr_params['SD']  # First Gaussian distribution
            mu2, std2 = distr_params['mean_op_degree_disapr'], distr_params['SD']   # Second Gaussian distribution
            
            # Generate data points from the two Gaussian distributions
            data1 = np.random.normal(mu1, std1, int(self.num_players*distr_params['apr_weight']))
            data2 = np.random.normal(mu2, std2, int(self.num_players*(1-distr_params['apr_weight'])))
            
            # Create a U-shaped distribution by combining the two datasets
            ops = np.concatenate((data1, data2))
            ops = list(np.clip(ops, 0, 1))
            np.random.shuffle(ops)
            if len(ops) < self.num_players:
                ops.extend([np.random.random()])
            return ops
            

    def __init__(self, render_mode=None, attr_dict = None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        if attr_dict is not None:
            for key in attr_dict:
                setattr(self, key, attr_dict[key])
        self.num_players = 100
        self.update_rate = 10
        self.sanc_marginal_target = 0.2
        #self.norm_context_list = ['n1','n2','n3','n4']
        self.norm_context_list = ['n1']
        self.security_util = 0.2
        sanctioning_vals = np.random.normal(0.5, 0.1, self.num_players)
        self.mean_sanction, self.mean_sanction_baseline = 0.5,0.5
        sanctioning_vals = np.clip(sanctioning_vals, 0, 1)
        self.possible_agents = [Player(r,self) for r in range(self.num_players)]
        self.results_map = dict()
        self.observations = None
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        
        ''' Define list of normative contexts and initial distributions '''
        #self.norm_contexts_distr = {x:0.25 for x in self.norm_context_list}
        if not hasattr(self, 'norm_contexts_distr'):
            #self.norm_contexts_distr = {'n1':0.4,'n2':0.2,'n3':0.3,'n4':0.1} 
            self.norm_contexts_distr = {'n1':1} 
        #self.norm_contexts_distr = {k:v/np.sum(list(self.norm_contexts_distr.values())) for k,v in self.norm_contexts_distr.items()}
        ''' Sample the player opinions based on their private contexts sampled from the context distribution '''
        try_ct = 0
        if not hasattr(self, 'players_private_contexts'):
            players_private_contexts = np.random.choice(a=list(self.norm_contexts_distr.keys()),size=self.num_players,p=list(self.norm_contexts_distr.values()))
            while(set(players_private_contexts) != set(self.norm_context_list)):
                try_ct+=1
                print('trying...',try_ct)
                players_private_contexts = np.random.choice(a=list(self.norm_contexts_distr.keys()),size=self.num_players,p=list(self.norm_contexts_distr.values()))
            self.players_private_contexts = players_private_contexts
        players_private_contexts  = self.players_private_contexts
        for idx,op in enumerate(players_private_contexts): self.possible_agents[idx].norm_context = players_private_contexts[idx]
        
        #distr_params = {'mean_op_degree':self.true_state['n1']}
        #ops = self.generate_opinions('uniform',distr_params)
        
        #distr_params = {'mean_op_degree':self.true_state['n1'],'SD':0.2}
        #ops = self.generate_opinions('gaussian',distr_params)
        
        distr_params = {'mean_op_degree_apr':0.7,'mean_op_degree_disapr':0.4,'apr_weight':0.5,'SD':0.05}
        ops = self.generate_opinions('U',distr_params)
        
        '''
        opinions = np.random.choice([1,0],size=self.num_players,p=[norm_context_appr_rate['n1'], 1-norm_context_appr_rate['n1']])
        opinions = opinions.reshape((self.num_players,len(self.norm_context_list)))
        
        self.opinion_marginals = dict()
        for n_idx,norm_context in enumerate(self.norm_context_list):
            ops = opinions[:,n_idx]
            for idx,op in enumerate(ops): 
                self.possible_agents[idx].opinion[norm_context] = np.random.uniform(0.5,1) if op == 1 else np.random.uniform(0,0.5)
        '''
        for idx,op in enumerate(ops): 
            self.possible_agents[idx].opinion['n1'] = op
        for ag in self.possible_agents:
            ag.init_beliefs(self)
        
        for idx,s in enumerate(sanctioning_vals):
            agent_op = self.possible_agents[idx].opinion['n1']
            self.possible_agents[idx].sanction_capacity = np.random.normal(0.5,.1)#np.random.normal(agent_op, 0.1)
        ''' Define the marginal approval means'''
        
        
    
    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=0, high=1.0, shape=(1, 2), dtype=np.float16)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)
    
    def state(self,agent=None):
        pass
        

    def render(self,msg):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        ''' Maybe just display the mean beliefs of approval and payoffs stratified by each norm context.'''
        

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent.id: None for agent in self.agents}

        if not return_info:
            return observations
        else:
            infos = {agent: {} for agent in self.agents}
            return observations, infos

    def step(self, actions, iter_no, baseline):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # rewards for all agents are placed in the rewards dictionary to be returned
        ''' Since only the sender agent is learning, we do not need a full dict '''
        observed_action_values = np.array([ag.action[1] for ag in self.agents if ag.action[0]!=-1])
        num_observation = len(observed_action_values)
        num_participation = len([ag for ag in self.agents if ag.action[0]!=-1])/self.num_players
        if not baseline:
            ''' Let the reward be inversely proportional to the opinion value extremity'''
            
            
            baseline_op_mean = np.mean([ag.opinion[ag.norm_context] for ag in self.agents])
            
                
            terminations = {agent.id: False for agent in self.agents}
    
            self.num_moves += 1
            env_truncation = self.num_moves >= self.NUM_ITERS
            truncations = {agent.id: env_truncation for agent in self.agents}
    
            ''' Observation is the next state, or the common prior change '''
            num_appr = len([ag.action[0] for ag in self.agents if ag.action[0]==1 and ag.action[0]!=-1])
            num_disappr = len([ag.action[0] for ag in self.agents if ag.action[0]==0 and ag.action[0]!=-1])
            
            if num_observation > 0:
                
                theta_prime_rate = np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1])
                theta_prime_rate_appr = np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] >=0.5])
                theta_prime_rate_disappr = np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] <0.5])
                
                
                
                for agent in self.agents:
                    if agent.common_posterior_outgroup<0.5:
                        f=1
                
                    _var = utils.beta_var(agent.common_prior_outgroup[0], agent.common_prior_outgroup[1])
                    agent.common_prior_outgroup = utils.est_beta_from_mu_sigma(agent.common_posterior_outgroup, math.sqrt(_var))
                    
                    if agent.listened_to == 'extensive':
                        a_prime = theta_prime_rate*self.update_rate
                        b_prime =  self.update_rate-a_prime
                        agent.common_prior = (agent.common_prior[0]+a_prime, agent.common_prior[1]+b_prime)
                    else:
                        
                        if agent.opinion[agent.norm_context] >=0.5:
                            if not math.isnan(theta_prime_rate_appr):
                                a_prime = theta_prime_rate_appr*self.update_rate
                                b_prime =  self.update_rate-a_prime
                                agent.common_prior_ingroup = (agent.common_prior_ingroup[0]+a_prime, agent.common_prior_ingroup[1]+b_prime)
                            
                            if not math.isnan(theta_prime_rate_disappr):
                                if not hasattr(self, 'outgroup_update_rate'):
                                    self.outgroup_update_rate = self.update_rate/10
                                b_prime =  (1-theta_prime_rate_disappr)*self.outgroup_update_rate
                                a_prime =  self.outgroup_update_rate-b_prime
                                agent.common_prior_outgroup = (agent.common_prior_outgroup[0]+a_prime, agent.common_prior_outgroup[1]+b_prime)
                        else:
                            if not math.isnan(theta_prime_rate_disappr):
                                b_prime =  (1-theta_prime_rate_disappr)*self.update_rate
                                a_prime =  self.update_rate-b_prime
                                agent.common_prior_ingroup = (agent.common_prior_ingroup[0]+a_prime, agent.common_prior_ingroup[1]+b_prime)
                            
                            if not math.isnan(theta_prime_rate_appr):
                                if not hasattr(self, 'outgroup_update_rate'):
                                    self.outgroup_update_rate = self.update_rate/10
                                
                                a_prime = theta_prime_rate_appr*self.outgroup_update_rate
                                b_prime =  self.outgroup_update_rate-a_prime
                                agent.common_prior_outgroup = (agent.common_prior_outgroup[0]+a_prime, agent.common_prior_outgroup[1]+b_prime)
                        
                                
                
            
            ''' Assign (institutional) rewards'''
            if self.extensive:
                rewards = (num_participation-0.5)*2
            else:
                mean_appr_degree = np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1])
                mean_sanctioning_capacity = np.mean([ag.action[3] for ag in self.agents if ag.action[0]!=-1])
                rewards = mean_sanctioning_capacity*mean_appr_degree
            
            observations = self.common_prior
            if self.only_intensive:
                x = [(ag.opinion[ag.norm_context],ag.listened_to,ag.action[0],ag.action[5],ag.action[6]) for ag in self.agents if ag.opinion[ag.norm_context]>=0.5]
                x.sort(key=lambda x: x[0])
            else:
                x = [(ag.opinion[ag.norm_context],ag.listened_to,ag.action[0],ag.action[5],ag.action[6]) for ag in self.agents if ag.opinion[ag.norm_context]>=0.5]
                x.sort(key=lambda x: x[0])
            # typically there won't be any information in the infos, but there must
            # still be an entry for each agent
            
            infos = x
    
            if env_truncation:
                self.agents = []
    
            if self.render_mode == "human":
                self.render(iter_no)
            return observations, rewards, terminations, truncations, infos
        else:
            num_appr = len([ag.action[0] for ag in self.agents if ag.action[0]==1 and ag.action[0]!=-1])
            num_disappr = len([ag.action[0] for ag in self.agents if ag.action[0]==0 and ag.action[0]!=-1])
            if num_observation > 0:
                theta_prime_rate = np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1])
                #theta_prime_by_nums = num_appr /(num_appr+num_disappr)
                #theta_prime_rate = theta_prime_by_nums
                a_prime = theta_prime_rate*self.update_rate
                b_prime =  self.update_rate-a_prime
                self.prior_baseline = (self.prior_baseline[0]+a_prime, self.prior_baseline[1]+b_prime)
                
                a_prime_prop = num_participation*self.update_rate
                b_prime_prop =  self.update_rate-a_prime_prop
                self.prior_prop_baseline = (self.prior_prop_baseline[0]+a_prime, self.prior_prop_baseline[1]+b_prime)
                
                mean_appr_degree = np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1])
                mean_sanctioning_capacity = np.mean([ag.action[3] for ag in self.agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] >= 0.5])
                #print('------>',self.common_proportion_prior[0]/np.sum(self.common_proportion_prior), self.common_prior[0]/np.sum(self.common_prior), '||', mean_appr_degree,mean_sanctioning_capacity,)
                self.mean_sanction_baseline = mean_sanctioning_capacity*mean_appr_degree
    
    
    @property
    def common_prior_mean(self):   
        return self.common_prior[0]/sum(self.common_prior)
    
        
class Player():
    
    def __init__(self,id, env):
        self.id = id
        self.payoff_tol = constants.payoff_tol
        self.opinion = dict()
        self.opinion_val = dict()
        self.norm_context = 'n1'
        '''
        if real_p:
            self.shadow_player = Player(-id,False)
        '''
        
        self.total_reward = 0
        self.total_participation = 0
    
    def init_beliefs(self,env):
        self.common_prior = env.common_prior
        if self.opinion['n1'] >= 0.5:
            self.common_prior_ingroup = min(halfnorm.rvs(loc=env.common_prior_appr_input[0], scale=0.5),4.99)
            self.common_prior_ingroup = (self.common_prior_ingroup,5-self.common_prior_ingroup)
            
            self.common_prior_outgroup = (self.common_prior_ingroup[1],self.common_prior_ingroup[0])
        else:
            sample = min(halfnorm.rvs(loc=env.common_prior_appr_input[0], scale=0.5),4.99)
             
            self.common_prior_outgroup = (sample,5-sample)
            self.common_prior_ingroup = (self.common_prior_outgroup[1],self.common_prior_outgroup[0])
        self.common_posterior = env.common_prior
        self.common_posterior_ingroup = env.common_prior_ingroup
        self.common_posterior_outgroup = env.common_prior_outgroup
        self.common_proportion_prior = env.common_proportion_prior
        self.common_proportion_posterior = env.common_proportion_prior
    
    def act(self, env, run_type, baseline):
        if run_type in ['baseline','self-ref']:
            return self.act_with_sanction_cap(env,baseline)
        
        
    
    def act_with_sanction_cap(self, env,baseline):
        
        
        
        u_bar = env.security_util
        op = self.opinion[self.norm_context]
        
        if op<0.5:
            f=1
        
        if not baseline:
            theta = self.common_posterior
            theta_ingroup = self.common_posterior_ingroup
            theta_outgroup = self.common_posterior_outgroup
            n_p = self.common_proportion_posterior
            
            op_degree = op if op >= 0.5 else (1-op)
            conc_prop = n_p if op >= 0.5 else (1-n_p)
            
            conc_deg = theta if op >= 0.5 else (1-theta)
            if op_degree*conc_prop*(1-conc_deg)*1**(-conc_deg) > env.sanc_marginal_target:
                opt_sanc = 1
                opt_sanc_o = 1
            else:
                opt_sanc = math.pow(env.sanc_marginal_target/(op_degree*conc_prop*(1-conc_deg)),-1/conc_deg)
                opt_sanc_o = math.pow(env.sanc_marginal_target/(op_degree*conc_prop*(conc_deg)),-1/(1-conc_deg))
            self.sanction_intensity = min(self.sanction_capacity,opt_sanc)
            self.sanction_intensity_o = min(self.sanction_capacity,opt_sanc_o)
            
            conc_deg_ingroup = theta_ingroup if op >= 0.5 else (1-theta_ingroup)
            if op_degree*conc_prop*(1-conc_deg_ingroup)*1**(-conc_deg_ingroup) > env.sanc_marginal_target:
                opt_sanc = 1
            else:
                opt_sanc = math.pow(env.sanc_marginal_target/(op_degree*conc_prop*(1-conc_deg_ingroup)),-1/conc_deg_ingroup)
            self.sanction_intensity_ingroup = min(self.sanction_capacity,opt_sanc)
            
            conc_deg_outgroup = theta_outgroup if op >= 0.5 else (1-theta_outgroup)
            if op_degree*conc_prop*(1-conc_deg_outgroup)*1**(-conc_deg_outgroup) > env.sanc_marginal_target:
                opt_sanc = 1
            else:
                opt_sanc = math.pow(env.sanc_marginal_target/(op_degree*conc_prop*(1-conc_deg_outgroup)),-1/conc_deg_outgroup)
            self.sanction_intensity_outgroup = min(self.sanction_capacity,opt_sanc)
            
            util = lambda op : op*(self.sanction_intensity**(1-theta))*n_p if op >= 0.5 else (1-op)*(self.sanction_intensity**theta)*(1-n_p)
            util_ingroup = lambda op : op*(self.sanction_intensity_ingroup**(1-theta_ingroup))*n_p if op >= 0.5 else (1-op)*(self.sanction_intensity_ingroup**theta_ingroup)*(1-n_p)
            util_outgroup = lambda op : op*(self.sanction_intensity_outgroup**(1-theta_outgroup))*n_p if op >= 0.5 else (1-op)*(self.sanction_intensity_outgroup**theta_outgroup)*(1-n_p)
            
            util_val = util(op)
            util_val_ingroup = util_ingroup(op)
            util_val_outgroup = util_outgroup(op)
            
            if env.only_intensive:
                if util_val_ingroup > u_bar or util_val_outgroup > u_bar:
                    self.action_code = 1 if op >= 0.5 else 0
                    self.listened_to = 'intensive'
                    self.action_util = np.mean([util_val_ingroup,util_val_outgroup])
                    if util_val_outgroup < u_bar:
                        self.common_posterior_outgroup = self.common_prior_outgroup[0]/np.sum(self.common_prior_outgroup)
                    
                else:
                    self.action_code = -1
                    self.action_util = u_bar
                    self.listened_to = 'intensive'
                self.action =(self.action_code,self.action_util,self.opinion[self.norm_context],self.sanction_intensity,self.listened_to,0,
                          self.common_posterior_outgroup)
            else:
                if util_val_ingroup > u_bar and util_val_outgroup > u_bar and util_val > u_bar:
                    self.action_code = 1 if op >= 0.5 else 0
                    self.listened_to = 'extensive' if self.sanction_intensity < np.max([self.sanction_intensity_ingroup,self.sanction_intensity_outgroup]) else 'intensive'
                    self.listened_to = 'both'
                    self.action_util = max(util_val, np.mean([util_val_ingroup,util_val_outgroup]))
                elif (util_val > u_bar) and (util_val_ingroup <= u_bar or util_val_outgroup <= u_bar):
                    self.action_code = 1 if op >= 0.5 else 0
                    self.listened_to = 'extensive'
                    self.action_util = util_val
                elif util_val_ingroup > u_bar and util_val_outgroup > u_bar and util_val < u_bar:
                    self.action_code = 1 if op >= 0.5 else 0
                    self.listened_to = 'intensive'
                    self.action_util = np.mean([util_val_ingroup,util_val_outgroup])
                else:
                    self.action_code = -1
                    self.action_util = u_bar
                    if util_val > np.mean([util_val_ingroup,util_val_outgroup]):
                        utilmax = util_val
                        self.listened_to = 'extensive'
                        self.sanction_intensity = self.sanction_intensity
                    else:
                        utilmax = np.mean([util_val_ingroup,util_val_outgroup])
                        self.listened_to = 'intensive'
                        self.sanction_intensity = self.sanction_intensity_outgroup
                
                if self.listened_to == 'both':
                    self.ext_int_diff =  (n_p*op_degree*(1-conc_deg_ingroup)*self.sanction_intensity_ingroup**(-conc_deg_ingroup)+ n_p*op_degree*(1-conc_deg_outgroup)*self.sanction_intensity_outgroup**(-conc_deg_outgroup))/2 - n_p*op_degree*(1-conc_deg)*self.sanction_intensity**(-conc_deg)
                    #self.ext_int_diff = (util_val_ingroup+util_val_outgroup)/2 - util_val
                    self.ext_int_diff1 = self.ext_int_diff
                    self.ext_int_diff = self.ext_int_diff*10
                    if self.ext_int_diff < 0:
                        self.ext_int_diff = 1-abs(self.ext_int_diff)
                    self.ext_int_diff = 0 if self.ext_int_diff < 0 else self.ext_int_diff
                    int_ingr_sig_val = n_p*op_degree*self.sanction_intensity_ingroup**(1-conc_deg_ingroup)
                    int_outgr_sig_val = n_p*op_degree*self.sanction_intensity_outgroup**(1-conc_deg_outgroup)
                    ext_sig_val_ingr = n_p*op_degree*self.sanction_intensity**(1-conc_deg)
                    ext_sig_val_outgr = n_p*op_degree*self.sanction_intensity_o**(conc_deg)
                    
                    signal_values = [(n_p*op_degree*(1-conc_deg_ingroup)*self.sanction_intensity_ingroup**(-conc_deg_ingroup)+ n_p*op_degree*(1-conc_deg_outgroup)*self.sanction_intensity_outgroup**(-conc_deg_outgroup))/2,n_p*op_degree*(1-conc_deg)*self.sanction_intensity**(-conc_deg)]
                    signal_values = [(int_ingr_sig_val+int_outgr_sig_val)/2,ext_sig_val_ingr]
                    #inst_weights = scipy.special.softmax(signal_values)
                    self.ext_int_diff1 = signal_values[0]-signal_values[1]
                    inst_weights_inten = (5/((op_degree)*10))*(self.ext_int_diff1+(op_degree))
                    inst_weights = [inst_weights_inten,1-inst_weights_inten]
                    inst_weights_sm = scipy.special.softmax(signal_values)
                    inst_weights = [s/np.sum(signal_values) for s in signal_values]
                    f=1
                    #inst_weights = scipy.special.softmax(signal_values)
                    
                    #inst_weights = [self.ext_int_diff,(1-self.ext_int_diff)]
                    '''
                    if self.listened_to == 'both':
                        if self.opinion[self.norm_context] >=0.5:
                            self.common_posterior_outgroup = (1-self.ext_int_diff)*(1-self.common_posterior) + self.ext_int_diff*self.common_posterior_outgroup #if self.listened_to == 'extensive' else np.mean([self.common_posterior,self.common_posterior_outgroup])
                        else:
                            self.common_posterior_outgroup = (1-self.ext_int_diff)*self.common_posterior + self.ext_int_diff*self.common_posterior_outgroup
                    elif self.listened_to == 'extensive':
                        self.post = self.common_posterior
                        if self.opinion[self.norm_context] >=0.5:
                            self.common_posterior_outgroup = (1-self.common_posterior)
                        else:
                            self.common_posterior_outgroup = self.common_posterior
                    else:
                        self.common_posterior_outgroup = self.common_posterior_outgroup
                    
                    '''
                    '''
                    if self.ext_int_diff < 0:
                        print('Warning!!!!!!!!')
                    if self.opinion[self.norm_context] >=0.5:
                        if self.common_posterior >= 0.5:
                            self.common_posterior_outgroup = (1-self.ext_int_diff)*(1-self.common_posterior) + self.ext_int_diff*self.common_posterior_outgroup
                            
                        else:
                            self.common_posterior_outgroup = (1-self.ext_int_diff)*self.common_posterior + self.ext_int_diff*self.common_posterior_outgroup
                    else:
                        if self.common_posterior >= 0.5:
                            self.common_posterior_outgroup = (1-self.ext_int_diff)*self.common_posterior + self.ext_int_diff*self.common_posterior_outgroup
                        else:
                            self.common_posterior_outgroup = (1-self.ext_int_diff)*(1-self.common_posterior) + self.ext_int_diff*self.common_posterior_outgroup
                       
                    f=1
                    '''
                    
                    #print(_diff,inst_weights[0])
                    if self.opinion[self.norm_context] >=0.5:
                        if self.common_posterior >= 0.5:
                            self.common_posterior_outgroup = inst_weights[1]*(1-self.common_posterior) + inst_weights[0]*self.common_posterior_outgroup
                            
                        else:
                            self.common_posterior_outgroup = inst_weights[1]*self.common_posterior + inst_weights[0]*self.common_posterior_outgroup
                    else:
                        if self.common_posterior >= 0.5:
                            self.common_posterior_outgroup = inst_weights[1]*self.common_posterior + inst_weights[0]*self.common_posterior_outgroup
                        else:
                            self.common_posterior_outgroup = inst_weights[1]*(1-self.common_posterior) + inst_weights[0]*self.common_posterior_outgroup                    
                    
                    #self.ext_int_diff = [self.sanction_intensity, self.sanction_intensity_ingroup ,self.sanction_intensity_outgroup]
                    ''' The Bayesian Nash Eq action thresholds. '''
                    '''
                    if utilmax < u_bar:
                        self.action_code = -1
                        self.action_util = u_bar
                    else:
                        self.action_code = 1 if op >= 0.5 else 0
                        self.action_util = utilmax
                    '''
                    
                    ''' Now the sanctioning selection '''
                    ''' Sanction is selected based on the marginal utility as calculated by taking the derivative of the utility o_{i}*s^{1-o_-i}*n or (1-o_i)s^{o_-i}*(1-n) as applicable '''
                    ''' The derivative of the above function is o_{i}*n*(1-o_-i)s^{-o_-i} or (1-o_{i})*(1-n)*(o_-i)s^{o_-i-1} as applicable '''
                    ''' If the marginal utility threshold is s_max_i  '''
                else:
                    if self.listened_to == 'intensive':
                        self.common_posterior_outgroup = self.common_posterior_outgroup
                        self.ext_int_diff = (util_val_outgroup - util_val)
                    else:
                        if self.opinion[self.norm_context] >=0.5:
                            self.common_posterior_outgroup = (1-self.common_posterior) if self.common_posterior >= 0.5 else self.common_posterior
                        else:
                            self.common_posterior_outgroup = self.common_posterior if self.common_posterior >= 0.5 else (1-self.common_posterior)
                        self.ext_int_diff = (util_val - util_val_outgroup)
            
                self.action =(self.action_code,self.action_util,self.opinion[self.norm_context],self.sanction_intensity,self.listened_to,self.ext_int_diff,
                              self.common_posterior_outgroup)
        else:
            theta_baseline = env.prior_baseline[0]/sum(env.prior_baseline)
            prop_baseline = env.prior_prop_baseline[0]/sum(env.prior_prop_baseline)
            op_degree = op if op >= 0.5 else (1-op)
            conc_prop = prop_baseline if op >= 0.5 else (1-prop_baseline)
            conc_deg = theta_baseline if op >= 0.5 else (1-theta_baseline)
            if op_degree*conc_prop*(1-conc_deg)*1**(-conc_deg) > env.sanc_marginal_target:
                opt_sanc = 1
            else:
                opt_sanc = math.pow(env.sanc_marginal_target/(op_degree*conc_prop*(1-conc_deg)),-1/conc_deg)
            self.sanction_intensity = opt_sanc
            
            util_baseline = lambda op : op*(self.sanction_intensity**(1-theta_baseline))*prop_baseline if op >= 0.5 else (1-op)*(self.sanction_intensity**theta_baseline)*(1-prop_baseline)
            
            if util_baseline(op) < u_bar:
                self.action_code_baseline = -1
                self.action_util_baseline = u_bar
            else:
                self.action_code_baseline = 1 if op >= 0.5 else 0
                self.action_util_baseline = util_baseline(op)
            
            
            
            self.action =(self.action_code_baseline,self.action_util_baseline,self.opinion[self.norm_context],self.sanction_intensity)
        
        return self.action
    
        
    def generate_posteriors(self,env,opt_signals_in,common_proportion_prior,update_type):
        opt_signals = dict(opt_signals_in)
        if update_type == 'common':
            common_prior = self.common_prior
            common_prior_mean = common_prior[0]/np.sum(common_prior)
            curr_state = common_prior_mean
            signal_distribution = opt_signals[round(curr_state,1)]
        else:
            if self.opinion[self.norm_context] < 0.5:
                ''' signals need to be reversed'''
                opt_signals = {round(1-k,1):(1-v) for k,v in opt_signals.items()}
            common_prior = self.common_prior_ingroup if update_type=='ingroup' else self.common_prior_outgroup
            common_prior_mean = common_prior[0]/np.sum(common_prior)
            curr_state = common_prior_mean
            try:
                signal_distribution = opt_signals[round(curr_state,1)]
            except KeyError:
                print('Info:')
                print(self.opinion[self.norm_context])
                print(update_type)
                print(curr_state)
                print(opt_signals)
                print(common_prior)
                raise
                
        if abs(signal_distribution-common_prior_mean) > env.normal_constr_w:
            common_posterior = common_prior_mean
            common_proportion_posterior = common_proportion_prior[0]/np.sum(common_proportion_prior)
            return common_posterior, common_proportion_posterior
        '''
            This method updates the posterior for the population (posterior over the rate of approval) based on the signal dristribution.
            Since signal distribution is a Bernoulli, we can get individual realizations of 0 and 1 separately, and then take the expectation.
        '''
        def _post(x,priors_rescaled,likelihood_rescaled):
            prior_x = priors_rescaled[x]
            
            ''' This evaluates the likelohood as conditioned on the state value x
                Find prob of signal distr. (this connects to the 'state', i.e., have state information)
                Then calc liklihood of the signal realization
            '''
            signal_param_prob = likelihood_rescaled[x]
            lik = lambda x: signal_param_prob*signal_distribution if x == 1 else signal_param_prob*(1-signal_distribution)
            post = (prior_x*lik(1),prior_x*lik(0))
            return post
        all_posteriors = []
        priors_rescaled, likelihood_rescaled = dict(), dict()
        for x in np.linspace(0.01,0.99,100):
            priors_rescaled[x] = utils.beta_pdf(x, common_prior[0], common_prior[1])
            _constr_distr = utils.Gaussian_plateu_distribution(signal_distribution,.01,env.normal_constr_w)
            likelihood_rescaled[x] = _constr_distr.pdf(x)
            #_constr_distr = utils.Gaussian_plateu_distribution(0,.01,self.normal_constr_sd)
            #likelihood_rescaled[x] = _constr_distr.pdf(abs(x-signal_distribution))
        priors_rescaled = {k:v/sum(list(priors_rescaled.values())) for k,v in priors_rescaled.items()}
        likelihood_rescaled = {k:v/sum(list(likelihood_rescaled.values())) for k,v in likelihood_rescaled.items()}
        
        
        for x in np.linspace(0.01,0.99,100):
            posteriors = _post(x,priors_rescaled,likelihood_rescaled)
            ''' Since the signal realization will be based on the signal distribution, we can take the expectation of the posterior w.r.t each realization.'''
            expected_posterior_for_state_x = (signal_distribution*posteriors[0]) + ((1-signal_distribution)*posteriors[1])
            all_posteriors.append(expected_posterior_for_state_x)
        all_posteriors = [x/np.sum(all_posteriors) for x in all_posteriors]
        exp_x = np.sum([x*prob_x for x,prob_x in zip(np.linspace(0.01,0.99,100),all_posteriors)])
        '''
        print(exp_x)
        plt.figure()
        plt.plot(list(priors_rescaled.keys()),list(priors_rescaled.values()))
        plt.plot(list(likelihood_rescaled.keys()),list(likelihood_rescaled.values()))
        plt.plot(np.linspace(0.01,0.99,100),all_posteriors)
        plt.title('likelihood:'+str(signal_distribution)+','+str(self.common_prior[0]/sum(self.common_prior)))
        plt.show()
        '''
        common_posterior = exp_x
        common_proportion_posterior = common_proportion_prior[0]/np.sum(common_proportion_prior)
        if math.isnan(common_posterior):
            f=1
        return common_posterior, common_proportion_posterior
        ''' Generate posteriors for norm support '''
        '''
        self.st_signal = (1,0) if signal_distribution > 0.5 else (0,1) if signal_distribution < 0.5 else (0.5,0.5)
        update_rate = 2
        self.common_proportion_posterior = (self.common_proportion_prior[0]+(update_rate*self.st_signal[0]),self.common_proportion_prior[1]+(update_rate*self.st_signal[1]))
        self.common_proportion_posterior = self.common_proportion_posterior[0]/np.sum(self.common_proportion_posterior).
        '''
        #self.common_proportion_posterior = exp_x
        

    
class StewardAgent():
    
    def __init__(self,qnetwork):
        self.qnetwork = qnetwork       
        
    
class RunInfo():
    
    def __init__(self,iter):
        self.iter = iter
        
def run_sim(run_param):
    """ ENV SETUP """
    common_prior, common_prior_ingroup, common_prior_outgroup = run_param['common_prior'],run_param['common_prior_ingroup'],run_param['common_prior_outgroup']
    common_proportion_prior = run_param['common_proportion_prior']
    normal_constr_w = run_param['normal_constr_w']
    common_prior_mean = common_prior[0]/sum(common_prior)
    state_evolution,state_evolution_baseline = dict(), dict()
    lst = []
    cols = ['run_id', 'time_step', 'listened', 'opinion', 'out_belief']
    lst_df = pd.DataFrame(lst, columns=cols)
    #for signal_distr_theta_idx, signal_distr_theta in enumerate([common_prior_mean-(normal_constr_w+0.05),common_prior_mean-(normal_constr_w-0.05),common_prior_mean+(normal_constr_w+0.05),common_prior_mean+(normal_constr_w-0.05)]):
    '''
        opt_signals acquired from running solving_tools.py separately
    '''
    opt_signals, opt_signals_ingroup, opt_signals_outgroup = run_param['opt_signals'], run_param['opt_signals_ingroup'], run_param['opt_signals_outgroup']
    for batch_num in np.arange(100):
        env = parallel_env(render_mode='human',attr_dict={'true_state':{'n1':0.55},'extensive':False,
                                                            'common_prior' : common_prior,
                                                            'common_prior_ingroup' : common_prior_ingroup,
                                                            'common_prior_outgroup' : common_prior_outgroup,
                                                            'common_proportion_prior' : common_proportion_prior,
                                                            'common_prior_appr_input':run_param['common_prior_appr_input'],
                                                            'only_intensive':run_param['only_intensive']})
        ''' Check that every norm context has at least one agent '''
        if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
            raise Exception()
        env.reset()
        env.no_print = True
        env.NUM_ITERS = 100
        
        env.prior_baseline = env.common_prior
        env.prior_prop_baseline = common_proportion_prior
        env.normal_constr_w = normal_constr_w
        #env.constraining_distribution = utils.Gaussian_plateu_distribution(env.common_prior[0]/sum(env.common_prior),.01,.3)
        #env.constraining_distribution = utils.Gaussian_plateu_distribution(.3,.01,.3)
        dataset = []
        history = [[(ag.opinion[ag.norm_context],'intensive' if env.only_intensive else 'both',1,0,ag.common_posterior_outgroup[0]/np.sum(ag.common_posterior_outgroup)) for ag in env.possible_agents if ag.opinion[ag.norm_context]>=0.5]]
        #history = []
        '''
        plt.figure()
        plt.hist([ag.opinion[ag.norm_context] for ag in env.possible_agents])
        plt.show()
        '''
        for i in np.arange(100):
            mean_common_prior_var = np.mean([utils.beta_var(agent.common_prior[0],agent.common_prior[1]) for agent in env.possible_agents])
            mean_common_prior_ingroup_var = np.mean([utils.beta_var(agent.common_prior_ingroup[0],agent.common_prior_ingroup[1]) for agent in env.possible_agents])
            mean_common_prior_outgroup_var = np.mean([utils.beta_var(agent.common_prior_outgroup[0],agent.common_prior_outgroup[1]) for agent in env.possible_agents])
            
            if min(mean_common_prior_var,mean_common_prior_ingroup_var,mean_common_prior_outgroup_var) < 0.005:
                break
            
            print(common_prior,batch_num,i)
            #curr_state = np.mean([agent.common_prior[0]/sum(agent.common_prior) for agent in env.possible_agents])
            #curr_state_ingroup = np.mean([agent.common_prior_ingroup[0]/sum(agent.common_prior_ingroup) for agent in env.possible_agents])
            #curr_state_outgroup = np.mean([agent.common_prior_outgroup[0]/sum(agent.common_prior_outgroup) for agent in env.possible_agents])
            #signal_distr_theta = curr_state - 0.3
            
            #signal_distr_theta = opt_signals[round(curr_state,1)]
            #signal_distr_theta_ingroup = opt_signals_ingroup[round(curr_state_ingroup,1)]
            #signal_distr_theta_outgroup = opt_signals_outgroup[round(curr_state_outgroup,1)]
            '''
            _d = abs(signal_distr_theta-env.common_prior_mean)
            if _d <= env.normal_constr_w:
                valid_distr = True
            else:
                valid_distr = False
            '''
            
            if i not in  state_evolution:
                state_evolution[i] = []
            state_evolution[i].append((env.common_prior[0]/sum(env.common_prior),env.mean_sanction))
            if i not in  state_evolution_baseline:
                state_evolution_baseline[i] = []
            state_evolution_baseline[i].append((env.prior_baseline[0]/sum(env.prior_baseline),env.mean_sanction_baseline))
            
            ''' break if the mean beliefs (common or any of ingroup and outgroup) variance is very low. Because then information is stable '''
            
            ''' act is based on the new posterior acting as prior '''
            for agent in env.possible_agents:
                if math.isnan(agent.common_prior[0]/np.sum(agent.common_prior)) or math.isnan(agent.common_prior_outgroup[0]/np.sum(agent.common_prior_outgroup)) or math.isnan(agent.common_prior_ingroup[0]/np.sum(agent.common_prior_ingroup)):
                    continue
                agent.common_posterior, agent.common_proportion_posterior = agent.generate_posteriors(env,opt_signals,agent.common_proportion_prior,'common')
                agent.common_posterior_ingroup, agent.common_proportion_posterior = agent.generate_posteriors(env,opt_signals_ingroup,agent.common_proportion_prior,'ingroup')
                agent.common_posterior_outgroup, agent.common_proportion_posterior = agent.generate_posteriors(env,opt_signals_outgroup,agent.common_proportion_prior,'outgroup')
            
                
            actions = {agent.id:agent.act(env,run_type='self-ref',baseline=False) for agent in env.possible_agents}
            '''
            plt.figure()
            plt.hist([ag.opinion[ag.norm_context] for ag in env.possible_agents if ag.action[0]!=-1])
            plt.show()
            '''
            ''' common prior is updated based on the action observations '''
            observations, rewards, terminations, truncations, infos = env.step(actions,i,baseline=False)
            history.append(infos)
            
            #actions = {agent.id:agent.act(env,run_type='self-ref',baseline=True) for agent in env.possible_agents}
            #env.step(actions,i,baseline=True)
        data = {'run_id':[batch_num]*len(history[0]), 'time_step':[1]*len(history[0]), 
                'listened': [d[1] for d in history[0]],
                'opinion': [d[0] for d in history[0]], 'out_belief': [d[4] for d in history[0]] }
        df = pd.DataFrame(data)
        data = df.dropna()
        lst_df = lst_df.append(data)
        
        data = {'run_id':[batch_num]*len(history[-1]), 'time_step':[len(history)+1]*len(history[-1]), 
                'listened': [d[1] for d in history[-1]],
                'opinion': [d[0] for d in history[-1]], 'out_belief': [d[4] for d in history[-1]] }
        df = pd.DataFrame(data)
        data = df.dropna()
        lst_df = lst_df.append(data)
        '''
        sns.lmplot(x="x", y="y", data=df, ci=95, hue='listened')  
        subset_data = data[data['listened'] == 'both']
        r, p = pearsonr(subset_data['x'], subset_data['y'])
        ax = plt.gca()
        ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),transform=ax.transAxes)
        
        data = {'x': [d[0] for d in history[-1]], 'y': [d[4] for d in history[-1]], 'listened': [d[1] for d in history[-1]]}
        df = pd.DataFrame(data)
        data = df.dropna()
        
        sns.lmplot(x="x", y="y", data=df, ci=95, hue='listened')  
        subset_data = data[data['listened'] == 'both']
        r, p = pearsonr(subset_data['x'], subset_data['y'])
        ax = plt.gca()
        ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),transform=ax.transAxes)
        
        plt.show()  
        
            
            #env.common_prior = (np.random.randint(low=1,high=4),np.random.randint(low=1,high=4))
        cols = ['run_id', 'time_step', 'listened', 'opinion', 'out_belief']
        only_baseline_plot = False
        
        
        if not only_baseline_plot:
            for k,v in state_evolution.items():
                for _v in v:
                    lst.append([k,_v[0],'signal',_v[1]])
            
        
        for k,v in state_evolution_baseline.items():
            for _v in v:
                lst.append([k,_v[0],'no signal',_v[1]])
        '''
    
    return lst_df

if __name__ == "__main__":
    '''
    df_list = dict()#(3,1.3),(5,2),(2,3.3)
    for runlist in [((4,2),(2,4))]:
        run_param ={'common_prior':runlist[0],
                    'common_prior_appr_input':runlist[0],
                    'common_prior_ingroup':runlist[0],
                    'common_prior_outgroup':runlist[1],
                    'common_proportion_prior':(5,5),
                    'normal_constr_w':0.3,
                    'opt_signals':{0:0.1,0.1:0.1, 0.2:0.5, 0.3:0.5, 0.4:0.5, 0.5:0.5, 0.6:0.5, 0.7:0.5, 0.8:0.5, 0.9:0.6,1:0.7},
                    #'opt_signals_ingroup':{0.5:0.5, 0.6:0.5, 0.7:0.5, 0.8:0.6, 0.9:0.7,1:0.7},
                    'opt_signals_ingroup':{0.5:0.3, 0.6:0.3, 0.7:0.5, 0.8:0.6, 0.9:0.6,1:0.7},
                    'opt_signals_outgroup':{0:0.1,0.1:0.1, 0.2:0.1, 0.3:0.1, 0.4:0.1, 0.5:0.2},
                    'only_intensive':False}
        
        run_df = run_sim(run_param)
        df_list[str(runlist)] = run_df
    print(tabulate(run_df, headers='keys', tablefmt='psql'))
    if run_param['only_intensive']:
        run_df.to_csv('data\\lst_data_intensive_const_sanc.csv', index=True)
    else:
        run_df.to_csv('data\\lst_data_const_sanc.csv', index=True)
    '''
    
    run_df = pd.read_csv('data\\lst_data_intensive.csv', header=0).applymap(lambda x: x.strip() if isinstance(x, str) else x)
    run_df_both = pd.read_csv('data\\lst_data.csv', header=0).applymap(lambda x: x.strip() if isinstance(x, str) else x)
    run_df.columns = [col.strip() for col in run_df.columns]
    run_df_both.columns = [col.strip() for col in run_df_both.columns]
    run_df_both.loc[run_df_both['time_step'] > 1, 'time_step'] = 'Final_Both'
    run_df_both.loc[run_df_both['time_step'] == 1, 'time_step'] = 'Initial'
    run_df_both = run_df_both[run_df_both['time_step'] == 'Final_Both']
    
    run_df.loc[run_df['time_step'] > 1, 'time_step'] = 'Final_intensive'
    run_df.loc[run_df['time_step'] == 1, 'time_step'] = 'Initial'
    run_df = run_df.append(run_df_both)
    sns.set_theme(style="darkgrid")
    g = sns.lmplot(x="opinion", y="out_belief", data=run_df, ci=95, hue='time_step',scatter=False)  
    
    
    ax = sns.scatterplot(data=run_df, x='opinion', y='out_belief', hue='time_step',legend=False)
    ax.set_xlabel('Opinion (approval)')
    ax.set_ylabel('Second order beliefs (beliefs about out-group)')
    
    df = {'x':[0.5,0.85],'y':[0.4,0.4]}
    sns.lineplot(data=df, x='x', y='y', linestyle='dashed', markers=True, dashes=(2, 2), label='Dashed Line', color='black',legend=False)
    groups = run_df['time_step'].unique()
    locs = {'Final_Both':0.33,'Final_intensive':0.14,'Initial':0.38}
    for idx,group in enumerate(groups):
        subset = run_df[run_df['time_step'] == group]
        r, p = pearsonr(subset['opinion'], subset['out_belief'])
        #if not math.isnan(r):
        print(group,r,p)
    
    #r, p = pearsonr(subset_data['opinion'], subset_data['out_belief'])
    #ax = plt.gca()
    #ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),transform=ax.transAxes)
    plt.show()
    
    