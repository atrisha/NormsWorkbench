'''
Created on 20 Jan 2023

@author: Atrisha
'''
import functools

import gymnasium
from gymnasium.spaces import Discrete, Box
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from collections import namedtuple
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
from scipy.stats import beta, norm, dirichlet
import seaborn as sns
import pandas as pd
import math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class heterogenous_parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "sim_s1"}

    def __init__(self, render_mode=None, attr_dict = None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        self.num_players = 100
        self.update_rate = 10
        self.norm_context_list = ['n1','n2','n3']
        self.norm_context_list_len = len(self.norm_context_list)
        self.security_util = 0.3
        self.possible_agents = [Player(r) for r in range(self.num_players)]
        self.results_map = dict()
        self.observations = None
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        if attr_dict is not None:
            for key in attr_dict:
                setattr(self, key, attr_dict[key])
        ''' Define list of normative contexts and initial distributions '''
        #self.norm_contexts_distr = {x:0.25 for x in self.norm_context_list}
        self.true_state = {k:beta(a=v[0],b=v[1]).mean() for k,v in self.true_state_distr_params.items()}     
        self.norm_contexts_distr = {k:v/np.sum(list(self.norm_contexts_distr.values())) for k,v in self.norm_contexts_distr.items()}
        self.common_prior_means = self.true_state_distr_params
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
        
        self.true_state_thetas = [self.true_state[n] for n in self.norm_context_list]
        self.opinions,self.corr_mat,self.mutual_info_mat = utils.generate_samples_copula([self.true_state_distr_params[n] for n in self.norm_context_list] ,(0,None), n_samples=self.num_players)
        self.corr_mat_ref_sum = np.sum(self.corr_mat[0,:])-1
        self.constructed_corr_mat = self.construct_correlation_from_opinions()
        self.constructed_corr_mat_ref_sum = np.sum(self.constructed_corr_mat[0,:])-1
        
        opinions = self.opinions
        
        self.opinion_marginals = dict()
        for n_idx,norm_context in enumerate(self.norm_context_list):
            ops = opinions[:,n_idx]
            for idx,op in enumerate(ops): 
                self.possible_agents[idx].opinion[norm_context] = np.random.uniform(0.5,1) if op == 1 else np.random.uniform(0,0.5)
                #self.possible_agents[idx].complete_information = True if self.baseline else False
        for _ag in self.possible_agents:
            if _ag.norm_context not in self.opinion_marginals:
                self.opinion_marginals[_ag.norm_context] = []
            self.opinion_marginals[_ag.norm_context].append(_ag.opinion[_ag.norm_context])
        self.opinion_marginals = [np.mean(self.opinion_marginals[k]) for k in self.norm_context_list]   
        ''' reconcile the true state based on the generated samples '''
        self.true_state = {n:self.opinion_marginals[idx] for idx,n in enumerate(self.norm_context_list)}
        
        weight_samples = utils.runif_in_simplex(10000,3)
        theta_samples = np.random.uniform(size=(10000,3))
        self.domain_samples =  np.hstack((weight_samples,theta_samples))
        
        
        
    def construct_correlation_from_opinions(self):
        num_contexts = self.opinions.shape[1]
        corr_mat = np.ones(shape=(num_contexts,num_contexts))
        for i in np.arange(num_contexts):
            for j in np.arange(num_contexts):
                if j>i:
                    _op_view = np.take(self.opinions,[i,j],axis=1)
                    _op_ct = Counter([tuple(_op_view[pl_idx,:]) for pl_idx in np.arange(self.num_players)])
                    corr_val = (_op_ct[(0.0,0.0)]+_op_ct[(1.0,1.0)])/np.sum(list(_op_ct.values()))
                    corr_mat[i,j] = corr_val
                    corr_mat[j,i] = corr_val
        return corr_mat   
    
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
        if not baseline:
            ''' Let the reward be inversely proportional to the opinion value extremity'''
            
            
            baseline_op_mean = np.mean([ag.opinion[ag.norm_context] for ag in self.agents])
            num_participation = len([ag for ag in self.agents if ag.action[0]!=-1])/self.num_players
            rewards = (num_participation-0.4)*2
            terminations = {agent.id: False for agent in self.agents}
    
            self.num_moves += 1
            env_truncation = self.num_moves >= self.NUM_ITERS
            truncations = {agent.id: env_truncation for agent in self.agents}
    
            ''' Observation is the next state, or the common prior change '''
            num_appr = len([ag.action[0] for ag in self.agents if ag.action[0]==1 and ag.action[0]!=-1])
            num_disappr = len([ag.action[0] for ag in self.agents if ag.action[0]==0 and ag.action[0]!=-1])
            if num_observation > 0:
                for nidx,n in enumerate(self.norm_context_list):
                    obs_op_vals = [ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1 and ag.norm_context==n]
                    theta_prime_rate = np.mean(obs_op_vals) if len(obs_op_vals) > 0 else None
                    obs_weight = len([ag.action[1] for ag in self.agents if ag.action[0]!=-1 and ag.norm_context==n])/num_observation
                    if theta_prime_rate is not None:
                        a_prime = theta_prime_rate*self.update_rate
                        b_prime =  self.update_rate-a_prime
                        self.common_prior[n] = (self.common_prior[n][0]+a_prime, self.common_prior[n][1]+b_prime)
                    self.common_prior_w[n] = self.common_prior_w[n]+(obs_weight*self.update_rate)
            
            observations = self.common_prior
            ''' now update the weights '''
            
            # typically there won't be any information in the infos, but there must
            # still be an entry for each agent
            infos = {agent.id: {} for agent in self.agents}
    
            if env_truncation:
                self.agents = []
    
            if self.render_mode == "human":
                self.render(iter_no)
            return observations, rewards, terminations, truncations, infos
        else:
            #num_appr = len([ag.action[0] for ag in self.agents if ag.action[0]==1 and ag.action[0]!=-1])
            #num_disappr = len([ag.action[0] for ag in self.agents if ag.action[0]==0 and ag.action[0]!=-1])
            '''
            if num_observation > 0:
                obs = np.array([(ag.action[2],ag.norm_context) for ag in self.agents if ag.action[0]!=-1])
                obs_mean = np.mean(obs[:,0].astype(np.float32))
                obs_samples = [(np.sum(np.random.choice([1,0],size=10,p=[op_val,1-op_val])),) for op_val in obs[:,0].astype(np.float32)]
                desc_bel = env.context_weights_baseline @ env.common_prior_means_baseline_asarray.T
                
                plt.figure()
                plt.title(str(desc_bel))
                plt.hist(obs[:,0].astype(np.float32), bins=20)
                plt.show()
                
                xs = [(s[0],10-s[0]) for s in obs_samples]
                thetas = np.array([[self.common_prior_means_baseline_asarray[n], 1-self.common_prior_means_baseline_asarray[n]] for n in np.arange(env.norm_context_list_len)])
                i, thetas, ws = utils.em(xs, thetas)
                ws = np.mean(ws,axis=1)
                for nidx,n in enumerate(self.norm_context_list):
                    a_prime = thetas[nidx,0]*self.update_rate
                    b_prime =  self.update_rate-a_prime
                    self.prior_baseline[n] = (self.prior_baseline[n][0]+a_prime, self.prior_baseline[n][1]+b_prime)
                    self.prior_baseline_w[n] = self.prior_baseline_w[n]+(ws[nidx]*self.update_rate)
            '''
            if num_observation > 0:
                for nidx,n in enumerate(self.norm_context_list):
                    obs_op_vals = [ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1]
                    theta_prime_rate = np.mean(obs_op_vals) if len(obs_op_vals) > 0 else None
                    #obs_weight = len([ag.action[1] for ag in self.agents if ag.action[0]!=-1 and ag.norm_context==n])/num_observation
                    if theta_prime_rate is not None:
                        a_prime = theta_prime_rate*self.update_rate
                        b_prime =  self.update_rate-a_prime
                        #self.prior_baseline[n] = (self.prior_baseline[n][0]+a_prime, self.prior_baseline[n][1]+b_prime)
                        for ag in self.agents:
                            if ag.norm_context==n:
                                ag.desc_bel = (ag.desc_bel[0]+a_prime, ag.desc_bel[1]+b_prime)
                    #self.prior_baseline_w[n] = self.prior_baseline_w[n]+(obs_weight*self.update_rate)
                    
                    
    def _posterior_calc(self,all_samples,steward_distr):
        if all_samples is None:
            self.weight_samples = utils.runif_in_simplex(10000,4)
            self.theta_samples = np.random.uniform(size=(10000,4))
            self.all_samples = np.hstack((self.weight_samples,self.theta_samples))
        constr_f = lambda x,signal_distr : utils.Gaussian_plateu_distribution(0,0.1,self.normal_constr_sd).pdf(abs(x-signal_distr))
        def calc_pos(_state_vec):
            return constr_f(_state_vec[:4]@_state_vec[4:].T,steward_distr.w@steward_distr.t.T)*(utils.dirichlet_pdf(x=_state_vec[:4], alpha = [self.common_prior_w[n] for n in self.norm_context_list])*math.prod([utils.beta_pdf(_state_vec[nidx], self.common_prior[n][0], self.common_prior[n][1]) for nidx,n in enumerate(self.norm_context_list)]))
        posteriors = np.apply_along_axis(calc_pos, 1, self.all_samples)
        return posteriors
    
    def generate_posteriors(self,signal_distribution):
        '''
            This method updates the posterior for the population (posterior over the weights and rates) based on the signal dristribution (weights and rates).
            Since signal distribution is a Bernoulli, we can get individual realizations of 0 and 1 separately, and then take the expectation.
        '''
        exp_signal_appr_rate = signal_distribution.w@signal_distribution.t.T
        posteriors_part1 = self._posterior_calc(self.all_samples if hasattr(self, 'all_samples') else None, signal_distribution)
        ''' appr _signal'''
        appr_post = posteriors_part1 * exp_signal_appr_rate
        appr_post_marginal = np.mean(appr_post)*(1/math.factorial(len(self.norm_context_list))) # Multiplied by 1/n! for the volume of the simples and everything else lies in [0,1]
        appr_post_by_marginal = appr_post/appr_post_marginal
        ''' disappr signal '''
        disappr_post = posteriors_part1 * (1-exp_signal_appr_rate)
        disappr_post_marginal = np.mean(disappr_post)*(1/math.factorial(len(self.norm_context_list))) # Multiplied by 1 just to remind that the volume of the space is still 1 since everything lies in [0,1]
        disappr_post_by_marginal = disappr_post/disappr_post_marginal
        ''' approvals and diapproval signals are going to come based on the signal distribution proportion due to commitment constraint '''
        ''' therefore, calculate the expectations of the posterior function '''
        exp_posterior_full_distr_samples = exp_signal_appr_rate*appr_post_by_marginal + (1-exp_signal_appr_rate)*disappr_post_by_marginal
        exp_posterior_full_distr_samples = np.reshape(exp_posterior_full_distr_samples,newshape=(exp_posterior_full_distr_samples.shape[0],1))
        
        ''' the posterior update will be different for different models of behaviour.'''
        ''' this is the common expected posterior approval rate, which can be used as a descriptive belief'''
        checksum = np.sum(exp_posterior_full_distr_samples)
        exp_posterior_full_distr_samples = exp_posterior_full_distr_samples/checksum
        ''' just for better readability E[\sigma (w_i.\theta_i)] '''
        x1 = np.multiply(env.all_samples[:,:4],env.all_samples[:,4:])
        x2 = np.sum(x1, axis=1)
        x3 = np.multiply(np.reshape(x2,newshape=(x2.shape[0],1)),exp_posterior_full_distr_samples)
        self.common_posterior_as_single_expectation = np.sum(x3)
        self.exp_weight_distr_posterior = np.sum(np.multiply(env.all_samples[:,:4],exp_posterior_full_distr_samples), axis=0)
        self.exp_theta_distr_mean = np.sum(np.multiply(env.all_samples[:,4:],exp_posterior_full_distr_samples), axis=0)
        self.exp_weight_distr_posterior = {n:self.exp_weight_distr_posterior[nidx] for nidx,n in enumerate(self.norm_context_list)}
        self.exp_theta_distr_mean = {n:self.exp_theta_distr_mean[nidx] for nidx,n in enumerate(self.norm_context_list)}
        
        x1, x2, x3 = None, None, None
        ''' now the hard part. updating the posterior for the full distribution.'''
        ''' the action needs two things separately:
            1. the weight distribution
            2. the correlation matrix for the joint distribution of thetas.
        '''
        '''
        exp_weight_distr = np.sum(np.multiply(env.all_samples[:,:4],exp_posterior_full_distr_samples), axis=0)
        exp_theta_distr_mean = np.sum(np.multiply(env.all_samples[:,4:],exp_posterior_full_distr_samples), axis=0)
        cov_matrix, sd_matrix = np.zeros(shape=(len(env.norm_context_list),len(env.norm_context_list))), np.zeros(shape=(len(env.norm_context_list),1))
        for i in np.arange(sd_matrix.shape[0]):
            x1 = env.all_samples[:,i]-exp_theta_distr_mean[i]
            x2 = np.square(x1)
            x3 = np.multiply(np.reshape(x2,newshape=(x2.shape[0],1)),exp_posterior_full_distr_samples)
            sd_matrix[i] = np.sum(np.sqrt(x3), axis=0)
        with np.nditer(cov_matrix, flags=['multi_index']) as it:
            for x in it:
                i,j = it.multi_index
                x1 = env.all_samples[:,i]-exp_theta_distr_mean[i]
                x2 = env.all_samples[:,j]-exp_theta_distr_mean[j]
                x3 = np.multiply(np.reshape(x1,newshape=(x1.shape[0],1)), np.reshape(x2,newshape=(x2.shape[0],1)))
                x3 = np.multiply(np.reshape(x3,newshape=(x3.shape[0],1)),exp_posterior_full_distr_samples)
                cov_matrix[i,j] = np.sum(x3, axis=0)
        corr_matrix = np.empty_like(cov_matrix)
        with np.nditer(corr_matrix, flags=['multi_index']) as it:
            for x in it:  
                i,j = it.multi_index
                corr_matrix[i,j] = cov_matrix[i,j]/(sd_matrix[i]*sd_matrix[j])
        x1, x2, x3 = None, None, None      
        ''' 
        ''' let it be, assuming the correlation matrix stays constant (because  it is more reasonable as correlation gets lost very fast with posterior updates)'''
        ''' so we just need the new expected weights that '''      
        return exp_posterior_full_distr_samples
    
    
    
    @property
    def context_weights_baseline(self):
        return np.array([self.norm_contexts_distr[n]/np.sum(list(self.norm_contexts_distr.values())) for n in self.norm_context_list])
    
    @property
    def common_prior_means_baseline_asarray(self):
        return np.array([self.prior_baseline[n][0]/(self.prior_baseline[n][0]+self.prior_baseline[n][1]) for n in self.norm_context_list])
    
    
    @property
    def context_weights(self):
        return np.array([self.common_prior_w[n] for n in self.norm_context_list])
    
        
    @property
    def common_prior_means_asarray(self):   
        return np.array([self.common_prior[n][0]/(self.common_prior[n][0]+self.common_prior[n][1]) for n in self.norm_context_list])
        
        
class Player():
    
    def __init__(self,id):
        self.id = id
        self.payoff_tol = constants.payoff_tol
        self.opinion = dict()
        self.opinion_val = dict()
        
        '''
        if real_p:
            self.shadow_player = Player(-id,False)
        '''
        
        self.total_reward = 0
        self.total_participation = 0
    
    
    
    def act(self, env, baseline, norm_context):
        w_t = env.w_t
        theta_t = {k:v[0]/sum(v) for k,v in env.theta_t.items()}
        
        ''' bels -1,2,3,4 are binary ordering with context_same, opinion_same'''
        prob_diff_context = 1-w_t[self.norm_context]
        checksame = lambda o1,o2: True if (o1>=0.5 and o2 >= 0.5) or (o1<0.5 and o2 < 0.5) else False
        diff_context_diff_opinion = sum([(1-o)*w_t[n] if self.opinion[self.norm_context] >=0.5 else o*w_t[n] for n,o in theta_t.items() if n!=self.norm_context])
        diff_context_same_opinion = sum([o*w_t[n] if self.opinion[self.norm_context] >=0.5 else (1-o)*w_t[n] for n,o in theta_t.items() if n!=self.norm_context])
        same_context_same_opinion = theta_t[self.norm_context]*w_t[self.norm_context] if self.opinion[self.norm_context] >=0.5 else (1-theta_t[self.norm_context])*w_t[self.norm_context]
        same_context_diff_opinion = (1-theta_t[self.norm_context])*w_t[self.norm_context] if self.opinion[self.norm_context] >=0.5 else theta_t[self.norm_context]*w_t[self.norm_context]
        all_bels = [diff_context_diff_opinion,diff_context_same_opinion,same_context_diff_opinion,same_context_same_opinion]
        f = sum(all_bels)
        f=1
        
    
    def _util(self,desc_bel_op,corr_val_ctx):
            '''Maps from -1,1 to 0,1'''
            scaled_u_corr = lambda u : (u+1)/2 
            return desc_bel_op*scaled_u_corr(corr_val_ctx)
        
    def act_context_misinterpretation(self,env,baseline,norm_context):
        util = lambda op : op if op >= 0.5 else (1-op)
        u_bar = env.security_util
        if not baseline:
            assert norm_context==self.norm_context, 'norm context check failed'
            desc_bel = env.common_posterior_as_single_expectation
            desc_bel_op = desc_bel if self.opinion[self.norm_context] >= 0.5 else 1-desc_bel
            self_norm_index = env.norm_context_list.index(self.norm_context)
            expected_corr = np.array([env.exp_weight_distr_posterior[n] for n in env.norm_context_list]) @ env.corr_mat[self_norm_index,:].T
            util_val = self._util(desc_bel_op,expected_corr)
            if util_val < u_bar:
                self.action_code_baseline = -1
                self.action_util_baseline = u_bar
            else:
                self.action_code_baseline = 1 if self.opinion[self.norm_context] >= 0.5 else 0
                self.action_util_baseline = desc_bel_op*expected_corr
                    
            
            self.action =(self.action_code_baseline,self.action_util_baseline,self.opinion[self.norm_context])
        else:
            
            desc_bel = np.mean(self.desc_bel)
            #desc_bel = np.sum([x[0]*x[1] for x in zip(env.context_weights_baseline,env.common_prior_means_baseline_asarray)])
            #desc_bel = env.common_prior_means_baseline_asarray[env.norm_context_list.index(self.norm_context)]
            desc_bel_op = desc_bel if self.opinion[self.norm_context] >= 0.5 else 1-desc_bel
            #op_val = self.opinion[self.norm_context] if self.opinion[self.norm_context] >= 0.5 else 1-self.opinion[self.norm_context]
            self_norm_index = env.norm_context_list.index(self.norm_context)
            #expected_corr = env.context_weights_baseline @ env.corr_mat[self_norm_index,:].T
            expected_corr = np.sum([x[0]*x[1] for x in zip(env.context_weights_baseline , env.corr_mat[self_norm_index,:])])
            #assert norm_context==self.norm_context, 'norm context check failed'
            #util_val = _util(desc_bel_op,expected_corr)
            util_val = desc_bel_op*util(self.opinion[self.norm_context])
            if util_val < u_bar:
                self.action_code_baseline = -1
                self.action_util_baseline = u_bar
            else:
                self.action_code_baseline = 1 if self.opinion[self.norm_context] >= 0.5 else 0
                self.action_util_baseline = util_val
            '''        
            if (0.4 < self.opinion[self.norm_context] < 0.5) and self.action_code_baseline != -1:
                f=1
            '''
            self.action =(self.action_code_baseline,self.action_util_baseline,self.opinion[self.norm_context])
            
        
    def act_self_ref(self, env,baseline,norm_context):
        ''' This is a petting zoo framework method '''
        '''
        The required information are:
        belief about the distribution on opinion (already should be in player object)
        self opinion (already should have been initialized)
        '''
        util = lambda op : op if op >= 0.5 else (1-op)
        u_bar = env.security_util
        op = self.opinion[self.norm_context]
        
        if not baseline:
            assert norm_context==self.norm_context, 'norm context check failed'
            theta_posterior = env.common_posterior_as_single_expectation
            disappr_bar_posterior = 1 - (u_bar/(1-theta_posterior))
            appr_bar_posterior = u_bar/theta_posterior
            
            if op >= 0.5:
                if op < appr_bar_posterior:
                    self.action_code = -1
                    self.action_util = u_bar
                else:
                    self.action_code = 1
                    self.action_util = util(op)
            else:
                if op > disappr_bar_posterior:
                    self.action_code = -1
                    self.action_util = u_bar
                else:
                    self.action_code = 0
                    self.action_util = util(op)
            self.action =(self.action_code,self.action_util,self.opinion[self.norm_context])
        else:
            assert norm_context==self.norm_context, 'norm context check failed'
            theta_baseline = env.context_weights_baseline @ env.prior_baseline.T
            disappr_bar_baseline = 1 - (u_bar/(1-theta_baseline))
            appr_bar_baseline = u_bar/theta_baseline
            
            if op >= 0.5:
                if op < appr_bar_baseline:
                    self.action_code_baseline = -1
                    self.action_util_baseline = u_bar
                else:
                    self.action_code_baseline = 1
                    self.action_util_baseline = util(op)
            else:
                if op > disappr_bar_baseline:
                    self.action_code_baseline = -1
                    self.action_util_baseline = u_bar
                else:
                    self.action_code_baseline = 0
                    self.action_util_baseline = util(op)
            self.action =(self.action_code_baseline,self.action_util_baseline,self.opinion[self.norm_context])
        
        return self.action





if __name__ == "__main__":
    """ ENV SETUP """
    baseline_only_run = True
    #,[0.2,0.3,0.4,0.1]
    lstb,lst = [],[]
    p_list = [1]
    for runid, prior_list in enumerate(p_list):
        weight_list = [0.5,0.3,0.2]
        state_evolution_baseline, state_evolution = dict(), dict()
        summary_statistic_baseline = {'participation':{},'community_correlation':{},'mean_theta':{}}
        summary_statistic = {k:v for k,v in summary_statistic_baseline.items()}
        for batch_num in np.arange(100):
            attr_dict = {'norm_contexts_distr': {_n:weight_list[_nidx] for _nidx,_n in enumerate(['n1','n2','n3'])},
                        'true_state_distr_params':{'n1':(2,3.3),'n2':(3,4),'n3':(3,1.3)}
                        }
            env = heterogenous_parallel_env(render_mode='human',attr_dict=attr_dict)
            env.w_t = {'n1':0.5,'n2':0.3,'n3':0.2}
            env.theta_t = {'n1':(2,3.3),'n2':(3,4),'n3':(3,1.3)}
            ''' Check that every norm context has at least one agent '''
            if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
                raise Exception()
            env.reset()
            env.no_print = True
            env.NUM_ITERS = 100
            common_prior_var = 0.1
            env.common_prior = {k:utils.est_beta_from_mu_sigma(mu=v+np.random.normal(scale=common_prior_var), sigma=common_prior_var) for k,v in env.true_state.items()}
            for ag in env.possible_agents:
                ag.desc_bel = env.common_prior[ag.norm_context]
            env.common_prior = {k:utils.est_beta_from_mu_sigma(mu=v, sigma=common_prior_var) for k,v in env.true_state.items()}
            env.common_prior_w = {k:int(v*10) for k,v in attr_dict['norm_contexts_distr'].items()}
            env.prior_baseline = {k:v for k,v in env.common_prior.items()}
            #env.prior_baseline_w = {k:int(v*10) for k,v in attr_dict['norm_contexts_distr'].items()}
            #env.prior_baseline_w = {'n1':0.1,'n2':0.1,'n3':0.8}
            env.normal_constr_sd = 0.3
            #env.constraining_distribution = utils.Gaussian_plateu_distribution(env.common_prior[0]/sum(env.common_prior),.01,.3)
            #env.constraining_distribution = utils.Gaussian_plateu_distribution(.3,.01,.3)
            dataset = []
            for k,v in summary_statistic_baseline.items():
                if len(v) == 0:
                    summary_statistic_baseline[k] = {n:[] for n in env.norm_context_list}
            for k,v in summary_statistic.items():
                if len(v) == 0:
                    summary_statistic[k] = {n:[] for n in env.norm_context_list}
            
            for i in np.arange(100):
                print(runid,batch_num,i,':', env.context_weights_baseline @ env.common_prior_means_baseline_asarray.T)
                
                actions = dict()
                for norm_context in env.norm_context_list:
                    #curr_state = env.common_prior[norm_context][0]/sum(env.common_prior[norm_context])
                    #baseline_bels_mean = env.prior_baseline[norm_context][0]/sum(env.prior_baseline[norm_context])
                    actions.update({agent.id:agent.act(env,baseline=True,norm_context=norm_context) for agent in env.possible_agents if agent.norm_context==norm_context })
                    #actions.update({agent.id:agent.act(env,run_type='self-ref',baseline=True,norm_context=norm_context) for agent in env.possible_agents if agent.norm_context==norm_context })
                env.step(actions,i,baseline=True)
                for _n in env.norm_context_list:
                    ''' desc beliefs for all agents within a context is the same, so just grab one.'''
                    for ag in env.possible_agents:
                        if ag.norm_context == _n:
                            env.prior_baseline[_n] = ag.desc_bel
                            break
                ''' common prior is updated based on the action observations for both w and t if baseline is False '''
                for norm_context in env.norm_context_list:
                    if norm_context not in state_evolution_baseline:
                        state_evolution_baseline[norm_context] = dict()
                    if i not in  state_evolution_baseline[norm_context]:
                        state_evolution_baseline[norm_context][i] = []
                    state_evolution_baseline[norm_context][i].append(env.prior_baseline[norm_context][0]/sum(env.prior_baseline[norm_context]))
                for nidx,n in enumerate(env.norm_context_list):
                    part = len([ag for ag in env.possible_agents if ag.norm_context==n and ag.action[0]!=-1])/len([ag for ag in env.possible_agents if ag.norm_context==n])
                    summary_statistic_baseline['participation'][n].append(part)
                    corr_val = np.sum(env.corr_mat[nidx,:])
                    summary_statistic_baseline['community_correlation'][n].append(corr_val)
                    mean_theta = env.true_state[n]
                    summary_statistic_baseline['mean_theta'][n].append(mean_theta)
                if not baseline_only_run:
                    ''' with baseline=False'''
                    signal_distribution = namedtuple('signal_distribution','w t')
                    steward_signal_distribution = signal_distribution(np.array([0.2,0.3,0.4,0.1]), np.array([0.45,0.45,0.55,0.45]))
                    ''' the posterior gets updated inside this '''
                    exp_posterior_full_distr_samples = env.generate_posteriors(steward_signal_distribution)
                    ''' act is based on the new posterior acting as prior if baseline is False'''
                    actions = dict()
                    for norm_context in env.norm_context_list:
                        actions.update({agent.id:agent.act_context_misinterpretation(env,baseline=False,norm_context=norm_context) for agent in env.possible_agents if agent.norm_context==norm_context })
                        #actions.update({agent.id:agent.act(env,run_type='self-ref',baseline=True,norm_context=norm_context) for agent in env.possible_agents if agent.norm_context==norm_context })
                    observations, rewards, terminations, truncations, infos = env.step(actions,i,baseline=False)
                    for norm_context in env.norm_context_list:
                        if norm_context not in state_evolution:
                            state_evolution[norm_context] = dict()
                        if i not in  state_evolution[norm_context]:
                            state_evolution[norm_context][i] = []
                        state_evolution[norm_context][i].append(env.common_prior[norm_context][0]/sum(env.common_prior[norm_context]))
                    for nidx,n in enumerate(env.norm_context_list):
                        part = len([ag for ag in env.possible_agents if ag.norm_context==n and ag.action[0]!=-1])/len([ag for ag in env.possible_agents if ag.norm_context==n])
                        summary_statistic['participation'][n].append(part)
                    
            
        for k,v in summary_statistic_baseline.items():
            print('------',k,'-------')
            for n,val in v.items():
                print(n,':',np.mean(val) if len(val)>0 else 'None')
        if not baseline_only_run:
            print('----with signal summary statistic-----')
            for k,v in summary_statistic.items():
                print('------',k,'-------')
                for n,val in v.items():
                    print(n,':',np.mean(val) if len(val)>0 else 'None')
        cols = ['time', 'belief','norm','model','run_id']
        
        for norm,norm_data in state_evolution_baseline.items():
            for k,v in norm_data.items():
                for _v in v:
                    lstb.append([k,_v,norm,'no signal',runid])
        if not baseline_only_run:
            for norm,norm_data in state_evolution.items():
                for k,v in norm_data.items():
                    for _v in v:
                        lst.append([k,_v,norm,'signal',runid])
    df = pd.DataFrame(lst, columns=cols)
    df['norm'] = df['norm'].map({k:k+' ('+str(attr_dict['norm_contexts_distr'][k])+','+str(beta.mean(a=attr_dict['true_state_distr_params'][k][0],b=attr_dict['true_state_distr_params'][k][1]))+')' for k in env.norm_context_list})
    dfb = pd.DataFrame(lstb, columns=cols)
    dfb['norm'] = dfb['norm'].map({k:k+' ('+str(attr_dict['norm_contexts_distr'][k])+','+str(beta.mean(a=attr_dict['true_state_distr_params'][k][0],b=attr_dict['true_state_distr_params'][k][1]))+')' for k in env.norm_context_list})
    
    sns.set_theme(style="darkgrid")
    '''
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')   
    ax.scatter([x[0] for x in dataset], [x[1] for x in dataset], [x[2] for x in dataset])
    ax.set_xlabel('state as prior mean')
    ax.set_ylabel('signal distr theta')
    ax.set_zlabel('Reward')
    '''
    coloridx = {0:'black',1:'blue'}
    if not baseline_only_run:
        fig, ax = plt.subplots(2)
    else:
        fig, ax = plt.subplots(1)
    
    if not baseline_only_run:
        g1 = sns.lineplot(hue="norm", x="time", y="belief", ci="sd", estimator='mean', data=dfb, ax=ax[0])
        g2 = sns.lineplot(hue="norm", x="time", y="belief", ci="sd", estimator='mean', data=df, ax=ax[1])
    else:
        for runid in np.arange(len(p_list)):
            _dfb = dfb.loc[dfb['run_id'] == runid]
            sns.lineplot(hue="norm", x="time", y="belief", ci="sd", estimator='mean', data=_dfb, palette=sns.color_palette([coloridx[runid]], 3), ax=ax)
    plt.tight_layout()
    plt.show()