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

NUM_ITERS = 10

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
    
    def get_act_for_apr_theta(self,apr_theta):
        if self.opinion[self.norm_context] >= 0.5:
            op = 1
        else:
            op = 0
        util = lambda op : op if op > 0.5 else (1-op)
        bel_op = apr_theta if op==1 else 1-apr_theta
        prob_of_N = (bel_op*util(self.opinion[self.norm_context]))/self.payoff_tol
        if prob_of_N > 1:
            return self.opinion[self.norm_context]
        else:
            return -1
    
    def act(self, env, run_type):
        if run_type in ['baseline','self-ref']:
            return self.act_self_ref(env)
        else:
            if run_type in ['community-ref']:
                return self.act_community_ref_bne(env)
            else:
                return self.act_moderator_ref_bne(env)
        
        
    def act_self_ref(self, env):
        ''' This is a petting zoo framework method '''
        '''
        The required information are:
        belief about the distribution on opinion (already should be in player object)
        self opinion (already should have been initialized)
        '''
        util = lambda op : op if op > 0.5 else (1-op)
        if self.opinion[self.norm_context] >= 0.5:
            op = 1
        else:
            op = 0
        if not hasattr(self, 'belief'):
            self.belief = utils.get_priors_from_true(env.opinion_marginals)
            self.belief = {n:self.belief[nidx] for nidx,n in enumerate(env.norm_context_list)}
            self.belief = {self.norm_context:self.belief[self.norm_context]}
        
        ''' Update the beliefs from the previous observations '''
        if env.observations is not None:
            norm_consideration_list = [self.norm_context]
            obs = env.complete_observations
            obs_samples = [(np.sum(np.random.choice([1,0],size=10,p=[op_val,1-op_val])),) for op_val in obs]
            xs = [(s[0],10-s[0]) for s in obs_samples]
            thetas = np.array([[self.belief[n], 1-self.belief[n]] for n in norm_consideration_list])
            i, thetas, ws = utils.em(xs, thetas)
            ws = np.mean(ws,axis=1)
            self.belief = {n:thetas[nidx][0] for nidx,n in enumerate(norm_consideration_list)}
        bel_op = self.belief[self.norm_context] if op == 1 else 1-self.belief[self.norm_context]
        prob_of_N = (bel_op*util(self.opinion[self.norm_context]))/self.payoff_tol
        if prob_of_N > 1:
            self.action_util = util(self.opinion[self.norm_context])
            self.action_code = op
        else:
            self.action_code = -1
            self.action_util = self.payoff_tol
        self.action =(self.action_code,self.action_util,self.opinion[self.norm_context])
        return self.action
    
    def act_community_ref_bne(self, env):
        ''' This is a petting zoo framework method '''
        '''
        The required information are:
        belief about the distribution on opinion (already should be in player object)
        self opinion (already should have been initialized)
        '''
        util = lambda op : op if op > 0.5 else (1-op)
        if self.opinion[self.norm_context] >= 0.5:
            op = 1
        else:
            op = 0
        if not hasattr(self, 'belief'):
            self.belief = utils.get_priors_from_true(env.opinion_marginals)
            self.belief = {n:self.belief[nidx] for nidx,n in enumerate(env.norm_context_list)}
            self.norm_prop = {n:0.25 for nidx,n in enumerate(env.norm_context_list)}
        ''' Update the beliefs from the previous observations '''
        if env.observations is not None:
            ''' Observations are common and public, so no need for agent index'''
            
            obs = env.complete_observations if not self.complete_information else env.get_complete_information()
            
            if self.complete_information:
                obs_samples = [(np.sum(np.random.choice([1,0],size=10,p=[op_val,1-op_val])),) for op_val in [float(x[0]) for x in obs]]
                obs_norm_contexts = [x[1] for x in obs]
                xs = [(s[0],10-s[0]) for s in obs_samples]
                updated_thetas,updated_props = utils.mle(zip(xs,obs_norm_contexts))
                self.belief.update(updated_thetas)
                self.norm_prop.update(updated_props)
            else:
                obs_samples = [(np.sum(np.random.choice([1,0],size=10,p=[op_val,1-op_val])),) for op_val in obs]
                xs = [(s[0],10-s[0]) for s in obs_samples]
                thetas = np.array([[self.belief[n], 1-self.belief[n]] for n in env.norm_context_list])
                i, thetas, ws = utils.em(xs, thetas)
                ws = np.mean(ws,axis=1)
                self.belief = {n:thetas[nidx][0] for nidx,n in enumerate(env.norm_context_list)}
                self.norm_prop = {n:ws[nidx] for nidx,n in enumerate(env.norm_context_list)}
            
            #self.belief = {n:env.opinion_marginals[nidx] for nidx,n in enumerate(env.norm_context_list)}
            #self.norm_prop = {n:env.norm_contexts_distr[n] for nidx,n in enumerate(env.norm_context_list)}
        bel_op_all_norms = {n:self.belief[n] if op == 1 else 1-self.belief[n] for nidx,n in enumerate(['n1','n2','n3','n4'])}
        exp_n = 0
        for n,bel_for_n in bel_op_all_norms.items():
            exp_prob_n_for_bel = self.norm_prop[n]*(bel_for_n*util(self.opinion[self.norm_context]))/self.payoff_tol
            exp_n += exp_prob_n_for_bel 
        if exp_n > 1:
            self.action_util = util(self.opinion[self.norm_context])
            self.action_code = op
        else:
            self.action_code = -1
            self.action_util = self.payoff_tol
        self.action =(self.action_code,self.action_util,self.opinion[self.norm_context])
        return self.action
    
    def act_moderator_ref_bne(self, env):
        ''' This is a petting zoo framework method '''
        '''
        The required information are:
        belief about the distribution on opinion (already should be in player object)
        self opinion (already should have been initialized)
        '''
        util = lambda op : op if op > 0.5 else (1-op)
        if self.opinion[self.norm_context] >= 0.5:
            op = 1
        else:
            op = 0
        if env.moderator_context_signal!=self.norm_context:
            norm_consideration_list = [env.moderator_context_signal,self.norm_context]
        else:
            norm_consideration_list = [self.norm_context]
        if not hasattr(self, 'belief'):
            self.belief = utils.get_priors_from_true(env.opinion_marginals)
            self.belief = {n:self.belief[nidx] for nidx,n in enumerate(env.norm_context_list)}
            self.belief = {n:self.belief[n] for nidx,n in enumerate(norm_consideration_list)}
            self.norm_prop = {n:1/len(norm_consideration_list) for nidx,n in enumerate(norm_consideration_list)}
            
        ''' Update the beliefs from the previous observations '''
        if env.observations is not None:
            ''' Observations are common and public, so no need for agent index'''
            
            obs = env.complete_observations
            obs_samples = [(np.sum(np.random.choice([1,0],size=10,p=[op_val,1-op_val])),) for op_val in obs]
            xs = [(s[0],10-s[0]) for s in obs_samples]
            thetas = np.array([[self.belief[n], 1-self.belief[n]] for n in norm_consideration_list])
            i, thetas, ws = utils.em(xs, thetas)
            ws = np.mean(ws,axis=1)
            self.belief = {n:thetas[nidx][0] for nidx,n in enumerate(norm_consideration_list)}
            self.norm_prop = {n:ws[nidx] for nidx,n in enumerate(norm_consideration_list)}
            
            #self.belief = {n:env.opinion_marginals[nidx] for nidx,n in enumerate(env.norm_context_list)}
            #self.norm_prop = {n:env.norm_contexts_distr[n] for nidx,n in enumerate(env.norm_context_list)}
        bel_op_all_norms = {n:self.belief[n] if op == 1 else 1-self.belief[n] for nidx,n in enumerate(norm_consideration_list)}
        exp_n = 0
        for n,bel_for_n in bel_op_all_norms.items():
            exp_prob_n_for_bel = self.norm_prop[n]*(bel_for_n*util(self.opinion[n]))/self.payoff_tol
            exp_n += exp_prob_n_for_bel 
        if exp_n > 1:
            if len(norm_consideration_list) > 1:
                optimal_context_for_player = self.norm_context if util(self.opinion[self.norm_context])>=util(self.opinion[env.moderator_context_signal]) else env.moderator_context_signal
            else:
                optimal_context_for_player = self.norm_context
            self.action_util = util(self.opinion[optimal_context_for_player])
            self.action_code = 1 if self.opinion[optimal_context_for_player] >= 0.5 else 0
        else:
            self.action_code = -1
            self.action_util = self.payoff_tol
        self.action =(self.action_code,self.action_util,self.opinion[self.norm_context])
        return self.action
        
    def _setup_player(self,r_info):
        self.appr_val = np.random.beta(self.opinion_alpha, self.opinion_beta)
        self.shadow_player.appr_val = np.random.beta(self.prior_belief[0], self.prior_belief[1])
            
        self.belief = r_info.prior_belief[0], r_info.prior_belief[1]
        self.belief_alt = r_info.prior_belief_alt[0], r_info.prior_belief_alt[1]
        
        if self.appr_val > .5:
            self.opinion = 'A'
            self.opinion_val = self.appr_val
        elif self.appr_val < .5:
            self.opinion = 'D'
            self.opinion_val = 1-self.appr_val
        else:
            self.opinion = np.random.choice(['A','D'])
            self.opinion_val = self.appr_val
        
        if self.shadow_player.appr_val > .5:
            self.shadow_player.opinion = 'A'
            self.shadow_player.opinion_val = self.shadow_player.appr_val
        elif self.shadow_player.appr_val < .5:
            self.shadow_player.opinion = 'D'
            self.shadow_player.opinion_val = 1-self.shadow_player.appr_val
        else:
            self.shadow_player.opinion = np.random.choice(['A','D'])
            self.shadow_player.opinion_val = self.shadow_player.appr_val
        self.oth_opinion_bel = self.shadow_player.opinion
        self.shadow_player.oth_opinion_bel = self.shadow_player.opinion
        '''
        people with minority opinion having higher minority opinion risk tolerance
        people with majority opinion having lower majority opinion risk tolerance
        '''
        if constants.risk_tol is None:
            if self.opinion != self.shadow_player.opinion:
                self.risk_tol = .9*self.opinion_val
            else:
                self.risk_tol = .9*self.opinion_val
        else:
            self.risk_tol = constants.risk_tol
        if constants.payoff_tol is not None:
            self.payoff_tol = constants.payoff_tol
            
        
    
    
        
    def choose_action(self,payoff_dict):
        #effort_fun = lambda x : 1-x
        effort_fun = lambda x : entropy(x,1-x)
        if self.opinion == self.oth_opinion_bel:
                risk_of_opinion = utils.get_act_risk(payoff_dict, 0, self.opinion)
                if constants.op_mode == 'payoff_based':
                    effective_op_val = self.opinion_val - effort_fun(self.opinion_val)
                    if effective_op_val < self.payoff_tol:
                        self.action = 'N'
                    else:
                        self.action = self.opinion
                
                if constants.op_mode == 'risk_based':
                    if risk_of_opinion > self.risk_tol:
                        self.action = 'N'
                    else:
                        self.action = self.opinion
        else:
            pl_not_op = utils.get_oth_opinion(self.opinion)
            payoff_not_opinion = payoff_dict[(pl_not_op,pl_not_op)][0]
            payoff_opinion = payoff_dict[(self.opinion,self.opinion)][0]
            risk_of_opinion = utils.get_act_risk(payoff_dict, 0 , self.opinion)
            if constants.op_mode == 'payoff_based':
                effective_op_val = self.opinion_val - effort_fun(self.opinion_val)
                if effective_op_val < self.payoff_tol:
                    self.action = 'N'
                else:
                    self.action = self.opinion
            
            if constants.op_mode == 'risk_based':
                if risk_of_opinion > self.risk_tol:
                    self.action = 'N'
                else:
                    self.action = self.opinion
    
    def choose_action_simplified(self,prior_belief):
        effort_fun = lambda x : 0.5*entropy([x,1-x])
        effective_op_val = self.opinion_val - effort_fun(self.opinion_val)
        
        bel_op = sum(prior_belief[:2])/sum(prior_belief) if self.opinion == 'D' else sum(prior_belief[2:])/sum(prior_belief)
        op_selection_ratio = (bel_op*(effective_op_val))/self.payoff_tol
        if op_selection_ratio > 1:
            self.action = self.opinion_expanded
        else:
            self.action = 'N'
    
    def add_regrets(self,act_distribution):
        other_action = 'N' if self.action != 'N' else self.opinion
        obs_opinion_support = sum(act_distribution[:2])/sum(act_distribution) if self.opinion == 'D' else sum(act_distribution[2:])/sum(act_distribution)
        if other_action == 'N':
            curr_payoff = self.opinion_val*obs_opinion_support
            self.regret_map = {other_action:max(0,self.payoff_tol-curr_payoff)}
        else:
            curr_payoff = self.payoff_tol
            regret_payoff = obs_opinion_support*self.opinion_val
            self.regret_map = {other_action:max(0,regret_payoff-curr_payoff)}
        self.regret_map[self.action[-1]] = 0
    
class ModeratorAgent():
    
    def __init__(self,moderator_action = None):
        if moderator_action is not None:
            self.moderator_action = moderator_action            

class RunInfo():
    
    def __init__(self,iter):
        self.iter = iter

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env



class parallel_env(ParallelEnv):
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
        self.norm_context_list = ['n1','n2','n3','n4']
        constants.payoff_tol = 0.3
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
        if not hasattr(self, 'norm_contexts_distr'):
            self.norm_contexts_distr = {'n1':0.4,'n2':0.2,'n3':0.3,'n4':0.1} 
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
        #opinions = np.genfromtxt('..//r_scripts//samples.csv', delimiter=',')
        if not hasattr(self, 'ref_op_marginal_theta'):
            self.ref_op_marginal_theta = [0.3,0.61,0.58,0.8]
        if not hasattr(self, 'corr_mat'):
            if not hasattr(self, 'corr_idx'):
                self.opinions,self.corr_mat,self.mutual_info_mat = utils.generate_samples(self.ref_op_marginal_theta,(0,None))
            else:
                self.corr_mat_grid = utils.generate_corr_mat_grid(self.ref_op_marginal_theta, (0,self.corr_idx))
                self.opinions,self.corr_mat,self.mutual_info_mat = utils.generate_grid_samples(self.corr_mat_grid,self.ref_op_marginal_theta,(0,self.corr_idx))
        self.corr_mat_ref_sum = np.sum(self.corr_mat[0,:])-1
        self.constructed_corr_mat = self.construct_correlation_from_opinions()
        self.constructed_corr_mat_ref_sum = np.sum(self.constructed_corr_mat[0,:])-1
        
        opinions = self.opinions
        
        self.opinion_marginals = dict()
        for n_idx,norm_context in enumerate(self.norm_context_list):
            ops = opinions[:,n_idx]
            for idx,op in enumerate(ops): 
                self.possible_agents[idx].opinion[norm_context] = np.random.uniform(0.5,1) if op == 1 else np.random.uniform(0,0.5)
                self.possible_agents[idx].complete_information = True if self.baseline else False
        for _ag in self.possible_agents:
            if _ag.norm_context not in self.opinion_marginals:
                self.opinion_marginals[_ag.norm_context] = []
            self.opinion_marginals[_ag.norm_context].append(_ag.opinion[_ag.norm_context])
        self.opinion_marginals = [np.mean(self.opinion_marginals[k]) for k in self.norm_context_list]   
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
        if env.run_type in ['self-ref']:
            n_context_appr_beliefs = {norm_context:None for n_idx,norm_context in enumerate(self.norm_context_list)}
            for n_context in n_context_appr_beliefs.keys():
                pl_by_norm_contexts = [pl for pl in self.possible_agents if pl.norm_context==n_context]
                appr_list,true_appr_list,act_distort_list = [],[],[]
                for pl in pl_by_norm_contexts:
                    apr_from_bel = pl.belief[pl.norm_context]
                    appr_list.append(apr_from_bel)
                    appr_true = np.sum([self.norm_contexts_distr[_n]*self.opinion_marginals[_nidx] for _nidx,_n in enumerate(self.norm_context_list)])
                    true_appr_list.append(appr_true)
                    binary_op_from_bel = pl.get_act_for_apr_theta(apr_from_bel)
                    binary_op_from_true_apr = pl.get_act_for_apr_theta(appr_true)
                    act_distort_list.append(binary_op_from_bel-binary_op_from_true_apr)
                appr_distortion_mean = np.mean(np.array(appr_list)-np.array(true_appr_list))
                payoff_mean = np.mean([pl.total_reward for pl in pl_by_norm_contexts])
                participation_mean = np.mean([pl.total_participation for pl in pl_by_norm_contexts])
                n_context_appr_beliefs[n_context] = (appr_distortion_mean,payoff_mean,participation_mean,act_distort_list)
                
            belief_distort_map = {_n:n_context_appr_beliefs[_n][0] for _nidx,_n in enumerate(self.norm_context_list)}
            payoff_map = {_n:n_context_appr_beliefs[_n][1] for _n in self.norm_context_list}
            action_distortion_map = {_n:n_context_appr_beliefs[_n][3] for _nidx,_n in enumerate(self.norm_context_list)}
            
        elif env.run_type in ['community-ref']:
            n_context_appr_beliefs = {norm_context:None for n_idx,norm_context in enumerate(self.norm_context_list)}
            for n_context in n_context_appr_beliefs.keys():
                pl_by_norm_contexts = [pl for pl in self.possible_agents if pl.norm_context==n_context]
                appr_list,true_appr_list,act_distort_list = [],[],[]
                for pl in pl_by_norm_contexts:
                    ''' Use the norm_prop that the player holds to calculate the expected belief '''
                    apr_from_bel = np.sum([pl.norm_prop[k]*pl.belief[k] for k in pl.belief.keys()])
                    appr_list.append(apr_from_bel)
                    appr_true = np.sum([self.norm_contexts_distr[_n]*self.opinion_marginals[_nidx] for _nidx,_n in enumerate(self.norm_context_list)])
                    true_appr_list.append(appr_true)
                    binary_op_from_bel = pl.get_act_for_apr_theta(apr_from_bel)
                    binary_op_from_true_apr = pl.get_act_for_apr_theta(appr_true)
                    act_distort_list.append(binary_op_from_bel-binary_op_from_true_apr)
                appr_distortion_mean = np.mean(np.array(appr_list)-np.array(true_appr_list))
                payoff_mean = np.mean([pl.total_reward for pl in pl_by_norm_contexts])
                participation_mean = np.mean([pl.total_participation for pl in pl_by_norm_contexts])
                n_context_appr_beliefs[n_context] = (appr_distortion_mean,payoff_mean,participation_mean,act_distort_list)
                
            belief_distort_map = {_n:n_context_appr_beliefs[_n][0] for _nidx,_n in enumerate(self.norm_context_list)}
            payoff_map = {_n:n_context_appr_beliefs[_n][1] for _n in self.norm_context_list}
            action_distortion_map = {_n:n_context_appr_beliefs[_n][3] for _nidx,_n in enumerate(self.norm_context_list)}
            
        else:
            ''' This is moderator ref branch '''
            n_context_appr_beliefs = {norm_context:None for n_idx,norm_context in enumerate(self.norm_context_list)}
            for n_context in n_context_appr_beliefs.keys():
                pl_by_norm_contexts = [pl for pl in self.possible_agents if pl.norm_context==n_context]
                appr_list,true_appr_list,act_distort_list = [],[],[]
                for pl in pl_by_norm_contexts:
                    apr_from_bel = np.sum([pl.norm_prop[k]*pl.belief[k] for k in pl.belief.keys()])
                    appr_list.append(apr_from_bel)
                    appr_true = np.sum([self.norm_contexts_distr[_n]*self.opinion_marginals[_nidx] for _nidx,_n in enumerate(self.norm_context_list)])
                    true_appr_list.append(appr_true)
                    binary_op_from_bel = pl.get_act_for_apr_theta(apr_from_bel)
                    binary_op_from_true_apr = pl.get_act_for_apr_theta(appr_true)
                    act_distort_list.append(binary_op_from_bel-binary_op_from_true_apr)
                appr_distortion_mean = np.mean(np.array(appr_list)-np.array(true_appr_list)) if len(appr_list) > 0 else None
                rewards_list = [pl.total_reward for pl in pl_by_norm_contexts]
                payoff_mean = np.mean(rewards_list)
                participation_mean = np.mean([pl.total_participation for pl in pl_by_norm_contexts])
                n_context_appr_beliefs[n_context] = (appr_distortion_mean,payoff_mean,participation_mean,act_distort_list)
            payoff_map = {_n:n_context_appr_beliefs[_n][1] for _n in self.norm_context_list}
            belief_distort_map = {_n:n_context_appr_beliefs[_n][0] for _nidx,_n in enumerate(self.norm_context_list)}
            action_distortion_map = {_n:n_context_appr_beliefs[_n][3] for _nidx,_n in enumerate(self.norm_context_list)}
            
        if not self.no_print:
            print('-------------------------------------------------------------------------------------------------------')
            print('iter:',msg,'corr_mat sums:',np.sum(self.corr_mat,axis=0))
            print('iter:',msg,'mutual info_mat sums:',np.sum(self.mutual_info_mat,axis=0))
            print('iter:',msg,'Mean distort.:',belief_distort_map)         
            print('iter:',msg,'Mean total payoffs:',payoff_map)
            print('iter:',msg,'Mean total participation:',{_n:n_context_appr_beliefs[_n][2] for _n in self.norm_context_list})
        
        if msg == NUM_ITERS - 1:
            self.results_map['belief_distortion'] = belief_distort_map
            self.results_map['payoff'] = payoff_map
            self.results_map['action_distortion'] = action_distortion_map
            if not self.no_print:
                print(self.opinion_marginals,np.sum(self.opinion_marginals)/4)
                print(np.sum(self.corr_mat,axis=0))

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

    def step(self, actions, iter_no):
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
        rewards = {}
        if self.baseline:
            observed_actions = np.array([ag.opinion[ag.norm_context] for ag in self.agents])
        else:
            observed_actions = np.array([ag.action[2] for ag in self.agents if ag.action[0]!=-1])
        
        
        appr_mean,appr_var = np.mean(observed_actions), np.var(observed_actions)
        
        
        
        maj_op = 1 if appr_mean >= 0.5 else 0
        for ag in self.agents:
            if ag.action[0] == maj_op:
                rewards[ag.id] = ag.action[1] if ag.action[0] != -1 else ag.payoff_tol
            else:
                rewards[ag.id] = 0 if ag.action[0] != -1 else ag.payoff_tol 
        

        terminations = {agent.id: False for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent.id: env_truncation for agent in self.agents}

        # current observation is just the other player's most recent action
        ''' All observation are the same '''
        #observation = np.array([appr_mean,appr_var]).reshape((1,2))
        appr_sum = np.sum([x for x in observed_actions if x >=0.5])
        disappr_sum = np.sum([1-x for x in observed_actions if x <0.5])
        self.complete_observations = observed_actions
        observation = np.array([appr_sum,disappr_sum]).reshape((1,2))
        observations = {ag.id:observation for ag in self.agents}
        self.observations = observations
        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent.id: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render(iter_no)
        return observations, rewards, terminations, truncations, infos
    
    def get_complete_information(self):
        if self.baseline:
            observed_actions = np.array([(ag.opinion[ag.norm_context],ag.norm_context) for ag in self.agents])
        else:
            observed_actions = np.array([(ag.action[2],ag.norm_context) for ag in self.agents if ag.action[0]!=-1])
        return observed_actions
    
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

def process_moderator_reward(env):
    moderator_reward_socialwelfare_max = np.sum([abs(x) for x in env.results_map['payoff'].values()]) - np.sum([abs(x) for x in env.results_map['belief_distortion'].values()])
    moderator_reward_distortionwelfare_max = 4-np.sum([abs(x) for x in env.results_map['belief_distortion'].values()])
    return moderator_reward_socialwelfare_max,moderator_reward_distortionwelfare_max

def calc_belief_distortion(env):
    bel_distor = np.sum([abs(x) for x in env.results_map['belief_distortion'].values() if x is not None])
    return bel_distor

def calc_action_distortion(env):
    num_distors, pos_distors, neg_distors =[], [], []
    for x in env.results_map['action_distortion'].values():
        act_distor = np.array(x)
        num_distors.append(np.count_nonzero(act_distor))
        pos_distors.extend(np.extract(act_distor>0,act_distor).tolist())
        neg_distors.extend(np.extract(act_distor<0,act_distor).tolist())
    return [np.sum(num_distors), np.mean(pos_distors) if len(pos_distors) > 0 else np.nan, np.mean(neg_distors) if len(pos_distors) > 0 else np.nan]

def calc_payoff(env):
    payoff = np.sum([abs(x) for x in env.results_map['payoff'].values()])
    return payoff



def train_moderator_policy():
    #cumul_res_dict = dict()
    batch_size,process_id = 10,0
    #y_array_info = dict()
    y_array_info = []
    data_gen_mode = False
    all_data_X,all_data_Y = None,None
    filename_x,filename_y = 'all_data_X_'+str(process_id)+'.csv','all_data_Y_'+str(process_id)+'.csv'
    #for rt_idx,run_type in enumerate(['baseline','self-ref','moderator-ref','community-ref]):
    for numrows in np.arange(batch_size):
        
        print('----------------------------->',numrows)
        for rt_idx,run_type in enumerate(['moderator-ref']):
            sim_repeat_num = 1 if data_gen_mode else 30
            for r_itr in np.arange(sim_repeat_num):
                prev_env,attr_dict = None,{'run_type':run_type,'baseline':False}
                for sidx,moderator_context_signal in enumerate(['n1']):
                    if sidx > 0:
                        attr_dict = {'players_private_contexts':prev_env.players_private_contexts,
                                     'opinions':prev_env.opinions,
                                     'corr_mat':prev_env.corr_mat,
                                     'mutual_info_mat':prev_env.mutual_info_mat,
                                     'run_type':run_type,
                                     'baseline':False}
                    
                    env = parallel_env(render_mode='human',attr_dict=attr_dict)
                    if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
                        break
                    env.reset()
                    env.no_print = True
                    env.moderator_context_signal = moderator_context_signal
                    for ag in env.possible_agents:
                        if hasattr(ag, 'belief'): 
                            del ag.belief
                    for i in np.arange(NUM_ITERS):
                        actions = {agent.id:agent.act(env,run_type) for agent in env.possible_agents}
                        observations, rewards, terminations, truncations, infos = env.step(actions,i)
                        for agent in env.possible_agents:
                            agent.step_reward = rewards[agent.id]
                            agent.total_reward += rewards[agent.id]
                            agent.total_participation = agent.total_participation + 1 if agent.action[0] != -1 else agent.total_participation
                    '''
                    moderator_reward,moderator_reward_bd = process_moderator_reward(env)
                    y_array_info[env.moderator_context_signal] = moderator_reward_bd
                    '''
                    bel_distor = calc_action_distortion(env)
                    #y_array_info[moderator_context_signal] = bel_distor[moderator_context_signal]
                    y_array_info.append(bel_distor)
                    prev_env = copy.copy(env)
    
                '''
                if all_data_X is None:
                    all_data_X = np.reshape(np.array(np.sum(env.corr_mat,axis=0).tolist()+ env.opinion_marginals.tolist() + [env.norm_contexts_distr[x] for x in env.norm_context_list]) ,\
                                             newshape=(1,12))
                    #all_data_Y = np.reshape(np.array([y_array_info[x] for x in env.norm_context_list]),newshape = (1,4))
                else:
                    all_data_X = np.append(all_data_X,np.array(np.sum(env.corr_mat,axis=0).tolist()+ env.opinion_marginals.tolist() + [env.norm_contexts_distr[x] for x in env.norm_context_list]).reshape((1,12)),axis=0)
                    #all_data_Y = np.append(all_data_Y,np.array([y_array_info[x] for x in env.norm_context_list]).reshape((1,4)),axis=0)
                ''' 
                '''
                part_data = [(agent.opinion[agent.norm_context],agent.total_participation/NUM_ITERS)for agent in env.possible_agents]
                part_data.sort(key=lambda tup: tup[0])
                plt.plot([x[0] for x in part_data],[x[1] for x in part_data],'x')
                plt.show()
                '''
                '''
                for res_tag,res_dict in env.results_map.items():
                    if res_tag not in cumul_res_dict:
                        cumul_res_dict[res_tag] = dict()
                    for k,v in res_dict.items():
                        if k not in cumul_res_dict[res_tag]:
                            cumul_res_dict[res_tag][k] = [[],[],[]]
                        cumul_res_dict[res_tag][k][rt_idx].append(v)
                '''
        '''
        if numrows%10 == 0:
            with open(filename_x, "a") as f:
                np.savetxt(f, all_data_X, delimiter=",")
            with open(filename_y, "a") as f:
                np.savetxt(f, all_data_Y, delimiter=",")
            all_data_X,all_data_Y = None,None
    if all_data_X is not None:
        with open(filename_x, "a") as f:
            np.savetxt(f, all_data_X, delimiter=",")
        with open(filename_y, "a") as f:
            np.savetxt(f, all_data_Y, delimiter=",")
    '''
    '''    
    for res_tag,res_dict in cumul_res_dict.items():    
        data_a = [res_dict[k][0] for k in ['n1','n2','n3','n4']]
        data_b = [res_dict[k][1] for k in ['n1','n2','n3','n4']]
        data_c = [res_dict[k][2] for k in ['n1','n2','n3','n4']]
        
        
        ticks = ['n1','n2','n3','n4']
        
        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)
        
        plt.figure()
        plt.title(res_tag)
        bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
        bpc = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
        bpr = plt.boxplot(data_c, positions=np.array(range(len(data_b)))*2.0+0.8, sym='', widths=0.6)
        set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
        set_box_color(bpc, '#2C7BB6')
        set_box_color(bpr, '#7fcdbb')
        
        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c='#D7191C', label='Baseline')
        plt.plot([], c='#2C7BB6', label='Self-ref. BNE')
        plt.plot([], c='#7fcdbb', label='Commu-ref. BNE')
        plt.legend()
        
        plt.xticks(range(0, len(ticks) * 2, 2), ticks)
        plt.xlim(-2, len(ticks)*2)
        plt.show()
    '''
def show_group_results():
    batch_size,process_id = 10,0
    #y_array_info = dict()
    y_array_info = []
    data_gen_mode = True
    all_data_X,all_data_Y = None,None
    filename_x,filename_y = 'all_data_X_'+str(process_id)+'.csv','all_data_Y_'+str(process_id)+'.csv'
    #for rt_idx,run_type in enumerate(['baseline','self-ref','moderator-ref','community-ref]):
    is_bel_distor_run = False
    for numrows in np.arange(batch_size):
        
        print('----------------------------->',numrows)
        for rt_idx,run_type in enumerate(['community-ref']):
            sim_repeat_num = 1 if data_gen_mode else 30
            for r_itr in np.arange(sim_repeat_num):
                prev_env,attr_dict = None,{'run_type':run_type,'baseline':True}
                opt_mod_signal_map = dict()
                signals = ['n1','n2','n3','n4'] if run_type == 'moderator-ref' else ['n1']
                for sidx,moderator_context_signal in enumerate(signals):
                    if sidx > 0:
                        attr_dict = {'players_private_contexts':prev_env.players_private_contexts,
                                     'opinions':prev_env.opinions,
                                     'corr_mat':prev_env.corr_mat,
                                     'mutual_info_mat':prev_env.mutual_info_mat,
                                     'run_type':run_type,
                                     'baseline':True}
                    
                    env = parallel_env(render_mode='human',attr_dict=attr_dict)
                    if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
                        break
                    env.reset()
                    env.no_print = True
                    env.moderator_context_signal = moderator_context_signal
                    for ag in env.possible_agents:
                        if hasattr(ag, 'belief'): 
                            del ag.belief
                    for i in np.arange(NUM_ITERS):
                        actions = {agent.id:agent.act(env,run_type) for agent in env.possible_agents}
                        observations, rewards, terminations, truncations, infos = env.step(actions,i)
                        for agent in env.possible_agents:
                            agent.step_reward = rewards[agent.id]
                            agent.total_reward += rewards[agent.id]
                            agent.total_participation = agent.total_participation + 1 if agent.action[0] != -1 else agent.total_participation
                    if is_bel_distor_run:
                        distor = calc_belief_distortion(env)
                        if run_type == 'moderator-ref':
                            opt_mod_signal_map[moderator_context_signal] = distor
                    else:
                        distor = calc_action_distortion(env)   
                        if run_type == 'moderator-ref':
                            opt_mod_signal_map[moderator_context_signal] = distor
                    prev_env = copy.copy(env)
                if is_bel_distor_run:
                    if run_type == 'moderator-ref':
                        y_array_info.append(np.min(list(opt_mod_signal_map.values())))
                    else:
                        y_array_info.append(distor)
                else:
                    if run_type == 'moderator-ref':
                        sorted_distor = [tuple(x) for x in opt_mod_signal_map.values()]
                        sorted_distor.sort(key=lambda tup: tup[0])
                        y_array_info.append(sorted_distor[0])
                    else:
                        y_array_info.append(distor)
    f = open('results_by_group.csv', 'a')
    writer = csv.writer(f)
    baseline_tag = ' (beseline)' if env.baseline else ''
    if is_bel_distor_run:
        row = run_type+baseline_tag+' belief mean and sd distor',np.mean(y_array_info),np.std(y_array_info)
    else:
        y_array_info = np.array(y_array_info)
        row = run_type+baseline_tag+' action distor',np.nanmean(y_array_info, axis = 0),np.nanstd(y_array_info, axis = 0)
    writer.writerow(row)
    f.close()
    print(row)
    
def generate_moderator_optimal_action_grid():
    x_array_info,y_array_info = None, None
    ref_op_marginal_theta = [0.3,0.61,0.58,0.8]
    processed_list = []
    x_array_path = Path('grid_run_values_x_'+','.join(utils.list_to_str(ref_op_marginal_theta))+'.csv')
    if x_array_path.is_file():
        x_processed = np.genfromtxt('grid_run_values_x_'+','.join(utils.list_to_str(ref_op_marginal_theta))+'.csv', delimiter=',')
        processed_list = [(x_processed[i,0],x_processed[i,1]) for i in np.arange(x_processed.shape[0])]
    for corr_idx in np.arange(5):
        for distr_idx in np.arange(10):
            if (corr_idx,distr_idx) in processed_list:
                continue
            opt_mod_signal_map = dict()
            for rep_idx in np.arange(10):
                norm_contexts_distr = {'n1':np.linspace(0.1,.9,10)[distr_idx]}
                norm_contexts_distr.update({x:(1-norm_contexts_distr['n1'])/3 for x in ['n2','n3','n4']})
                
                print('----------------------------->',corr_idx,distr_idx)
                run_type= 'moderator-ref'
                prev_env,attr_dict = None,{'run_type':run_type,
                                           'ref_op_marginal_theta':ref_op_marginal_theta,
                                           'corr_idx':corr_idx,
                                           'norm_contexts_distr':norm_contexts_distr,
                                           'baseline':False}
                
                signals = ['n1','n2','n3','n4'] if run_type == 'moderator-ref' else ['n1']
                for sidx,moderator_context_signal in enumerate(signals):
                    if sidx > 0:
                        attr_dict = {'players_private_contexts':prev_env.players_private_contexts,
                                     'opinions':prev_env.opinions,
                                     'corr_mat':prev_env.corr_mat,
                                     'mutual_info_mat':prev_env.mutual_info_mat,
                                     'run_type':run_type,
                                     'ref_op_marginal_theta':ref_op_marginal_theta,
                                     'corr_idx':corr_idx,
                                     'norm_contexts_distr':norm_contexts_distr,
                                     'baseline':False}
                    
                    env = parallel_env(render_mode='human',attr_dict=attr_dict)
                    if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
                        break
                    env.reset()
                    env.no_print = True
                    env.moderator_context_signal = moderator_context_signal
                    for ag in env.possible_agents:
                        if hasattr(ag, 'belief'): 
                            del ag.belief
                    for i in np.arange(NUM_ITERS):
                        actions = {agent.id:agent.act(env,run_type) for agent in env.possible_agents}
                        observations, rewards, terminations, truncations, infos = env.step(actions,i)
                        for agent in env.possible_agents:
                            agent.step_reward = rewards[agent.id]
                            agent.total_reward += rewards[agent.id]
                            agent.total_participation = agent.total_participation + 1 if agent.action[0] != -1 else agent.total_participation
                    bel_distor = calc_belief_distortion(env)
                    if moderator_context_signal not in opt_mod_signal_map:
                        opt_mod_signal_map[moderator_context_signal] = []
                    opt_mod_signal_map[moderator_context_signal].append(bel_distor)
                    prev_env = copy.copy(env)
                norm_sum_corr_val = env.corr_mat_ref_sum
                norm_sum_constructed_corr_val = env.constructed_corr_mat_ref_sum
            
            norm_occurance_support_val = norm_contexts_distr['n1']
            x_entry = np.array([corr_idx,distr_idx,norm_sum_corr_val,norm_sum_constructed_corr_val,norm_occurance_support_val]).reshape((1,5))
            y_entry = np.array([np.mean(opt_mod_signal_map[x]) for x in env.norm_context_list]).reshape((1,4))
            
            if x_array_info is None:
                x_array_info = np.copy(x_entry)
            else:
                x_array_info = np.append(x_array_info,x_entry,axis=0)
            
            if y_array_info is None:
                y_array_info = np.copy(y_entry)
            else:
                y_array_info = np.append(y_array_info,y_entry,axis=0)
    
            with open('grid_run_values_x_'+','.join(utils.list_to_str(ref_op_marginal_theta))+'.csv', "a") as f:
                np.savetxt(f, x_array_info, delimiter=",")
            with open('grid_run_values_y_'+','.join(utils.list_to_str(ref_op_marginal_theta))+'.csv', "a") as f:
                np.savetxt(f, y_array_info, delimiter=",")                
                            
            x_array_info,y_array_info = None, None
    

def process_grid_results():
    
    x_arr = np.genfromtxt('grid_run_values_x_0.3,0.61,0.58,0.8.csv', delimiter=',')
    y_arr = np.genfromtxt('grid_run_values_y_0.3,0.61,0.58,0.8.csv', delimiter=',')
    '''
    x_arr = np.genfromtxt('grid_run_values_x.csv', delimiter=',')
    y_arr = np.genfromtxt('grid_run_values_y.csv', delimiter=',')
    '''
    y_arr = 1-y_arr
    #y_arr = softmax(y_arr,axis=1)
    corr_arr = x_arr[:,3]
    y_arr =y_arr[:,0]
    x_arr = x_arr[:,:3].astype(np.int32)
    fig, ax = plt.subplots()
    
    intersection_matrix = np.full(shape=(5,10), fill_value=np.nan)
    for xrow,yrow in zip(x_arr,y_arr):
        intersection_matrix[xrow[0],xrow[1]] = yrow
    
    
    ax.matshow(intersection_matrix, cmap=plt.cm.Blues)
    
    for i in range(intersection_matrix.shape[0]):
        ax.text(-1.5, i, str(round(corr_arr[int(i*10)],2)), va='center', ha='center')
        for j in range(intersection_matrix.shape[1]):
            c = intersection_matrix[i,j]
            ax.text(j, i, str(round(c,2)), va='center', ha='center')
            
    
    plt.show()

#process_grid_results()

def process_all_grids():
    _files =os.listdir('grid_run_results_30')
    all_files_dict = dict()
    for f in _files:
        x_file = f.split('_')[3]
        apr_distr_tag = f.split('_')[-1][:-4]
        if apr_distr_tag not in all_files_dict:
            all_files_dict[apr_distr_tag] = [None,None]
        if x_file == 'x':
            all_files_dict[apr_distr_tag][0] = f
        else:
            all_files_dict[apr_distr_tag][1] = f
    f=1
    fig, ax = plt.subplots(nrows=1,ncols=5)
    
    ax_ctr = 0
    for k,v in all_files_dict.items():
        apr_distr_tag = v[0].split('_')[-1][:-4]
        x_arr = np.genfromtxt('grid_run_results_30\\'+str(v[0]), delimiter=',')
        y_arr = np.genfromtxt('grid_run_results_30\\'+str(v[1]), delimiter=',')
        y_arr = 1-y_arr
        y_arr_softmax = softmax(y_arr,axis=1)
        distr_arr = x_arr[:,3]
        corr_arr = x_arr[:,2]
        y_argmax_indices = np.argmax(y_arr,axis=1)
        y_arr = y_arr[:,0]
        '''
        y_arr_ind_sel = np.copy(y_arr)[:,0]
        for i in np.arange(y_arr_ind_sel.shape[0]):
            y_arr_ind_sel[i] = 1 if y_argmax_indices[i] == 0 else 0
        y_arr = y_arr_ind_sel
        '''
        x_arr = x_arr[:,:2].astype(np.int32)
        
        
        intersection_matrix = np.full(shape=(5,10), fill_value=np.nan)
        for xrow,yrow in zip(x_arr,y_arr):
            intersection_matrix[xrow[0],xrow[1]] = yrow
        
        
        ax[ax_ctr].matshow(intersection_matrix, cmap=plt.cm.Blues)
        
        for i in range(intersection_matrix.shape[0]):
            ax[ax_ctr].text(-1.5, i, str(round(corr_arr[int(i*10)],2)), va='center', ha='center',fontsize='xx-small')
            for j in range(intersection_matrix.shape[1]):
                if i==0:
                    ax[ax_ctr].text(j, -1.5, str(round(distr_arr[j],2)), va='center', ha='center',fontsize='xx-small')
                c = intersection_matrix[i,j]
                ax[ax_ctr].text(j, i, str(round(c,2)), va='center', ha='center',fontsize='xx-small')
        ax[ax_ctr].set_title(str(apr_distr_tag),fontsize='xx-small')
        ax[ax_ctr].set_axis_off()
        ax_ctr += 1
    plt.show()
#process_all_grids()

def plot_action_distortion_by_group():
    fig, ax = plt.subplots(nrows=4, sharex=True)
    ctr = 0
    with open('results_by_group.csv') as file:
        for line in file:
            line.rstrip()
            if 'action distor' in line:
                m = re.findall('\[(.+?)\]', line)
                mean_str = [float(x) for x in m[0].split(',')]
                std_str = [float(x) for x in m[1].split(',')]
                for i in [1,2]:
                    utils.plot_gaussian(mean_str[i], std_str[i]**2,ax=ax[ctr],vert_line_at=1.5 if i==2 else -1.5)
                if 'baseline' in line:
                    ax[ctr].set_title(''.join(line.split(' ')[:2]))
                else:
                    ax[ctr].set_title(''.join(line.split(' ')[0]))
                ctr += 1  
    fig, ax = plt.subplots(nrows=4, sharex=True)
    ctr = 0
    with open('results_by_group.csv') as file:
        for line in file:
            line.rstrip()
            if 'action distor' in line:
                m = re.findall('\[(.+?)\]', line)
                mean_str = [float(x) for x in m[0].split(',')]
                std_str = [float(x) for x in m[1].split(',')]
                utils.plot_gaussian(mean_str[0], std_str[0]**2,ax=ax[ctr])
                if 'baseline' in line:
                    ax[ctr].set_title(''.join(line.split(' ')[:2]))
                else:
                    ax[ctr].set_title(''.join(line.split(' ')[0]))
                ctr += 1  
    plt.show()

def plot_belief_distortion_by_group():
    fig, ax = plt.subplots(nrows=4, sharex=True)
    ctr = 0
    with open('results_by_group.csv') as file:
        for line in file:
            line.rstrip()
            if 'belief' in line:
                m = re.findall('\[(.+?)\]', line)
                mean_str = float(line.split(',')[-2])
                std_str = float(line.split(',')[-1])
                utils.plot_gaussian(mean_str, std_str**2,ax=ax[ctr])
                if 'baseline' in line:
                    ax[ctr].set_title(''.join(line.split(' ')[:2]))
                else:
                    ax[ctr].set_title(''.join(line.split(' ')[0]))
                ctr += 1  
    plt.show()
                
                

#plot_belief_distortion_by_group()        

    
    