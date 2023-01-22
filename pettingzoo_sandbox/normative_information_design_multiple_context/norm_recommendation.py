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
from scipy.stats import beta, norm
import seaborn as sns
import pandas as pd


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
        self.norm_context_list = ['n1','n2','n3','n4']
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
        self.opinions,self.corr_mat,self.mutual_info_mat = utils.generate_samples_copula([self.true_state_distr_params[n] for n in self.norm_context_list] ,(0,None))
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
        ''' reconcile the true state based on the generated sampled '''
        self.true_state = {n:self.opinion_marginals[idx] for idx,n in enumerate(self.norm_context_list)}
        
        
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
                theta_prime_rate = np.mean([ag.opinion[ag.norm_context] for ag in self.agents if ag.action[0]!=-1])
                a_prime = theta_prime_rate*self.update_rate
                b_prime =  self.update_rate-a_prime
                self.common_prior = (self.common_prior[0]+a_prime, self.common_prior[1]+b_prime)
            
            observations = self.common_prior
            # typically there won't be any information in the infos, but there must
            # still be an entry for each agent
            infos = {agent.id: {} for agent in self.agents}
    
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
                a_prime = theta_prime_rate*self.update_rate
                b_prime =  self.update_rate-a_prime
                for norm_context in self.norm_context_list:
                    self.prior_baseline[norm_context] = (self.prior_baseline[norm_context][0]+a_prime, self.prior_baseline[norm_context][1]+b_prime)
    
    def generate_posteriors(self,signal_distribution):
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
        for x in np.linspace(0,1,100):
            priors_rescaled[x] = beta.pdf(x, self.common_prior[0], self.common_prior[1])
            _constr_distr = utils.Gaussian_plateu_distribution(x,.01,self.normal_constr_sd)
            likelihood_rescaled[x] = _constr_distr.pdf(signal_distribution)
        priors_rescaled = {k:v/sum(list(priors_rescaled.values())) for k,v in priors_rescaled.items()}
        likelihood_rescaled = {k:v/sum(list(likelihood_rescaled.values())) for k,v in likelihood_rescaled.items()}
        
        for x in np.linspace(0,1,100):
            posteriors = _post(x,priors_rescaled,likelihood_rescaled)
            ''' Since the signal realization will be based on the signal distribution, we can take the expectation of the posterior w.r.t each realization.'''
            expected_posterior_for_state_x = (signal_distribution*posteriors[0]) + ((1-signal_distribution)*posteriors[1])
            all_posteriors.append(expected_posterior_for_state_x)
        all_posteriors = [x/np.sum(all_posteriors) for x in all_posteriors]
        '''
        plt.plot(np.linspace(0,1,100),all_posteriors)
        plt.xlim(0,1)
        plt.show()
        '''
        exp_x = np.sum([x*prob_x for x,prob_x in zip(np.linspace(0,1,100),all_posteriors)])
        self.common_posterior = exp_x
        return exp_x
        
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
    
    
    
    def act(self, env, run_type, baseline,norm_context):
        if run_type in ['baseline','self-ref']:
            return self.act_self_ref(env,baseline,norm_context)
        
    
    def act_context_misinterpretation(self,env,baseline,norm_context):
        f=1
        for pl    
        
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
            ''' The Bayesian Nash Eq action thresholds. '''
            disappr_bar = 1 - (u_bar/(1-env.common_posterior))
            appr_bar = u_bar/env.common_posterior
            
            if op >= 0.5:
                if op < appr_bar:
                    self.action_code = -1
                    self.action_util = u_bar
                else:
                    self.action_code = 1
                    self.action_util = util(op)
            else:
                if op > disappr_bar:
                    self.action_code = -1
                    self.action_util = u_bar
                else:
                    self.action_code = 0
                    self.action_util = util(op)
                    
            
            self.action =(self.action_code,self.action_util,self.opinion[self.norm_context])
        else:
            assert norm_context==self.norm_context, 'norm context check failed'
            theta_baseline = env.prior_baseline[norm_context][0]/sum(env.prior_baseline[norm_context])
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
    state_evolution_baseline = dict()
    for batch_num in np.arange(100):
        attr_dict = {'norm_contexts_distr': {'n1':0.3,'n2':0.2,'n3':0.1,'n4':0.4},
                    'true_state_distr_params':{'n1':(5,3),'n2':(3.7,3),'n3':(2,3),'n4':(9,2)}}
        env = heterogenous_parallel_env(render_mode='human',attr_dict=attr_dict)
        ''' Check that every norm context has at least one agent '''
        if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
            raise Exception()
        env.reset()
        env.no_print = True
        env.NUM_ITERS = 100
        common_prior_var = 0.1
        env.common_prior = {k:utils.est_beta_from_mu_sigma(mu=v+np.random.normal(scale=common_prior_var), sigma=common_prior_var) for k,v in env.true_state.items()}
        env.prior_baseline = {k:v for k,v in env.common_prior.items()}
        env.normal_constr_sd = 0.3
        #env.constraining_distribution = utils.Gaussian_plateu_distribution(env.common_prior[0]/sum(env.common_prior),.01,.3)
        #env.constraining_distribution = utils.Gaussian_plateu_distribution(.3,.01,.3)
        dataset = []
    
        
        for i in np.arange(100):
            print(batch_num,i)
            actions = dict()
            for norm_context in env.norm_context_list:
                if norm_context not in state_evolution_baseline:
                    state_evolution_baseline[norm_context] = dict()
                if i not in  state_evolution_baseline[norm_context]:
                    state_evolution_baseline[norm_context][i] = []
                state_evolution_baseline[norm_context][i].append(env.prior_baseline[norm_context][0]/sum(env.prior_baseline[norm_context]))
                
                curr_state = env.common_prior[norm_context][0]/sum(env.common_prior[norm_context])
                baseline_bels_mean = env.prior_baseline[norm_context][0]/sum(env.prior_baseline[norm_context])
                actions.update({agent.id:agent.act(env,run_type='self-ref',baseline=True,norm_context=norm_context) for agent in env.possible_agents if agent.norm_context==norm_context })
            env.step(actions,i,baseline=True)
            
        #env.common_prior = (np.random.randint(low=1,high=4),np.random.randint(low=1,high=4))
    cols = ['time', 'belief','norm']
    lst = []
    for norm,norm_data in state_evolution_baseline.items():
        for k,v in norm_data.items():
            for _v in v:
                lst.append([k,_v,norm])
    df = pd.DataFrame(lst, columns=cols)
    
    sns.set_theme(style="darkgrid")
    '''
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')   
    ax.scatter([x[0] for x in dataset], [x[1] for x in dataset], [x[2] for x in dataset])
    ax.set_xlabel('state as prior mean')
    ax.set_ylabel('signal distr theta')
    ax.set_zlabel('Reward')
    '''
    fig = plt.figure(figsize=(12, 12))
    ax = sns.lineplot(hue="norm", x="time", y="belief", ci="sd", estimator='mean', data=df)
    
    plt.show()