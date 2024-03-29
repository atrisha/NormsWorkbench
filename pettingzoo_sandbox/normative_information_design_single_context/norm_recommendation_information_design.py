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
from pettingzoo_sandbox.all_networks import QNetwork
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
import torch




        

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
        self.update_rate = 10
        #self.norm_context_list = ['n1','n2','n3','n4']
        self.norm_context_list = ['n1']
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
        
        norm_context_appr_rate = self.true_state
        ops = [o for o in np.random.uniform(low=0.5,high=1,size=int(self.num_players*norm_context_appr_rate['n1']))]
        ops.extend([o for o in np.random.uniform(low=0,high=0.5,size=int(self.num_players*(1-norm_context_appr_rate['n1'])))])
        np.random.shuffle(ops)
        if len(ops) < self.num_players:
            ops.extend([np.random.random()])
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
                self.prior_baseline = (self.prior_baseline[0]+a_prime, self.prior_baseline[1]+b_prime)
    
    def generate_posteriors(self,signal_distribution):
        if abs(signal_distribution-self.common_prior_mean) > self.normal_constr_w:
            self.common_posterior = self.common_prior_mean
            return self.common_prior_mean
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
            priors_rescaled[x] = utils.beta_pdf(x, self.common_prior[0], self.common_prior[1])
            _constr_distr = utils.Gaussian_plateu_distribution(signal_distribution,.01,self.normal_constr_w)
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
        self.common_posterior = exp_x
        return exp_x
    
    @property
    def common_prior_mean(self):   
        return self.common_prior[0]/sum(self.common_prior)
    
        
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
    
    
    
    def act(self, env, run_type, baseline):
        if run_type in ['baseline','self-ref']:
            return self.act_self_ref(env,baseline)
        
        
        
    def act_self_ref(self, env,baseline):
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
            theta_baseline = env.prior_baseline[0]/sum(env.prior_baseline)
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
    
    
    
    
class StewardAgent():
    
    def __init__(self,qnetwork):
        self.qnetwork = qnetwork       
        
    
class RunInfo():
    
    def __init__(self,iter):
        self.iter = iter
        
if __name__ == "__main__":
    """ ENV SETUP """
    common_prior = (2,3.3)
    normal_constr_w = 0.1
    common_prior_mean = common_prior[0]/sum(common_prior)
    state_evolution,state_evolution_baseline = dict(), dict()
    lst = []
    #for signal_distr_theta_idx, signal_distr_theta in enumerate([common_prior_mean-(normal_constr_w+0.05),common_prior_mean-(normal_constr_w-0.05),common_prior_mean+(normal_constr_w+0.05),common_prior_mean+(normal_constr_w-0.05)]):
    '''
    if signal_distr_theta <=0 or signal_distr_theta >=1:
        continue
    '''
    opt_signals = {0.1:0.1, 0.2:0.1, 0.3:0.4, 0.4:0.5, 0.5:0.5, 0.6:0.5, 0.7:0.7, 0.8:0.7, 0.9:0.8}
    for batch_num in np.arange(100):
        env = parallel_env(render_mode='human',attr_dict={'true_state':{'n1':0.55}})
        ''' Check that every norm context has at least one agent '''
        if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
            raise Exception()
        env.reset()
        env.no_print = True
        env.NUM_ITERS = 100
        env.common_prior = common_prior
        env.prior_baseline = env.common_prior
        env.normal_constr_w = normal_constr_w
        #env.constraining_distribution = utils.Gaussian_plateu_distribution(env.common_prior[0]/sum(env.common_prior),.01,.3)
        #env.constraining_distribution = utils.Gaussian_plateu_distribution(.3,.01,.3)
        dataset = []
    
    
        for i in np.arange(100):
            print(batch_num,i)
            curr_state = env.common_prior[0]/sum(env.common_prior)
            #signal_distr_theta = curr_state - 0.3
            
            signal_distr_theta = opt_signals[round(curr_state,1)]
            '''
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            q_model = QNetwork(input_state_size=3)
            q_model.load_state_dict(torch.load('../agent_qnetwork.model'))
            q_model = q_model.to(device)
            opt_act = []
            for _act in np.arange(0.1,1,.1):
                
                action = torch.tensor([[_act]], device=device, dtype=torch.float)
                #state_ = torch.FloatTensor([env.common_prior[0]/sum(env.common_prior), beta(a=env.common_prior[0], b=env.common_prior[1]).var()])
                state_ = torch.tensor([env.common_prior[0]/sum(env.common_prior), utils.beta_var(a=env.common_prior[0], b=env.common_prior[1])], dtype=torch.float32, device=device).unsqueeze(0)
                input_tensor = torch.cat((state_,action),axis=1)
                current_reward = q_model.forward(input_tensor)
                opt_act.append((_act,current_reward.item()))
            opt_act = sorted(opt_act, key=lambda tup: tup[0])[-1][0]
            
            signal_distr_theta = opt_act
            '''
            _d = abs(signal_distr_theta-env.common_prior_mean)
            if _d <= env.normal_constr_w:
                valid_distr = True
            else:
                valid_distr = False
            
            
            if i not in  state_evolution:
                state_evolution[i] = []
            state_evolution[i].append(env.common_prior[0]/sum(env.common_prior))
            if i not in  state_evolution_baseline:
                state_evolution_baseline[i] = []
            state_evolution_baseline[i].append(env.prior_baseline[0]/sum(env.prior_baseline))
            
            
            #signal_distr_theta = np.random.uniform()
            baseline_bels_mean = env.prior_baseline[0]/sum(env.prior_baseline)
            ''' the posterior gets updated inside this '''
            _d = abs(signal_distr_theta-env.common_prior_mean)
            if valid_distr:
                posterior_mean = env.generate_posteriors(signal_distr_theta)
            else:
                posterior_mean = baseline_bels_mean
            
            
            if valid_distr:
                ''' act is based on the new posterior acting as prior '''
                actions = {agent.id:agent.act(env,run_type='self-ref',baseline=False) for agent in env.possible_agents}
                ''' common prior is updated based on the action observations '''
                observations, rewards, terminations, truncations, infos = env.step(actions,i,baseline=False)
                s,a,r = curr_state, signal_distr_theta, rewards
                new_common_prior_mean = env.common_prior[0]/sum(env.common_prior)
                dataset.append((s,a,r))
                next_state = observations
            else:
                env.common_prior = env.prior_baseline
            actions = {agent.id:agent.act(env,run_type='self-ref',baseline=True) for agent in env.possible_agents}
            env.step(actions,i,baseline=True)
            
            
            
            #env.common_prior = (np.random.randint(low=1,high=4),np.random.randint(low=1,high=4))
        cols = ['time', 'belief','model']
        only_baseline_plot = False
        
        
        if not only_baseline_plot:
            for k,v in state_evolution.items():
                for _v in v:
                    lst.append([k,_v,'signal'])
            
        
        for k,v in state_evolution_baseline.items():
            for _v in v:
                lst.append([k,_v,'no signal'])
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
    fig = plt.figure(figsize=(6, 6))
    ax = sns.lineplot(hue="model", x="time", y="belief", ci="sd", estimator='mean', data=df)
    #if only_baseline_plot:
    plt.legend([],[], frameon=False)
    plt.title(common_prior)
    plt.show()