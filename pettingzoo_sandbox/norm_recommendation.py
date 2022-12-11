import functools

import gymnasium
from gymnasium.spaces import Discrete, Box
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from CENormEnvironmentSimulation import Player
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

NUM_ITERS = 10


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
        if not hasattr(self, 'players_private_contexts'):
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
        
            
                
        opinions = self.opinions
        
        self.opinion_marginals = np.sum(opinions,axis=0)/100
        
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
                appr_mean = np.mean([pl.belief[pl.norm_context] for pl in pl_by_norm_contexts])
                payoff_mean = np.mean([pl.total_reward for pl in pl_by_norm_contexts])
                participation_mean = np.mean([pl.total_participation for pl in pl_by_norm_contexts])
                n_context_appr_beliefs[n_context] = (appr_mean,payoff_mean,participation_mean)
                
            belief_distort_map = {_n:n_context_appr_beliefs[_n][0]-self.opinion_marginals[_nidx] for _nidx,_n in enumerate(self.norm_context_list)}
            payoff_map = {_n:n_context_appr_beliefs[_n][1] for _n in self.norm_context_list}
        elif env.run_type in ['community-ref']:
            n_context_appr_beliefs = {norm_context:None for n_idx,norm_context in enumerate(self.norm_context_list)}
            for n_context in n_context_appr_beliefs.keys():
                pl_by_norm_contexts = [pl for pl in self.possible_agents if pl.norm_context==n_context]
                appr_list,true_appr_list = [],[]
                for pl in pl_by_norm_contexts:
                    ''' Use the norm_prop that the player holds to calculate the expected belief '''
                    appr_list.append(np.sum([pl.norm_prop[k]*pl.belief[k] for k in pl.belief.keys()]))
                    true_appr_list.append(np.sum([self.norm_contexts_distr[_n]*self.opinion_marginals[_nidx] for _nidx,_n in enumerate(self.norm_context_list)]))
                appr_distortion_mean = np.mean(np.array(appr_list)-np.array(true_appr_list))
                payoff_mean = np.mean([pl.total_reward for pl in pl_by_norm_contexts])
                participation_mean = np.mean([pl.total_participation for pl in pl_by_norm_contexts])
                n_context_appr_beliefs[n_context] = (appr_distortion_mean,payoff_mean,participation_mean)
                
            belief_distort_map = {_n:n_context_appr_beliefs[_n][0] for _nidx,_n in enumerate(self.norm_context_list)}
            payoff_map = {_n:n_context_appr_beliefs[_n][1] for _n in self.norm_context_list}
        else:
            ''' This is moderator ref branch '''
            n_context_appr_beliefs = {norm_context:None for n_idx,norm_context in enumerate(self.norm_context_list)}
            for n_context in n_context_appr_beliefs.keys():
                pl_by_norm_contexts = [pl for pl in self.possible_agents if pl.norm_context==n_context]
                appr_list,true_appr_list = [],[]
                for pl in pl_by_norm_contexts:
                    appr_list.append(np.sum([pl.norm_prop[k]*pl.belief[k] for k in pl.belief.keys()]))
                    true_appr_list.append(np.sum([self.norm_contexts_distr[_n]*self.opinion_marginals[_nidx] for _nidx,_n in enumerate(self.norm_context_list)]))
                appr_distortion_mean = np.mean(np.array(appr_list)-np.array(true_appr_list)) if len(appr_list) > 0 else None
                rewards_list = [pl.total_reward for pl in pl_by_norm_contexts]
                payoff_mean = np.mean(rewards_list)
                participation_mean = np.mean([pl.total_participation for pl in pl_by_norm_contexts])
                n_context_appr_beliefs[n_context] = (appr_distortion_mean,payoff_mean,participation_mean)
            payoff_map = {_n:n_context_appr_beliefs[_n][1] for _n in self.norm_context_list}
            belief_distort_map = {_n:n_context_appr_beliefs[_n][0] for _nidx,_n in enumerate(self.norm_context_list)}
            
        '''
        obs = self.observations[0]
        obs_appr_mean,obs_appr_var = obs[0,0], obs[0,1]
        #self.belief = self.belief + np.rint(obs*10)
        mom_alpha_est = lambda u,v : u * (((u*(1-u))/v) - 1)
        mom_beta_est = lambda u,v : (1-u) * (((u*(1-u))/v) - 1)
        mom_cond = True if obs_appr_var < obs_appr_mean*(1-obs_appr_mean) else False
        belief = np.array([mom_alpha_est(obs_appr_mean,obs_appr_var), mom_beta_est(obs_appr_mean,obs_appr_var)]).reshape((1,2))
        '''
        if not self.no_print:
            print('-------------------------------------------------------------------------------------------------------')
            print('iter:',msg,'corr_mat sums:',np.sum(self.corr_mat,axis=0))
            print('iter:',msg,'mutual info_mat sums:',np.sum(self.mutual_info_mat,axis=0))
            print('iter:',msg,'Mean distort.:',belief_distort_map)         
            print('iter:',msg,'Mean total payoffs:',payoff_map)
            print('iter:',msg,'Mean total participation:',{_n:n_context_appr_beliefs[_n][2] for _n in self.norm_context_list})
        #utils.plot_gaussian(obs_appr_mean, obs_appr_var)
        if msg == NUM_ITERS - 1:
            self.results_map['belief_distortion'] = belief_distort_map
            self.results_map['payoff'] = payoff_map
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

def process_moderator_reward(env):
    moderator_reward_socialwelfare_max = np.sum([abs(x) for x in env.results_map['payoff'].values()]) - np.sum([abs(x) for x in env.results_map['belief_distortion'].values()])
    moderator_reward_distortionwelfare_max = 4-np.sum([abs(x) for x in env.results_map['belief_distortion'].values()])
    return moderator_reward_socialwelfare_max,moderator_reward_distortionwelfare_max

def calc_belief_distortion(env):
    bel_distor = np.sum([abs(x) for x in env.results_map['belief_distortion'].values() if x is not None])
    return bel_distor

def calc_action_distortion(env):
    f=1

def calc_payoff(env):
    payoff = np.sum([abs(x) for x in env.results_map['payoff'].values()])
    return payoff



def train_moderator_polict():
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
        for rt_idx,run_type in enumerate(['community-ref']):
            sim_repeat_num = 1 if data_gen_mode else 30
            for r_itr in np.arange(sim_repeat_num):
                prev_env,attr_dict = None,{'run_type':run_type,'baseline':True}
                for sidx,moderator_context_signal in enumerate(['n1']):
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
                    '''
                    moderator_reward,moderator_reward_bd = process_moderator_reward(env)
                    y_array_info[env.moderator_context_signal] = moderator_reward_bd
                    '''
                    bel_distor = calc_belief_distortion(env)
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
    for numrows in np.arange(batch_size):
        
        print('----------------------------->',numrows)
        for rt_idx,run_type in enumerate(['moderator-ref']):
            sim_repeat_num = 1 if data_gen_mode else 30
            for r_itr in np.arange(sim_repeat_num):
                prev_env,attr_dict = None,{'run_type':run_type,'baseline':False}
                opt_mod_signal_map = dict()
                signals = ['n1','n2','n3','n4'] if run_type == 'moderator-ref' else ['n1']
                for sidx,moderator_context_signal in enumerate(signals):
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
                    bel_distor = calc_belief_distortion(env)
                    #bel_distor = calc_payoff(env)
                    if run_type == 'moderator-ref':
                        opt_mod_signal_map[moderator_context_signal] = bel_distor
                    #y_array_info[moderator_context_signal] = bel_distor[moderator_context_signal]
                    prev_env = copy.copy(env)
                if run_type == 'moderator-ref':
                    y_array_info.append(np.min(list(opt_mod_signal_map.values())))
                else:
                    y_array_info.append(bel_distor)
    f = open('results_by_group.csv', 'a')
    writer = csv.writer(f)
    baseline_tag = ' (beseline)' if env.baseline else ''
    row = run_type+baseline_tag+' belief mean and sd distor',np.mean(y_array_info),np.std(y_array_info)
    writer.writerow(row)
    f.close()
    print(row)
    
def generate_moderator_optimal_action_grid():
    x_array_info,y_array_info = None, None
    ref_op_marginal_theta = [0.61,0.3,0.58,0.8]
    processed_list = []
    x_array_path = Path('grid_run_values_x.csv')
    if x_array_path.is_file():
        x_processed = np.genfromtxt('grid_run_values_x.csv', delimiter=',')
        processed_list = [(x_processed[i,0],x_processed[i,1]) for i in np.arange(x_processed.shape[0])]
    for corr_idx in np.arange(5):
        for distr_idx in np.arange(10):
            if (corr_idx,distr_idx) in processed_list:
                continue
            norm_contexts_distr = {'n1':np.linspace(0.1,.9,10)[distr_idx]}
            norm_contexts_distr.update({x:(1-norm_contexts_distr['n1'])/3 for x in ['n2','n3','n4']})
            
            print('----------------------------->',corr_idx,distr_idx)
            run_type= 'moderator-ref'
            prev_env,attr_dict = None,{'run_type':run_type,
                                       'ref_op_marginal_theta':ref_op_marginal_theta,
                                       'corr_idx':corr_idx,
                                       'norm_contexts_distr':norm_contexts_distr,
                                       'baseline':False}
            opt_mod_signal_map = dict()
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
                opt_mod_signal_map[moderator_context_signal] = bel_distor
                prev_env = copy.copy(env)
            norm_sum_corr_val = env.corr_mat_ref_sum
            norm_occurance_support_val = norm_contexts_distr['n1']
            x_entry = np.array([corr_idx,distr_idx,norm_sum_corr_val,norm_occurance_support_val]).reshape((1,4))
            y_entry = np.array([opt_mod_signal_map[x] for x in env.norm_context_list]).reshape((1,4))
            
            if x_array_info is None:
                x_array_info = np.copy(x_entry)
            else:
                x_array_info = np.append(x_array_info,x_entry,axis=0)
            
            if y_array_info is None:
                y_array_info = np.copy(y_entry)
            else:
                y_array_info = np.append(y_array_info,y_entry,axis=0)
    
            with open('grid_run_values_x.csv', "a") as f:
                np.savetxt(f, x_array_info, delimiter=",")
            with open('grid_run_values_y.csv', "a") as f:
                np.savetxt(f, y_array_info, delimiter=",")                
                            
            x_array_info,y_array_info = None, None
    

def process_grid_results():
    
    x_arr = np.genfromtxt('grid_run_results\\grid_run_values_x_0.61,0.3,0.58,0.8.csv', delimiter=',')
    y_arr = np.genfromtxt('grid_run_results\\grid_run_values_y_0.61,0.3,0.58,0.8.csv', delimiter=',')
    '''
    x_arr = np.genfromtxt('grid_run_values_x.csv', delimiter=',')
    y_arr = np.genfromtxt('grid_run_values_y.csv', delimiter=',')
    '''
    y_arr = 1-y_arr
    #y_arr = softmax(y_arr,axis=1)
    corr_arr = x_arr[:,2]
    y_arr =y_arr[:,0]
    x_arr = x_arr[:,:2].astype(np.int32)
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
process_grid_results()
    
    