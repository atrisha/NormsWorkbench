'''
Created on 7 Oct 2022

@author: Atrisha
'''
import numpy as np
import utils
from Equilibria import CorrelatedEquilibria, PureNashEquilibria
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import constants
import matplotlib.animation as animation
from scipy.interpolate import RectBivariateSpline
import sys
import itertools
from scipy.interpolate import UnivariateSpline
import scipy.integrate as integrate
from scipy.stats import entropy
from constants import op_mode, num_players
from scipy import interpolate
from sklearn.preprocessing import normalize
from numpy import genfromtxt
from collections import Counter

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
        
def extend_table_with_noop(r_info, two_players, payoff_dict):
    all_strats = list(itertools.product(*[['A','D','N'],['A','D','N']]))
    ext_payoff_dict = {k:v for k,v in payoff_dict.items()}
    for strat in all_strats:
        for pl_idx,pl in enumerate(two_players):
            op = pl.opinion
            op_bar = utils.get_oth_opinion(op)
            ''' equilibrium constraints and no-op selection constraints'''
            s_n_op,u_n_op = ('N',op),np.random.uniform(payoff_dict[(op,op_bar)][pl_idx],payoff_dict[(op,op)][pl_idx])
            s_n_opbar,u_n_opbar = ('N',op_bar),np.random.uniform(payoff_dict[(op,op_bar)][pl_idx],payoff_dict[(op_bar,op_bar)][pl_idx])
            s_n_n,u_n_n = ('N','N'),np.random.uniform(payoff_dict[(op,op_bar)][pl_idx],payoff_dict[(op,op)][pl_idx])
            ''' risk dominance constraints '''
            s_op_n,u_op_n = (op,'N'),np.random.uniform(payoff_dict[(op,op_bar)][pl_idx],payoff_dict[(op,op)][pl_idx])
            s_opbar_n,u_opbar_n = (op_bar,'N'),np.random.uniform(payoff_dict[(op_bar,op)][pl_idx],payoff_dict[(op_bar,op_bar)][pl_idx])
            ext_list = zip([s_n_op,s_n_opbar,s_n_n,s_op_n,s_opbar_n],[u_n_op,u_n_opbar,u_n_n,u_op_n,u_opbar_n])
            for exts in ext_list:
                if exts[0] not in ext_payoff_dict:
                    ext_payoff_dict[exts[0]] = [exts[1]]
                else:
                    ext_payoff_dict[exts[0]].append(exts[1])
    return ext_payoff_dict

def add_participation_constraints(args):
    existing_constr,ext_payoff_dict = args[0], args[1]
    constr_1 = (['pAA','pAD'],[ext_payoff_dict[('A','A')][0]-ext_payoff_dict[('N','A')][0], ext_payoff_dict[('A','D')][0]-ext_payoff_dict[('N','D')][0]])
    constr_2 = (['pDA','pDD'],[ext_payoff_dict[('D','A')][0]-ext_payoff_dict[('N','A')][0], ext_payoff_dict[('D','D')][0]-ext_payoff_dict[('N','D')][0]])
    
    constr_3 = (['pAA','pDA'],[ext_payoff_dict[('A','A')][1]-ext_payoff_dict[('A','N')][1], ext_payoff_dict[('D','A')][1]-ext_payoff_dict[('D','N')][1]])
    constr_4 = (['pAD','pDD'],[ext_payoff_dict[('A','D')][1]-ext_payoff_dict[('A','N')][1], ext_payoff_dict[('D','D')][1]-ext_payoff_dict[('D','N')][1]])
    
    existing_constr += [constr_1,constr_2,constr_3,constr_4]
    
    return existing_constr
                
def add_correleted_equilibrium_action(players,r_info):
    for i in np.arange(2,len(players)+2,2):   
        two_players = players[i-2:i]
        player_1, player_2 = two_players[0], two_players[1]
        payoff_dict = utils.generate_2_x_2_staghunt(r_info,two_players,True)
        for pl_idx in [0,1]:
            utils.check_pdrd_constraint(two_players[pl_idx].opinion,r_info.maj_opinion,pl_idx,payoff_dict)
        ext_payoff_dict = extend_table_with_noop(r_info, two_players, payoff_dict)
        ce = CorrelatedEquilibria(add_participation_constraints,ext_payoff_dict)
        ce_res = ce.solve(payoff_dict, two_players)
        player_1.ce_action = list(ce_res.keys())[0][0]
        player_2.ce_action = list(ce_res.keys())[0][1]

def construct_ce_strategy_from_regrets(longitudinal_evol):
    all_regrets = dict()
    for run_info in longitudinal_evol:
        for pl in run_info.players:
            op_val = pl.appr_val
            if op_val not in all_regrets:
                all_regrets[op_val] = dict()
            for k,v in pl.regret_map.items():
                if k in all_regrets[op_val]:
                    all_regrets[op_val][k].append(v)
                else:
                    all_regrets[op_val][k] = [v]
    all_regrets = {k:{k1:np.mean(v1) for k1,v1 in v.items()} for k,v in all_regrets.items()}
    return all_regrets

def simulate(plot_iter=None,N_iter = 1,add_ce=None):
    risk_tol, payoff_tol = constants.risk_tol, constants.payoff_tol
    if plot_iter is None:
        plot = -1
    else:
        if plot_iter == 'all':
            plot = np.inf
        else:
            plot = plot_iter
            
    print('called:'+str(N_iter),risk_tol,payoff_tol)
    caller = sys._getframe(1).f_code.co_name
    if 'sweeps' in caller:
        sweep = True
    else:
        sweep = False
    
    opinion_alpha_1, opinion_alpha_2, opinion_alpha_3, opinion_alpha_4 = constants.cen_true_distr
    longitudinal_evol = []
    
    prob_ptol_decision = lambda x : 2*x if x <=0.5 else 2-2*x          #\\
    #prob_ptol_decision = lambda x : 0.5 #
    
    
    for run_iter in np.arange(N_iter):
        if not sweep:
            print('iter:'+str(run_iter))
        if plot is np.inf or plot==run_iter:
            fig, ax = plt.subplots(nrows=2, ncols=3)
            if risk_tol is not None and payoff_tol is not None:
                fig.suptitle(str(risk_tol)+','+str(payoff_tol))
        r_info = RunInfo(run_iter)
        r_info.risk_tol = risk_tol
        r_info.payoff_tol = payoff_tol
        r_info.opinion_distr = (opinion_alpha_1, opinion_alpha_2, opinion_alpha_3, opinion_alpha_4)
        
        r_info.maj_opinion = constants.get_maj_opinion()
        r_info.ce_added = False
        if len(longitudinal_evol) == 0:
            r_info.prior_belief = constants.cen_belief
            r_info.prior_belief_alt = constants.cen_belief
        else:
            '''TODO: change this to dirichlet '''
            r_info.prior_belief = np.array([p.belief for p in longitudinal_evol[-1].players])
            r_info.prior_belief_alt = np.array([p.belief_alt for p in longitudinal_evol[-1].players])
            
        players = [Player(x) for x in np.arange(1,constants.num_players + 1)]
        ghost_players = [Player(x) for x in np.arange(-constants.num_players,0)[::-1]]
        pidx = 0
        for p,gp in zip(players,ghost_players):
            if run_iter == 0:
                '''TODO: change this to sample from dirichlet and then map this to interval'''
                p.appr_val = np.random.dirichlet()
            else:
                p.appr_val = longitudinal_evol[-1].players[pidx].appr_val
            
            '''TODO: change this to sample from dirichlet and then map this to interval'''
            gp.appr_val = np.random.beta(r_info.prior_belief[0], r_info.prior_belief[1])
            
            '''TODO: change this to dirichlet '''
            p.belief = r_info.prior_belief[0], r_info.prior_belief[1]
            p.belief_alt = r_info.prior_belief_alt[0], r_info.prior_belief_alt[1]
            
            if p.appr_val > .5:
                p.opinion = 'A'
                p.opinion_val = p.appr_val
            elif p.appr_val < .5:
                p.opinion = 'D'
                p.opinion_val = 1-p.appr_val
            else:
                p.opinion = np.random.choice(['A','D'])
                p.opinion_val = p.appr_val
            
            if gp.appr_val > .5:
                gp.opinion = 'A'
                gp.opinion_val = gp.appr_val
            elif gp.appr_val < .5:
                gp.opinion = 'D'
                gp.opinion_val = 1-gp.appr_val
            else:
                gp.opinion = np.random.choice(['A','D'])
                gp.opinion_val = gp.appr_val
            p.oth_opinion_bel = gp.opinion
            gp.oth_opinion_bel = gp.opinion
            '''
            people with minority opinion having higher minority opinion risk tolerance
            people with majority opinion having lower majority opinion risk tolerance
            '''
            if constants.risk_tol is None:
                if p.opinion != gp.opinion:
                    p.risk_tol = .9*p.opinion_val
                else:
                    p.risk_tol = .9*p.opinion_val
            else:
                p.risk_tol = constants.risk_tol
            if constants.payoff_tol is not None:
                p.payoff_tol = constants.payoff_tol
            '''
            p.maj_opinion_risk_tol =  maj_opinion_risk_tol
            p.min_opinion_risk_tol =  min_opinion_risk_tol
            '''
            pidx += 1
        
        '''TODO: change this to dirichlet '''    
        opinion_distr = {k:len([pl.opinion for pl in players if pl.opinion==k]) for k in ['A','D']}
        
        if plot is np.inf or plot==run_iter:
            ax[0,0].bar(list(opinion_distr.keys()),list(opinion_distr.values()))
            '''TODO: change this to dirichlet '''
            utils.plot_beta(opinion_alpha, opinion_beta,ax[0,2])   
            '''TODO: change this to dirichlet '''
            utils.plot_beta(opinion_alpha,opinion_beta,ax[1,0],color='black',label='true op.',linestyle='dotted')
        for i in np.arange(len(players)):   
            two_players = [players[i],ghost_players[i]]
            player_1, player_2 = two_players[0], two_players[1]
            payoff_dict = utils.generate_2_x_2_staghunt(r_info,two_players,False)
            utils.check_pdrd_constraint(two_players[0].opinion,two_players[1].opinion,0,payoff_dict)
            pne = PureNashEquilibria()
            pne_res = pne.solve(payoff_dict)
            print_str = ''+str(i)+'pne num = '+str(len(pne_res))+','+str(list(pne_res.keys()))+'\n'+str(payoff_dict)
            assert len(pne_res) == 2 , print_str
            pl = two_players[0]
            pl.oth_opinion_bel = two_players[1].opinion
            
            constants.op_mode = 'payoff_based' if np.random.binomial(1,prob_ptol_decision(pl.appr_val))==1 else 'risk_based'
            
            pl.choose_action(payoff_dict)
                        
        add_ce = add_ce if add_ce is not None else not sweep
        if add_ce:
            add_correleted_equilibrium_action(players,r_info)
            r_info.ce_added = True
        
        '''TODO: change this to multinomial '''                 
        action_distr = {k:len([pl.action for pl in players if pl.action==k]) for k in ['A','D','N']}
        if r_info.ce_added:
            ce_action_distr = {k:len([pl.ce_action for pl in players if pl.ce_action==k]) for k in ['A','D']}
        if plot is np.inf or plot==run_iter:
            ax[0,1].bar(list(action_distr.keys()),list(action_distr.values()))
            if r_info.ce_added:
                ax[1,1].bar(list(ce_action_distr.keys()),list(ce_action_distr.values()))
                print(ce_action_distr['A']-opinion_distr['A'],ce_action_distr['D']-opinion_distr['D'])
        
        ''' TODO: change the update to multinomial '''
        observed_op_histogram = [pl.appr_val for pl in players if pl.action!='N']
        obs_hist_mean, obs_hist_var = np.mean(observed_op_histogram), np.var(observed_op_histogram)
        observed_op_histogram_alt = [pl.appr_val for pl in players]
        obs_hist_mean_alt, obs_hist_var_alt = np.mean(observed_op_histogram_alt), np.var(observed_op_histogram_alt)
        mom_alpha_est = lambda u,v : u * (((u*(1-u))/v) - 1)
        mom_beta_est = lambda u,v : (1-u) * (((u*(1-u))/v) - 1)
        
        for pl in players:
            #pl.belief = (pl.belief[0]+(action_distr['A']/100), pl.belief[1]+(action_distr['D']/100))
            #pl.belief_alt =(pl.belief_alt[0]+(opinion_distr['A']/100), pl.belief_alt[1]+(opinion_distr['D']/100))
            pl.belief = (mom_alpha_est(obs_hist_mean,obs_hist_var), mom_beta_est(obs_hist_mean,obs_hist_var))
            pl.belief_alt = (mom_alpha_est(obs_hist_mean_alt,obs_hist_var_alt), mom_beta_est(obs_hist_mean_alt,obs_hist_var_alt))
        
        ''' TODO: change the belief distortion calculation to multinomial '''    
        alpha_beta_no_distortion = (np.mean([p.belief_alt[0] for p in players]), np.mean([p.belief_alt[1] for p in players]))
        alpha_beta = (np.mean([p.belief[0] for p in players]), np.mean([p.belief[1] for p in players]))
        mean_beliefs = alpha_beta[0]/(alpha_beta[0]+alpha_beta[1]) 
        mean_beliefs_no_distortion = alpha_beta_no_distortion[0]/(alpha_beta_no_distortion[0]+alpha_beta_no_distortion[1]) 
        
        r_info.distortion_of_mean = mean_beliefs - mean_beliefs_no_distortion
        r_info.alpha_beta = alpha_beta
        
        
        if plot is np.inf or plot==run_iter:
            lik_appr,like_disappr,lik_nop = [],[],[]
            act_dict = dict()
            for pidx in np.arange(constants.num_players):
                
                act_dict[longitudinal_evol[0].players[pidx].appr_val] = {'A':sum([True if rinfo.players[pidx].action=='A' else False for rinfo in longitudinal_evol]),
                                                       'D':sum([True if rinfo.players[pidx].action=='D' else False for rinfo in longitudinal_evol]),
                                                       'N':sum([True if rinfo.players[pidx].action=='N' else False for rinfo in longitudinal_evol])}
            act_dict = dict(sorted(act_dict.items()))
            for k,v in act_dict.items():
                lik_appr.append((k,v['A']/N_iter))
                like_disappr.append((k,v['D']/N_iter))
                lik_nop.append((k,v['N']/N_iter))
            ax[1,2].plot([x[0] for x in lik_appr if x[1] != 0],[x[1] for x in lik_appr if x[1] != 0],'.',color='blue',label='appr')
            ax[1,2].plot([x[0] for x in like_disappr if x[1] != 0],[x[1] for x in like_disappr if x[1] != 0],'.',color='red',label='disappr')
            ax[1,2].plot([x[0] for x in lik_nop if x[1] != 0],[x[1] for x in lik_nop if x[1] != 0],'.',color='black',label='nop')   
            ax[1,2].plot([0.5]*100,np.linspace(0,1,100),'--',color='black') 
            '''
            lik_appr = UnivariateSpline([x[0] for x in lik_appr],[x[1] for x in lik_appr])
            like_disappr = UnivariateSpline([x[0] for x in like_disappr],[x[1] for x in like_disappr])
            lik_nop = UnivariateSpline([x[0] for x in lik_nop],[x[1] for x in lik_nop])
            x = np.linspace(0,1,100)
            p = lik_appr
            ax[1,2].plot(x,p(x),'-',color='blue',label='appr')
            p = like_disappr
            ax[1,2].plot(x,p(x),'-',color='red',label='disappr')
            p = lik_nop
            ax[1,2].plot(x,p(x),'-',color='black',label='nop')  
            '''
            utils.plot_beta(np.mean([p.belief[0] for p in players]), np.mean([p.belief[1] for p in players]),ax[1,0],color='red',label='distor.')
            utils.plot_beta(alpha_beta_no_distortion[0],alpha_beta_no_distortion[1],ax[1,0],color='green',label='no distor.')
            ax[0,0].set_title('opinion histogram',fontsize='small')
            ax[0,1].set_title('selected action histogram',fontsize='small')
            ax[0,2].set_title('true opinion distr.',fontsize='small')
            ax[1,0].set_title('belief distr. change (mean,N=100)',fontsize='small')
            ax[1,0].legend(loc="lower right",fontsize="x-small")
            ax[1,2].set_title('action likelihood',fontsize='small')
            ax[1,2].legend(loc="upper left",fontsize="x-small")
            if r_info.ce_added:
                ax[1,1].set_title('CE histogram',fontsize='small')
            
            plt.show()   
        r_info.players = players    
        longitudinal_evol.append(r_info)
    return longitudinal_evol

def simulate_simple(plot_iter=None,N_iter = 1,add_ce=None):
    payoff_tol = constants.payoff_tol
    if plot_iter is None:
        plot = -1
    else:
        if plot_iter == 'all':
            plot = np.inf
        else:
            plot = plot_iter
    caller = sys._getframe(1).f_code.co_name
    if 'sweeps' in caller:
        sweep = True
    else:
        sweep = False
    
    opinion_alpha_1, opinion_alpha_2, opinion_alpha_3, opinion_alpha_4 = constants.cen_true_distr
    longitudinal_evol = []
    
    prob_ptol_decision = lambda x : 2*x if x <=0.5 else 2-2*x          #\\
    #prob_ptol_decision = lambda x : 0.5 #
    
    
    for run_iter in np.arange(N_iter):
        if not sweep:
            print('iter:'+str(run_iter))
        if plot is np.inf or plot==run_iter:
            fig, ax = plt.subplots(nrows=2, ncols=3)
        r_info = RunInfo(run_iter)
        r_info.payoff_tol = payoff_tol
        r_info.opinion_distr = [opinion_alpha_1, opinion_alpha_2, opinion_alpha_3, opinion_alpha_4]
        
        r_info.maj_opinion = constants.get_maj_opinion()
        r_info.ce_added = False
        
        if len(longitudinal_evol) == 0:
            r_info.prior_belief = constants.cen_belief
            r_info.prior_belief_alt = constants.cen_belief
        else:
            '''TODO: change this to dirichlet '''
            r_info.prior_belief = longitudinal_evol[-1].mean_belief_param
            r_info.prior_belief_alt = longitudinal_evol[-1].mean_belief_param_no_distortion
            
        players = [Player(x) for x in np.arange(1,constants.num_players + 1)]
        ghost_players = [Player(x) for x in np.arange(-constants.num_players,0)[::-1]]
        pidx = 0
        for p,gp in zip(players,ghost_players):
            if run_iter == 0:
                '''TODO: change this to sample from dirichlet '''
                utils.sample_op_val(r_info.opinion_distr,p)
            else:
                utils.sample_op_val(r_info.opinion_distr,p,longitudinal_evol[-1].players[pidx])
            
            '''TODO: change this to sample from dirichlet and then map this to interval'''
            gp.appr_val = utils.sample_op_val(r_info.prior_belief,gp)
            
            '''TODO: change this to dirichlet '''
            p.belief = r_info.prior_belief
            p.belief_alt = r_info.prior_belief_alt
            
            p.oth_opinion_bel = gp.opinion
            gp.oth_opinion_bel = gp.opinion
            
            pidx += 1
        
        '''TODO: change this to dirichlet '''    
        opinion_distr = {k:len([pl.opinion_expanded for pl in players if pl.opinion_expanded==k]) for k in utils.op_category_list}
        
        
        if plot is np.inf or plot==run_iter:
            ax[0,0].bar(list(opinion_distr.keys()),list(opinion_distr.values()))
            
        for i in np.arange(len(players)):   
            two_players = [players[i],ghost_players[i]]
            player_1, player_2 = two_players[0], two_players[1]
            payoff_dict = utils.generate_2_x_2_staghunt(r_info,two_players,False)
            pl = two_players[0]
            pl.oth_opinion_bel = two_players[1].opinion
            pl.choose_action_simplified(r_info.prior_belief)
        
        '''TODO: change this to multinomial '''                 
        action_distr = {k:len([pl.action for pl in players if pl.action==k]) for k in utils.op_category_list+['N']}
        
        if plot is np.inf or plot==run_iter:
            ax[0,1].bar(list(action_distr.keys()),list(action_distr.values()))
            
        ''' TODO: change the update to multinomial '''
        observed_op_histogram = [pl.appr_val for pl in players if pl.action!='N']
        observed_op_expanded_category_nums = [_f/constants.num_players for _f in utils.get_expanded_action_freq(players)]
        true_op_expanded_category_nums = [opinion_distr[_k]/num_players for _k in utils.op_category_list]
        for pl in players:
            pl.belief = np.add(pl.belief,np.array(observed_op_expanded_category_nums)) 
            pl.belief_alt = np.add(p.belief_alt,np.array(true_op_expanded_category_nums)) 
        
        for pl in players:
            pl.add_regrets(observed_op_expanded_category_nums)
        ''' TODO: change the belief distortion calculation to multinomial '''    
        mean_belief_param_no_distortion = np.mean(np.array([pl.belief_alt for pl in players]), axis=0)
        mean_belief_param = np.mean(np.array([pl.belief for pl in players]), axis=0)
        
        r_info.mean_belief_param_no_distortion = mean_belief_param_no_distortion
        r_info.mean_belief_param = mean_belief_param
        r_info.distortion_of_mean = mean_belief_param - mean_belief_param_no_distortion
        
        
        
        if plot is np.inf or plot==run_iter:
            lik_appr,like_disappr,lik_nop = [],[],[]
            act_dict = dict()
            for pidx in np.arange(constants.num_players):
                
                act_dict[longitudinal_evol[0].players[pidx].appr_val] = {'SD':sum([True if rinfo.players[pidx].action=='SD' else False for rinfo in longitudinal_evol]),
                                                                         'D':sum([True if rinfo.players[pidx].action=='D' else False for rinfo in longitudinal_evol]),
                                                                         'A':sum([True if rinfo.players[pidx].action=='A' else False for rinfo in longitudinal_evol]),
                                                                         'SA':sum([True if rinfo.players[pidx].action=='SA' else False for rinfo in longitudinal_evol]),
                                                                         'N':sum([True if rinfo.players[pidx].action=='N' else False for rinfo in longitudinal_evol])}
            act_dict = dict(sorted(act_dict.items()))
            for k,v in act_dict.items():
                like_disappr.append((k,v['SD']/N_iter))
                like_disappr.append((k,v['D']/N_iter))
                lik_appr.append((k,v['A']/N_iter))
                lik_appr.append((k,v['SA']/N_iter))
                lik_nop.append((k,v['N']/N_iter))
            ax[1,2].plot([x[0] for x in lik_appr if x[1] != 0],[x[1] for x in lik_appr if x[1] != 0],'.',color='blue',label='appr')
            ax[1,2].plot([x[0] for x in like_disappr if x[1] != 0],[x[1] for x in like_disappr if x[1] != 0],'.',color='red',label='disappr')
            ax[1,2].plot([x[0] for x in lik_nop if x[1] != 0],[x[1] for x in lik_nop if x[1] != 0],'.',color='black',label='nop')   
            ax[1,2].plot([0.5]*100,np.linspace(0,1,100),'--',color='black') 
            f=r_info.distortion_of_mean
            ax[0,0].set_title('opinion histogram',fontsize='small')
            ax[0,1].set_title('selected action histogram',fontsize='small')
            ax[0,2].set_title('true opinion distr.',fontsize='small')
            ax[1,0].set_title('belief distr. change (mean,N=100)',fontsize='small')
            ax[1,0].legend(loc="lower right",fontsize="x-small")
            ax[1,2].set_title('action likelihood',fontsize='small')
            ax[1,2].legend(loc="upper left",fontsize="x-small")
            if r_info.ce_added:
                ax[1,1].set_title('CE histogram',fontsize='small')
            
            plt.show()   
        r_info.players = players    
        longitudinal_evol.append(r_info)
    all_regrets = construct_ce_strategy_from_regrets(longitudinal_evol)
    return longitudinal_evol

def get_act_counts(sim_res):
    act_dict = dict()
    for pidx in np.arange(constants.num_players):
        act_dict[sim_res[0].players[pidx].appr_val] = {'A':sum([True if rinfo.players[pidx].action=='A' else False for rinfo in sim_res]),
                                                       'D':sum([True if rinfo.players[pidx].action=='D' else False for rinfo in sim_res]),
                                                       'N':sum([True if rinfo.players[pidx].action=='N' else False for rinfo in sim_res])}
    act_dict = dict(sorted(act_dict.items()))
    for k,v in act_dict.items():
        print(k,v)
        
                
def run_parameter_sweeps():
    ax = plt.axes(projection='3d')
    

    def f(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))
    
    x = np.linspace(0,1,20)
    y = np.linspace(0,1,20)
    
    #constants.cen_belief = 2.5,2
    #constants.cen_true_distr = 4,2
    
    
    xs, ys = np.meshgrid(x, y)
    Z=[]
    for i in range(len(xs)):
        for j in range(len(xs[0])):
            v = xs[i][j], ys[i][j]
            sim_res = simulate(maj_opinion_risk_tol=v[0],min_opinion_payoff_tol=v[1],plot_iter=None,N_iter=20,add_ce=False)
            #Z.append(sim_res[-1].alpha_beta[0]/(sim_res[-1].alpha_beta[0]+sim_res[-1].alpha_beta[1]))
            Z.append(sim_res[-1].alpha_beta[0]/(sim_res[-1].alpha_beta[0]+sim_res[-1].alpha_beta[1]) - (constants.cen_true_distr[0]/(constants.cen_true_distr[0]+constants.cen_true_distr[1])))
    
    # reshape Z
    
    Z = np.array(Z).reshape(xs.shape)
    # interpolate your values on the grid defined above
    f_interp = RectBivariateSpline(x,y, Z)
    X_grid, Y_grid = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    Z_grid = f_interp(np.linspace(0,1,100), np.linspace(0,1,100))
    ax.plot_surface(xs,ys,Z,
                cmap='viridis', edgecolor='none', rcount=200, ccount=200)
    ax.set_xlabel('Maj op. risk tolerance \n (lower selects N)')
    label_str = 'risk tolerance \n (lower selects N)' if constants.minority_op_mode == 'risk_based' else 'payoff tolerance \n (higher selects N)'
    ax.set_ylabel('Min op. '+label_str)
    ax.set_zlabel('Mean error of opinion')
    ax.set_title('true_op:'+str(constants.cen_true_distr)+'; belief:'+str(constants.cen_belief))

    

    plt.show()


    

def setup_state_matrices():
    P_size, N_size = 100, 4
    app_prob_lik = lambda x : 0.9-0.047*x
    PN_matrix = np.empty(shape=(P_size, N_size))
    for _p in np.arange(PN_matrix.shape[0]):
        _n_entry = np.random.choice([0,1],N_size,True,[1-app_prob_lik(_p),app_prob_lik(_p)])
        PN_matrix[_p,:] = _n_entry
    app_prob_lik = lambda x : 0.5
    DB_matrix = np.empty(shape=(P_size, N_size))
    for _p in np.arange(DB_matrix.shape[0]):
        _n_entry = np.random.choice([0,1],N_size,True,[1-app_prob_lik(_p),app_prob_lik(_p)])
        DB_matrix[_p,:] = _n_entry
    return PN_matrix, DB_matrix

def environment_transition(opinion_vect, db_param):
    risk_tol, payoff_tol = constants.risk_tol, constants.payoff_tol
    prob_ptol_decision = lambda x : 2*x if x <=0.5 else 2-2*x          #\\
    r_info = RunInfo(0)
    r_info.maj_opinion = constants.get_maj_opinion()
    players = [Player(x) for x in np.arange(1,opinion_vect.shape[0] + 1)]
    ghost_players = [Player(x) for x in np.arange(-constants.num_players,0)[::-1]]
    pidx = 0
    for p,gp in zip(players,ghost_players):
        
        p.appr_val = opinion_vect[pidx,0]
        
        gp.appr_val = np.random.beta(db_param[0],db_param[1])
        
        if p.appr_val > .5:
            p.opinion = 'A'
            p.opinion_val = p.appr_val
        elif p.appr_val < .5:
            p.opinion = 'D'
            p.opinion_val = 1-p.appr_val
        else:
            p.opinion = np.random.choice(['A','D'])
            p.opinion_val = p.appr_val
        
        if gp.appr_val > .5:
            gp.opinion = 'A'
            gp.opinion_val = gp.appr_val
        elif gp.appr_val < .5:
            gp.opinion = 'D'
            gp.opinion_val = 1-gp.appr_val
        else:
            gp.opinion = np.random.choice(['A','D'])
            gp.opinion_val = gp.appr_val
        p.oth_opinion_bel = gp.opinion
        gp.oth_opinion_bel = gp.opinion
        '''
        people with minority opinion having higher minority opinion risk tolerance
        people with majority opinion having lower majority opinion risk tolerance
        '''
        if constants.risk_tol is None:
            if p.opinion != gp.opinion:
                p.risk_tol = .9*p.opinion_val
            else:
                p.risk_tol = .9*p.opinion_val
        else:
            p.risk_tol = constants.risk_tol
        if constants.payoff_tol is not None:
            p.payoff_tol = constants.payoff_tol
        '''
        p.maj_opinion_risk_tol =  maj_opinion_risk_tol
        p.min_opinion_risk_tol =  min_opinion_risk_tol
        '''
        pidx += 1
        
    opinion_distr = {k:len([pl.opinion for pl in players if pl.opinion==k]) for k in ['A','D']}
    
    for i in np.arange(len(players)):   
        two_players = [players[i],ghost_players[i]]
        player_1, player_2 = two_players[0], two_players[1]
        payoff_dict = utils.generate_2_x_2_staghunt(r_info,two_players,False)
        utils.check_pdrd_constraint(two_players[0].opinion,two_players[1].opinion,0,payoff_dict)
        pne = PureNashEquilibria()
        pne_res = pne.solve(payoff_dict)
        print_str = ''+str(i)+'pne num = '+str(len(pne_res))+','+str(list(pne_res.keys()))+'\n'+str(payoff_dict)
        assert len(pne_res) == 2 , print_str
        pl = two_players[0]
        pl.oth_opinion_bel = two_players[1].opinion
        
        constants.op_mode = 'payoff_based' if np.random.binomial(1,prob_ptol_decision(pl.appr_val))==1 else 'risk_based'
        
        pl.choose_action(payoff_dict)
                    
    action_distr = {k:len([pl.action for pl in players if pl.action==k]) for k in ['A','D','N']}
    observed_op_histogram = [pl.appr_val for pl in players if pl.action!='N']
    obs_hist_mean, obs_hist_var = np.mean(observed_op_histogram), np.var(observed_op_histogram)
    observed_op_histogram_alt = [pl.appr_val for pl in players]
    obs_hist_mean_alt, obs_hist_var_alt = np.mean(observed_op_histogram_alt), np.var(observed_op_histogram_alt)
    mom_alpha_est = lambda u,v : u * (((u*(1-u))/v) - 1)
    mom_beta_est = lambda u,v : (1-u) * (((u*(1-u))/v) - 1)
    
    
    new_db_param = (mom_alpha_est(obs_hist_mean,obs_hist_var), mom_beta_est(obs_hist_mean,obs_hist_var))
    
    reward = (action_distr['A']+action_distr['D'])/sum(list(action_distr.values())) 
    return reward,new_db_param
    
def plot_action_selection_charts(op_strat=True):
    theta = 0.1
    cost = lambda x : entropy([x,1-x])
    util = lambda op : op if op > 0.5 else (1-op)
    bel_op = np.linspace(0,1,100)
    ops = np.linspace(0,1,100)
    ax = plt.axes(projection='3d')
    

    def f(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))
    
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    
    #constants.cen_belief = 2.5,2
    #constants.cen_true_distr = 4,2
    
    
    xs, ys = np.meshgrid(x, y)
    Z,Z2=[],[]
    for i in range(len(xs)):
        for j in range(len(xs[0])):
            op, bel_op = xs[i][j], ys[i][j]
            prob_of_N = (bel_op*(util(op)-cost(op)))/theta
            Z.append(prob_of_N)
            if op_strat:
                if prob_of_N > 1:
                    Z2.append(1)
                else:
                    Z2.append(0)
            
    
    # reshape Z
    
    Z = np.array(Z).reshape(xs.shape)
    Z2 = np.array(Z2).reshape(xs.shape)
    # interpolate your values on the grid defined above
    f_interp = RectBivariateSpline(x,y, Z)
    X_grid, Y_grid = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    Z_grid = f_interp(np.linspace(0,1,100), np.linspace(0,1,100))
    Z2_grid = f_interp(np.linspace(0,1,10), np.linspace(0,1,10))
    if not op_strat:
        ax.plot_surface(xs,ys,Z,
                cmap='viridis', edgecolor='none', rcount=200, ccount=200)
    else:
        ax.plot_surface(xs,ys,Z2-1,cmap='cividis', edgecolor='none', rcount=20, ccount=20)
    ax.set_xlabel('opinion value')
    ax.set_zlabel('Opinion selection ratio')
    ax.set_ylabel('Descriptive supporting belief')
    plt.show()
'''
constants.cen_belief = [2,2,2,2]
constants.cen_true_distr = 3,2,1,2 
constants.payoff_tol, constants.risk_tol = 0.3, None
sim_res = simulate_simple(499,500,False)
get_act_counts(sim_res)
print('distortion',sim_res[-1].distortion_of_mean)

'''
#PN_matrix, DB_matrix = setup_state_matrices()
#opinion_vect = np.mean(PN_matrix,axis=1).reshape(PN_matrix.shape[0],1)
#db_vect = np.sum(PN_matrix,axis=1).reshape(PN_matrix.shape[0],1)
#reward,new_db_param = environment_transition(opinion_vect, (2,2))
#print(reward,new_db_param)
#fig, ax = plt.subplots(1,1)
#ax.matshow(PN_matrix, cmap='bwr_r')
#ax[1].matshow(np.random.beta(2,2,(20,1)), cmap='bwr_r')
#ax[2].matshow(np.random.beta(new_db_param[0],new_db_param[1],(20,1)), cmap='bwr_r')
#plt.show()
'''


#run_parameter_sweeps()

plot_action_selection_charts(True)
'''

class SingleContextModel():
    
    def simple_repeated_interaction(self):
        class Player():
            def __init__(self,op,u_bar):
                self.op = op
                self.u_bar = u_bar
        
        u_bar = 0.3
        cost = lambda x : entropy([x,1-x])
        util = lambda op : op if op > 0.5 else (1-op)
        theta_prior = (2,1)
        theta_prime = theta_prior
        op_distr = None
        players = [Player(o,u_bar) for o in np.random.uniform(size=100)]
        bels = []
        print('prior:',theta_prior[0]/sum(theta_prior))
        for run_idx in np.arange(300):
            '''
            if run_idx ==0:
                plt.hist([pl.op for pl in players],bins=10)
                plt.show()
            '''
            for pl in players:
                if pl.op>=0.5:
                    bel_op = theta_prime[0]/sum(theta_prime)
                else:
                    bel_op = 1-(theta_prime[0]/sum(theta_prime))
                prob_of_N = (bel_op*util(pl.op))/u_bar
                pl.act = 'e' if prob_of_N > 1 else 'n'
            op_distr = np.mean([pl.op if pl.op > 0.5 else 1-pl.op for pl in players if pl.act=='e'])
            theta_prime_rate = np.mean([pl.op for pl in players if pl.act=='e'])
            h = theta_prime_rate*10
            theta_prime = (theta_prime[0]+h,theta_prime[1]+(10-h))
            print(run_idx,theta_prime[0]/sum(theta_prime),Counter([pl.act for pl in players]))
            '''
            plt.hist([pl.op for pl in players if pl.act=='e'],bins=10)
            plt.show()
            '''
            bels.append(theta_prime[0]/sum(theta_prime))
        theta_prior_h = theta_prior[0]/sum(theta_prior)
        eq_ratio = ((u_bar/theta_prior_h)-0.5)/((u_bar/theta_prior_h)+(u_bar/(1-theta_prior_h))-1)
        
        print(theta_prime[0]/sum(theta_prime),op_distr)
        return bels


class HeterogenousContextSimulation():
    
    def sample_multivariate_gaussian(self,mean,corr):
        cov_val =  corr * np.sqrt(0.07) * np.sqrt(0.01)
        cov = np.array([[.07, cov_val], [cov_val, .01]])
        pt = np.random.multivariate_normal([0, 0], cov)
        return pt

    
    def draw_opinion_samples(self,app_lik,N,corr=[0.7,-0.6,0.4]):
        disapp_lik = 1- app_lik
        #covs = [c*]
        sa = np.random.uniform(low=0.2,high=0.8)*app_lik
        a = app_lik - sa
        sd = np.random.uniform(low=0.2,high=0.8)*disapp_lik
        d = disapp_lik - sd
        f = interpolate.interp1d([0,0.25,0.75,1],[sd,d,a,sa])
        xnew = np.arange(0, 1.01, 0.01)
        ynew = f(xnew) 
        '''
        plt.plot([0,0.25,0.75,1],[sd,d,a,sa], 'o')
        plt.plot( xnew, ynew, '-')
        plt.title(str(app_lik))
        plt.show()
        '''
        ynew = [x/sum(ynew) for x in ynew]
        ops = np.random.choice(a=xnew,size=N,p=ynew)
        #plt.hist(ops, density=True, bins=N)
        #plt.show()
        return ops

    def generate_another_context_opinion(self,base_context_app_lik,new_context_app_lik,corr_deg):
        #https://stats.stackexchange.com/questions/284996/generating-correlated-binomial-random-variables
        base_context_disapp_lik = 1 - base_context_app_lik
        #new_context_app_lik = (base_context_app_lik*corr_deg) + ((1-corr_deg)*base_context_disapp_lik)
        new_context_disapp_lik = 1 - new_context_app_lik
        p,q,rho = base_context_app_lik, new_context_app_lik, corr_deg
        p1 = rho * np.sqrt(p * q * (1 - p) * (1 - q)) + (1 - p) * (1 - q)
        p2 = 1 - p - p1
        p3 = 1 - q - p1
        p4 = p1 + p + q - 1
        join_distr = np.array([[p1,p2],[p3,p4]])
        return join_distr
    
    def __init__(self):
        util = lambda op : op if op > 0.5 else (1-op)
        theta = 0.3
        norm_contexts_distr = {'n1':0.25,'n2':0.25,'n3':0.25,'n4':0.25}
        players = [Player(i) for i in np.arange(100)]
        players_private_contexts = np.random.choice(a=list(norm_contexts_distr.keys()),size=100,p=list(norm_contexts_distr.values()))
        for idx,op in enumerate(players_private_contexts): players[idx].norm_context = players_private_contexts[idx]
        opinions = genfromtxt('.//r_scripts//samples.csv', delimiter=',')
        '''
        opinions_n1 = np.random.choice([0,1],size=(100,),p=[0.4,0.6])
        opinions = opinions_n1
        f = lambda x,corr_prob: x if np.random.random() < corr_prob else 1-x
        for n in [0.6,0.5,0.2]:
            _op_vect = np.array(f(np.array(opinions_n1),n))
            opinions = np.vstack((opinions,_op_vect))
        opinions = opinions.T
        '''
        opinion_marginals = np.sum(opinions,axis=0)/100
        bels = {x:opinion_marginals[nidx] for nidx,x in enumerate(['n1','n2','n3','n4'])}
        for n_idx,norm_context in enumerate(['n1','n2','n3','n4']):
            ops = opinions[:,n_idx]
            for idx,op in enumerate(ops): players[idx].opinion[norm_context] = np.random.uniform(0.5,1) if op == 1 else np.random.uniform(0,0.5)
            for idx,op in enumerate(ops): players[idx].opinion_val[norm_context] = util(players[idx].opinion[norm_context])
            
        for pl in players:
            
            if pl.opinion[pl.norm_context] >= 0.5:
                op = 1
            else:
                op = 0
            pl.op = op
            pl.bel_op = bels[pl.norm_context] if op == 1 else 1-bels[pl.norm_context]
            prob_of_N = (pl.bel_op*util(pl.opinion[pl.norm_context]))/theta
            if prob_of_N > 1:
                pl.action = pl.opinion[pl.norm_context]
            else:
                pl.action = -1
            pl.bel_op_all_norms = {n:bels[n] if op == 1 else 1-bels[n] for n in ['n1','n2','n3','n4']}
            exp_n = 0
            for n,bel_for_n in pl.bel_op_all_norms.items():
                exp_prob_n_for_bel = norm_contexts_distr[n]*(bel_for_n*util(pl.opinion[pl.norm_context]))/theta
                exp_n += exp_prob_n_for_bel
            if exp_n > 1:
                pl.action_all_context_bels = pl.opinion[pl.norm_context]
            else:
                pl.action_all_context_bels = -1
                
                
        true_obs = np.sum([True for pl in players if pl.action >= 0.5])/np.sum([True for pl in players if pl.action != -1])
        if true_obs > 0.5:
            true_op_obs = 1
        else:
            true_op_obs = 0
        true_obs_all_contexts = np.sum([True for pl in players if pl.action_all_context_bels >= 0.5])/100
        regret_map, regret_map_all_contexts = {},{}
        payoff_map, payoff_map_all_contexts = {},{}
        for pl in players:
            if pl.opinion[pl.norm_context] >= 0.5:
                op = 1
            else:
                op = 0
            bels_from_obs = true_obs if op == 1 else 1-true_obs
            bels_from_obs_all_contextx = true_obs if op == 1 else 1-true_obs
            prob_of_N = (bels_from_obs*util(pl.opinion[pl.norm_context]))/theta
            if prob_of_N > 1 and pl.action == -1:
                pl.regret = pl.opinion_val[pl.norm_context]
            elif prob_of_N <= 1 and pl.action > -1:
                pl.regret = theta
            else:
                pl.regret = 0
            
            prob_of_N = (bels_from_obs_all_contextx*util(pl.opinion[pl.norm_context]))/theta
            if prob_of_N > 1 and pl.action_all_context_bels == -1:
                pl.regret_all_context_bels = pl.opinion_val[pl.norm_context]
            elif prob_of_N <= 1 and pl.action_all_context_bels > -1:
                pl.regret_all_context_bels = theta
            else:
                pl.regret_all_context_bels = 0
            if pl.action == -1:
                pl.payoff = 0
            else:
                if pl.op == true_op_obs:
                    pl.payoff = 1
                else:
                    pl.payoff = 1
            if pl.action_all_context_bels == -1:
                pl.payoff_all_context_bels = 0
            else:
                if pl.op == true_op_obs:
                    pl.payoff_all_context_bels = 1
                else:
                    pl.payoff_all_context_bels = 1
            
            if pl.norm_context not in regret_map:
                regret_map[pl.norm_context] = []
                regret_map_all_contexts[pl.norm_context] = []
            if pl.norm_context not in payoff_map:
                payoff_map[pl.norm_context] = []
                payoff_map_all_contexts[pl.norm_context] = []
            regret_map[pl.norm_context].append(pl.regret)
            regret_map_all_contexts[pl.norm_context].append(pl.regret_all_context_bels)
            payoff_map[pl.norm_context].append(pl.payoff)
            payoff_map_all_contexts[pl.norm_context].append(pl.payoff_all_context_bels)
        for k,v in regret_map.items():
            print(k,np.mean(v))
        regret_map = {k:np.mean(v) for k,v in regret_map.items()}
        payoff_map = {k:np.mean(v) for k,v in payoff_map.items()}
        print('total regret',np.sum(list(regret_map.values())))
        self.total_regret_self_ref = np.sum(list(regret_map.values()))
        print('-----')
        for k,v in regret_map_all_contexts.items():
            print(k,np.mean(v))
        regret_map_all_contexts = {k:np.mean(v) for k,v in regret_map_all_contexts.items()}
        payoff_map_all_contexts = {k:np.mean(v) for k,v in payoff_map_all_contexts.items()}
        print('total regret',np.sum(list(regret_map_all_contexts.values())))
        self.total_regret_comm_ref = np.sum(list(regret_map_all_contexts.values()))
        self.regret_map = regret_map
        self.regret_map_all_contexts = regret_map_all_contexts
        self.payoff_map = payoff_map
        self.payoff_map_all_contexts = payoff_map_all_contexts
        '''
        fig, ax = plt.subplots()
        for norm_ctx,color in zip(['n1','n2','n3','n4'],['red','blue','green','cyan']):
            
            pl_ops = [pl.opinion[pl.norm_context] for pl in players if pl.norm_context==norm_ctx]
            #pl_acts = [pl.action for pl in players if pl.norm_context==norm_ctx]
            pl_regrets = [pl.regret for pl in players if pl.norm_context==norm_ctx]
            ax.plot(pl_ops,pl_regrets,'.',label=norm_ctx,color=color)
        ax.legend()
        plt.show()
        '''
def heterogen_sim_run():
    data,data_payoff = {},{}       
    for iter in np.arange(100):           
        print('ITER------------------->',iter)
        sim_run = HeterogenousContextSimulation()
        for k,v in sim_run.regret_map.items():
            if k not in data:
                data[k] = [[],[]]
            data[k][0].append(v)
            
        for k,v in sim_run.regret_map_all_contexts.items():
            if k not in data:
                data[k] = [[],[]]
            data[k][1].append(v)
            
        for k,v in sim_run.payoff_map.items():
            if k not in data_payoff:
                data_payoff[k] = [[],[]]
            data_payoff[k][0].append(v)
            
        for k,v in sim_run.payoff_map_all_contexts.items():
            if k not in data_payoff:
                data_payoff[k] = [[],[]]
            data_payoff[k][1].append(v)
        
    
    for i in np.arange(2):       
        if i== 0:
            data_a = [data[k][0] for k in ['n1','n2','n3','n4']]
            data_b = [data[k][1] for k in ['n1','n2','n3','n4']]
        else:
            data_a = [data_payoff[k][0] for k in ['n1','n2','n3','n4']]
            data_b = [data_payoff[k][1] for k in ['n1','n2','n3','n4']]
        
        ticks = ['n1','n2','n3','n4']
        
        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)
        
        plt.figure()
        plt.title('Regret' if i ==0 else 'Payoff')
        bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
        bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
        set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
        set_box_color(bpr, '#2C7BB6')
        
        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c='#D7191C', label='Self-ref')
        plt.plot([], c='#2C7BB6', label='Community-ref')
        if i==0:
            plt.legend()
        
        plt.xticks(range(0, len(ticks) * 2, 2), ticks)
        plt.xlim(-2, len(ticks)*2)
    plt.show()




























