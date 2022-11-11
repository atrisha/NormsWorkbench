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
from constants import op_mode


class Player():
    
    def __init__(self,id,real_p=True):
        self.id = id
        if real_p:
            self.shadow_player = Player(-id,False)
    
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
        effort_fun = lambda x : 0
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
    
    def choose_action_cost_based(self,payoff_dict):
        f=1

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
    '''TODO: change this to dirichlet '''
    opinion_alpha, opinion_beta = constants.cen_true_distr
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
        r_info.opinion_distr = (opinion_alpha, opinion_beta)
        '''TODO: change this function '''
        r_info.maj_opinion = constants.get_maj_opinion()
        r_info.ce_added = False
        if len(longitudinal_evol) == 0:
            r_info.prior_belief = constants.cen_belief
            r_info.prior_belief_alt = constants.cen_belief
        else:
            '''TODO: change this to dirichlet '''
            r_info.prior_belief = (np.mean([p.belief[0] for p in longitudinal_evol[-1].players]), np.mean([p.belief[1] for p in longitudinal_evol[-1].players]))
            r_info.prior_belief_alt = (np.mean([p.belief_alt[0] for p in longitudinal_evol[-1].players]), np.mean([p.belief_alt[1] for p in longitudinal_evol[-1].players]))
            
        players = [Player(x) for x in np.arange(1,constants.num_players + 1)]
        ghost_players = [Player(x) for x in np.arange(-constants.num_players,0)[::-1]]
        pidx = 0
        for p,gp in zip(players,ghost_players):
            if run_iter == 0:
                '''TODO: change this to sample from dirichlet and then map this to interval'''
                p.appr_val = np.random.beta(opinion_alpha, opinion_beta)
            else:
                p.appr_val = longitudinal_evol[-1].players[pidx].appr_val
            
            gp.appr_val = np.random.beta(r_info.prior_belief[0], r_info.prior_belief[1])
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
            
        opinion_distr = {k:len([pl.opinion for pl in players if pl.opinion==k]) for k in ['A','D']}
        
        if plot is np.inf or plot==run_iter:
            ax[0,0].bar(list(opinion_distr.keys()),list(opinion_distr.values()))
            utils.plot_beta(opinion_alpha, opinion_beta,ax[0,2])   
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
                        
        action_distr = {k:len([pl.action for pl in players if pl.action==k]) for k in ['A','D','N']}
        if r_info.ce_added:
            ce_action_distr = {k:len([pl.ce_action for pl in players if pl.ce_action==k]) for k in ['A','D']}
        if plot is np.inf or plot==run_iter:
            ax[0,1].bar(list(action_distr.keys()),list(action_distr.values()))
            if r_info.ce_added:
                ax[1,1].bar(list(ce_action_distr.keys()),list(ce_action_distr.values()))
                print(ce_action_distr['A']-opinion_distr['A'],ce_action_distr['D']-opinion_distr['D'])
        
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
    P_size, N_size = 20, 20
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
    

constants.cen_belief = (0.5,0.5 )
constants.cen_true_distr = 0.5,0.5 
constants.payoff_tol, constants.risk_tol = 0.6, None
sim_res = simulate(499,500,False)
get_act_counts(sim_res)
print('distortion',sim_res[-1].alpha_beta[0]/(sim_res[-1].alpha_beta[0]+sim_res[-1].alpha_beta[1]) - (constants.cen_true_distr[0]/(constants.cen_true_distr[0]+constants.cen_true_distr[1])))

'''
PN_matrix, DB_matrix = setup_state_matrices()
opinion_vect = np.mean(PN_matrix,axis=1).reshape(PN_matrix.shape[0],1)
db_vect = np.sum(PN_matrix,axis=1).reshape(PN_matrix.shape[0],1)
reward,new_db_param = environment_transition(opinion_vect, (2,2))
print(reward,new_db_param)
fig, ax = plt.subplots(1,3)
ax[0].matshow(opinion_vect, cmap='bwr_r')
ax[1].matshow(np.random.beta(2,2,(20,1)), cmap='bwr_r')
ax[2].matshow(np.random.beta(new_db_param[0],new_db_param[1],(20,1)), cmap='bwr_r')
plt.show()



#run_parameter_sweeps()
'''


































