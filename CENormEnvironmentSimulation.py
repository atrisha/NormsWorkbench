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

class Player():
    
    def __init__(self,id):
        self.id = id

class RunInfo():
    
    def __init__(self,iter):
        self.iter = iter


def simulate(maj_opinion_risk_tol=None,min_opinion_payoff_tol=None,plot=True,N_iter = 1):
    
    opinion_alpha, opinion_beta = constants.cen_true_distr
    longitudinal_evol = []
    for run_iter in np.arange(N_iter):
        if plot:
            fig, ax = plt.subplots(nrows=2, ncols=3)
        r_info = RunInfo(run_iter)
        r_info.maj_opinion_risk_tol = maj_opinion_risk_tol
        r_info.min_opinion_payoff_tol = min_opinion_payoff_tol
        r_info.opinion_distr = (opinion_alpha, opinion_beta)
        if len(longitudinal_evol) == 0:
            r_info.prior_belief = constants.cen_belief
        else:
            r_info.prior_belief = (np.mean([p.belief[0] for p in longitudinal_evol[-1].players]), np.mean([p.belief[1] for p in longitudinal_evol[-1].players]))
        players = [Player(x) for x in np.arange(1,101)]
        for p in players:
            p.appr_val = np.random.beta(opinion_alpha, opinion_beta)
            #p.belief = (np.random.uniform(0,opinion_alpha),np.random.uniform(0,opinion_beta))
            p.belief = r_info.prior_belief
            if p.appr_val > .5:
                p.opinion = 'A'
                p.opinion_val = p.appr_val
            elif p.appr_val < .5:
                p.opinion = 'D'
                p.opinion_val = 1-p.appr_val
            else:
                p.opinion = np.random.choice(['A','D'])
                p.opinion_val = p.appr_val
            disappr_belief = scipy.stats.beta.cdf(0.5,p.belief[0],p.belief[1])
            if disappr_belief <.5:
                p.oth_opinion_bel = 'A'
            elif disappr_belief >.5:
                p.oth_opinion_bel = 'D'
            else:
                p.oth_opinion_bel = np.random.choice(['A','D'])
            p.maj_opinion_risk_tol = np.random.beta(150,100) if maj_opinion_risk_tol is None else maj_opinion_risk_tol
            p.min_opinion_payoff_tol = np.random.beta(50,100) if min_opinion_payoff_tol is None else min_opinion_payoff_tol
        opinion_distr = {k:len([pl.opinion for pl in players if pl.opinion==k]) for k in ['A','D']}
        
        if plot:
            ax[0,0].bar(list(opinion_distr.keys()),list(opinion_distr.values()))
            utils.plot_beta(np.mean([p.belief[0] for p in players]), np.mean([p.belief[1] for p in players]),ax[1,0],color='blue',label="prior bel.")
            utils.plot_beta(opinion_alpha, opinion_beta,ax[0,2])   
            utils.plot_beta(opinion_alpha,opinion_beta,ax[1,0],color='black',label='true op.',linestyle='dotted')
        for i in np.arange(2,len(players)+2,2):   
            two_players = players[i-2:i]
            player_1, player_2 = two_players[0], two_players[1]
            payoff_dict = utils.generate_2_x_2_staghunt(two_players)
            for pl_idx in [0,1]:
                utils.check_pdrd_constraint(two_players[pl_idx].opinion,two_players[pl_idx].oth_opinion_bel,pl_idx,payoff_dict)
            pne = PureNashEquilibria()
            pne_res = pne.solve(payoff_dict)
            print_str = ''+str(i)+'pne num = '+str(len(pne_res))+','+str(list(pne_res.keys()))+'\n'+str(payoff_dict)
            assert len(pne_res) == 2 , print_str
            ce = CorrelatedEquilibria()
            ce_res = ce.solve(payoff_dict, two_players)
            player_1.ce_action = list(ce_res.keys())[0][0]
            player_2.ce_action = list(ce_res.keys())[0][1]
            for pl_idx,pl in enumerate(two_players):
                if pl.opinion == pl.oth_opinion_bel:
                    risk_of_opinion = utils.get_act_risk(payoff_dict, pl_idx, pl.opinion)
                    if risk_of_opinion > pl.maj_opinion_risk_tol:
                        pl.action = 'N'
                    else:
                        pl.action = pl.opinion
                else:
                    pl_not_op = utils.get_oth_opinion(pl.opinion)
                    payoff_not_opinion = payoff_dict[(pl_not_op,pl_not_op)][pl_idx]
                    risk_of_opinion = utils.get_act_risk(payoff_dict, pl_idx, pl.opinion)
                    if constants.minority_op_mode == 'payoff_based':
                        if payoff_not_opinion < pl.min_opinion_payoff_tol:
                            pl.action = 'N'
                        else:
                            pl.action = pl_not_op
                    
                    if constants.minority_op_mode == 'risk_based':
                        if risk_of_opinion > pl.min_opinion_payoff_tol:
                            pl.action = 'N'
                        else:
                            pl.action = pl.opinion
                    
        action_distr = {k:len([pl.action for pl in players if pl.action==k]) for k in ['A','D','N']}
        ce_action_distr = {k:len([pl.ce_action for pl in players if pl.ce_action==k]) for k in ['A','D']}
        if plot:
            ax[0,1].bar(list(action_distr.keys()),list(action_distr.values()))
            ax[1,1].bar(list(ce_action_distr.keys()),list(ce_action_distr.values()))
        
        for pl in players:
            pl.belief =(pl.belief[0]+(action_distr['A']/100), pl.belief[1]+(action_distr['D']/100))
            pl.belief_no_distortion =(pl.belief[0]+(opinion_distr['A']/100), pl.belief[1]+(opinion_distr['D']/100))
        alpha_beta_no_distortion = (np.mean([p.belief_no_distortion[0] for p in players]), np.mean([p.belief_no_distortion[1] for p in players]))
        alpha_beta = (np.mean([p.belief[0] for p in players]), np.mean([p.belief[1] for p in players]))
        mean_beliefs = alpha_beta[0]/(alpha_beta[0]+alpha_beta[1]) 
        mean_beliefs_no_distortion = alpha_beta_no_distortion[0]/(alpha_beta_no_distortion[0]+alpha_beta_no_distortion[1]) 
        r_info.players = players
        r_info.distortion_of_mean = mean_beliefs - mean_beliefs_no_distortion
        r_info.alpha_beta = alpha_beta
        print('called:'+str(run_iter),maj_opinion_risk_tol,min_opinion_payoff_tol,alpha_beta,alpha_beta_no_distortion,r_info.distortion_of_mean)
        if plot:
            utils.plot_beta(np.mean([p.belief[0] for p in players]), np.mean([p.belief[1] for p in players]),ax[1,0],color='red',label='distor.')
            if N_iter == 1:
                utils.plot_beta(alpha_beta_no_distortion[0],alpha_beta_no_distortion[1],ax[1,0],color='green',label='no distor.')
            ax[0,0].set_title('opinion histogram',fontsize='small')
            ax[0,1].set_title('selected action histogram',fontsize='small')
            ax[0,2].set_title('true opinion distr.',fontsize='small')
            ax[1,0].set_title('belief distr. change (mean,N=100)',fontsize='small')
            ax[1,0].legend(loc="lower right",fontsize="x-small")
            ax[1,1].set_title('CE histogram',fontsize='small')
            plt.show()   
        longitudinal_evol.append(r_info)
    return longitudinal_evol
                
def run_parameter_sweeps():
    ax = plt.axes(projection='3d')
    

    def f(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))
    
    x = np.linspace(0,1,10)
    y = np.linspace(0,1,10)
    
    #constants.cen_belief = 2.5,2
    #constants.cen_true_distr = 4,2
    
    
    xs, ys = np.meshgrid(x, y)
    Z=[]
    for i in range(len(xs)):
        for j in range(len(xs[0])):
            v = xs[i][j], ys[i][j]
            sim_res = simulate(v[0],v[1],False,N_iter=20)
            #Z.append(sim_res[-1].alpha_beta[0]/(sim_res[-1].alpha_beta[0]+sim_res[-1].alpha_beta[1]))
            Z.append(sim_res[-1].alpha_beta[0]/(sim_res[-1].alpha_beta[0]+sim_res[-1].alpha_beta[1]) - (3/5))
    
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
    ax.set_title('true_op:'+str(constants.cen_true_distr)+'; belief:'+str(constants.cen_belief))

    

    plt.show()

constants.cen_belief = (1,1)
constants.cen_true_distr = 3,2  
constants.minority_op_mode = 'risk_based'
simulate(0.3,0.2,True,20)

#run_parameter_sweeps()



































