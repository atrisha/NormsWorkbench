'''
Created on 7 Sept 2022

@author: Atrisha
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import itertools
from Equilibria import CorrelatedEquilibria, PureNashEquilibria

def plot_beta(a,b,ax=None,color=None,label=None,linestyle='-'):
    x = np.linspace(beta.ppf(0.01, a, b),beta.ppf(0.99, a, b), 100)
    plot_color = 'black' if color is None else color
    if ax is None:
        plt.figure(figsize=(7,7))
        plt.xlim(0, 1)
        plt.plot(x, beta.pdf(x, a, b), linestyle='-', color=plot_color)
        plt.title('Beta Distribution', fontsize='15')
        plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
        plt.ylabel('Probability', fontsize='15')
        plt.show()
    else:
        ax.set_xlim(0, 1)
        ax.plot(x, beta.pdf(x, a, b), linestyle=linestyle if linestyle is not None else '-', color=plot_color,label=label)
        #ax.set_xlabel('Values of Random Variable X (0, 1)', fontsize='15')
        #ax.set_ylabel('Probability', fontsize='15')
        
    
def eq(a,b):
    return a==b 

def less(a,b):
    return a < b

def leq(a,b):
    return a <= b

def generate_2_x_2_staghunt(r_info,players,for_recommendation_ex_post):
    get_oth_opinion = lambda x : 'A' if x == 'D' else 'D'
    num_players = 2
    all_strats = list(itertools.product(*[['A','D'],['A','D']]))
    max_payoff = 1
    payoff_dict = dict()
    for pl_idx, player in enumerate(players):
        assert(player.opinion_val<=1)
        strat_player_opinion = ''.join([player.opinion]*num_players)
        strat_not_player_opinion = ''.join([get_oth_opinion(player.opinion)]*num_players)
        player.act_payoffs = {strat_player_opinion:player.opinion_val,strat_not_player_opinion:max_payoff-player.opinion_val}
        if for_recommendation_ex_post:
            minorty_op_check = True if player.opinion != r_info.maj_opinion else False
        else:
            minorty_op_check = True if player.opinion != player.oth_opinion_bel else False
        if minorty_op_check:
            ''' Player descriptive opinion belief in minority. So risk of sharing not opinion is less that sharing opinion '''
            r_player_opinion = np.random.uniform(max(0,player.opinion_val-(1-player.opinion_val)),max_payoff)
            r_not_player_opinion = np.random.uniform()*r_player_opinion
        else:
            ''' Player descriptive opinion belief in majority '''
            r_not_player_opinion = np.random.uniform(player.opinion_val-(1-player.opinion_val),max_payoff)
            r_player_opinion = np.random.uniform(player.opinion_val-(1-player.opinion_val),r_not_player_opinion)
        strat_player_opinion_defection = ''.join([player.opinion if i==pl_idx else get_oth_opinion(player.opinion) for i in np.arange(num_players)])
        strat_not_player_opinion_defection = ''.join([get_oth_opinion(player.opinion) if i==pl_idx else player.opinion for i in np.arange(num_players)])
        player.act_payoffs[strat_player_opinion_defection] = player.act_payoffs[strat_player_opinion] - r_player_opinion
        player.act_payoffs[strat_not_player_opinion_defection] = player.act_payoffs[strat_not_player_opinion] - r_not_player_opinion
    payoff_dict = {}
    for k in list(players[0].act_payoffs.keys()):
        payoff_dict[tuple(list(k))] = [players[i].act_payoffs[k] for i in np.arange(num_players)]
    #pne = PureNashEquilibria()
    #pne_res = pne.solve(payoff_dict)
    return payoff_dict

get_oth_opinion = lambda x : 'A' if x == 'D' else 'D'

def check_pdrd_constraint(pl_op,pl_bel,pl_idx,payoff_dict_inp):
    payoff_dict = {''.join(list(k)):v for k,v in payoff_dict_inp.items()}
    try:
        if pl_op != pl_bel:
            if pl_idx == 0:
                assert abs(payoff_dict[''.join([pl_op,pl_op])][pl_idx]-payoff_dict[''.join([pl_op,get_oth_opinion(pl_op)])][pl_idx]) >= abs(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx]-payoff_dict[''.join([get_oth_opinion(pl_op),pl_op])][pl_idx])
            else:
                assert abs(payoff_dict[''.join([pl_op,pl_op])][pl_idx]-payoff_dict[''.join([get_oth_opinion(pl_op),pl_op])][pl_idx]) >= abs(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx]-payoff_dict[''.join([pl_op,get_oth_opinion(pl_op)])][pl_idx])   
            #assert payoff_dict[''.join([pl_op,pl_op])][pl_idx] < payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx], str(payoff_dict)+'\n'+str([pl_op,pl_bel,pl_idx])
        else:
            if pl_idx == 0:
                assert abs(payoff_dict[''.join([pl_op,pl_op])][pl_idx]-payoff_dict[''.join([pl_op,get_oth_opinion(pl_op)])][pl_idx]) <= abs(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx]-payoff_dict[''.join([get_oth_opinion(pl_op),pl_op])][pl_idx])
            else:
                assert abs(payoff_dict[''.join([pl_op,pl_op])][pl_idx]-payoff_dict[''.join([get_oth_opinion(pl_op),pl_op])][pl_idx]) <= abs(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx]-payoff_dict[''.join([pl_op,get_oth_opinion(pl_op)])][pl_idx])
        assert payoff_dict[''.join([pl_op,pl_op])][pl_idx] >= payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx], str(payoff_dict[''.join([pl_op,pl_op])][pl_idx]) +  str(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx])
    except AssertionError:
        f=1
        raise
        
def get_rd_eq(payoff_dict, pl_idx, pl):
    pl_op = pl.opinion
    pl_not_op = get_oth_opinion(pl_op)
    risk_of_op_val = abs(payoff_dict[(pl_op,pl_op)][pl_idx] - payoff_dict[(pl_op,get_oth_opinion(pl_op))][pl_idx])
    risk_of_not_op_val = abs(payoff_dict[(pl_not_op,pl_not_op)][pl_idx] - payoff_dict[(pl_not_op,get_oth_opinion(pl_not_op))][pl_idx])
    if pl_idx == 0:
        if  risk_of_op_val > risk_of_not_op_val : \
                   return (pl_not_op,pl_not_op)
        else:
            return (pl_op,pl_op)
    else:
        if abs(payoff_dict[(pl_op,pl_op)][pl_idx] - payoff_dict[(get_oth_opinion(pl_op),pl_op)][pl_idx]) \
               > abs(payoff_dict[(pl_not_op,pl_not_op)][pl_idx] - payoff_dict[(get_oth_opinion(pl_not_op),pl_not_op)][pl_idx]): \
                   return (pl_not_op,pl_not_op)
        else:
            return (pl_op,pl_op)
def get_act_risk(payoff_dict, pl_idx, act):
    if pl_idx == 0:
        risk_of_act_val = abs(payoff_dict[(act,act)][pl_idx] - payoff_dict[(act,get_oth_opinion(act))][pl_idx])
    else:
        risk_of_act_val = abs(payoff_dict[(act,act)][pl_idx] - payoff_dict[(get_oth_opinion(act),act)][pl_idx])
    return risk_of_act_val

def get_pd_eq(payoff_dict, pl_idx, pl):
    pl_op = pl.opinion
    pl_not_op = get_oth_opinion(pl_op)
    return (pl_op,pl_op) if payoff_dict[(pl_op,pl_op)][pl_idx] > payoff_dict[(pl_not_op,pl_not_op)][pl_idx] else (pl_not_op,pl_not_op)
        
                   
    
               
        
    
        
    
    

    
    
