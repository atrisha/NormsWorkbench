'''
Created on 9 May 2023

@author: Atrisha
'''
import numpy as np
import math
from whiteboard.copula import plot_3_way_corr
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from scipy.stats import binom

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

gamma = 0.1

class CommunityAgent():
    
    def __init__(self,op_vect):
        self.opinion_vect = op_vect
        self.u_bar = 0.2
        self.mean_op = np.mean(self.opinion_vect)
        
        
    def expr_eval(self,theta_s):
        mean_op = self.mean_op
        expr = False
        if mean_op >=0.5:
            expr = True if self.u_bar/theta_s < mean_op else False
        else:
            expr = True if 1-(self.u_bar/(1-theta_s)) > mean_op else False
        exp_util_expr = mean_op*theta_s if mean_op >=0.5 else (1-mean_op)*(1-theta_s)
        self.expr= expr
        return exp_util_expr,expr
    
    def intervening_cost(self, theta_s, confl_agents):
        mean_op = self.mean_op
        diagreement_prob = 2*theta_s*(1-theta_s)
        l,h = np.clip(1-(0.2/(1-theta_s)),0,0.5), np.clip(0.2/theta_s,0.5,1)
        l,h = l/2,(h+1)/2 # the expected opinion for disagreement
        cost_Y = min([abs(mean_op-confl_agents[0].mean_op),abs(mean_op-confl_agents[1].mean_op)])
        intv_cost = cost_Y
        return intv_cost
    
    def intervening_gain(self, theta_s, confl_agents):
        l,h = np.clip(1-(0.2/(1-theta_s)),0,0.5), np.clip(0.2/theta_s,0.5,1)
        l,h = l/2,(h+1)/2 # the expected opinion for disagreement
        mean_op = self.mean_op
        theta_prime = l if abs(mean_op-l)<abs(mean_op-h) else h
        new_theta = (0.95*theta_s) + (0.05*theta_prime)
        if self.expr:
            exp_expr_gain = max(0,self.expr_eval(new_theta)[0] - self.u_bar)
        else:
            exp_expr_gain = max(0,self.expr_eval(new_theta)[0] - self.expr_eval(theta_s)[0])
        return exp_expr_gain/(1-gamma)
    
same_sign = lambda x,y : True if (x-0.5)*(y-0.5) > 0 else False       
beta_params=[(2,4),(4,2),(3.2,5)]
corr_mat = np.asarray([[1, 0.75,0.56], 
                        [0.75, 1,0.3],
                        [0.56, 0.3,1]])
data_samples = plot_3_way_corr(beta_params,corr_mat,show_plot=False).T
comm_agents = [CommunityAgent(data_samples[i,:]) for i in np.arange(data_samples.shape[0])]
theta_s = 0.7
X = []
for c in comm_agents:
    c.expr_eval(theta_s)
expressing_agents = [c for c in comm_agents if c.expr]
expr_len = len(expressing_agents)
t1,t2 = expressing_agents[:int(expr_len/2)],expressing_agents[int(expr_len/2):-1]
conflicts = []
for c1,c2 in zip(t1,t2):
    if not same_sign(c1.mean_op,c2.mean_op):
        conflicts.append((c1,c2))
punishment_pool = dict()
for confls in conflicts:
    punishment_pool[confls] = []
    X = []
    for c in comm_agents:
        gain_from_intervention = c.intervening_gain(theta_s,confls)-c.intervening_cost(theta_s, confls)
        if gain_from_intervention > 0:
            punishment_pool[confls].append(c)
            X.append((c.mean_op,gain_from_intervention))
    X.sort()
    plt.title(','.join([str(confls[0].mean_op),str(confls[1].mean_op)]))
    plt.plot([x[0] for x in X],[x[1] for x in X])
    plt.show()
plt.hist([np.mean([c.mean_op for c in v]) for k,v in punishment_pool.items()],bins=20)
plt.show()
    