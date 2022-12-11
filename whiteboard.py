'''
Created on 25 Sept 2022

@author: Atrisha
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
#from z3 import *
from scipy.stats import entropy
from scipy.interpolate import RectBivariateSpline
from scipy.special import expit
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr


def show_convergence():
    R,c = 1,-0.5
    mu, sigma = 0, 0.1
    N = 1000
    players = {i:[] for i in np.arange(N)}
    
    for run_id in np.arange(100):
        samples = np.random.normal(mu, sigma, N)
        for i in np.arange(0,len(samples),3):
            if i + 2 < len(samples):
                ids = [(i,samples[i]),(i+1,samples[i+1]),(i+2,samples[i+2])]
                ids.sort(key=lambda tup: tup[1]) 
                if abs(ids[1][1]-ids[0][1]) < abs(ids[1][1]-ids[2][1]):
                    players[ids[0][0]].append((ids[0][1],R))
                    players[ids[1][0]].append((ids[1][1],0))
                    players[ids[2][0]].append((ids[2][1],-R))
                    
                else:
                    players[ids[0][0]].append((ids[0][1],-R))
                    players[ids[1][0]].append((ids[0][1],0))
                    players[ids[2][0]].append((ids[0][1],R))
        
    count, bins, ignored = plt.hist(samples, 30, density=True)
    plt.show()
    opinion_distr = []
    for k,v in players.items():
        if sum([x[1] for x in v]) > 0:
            opinion_distr = opinion_distr + v
    all_costs = [sum([x[1] for x in v]) for k,v in players.items()]
    plt.hist(all_costs)
    plt.show()
    
def correleted_eq_staghunt_conflict():
    utilatarian_welfare = [0,0,0,1]
    utilatarian_welfare = [x/sum(utilatarian_welfare) for x in utilatarian_welfare]
    aristrocatic_welfare_g1 = [1,0,0,0]
    aristrocatic_welfare_g1 = [x/sum(aristrocatic_welfare_g1) for x in aristrocatic_welfare_g1]
    outcomes = np.random.choice(a=['p1','p2','p3','p4'],size=1000,p=utilatarian_welfare)
    payoffs = {'p1':(10,15),'p2':(4,14),'p3':(6,10),'p4':(5,20)}
    g1, g2 = [],[]
    for o in outcomes:
        g1.append(payoffs[o][0])
        g2.append(payoffs[o][1])
    g1 = [sum(g1[:i])for i in np.arange(len(g1))]
    g2 = [sum(g2[:i])for i in np.arange(len(g2))]
    plt.plot(np.arange(len(outcomes)),g1,'-',color='r',label='row(utilitarian)')
    plt.plot(np.arange(len(outcomes)),g2,'-',color='b',label='column(utilitarian)')
    
    outcomes = np.random.choice(a=['p1','p2','p3','p4'],size=1000,p=aristrocatic_welfare_g1)
    #payoffs = {'p1':(10,10),'p2':(1,8),'p3':(8,1),'p4':(5,5)}
    g1, g2 = [],[]
    for o in outcomes:
        g1.append(payoffs[o][0])
        g2.append(payoffs[o][1])
    g1 = [sum(g1[:i])for i in np.arange(len(g1))]
    g2 = [sum(g2[:i])for i in np.arange(len(g2))]
    plt.plot(np.arange(len(outcomes)),g1,'--',color='r',label='row(maxmin)')
    plt.plot(np.arange(len(outcomes)),g2,'--',color='b',label='column(maxmin)')
    plt.legend(loc="upper left")
    plt.xlabel("time steps")
    plt.ylabel("payoffs")
    
    plt.show()

def bayesian_gaussian_gaussian_update():
    mu_0, sd_0 = 0, 1
    mu_true, sd_true = 7, 2
    mu_prev, sd_prev = 0, 10
    mu_curr, sd_curr = 0, 10
    sigma_ratio = lambda num, s, s_prior : num**2/(s**2 + s_prior**2) 
    mus = []
    for i in np.arange(100):
        obs = np.random.normal(mu_true, sd_true)
        mu_curr = sigma_ratio(sd_true,sd_true,sd_curr)*mu_curr + sigma_ratio(sd_curr,sd_true,sd_curr)*obs
        sd_curr =  sigma_ratio(sd_true,sd_true,sd_curr)*(sd_curr**2)
        print('mu_curr',mu_curr, sd_curr)
        mus.append(mu_curr)
    bins = np.arange(0,30,0.1)
    plt.plot(bins, 1/(sd_curr * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu_curr)**2 / (2 * sd_curr**2) ), linewidth=2, color='r')
    plt.figure()
    plt.plot(np.arange(len(mus)),mus)
    print("all mu means",np.mean(mus))
    plt.show()
    

def z3_test():
    x = Real('x')
    y = Real('y')
    s = Solver()
    s.add(x + y > 5, x > 1, y > 1)
    print(s.check())
    print(s.model())
    
#z3_test()

def plot_payoff_based_selection_prob():
    x = np.linspace(0,1,100)
    fy =  lambda x : 2*x if x <=0.5 else 2-2*x 
    plt.plot(x,[fy(_x) for _x in x])
    plt.xlabel('op(A)')
    plt.ylabel('probability of payoff based selection')
    plt.show()

def plot_exp():
    x = np.linspace(0.5,1,100)
    #fy =  lambda x : np.exp(1)*np.exp(-x)/(np.exp(1)-1)
    fy =  lambda x : 1-x
    plt.plot(x,[fy(_x) for _x in x])
    plt.show()
    
def plot_line():
    a,b = 2,2
    x = np.linspace(beta.ppf(0.01, a, b),beta.ppf(0.99, a, b), 100)
    plt.figure(figsize=(7,7))
    plt.xlim(0, 1)
    y = beta.pdf(x, a, b)
    plt.plot(x, [_e/max(y) for _e in y], linestyle='-')
    plt.xlabel('op(A)', fontsize='15')
    plt.ylabel('Cost of context mismatch', fontsize='15')
    plt.show()

#plot_line()            
 
def plot_PN_matrix():
    fig, ax = plt.subplots(1,4)

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
    ax[0].matshow(PN_matrix, cmap='bwr_r')
    ax[2].matshow(DB_matrix, cmap='bwr_r')
    ax[1].matshow(np.sum(PN_matrix,axis=1).reshape(PN_matrix.shape[0],1), cmap='bwr_r')
    ax[3].matshow(np.sum(DB_matrix,axis=1).reshape(DB_matrix.shape[0],1), cmap='bwr_r')
    
    plt.show()

'''
#N_dict = {'n1':0.3,'n2':0.2,'n3':0.1,'n4':0.4}
N_dict = [0.3,0.2,0.1,0.4]
prob_app = [1,1,1,1]
prob_dis = [1-x for x in prob_app]
binom_param =np.sum([N_dict[i] for i in np.arange(len(N_dict)) if prob_app[i] == 1])
#prob_app = [0.1,0.1,0.1,0.1]

cost_entropy = entropy([binom_param, 1-binom_param], base=2)

print(cost_entropy)
'''
def plot_multivariate_gaussian(mean,corr):
    cov_val =  corr * np.sqrt(0.07) * np.sqrt(0.01)
    cov = np.array([[.07, cov_val], [cov_val, .01]])
    pts = np.random.multivariate_normal([0, 0], cov, size=800)
    pts = expit(pts)
    plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.5)
    plt.grid()
    plt.show()



 
    