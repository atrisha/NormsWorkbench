'''
Created on 25 Sept 2022

@author: Atrisha
'''

import numpy as np
import matplotlib.pyplot as plt

from z3 import *



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
    
z3_test()
    
    
            
     