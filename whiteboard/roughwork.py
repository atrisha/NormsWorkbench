'''
Created on 15 Jan 2023

@author: Atrisha
'''
from sympy.stats import P, E, variance, Beta, Normal
from sympy import simplify
import numpy as np

import math

def func1(x):
    # function f(x)=x^2
    return x**2

def func1_int(a, b):
    # analytical solution to integral of f(x)
    return (1/3)*(b**3-a**3)
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
def mc_integrate(func, a, b, n = 1000):
    # Monte Carlo integration between x1 and x2 of given function from a to b
    
    vals = np.random.uniform(a, b, n)
    y = [func(val) for val in vals]
    
    #y_mean = np.sum(y)/n
    #integ = (b-a) * y_mean
    integ = np.sum(y)
    return integ

print(math.pow(0.8, -1))
