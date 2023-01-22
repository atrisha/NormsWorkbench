'''
Created on 15 Jan 2023

@author: Atrisha
'''
from sympy.stats import P, E, variance, Beta, Normal
from sympy import simplify
import numpy as np

x= 0.9
prior_distr = Beta("X", 3,2)
prec = 0.001
prior_x = 1 - P(prior_distr<(x-prec)) - P(prior_distr>(x+prec))
prior_x = prior_x.evalf()
print(prior_x)