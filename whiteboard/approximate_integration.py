'''
Created on 25 Jan 2023

@author: Atrisha
'''
import utils
import numpy as np
from scipy.stats import beta, norm, dirichlet
import math
import time
import operator
import functools
import scipy.special

def beta_pdf(x,a,b):
    return ((x**(a-1))*((1-x)**(b-1)))/scipy.special.beta(a,b)

def dirichlet_pdf(x, alpha):
    return (math.gamma(sum(alpha)) / 
          functools.reduce(operator.mul, [math.gamma(a) for a in alpha]) *
          functools.reduce(operator.mul, [x[i]**(alpha[i]-1.0) for i in range(len(alpha))]))
  
def runif_in_simplex(n_samples,n_dim):
    ''' Return uniformly random vector in the n-simplex '''

    k = np.random.exponential(scale=1.0, size=(n_samples,n_dim))
    return k / np.sum(k,axis=1)[:,None]

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

start_time = time.time()
'''
weight_samples = runif_in_simplex(100,4)
theta_samples = np.random.uniform(size=(1000,4))
indices = np.ndindex(weight_samples.shape[0],theta_samples.shape[0])
all_samples = np.zeros(shape=(weight_samples.shape[0]*theta_samples.shape[0],4*2))
i = 0
for index in indices:
    f = np.hstack((weight_samples[index[0],:],theta_samples[index[1],:]))
    all_samples[i] = f
    i +=1
'''
#g = np.meshgrid(weight_samples, theta_samples)
#positions = np.hstack(map(np.ravel, g))
weight_samples = runif_in_simplex(10000,4)
theta_samples = np.random.uniform(size=(10000,4))
all_samples = np.hstack((weight_samples,theta_samples))
constr_f = lambda x,signal_distr : utils.Gaussian_plateu_distribution(0,0.1,0.3).pdf(abs(x-signal_distr))
print("--- generating samples done %s seconds ---" % (time.time() - start_time))
#calc_pos = lambda _state_vec : (dirichlet.pdf(x=_state_vec[:4], alpha = [1.5,1.5,1.5,1.5])*math.prod([beta.pdf(_state_vec[nidx], 2,3) for nidx,n in enumerate([1,2,3,4])]))*constr_f(_state_vec[:4]@_state_vec[4:].T,0.6)
def calc_pos(_state_vec):
    #x1 =  (dirichlet.pdf(x=_state_vec[:4], alpha = [1.5,1.5,1.5,1.5])*math.prod([beta.pdf(_state_vec[nidx], 2,3) for nidx,n in enumerate([1,2,3,4])]))*constr_f(_state_vec[:4]@_state_vec[4:].T,0.6)
    x2 = (dirichlet_pdf(x=_state_vec[:4], alpha = [1.5,1.5,1.5,1.5])*math.prod([beta_pdf(_state_vec[nidx], 2,3) for nidx,n in enumerate([1,2,3,4])]))*constr_f(_state_vec[:4]@_state_vec[4:].T,0.6)
    #print('diff',x1-x2)
    return x2
'''
posteriors = np.zeros(shape=(all_samples.shape[0],))
for x_idx in np.arange(all_samples.shape[0]):
    x = all_samples[x_idx]
    p1 = dirichlet.pdf(x=x[:4], alpha = [1.5,1.5,1.5,1.5])*math.prod([beta.pdf(x[nidx], 2,3) for nidx,n in enumerate([1,2,3,4])])
    _state = x[:4]@x[4:].T
    p2 = constr_f(_state,0.6)
    posteriors[x_idx] = p1*p2
'''
posteriors = np.apply_along_axis(calc_pos, 1, all_samples)
print("--- posterior calculation done %s seconds ---" % (time.time() - start_time))
#post_vect = np.vectorize(calc_pos,signature='(n,m)->(n)')
#posteriors = post_vect(all_samples)
print('done',np.sum(posteriors)/all_samples.shape[0])
print("--- %s seconds ---" % (time.time() - start_time))


  

