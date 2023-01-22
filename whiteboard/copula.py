'''
Created on Dec. 27, 2022


'''
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

def corr2cov(corr_mat,variance_list):
    cov_matrix = np.ones(shape=corr_mat.shape)
    var_matrix = np.zeros(shape=corr_mat.shape)
    np.fill_diagonal(var_matrix, variance_list)
    cov_matrix = var_matrix @ corr_mat @ var_matrix
    return cov_matrix

def n_way_samples_copula(beta_params, corr_matrix):
    beta_var = [stats.beta(a=beta_params[i][0],b=beta_params[i][1]).var() for i in np.arange(len(beta_params))]
    N = corr_matrix.shape[0]
    ''' we need to scale the covariance since the original covariance is with respect to beta distribution'''
    cov_matrix  = corr2cov(corr_matrix,beta_var)*(10**3)
    mvnorm = stats.multivariate_normal(mean=[0]*N, cov=cov_matrix)
    x = mvnorm.rvs(100)
    norm = stats.norm()
    x_unif = norm.cdf(x)
    m_list = [stats.beta(a=x[0],b=x[1]) for x in beta_params]
    x_trans = [m_list[i].ppf(x_unif[:,i]) for i in np.arange(len(m_list))]
    x_trans = np.asarray(x_trans).T
    marginals = np.mean(x_trans,axis=0)
    return x_trans
    
def plot_3_way_corr():
    beta_params=[(2,4),(4,2),(3.2,5)]
    beta_var = [stats.beta(a=beta_params[i][0],b=beta_params[i][1]).var() for i in np.arange(len(beta_params))]
    corr_mat = np.asarray([[1, -0.6,0.5], 
                            [-0.6, 1,0.3],
                            [0.5, 0.3,1]])
    cov_mat = corr2cov(corr_mat,beta_var)*(10**3)
    ''' we need to scale the covariance since the original covariance is with respect to beta distribution'''
    mvnorm = stats.multivariate_normal(mean=[0, 0, 0], cov=cov_mat)
    # Generate random samples from multivariate normal with correlation .5
    x = mvnorm.rvs(1000)
    #h = sns.jointplot(x[:, 0], x[:, 1], kind='kde',fill=True);
    #h.set_axis_labels('X1', 'X2', fontsize=16);
    norm = stats.norm()
    x_unif = norm.cdf(x)
    #h = sns.jointplot(x_unif[:, 0], x_unif[:, 1], kind='hex')
    #h.set_axis_labels('Y1', 'Y2', fontsize=16);
    m1 = stats.beta(a=2, b=4)
    m2 = stats.beta(a=4, b=2)
    m3 = stats.beta(a=3.2, b=5)
    x1_trans = m1.ppf(x_unif[:, 0])
    x2_trans = m2.ppf(x_unif[:, 1])
    x3_trans = m3.ppf(x_unif[:, 2])
    
    h = sns.jointplot(x1_trans, x2_trans, kind='kde', xlim=(0, 1), ylim=(0, 1.0), fill=True);
    h = sns.jointplot(x1_trans, x3_trans, kind='kde', xlim=(0, 1), ylim=(0, 1.0), fill=True);
    h = sns.jointplot(x2_trans, x3_trans, kind='kde', xlim=(0, 1), ylim=(0, 1.0), fill=True);
    plt.show()

def plot_2_way_corr():
    mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.6], 
                                                         [0.6, 1.]])
    # Generate random samples from multivariate normal with correlation .5
    x = mvnorm.rvs(1000)
    #h = sns.jointplot(x[:, 0], x[:, 1], kind='kde',fill=True);
    #h.set_axis_labels('X1', 'X2', fontsize=16);
    norm = stats.norm()
    x_unif = norm.cdf(x)
    #h = sns.jointplot(x_unif[:, 0], x_unif[:, 1], kind='hex')
    #h.set_axis_labels('Y1', 'Y2', fontsize=16);
    m1 = stats.beta(a=1, b=3)
    m2 = stats.beta(a=2, b=3)
    
    x1_trans = m1.ppf(x_unif[:, 0])
    x2_trans = m2.ppf(x_unif[:, 1])
    
    h = sns.jointplot(x1_trans, x2_trans, kind='kde', xlim=(0, 1), ylim=(0, 1.0), fill=True);
    plt.show()
    
#plot_3_way_corr()
n_way_samples_copula(beta_params=[(2,4),(4,2),(3.2,5)],corr_matrix=np.asarray([[1, -0.6,0.5], 
                            [-0.6, 1,0.3],
                            [0.5, 0.3,1]]))