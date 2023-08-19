'''
Created on Dec. 27, 2022


'''
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import gamma, beta
import utils


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
    
def plot_3_way_corr(beta_params,corr_mat,show_plot=False):
    beta_params=[(2,4),(4,2),(3.2,5)] if beta_params is None else beta_params
    beta_var = [stats.beta(a=beta_params[i][0],b=beta_params[i][1]).var() for i in np.arange(len(beta_params))]
    corr_mat = np.asarray([[1, -0.6,0.5], 
                            [-0.6, 1,0.3],
                            [0.5, 0.3,1]]) if corr_mat is None else corr_mat
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
    if show_plot:
        h = sns.jointplot(x=x1_trans, y=x2_trans, kind='kde', xlim=(0, 1), ylim=(0, 1.0), fill=True);
        h.fig.suptitle("n1n2")
        h = sns.jointplot(x=x1_trans, y=x3_trans, kind='kde', xlim=(0, 1), ylim=(0, 1.0), fill=True);
        h.fig.suptitle("n1n3")
        h = sns.jointplot(x=x2_trans, y=x3_trans, kind='kde', xlim=(0, 1), ylim=(0, 1.0), fill=True);
        h.fig.suptitle("n2n3")
        plt.show()
    return np.asarray([x1_trans, x2_trans, x3_trans])

def plot_2_way_corr():
    mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., -0.6], 
                                                         [-0.6, 1.]])
    # Generate random samples from multivariate normal with correlation .5
    x = mvnorm.rvs(1000)
    #h = sns.jointplot(x[:, 0], x[:, 1], kind='kde',fill=True);
    #h.set_axis_labels('X1', 'X2', fontsize=16);
    norm = stats.norm()
    x_unif = norm.cdf(x)
    #h = sns.jointplot(x_unif[:, 0], x_unif[:, 1], kind='hex')
    #h.set_axis_labels('Y1', 'Y2', fontsize=16);
    m1 = stats.beta(a=2, b=7)
    m2 = stats.beta(a=7, b=6)
    
    x1_trans = m1.ppf(x_unif[:, 0])
    x2_trans = m2.ppf(x_unif[:, 1])
    
    h = sns.jointplot(x=x1_trans, y=x2_trans, kind='kde', xlim=(0, 1), ylim=(0, 1.0), fill=True, color='red');
    plt.show()

def beta_pdf(x,a,b):
    return ((x**(a-1))*((1-x)**(b-1)))/beta(a,b)

def calc_bivariate_pdf(varx,vary,alpha):
    if vary is None:
        evals = []
        norm_constant_b_of_alpha = np.prod([gamma(a) for a in alpha])/gamma(np.sum(alpha))
        #samples = np.random.dirichlet(alpha, 1000)[:,1]
        #sns.displot(data=samples, kind="kde")
        #plt.show()
        
        for vary in np.linspace(0,1,100):
            N = 100
            accum = 0 
            '''
            all_xs = [x for x in np.random.dirichlet(alpha, 10000)[:,3] if max(0,varx+vary-1) < x < min(varx,vary)]
            if len(all_xs)==0:
                continue
            '''
            for i in range(N):
                x = np.random.uniform(max(0,varx+vary-1), min(varx,vary))
                if x!=0:
                    #x = np.random.choice(all_xs)
                    ''' marginal of dirichlet is beta: https://stats.stackexchange.com/questions/319292/find-marginal-distribution-of-k-variate-dirichlet'''
                    #accum += ((math.pow(x, alpha[3]-1) * math.pow(varx - x,alpha[2]-1)* math.pow(vary - x,alpha[1]-1) * math.pow(1 - varx - vary + x, alpha[0]-1)) * (beta_pdf(x, alpha[3], np.sum(alpha[:-1]))) )
                    accum += (math.pow(x, alpha[3]-1) * math.pow(varx - x,alpha[2]-1)* math.pow(vary - x,alpha[1]-1) * math.pow(1 - varx - vary + x, alpha[0]-1))
                else:
                    accum += 0
            measure =  min(varx,vary) - max(0,varx+vary-1)
            measure = (measure/float(N)) * accum
            evals.append((vary,measure/norm_constant_b_of_alpha))
        norm_evals = [(x,y/np.sum([x[1] for x  in evals])) for x,y in evals]
        print(np.sum([x*y for x,y in norm_evals]))
        plt.figure()
        plt.plot([x[0] for x in evals], [x[1] for x in evals])
        #utils.plot_beta(alpha[0], np.sum(alpha[1:]))
        plt.show()
        
def bivariate_beta_from_dirichlet(alpha):
    ''' https://www.sciencedirect.com/science/article/pii/S0167715214003241 '''
    ''' generate Dirichlet samples '''
    alpha_oo,alpha_o1,alpha_10,alpha_11 = alpha
    #alpha_oo,alpha_o1,alpha_10,alpha_11 = 4,4,4,0.5
    samples = np.random.dirichlet((alpha_oo,alpha_o1,alpha_10,alpha_11), 1000)
    new_variate_x = samples[:,2] + samples[:,3]
    new_variate_y = samples[:,1] + samples[:,3]
    h = sns.jointplot(x=new_variate_x, y=new_variate_y, kind='kde', xlim=(0, 1), ylim=(0, 1), fill=True)
    #h.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
    #h.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
    ''' correlation calculation '''
    alpha_one_plus,alpha_plus_one,alpha_zero_plus,alpha_plus_zero = alpha_11+alpha_10, alpha_11+alpha_o1, alpha_oo+alpha_o1, alpha_oo+alpha_10
    corr = ((alpha_11*alpha_oo) - (alpha_10*alpha_o1))/(math.sqrt(alpha_one_plus*alpha_plus_one*alpha_zero_plus*alpha_plus_zero))
    print('correlation',corr)
    means = np.mean(np.asarray([new_variate_x, new_variate_y]), axis=1)
    print('means',means)
    plt.show()

#alpha = (3,5,6,3)
#bivariate_beta_from_dirichlet(alpha)    
#calc_bivariate_pdf(None,0.3,alpha)
#plot_2_way_corr()

#n_way_samples_copula(beta_params=[(2,4),(4,2),(3.2,5)],corr_matrix=np.asarray([[1, -0.6,0.5], 
#                            [-0.6, 1,0.3],
#                            [0.5, 0.3,1]]))