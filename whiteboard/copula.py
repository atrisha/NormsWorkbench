'''
Created on Dec. 27, 2022


'''
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
def plot_3_way_corr():
    mvnorm = stats.multivariate_normal(mean=[0, 0, 0], cov=[[1., -0.6,0.5], 
                                                         [-0.6, 1.,0.3],
                                                         [0.5, 0.3,1]])
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
    m3 = stats.beta(a=2, b=2)
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
    m2 = stats.beta(a=2, b=4)
    
    x1_trans = m1.ppf(x_unif[:, 0])
    x2_trans = m2.ppf(x_unif[:, 1])
    
    h = sns.jointplot(x1_trans, x2_trans, kind='kde', xlim=(0, 1), ylim=(0, 1.0), fill=True);
    plt.show()
    
plot_2_way_corr()