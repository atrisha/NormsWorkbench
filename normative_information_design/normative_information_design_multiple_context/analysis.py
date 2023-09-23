'''
Created on 17 Apr 2023

@author: Atrisha
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import seaborn as sns
import pandas as pd
import sympy
import math
import scipy.stats as stats
import scipy.integrate as integrate

def weight_graph():
    bn, bn_bar = 0.6, 0.4
    w_n = lambda d1,d2 : (bn*(d1 - (0.5*d2) -0.5)) / ( (0.5*bn) - (bn_bar*(1-d1-0.5+(0.5*d2))) )
    ax = plt.axes(projection='3d')
    

    
    
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    
    #constants.cen_belief = 2.5,2
    #constants.cen_true_distr = 4,2
    
    
    xs, ys = np.meshgrid(x, y)
    Z=[]
    for i in range(len(xs)):
        for j in range(len(xs[0])):
            d1, d2 = xs[i][j], ys[i][j]
            #prob_of_N = (bel_op*(util(op)-cost(op)))/theta
            
            Z.append(w_n(d1,d2))
            
            
    
    # reshape Z
    
    Z = np.array(Z).reshape(xs.shape)
    
    # interpolate your values on the grid defined above
    f_interp = RectBivariateSpline(x,y, Z)
    
    
    ax.plot_surface(xs,ys,Z,
            cmap='viridis', edgecolor='none', rcount=200, ccount=200)
        
    ax.set_xlabel('delta 1')
    ax.set_zlabel('w')
    ax.set_ylabel('delta 2')
    
    plt.gca().invert_xaxis()
    #plt.gca().invert_yaxis()
    plt.show()

def wn_solve(b1, c1, e1, f1):
    #u,b,c,w,e,f,v = sympy.symbols('u b c w e f v')
    #expr = sympy.Eq( (u*b*w) + ((1-w)*c*(1-e)*u) - (v*b*w) - ((1-w)*c*(1-f)*v) , 0)
    #r = sympy.solveset(expr, w)
    #print(r)
    #subs_list = [(u,1),(b,b1),(c,c1),(e,e1),(f,f1),(v,0.5)]
    ex = c1*(e1*1 - f1*0.5 - 1 + 0.5)/(b1*1 - b1*0.5 + c1*e1*1 - c1*f1*0.5 - c1*1 + c1*0.5)
    return ex

def weight_graph_d1_d2_ratio():
    runidx = 0
    axs = None
    dfs = []
    fig, axs = plt.subplots(1, 1)
    for bn, bn_bar in [(0.8, 0.7),(0.8, 0.55)]:
        data_list,d2_list = [], [] 
        w_n = lambda d1,d2,bn,bn_bar : wn_solve(bn,bn_bar,d1,d2)
        x = np.linspace(0,1,100)
        y = np.linspace(0,1,100)
        xs, ys = np.meshgrid(x, y)
        for i in range(len(xs)):
            for j in range(len(xs[0])):
                d1, d2 = xs[i][j], ys[i][j]
                d2_d1_ratio = d2/d1
                wn_val = w_n(d1,d2,bn,bn_bar)
                if not isinstance(wn_val, float):
                    continue
                if 0 <= wn_val <= 1:
                    data_list.append([runidx,d2,d2_d1_ratio,wn_val])
                    exp_util_low = (0.5*bn*wn_val) + (0.5*bn_bar*(1-wn_val)*(1-d2))
                    exp_util_high = (bn*wn_val) + (bn_bar*(1-wn_val)*(1-d1))
                    
        runidx += 1            
        df = pd.DataFrame(data_list, columns=['runidx','d2','d2-d1-ratio','weight'])
        dfs.append(df)
    for i,df in enumerate(dfs):
        if i== 0:
            axs = sns.scatterplot( data=df, x="d2-d1-ratio", y="weight", hue='d2', palette=sns.color_palette("Blues", as_cmap=True))
        else:
            sns.scatterplot( data=df, x="d2-d1-ratio", y="weight", hue='d2', palette=sns.color_palette("Reds", as_cmap=True))
    axs.legend([],[], frameon=False)
    #data_list.sort(key=lambda tup: tup[0])
    #plt.scatter([x[0] for x in data_list],[x[1] for x in data_list],c=d2_list)
    #plt.gray()
    #plt.legend()
    #plt.ylim(0,1)
    
    plt.show()
def exp_util_diff_max_bounds():
    exp_util = lambda u,bn,bn_bar,wn_val,d : (u*bn*wn_val) + (u*bn_bar*(1-wn_val)*(1-d))
    d1, d2 = 0.6, 0.12
    
    runidx = 0
    fig, axs = plt.subplots(1, 2)
    for bn, bn_bar in [(0.8, 0.7),(0.8, 0.55)]:
        data_list = []
        for wn_val in np.linspace(0,1,100):
            ex_high = exp_util(1,bn, bn_bar, wn_val, d1)
            ex_low = exp_util(0.5,bn, bn_bar, wn_val, d2)
            data_list.append([wn_val,ex_high,'1'])
            data_list.append([wn_val,ex_low,'0.5'])
        df = pd.DataFrame(data_list, columns=['w','u','op'])
        sns.lineplot(data=df, x="w", y="u",hue='op',ax = axs[runidx])
        runidx += 1 
    axs[0].set_ylim(0,1)
    axs[1].set_ylim(0,1)
    plt.show()

def bn_bar_relation():
    d1, d2 = 0.6, 0.12
    col = ['black','gray']
    for idx,bn in enumerate([0.8, 0.6]):
        x,y = [],[]
        for bn_bar in np.linspace(0,1,100):
            w = wn_solve(bn,bn_bar,d1,d2)
            x.append(bn_bar)
            y.append(w)
        plt.plot(x,y,color= col[idx])
    
    plt.show()
    
def generate_2way_copula(mean = [0, 0],cov=[[1., -0.6], 
                                                [-0.6, 1.]]):
    mvnorm = stats.multivariate_normal(mean, cov)
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
    
    new_variate_x = m1.ppf(x_unif[:, 0])
    new_variate_y = m2.ppf(x_unif[:, 1])
    
    #h = sns.jointplot(x=x1_trans, y=x2_trans, kind='kde', xlim=(0, 1), ylim=(0, 1.0), fill=True);
    #plt.show()
    return np.asarray([new_variate_x, new_variate_y])

def bivariate_beta_from_dirichlet(param_tuple):
    ''' https://www.sciencedirect.com/science/article/pii/S0167715214003241 '''
    ''' generate Dirichlet samples '''
    #alpha_oo,alpha_o1,alpha_10,alpha_11 = 1.2,2.5,3,2
    alpha_oo,alpha_o1,alpha_10,alpha_11 = param_tuple
    samples = np.random.dirichlet((alpha_oo,alpha_o1,alpha_10,alpha_11), 1000)
    new_variate_x = samples[:,2] + samples[:,3]
    new_variate_y = samples[:,1] + samples[:,3]
    #h = sns.jointplot(x=new_variate_x, y=new_variate_y, kind='kde', xlim=(0, 1), ylim=(0, 1), fill=True)
    #h.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
    #h.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
    ''' correlation calculation '''
    alpha_one_plus,alpha_plus_one,alpha_zero_plus,alpha_plus_zero = alpha_11+alpha_10, alpha_11+alpha_o1, alpha_oo+alpha_o1, alpha_oo+alpha_10
    corr = ((alpha_11*alpha_oo) - (alpha_10*alpha_o1))/(math.sqrt(alpha_one_plus*alpha_plus_one*alpha_zero_plus*alpha_plus_zero))
    #print('correlation',corr)
    #plt.show()
    return np.asarray([new_variate_x, new_variate_y])

def calc_inequality_index(data_samples,w):
    
    means = np.mean(data_samples, axis=0)
    
    def pc_assignment(x):
        if abs(x[0]-0.5) > abs(x[1]-0.5):
            return np.asarray([1,0])
        elif abs(x[0]-0.5) < abs(x[1]-0.5):
            return np.asarray([0,1])
        else:
            return np.asarray([1,0]) if np.random.random_sample() < 0.5 else np.asarray([0,1])
    prerred_context_index = np.apply_along_axis(pc_assignment,1,data_samples)
    def bn_bnbar_assignment(x):
        op,pc,means = x[:2],x[2:4],x[4:6]
        pc_index = np.where(pc == 1)
        non_pc_index = np.where(pc == 0)
        bn = means[pc_index] if op[pc_index] >= 0.5 else 1-means[pc_index]
        bn_bar = means[non_pc_index] if op[non_pc_index] >= 0.5 else 1-means[non_pc_index]
        return np.array([bn[0],bn_bar[0]])
    inp_args = np.concatenate((data_samples,prerred_context_index,np.tile(means,(1000,1))),axis=1)
    bn_bn_bar = np.apply_along_axis(bn_bnbar_assignment,1,inp_args)
    def exp_util_assignment(x,w):
        op,pc,means,bn_bn_bar = x[:2],x[2:4],x[4:6],x[6:8]
        pc_index = np.where(pc == 1)
        non_pc_index = np.where(pc == 0)
        cw = w if pc_index==0 else (1-w)
        op_util = abs(op-0.5)+0.5
        op_util = np.asarray([op_util[pc_index],op_util[non_pc_index]])
        wD,bD = np.diag([cw,1-cw]), np.diag(bn_bn_bar)
        exp_util = wD @ bD @ op_util
        return np.sum(exp_util)
    inp_args = np.concatenate((data_samples,prerred_context_index,np.tile(means,(1000,1)),bn_bn_bar),axis=1)
    exp_util_samples = np.apply_along_axis(exp_util_assignment,1,inp_args,w)
    mean_util = np.mean(exp_util_samples)
    inequal_idx_MLD = np.sum(np.log((exp_util_samples**-1)*mean_util))/data_samples.shape[0 ]
    if w==0:
        print('pc props',np.sum(prerred_context_index/1000,axis=0))
    #print('ineqality index MLD',w,inequal_idx_MLD,np.min(exp_util_samples))
    
    return inequal_idx_MLD

def theta_theta_prime_relation():
    thetas = np.linspace(0.01,.99,100)
    c=1
    u_bar = 0.2
    f_x = lambda x,theta,c: 2*theta*c if x>0.5 else 2*(1-theta)*c
    def lower_int(l,theta,c):
        if l <= 0 :
            result = 0.0
        elif l >0.5:
            result = integrate.quad(lambda x: x*f_x(x,theta,c), 0, 0.5)
        else:
            result = integrate.quad(lambda x: x*f_x(x,theta,c), 0, l)
        return result if isinstance(result,float) else result[0]
    def upper_int(h,theta,c):
        if h > 1 :
            result = 0.0
        elif h <0.5:
            result = integrate.quad(lambda x: x*f_x(x,theta,c), 0.5, 1)
        else:
            result = integrate.quad(lambda x: x*f_x(x,theta,c), h, 1)
        return result if isinstance(result,float) else result[0]
    X,Y = [],[]
    for b in thetas:
        if b<=0.2:
            f=1
        l,h = 1-(u_bar/(1-b)), u_bar/b
        curr_x0, curr_x1 = 2*(1-b)*c, 2*b*c
        #c = 1/((curr_x1*max(0.5,1-h))+(curr_x0*max(0,l)))
        exp_res = lower_int(l,b,c) + upper_int(h, b, c)
        if isinstance(exp_res, np.ndarray):
            exp_res = exp_res[0]
        X.append(b)
        Y.append(exp_res)
        
    plt.plot(X,Y,'.')
    plt.plot(X,X,'--')
    plt.show()
        
        
        
'''
bn, bn_bar = 0.8, 0.8
w_n = lambda d1,d2 : (bn*(d1 - (0.5*d2) -0.5)) / ( (0.5*bn) - (bn_bar*(1-d1-0.5+(0.5*d2))) ) 
print(w_n(0.7,0.21))  
'''
#weight_graph_d1_d2_ratio()
#exp_util_diff_max_bounds()
#wn_solve(1,1,1,1)
#bn_bar_relation()
#data_samples = bivariate_beta_from_dirichlet(param_tuple=(4,4,4,0.5)).T
'''
data_samples = generate_2way_copula().T
print('red means',np.mean(data_samples, axis=0))
Y=[]
for w in np.linspace(0,1,100):
    Y.append(calc_inequality_index(data_samples,w))
plt.plot(np.linspace(0,1,100),Y,color='red')

data_samples = bivariate_beta_from_dirichlet(param_tuple=(5,5,5,5)).T
print('blue means',np.mean(data_samples, axis=0))
Y=[]
for w in np.linspace(0,1,100):
    Y.append(calc_inequality_index(data_samples,w))
plt.plot(np.linspace(0,1,100),Y,color='blue')
plt.show()
'''
#theta_theta_prime_relation()