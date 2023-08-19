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
from collections import Counter
import utils
import seaborn as sns
#import rpy2.robjects as robjects
#import rpy2.robjects.numpy2ri
#from rpy2.robjects.packages import importr



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
    
def action_selection_gird(op_strat=False):
    theta = 0.3
    cost = lambda x : entropy([x,1-x])
    util = lambda op : op if op > 0.5 else (1-op)
    bel_op = np.linspace(0,1,100)
    ops = np.linspace(0,1,100)
    ax = plt.axes(projection='3d')
    

    def f(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))
    
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    
    #constants.cen_belief = 2.5,2
    #constants.cen_true_distr = 4,2
    
    
    xs, ys = np.meshgrid(x, y)
    Z,Z2=[],[]
    for i in range(len(xs)):
        for j in range(len(xs[0])):
            op, bel_op = xs[i][j], ys[i][j]
            #prob_of_N = (bel_op*(util(op)-cost(op)))/theta
            prob_of_N = (bel_op*util(op))/theta
            Z.append(prob_of_N)
            Z2.append(1)
            if op_strat:
                if prob_of_N > 1:
                    Z2.append(1)
                else:
                    Z2.append(0)
            
    
    # reshape Z
    
    Z = np.array(Z).reshape(xs.shape)
    Z2 = np.array(Z2).reshape(xs.shape)
    
    # interpolate your values on the grid defined above
    f_interp = RectBivariateSpline(x,y, Z)
    X_grid, Y_grid = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    Z_grid = f_interp(np.linspace(0,1,100), np.linspace(0,1,100))
    Z2_grid = f_interp(np.linspace(0,1,10), np.linspace(0,1,10))
    if not op_strat:
        ax.plot_surface(xs,ys,Z,
                cmap='viridis', edgecolor='none', rcount=200, ccount=200)
        ax.plot_surface(xs,ys,Z2,cmap='cividis', edgecolor='none', rcount=20, ccount=20)
    else:
        Z2 = np.array(Z2).reshape(xs.shape)
        ax.plot_surface(xs,ys,Z2-1,cmap='cividis', edgecolor='none', rcount=20, ccount=20)
    ax.set_xlabel('opinion value $o_{i}$')
    ax.set_zlabel('action selection ratio')
    ax.set_ylabel('descriptive concordant beliefs $b(o_{i})$')
    plt.gca().invert_xaxis()
    #plt.gca().invert_yaxis()
    plt.show()

def plot_surface():
    ax = plt.axes(projection='3d')
    

    
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    
    #constants.cen_belief = 2.5,2
    #constants.cen_true_distr = 4,2
    
    
    xs, ys = np.meshgrid(x, y)
    Z=[]
    for i in range(len(xs)):
        for j in range(len(xs[0])):
            Z.append((xs[i][j]+ ys[i][j])/2)
            
            
    
    # reshape Z
    
    Z = np.array(Z).reshape(xs.shape)
    
    # interpolate your values on the grid defined above
    f_interp = RectBivariateSpline(x,y, Z)
    X_grid, Y_grid = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    Z_grid = f_interp(np.linspace(0,1,100), np.linspace(0,1,100))
    Z2_grid = f_interp(np.linspace(0,1,10), np.linspace(0,1,10))
    ax.plot_surface(xs,ys,Z,
                cmap='viridis', edgecolor='none', rcount=200, ccount=200)
    
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()

#plot_surface()

def simple_repeated_interaction():
    class Player():
        def __init__(self,op,u_bar):
            self.op = op
            self.u_bar = u_bar
    
    u_bar = 0.3
    cost = lambda x : entropy([x,1-x])
    util = lambda op : op if op > 0.5 else (1-op)
    orig_distr = (2,5)
    theta_prior = (2,1)
    theta_prime = theta_prior
    op_distr = None
    players = [Player(o,u_bar) for o in np.random.uniform(low=0.5,high=1,size=int(100*(orig_distr[0]/sum(orig_distr))))]
    players.extend([Player(o,u_bar) for o in np.random.uniform(low=0,high=0.5,size=int(100*(orig_distr[1]/sum(orig_distr))))])
    np.random.shuffle(players)
    bels = []
    print('prior:',theta_prior[0]/sum(theta_prior))
    for run_idx in np.arange(300):
        '''
        if run_idx ==0:
            plt.hist([pl.op for pl in players],bins=10)
            plt.show()
        '''
        for pl in players:
            if pl.op>=0.5:
                bel_op = theta_prime[0]/sum(theta_prime)
            else:
                bel_op = 1-(theta_prime[0]/sum(theta_prime))
            prob_of_N = (bel_op*util(pl.op))/u_bar
            pl.act = 'e' if prob_of_N > 1 else 'n'
        op_distr = np.mean([pl.op if pl.op > 0.5 else 1-pl.op for pl in players if pl.act=='e'])
        theta_prime_rate = np.mean([pl.op for pl in players if pl.act=='e'])
        n_appr = np.sum([True if pl.act=='e' and pl.op>=0.5 else False for pl in players])
        n_disappr = np.sum([True if pl.act=='e' and pl.op<0.5 else False for pl in players])
        #theta_prime_rate = n_appr/(n_appr+n_disappr)
        h = theta_prime_rate*10
        theta_prime = (theta_prime[0]+h,theta_prime[1]+(10-h))
        print(run_idx,theta_prime[0]/sum(theta_prime),Counter([pl.act for pl in players]))
        '''
        plt.hist([pl.op for pl in players if pl.act=='e'],bins=10)
        plt.show()
        '''
        bels.append(theta_prime[0]/sum(theta_prime))
    theta_prior_h = theta_prior[0]/sum(theta_prior)
    eq_ratio = ((u_bar/theta_prior_h)-0.5)/((u_bar/theta_prior_h)+(u_bar/(1-theta_prior_h))-1)
    
    print(theta_prime[0]/sum(theta_prime),op_distr)
    return bels
#simple_repeated_interaction()
'''
end_thetas = []
for i in np.arange(100):
    bels = simple_repeated_interaction()
    plt.plot(np.arange(len(bels)), bels, color='blue')
    end_thetas.append(bels[-1])
plt.figure()
plt.hist(end_thetas,bins=10)
plt.show()
'''
def simple_egocentric_two_context_model_repeated_interaction():
    class Player():
        def __init__(self,op,u_bar):
            self.op = op
            self.u_bar = u_bar
    
    u_bar = 0.3
    cost = lambda x : entropy([x,1-x])
    util = lambda op : op if op > 0.5 else (1-op)
    orig_distr_ctx1 = (4,2)
    orig_distr_ctx2 = (2.2,2)
    theta_prior = np.array([[1,2],[2,1]])
    theta_prime = theta_prior
    context_weights = [0.6,0.4]
    op_samples = utils.generate_correleted_opinions(marginal_params=[orig_distr_ctx1,orig_distr_ctx2],
                                                     correlation_val=0.6, 
                                                     size=100)
    #h = sns.jointplot(op_samples[:,0], op_samples[:,1], kind='kde', xlim=(0, 1), ylim=(0, 1), fill=True);

    #plt.show()
    players = [Player(None,u_bar) for i in np.arange(op_samples.shape[0])]
    for pl_idx,pl in enumerate(players):
        ct_idx = np.random.choice([0,1],p=context_weights)
        pl.norm_context = ct_idx
        pl.op = op_samples[pl_idx,ct_idx]
    bels_ctx1,bels_ctx2 = [],[]
    #print('prior:',theta_prior[0]/sum(theta_prior))
    for run_idx in np.arange(300):
        '''
        if run_idx ==0:
            plt.hist([pl.op for pl in players],bins=10)
            plt.show()
        '''
        for pl in players:
            if pl.op>=0.5:
                bel_op = theta_prime[pl.norm_context,0]/sum(theta_prime[pl.norm_context,:])
            else:
                bel_op = 1-(theta_prime[pl.norm_context,0]/sum(theta_prime[pl.norm_context,:]))
            prob_of_N = (bel_op*util(pl.op))/u_bar
            pl.act = 'e' if prob_of_N > 1 else 'n'
        op_distr = np.mean([pl.op if pl.op > 0.5 else 1-pl.op for pl in players if pl.act=='e'])
        theta_prime_rate = np.mean([pl.op for pl in players if pl.act=='e'])
        n_appr = np.sum([True if pl.act=='e' and pl.op>=0.5 else False for pl in players])
        n_disappr = np.sum([True if pl.act=='e' and pl.op<0.5 else False for pl in players])
        #theta_prime_rate = n_appr/(n_appr+n_disappr)
        h = theta_prime_rate*10
        theta_prime = np.array([[theta_prime[0,0]+h,theta_prime[0,1]+(10-h)],[theta_prime[1,0]+h,theta_prime[1,1]+(10-h)]])
        print(run_idx,(theta_prime[0,0]/sum(theta_prime[0,:]),theta_prime[1,0]/sum(theta_prime[1,:])),op_distr)
        '''
        plt.hist([pl.op for pl in players if pl.act=='e'],bins=10)
        plt.show()
        '''
        bels_ctx1.append(theta_prime[0,0]/sum(theta_prime[0,:]))
        bels_ctx2.append(theta_prime[1,0]/sum(theta_prime[1,:]))
    theta_prior_h = theta_prior[0]/sum(theta_prior)
    eq_ratio = ((u_bar/theta_prior_h)-0.5)/((u_bar/theta_prior_h)+(u_bar/(1-theta_prior_h))-1)
    
    print(theta_prime[0]/sum(theta_prime),op_distr)
    return bels_ctx1,bels_ctx2
'''
end_thetas = []
for i in np.arange(100):
    bels_ctx1,bels_ctx2 = simple_egocentric_two_context_model_repeated_interaction()
    plt.plot(np.arange(len(bels_ctx1)), bels_ctx1, color='blue')
    end_thetas.append(bels_ctx1[-1])
    plt.plot(np.arange(len(bels_ctx2)), bels_ctx2, color='red')
    end_thetas.append(bels_ctx2[-1])
plt.figure()
plt.hist(end_thetas,bins=10)
plt.show()   
'''
def simple_normative_signal_two_context_model_repeated_interaction(b_idx):
    class Player():
        def __init__(self,op,u_bar):
            self.op = op
            self.u_bar = u_bar
    
    u_bar = 0.3
    cost = lambda x : entropy([x,1-x])
    util = lambda op : op if op > 0.5 else (1-op)
    orig_distr_ctx1 = (4,2)
    orig_distr_ctx2 = (2.2,2)
    theta_prior = {0:[1,2],1:[2,1]}
    belief_prior = {0:0.33,1:0.66}
    theta_prime = theta_prior
    belief_prime = {k:v for k,v in belief_prior.items()}
    context_weights = [0.6,0.4]
    weights_prime = {0:0.6,1:0.4}
    signal_theta_prime = {0:0.66,1:0.66}
    signal_ab_prime = {0:[2,1],1:[2,1]}
    ''' What each context believes about the signal weights '''
    signal_weights_prime = {0:0.4,1:0.6}
    op_samples = utils.generate_correleted_opinions(marginal_params=[orig_distr_ctx1,orig_distr_ctx2],
                                                     correlation_val=0.6, 
                                                     size=100)
    #h = sns.jointplot(op_samples[:,0], op_samples[:,1], kind='kde', xlim=(0, 1), ylim=(0, 1), fill=True);

    #plt.show()
    players = [Player(None,u_bar) for i in np.arange(op_samples.shape[0])]
    for pl_idx,pl in enumerate(players):
        ct_idx = np.random.choice([0,1],p=context_weights)
        pl.norm_context = ct_idx
        pl.op = op_samples[pl_idx,ct_idx]
    bels_ctx1,bels_ctx2 = [],[]
    signal_interpretations_ctx1, signal_interpretations_ctx2 = [], []
    #print('prior:',theta_prior[0]/sum(theta_prior))
    for run_idx in np.arange(100):
        '''
        if run_idx ==0:
            plt.hist([pl.op for pl in players],bins=10)
            plt.show()
        '''
        print(b_idx,'-',run_idx)
        for pl in players:
            if pl.op>=0.5:
                bel_op = sum([weights_prime[pl.norm_context]*belief_prime[pl.norm_context],signal_weights_prime[pl.norm_context]*signal_theta_prime[pl.norm_context]])
            else:
                bel_op = 1-sum([weights_prime[pl.norm_context]*belief_prime[pl.norm_context],signal_weights_prime[pl.norm_context]*signal_theta_prime[pl.norm_context]])
            prob_of_N = (bel_op*util(pl.op))/u_bar
            pl.act = 'e' if prob_of_N > 1 else 'n'
        op_distr = np.mean([pl.op if pl.op > 0.5 else 1-pl.op for pl in players if pl.act=='e'])
        n_appr = np.sum([True if pl.act=='e' and pl.op>=0.5 else False for pl in players])
        n_disappr = np.sum([True if pl.act=='e' and pl.op<0.5 else False for pl in players])
        
        obs = [pl.op for pl in players if pl.act=='e']
        obs_samples = [(np.sum(np.random.choice([1,0],size=10,p=[op_val,1-op_val])),) for op_val in obs]
        xs = [(s[0],10-s[0]) for s in obs_samples]
        done_flag = {0:False,1:True}
        for nctx in [0,1]:
            for pl in players:
                if all(list(done_flag.values())):
                    break
                thetas = np.array([[belief_prime[pl.norm_context],1-belief_prime[pl.norm_context]],
                                    [signal_theta_prime[pl.norm_context],1-signal_theta_prime[pl.norm_context]]])
                i, thetas, ws = utils.em(xs, thetas)
                ws = np.mean(ws,axis=1)
                ''' This will be the same for a given context '''
                _n_h = thetas[0][0]*10 
                theta_prime[pl.norm_context] = [theta_prime[pl.norm_context][0]+_n_h,theta_prime[pl.norm_context][1]+(10-_n_h)]
                #belief_prime[pl.norm_context] = thetas[0][0]
                belief_prime[pl.norm_context] = theta_prime[pl.norm_context][0]/sum(theta_prime[pl.norm_context])
                weights_prime[pl.norm_context] = ws[0]
                _n_h = thetas[1][0]*10 
                signal_ab_prime[pl.norm_context] = [signal_ab_prime[pl.norm_context][0]+_n_h,signal_ab_prime[pl.norm_context][1]+(10-_n_h)]
                #signal_theta_prime[pl.norm_context] = thetas[1][0]
                signal_theta_prime[pl.norm_context] = signal_ab_prime[pl.norm_context][0]/sum(signal_ab_prime[pl.norm_context])
                signal_weights_prime[pl.norm_context] = ws[1]
                done_flag[pl.norm_context] = True
        
        '''
        plt.hist([pl.op for pl in players if pl.act=='e'],bins=10)
        plt.show()
        '''
        bels_ctx1.append(belief_prime[0])
        bels_ctx2.append(belief_prime[1])
        
        #signal_interpretations_ctx1.append(signal_theta_prime[0])
        #signal_interpretations_ctx2.append(signal_theta_prime[1])
        
        #signal_interpretations_ctx1.append(sum([weights_prime[0]*belief_prime[0],signal_weights_prime[0]*signal_theta_prime[0]]))
        #signal_interpretations_ctx2.append(sum([weights_prime[1]*belief_prime[1],signal_weights_prime[1]*signal_theta_prime[1]]))
                
    
    
    return bels_ctx1,bels_ctx2,signal_interpretations_ctx1,signal_interpretations_ctx2
#action_selection_gird()
'''
fig, ax = plt.subplots(nrows=1, ncols=2)
end_thetas,end_signal_w_thetas = [],[]
for i in np.arange(100):
    bels_ctx1,bels_ctx2,signal_interpretations_ctx1,signal_interpretations_ctx2 = simple_normative_signal_two_context_model_repeated_interaction(i)
    ax[0].plot(np.arange(len(bels_ctx1)), bels_ctx1, color='blue')
    end_thetas.append(bels_ctx1[-1])
    ax[0].plot(np.arange(len(bels_ctx2)), bels_ctx2, color='red')
    end_thetas.append(bels_ctx2[-1])
    
    ax[1].plot(np.arange(len(signal_interpretations_ctx1)), signal_interpretations_ctx1, color='lightblue')
    ax[1].plot(np.arange(len(signal_interpretations_ctx2)), signal_interpretations_ctx2, color='pink')
    
plt.figure()
plt.hist(end_thetas,bins=10)
plt.show()    
'''