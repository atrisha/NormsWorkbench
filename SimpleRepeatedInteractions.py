'''
Created on 12 Jan 2023

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


def simple_repeated_interaction():
    class Player():
        def __init__(self,op,u_bar):
            self.op = op
            self.u_bar = u_bar
    
    u_bar = 0.3
    cost = lambda x : entropy([x,1-x])
    util = lambda op : op if op > 0.5 else (1-op)
    orig_distr = (2,1.6)
    theta_prior = (2,1.3)
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

end_thetas = []
for i in np.arange(100):
    bels = simple_repeated_interaction()
    plt.plot(np.arange(len(bels)), bels, color='blue')
    end_thetas.append(bels[-1])
plt.figure()
plt.hist(end_thetas,bins=10)
plt.show()

class Simple_egocentric_two_context_model_repeated_interaction():
    def play_dynamic(self):
        class Player():
            def __init__(self,op,u_bar):
                self.op = op
                self.u_bar = u_bar
        
        u_bar = 0.3
        cost = lambda x : entropy([x,1-x])
        util = lambda op : op if op > 0.5 else (1-op)
        orig_distr_ctx1 = (4,2)
        orig_distr_ctx2 = (1,3)
        theta_prior = np.array([[8,4],[2,8]])
        theta_prime = theta_prior
        context_weights = [0.7,0.3]
        op_samples = utils.generate_correleted_opinions(marginal_params=[orig_distr_ctx1,orig_distr_ctx2],
                                                         correlation_val=-0.6, 
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
        
        #print(theta_prime[0]/sum(theta_prime),op_distr)
        print('op expectation:',((context_weights[0]*4/6) + (context_weights[1]*1/4)))
        print('bel expectation:',((context_weights[0]*8/12) + (context_weights[1]*2/8)))
        return bels_ctx1,bels_ctx2

    def run(self):
        end_thetas = []
        context_weights = [0.7,0.3]
        for i in np.arange(100):
            bels_ctx1,bels_ctx2 = self.play_dynamic()
            plt.plot(np.arange(len(bels_ctx1)), bels_ctx1, color='blue')
            #end_thetas.append(bels_ctx1[-1])
            plt.plot(np.arange(len(bels_ctx2)), bels_ctx2, color='red')
            end_thetas.append((context_weights[0]*bels_ctx1[-1]) + (context_weights[1]*bels_ctx2[-1]))
        
        plt.figure()
        plt.hist(end_thetas,bins=10)
        print('mean population end thetas:',np.mean(end_thetas))
        plt.show()   
'''       
env_obj = Simple_egocentric_two_context_model_repeated_interaction()
env_obj.run()
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