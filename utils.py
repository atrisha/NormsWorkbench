'''
Created on 7 Sept 2022

@author: Atrisha
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import scipy.stats as stats
from scipy.stats import beta
import itertools
from Equilibria import CorrelatedEquilibria, PureNashEquilibria
from collections import Counter
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
import warnings

def plot_beta(a,b,ax=None,color=None,label=None,linestyle='-'):
    x = np.linspace(beta.ppf(0.01, a, b),beta.ppf(0.99, a, b), 100)
    plot_color = 'black' if color is None else color
    if ax is None:
        plt.figure(figsize=(7,7))
        plt.xlim(0, 1)
        plt.plot(x, beta.pdf(x, a, b), linestyle='-', color=plot_color)
        plt.title('Beta Distribution', fontsize='15')
        plt.xlabel('Values of Random Variable X (0, 1)', fontsize='15')
        plt.ylabel('Probability', fontsize='15')
        plt.show()
    else:
        ax.set_xlim(0, 1)
        ax.plot(x, beta.pdf(x, a, b), linestyle=linestyle if linestyle is not None else '-', color=plot_color,label=label)
        #ax.set_xlabel('Values of Random Variable X (0, 1)', fontsize='15')
        #ax.set_ylabel('Probability', fontsize='15')
        
    
def eq(a,b):
    return a==b 

def less(a,b):
    return a < b

def leq(a,b):
    return a <= b

def generate_2_x_2_staghunt(r_info,players,for_recommendation_ex_post):
    get_oth_opinion = lambda x : 'A' if x == 'D' else 'D'
    num_players = 2
    all_strats = list(itertools.product(*[['A','D'],['A','D']]))
    max_payoff = 1
    payoff_dict = dict()
    for pl_idx, player in enumerate(players):
        assert(player.opinion_val<=1)
        strat_player_opinion = ''.join([player.opinion]*num_players)
        strat_not_player_opinion = ''.join([get_oth_opinion(player.opinion)]*num_players)
        player.act_payoffs = {strat_player_opinion:player.opinion_val,strat_not_player_opinion:max_payoff-player.opinion_val}
        if for_recommendation_ex_post:
            minorty_op_check = True if player.opinion != r_info.maj_opinion else False
        else:
            minorty_op_check = True if player.opinion != player.oth_opinion_bel else False
        if minorty_op_check:
            ''' Player descriptive opinion belief in minority. So risk of sharing not opinion is less that sharing opinion '''
            r_player_opinion = np.random.uniform(max(0,player.opinion_val-(1-player.opinion_val)),max_payoff)
            r_not_player_opinion = np.random.uniform()*r_player_opinion
        else:
            ''' Player descriptive opinion belief in majority '''
            r_not_player_opinion = np.random.uniform(player.opinion_val-(1-player.opinion_val),max_payoff)
            r_player_opinion = np.random.uniform(player.opinion_val-(1-player.opinion_val),r_not_player_opinion)
        strat_player_opinion_defection = ''.join([player.opinion if i==pl_idx else get_oth_opinion(player.opinion) for i in np.arange(num_players)])
        strat_not_player_opinion_defection = ''.join([get_oth_opinion(player.opinion) if i==pl_idx else player.opinion for i in np.arange(num_players)])
        player.act_payoffs[strat_player_opinion_defection] = player.act_payoffs[strat_player_opinion] - r_player_opinion
        player.act_payoffs[strat_not_player_opinion_defection] = player.act_payoffs[strat_not_player_opinion] - r_not_player_opinion
    payoff_dict = {}
    for k in list(players[0].act_payoffs.keys()):
        payoff_dict[tuple(list(k))] = [players[i].act_payoffs[k] for i in np.arange(num_players)]
    #pne = PureNashEquilibria()
    #pne_res = pne.solve(payoff_dict)
    return payoff_dict

get_oth_opinion = lambda x : 'A' if x == 'D' else 'D'

def check_pdrd_constraint(pl_op,pl_bel,pl_idx,payoff_dict_inp):
    payoff_dict = {''.join(list(k)):v for k,v in payoff_dict_inp.items()}
    try:
        if pl_op != pl_bel:
            if pl_idx == 0:
                assert abs(payoff_dict[''.join([pl_op,pl_op])][pl_idx]-payoff_dict[''.join([pl_op,get_oth_opinion(pl_op)])][pl_idx]) >= abs(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx]-payoff_dict[''.join([get_oth_opinion(pl_op),pl_op])][pl_idx])
            else:
                assert abs(payoff_dict[''.join([pl_op,pl_op])][pl_idx]-payoff_dict[''.join([get_oth_opinion(pl_op),pl_op])][pl_idx]) >= abs(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx]-payoff_dict[''.join([pl_op,get_oth_opinion(pl_op)])][pl_idx])   
            #assert payoff_dict[''.join([pl_op,pl_op])][pl_idx] < payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx], str(payoff_dict)+'\n'+str([pl_op,pl_bel,pl_idx])
        else:
            if pl_idx == 0:
                assert abs(payoff_dict[''.join([pl_op,pl_op])][pl_idx]-payoff_dict[''.join([pl_op,get_oth_opinion(pl_op)])][pl_idx]) <= abs(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx]-payoff_dict[''.join([get_oth_opinion(pl_op),pl_op])][pl_idx])
            else:
                assert abs(payoff_dict[''.join([pl_op,pl_op])][pl_idx]-payoff_dict[''.join([get_oth_opinion(pl_op),pl_op])][pl_idx]) <= abs(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx]-payoff_dict[''.join([pl_op,get_oth_opinion(pl_op)])][pl_idx])
        assert payoff_dict[''.join([pl_op,pl_op])][pl_idx] >= payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx], str(payoff_dict[''.join([pl_op,pl_op])][pl_idx]) +  str(payoff_dict[''.join([get_oth_opinion(pl_op),get_oth_opinion(pl_op)])][pl_idx])
    except AssertionError:
        f=1
        raise
        
def get_rd_eq(payoff_dict, pl_idx, pl):
    pl_op = pl.opinion
    pl_not_op = get_oth_opinion(pl_op)
    risk_of_op_val = abs(payoff_dict[(pl_op,pl_op)][pl_idx] - payoff_dict[(pl_op,get_oth_opinion(pl_op))][pl_idx])
    risk_of_not_op_val = abs(payoff_dict[(pl_not_op,pl_not_op)][pl_idx] - payoff_dict[(pl_not_op,get_oth_opinion(pl_not_op))][pl_idx])
    if pl_idx == 0:
        if  risk_of_op_val > risk_of_not_op_val : \
                   return (pl_not_op,pl_not_op)
        else:
            return (pl_op,pl_op)
    else:
        if abs(payoff_dict[(pl_op,pl_op)][pl_idx] - payoff_dict[(get_oth_opinion(pl_op),pl_op)][pl_idx]) \
               > abs(payoff_dict[(pl_not_op,pl_not_op)][pl_idx] - payoff_dict[(get_oth_opinion(pl_not_op),pl_not_op)][pl_idx]): \
                   return (pl_not_op,pl_not_op)
        else:
            return (pl_op,pl_op)
def get_act_risk(payoff_dict, pl_idx, act):
    if pl_idx == 0:
        risk_of_act_val = abs(payoff_dict[(act,act)][pl_idx] - payoff_dict[(act,get_oth_opinion(act))][pl_idx])
    else:
        risk_of_act_val = abs(payoff_dict[(act,act)][pl_idx] - payoff_dict[(get_oth_opinion(act),act)][pl_idx])
    return risk_of_act_val

def get_pd_eq(payoff_dict, pl_idx, pl):
    pl_op = pl.opinion
    pl_not_op = get_oth_opinion(pl_op)
    return (pl_op,pl_op) if payoff_dict[(pl_op,pl_op)][pl_idx] > payoff_dict[(pl_not_op,pl_not_op)][pl_idx] else (pl_not_op,pl_not_op)
        
op_category_list = ['SD','D','A','SA']

def sample_op_val(dirichlet_parms,person_object,from_person_object=None):
    multi_nom_parms = np.random.dirichlet(alpha=dirichlet_parms)
    ''' The opinion categories stand for disstrong approval to strong approval'''
                   
    op_category_sample = np.random.choice(op_category_list,p=multi_nom_parms)
    appr_val_map = {'SD':[0,.25],'D':[0.25,0.5],'A':[0.5,0.75],'SA':[0.75,1]}
    appr_val = np.random.uniform(low=appr_val_map[op_category_sample][0], high=appr_val_map[op_category_sample][1])  
    if from_person_object is None:
        person_object.opinion_expanded = op_category_sample
        person_object.opinion = 'A' if appr_val > 0.5 else 'D' if appr_val < 0.5 else np.random.choice(['A','D'])
        person_object.appr_val = appr_val
        person_object.opinion_val = appr_val if person_object.opinion == 'A' else 1-appr_val
    else:
        person_object.opinion_expanded = from_person_object.opinion_expanded
        person_object.opinion = from_person_object.opinion
        person_object.appr_val = from_person_object.appr_val
        person_object.opinion_val = from_person_object.opinion_val
    return appr_val

def get_expanded_action_freq(players_list):
    players_expanded_ops = [pl.action for pl in players_list if pl.action != 'N']
    freq_vals_counter = Counter(players_expanded_ops)
    return [freq_vals_counter['SD'],freq_vals_counter['D'],freq_vals_counter['A'],freq_vals_counter['SA']]
               

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
def correlation_constraints(ref_cor_index,opinion_param_samples):
    corr_matrix_upp_bound = np.full(shape=(4,4),fill_value=-np.inf) if ref_cor_index[1] is None else np.full(shape=(4,4,2),fill_value=-np.inf)
    mu_s = opinion_param_samples.tolist()
    psi_i = lambda mu : np.sqrt(mu/(1-mu)) 
    for i in np.arange(4):
        for j in np.arange(4):
            if i > j:
                r_ij = [max([-psi_i(mu_s[i])*psi_i(mu_s[j]),-1/(psi_i(mu_s[i])*psi_i(mu_s[j]))]),min([psi_i(mu_s[i])/psi_i(mu_s[j]),psi_i(mu_s[j])/psi_i(mu_s[i])])]
                #print('r_'+str(i)+','+str(j)+": "+str(r_ij))
                if ref_cor_index[1] is None:
                    corr_matrix_upp_bound[i,j] = np.random.uniform(0,r_ij[1]) if j!=ref_cor_index[0] else np.random.uniform(r_ij[0],0)
                else:
                    corr_matrix_upp_bound[i,j] = np.array([r_ij[0],r_ij[1]])
            if i==j:
                corr_matrix_upp_bound[i,j] = 1 if ref_cor_index[1] is None else np.array([1,1])
    for i in np.arange(4):
        for j in np.arange(4):
            if np.any(corr_matrix_upp_bound[i,j] == -np.inf):
                corr_matrix_upp_bound[i,j] = corr_matrix_upp_bound[j,i] if ref_cor_index[1] is None else corr_matrix_upp_bound[j,i,:]
    corr_sums = np.sum(corr_matrix_upp_bound,axis=0)
    return corr_matrix_upp_bound
    
def plot_gaussian(mu,variance):
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.show()

def generate_samples(op_marginals,ref_cor_index):
    success = False
    while(not success):
        success = True
        base = importr('base')
        mipfp = importr('mipfp')
        questionr = importr('questionr')
        effectsize = importr('effectsize')
        infotheo = importr('infotheo')
        
        robjects.r('''
            # create a function `f`
                    f <- function(or,p){
              #or <- Corr2Odds(or,marg.probs=p)$odds
              rownames(or) <- colnames(or) <- c("n1", "n2", "n3", "n4")
              # estimating the joint-distribution
              p.joint <- ObtainMultBinaryDist(corr = or, marg.probs = p)
              samples <- RMultBinary(n = 100, mult.bin.dist = p.joint)$binary.sequences
              samples <- data.frame(samples)
              return(samples)
            }
            
            ''')
        robjects.r('''
            # create a function `f`
                    f_mutual_info <- function(samples){
                    mutual_info_mat <- mutinformation(samples)
              return(mutual_info_mat)
            }
            
            ''')
        rpy2.robjects.numpy2ri.activate()
        r_f = robjects.globalenv['f']
        #neg_cor_index = np.random.choice(np.arange(0,4))
        #opinion_param_samples = np.random.uniform(low=0.2,high=0.9,size=4)
        opinion_param_samples = np.array(op_marginals)
        corr_mat = correlation_constraints(ref_cor_index,opinion_param_samples)
        nr,nc = corr_mat.shape
        Br = robjects.r.matrix(corr_mat, nrow=nr, ncol=nc)
        robjects.r.assign("C", Br)
        try:
            p = robjects.FloatVector(opinion_param_samples)
            res = r_f(Br,p)
            output = np.array([list([res[j][i] for j in range(res.ncol)]) for i in range(res.nrow)])
            
            robjects.r.assign("samples", res)
            robjects.r('mi_mat <- mutinformation(samples)')
            '''
            samples_r = robjects.r.matrix(output, nrow=output.shape[0], ncol=output.shape[1])
            r_f_mi = robjects.globalenv['f_mutual_info']
            print(samples_r)
            '''
            
            mi_matrix = robjects.r['mi_mat']
            mi_matrix = np.asarray(a=mi_matrix,dtype=np.float16)
        except Exception:
            print('ran with',corr_mat,opinion_param_samples)
            success = False
    marginals = np.sum(output,axis=0)/100
    return output,corr_mat,mi_matrix

def generate_corr_mat_grid(op_marginals,ref_cor_index):
    opinion_param_samples = np.array(op_marginals)
    corr_mat = correlation_constraints(ref_cor_index,opinion_param_samples)
    return corr_mat

def generate_grid_samples(corr_mat,op_marginals,ref_cor_index):
    success = False
    while(not success):
        success = True
        base = importr('base')
        mipfp = importr('mipfp')
        questionr = importr('questionr')
        effectsize = importr('effectsize')
        infotheo = importr('infotheo')
        
        robjects.r('''
            # create a function `f`
                    f <- function(or,p){
              #or <- Corr2Odds(or,marg.probs=p)$odds
              rownames(or) <- colnames(or) <- c("n1", "n2", "n3", "n4")
              # estimating the joint-distribution
              p.joint <- ObtainMultBinaryDist(corr = or, marg.probs = p)
              samples <- RMultBinary(n = 100, mult.bin.dist = p.joint)$binary.sequences
              samples <- data.frame(samples)
              return(samples)
            }
            
            ''')
        robjects.r('''
            # create a function `f`
                    f_mutual_info <- function(samples){
                    mutual_info_mat <- mutinformation(samples)
              return(mutual_info_mat)
            }
            
            ''')
        rpy2.robjects.numpy2ri.activate()
        r_f = robjects.globalenv['f']
        if ref_cor_index[1] is not None:
            extracted_corr_mat = np.full(shape=(4,4),fill_value=-np.inf)
            for i in np.arange(extracted_corr_mat.shape[0]):
                for j in np.arange(extracted_corr_mat.shape[1]):
                    if i==ref_cor_index[0] or j==ref_cor_index[0]:
                        extracted_corr_mat[i,j] = np.linspace(corr_mat[i,j,0],corr_mat[i,j,1],5)[ref_cor_index[1]]
                    else:
                        extracted_corr_mat[i,j] = corr_mat[i,j,1]
            corr_mat = extracted_corr_mat
        opinion_param_samples = np.array(op_marginals)
        nr,nc = corr_mat.shape
        Br = robjects.r.matrix(corr_mat, nrow=nr, ncol=nc)
        robjects.r.assign("C", Br)
        try:
            p = robjects.FloatVector(opinion_param_samples)
            res = r_f(Br,p)
            output = np.array([list([res[j][i] for j in range(res.ncol)]) for i in range(res.nrow)])
            
            robjects.r.assign("samples", res)
            robjects.r('mi_mat <- mutinformation(samples)')
            '''
            samples_r = robjects.r.matrix(output, nrow=output.shape[0], ncol=output.shape[1])
            r_f_mi = robjects.globalenv['f_mutual_info']
            print(samples_r)
            '''
            
            mi_matrix = robjects.r['mi_mat']
            mi_matrix = np.asarray(a=mi_matrix,dtype=np.float16)
        except Exception:
            print('ran with',corr_mat,opinion_param_samples)
            success = False
    marginals = np.sum(output,axis=0)/100
    return output,corr_mat,mi_matrix

def get_mom_update(sample_mean,sample_var):
    mom_alpha_est = lambda u,v : u * (((u*(1-u))/v) - 1)
    mom_beta_est = lambda u,v : (1-u) * (((u*(1-u))/v) - 1)
    return np.array([mom_alpha_est(sample_mean,sample_var), mom_beta_est(sample_mean,sample_var)]).reshape((1,2))

def em(xs, thetas, max_iter=100, tol=1e-6):
    """Expectation-maximization for coin sample problem."""
    ll_old = -np.infty
    thetas = np.where(thetas < 10**-6,thetas+(10**-6),thetas)
    thetas = np.where(thetas > (1-10**-6),thetas-(10**-6),thetas)
    #warnings.simplefilter('error',RuntimeWarning)
    for i in range(max_iter):
        try:
            ll = np.array([np.sum(xs * np.log(theta), axis=1) for theta in thetas])
            lik = np.exp(ll)
            ws = lik/lik.sum(0)
            exps = np.array([w[:, None] * xs for w in ws])
            thetas = np.array([expr.sum(0)/expr.sum() for expr in exps])
            thetas = np.where(thetas < 10**-6,thetas+(10**-6),thetas)
            thetas = np.where(thetas > (1-10**-6),thetas-(10**-6),thetas)
            ll_new = np.sum([w*l for w, l in zip(ws, ll)])
            if np.abs(ll_new - ll_old) < tol:
                break
            ll_old = ll_new
        except RuntimeWarning:
            f=1
            raise
    return i, thetas, ws

def mle(xs):
    xs = list(xs)
    updated_thetas = dict()
    updated_props = Counter([x[1] for x in xs])
    updated_props = {k:v/len(xs) for k,v in updated_props.items()}
    for x in xs:
        if x[1] not in updated_thetas:
            updated_thetas[x[1]] = []
        updated_thetas[x[1]].append(x[0][0]/10)
    updated_thetas = {k:np.mean(v) for k,v in updated_thetas.items()}
    return updated_thetas,updated_props

def get_priors_from_true(true_distr):
    ''' Convert true means to beta params '''
    alphas = [round(x*10) for x in true_distr]
    betas = [10-x for x in alphas]
    try:
        priors = [np.random.beta(x[0],x[1]) if (x[0]>0 and x[1]>0) else np.random.beta(1,10) if x[0] <= 0 else np.random.beta(10,1) for x in zip(alphas,betas)]
    except ValueError:
        print('ran with',true_distr)
        raise
    return priors

'''
opinions,corr_mat = generate_samples()
opinion_marginals = np.sum(opinions,axis=0)/100
opinion_marginals_props = [0.4,0.2,0.3,0.1]
sample_contexts = np.random.choice(np.arange(len(opinion_marginals_props)),size=100,p=opinion_marginals_props)
samples = [(np.sum(np.random.choice([1,0],size=10,p=[opinion_marginals[sample_contexts[i]],1-opinion_marginals[sample_contexts[i]]])),) for i in np.arange(100)]
xs = [(s[0],10-s[0]) for s in samples]
#xs = np.array([(5,5), (9,1), (8,2), (4,6), (7,3)])
#thetas = np.array([[0.6, 0.4], [0.5, 0.5]])
thetas = np.array([[x, 1-x] for x in opinion_marginals])

i, thetas, ws = em(xs, thetas)
print(i,'EM est.')
for theta in thetas:
    print(theta)
ws = np.mean(ws,axis=1)
print('prop',ws)
print('True est.')
print(opinion_marginals)
'''
'''
thetas = np.array([[1.52085510e-311, 9.99999000e-001],
 [5.76850271e-001, 4.23149729e-001],
 [9.45689684e-001, 5.43103164e-002],
 [8.11721432e-001, 1.88278568e-001]])
xs = [(8, 2), (8, 2), (9, 1), (10, 0), (8, 2), (10, 0), (8, 2), (7, 3), (9, 1), (10, 0), (10, 0), (8, 2), (10, 0), (6, 4), (10, 0), (6, 4), (7, 3), (10, 0), (9, 1), (10, 0), (4, 6), (10, 0), (5, 5), (7, 3), (8, 2), (4, 6), (10, 0), (8, 2), (8, 2), (5, 5), (9, 1), (9, 1), (10, 0), (5, 5), (9, 1), (6, 4), (8, 2), (7, 3), (5, 5), (8, 2), (7, 3), (6, 4), (7, 3), (5, 5), (10, 0), (8, 2), (5, 5), (9, 1), (10, 0), (10, 0), (7, 3), (7, 3), (5, 5), (9, 1), (6, 4), (6, 4), (6, 4), (8, 2), (9, 1), (5, 5), (5, 5), (7, 3)]
em(xs,thetas)
'''