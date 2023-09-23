import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sympy as sp
from sympy.integrals.intpoly import cross_product
import pandas as pd
from IPython.display import display

def solver(): 
    '''
    # Create symbols x1 to x4
    k, a, u = sp.symbols('k a u')
    
    # Create symbols p and q
    
    
    # Define the equations
    equations = [
    sp.Eq(((1-k)*a*(1-(u/k))) - (k*(1-(u/(1-k)))) , 0)
    ]
    solution = sp.solve(equations, k)
    '''
    k = sp.Symbol('k')
    a = sp.Symbol('a')
    u = sp.Symbol('u')
    b = sp.Symbol('b')
    x = sp.Symbol('x')
    #solution = sp.solveset(((1-k)*a*(1-(u/k))) - (k*a*(1-(u/(1-k)))), a, domain=sp.S.Reals)
    solution = sp.solveset(((1-k)*a*0.5*(1-(u**2/k**2))) - (k*a*0.5*(1-(u/(1-k)))**2) + ((1-k)*b*(1-(u/k))) - (k*b*(1-(u/(1-k)))), a, domain=sp.S.Reals)
    expr1 = ((1-k)*a*0.5*(1-(u**2/k**2))) - (k*a*0.5*(1-(u/(1-k)))**2) + ((1-k)*b*(1-(u/k))) - (k*b*(1-(u/(1-k))))
    # Print the solution
    print("Solution:", solution)
    print(sp.latex(eval('-0.8*b*k*(1.0*k - 1.0)*(1.0*k - 0.5)*(1.0*k**2 - 1.0*k + 1.0*u)/(0.4*k**5 + 0.4*k**4*u - 1.0*k**4 - 0.4*k**3*u + 0.8*k**3 + 0.6*k**2*u**2 - 0.2*k**2 - 0.6*k*u**2 + 0.2*u**2)')))
    expr = -0.8*b*k*(1.0*k - 1.0)*(1.0*k - 0.5)*(1.0*k**2 - 1.0*k + 1.0*u)/(0.4*k**5 + 0.4*k**4*u - 1.0*k**4 - 0.4*k**3*u + 0.8*k**3 + 0.6*k**2*u**2 - 0.2*k**2 - 0.6*k*u**2 + 0.2*u**2)
    for _k in np.linspace(0,1,10):
        for _u in np.linspace(0,1,10):
            for _b in np.linspace(0,1,10):
                eval_expr = expr.subs([(k, _k),(u, _u), (b, _b)])
                #print(_u,_k,_b,eval_expr)
                eval_expr1 = expr1.subs([(k, _k),(u, _u), (b, _b), (a,eval_expr)])
                print(eval_expr,eval_expr1)
                '''
                if eval_expr !=0:
                    int_l = sp.integrate(eval_expr*x**2 + _b*x, (x, 0, (1-(_u/(1-_k)))))
                    int_h = sp.integrate(eval_expr*x**2 + _b*x, (x, _u/_k, 1))
                    if int_l >0 and int_h >0:
                        print(int_l,int_h,int_h/(int_l+int_h),_k)
                '''
def executor():
    x1,x2,x3,x4 = 4/7,0,3/7,1
    p,q = 0.7, 0.3
    qri_cond_is = (x1*p)/(x1*p+x2*q)#*((x1+x2)/2))
    qri_cond_gs = (x3*p)/(x3*p+x4*q)#*((x3+x4)/2)
    qrg_cond_is = (x2*q)/(x1*p+x2*q)#*((x1+x2)/2))
    qrg_cond_gs = (x4*q)/(x3*p+x4*q)#*((x3+x4)/2)
    print(qri_cond_is,qri_cond_gs,qrg_cond_is,qrg_cond_gs)
    qri = qri_cond_is*((x1+x2)/2) + qri_cond_gs*((x3+x4)/2)
    print('guilty percentage times',p*qrg_cond_is+q*qrg_cond_gs)
    print('posterior guilty',(qrg_cond_is+qrg_cond_gs)/(qri_cond_is+qri_cond_gs+qrg_cond_is+qrg_cond_gs))
    
def beta_cdf():
    
    from scipy.stats import beta
    
    # Define the parameters of the Beta distribution
    a1 = 2  # Shape parameter
    b1 = 25    # Shape parameter
    a2 = 8  # Shape parameter
    b2 = 2   # Shape parameter
    mix = 0.15
    # Generate x values for the CDF plot
    x = np.linspace(0, 1, 1000)
    theta = mix*(a1/(a1+b1)) + (1-mix)*(a2/(a2+b2))
    u_bar = 0.2
    l_l_unclipped = 1-(u_bar/(1-theta))
    l_h_unclipped = u_bar/theta
    l_l = np.clip(1-(u_bar/(1-theta)),0,0.5)
    l_h = np.clip(u_bar/theta,0.5,1)

    # Calculate the CDF values using the beta distribution's CDF function
    cdf_l = mix*beta.cdf(l_l, a1, b1) + (1-mix)*beta.cdf(l_l, a2, b2)
    cdf_r = mix*(1 - beta.cdf(l_h, a1, b1)) + (1-mix)*(1 - beta.cdf(l_h, a2, b2))
    pdf = mix*beta.pdf(x, a1, b1) + (1-mix)*beta.pdf(x, a2, b2)
    pdf_e = []
    for _x in np.linspace(0,1,10000):
        if _x < l_l:
            pdf_e.append(mix*beta.pdf(_x, a1, b1) + (1-mix)*beta.pdf(_x, a2, b2))
        elif _x >= l_h:
            pdf_e.append(mix*beta.pdf(_x, a1, b1) + (1-mix)*beta.pdf(_x, a2, b2))
        else:
            pdf_e.append(0)
    norm_pdf_e = [x/np.sum(pdf_e) for x in pdf_e]
    exp_pdf_e = np.sum([x1*x2 for x1,x2 in zip(np.linspace(0,1,10000),norm_pdf_e)])
    plt.plot(x, pdf,'black')
    plt.plot(np.linspace(0,1,10000), pdf_e,'red')
    print('theta',theta)
    print('unclipped','l_l,l_h',l_l_unclipped,l_h_unclipped)
    print('l_l,l_h',l_l,l_h)
    print('new theta',exp_pdf_e)
    
    plt.show()


pr,pb = (0.3,0.7)
u_bar,theta = 0.4,pb   
def calc_bayes_plausible_distr(rr,rb,br,bb):
    #rr,rb,br,bb = sp.symbols('rr,rb,br,bb')
    #pr,pb = sp.symbols('pr,pb')
    #rr,rb,br,bb = 6/7,1/3,1/7,2/3
    pr,pb = (0.3,0.7)
    r_signal = (rr/(rr+br))*pr + (rb/(rb+bb))*pb
    b_signal = (br/(rr+br))*pr + (bb/(rb+bb))*pb
    pos_state_r_signal_r = (rr*pr)/(rr*pr + rb*pb)
    pos_state_r_signal_b = (br*pr)/(br*pr + bb*pb)
    pos_state_b_signal_r = (rb*pb)/(rb*pb + rr*pr)
    pos_state_b_signal_b = (bb*pb)/(bb*pb + br*pr)
    expected_posterior_r = r_signal*pos_state_r_signal_r + b_signal*pos_state_r_signal_b
    expected_posterior_b = r_signal*pos_state_b_signal_r + b_signal*pos_state_b_signal_b
    #equations = [expected_posterior_r-pr,expected_posterior_b-pb,rr+br-1,rb+bb-1]
    #solutions = sp.solve(equations, rr,rb,br,bb, dict=True)
    #solved = {bb: 1.0 - rb, br: 1.0 - rr}
    #print('expected posteriors',expected_posterior_r,expected_posterior_b)
    #print(solutions)
    #receivers = ['rs','rh','bh','bs']
    r_signal_ll = pos_state_r_signal_r*np.clip(1-(u_bar/pos_state_r_signal_r),0,0.5) + pos_state_b_signal_r*np.clip(1-(u_bar/(1-pos_state_b_signal_r)),0,0.5)
    r_signal_lh = pos_state_r_signal_r*np.clip(u_bar/(1-pos_state_r_signal_r),0.5,1)  + pos_state_b_signal_r*np.clip(u_bar/pos_state_b_signal_r,0.5,1) 
    b_signal_ll = pos_state_r_signal_b*np.clip(1-(u_bar/pos_state_r_signal_b),0,0.5) + pos_state_b_signal_b*np.clip(1-(u_bar/(1-pos_state_b_signal_b)),0,0.5)
    b_signal_lh = pos_state_r_signal_b*np.clip(u_bar/(1-pos_state_r_signal_b),0.5,1)  + pos_state_b_signal_b*np.clip(u_bar/pos_state_b_signal_b,0.5,1)
    exp_ll,exp_lh = r_signal*r_signal_ll + b_signal*b_signal_ll, r_signal*r_signal_lh + b_signal*b_signal_lh
    return exp_ll,exp_lh


def generate_bimodal_samples(mu,sigma = (0.1,0.1)):
    mean1 = mu[0]
    mean2 = mu[1]
    std_dev1 = 0.1
    std_dev2 = 0.1
    num_samples = 1000
    
    # Sample from the two normal distributions
    samples1 = np.random.normal(loc=mean1, scale=std_dev1, size=num_samples//2)
    samples2 = np.random.normal(loc=mean2, scale=std_dev2, size=num_samples//2)
    
    # Combine the samples to create the bimodal distribution
    bimodal_samples = np.concatenate([samples1, samples2])
    return bimodal_samples

class InfoDesign():
    
    def __init__(self):
        pr,pb = (0.3,0.7)
        u_bar,theta = 0.2,pb   
        self.exp_ll,self.exp_lh,self.util_b,self.util_r = None, None, None, None
    
    def single_sender_single_informed_receiver(self,rr,rb,br,bb):
        #rr,rb,br,bb = sp.symbols('rr,rb,br,bb')
        #pr,pb = sp.symbols('pr,pb')
        #rr,rb,br,bb = 6/7,1/3,1/7,2/3
        pr,pb = (0.3,0.7)
        u_bar,theta = 0.2,pb
        i_pref_b = 0.6
        r_signal = (rr/(rr+br))*pr + (rb/(rb+bb))*pb
        b_signal = (br/(rr+br))*pr + (bb/(rb+bb))*pb
        pos_state_r_signal_r = (rr*pr)/(rr*pr + rb*pb)
        pos_state_r_signal_b = (br*pr)/(br*pr + bb*pb)
        pos_state_b_signal_r = (rb*pb)/(rb*pb + rr*pr)
        pos_state_b_signal_b = (bb*pb)/(bb*pb + br*pr)
        expected_posterior_r = r_signal*pos_state_r_signal_r + b_signal*pos_state_r_signal_b
        expected_posterior_b = r_signal*pos_state_b_signal_r + b_signal*pos_state_b_signal_b
        exp_util_b = expected_posterior_b*i_pref_b
        exp_util_r = expected_posterior_r*(1-i_pref_b)
        #act = exp_util_b if exp_util_b>=exp_util_r else 0
        #equations = [expected_posterior_r-pr,expected_posterior_b-pb,rr+br-1,rb+bb-1]
        #solutions = sp.solve(equations, rr,rb,br,bb, dict=True)
        #solved = {bb: 1.0 - rb, br: 1.0 - rr}
        #print('expected posteriors',expected_posterior_r,expected_posterior_b)
        #print(solutions)
        #receivers = ['rs','rh','bh','bs']
        r_signal_ll = pos_state_r_signal_r*np.clip(1-(u_bar/pos_state_r_signal_r),0,0.5) + pos_state_b_signal_r*np.clip(1-(u_bar/(1-pos_state_b_signal_r)),0,0.5)
        r_signal_lh = pos_state_r_signal_r*np.clip(u_bar/(1-pos_state_r_signal_r),0.5,1)  + pos_state_b_signal_r*np.clip(u_bar/pos_state_b_signal_r,0.5,1) 
        b_signal_ll = pos_state_r_signal_b*np.clip(1-(u_bar/pos_state_r_signal_b),0,0.5) + pos_state_b_signal_b*np.clip(1-(u_bar/(1-pos_state_b_signal_b)),0,0.5)
        b_signal_lh = pos_state_r_signal_b*np.clip(u_bar/(1-pos_state_r_signal_b),0.5,1)  + pos_state_b_signal_b*np.clip(u_bar/pos_state_b_signal_b,0.5,1)
        exp_ll,exp_lh = r_signal*r_signal_ll + b_signal*b_signal_ll, r_signal*r_signal_lh + b_signal*b_signal_lh
        self.exp_ll,self.exp_lh,self.exp_util_b,self.exp_util_r = exp_ll, exp_lh,exp_util_b,exp_util_r
        ''' if the agent takes a cost to match the exp_lh'''
        
        return exp_ll,exp_lh
    
    def calc_pos_relative_to_representative(self,rep_post,rep_prior,rec_prior):
        ratio_1 = rec_prior/rep_prior
        ratio_2 = (1-rec_prior)/(1-rep_prior)
        rec_post = (rep_post*ratio_1)/(rep_post*ratio_1+(1-rep_post)*ratio_2)
        return rec_post
    
    def single_sender_multiple_receiver_nonnormative_nonmanipulative(self,rr,rb,br,bb,sender_prior):
        ''' generate all samples'''
        if hasattr(self, 'bimodal_samples'):
            bimodal_samples = self.bimodal_samples
        else:
            bimodal_samples = generate_bimodal_samples(mu=[0.4,0.6])
            self.bimodal_samples = bimodal_samples
        #pr,pb = (0.4,0.6)
        all_samples_res = []
        baseline_margin, moved_margin, moved_margin_norm = [],[],[]
        bimodal_samples.sort()
        
        rep_receiver_prior = sender_prior
        rep_receiver_prior_r,rep_receiver_prior_b = rep_receiver_prior
        pr,pb = rep_receiver_prior
        r_signal = (rr/(rr+br))*pr + (rb/(rb+bb))*pb
        b_signal = (br/(rr+br))*pr + (bb/(rb+bb))*pb
        repr_pos_r_r_signal = (rr*pr)/(rr*pr + rb*pb)
        repr_pos_r_b_signal = (br*pr)/(br*pr + bb*pb)
        repr_pos_b_b_signal = (bb*pb)/(bb*pb + br*pr)
        repr_pos_b_r_signal = (rb*pb)/(rb*pb + rr*pr)
        expected_posterior_r = r_signal*repr_pos_r_r_signal + b_signal*repr_pos_r_b_signal
        expected_posterior_b = r_signal*repr_pos_b_b_signal + b_signal*repr_pos_b_r_signal
        oth_group_bel,ref_grp_bel = 0.4,0.6
        for i in bimodal_samples:
            rec_prior_r,rec_prior_b = (1-i,i)
            pos_state_r_signal_r = self.calc_pos_relative_to_representative(repr_pos_r_r_signal,rep_receiver_prior_r,rec_prior_r)
            pos_state_r_signal_b = self.calc_pos_relative_to_representative(repr_pos_r_b_signal,rep_receiver_prior_r,rec_prior_r)
            pos_state_b_signal_r = self.calc_pos_relative_to_representative(repr_pos_b_r_signal,rep_receiver_prior_b,rec_prior_b)
            pos_state_b_signal_b = self.calc_pos_relative_to_representative(repr_pos_b_b_signal,rep_receiver_prior_b,rec_prior_b)
            #act_sig_b = 1 if pos_state_b_signal_b > i else 0
            #act_sig_r = 1 if pos_state_b_signal_r > i else 0
            if i>0.5:
                oth_group_bel,ref_grp_bel = 0.4,0.6
            else:
                oth_group_bel,ref_grp_bel = 0.6,0.4
            oth_ag_util = ref_grp_bel*i
            eb_util = i if ref_grp_bel*(pos_state_b_signal_b*b_signal + pos_state_b_signal_r*r_signal)>u_bar else 0
            s_util = u_bar
            
            #act_sig_b = 1 if (pos_state_b_signal_b*eb_util > s_util) and (i*eb_util <= s_util) else 0
            #act_sig_r = 1 if (pos_state_b_signal_r*eb_util > s_util) and (i*eb_util <= s_util) else 0
            act_sig_b = 1 if pos_state_b_signal_b > i and eb_util>s_util else 0
            act_sig_r = 1 if pos_state_b_signal_r > i and eb_util>s_util else 0
            
            
            
            exp_act_b = act_sig_b*b_signal + act_sig_r*r_signal
            exp_act_b_clip = 1 if exp_act_b > i else 0
            agent_gain =  (pos_state_b_signal_b*i*b_signal + pos_state_b_signal_r*i*r_signal) - i*i
            '''
            u_bar = 0.2
            r_signal_ll = pos_state_r_signal_r*np.clip(1-(u_bar/pos_state_r_signal_r),0,0.5) + pos_state_b_signal_r*np.clip(1-(u_bar/(1-pos_state_b_signal_r)),0,0.5)
            r_signal_lh = pos_state_r_signal_r*np.clip(u_bar/(1-pos_state_r_signal_r),0.5,1)  + pos_state_b_signal_r*np.clip(u_bar/pos_state_b_signal_r,0.5,1) 
            b_signal_ll = pos_state_r_signal_b*np.clip(1-(u_bar/pos_state_r_signal_b),0,0.5) + pos_state_b_signal_b*np.clip(1-(u_bar/(1-pos_state_b_signal_b)),0,0.5)
            b_signal_lh = pos_state_r_signal_b*np.clip(u_bar/(1-pos_state_r_signal_b),0.5,1)  + pos_state_b_signal_b*np.clip(u_bar/pos_state_b_signal_b,0.5,1)
            exp_ll,exp_lh = r_signal*r_signal_ll + b_signal*b_signal_ll, r_signal*r_signal_lh + b_signal*b_signal_lh
            act_sig_b_norm = 1 if b_signal_lh < i else 0
            act_sig_r_norm = 1 if r_signal_lh < i else 0
            exp_act_b_norm = act_sig_b_norm*b_signal + act_sig_r_norm*r_signal
            '''
            if i>0.5:
                all_samples_res.append(exp_act_b_clip)
            baseline_margin.append((rec_prior_b,i))
            moved_margin.append((i,agent_gain))
            #print(i,exp_act_b,pos_state_b_signal_b,pos_state_b_signal_r)
            #moved_margin_norm.append((pb,exp_act_b_norm))
            
        #plt.plot([x[0] for x in baseline_margin],[x[1] for x in baseline_margin],'blue')
        #plt.figure()
        #plt.plot([x[0] for x in moved_margin],[x[1] for x in moved_margin],'black')
        #plt.plot([x[0] for x in moved_margin_norm],[x[1] for x in moved_margin_norm],'pink')
        #plt.title('-'.join([str(rr),str(rb)]))
        #plt.show()
        self.all_samples_res = all_samples_res
        self.sender_val = np.sum([x for x in self.all_samples_res])
        
    def single_sender_multiple_receiver_normative(self,rr,rb,br,bb,sender_prior):
        ''' generate all samples'''
        if hasattr(self, 'bimodal_samples'):
            bimodal_samples = self.bimodal_samples
        else:
            bimodal_samples = generate_bimodal_samples(mu=[0.4,0.6])
            self.bimodal_samples = bimodal_samples
        #pr,pb = (0.4,0.6)
        all_samples_res = []
        baseline_margin, moved_margin, moved_margin_norm = [],[],[]
        bimodal_samples.sort()
        u_bar = 0.2
        rep_receiver_prior = sender_prior
        rep_receiver_prior_r,rep_receiver_prior_b = rep_receiver_prior
        pr,pb = rep_receiver_prior
        r_signal = (rr/(rr+br))*pr + (rb/(rb+bb))*pb
        b_signal = (br/(rr+br))*pr + (bb/(rb+bb))*pb
        repr_pos_r_r_signal = (rr*pr)/(rr*pr + rb*pb)
        repr_pos_r_b_signal = (br*pr)/(br*pr + bb*pb)
        repr_pos_b_b_signal = (bb*pb)/(bb*pb + br*pr)
        repr_pos_b_r_signal = (rb*pb)/(rb*pb + rr*pr)
        expected_posterior_r = r_signal*repr_pos_r_r_signal + b_signal*repr_pos_r_b_signal
        expected_posterior_b = r_signal*repr_pos_b_b_signal + b_signal*repr_pos_b_r_signal
        
        guilt_val = None
        for i in bimodal_samples:
            rec_prior_r,rec_prior_b = (1-i,i)
            pos_state_r_signal_r = self.calc_pos_relative_to_representative(repr_pos_r_r_signal,rep_receiver_prior_r,rec_prior_r)
            pos_state_r_signal_b = self.calc_pos_relative_to_representative(repr_pos_r_b_signal,rep_receiver_prior_r,rec_prior_r)
            pos_state_b_signal_r = self.calc_pos_relative_to_representative(repr_pos_b_r_signal,rep_receiver_prior_b,rec_prior_b)
            pos_state_b_signal_b = self.calc_pos_relative_to_representative(repr_pos_b_b_signal,rep_receiver_prior_b,rec_prior_b)
            
            if i>0.5:
                oth_group_bel,ref_grp_bel = 0.1,0.9
            else:
                oth_group_bel,ref_grp_bel = 0.9,0.1
            eb_util = i if ref_grp_bel*(pos_state_b_signal_b*b_signal + pos_state_b_signal_r*r_signal)>u_bar>u_bar else 0
            s_util = u_bar-max(0,ref_grp_bel-i) if ref_grp_bel > oth_group_bel else oth_group_bel
            
            act_sig_b = 1 if pos_state_b_signal_b > i and eb_util>s_util else 0
            act_sig_r = 1 if pos_state_b_signal_r > i and eb_util>s_util else 0
            
            
            
            exp_act_b = act_sig_b*b_signal + act_sig_r*r_signal
            exp_act_b_normative = 1 if exp_act_b > i else 0
            
              
            
            
            '''
            u_bar = 0.2
            r_signal_ll = pos_state_r_signal_r*np.clip(1-(u_bar/pos_state_r_signal_r),0,0.5) + pos_state_b_signal_r*np.clip(1-(u_bar/(1-pos_state_b_signal_r)),0,0.5)
            r_signal_lh = pos_state_r_signal_r*np.clip(u_bar/(1-pos_state_r_signal_r),0.5,1)  + pos_state_b_signal_r*np.clip(u_bar/pos_state_b_signal_r,0.5,1) 
            b_signal_ll = pos_state_r_signal_b*np.clip(1-(u_bar/pos_state_r_signal_b),0,0.5) + pos_state_b_signal_b*np.clip(1-(u_bar/(1-pos_state_b_signal_b)),0,0.5)
            b_signal_lh = pos_state_r_signal_b*np.clip(u_bar/(1-pos_state_r_signal_b),0.5,1)  + pos_state_b_signal_b*np.clip(u_bar/pos_state_b_signal_b,0.5,1)
            exp_ll,exp_lh = r_signal*r_signal_ll + b_signal*b_signal_ll, r_signal*r_signal_lh + b_signal*b_signal_lh
            act_sig_b_norm = 1 if b_signal_lh < i else 0
            act_sig_r_norm = 1 if r_signal_lh < i else 0
            exp_act_b_norm = act_sig_b_norm*b_signal + act_sig_r_norm*r_signal
            '''
            if i>0.5:
                all_samples_res.append(exp_act_b_normative)
            baseline_margin.append((rec_prior_b,i))
            #moved_margin.append((i,exp_act_b))
            #moved_margin_norm.append((pb,exp_act_b_norm))
            
        #plt.plot([x[0] for x in baseline_margin],[x[1] for x in baseline_margin],'blue')
        #plt.plot([x[0] for x in moved_margin],[x[1] for x in moved_margin],'red')
        #plt.plot([x[0] for x in moved_margin_norm],[x[1] for x in moved_margin_norm],'pink')
        #plt.title('-'.join([str(rr),str(rb)]))
        #plt.show()
        self.all_samples_res = all_samples_res
        self.sender_val = np.sum([x for x in self.all_samples_res])
        
        

def run_polyani_baseline():        
    models = InfoDesign()
    lst = []
    '''
    cols = ['rr','rb','orig_diff', 'changed_diff','orig_vals_l','orig_vals_h','changed_vals_l','changed_vals_h','exp_util_b','exp_util_r','util_gain']
    
    
    for rr in np.linspace(0.5,1,100):
        for rb in np.linspace(0.01,0.5,100):
            bb = 1 - rb
            br = 1 - rr
            models.single_sender_single_informed_receiver(rr,rb,br,bb)
            exp_ll,exp_lh,exp_util_b,exp_util_r = models.exp_ll,models.exp_lh,models.exp_util_b,models.exp_util_r
            l_l = np.clip(1-(u_bar/(1-theta)),0,0.5)
            l_h = np.clip(u_bar/theta,0.5,1) 
            print([abs(l_l-l_h),abs(exp_ll-exp_lh),l_l,l_h,exp_ll,exp_lh,exp_util_b,exp_util_r,exp_lh-exp_util_b])
            lst.append([rr,rb,abs(l_l-l_h),abs(exp_ll-exp_lh),l_l,l_h,exp_ll,exp_lh,exp_util_b,exp_util_r,exp_lh-exp_util_b])
    df = pd.DataFrame(lst, columns=cols)
    print(df.loc[df['changed_vals_l'].idxmin()])       
    print(df.loc[df['changed_vals_h'].idxmax()])
    print(df.loc[df['util_gain'].idxmax()])
    '''
    cols = ['rr','rb','target_rec','exp_util_b']
    for rr in np.linspace(0.5,1,10):
        for rb in np.linspace(0.01,0.5,10):
            data_res = []
            for select_rec in np.linspace(0.5,0.99,100)[::-1]:
                sender_prior = (1-select_rec,select_rec)
                bb = 1 - rb
                br = 1 - rr
                models.single_sender_multiple_receiver_nonnormative_nonmanipulative(rr,rb,br,bb,sender_prior)
                b_utils = np.sum([x for x in models.all_samples_res])
                lst.append([rr,rb,select_rec,models.sender_val])
                print([rr,rb,select_rec,models.sender_val])
                data_res.append((select_rec,models.sender_val))
            plt.plot([x[0] for x in data_res], [x[1] for x in data_res])
            plt.title('-'.join([str(rr),str(rb)]))
            plt.show()
    df = pd.DataFrame(lst, columns=cols)
    print(df.loc[df['exp_util_b'].idxmax()])
        
def run_normative():       
    models = InfoDesign()
    lst = []
    
    cols = ['rr','rb','orig_diff', 'changed_diff','orig_vals_l','orig_vals_h','changed_vals_l','changed_vals_h','exp_util_b','exp_util_r','util_gain']
    
    '''
    for rr in np.linspace(0.5,1,100):
        for rb in np.linspace(0.01,0.5,100):
            bb = 1 - rb
            br = 1 - rr
            models.single_sender_single_informed_receiver(rr,rb,br,bb)
            exp_ll,exp_lh,exp_util_b,exp_util_r = models.exp_ll,models.exp_lh,models.exp_util_b,models.exp_util_r
            l_l = np.clip(1-(u_bar/(1-theta)),0,0.5)
            l_h = np.clip(u_bar/theta,0.5,1) 
            print([abs(l_l-l_h),abs(exp_ll-exp_lh),l_l,l_h,exp_ll,exp_lh,exp_util_b,exp_util_r,exp_lh-exp_util_b])
            lst.append([rr,rb,abs(l_l-l_h),abs(exp_ll-exp_lh),l_l,l_h,exp_ll,exp_lh,exp_util_b,exp_util_r,exp_lh-exp_util_b])
    df = pd.DataFrame(lst, columns=cols)
    print(df.loc[df['changed_vals_l'].idxmin()])       
    print(df.loc[df['changed_vals_h'].idxmax()])
    print(df.loc[df['util_gain'].idxmax()])
    '''
    cols = ['rr','rb','target_rec','exp_util_b']
    for rr in np.linspace(0.5,1,10):
        for rb in np.linspace(0.01,0.5,10):
            data_res = []
            for select_rec in np.linspace(0.5,0.99,100)[::-1]:
                sender_prior = (1-select_rec,select_rec)
                bb = 1 - rb
                br = 1 - rr
                models.single_sender_multiple_receiver_normative(rr,rb,br,bb,sender_prior)
                b_utils = np.sum([x for x in models.all_samples_res])
                lst.append([rr,rb,select_rec,models.sender_val])
                print([rr,rb,select_rec,models.sender_val])
                data_res.append((select_rec,models.sender_val))
            plt.plot([x[0] for x in data_res], [x[1] for x in data_res],'red')
            plt.title('-'.join([str(rr),str(rb)]))
            
            data_res_base = []
            for select_rec in np.linspace(0.5,0.99,100)[::-1]:
                sender_prior = (1-select_rec,select_rec)
                bb = 1 - rb
                br = 1 - rr
                models.single_sender_multiple_receiver_nonnormative_nonmanipulative(rr,rb,br,bb,sender_prior)
                b_utils = np.sum([x for x in models.all_samples_res])
                lst.append([rr,rb,select_rec,models.sender_val])
                print([rr,rb,select_rec,models.sender_val])
                data_res_base.append((select_rec,models.sender_val))
            plt.plot([x[0] for x in data_res_base], [x[1] for x in data_res_base],'blue')
            plt.title('-'.join([str(rr),str(rb)]))
            plt.show()
    df = pd.DataFrame(lst, columns=cols)
    print(df.loc[df['exp_util_b'].idxmax()])
    
#run_normative()
        
        