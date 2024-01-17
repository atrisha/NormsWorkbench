'''
Created on 10 Mar 2023

@author: Atrisha
'''

from normative_information_design.normative_information_design_perceptiongap.perception_gap_information_design import parallel_env, Institution
import numpy as np
import mdptoolbox, mdptoolbox.example
import utils
import math
import matplotlib.pyplot as plt


def generate_transition_matrix():
    reward_matrix =  np.zeros(shape=(9,9))
    transition_map = dict()
    for action in [round(x,1) for x in np.arange(0.1,0.51,.1)]:
        print('\n')
        if action not in transition_map:
            transition_map[action] = dict()
        actlist_t, actlist_r = [],[]
        for run_iter in np.arange(5):
            common_prior_appr_input = (4,2)
            common_prior_appr = (4,2)
            common_prior_disappr = (2,4)
            common_proportion_prior = (5,5)
            institution = Institution('intensive')
            if action >= 0.5:
                institution.constant_disappr_signal = 0.4
            else:
                institution.constant_appr_signal = 0.6
            env = parallel_env(render_mode='human',attr_dict={'true_state':{'n1':0.55},'extensive':False,
                                                              'common_prior_appr' : common_prior_appr,
                                                            'common_prior_disappr' : common_prior_disappr,
                                                            'common_proportion_prior' : common_proportion_prior,
                                                            'common_prior_appr_input':common_prior_appr_input,
                                                            'only_intensive':True})
            ''' Check that every norm context has at least one agent '''
            if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
                raise Exception()
            env.single_institution_env = True
            number_of_iterations = 50000
            env.NUM_ITERS = number_of_iterations
            for state in [round(x,1) for x in np.arange(0.1,0.51,.1)]: 
                if (state-0.5)*(action-0.5) < 0:
                    continue
                print(action,':',state,':',run_iter)
                env.reset()
                if state >= 0.5:
                    env.common_prior_appr = utils.est_beta_from_mu_sigma(state, 0.2)
                    env.common_prior_disappr = 0.4
                else:
                    env.common_prior_disappr = utils.est_beta_from_mu_sigma(state, 0.2)
                    env.common_prior_appr = 0.6
                
                env.prior_baseline = (env.common_prior_appr + env.common_prior_disappr)/2
                env.normal_constr_w = 0.3
                for ag in env.possible_agents:
                    ag.init_beliefs(env)

                if institution.type == 'intensive':
                    if action >= 0.5:
                        if abs(action-env.common_prior_appr) <= env.normal_constr_w:
                            valid_distr = True
                        else:
                            valid_distr = False
                    else:
                        valid_distr = True
                else:
                    valid_distr = True
                    if action >= 0.5:
                        if abs(action-env.common_prior_appr) > env.normal_constr_w:
                            valid_distr = False
                    else:
                        if abs(action-env.common_prior_disappr) > env.normal_constr_w:
                            valid_distr = False
                
                

                if valid_distr:
                    for agent in env.possible_agents:
                        if math.isnan(agent.common_prior_outgroup[0]/np.sum(agent.common_prior_outgroup)) or math.isnan(agent.common_prior_ingroup[0]/np.sum(agent.common_prior_ingroup)):
                            continue
                        institution.opt_signals = {'disappr': {round(x,1):(institution.constant_disappr_signal,action) if action>= 0.5 else (action,institution.constant_appr_signal) for x in [round(x,1) for x in np.arange(0,0.5,.1)]},
                                        'appr': {round(x,1):(institution.constant_disappr_signal,action) if action>= 0.5 else (action,institution.constant_appr_signal) for x in [round(x,1) for x in np.arange(0.5,1,.1)]}
                                        }   
                        outgroup_posterior_intensive, agent.common_proportion_posterior = agent.generate_posteriors(env,institution,agent.common_proportion_prior,'outgroup')
                        ingroup_posterior_intensive, agent.common_proportion_posterior = agent.generate_posteriors(env,institution,agent.common_proportion_prior,'ingroup')
                        agent.pseudo_update_posteriors = {'intensive':{'outgroup':outgroup_posterior_intensive,'ingroup':ingroup_posterior_intensive}}
                    
                    actions = {agent.id:agent.act(env,run_type={'institutions':(None,institution),'update_type':'common'},baseline=False) for agent in env.possible_agents}
                    _po = [ag.common_posterior for ag in env.possible_agents][0]
                    _f = np.mean([ag.opinion[ag.norm_context] for ag in env.possible_agents if ag.action[0]!=-1 and ag.opinion[ag.norm_context] >= 0.5])
                    _p = {c:[ag.action_code for ag in env.possible_agents].count(c) for c in [-1,0,1]}    
                    '''
                    plt.figure()
                    plt.hist([ag.opinion[ag.norm_context] for ag in env.possible_agents if ag.action[0]!=-1])
                    plt.show()
                    '''
                    ''' common prior is updated based on the action observations '''
                    observations, reward, terminations, truncations, infos = env.step(actions,run_iter,baseline=False)
                else:
                    observations, reward, terminations, truncations, infos = env.common_prior, -1, {agent.id:False for agent in env.possible_agents}, {agent.id:False for agent in env.possible_agents}, {agent.id:{} for agent in env.possible_agents}
                #print(round(action,1),round(state,1),round(observations[0]/sum(observations),1))
                #print(round(action,1),round(state,1),round(reward,1))
                next_state = round(observations[0]/sum(observations),1)
                a_idx, s_idx, s_prime_idx = int(round(action,1)*10)-1, int(round(state,1)*10)-1, int(round(next_state,1)*10)-1
                if (s_idx,s_prime_idx) not in transition_map[action]:
                    transition_map[action][(s_idx,s_prime_idx)] = 1
                else:
                    transition_map[action][(s_idx,s_prime_idx)] += 1
                reward_matrix[s_idx,a_idx] += round(reward,1)
                #print('----')
    reward_matrix = reward_matrix/100    
    transition_matrix = np.zeros(shape=(9,9,9))
    for act,s_s_data in transition_map.items():
        a_idx = int(round(act,1)*10)-1
        for s_s_prime,ct in s_s_data.items():
            transition_matrix[a_idx,s_s_prime[0],s_s_prime[1]] = ct
    for a_idx in np.arange(transition_matrix.shape[0]):
        s_data = transition_matrix[a_idx]
        s_data_sum = np.sum(s_data,axis=1)
        s_data_prob = s_data / s_data_sum[:,None]
        transition_matrix[a_idx] = s_data_prob
    return transition_matrix, reward_matrix

P, R = generate_transition_matrix()

fh = mdptoolbox.mdp.FiniteHorizon(P, R, 0.5, 100)
#fh = mdptoolbox.mdp.QLearning(P, R, 0.9)
fh.run()
#print([np.round((x+1)/10,1) for x in list(fh.policy)])
print(fh.policy[:99])
plt.imshow(R, cmap='viridis', interpolation='gaussian')
plt.colorbar()
plt.title("Heatmap of R Matrix")
plt.xlabel("Action Index")
plt.ylabel("State Index")
plt.show()