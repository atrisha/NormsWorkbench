'''
Created on 19 Jan 2023

@author: Atrisha
'''
from pettingzoo_sandbox.all_networks import *
from pettingzoo_sandbox.normative_information_design_single_context.norm_recommendation_information_design import StewardAgent, parallel_env
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_target_expected_reward(model,state):
    rewards = []
    actions = torch.rand(size=(1000,1),dtype=torch.float32, device=device)
    state_repeat = state.repeat(1000,1)
    input_as_batch = torch.cat((state_repeat,actions),axis=1).unsqueeze(0)
    out = model.forward(input_as_batch)
    max_reward = torch.max(out)
    act_index = torch.argmax(out) 
    argmax_act = actions[act_index]
    return max_reward, argmax_act
    
use_cuda = torch.cuda.is_available()
env = parallel_env(render_mode='human',attr_dict={'true_state':{'n1':0.6}})
''' Check that every norm context has at least one agent '''
if not all([True if [_ag.norm_context for _ag in env.possible_agents].count(n) > 0 else False for n in env.norm_context_list]):
    raise Exception()
env.reset()
env.common_prior = (4, 2)
env.prior_baseline = env.common_prior
env.normal_constr_sd = 0.1

number_of_iterations = 20000
env.NUM_ITERS = number_of_iterations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qn = QNetwork(input_state_size=3)
agent = StewardAgent(qnetwork=qn)
optimizer = torch.optim.Adam(agent.qnetwork.parameters(), lr=0.5)
if use_cuda:
    agent.qnetwork.to(device)

GAMMA = 0.9
epsilon = 0.1
lossFunc = torch.nn.MSELoss()
loss_tracker,reward_tracker,loss_plot = [],[],[]
rewards_plot = []
print('Using device:', device)
print()
optimizer.zero_grad()
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
for iteration in np.arange(number_of_iterations): 
    #(3, 3) 0.6418616401671191 0.9196206661347156 0.5 -0.8
    ''' Sample a state '''
    if iteration%100 == 0:
        env.common_prior = (np.random.randint(low=1,high=10),np.random.randint(low=1,high=10))
    #state = torch.FloatTensor([env.common_prior[0]/sum(env.common_prior), beta(a=env.common_prior[0], b=env.common_prior[1]).var()])
    state = torch.tensor([env.common_prior[0]/sum(env.common_prior), beta(a=env.common_prior[0], b=env.common_prior[1]).var()], dtype=torch.float32, device=device).unsqueeze(0)
    ''' Sample an action '''
    if iteration == 0:
        action = np.random.random_sample()
    posterior_mean = env.generate_posteriors(action)
    population_actions = {agent.id:agent.act(env,run_type='self-ref',baseline=False) for agent in env.possible_agents}
    observations, reward, terminations, truncations, infos = env.step(population_actions,0,baseline=False)
    action = torch.tensor([[action]], device=device, dtype=torch.float)
    #state_ = torch.FloatTensor([env.common_prior[0]/sum(env.common_prior), beta(a=env.common_prior[0], b=env.common_prior[1]).var()])
    state_ = torch.tensor([env.common_prior[0]/sum(env.common_prior), beta(a=env.common_prior[0], b=env.common_prior[1]).var()], dtype=torch.float32, device=device).unsqueeze(0)
    input_tensor = torch.cat((state,action),axis=1)
    current_reward = agent.qnetwork.forward(input_tensor)
    target_reward = torch.add(torch.mul(get_target_expected_reward(agent.qnetwork,state_)[0], GAMMA), reward)
    loss = lossFunc(target_reward,current_reward)
    loss.backward()
    optimizer.step()
    loss_tracker.append(loss.item())
    reward_tracker.append([target_reward.item(),current_reward.item()])
    
    if np.random.random_sample() < epsilon:
        action = np.random.random_sample()
    else:
        action = get_target_expected_reward(agent.qnetwork,state_)[1].item()
    if iteration % 100 == 0:
        mean_loss=np.mean(loss_tracker)
        rewards_plot.append(mean_loss)
        target_mean_loss = np.mean([x[0] for x in reward_tracker])
        current_mean_loss = np.mean([x[1] for x in reward_tracker])
        print('iteration:',iteration,'mean loss=',np.mean(loss_tracker),'target=',target_mean_loss,'current=',current_mean_loss)
        #loss_plot.append([iteration,abs(target_mean_loss-current_mean_loss)])
        loss_plot.append([iteration,mean_loss])
        loss_tracker = []

PATH = '../agent_qnetwork.model'
torch.save(agent.qnetwork.state_dict(), PATH)

cols = ['instance', 'mean loss']
df = pd.DataFrame(loss_plot[1:], columns=cols)  
fig = plt.figure()
ax = sns.lineplot(x="instance", y="mean loss", data=df)

print("Model's state_dict:")
for param_tensor in agent.qnetwork.state_dict():
    print(param_tensor, "\t", agent.qnetwork.state_dict()[param_tensor].size())


opt_plot = []
for s in np.linspace(0,1,100):
    opt_act = []
    for v in np.linspace(0,.1,10):
        _state = torch.tensor([s,v], dtype=torch.float32, device=device).unsqueeze(0)
        max_reward, argmax_act = get_target_expected_reward(agent.qnetwork,_state)
        opt_act.append(argmax_act.item())
    opt_act = np.mean(opt_act)
    opt_plot.append([s,opt_act])

cols = ['state', 'action']
df = pd.DataFrame(opt_plot, columns=cols)  
#fig = plt.figure()
#ax = sns.lineplot(x="state", y="action", ci="sd", estimator='mean', data=df)
sns.lmplot(x="state", y="action", data=df,
           order=4, ci=None);
plt.show()       






    
    
    