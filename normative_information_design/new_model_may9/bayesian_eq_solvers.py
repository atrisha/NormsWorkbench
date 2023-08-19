'''
Created on 14 May 2023

@author: Atrisha
'''
import itertools

class BayesianNashEquilibrium:
    def __init__(self):
        self._num_agents = None
        self._action_dict = None
        self._type_dict = None
        self._type_space = None

    @property
    def num_agents(self):
        return self._num_agents

    @num_agents.setter
    def num_agents(self, value):
        self._num_agents = value

    @property
    def action_dict(self):
        return self._action_dict

    @action_dict.setter
    def action_dict(self, value):
        self._action_dict = value

    @property
    def type_dict(self):
        return self._type_dict

    @type_dict.setter
    def type_dict(self, value):
        self._type_dict = value

    @property
    def type_space(self):
        return self._type_space

    @type_space.setter
    def type_space(self, value):
        self._type_space = value

theta_s = 0.7
agents = ["R1", "R2"]
num_agents = len(agents)
common_prior = theta_s
payoff_dict = list(itertools.product(['pr','pb','s'], ['pr','pb','s']))
def assign_payoffs(strat_tuple, r1_type, r2_type):
    if strat_tuple[0][-1]=='r' and strat_tuple[1][-1]=='r':
        
    