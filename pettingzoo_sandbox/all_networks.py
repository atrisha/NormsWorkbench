'''
Created on 18 Jan 2023

@author: Atrisha
'''
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    @property
    def reached_capacity(self):
        self._reached_capacity = True if len(self.memory) == self.capacity else False
        return self._reached_capacity


class QNetwork(nn.Module):
    def __init__(self,input_state_size):
        super(QNetwork,self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_state_size, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )
        
        self.network.apply(init_weights)
    
    def forward(self,x):
        o = self.network(x)
        return o
    
    def printInfo(self):
        print("Weight of linear : \n     ", self.network[0].weight)
        print("Bias of linear : \n     ", self.network[0].bias)
        print("Grad of weight : \n     ", self.network[0].weight.grad)
        print("Grad of bias : \n     ", self.network[0].bias.grad)
        
'''
ag = Agent()
x = torch.tensor([[.7,.3]])
o = ag.forward(x)
print('output',o)
ag.printInfo()
lossFunc = torch.nn.L1Loss()
t = torch.tensor([[.5]])
loss = lossFunc(o,t)
#print('loss',loss)
loss.backward()
optimizer = torch.optim.SGD(ag.parameters(), lr=0.5)
optimizer.step()
ag.printInfo()
print('output',ag.forward(x))
'''