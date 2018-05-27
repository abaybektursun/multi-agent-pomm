# Agent based on regular FC network with replay

from pommerman.agents import BaseAgent
from pommerman        import constants
from pommerman        import utility

import os
import math
import random

import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple
from itertools   import count

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.autograd import Variable

from utils import _utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class _ReplayMemory(object):
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


class FCAgent(BaseAgent):
    def __init__(self, board_h=11, board_w=11, *args, **kwargs):
        self.name = 'FC Agent'
        super(FCAgent, self).__init__(*args, **kwargs)
        # Common functionalities among learning agents
        self.utils = _utils(board_h, board_w, 'fc_agent/save.tar')
        self.input_size = self.utils.input_size

        # Network -----------------------------------------------------------------------
        N, D_in, H1, H2, D_out = 1, self.input_size, 128, 64, 6

        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H1),
            torch.nn.ReLU(),
            torch.nn.Linear(H1, H2),
            torch.nn.ReLU(),
            torch.nn.Linear(H2, D_out),
        )
        # Hyper Params ------------------------------------------------------------------
        # Hyper Parameters
        self.BATCH_SIZE = 32
        self.LR = 0.01                   # learning rate
        self.EPSILON = 0.9               # greedy policy
        self.GAMMA = 0.9                 # reward discount
        self.TARGET_REPLACE_ITER = 100   # target update frequency
        self.MEMORY_CAPACITY = 2000
        self.N_ACTIONS = 6
        self.N_STATES = self.input_size
        #--------------------------------------------------------------------------------

        self.eval_net, self.target_net = self.model.to(device), self.model.to(device)
        self.learn_step_counter = 0                                               # for target updating
        self.memory_counter = 0                                                   # for storing memory
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

        if os.path.isfile(self.utils.save_file):
            checkpoint = torch.load(self.utils.save_file)
            self.eval_net.load_state_dict(checkpoint['eval_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            print("=> loaded checkpoint '{}'".format(checkpoint['iter']))

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1 

    def choose_action(self, x):
        # input only one sample
        if np.random.uniform() < self.EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0, 0]     # return the argmax
        else:   # random
            action = np.random.randint(0, self.N_ACTIONS)
        return action
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def act(self, obs, action_space):
        x_torch = self.utils.input(obs) 
        action = self.choose_action(x_torch)
        return action
 
