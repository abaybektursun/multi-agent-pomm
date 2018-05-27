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
        self.prev_x_np = None

        # Network -----------------------------------------------------------------------
        N, D_in, H1, H2, D_out = 1, self.input_size, 128, 64, 6

        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H1),
            torch.nn.ReLU(),
            torch.nn.Linear(H1, H2),
            torch.nn.ReLU(),
            torch.nn.Linear(H2, D_out),
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #--------------------------------------------------------------------------------
        
        self.step_num = 0
        self.policy_net = self.model.to(device)
        self.target_net = self.model.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # Hyper Params ------------------------------------------------------------------
        self.BATCH_SIZE = 128
        self.GAMMA      = 0.999
        self.EPS_START  = 0.9
        self.EPS_END    = 0.05
        self.EPS_DECAY  = 20_0
        self.TARGET_UPDATE = 10
        #--------------------------------------------------------------------------------

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = _ReplayMemory(10000)

        self.episode_durations = []

        if os.path.isfile(self.utils.save_file):
            checkpoint = torch.load(self.utils.save_file)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            print("=> loaded checkpoint '{}'".format(checkpoint['iter']))

    #def optimize_model():
    def _train(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # This mask indicates whether vector at corresponding index is final or not
        non_final_mask = torch.tensor(
                tuple(map(
                        lambda s: s is not None,
                        batch.next_state
                    )), device=device, dtype=torch.uint8)

        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).to(device)
        state_batch  = torch.stack(batch.state).to(device)
        action_batch = torch.stack(batch.action).to(device)
        reward_batch = torch.stack(batch.reward).to(device) 

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.view(-1, 1))

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).detach().max(1)[0]
        # Compute the expected Q values
        reward_batch = reward_batch.view(-1)
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def _select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.step_num / self.EPS_DECAY)
        self.step_num += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state.to(device)).max(0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(6)]], device=device, dtype=torch.long)
    
    
    def act(self, obs, action_space):
        x_torch = self.utils.input(obs) 
        action = self._select_action(x_torch)
        return action
 
