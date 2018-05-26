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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class _utils:
    def __init__(self, board_h, board_w, save_file):
        self.save_file = save_file
        self.num_actions = 6
        self.board_area = board_h * board_w 

        self.int2vec = {
            1  : np.zeros((self.board_area,)),
            2  : np.zeros((self.board_area,)),
            4  : np.zeros((self.board_area,)),
            6  : np.zeros((self.board_area,)),
            7  : np.zeros((self.board_area,)),
            8  : np.zeros((self.board_area,)),
            10 : np.zeros((self.board_area,)),
            11 : np.zeros((self.board_area,)),
            12 : np.zeros((self.board_area,)),
            13 : np.zeros((self.board_area,))
        }
        self.blast_strength_vec = np.zeros((max(board_h, board_w)+1,))

        self.max_ammo = 4
        self.ammo = np.zeros((self.max_ammo,))

        self.this_agent = np.zeros((5,))
        self.friend = np.zeros((5,))
        self.enemy1 = np.zeros((5,))
        self.enemy2 = np.zeros((5,))
        self.enemy3 = np.zeros((5,))

        # Different symbolic objects
        self.input_size = self.board_area*len(self.int2vec) + \
            max(board_h, board_w)+1 + \
            self.max_ammo + \
            5*5 + \
            self.board_area + \
            self.board_area
        # Action and reward
        #self.input_size += (6 + 1)


   
    def input(self, obs):
        blast_strength = int(obs['blast_strength'])
        ammo        = int(obs['ammo'])
        my_position = tuple(obs['position'])
        teammate    = int(obs['teammate'].value) - 9
        enemies     = np.array([e.value for e in obs['enemies']]) - 9
        board       = np.array(obs['board'])
        bombs       = np.array(obs['bomb_blast_strength'])/2.0
        bombs_life  = np.array(obs['bomb_life'])/9.0
        
        # Symbolic objects to vector of boards
        for idx, cell in enumerate(board.flatten().tolist()):
            if cell in self.int2vec:
                self.int2vec[cell][idx] = 1.0
        
        # !TODO Test this assumption
        self.blast_strength_vec[blast_strength] = 1.0

        # If ammo > 10, ammo = 10 (as one hot)
        self.ammo[min(self.max_ammo,ammo)-1] = 1.0

        agent_ids = [0,1,2,3,4]
        # Agents
        for an_enemy_id, an_enemy_vec in zip(enemies, [self.enemy1, self.enemy2, self.enemy3]):
            an_enemy_vec[an_enemy_id] = 1.0
            agent_ids.remove(an_enemy_id)
        self.friend[teammate] = 1.0 
        agent_ids.remove(teammate)
        # DEBUG
        if len(agent_ids) != 1: raise ValueError('Error! agent_ids has more/less than one id left!')
        # DEBUG
        self.this_agent[agent_ids[0]] = 1.0


        # !TODO Concatenate all the vectors 
        input_data = np.array([])
        for idx in self.int2vec:
            input_data = np.concatenate((input_data, self.int2vec[idx]))

        input_data = np.concatenate((input_data, self.blast_strength_vec))
        input_data = np.concatenate((input_data, self.ammo))
        input_data = np.concatenate((input_data, self.this_agent))
        input_data = np.concatenate((input_data, self.friend))
        input_data = np.concatenate((input_data, self.enemy1))
        input_data = np.concatenate((input_data, self.enemy2))
        input_data = np.concatenate((input_data, self.enemy3))
        input_data = np.concatenate((input_data, bombs.flatten()))
        input_data = np.concatenate((input_data, bombs_life.flatten()))
       
        #print("Data vector: {} v.s. input_size: {}".format(input_data.shape, self.input_size))

        return torch.Tensor(input_data.flatten(), device=device)

    def action_onehot(self, action):
        action_vec = [0]*self.num_actions
        action_vec[action] = 1
        return torch.tensor(action_vec, device=device, dtype=torch.long)
        
    def save(self, model): 
        torch.save(model, self.save_file)


        
class _ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200
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

        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).to(device)
        #print("non_final_next_states", non_final_next_states.shape)
        state_batch  = torch.stack(batch.state).to(device)
        #action_batch = torch.stack(batch.action).to(device)
        reward_batch = torch.stack(batch.reward).to(device) 

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        #print("state_batch shape: ", state_batch.shape)
        #state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        state_action_values = self.policy_net(state_batch)

        next_state_values = self.target_net(non_final_next_states).detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        #print("State action: ", state_action_values.shape)
        #print("Ex. State action: ", expected_state_action_values.shape)
        #print("Ex. State action (unsq): ", expected_state_action_values.unsqueeze(0).shape)
        #print('-'*90)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

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
        # Initialize the environment ae
        action = self._select_action(x_torch)
        return action.cpu().numpy()[0][0]
           

    def episode_end(self, reward): 
        pass

if __name__ == '__main__':
    # Training
    import pommerman
    from pommerman import agents
    
    # Hyperparams
    EPISODES = 300

    fc_agent = FCAgent()
    agent_list = [fc_agent, agents.SimpleAgent(), agents.RandomAgent(), agents.SimpleAgent()]
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    
    wins = {}; iter_num = 0
    target_update = 10
    for an_episode in range(EPISODES):
        state = env.reset()
        
        current_x = fc_agent.utils.input(state[0])
        last_x    = fc_agent.utils.input(state[0])
         
        #-------------------------------------------------------------------
        done = False
        while not done:
            #env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

            fca_reward = torch.tensor([float(reward[0])], device=device)
            fca_action = fc_agent.utils.action_onehot(actions[0])
            # Observe new state
            last_x = current_x
            current_x = fc_agent.utils.input(state[0])

            # Store the transition in memory
            fc_agent.memory.push(last_x, fca_action, current_x, fca_reward)
            
            # Perform one step of the optimization (on the target network)
            fc_agent._train()
            iter_num += 1
        #-------------------------------------------------------------------
        
        #for agent in agent_list:
        #    agent.episode_end(reward[agent.agent_id], obs[agent.agent_id])

        env.close()
        print(info)
        if 'winners' in info:
            wins[info['winners'][0]] = wins.get(info['winners'][0], 0) + 1
        print(wins)

        # Update the target network
        if an_episode % target_update == 0:
            fc_agent.target_net.load_state_dict(fc_agent.policy_net.state_dict())

        fc_agent.utils.save({
            "target_net": fc_agent.target_net.state_dict(),
            "policy_net": fc_agent.policy_net.state_dict(),
            "iter" : iter_num
        })

