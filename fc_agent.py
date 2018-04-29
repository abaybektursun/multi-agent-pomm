# Agent based on regular Fully Connected network

from pommerman.agents import BaseAgent
from pommerman        import constants
from pommerman        import utility

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
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
    def __init__(self, board_h=13, board_w=13, *args, **kwargs):
        super(FCAgent, self).__init__(*args, **kwargs)
        board_size = board_h*board_w
        N, D_in, H1, H2, D_out = 1, board_size, 128, 64, 6
        
        # Network -----------------------------------------------------------------------
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
        self.debug = 0
    
    def act(self, obs, action_space):
        self.debug += 1
        blast_strength = int(obs['blast_strength'])
        my_position = tuple(obs['position'])
        enemies     = [constants.Item(e) for e in obs['enemies']]
        board       = np.array(obs['board'])
        bombs       = np.array(obs['bomb_blast_strength'])
        bombs_life  = np.array(obs['bomb_life'])
        ammo        = int(obs['ammo'])

        for locations = np.where(bombs > 0)
        x = board
        y_pred = self.model(x)
        loss = loss_fn(y_pred, y)

        if self.debug == 10:
            print(blast_strength)
            print(my_position)
            print(enemies)
            print(board)
            print(bombs)
            print(ammo)
            print('-'*60)
            
    def episode_end(self, reward):


        return constants.Action.Down.value

if __name__ == '__main__':
    import pommerman
    from pommerman import agents
    agent_list = [FCAgent(), agents.SimpleAgent(), agents.RandomAgent(), agents.SimpleAgent()]
    env = pommerman.make('PommeFFA-v0', agent_list)
    
    state = env.reset()
    done = False
    while not done:
        actions = env.act(state)
        state, reward, done, info = env.step(actions)
    env.close()
    print(info)
