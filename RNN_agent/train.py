# Training
import pommerman
import torch

from pommerman import agents

from rnn_agent import RNN_Agent   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
EPISODES = 1

rnn_agent = RNN_Agent()
agent_list = [rnn_agent, agents.SimpleAgent(), agents.RandomAgent(), agents.SimpleAgent()]
env = pommerman.make('PommeFFACompetition-v0', agent_list)
    
wins = {}; iter_num = 0 
for an_episode in range(EPISODES):
    state = env.reset()
         
    #-------------------------------------------------------------------
    done  = False
    while not done:
        #env.render()
        actions = env.act(state); 
        state, reward, done, info = env.step(actions)

        iter_num += 1
    #-------------------------------------------------------------------

    env.close()
    print(info)
    if 'winners' in info:
        wins[info['winners'][0]] = wins.get(info['winners'][0], 0) + 1 
    print(wins)
