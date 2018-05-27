# Training
import pommerman
import torch

from pommerman import agents

from fc_agent import FCAgent    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
EPISODES = 1

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
    done  = False
    while not done:
        #env.render()
        actions = env.act(state); 
        state, reward, done, info = env.step(actions)
    #-------------------------------------------------------------------
        
    #for agent in agent_list:
    #    agent.episode_end(reward[agent.agent_id], obs[agent.agent_id])

    env.close()
    print(info)
    if 'winners' in info:
        wins[info['winners'][0]] = wins.get(info['winners'][0], 0) + 1 
    print(wins)

