# Training
import pommerman
import torch

from pommerman import agents

from fc_agent import FCAgent    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
EPISODES = 100


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
    memory_stop = False
    while not done:
        #env.render()
        actions = env.act(state); 
        if fc_agent.is_alive: actions[0] = actions[0].item()
        state, reward, done, info = env.step(actions)

        fca_reward = torch.tensor([float(reward[0])], device=device)
        fca_action = torch.tensor(actions[0], device=device)
        # Observe new state
        last_x = current_x
        current_x = fc_agent.utils.input(state[0])

        # Store the transition in memory
        # Game over
        if done or (not fc_agent.is_alive and not memory_stop): 
            fc_agent.memory.push(last_x, fca_action, None, fca_reward)
            memory_stop = True
        # Game on
        else:
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
