# Training
import pommerman

import numpy as np
from random import shuffle

import torch
import tensorflow as tf

from pommerman import agents

from rnn_agent import RNN_Agent   
from pomm_dataset import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_data(EPISODES, save_file_nm, shuffle_agents=False):
    rnn_agent = RNN_Agent()
    
    # Init dataset
    dset = dataset(rnn_agent.RNN_SEQUENCE_LENGTH, save_file_nm, rnn_agent.utils)

    agent_list = [rnn_agent, agents.SimpleAgent(), agents.RandomAgent(), agents.SimpleAgent()]
    if shuffle_agents: shuffle(agent_list)
    rnn_agent_index = agent_list.index(rnn_agent)

    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    rnn_agent.sess.run(tf.global_variables_initializer())

    wins = {}; iter_num = 0 
    for an_episode in range(EPISODES):
        state = env.reset()
             
        #-------------------------------------------------------------------
        done  = False; episode_obs = []; episode_acts = []
        #while not done and rnn_agent.is_alive:
        while not done:
            #env.render()
            actions = env.act(state)
            episode_acts.append(actions[rnn_agent_index])
            episode_obs.append(rnn_agent.utils.input(state[rnn_agent_index]))
            state, reward, done, info = env.step(actions)
            
            iter_num += 1
        #-------------------------------------------------------------------
        
        # Final timestep observation
        episode_obs.append(rnn_agent.utils.input(state[rnn_agent_index]))
        dset.add_episode(episode_obs, episode_acts)
        
        env.close()
        print(info)
        if 'winners' in info:
            wins[info['winners'][0]] = wins.get(info['winners'][0], 0) + 1 
        print(wins)
        print("Median Act Time: {} seconds".format(np.median(np.array(rnn_agent.act_times))))
    
    dset.save()
    rnn_agent.sess.close()
    tf.reset_default_graph()

def train_M(epochs, save_file_nm):
    rnn_agent = RNN_Agent()
    dset = dataset(rnn_agent.RNN_SEQUENCE_LENGTH, save_file_nm, rnn_agent.utils)
    dset.load()
    rnn_agent.sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        dset.sampler(rnn_agent.RNN_BATCH_SIZE)

    rnn_agent.sess.close()
    tf.reset_default_graph()

if __name__ == '__main__':
    lvl1 = "dataset_lvl1.pickle" 
    print('-'*150); print('*'*90); print("Generating dataset ", lvl1); print('*'*90);
    generate_data(10, lvl1) 
    print('-'*150); print('*'*90); print("Training M (RNN) on dataset ", lvl1); print('*'*90);  
    train_M(1, lvl1)

