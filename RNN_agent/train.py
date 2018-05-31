# Training
import pommerman

import gc
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from random   import shuffle
from datetime import datetime
from collections import deque

import tensorflow as tf

import easy_tf_log
from easy_tf_log import tflog
easy_tf_log.set_dir('tboard/')

from pommerman import agents

from rnn_agent import RNN_Agent   
from pomm_dataset import dataset

tf_tag = tf.saved_model.tag_constants

# Generate Data ---------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
def generate_data(EPISODES, save_file_nm, shuffle_agents=False):
    rnn_agent = RNN_Agent()
    
    # Init dataset
    dset = dataset(rnn_agent.RNN_SEQUENCE_LENGTH, save_file_nm, rnn_agent.utils)
    if os.path.exists(save_file_nm): dset.load()

    agent_list = [rnn_agent, agents.SimpleAgent(), agents.RandomAgent(), agents.SimpleAgent()]
    rnn_agent_index = agent_list.index(rnn_agent)

    if shuffle_agents: shuffle(agent_list)
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

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
        
        #print(info)
    #print("Median Act Time: {} seconds".format(np.median(np.array(rnn_agent.act_times))))
    
    env.close()
    dset.save()
    rnn_agent.sess.close()
    tf.reset_default_graph()
#------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------
def train_M(epochs, save_file_nm, chk_point_folder, load_model=None):
    if not os.path.exists(chk_point_folder):
        os.makedirs(chk_point_folder)
    # Init the agent
    rnn_agent = RNN_Agent(model_training='M')

    # For saving model
    saver = tf.train.Saver()
    
    # Try to recover previous model
    if load_model is not None: load_folder = load_model
    else: load_folder = chk_point_folder
    latest_model = tf.train.latest_checkpoint(load_folder)
    if latest_model is not None:
        saver.restore(
            rnn_agent.sess, 
            latest_model
        )
        print("Restored ", latest_model)

    # Load the saved dataset
    dset = dataset(rnn_agent.RNN_SEQUENCE_LENGTH, save_file_nm, rnn_agent.utils)
    dset.load()
   
    # TensorBoard writer
    experimentFolder = datetime.now().isoformat(timespec='minutes')
    train_writer = tf.summary.FileWriter('./tboard/train_{}_{}'.format(save_file_nm.split('.')[0], experimentFolder), rnn_agent.sess.graph)
    # Train that bad boy
    for epoch in range(epochs):
        train_losses = []
        for x_train, y_train in zip(*dset.sampler(rnn_agent.RNN_BATCH_SIZE)):
            loss, _, summary = rnn_agent.sess.run([rnn_agent.loss, rnn_agent.optimizer, rnn_agent.merged],
                feed_dict={
                    rnn_agent.batch_ph: x_train,
                    rnn_agent.target_ph: y_train
                }
            )
            train_losses.append(loss); 
            iter_num = rnn_agent.global_step.eval(session=rnn_agent.sess)
            train_writer.add_summary(summary, iter_num)
        train_losses = np.array(train_losses)
        print('mean |   ||    (train):', np.mean(train_losses), ' Epoch: ', epoch, ' Iter: ', iter_num)
        print('     ||  |_ \n')

    # Save the model

    saver.save(rnn_agent.sess, chk_point_folder, global_step=rnn_agent.global_step)
    
    # Test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # TensorBoard writer
    test_writer = tf.summary.FileWriter('./tboard/test_{}_{}'.format(save_file_nm.split('.')[0], experimentFolder), rnn_agent.sess.graph)
    for x_test, y_test in zip(*dset.sampler(rnn_agent.RNN_BATCH_SIZE, test=True)):
        test_losses = []
        loss, summary = rnn_agent.sess.run([rnn_agent.loss, rnn_agent.merged],
            feed_dict={
                rnn_agent.batch_ph: x_test,
                rnn_agent.target_ph: y_test
            }
        )
        test_losses.append(loss)
        iter_num = rnn_agent.global_step.eval(session=rnn_agent.sess)
        test_writer.add_summary(summary, iter_num)
    test_losses = np.array(test_losses)
    print('mean |   ||    (test):', np.mean(test_losses), ' Iter: ', rnn_agent.global_step.eval(session=rnn_agent.sess))
    print('     ||  |_')

    
    rnn_agent.sess.close()
    tf.reset_default_graph()
#------------------------------------------------------------------------------------------------------------

# Train the controller --------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
def train_C_generate_data(EPISODES, save_file_nm, chk_point_folder, sess_save_step=100, load_model=None, shuffle_agents=False, record=False, plot_reward=False, add_agents=[agents.SimpleAgent(), agents.RandomAgent(), agents.SimpleAgent()], encourage_win=False):
    if plot_reward:
        plt.xlabel('Episode #')
        plt.ylabel('Average reward for last 100 episodes')
    # Init the agent
    rnn_agent = RNN_Agent(model_training='C')

    # For saving model
    saver = tf.train.Saver()

    
    if not os.path.exists(chk_point_folder):
        os.makedirs(chk_point_folder)
    # Try to recover previous model
    if load_model is not None: load_folder = load_model
    else: load_folder = chk_point_folder
    latest_model = tf.train.latest_checkpoint(load_folder)
    if latest_model is not None:
        saver.restore(
            rnn_agent.sess, 
            latest_model
        )
        print("Restored ", latest_model)

    # Init dataset
    if record:
        dset = dataset(rnn_agent.RNN_SEQUENCE_LENGTH, save_file_nm, rnn_agent.utils)
        if os.path.exists(save_file_nm): dset.load()
    
    # TensorBoard writer
    experimentFolder = datetime.now().isoformat(timespec='minutes')
    C_writer = tf.summary.FileWriter('./tboard/train_C_{}_{}'.format(save_file_nm.split('.')[0], experimentFolder), rnn_agent.sess.graph)
    rnn_agent.summary_writer = C_writer

    agent_list =  [rnn_agent] + add_agents

    rnn_agent_index = agent_list.index(rnn_agent)

    if shuffle_agents: shuffle(agent_list)
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    mean_rewards_list = []
    episode_history = deque(maxlen=100)
    ties = deque(maxlen=100)
    
    rnn_wins = deque(maxlen=100)
    other_wins = deque(maxlen=100)

    for i_episode in range(EPISODES):
        # initialize
        state = env.reset()
        prev_state = np.copy(state)
        total_rewards = 0
        
        #-------------------------------------------------------------------
        done  = False; episode_obs = []; episode_acts = []
        #while not done and rnn_agent.is_alive:
        t = 0;  wins = {}
        while not done and rnn_agent.is_alive:
            t += 1
            #env.render()
            actions = env.act(state)

            episode_acts.append(actions[rnn_agent_index])
            episode_obs.append(rnn_agent.utils.input(state[rnn_agent_index]))
            
            state, reward, done, info = env.step(actions)
            if not encourage_win:
                reward[rnn_agent_index] = reward[rnn_agent_index] if not rnn_agent.is_alive else 0.1
            else:
                reward[rnn_agent_index] = reward[rnn_agent_index] if not rnn_agent.is_alive else 0.03
            if encourage_win and done and 'winners' in info:
                reward[rnn_agent_index] = 10 if info['winners'][0] == rnn_agent_index else -10
            #print("t: {} \t reward: {}\t Agent alive: {}".format(t, reward[rnn_agent_index], rnn_agent.is_alive) )
            
            total_rewards += reward[rnn_agent_index]
            rnn_agent.storeRollout(
                np.concatenate(( rnn_agent.utils.input(prev_state[rnn_agent_index]), rnn_agent.rnn_state )), 
                actions[rnn_agent_index], reward[rnn_agent_index]
            )
            prev_state = np.copy(state)
        #-------------------------------------------------------------------
        if 'winners' in info:
            rnn_wins.append(1 if info['winners'][0] == rnn_agent_index else 0)
            other_wins.append(1 if info['winners'][0] != rnn_agent_index else 0)
        wins_ratio = np.mean(other_wins)/np.mean(rnn_wins) 
        tflog('Other wins/agent wins ratio (100 wins)',  wins_ratio)
        
        ties.append(1 if 'Tie' in info else 0) 
        tie_ratio = np.mean(ties)/np.mean(rnn_wins)
        tflog('ties/agent wins ratio (100 steps)',  tie_ratio)

        

        # Final timestep observation
        episode_obs.append(rnn_agent.utils.input(state[rnn_agent_index]))
        if record: dset.add_episode(episode_obs, episode_acts)
        
        rnn_agent.update_C()

        episode_history.append(total_rewards)
        mean_rewards = np.mean(episode_history)

        print("Episode {}".format(i_episode))
        print("Finished after {} timesteps".format(t+1))
        print("Reward for this episode: {}".format(total_rewards))
        print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
        mean_rewards_list.append(mean_rewards)
        tflog('Iteration Number',  rnn_agent.train_iteration)
        tflog('Average reward for last 100 episodes',  mean_rewards)

        # Save the model
        if i_episode % sess_save_step == 0:
            saver.save(rnn_agent.sess, chk_point_folder, global_step=rnn_agent.C_step)
            if record: dset.save()

        # Plot rewards
        if plot_reward:
            x = np.arange(i_episode+1)
            # Linear Reg
            fit = np.polyfit(x,mean_rewards_list,1)
            fit_fn = np.poly1d(fit) 
            
            plt.plot(x, mean_rewards_list, '.', x, fit_fn(x), '--k') 
            plt.savefig("test.png")
            plt.gcf().clear()
        #print(info)
    print("Median Act Time: {} seconds".format(np.median(np.array(rnn_agent.act_times))))

    env.close()
    rnn_agent.sess.close()
    tf.reset_default_graph()

#------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    models = 'models/'
    if not os.path.exists(models):
        os.makedirs(models)
    
    # Random Actions
    # Agents at same positions
    lvl_ = "dataset_lvl{}.pickle"
    lvl = ''
    lvl1 = "dataset_lvl1.pickle" 
    # Random Actions
    # Agent positions are shuffled
    lvl2 = "dataset_lvl2.pickle" 
    # Data is generated while training controller
    # Agents at same positions
    lvl3 = "dataset_lvl3.pickle"
    #
    lvl4 = "dataset_lvl4.pickle"
    #
    lvl5 = "dataset_lvl5.pickle"
    #
    lvl6 = "dataset_lvl6.pickle"
    #
    lvl7 = "dataset_lvl7.pickle"
    lvl8 = "dataset_lvl8.pickle"
    lvl9 = "dataset_lvl9.pickle"
    lvl10 = "dataset_lvl10.pickle"
    lvl11 = "dataset_lvl11.pickle"
    lvl12 = "dataset_lvl12.pickle"
    lvl13 = "dataset_lvl13.pickle"
    lvl14 = "dataset_lvl14.pickle"
    lvl15 = "dataset_lvl15.pickle"
    lvl16 = "dataset_lvl16.pickle"
    lvl17 = "dataset_lvl17.pickle"
    lvl18 = "dataset_lvl18.pickle"
    lvl19 = "dataset_lvl19.pickle"

    # Level 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #print('-'*150); print('*'*90); print("Generating dataset ", lvl1); print('*'*90);
    #generate_data(400, lvl1) 
    
    #print('-'*150); print('*'*90); print("Training M (RNN) on dataset ", lvl1); print('*'*90);  
    #train_M(10, lvl1, models + lvl1 + '/')
    
    # Level 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''pbar = tqdm(total=1*10)
    for _ in range(1): 
        print('-'*150); print('*'*90); print("Generating dataset ", lvl2); print('*'*90);
        generate_data(10, lvl2, shuffle_agents=True) 
        pbar.update(10)
        gc.collect()
    pbar.close()
    '''
    
    #print('-'*150); print('*'*90); print("Training M (RNN) on dataset ", lvl2); print('*'*90);  
    #train_M(10, lvl2, models + lvl2 + '/', load_model=lvl1)
    
    #print('-'*150); print('*'*90); print("Training M (RNN) on dataset ", lvl2); print('*'*90);  
    #train_M(7, lvl2, models+lvl2+'/', load_model=models+lvl2+'/')
    
    # Level 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #train_C_generate_data(3000, lvl3, models + lvl3 + '/', plot_reward=False, add_agents=[agents.RandomAgent(), agents.SimpleAgent()])
    
    # Level 4 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #train_C_generate_data(1000, lvl4, models + lvl4 + '/', load_model=models+lvl3+'/', record=True, add_agents=[agents.RandomAgent(), agents.SimpleAgent()])
    
    # Level 5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #train_M(5, lvl4, models+lvl5+'/', load_model=models+lvl4+'/')

    # Level 6 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hory shit really good
    #train_C_generate_data(2000, lvl6, models + lvl6 + '/', load_model=models+lvl5+'/', shuffle_agents=True, add_agents=[agents.RandomAgent(), agents.SimpleAgent()])
    
    # Level 7 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ pretty good! 
    #train_C_generate_data(1000, lvl7, models + lvl7 + '/', load_model=models+lvl6+'/', shuffle_agents=True, record=True, add_agents=[agents.RandomAgent(), agents.SimpleAgent()])
    
    # Level 8 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #train_M(10, lvl4, models+lvl8+'/', load_model=models+lvl7+'/')
    
    # Level 9 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ oblitirated simple agent  
    #train_C_generate_data(600, lvl9, models + lvl9 + '/', load_model=models+lvl8+'/', shuffle_agents=True, record=True, add_agents=[agents.SimpleAgent()])

    # Level 10 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #train_M(20, lvl9, models+lvl10+'/', load_model=models+lvl9+'/')
    
    # Level 11 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    #train_C_generate_data(1000, lvl11, models + lvl11 + '/', load_model=models+lvl10+'/', shuffle_agents=True, record=True, add_agents=[agents.SimpleAgent(), agents.SimpleAgent()],encourage_win = True )
    # Level 12 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #train_M(20, lvl11, models+lvl12+'/', load_model=models+lvl11+'/')
    # Level 13 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    #train_C_generate_data(1000, lvl13, models + lvl13 + '/', load_model=models+lvl12+'/', shuffle_agents=True, record=True, add_agents=[agents.RandomAgent(), agents.SimpleAgent(), agents.SimpleAgent()],encourage_win = True )
    # Level 14 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #train_M(20, lvl13, models+lvl14+'/', load_model=models+lvl13+'/')
    # Level 15 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    #train_C_generate_data(1000, lvl15, models + lvl15 + '/', load_model=models+lvl14+'/', shuffle_agents=True, record=True, add_agents=[agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()],encourage_win = True )
    # Level 16 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #train_M(20, lvl15, models+lvl16+'/', load_model=models+lvl15+'/')
    

    
    curr_lev = 16;
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    curr_lev += 1
    lvl = lvl_.format(curr_lev)
    lvl_prev = lvl_.format(curr_lev-1)
    print('Level: ', curr_lev, '~~'*70)

    train_C_generate_data(1500, lvl, models + lvl + '/', load_model=models + lvl_prev +'/', shuffle_agents=True, record=True, add_agents=[agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()],encourage_win = True )
    
    train_M(20, lvl_prev, models+lvl+'/', load_model=models+lvl_prev+'/')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    


    curr_lev = 17;
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    curr_lev += 1
    lvl = lvl_.format(curr_lev)
    lvl_prev = lvl_.format(curr_lev-1)
    print('Level: ', curr_lev, '~~'*70)

    train_C_generate_data(1500, lvl, models + lvl + '/', load_model=models + lvl_prev +'/', shuffle_agents=True, record=True, add_agents=[agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()],encourage_win = True )
    
    train_M(20, lvl_prev, models+lvl+'/', load_model=models+lvl_prev+'/')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
