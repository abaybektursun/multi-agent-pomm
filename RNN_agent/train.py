# Training
import pommerman

import gc
import os
from tqdm import tqdm
import numpy as np
from random   import shuffle
from datetime import datetime

import torch
import tensorflow as tf

from pommerman import agents

from rnn_agent import RNN_Agent   
from pomm_dataset import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    rnn_agent.sess.run(tf.global_variables_initializer())
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
        
        env.close()
        #print(info)
    #print("Median Act Time: {} seconds".format(np.median(np.array(rnn_agent.act_times))))
    
    dset.save()
    rnn_agent.sess.close()
    tf.reset_default_graph()
#------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------
def train_M(epochs, save_file_nm, chk_point_folder, load_model=None):
    if not os.path.exists(chk_point_folder):
        os.makedirs(chk_point_folder)
    # Init the agent
    rnn_agent = RNN_Agent()
    rnn_agent.sess.run(tf.global_variables_initializer())

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
def train_C(EPISODES, save_file_nm):
    pass
#------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    models = 'models/'
    if not os.path.exists(models):
        os.makedirs(models)
    
    # Random Actions
    # Agents at same positions
    lvl1 = "dataset_lvl1.pickle" 
    # Random Actions
    # Agent positions are shuffled
    lvl2 = "dataset_lvl2.pickle" 
    
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
    
    print('-'*150); print('*'*90); print("Training M (RNN) on dataset ", lvl2); print('*'*90);  
    train_M(1, lvl2, models + lvl2 + '/')
    
    # Level 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

