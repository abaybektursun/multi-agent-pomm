from sklearn.model_selection import train_test_split
from sklearn.utils           import resample

import numpy as np


class dataset:
    def __init__(self, max_len):
        self.max_len = max_len 
        self.batch   = []
        self.target  = []
    def add_episode(self, obs, actions):
        num_obs, num_act = len(obs), len(actions)
        assert num_obs == num_act + 1
        
        obs_data   = obs[:-1]
        obs_target = obs[1:]

        num_batches = num_act // self.max_len 

        batch_id = 0
        while batch_id < num_batches:
            self.batch.append(obs_data[batch_id*self.max_len : (batch_id+1)*self.max_len])
            self.target.append(obs_target[batch_id*self.max_len : (batch_id+1)*self.max_len])
            batch_id += 1

        if num_act % self.max_len != 0:
            hanging_obs    = [0] * self.max_len
            hanging_target = [0] * self.max_len
            for idx in range(batch_id*self.max_len, num_act):
                hanging_obs[idx - batch_id*self.max_len] = obs_data[idx]
                hanging_target[idx - batch_id*self.max_len] = obs_target[idx]
            self.batch.append(hanging_obs)
            self.target.append(hanging_target)
            
            

