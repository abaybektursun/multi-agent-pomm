from sklearn.model_selection import train_test_split
from sklearn.utils           import shuffle

import numpy as np

import pickle


class dataset:
    def __init__(self, max_len, save_file_nm, utils):
        self.save_file_nm = save_file_nm
        self.max_len = max_len 
        self.utils   = utils
        self.batch   = []
        self.target  = []
        self.x_train = None
        self.y_train = None
        self.x_test  = None
        self.y_test  = None

    def add_episode(self, obs, actions):
        num_obs, num_act = len(obs), len(actions)
        assert num_obs == num_act + 1
        
        obs_data   = obs[:-1]
        # Concat observation and action
        for t in range(num_act):
            obs_data[t] = np.concatenate((obs_data[t], self.utils.action_onehot(actions[t]) ))
        obs_target = obs[1:]

        num_batches = num_act // self.max_len 

        batch_id = 0
        while batch_id < num_batches:
            self.batch.append(obs_data[batch_id*self.max_len : (batch_id+1)*self.max_len])
            self.target.append(obs_target[batch_id*self.max_len : (batch_id+1)*self.max_len])
            batch_id += 1

        if num_act % self.max_len != 0:
            dta_vec_shape = obs_data[0].shape 
            tar_vec_shape = obs_target[0].shape
            hanging_obs    = [np.zeros(dta_vec_shape)]*self.max_len
            hanging_target = [np.zeros(tar_vec_shape)]*self.max_len
            for idx in range(batch_id*self.max_len, num_act):
                hanging_obs[idx - batch_id*self.max_len] = obs_data[idx]
                hanging_target[idx - batch_id*self.max_len] = obs_target[idx]
            self.batch.append(hanging_obs)
            self.target.append(hanging_target)
        
        # DEBUG
        '''
        print('Batch Debug')
        for b in self.batch:
            print(len(b))
            for bv in b:
                try: print('\t', bv.shape)
                except: print(bv); exit(1)
        print('Target Debug')
        for t in self.target:
            print(len(t))
            for tv in t:
                print('\t', tv.shape)
        '''

    
    def sampler(self, sample_size, test=False):
        train_size = self.x_train.shape[0] 
        test_size  = self.x_test.shape[0]  
        assert test_size > sample_size and train_size > sample_size
        if test:
            set_size = test_size
            x = self.x_test
            y = self.y_test
        else:
            set_size = train_size
            x = self.x_train
            y = self.y_train
        
        num_batches = set_size // sample_size
        cut_idx     = num_batches * sample_size
        
        return np.split(x[:cut_idx], num_batches), np.split(y[:cut_idx], num_batches)
            

    def save(self):
        with open(self.save_file_nm, 'wb') as f:
            pickle.dump({
                'batch': self.batch,
                'target': self.target,
                'max_len': self.max_len
            }, f)
        print("Pickled the data")
        print("Batch size: {} \nTarget Size: {}".format(len(self.batch), len(self.target)))
        

    def load(self):
        with open(self.save_file_nm, 'rb') as f:
            data = pickle.load(f)
        assert data['max_len'] == self.max_len
        self.batch  = data['batch']
        self.target = data['target']

        x_train, x_test, y_train, y_test = train_test_split(self.batch, self.target, test_size=0.07)
        x_train, y_train = shuffle(x_train, y_train)
        x_test,  y_test  = shuffle(x_test,  y_test)
        self.x_train, self.x_test, self.y_train, self.y_test = np.concatenate(x_train), np.concatenate(x_test), np.concatenate(y_train), np.concatenate(y_test)
