# Agent

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

# Pomm -------------------------------------
from pommerman.agents import BaseAgent
from pommerman        import constants
from pommerman        import utility
#-------------------------------------------

# Regulars ----------------------------------
import os
import math
import time
import random

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple
from itertools   import count
from tqdm        import tqdm
#--------------------------------------------

# Fancy schmancy libraries -------------------------------------------------
import tensorflow as tf

from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
#---------------------------------------------------------------------------

# Custom Modules -----------------
from utils import _utils
#---------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class RNN_Agent(BaseAgent):
    def __init__(self, board_h=11, board_w=11, *args, **kwargs):
        self.name = 'FC Agent'
        super(RNN_Agent, self).__init__(*args, **kwargs)
        # Common functionalities among learning agents
        self.utils = _utils(board_h, board_w, 'checkpoints/save.tar')
        self.input_size = self.utils.input_size
        session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=session_conf)

        self.act_times = []

        # Hyperparameters --------------------------------------------
        self.RNN_SEQUENCE_LENGTH = 32
        self.RNN_HIDDEN_SIZE     = 256
        self.RNN_ATTENTION_SIZE  = 16
        self.RNN_KEEP_PROB       = 0.9
        self.RNN_BATCH_SIZE      = 64
        self.RNN_MODEL_PATH      = './model'

        self.NUM_ACTIONS = 6
        #-------------------------------------------------------------
        self.prev_action = np.zeros((self.NUM_ACTIONS,))

        # Different placeholders
        with tf.name_scope('Inputs'):
            self.batch_ph     = tf.placeholder(tf.float32, [self.RNN_BATCH_SIZE, self.RNN_SEQUENCE_LENGTH, self.input_size + self.NUM_ACTIONS], name='batch_ph')
            self.target_ph    = tf.placeholder(tf.float32, [self.RNN_BATCH_SIZE, self.RNN_SEQUENCE_LENGTH, self.input_size], name='target_ph')
            self.keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')
        
        # RNN layers
        self.rnn_cell         = tf.nn.rnn_cell.LSTMCell(self.RNN_HIDDEN_SIZE)
        #self.rnn_cell_attent  = tf.contrib.rnn.AttentionCellWrapper(self.rnn_cell, self.RNN_ATTENTION_SIZE) 
        #self.rnn_cell_drop    = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_attent, output_keep_prob=self.RNN_KEEP_PROB)
        self.rnn_cell_predict = tf.contrib.rnn.OutputProjectionWrapper(self.rnn_cell, output_size=self.input_size)
        # RNN trainer
        self.rnn_outputs_pred, self.rnn_final_state = \
        tf.nn.dynamic_rnn(
            self.rnn_cell_predict,
            inputs=self.batch_ph, 
            dtype=tf.float32
        )

        # !DEBUG
        print("self.rnn_outputs_pred size: ", self.rnn_outputs_pred.shape)
        for state in self.rnn_final_state: print(state)
        
        with tf.name_scope('Metrics'):
            loss = tf.reduce_mean(tf.squared_difference(self.rnn_outputs_pred, self.target_ph))
            tf.summary.scalar('loss', loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
        merged = tf.summary.merge_all()
        
        # For single step iterations
        self.input_single = tf.placeholder(tf.float32, [1, self.input_size + self.NUM_ACTIONS], name='input_single')
        state_single = self.rnn_cell.zero_state(1, tf.float32)
        (self.output_single, state_single) = self.rnn_cell(self.input_single, state_single)


        
    def act(self, obs, action_space):
        start = time.time()
        
        x = self.utils.input(obs)
        x = np.concatenate((x, self.prev_action))
        x = x.reshape(1, x.shape[0])
        self.sess.run(
            [self.output_single],
            feed_dict={
                self.input_single: x
            }
        )
        
        end = time.time()
        self.act_times.append(end - start)
        return random.randrange(self.NUM_ACTIONS)

 
