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
#from attention import attention
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

        # Hyperparameters --------------------------------------------
        self.RNN_NUM_WORDS       = 10000
        self.RNN_INDEX_FROM      = 3
        self.RNN_SEQUENCE_LENGTH = 250
        self.RNN_EMBEDDING_DIM   = 100
        self.RNN_HIDDEN_SIZE     = 150
        self.RNN_ATTENTION_SIZE  = 50
        self.RNN_KEEP_PROB       = 0.8
        self.RNN_BATCH_SIZE      = 256
        self.RNN_DELTA           = 0.5
        self.RNN_MODEL_PATH      = './model'
        #-------------------------------------------------------------

        # Different placeholders
        with tf.name_scope('Inputs'):
            self.batch_ph     = tf.placeholder(tf.float32, [None, self.SEQUENCE_LENGTH], name='batch_ph')
            self.target_ph    = tf.placeholder(tf.float32, [None], name='target_ph')
            self.keep_prob_ph = tf.placeholder(tf.float32,         name='keep_prob_ph')
        
        # RNN layer
        self.rnn_outputs, self.rnn_states = \
        tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(self.RNN_HIDDEN_SIZE),
            inputs=self.batch_ph, 
            dtype=tf.float32
        )
        tf.summary.histogram('RNN_outputs', self.rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output, alphas = \
                attention(
                    self.rnn_outputs, 
                    self.ATTENTION_SIZE, 
                    return_alphas=True
                )
        tf.summary.histogram('alphas', alphas)
        
        with tf.name_scope('Metrics'):
            loss = tf.reduce_mean(tf.squared_difference(self.rnn_outputs, self.target_ph))
            tf.summary.scalar('loss', loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
            tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()


        
    def act(self, obs, action_space):
        x = self.utils.input(obs) 
        return rand.randrange(6)
 
