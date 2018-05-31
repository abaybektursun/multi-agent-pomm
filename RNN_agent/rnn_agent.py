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
#---------------------------------------------------------------------------

# Custom Modules -----------------
from utils import _utils
#---------------------------------



class RNN_Agent(BaseAgent):
    def __init__(self, board_h=11, board_w=11, *args, **kwargs):
        self.name = 'FC Agent'
        self.model_training=kwargs['model_training']
        kwargs.pop('model_training')
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
        self.RNN_KEEP_PROB       = 0.8
        self.RNN_BATCH_SIZE      = 16
        self.RNN_MODEL_PATH      = './model'

        self.NUM_ACTIONS = self.utils.num_actions
        #-------------------------------------------------------------
        self.prev_action = np.zeros((self.NUM_ACTIONS,))

        # Different placeholders
        with tf.name_scope('RNN_Inputs'):
            self.batch_ph  = tf.placeholder(tf.float32, [self.RNN_BATCH_SIZE, self.RNN_SEQUENCE_LENGTH, self.input_size + self.NUM_ACTIONS], name='batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [self.RNN_BATCH_SIZE, self.RNN_SEQUENCE_LENGTH, self.input_size], name='target_ph')
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.C_step      = tf.Variable(0, name='C_step', trainable=False)
        
        # RNN layers
        self.rnn_cell         = tf.nn.rnn_cell.LSTMCell(self.RNN_HIDDEN_SIZE)
        #self.rnn_cell_attent  = tf.contrib.rnn.AttentionCellWrapper(self.rnn_cell, self.RNN_ATTENTION_SIZE) 
        self.rnn_cell_drop    = tf.contrib.rnn.DropoutWrapper(self.rnn_cell, output_keep_prob=self.RNN_KEEP_PROB)
        self.rnn_cell_predict = tf.contrib.rnn.OutputProjectionWrapper(self.rnn_cell_drop, output_size=self.input_size)
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
            self.loss = tf.reduce_mean(tf.squared_difference(self.rnn_outputs_pred, self.target_ph))
            if self.model_training == 'M': tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss, global_step=self.global_step)
        if self.model_training == 'M': self.merged = tf.summary.merge_all()
        
        # For single step iterations
        self.input_single = tf.placeholder(tf.float32, [1, self.input_size + self.NUM_ACTIONS], name='input_single')
        state_single = self.rnn_cell.zero_state(1, tf.float32)
        (self.output_single, state_single) = self.rnn_cell(self.input_single, state_single)
        
        # Controller ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.reinforce()

        self.sess.run(tf.global_variables_initializer())

        # make sure all variables are initialized
        self.session.run(tf.assert_variables_initialized())
    
    def policy_network(self, states):
        # define policy neural network
        W1 = tf.get_variable("W1", [self.state_dim, 20],
                             initializer=tf.random_normal_initializer())
        b1 = tf.get_variable("b1", [20],
                             initializer=tf.constant_initializer(0))
        h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
        W2 = tf.get_variable("W2", [20, self.num_actions],
                             initializer=tf.random_normal_initializer(stddev=0.1))
        b2 = tf.get_variable("b2", [self.num_actions],
                             initializer=tf.constant_initializer(0))
        p = tf.matmul(h1, W2) + b2
        return p


    
    def reinforce(self,  
                     init_exp=0.5,         # initial exploration prob
                     anneal_steps=10000,   # N steps for annealing exploration
                     final_exp=0.0,        # final exploration prob
                     summary_writer=None,
                     summary_every=100
                     ):
        self.summary_writer = summary_writer
        # tensorflow machinery
        self.C_optimizer    = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)

        # training parameters
        self.session         = self.sess
        self.state_dim       = self.input_size + self.RNN_HIDDEN_SIZE 
        self.num_actions     = self.NUM_ACTIONS
        self.discount_factor = 0.99              # discount future rewards
        self.max_gradient    = 5                 # max gradient norms
        self.reg_param       = 0.001             # regularization constants

        # exploration parameters
        self.exploration  = init_exp
        self.init_exp     = init_exp
        self.final_exp    = final_exp
        self.anneal_steps = anneal_steps

        # counters
        self.train_iteration = 0

        # rollout buffer
        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []

        # record reward history for normalization
        self.all_rewards = []
        self.max_reward_length = 1000000

        # create and initialize variables
        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #self.session.run(tf.variables_initializer(var_lists))

        self.summary_every = summary_every
        if self.summary_writer is not None:
            # graph was not available when journalist was created
            #self.summary_writer.add_graph(self.session.graph)
            pass

       
        
    def resetModel(self):
        self.cleanUp()
        self.train_iteration = 0
        self.exploration     = self.init_exp
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.session.run(tf.variables_initializer(var_lists))
  
    def create_variables(self):
        with tf.name_scope("C_inputs"):
            # raw state representation
            #self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
            self.states = tf.placeholder_with_default( tf.zeros((1, self.state_dim)), (None, self.state_dim))
  
        # rollout action based on current policy
        with tf.name_scope("C_predict_actions"):
            # initialize policy network
            with tf.variable_scope("policy_network"):
                self.policy_outputs = self.policy_network(self.states)
  
            # predict actions from policy network
            self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
            # Note 1: tf.multinomial is not good enough to use yet
            # so we don't use self.predicted_actions for now
            self.predicted_actions = tf.multinomial(self.action_scores, 1)
  
        # regularization loss
        policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")
  
        # compute loss and gradients
        with tf.name_scope("compute_pg_gradients"):
            # gradients for selecting action from policy network
            #self.taken_actions = tf.placeholder(tf.int32, (None,), name="taken_actions")
            #self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
            self.taken_actions = tf.placeholder_with_default( tf.zeros((1, ), dtype=tf.int32), (None, ) )
            self.discounted_rewards = tf.placeholder_with_default(  tf.zeros((1, )), (None,) )
  
            with tf.variable_scope("policy_network", reuse=True):
                self.logprobs = self.policy_network(self.states)
  
            # compute policy loss and regularization loss
            self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logprobs, labels=self.taken_actions)
            self.pg_loss            = tf.reduce_mean(self.cross_entropy_loss)
            self.reg_loss           = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
            self.loss               = self.pg_loss + self.reg_param * self.reg_loss
  
            # compute gradients
            self.gradients = self.C_optimizer.compute_gradients(self.loss)
  
            # compute policy gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)
  
            for grad, var in self.gradients:
                if self.model_training == 'C': tf.summary.histogram(var.name, var)
                if grad is not None:
                    if self.model_training == 'C':tf.summary.histogram(var.name + '/gradients', grad)
  
            # emit summaries
            if self.model_training == 'C':tf.summary.scalar("policy_loss", self.pg_loss)
            if self.model_training == 'C':tf.summary.scalar("reg_loss", self.reg_loss)
            if self.model_training == 'C':tf.summary.scalar("total_loss", self.loss)
  
        # training update
        with tf.name_scope("train_policy_network"):
            # apply gradients to update policy network
            self.train_op = self.C_optimizer.apply_gradients(self.gradients)
  
        if self.model_training == 'C':self.summarize = tf.summary.merge_all()
        self.no_op = tf.no_op()

    def sampleAction(self, states):
        # TODO: use this code piece when tf.multinomial gets better
        # sample action from current policy
        # actions = self.session.run(self.predicted_actions, {self.states: states})[0]
        # return actions[0]
        self.train_iteration = self.C_step.eval(session=self.session)
        # temporary workaround
        def softmax(y):
            """ simple helper function here that takes unnormalized logprobs """
            maxy = np.amax(y)
            e = np.exp(y - maxy)
            return e / np.sum(e)
  
        # epsilon-greedy exploration strategy
        if random.random() < self.exploration:
            return random.randint(0, self.num_actions-1)
        else:
            action_scores = self.session.run(self.action_scores, {self.states: states})[0]
            action_probs  = softmax(action_scores) - 1e-5
            action = np.argmax(np.random.multinomial(1, action_probs))
            return action
  
    def update_C(self):
        N = len(self.reward_buffer)
        r = 0 # use discounted reward to approximate Q value
  
        # compute discounted future rewards
        discounted_rewards = np.zeros(N)
        for t in reversed(range(N)):
            # future discounted reward from now on
            r = self.reward_buffer[t] + self.discount_factor * r
            discounted_rewards[t] = r
  
        # reduce gradient variance by normalization
        self.all_rewards += discounted_rewards.tolist()
        self.all_rewards = self.all_rewards[:self.max_reward_length]
        discounted_rewards -= np.mean(self.all_rewards)
        discounted_rewards /= np.std(self.all_rewards)
 
        # whether to calculate summaries
        calculate_summaries = self.summary_writer is not None and self.train_iteration % self.summary_every == 0
  
        # update policy network with the rollout in batches
        for t in range(N-1):
            # prepare inputs
            states  = self.state_buffer[t][np.newaxis, :]
            actions = np.array([self.action_buffer[t]])
            rewards = np.array([discounted_rewards[t]])
  
            # evaluate gradients
            grad_evals = [grad for grad, var in self.gradients]
             
            # perform one update of training
            _, summary_str = \
            self.session.run(
                [
                    self.train_op,
                    self.summarize if calculate_summaries else self.no_op
                ], 
                {
                    self.states:             states,
                    self.taken_actions:      actions,
                    self.discounted_rewards: rewards
                }
            )
  
        self.annealExploration()
        #self.train_iteration += 1
        new_C_step = self.C_step.eval(session=self.session) + 1
        self.session.run(self.C_step.assign(
            new_C_step
        )) 

  
        # clean up
        self.cleanUp()
  
    def annealExploration(self, stategy='linear'):
        ratio = max((self.anneal_steps - self.train_iteration)/float(self.anneal_steps), 0)
        self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp
  
    def storeRollout(self, state, action, reward):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)
  
    def cleanUp(self):
        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []

    # ----------------------------------------------------
    def act(self, obs, action_space):
        start = time.time()
        
        x = self.utils.input(obs)

        x_rnn = np.concatenate((x, self.prev_action))
        x_rnn = x_rnn.reshape(1, x_rnn.shape[0])
        self.rnn_state = self.sess.run(
            [self.output_single],
            feed_dict={
                self.input_single: x_rnn
            }
        )

        self.rnn_state = np.array(self.rnn_state).flatten()
        x = np.concatenate((x, self.rnn_state))

        end = time.time()
        self.act_times.append(end - start)
        
        return self.sampleAction(x[np.newaxis,:])
    # ----------------------------------------------------

 
