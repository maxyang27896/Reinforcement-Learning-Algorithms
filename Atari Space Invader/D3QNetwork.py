# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:30:10 2020

@author: MY2
"""
import tensorflow as tf

class DQNetworks:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            self.IS_weights = tf.placeholder(tf.float32, [None], name="IS_weights")
            self.mean_target = tf.reduce_mean(self.target_Q)
                        
            ''' Convolution Layers'''
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            with tf.variable_scope('conv1', reuse=True):
                self.conv1w = tf.get_variable('kernel')
                self.conv1b = tf.get_variable('bias')
                
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
                
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
            
            with tf.variable_scope('conv2', reuse=True):
                self.conv2w = tf.get_variable('kernel')
                self.conv2b = tf.get_variable('bias')
                

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")            

            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 64,
                                 kernel_size = [3,3],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")
            
            with tf.variable_scope('conv3', reuse=True):
                self.conv3w = tf.get_variable('kernel')
                self.conv3b = tf.get_variable('bias')

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            
            '''## Here to separate into two streams'''
            # The one to calculate V(s)
            self.value_fc = tf.layers.dense(inputs = self.flatten,
                                            units = 512,
                                            activation = tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")
            
            with tf.variable_scope('value_fc', reuse=True):
                self.value_fcw = tf.get_variable('kernel')
                self.value_fcb = tf.get_variable('bias')
                
            self.value =  tf.layers.dense(inputs = self.value_fc,
                                          units = 1,
                                          activation = None,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="value")
            
            # The one to calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs = self.flatten,
                                                units = 512,
                                                activation = tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")
            
            with tf.variable_scope('advantage_fc', reuse=True):
                self.advantage_fcw = tf.get_variable('kernel')
                self.advantage_fcb = tf.get_variable('bias')
            
            self.advantage = tf.layers.dense(inputs = self.advantage_fc, 
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             units = self.action_size, 
                                             activation=None,
                                             name = "advantage")
            
            
            # Agregating layer 
            self.output = tf.add(self.value, tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True)))
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis = -1)
            self.abs_TD_error = tf.abs(self.target_Q - self.Q)
            self.loss = tf.reduce_mean(self.IS_weights * tf.square(self.abs_TD_error))            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


            ''' #### FOR LOGGING #####'''
            self.max_Q_pred = tf.reduce_max(self.output, axis = 1)
            self.max_Q_mean_pred = tf.reduce_mean(self.max_Q_pred)
            self.argmax_Q_pred = tf.argmax(self.output, axis = 1)
            '''#### FOR LOGGING #####'''