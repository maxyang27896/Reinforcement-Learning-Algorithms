#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 21:41:56 2020

@author: max
"""

import gym
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# create the envirnment
env = gym.make("CartPole-v1")

# Get state spaces
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

GAMMA = 0.9
LEARNING_RATE = 0.001

def get_discounted_rewards(episode_rewards):
    discounted_rewards_mb = np.zeros(len(episode_rewards))
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative*GAMMA +  episode_rewards[i]
        discounted_rewards_mb[i] = cumulative
    # Normalise the discounted rewards
    mean = np.mean(discounted_rewards_mb)
    std = np.std(discounted_rewards_mb)
    discounted_rewards_mb = (discounted_rewards_mb - mean) / (std)
    return discounted_rewards_mb


class NNetwork:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
    
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.int32, [None, self.action_size], name="actions")
            self.discounted_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_rewards")

            self.fc1 = tf.contrib.layers.fully_connected(inputs = self.input_,
                                                num_outputs = 10,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
            
            self.fc2 = tf.contrib.layers.fully_connected(inputs = self.fc1,
                                                num_outputs = self.action_size,
                                                activation_fn= tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
            
            # Output logits of each action
            self.fc3 = tf.contrib.layers.fully_connected(inputs = self.fc2,
                                                num_outputs = self.action_size,
                                                activation_fn= None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    
            # Output logits probabiltiy of distriibution of each action, the output for taking an action
            self.action_distribution = tf.nn.softmax(self.fc3)
            
            # The action is one hot encoded action
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.fc3, labels = self.actions_)
            self.loss = tf.reduce_mean(neg_log_prob * self.discounted_rewards_) 
            
            self.optimiser = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

# Create the network and initialise
tf.reset_default_graph()
PGNetwork = NNetwork(observation_space, action_space, LEARNING_RATE, name = "PGNetwork")
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# run for N episodes
N = 1
for i in range(0, N):
    
    # Initialise episode
    state = env.reset()
    terminal = False
    step = 0
    
    # For storing 
    state_mb = []
    action_mb = []
    episode_rewards = []
    
    while not terminal:
        step += 1
        # Take a step
        action = np.random.randint(action_space)
        state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -reward
        
        # store values
        episode_rewards.append(reward) 
        action_mb.append(action)
        state_mb.append(state)
        
        state = state_next
        
        if terminal:
            discounted_rewards_mb = get_discounted_rewards(episode_rewards)
            
            # Train here
            

print('Done Training!')    