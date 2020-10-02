#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 21:41:56 2020

@author: max
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# create the envirnment
env = gym.make("CartPole-v1")
env.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Get state spaces
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

GAMMA = 0.95
LEARNING_RATE = 0.01

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
            self.inputs_ = tf.placeholder(tf.float32, [None, 4], name="inputs")
            self.actions_ = tf.placeholder(tf.int32, [None, 2], name="actions")
            self.discounted_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_rewards")

            self.fc1 = tf.contrib.layers.fully_connected(inputs = self.inputs_,
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
            
def act(state, greedy=False):
    action_distribution = sess.run(PGNetwork.action_distribution, feed_dict={PGNetwork.inputs_: state[tf.newaxis,...]})
    if greedy:
        action = np.argmax(action_distribution[0])
    else:
        action = np.random.choice(action_space, p = action_distribution[0])
    return action


def test(n_episodes, model=True, with_limit=False, render=False):
    ''' 
    function to test the result of the model for n_episodes and return average
    rewards
    '''
    # store average rewards
    avg_rewards = 0

    for i in range(1, n_episodes+1):
        state = env.reset()
        done = False 
        total_rewards = 0
        step = 0 
        
        # until done 
        while not done:
            if render:
                env.render()
                time.sleep(0.01)
                
            # take an action in the max q_table
            if model:
                action = act(state, greedy=True)
            else:
                action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            # acculmulate rewards
            total_rewards += reward
            step += 1
            
            # End the epsiode if step > 200 
            if with_limit:
                if step > 200:
                    done = True
                    
        if render:
           env.close()
        
        avg_rewards = avg_rewards + 1/(i) * (total_rewards - avg_rewards)
          
    return avg_rewards

# Create the network and initialise
tf.reset_default_graph()
PGNetwork = NNetwork(observation_space, action_space, LEARNING_RATE, name = "PGNetwork")
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# run for N episodes
N = 500
rewards = []
steps = []
losses = []
for i in range(0, N):
    
    # Initialise episode
    state = env.reset()
    terminal = False
    step = 0
    
    # For storing 
    states_mb = []
    actions_mb = []
    episode_rewards = []
    
    while not terminal:
        step += 1
        # Take a step
        action = act(state)
        state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -reward
        
        # store values
        episode_rewards.append(reward)
        states_mb.append(state)
        
        action_ = np.zeros(action_space)
        action_[action] = 1
        actions_mb.append(action_)
        
        state = state_next
        
        if terminal:
            discounted_rewards_mb = get_discounted_rewards(episode_rewards)
            
            actions_mb = np.array(actions_mb)
            states_mb = np.array(states_mb)
            
            # Fit the data 
            _, loss = sess.run([PGNetwork.optimiser, PGNetwork.loss],
                                feed_dict={PGNetwork.inputs_: states_mb,
                                            PGNetwork.actions_: actions_mb,
                                            PGNetwork.discounted_rewards_: discounted_rewards_mb})
            
            print("Episode: {}, Score: {}, loss: {}".format(i, step, loss))
            
            rewards.append(sum(episode_rewards))
            steps.append(step)
            losses.append(loss)
    
print('Done Training!')    

# plot result
plt.plot(rewards)
plt.plot(steps)
plt.ylabel('Scores')
plt.xlabel('Runs')
plt.legend(['Average Rewards', 'Steps'])
plt.show()

# plot loss
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Steps')
plt.show()

# Test results
test_reward = test(10, model=True, with_limit=False, render=False)
print("Final test rewards, ", test_reward)