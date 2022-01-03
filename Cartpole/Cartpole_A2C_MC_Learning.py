# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:24:28 2020

@author: MY2
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

# Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.005

# Define Network
class ACNetworks:
     def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
    
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, 4], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 2], name="actions")
            self.discounted_rewards_ = tf.placeholder(tf.float32, [None,1], name="discounted_rewards")

            self.fc_shared = tf.layers.dense(inputs = self.inputs_,
                                                units = 24,
                                                activation=tf.nn.relu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            # Actor stream
            self.fc_actor = tf.layers.dense(inputs = self.fc_shared,
                                                units = 24,
                                                activation= tf.nn.relu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            # Output action probabilities           
            self.action_distribution = tf.layers.dense(inputs = self.fc_actor,
                                                units = self.action_size,
                                                activation= tf.nn.softmax,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            
            # Value Stream 
            self.fc_value = tf.layers.dense(inputs = self.fc_shared,
                                    units = 24,
                                    activation= tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            # output value
            self.value = tf.layers.dense(inputs = self.fc_value,
                                    units = 1,
                                    activation= None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            # Calculate the advantage
            self.advantage = self.discounted_rewards_ - self.value
            
            # Actor optimisation
            self.prob = tf.reduce_sum(tf.multiply(self.action_distribution, self.actions_), axis=1, keep_dims=True)
            self.neg_log_prob = -tf.log(self.prob)
            self.actor_loss = tf.reduce_mean(self.neg_log_prob * tf.stop_gradient(self.advantage))
            self.actor_optimiser = tf.train.AdamOptimizer(self.learning_rate).minimize(self.actor_loss)
            
            # # Critic optimsation
            self.critic_loss = tf.reduce_mean(tf.square(self.advantage))
            self.critic_optimiser = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)
       

def act(state, greedy=False):
    '''
    Function to return action based on probability distribution
    '''

    action_distribution = sess.run(ACNetwork.action_distribution, feed_dict={ACNetwork.inputs_: state[tf.newaxis,...]})
    if greedy:
        action = np.argmax(action_distribution[0])
    else:
        action = np.random.choice(action_space, p = action_distribution[0])
    return action


def get_discounted_rewards(episode_rewards):
    '''
    Function returns discounte rewards at the end of an episode

    '''
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


# Create the graph
tf.reset_default_graph()
ACNetwork = ACNetworks(observation_space, action_space, 0.001, "DQN")    
sess = tf.Session()
sess.run(tf.global_variables_initializer()) 


# run for N episodes
N = 5000
rewards = []
steps = []
actor_losses = []
critic_losses = []
for episode in range(0, N):
    
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
         
        '''Monte Carlo  train'''
        if terminal:
               discounted_rewards_mb = get_discounted_rewards(episode_rewards)
               
               actions_mb = np.array(actions_mb)
               states_mb = np.array(states_mb)
               discounted_rewards_mb = np.expand_dims(discounted_rewards_mb, axis=-1)
               
               # Fit the data 
               _, actor_loss, _, critic_loss = sess.run([ACNetwork.actor_optimiser, ACNetwork.actor_loss, ACNetwork.critic_optimiser, ACNetwork.critic_loss],
                                                       feed_dict={ACNetwork.inputs_: states_mb,
                                                                   ACNetwork.actions_: actions_mb,
                                                                   ACNetwork.discounted_rewards_: discounted_rewards_mb})
                                   
               print("Episode: {}, Score: {}, actor loss: {}, critic loss: {}".format(episode, step, actor_loss, critic_loss))
               
               rewards.append(sum(episode_rewards))
               steps.append(step)
               actor_losses.append(actor_loss)
               critic_losses.append(critic_loss)
        
        state = state_next
                
print('Done Training!')    

# plot result
plt.plot(rewards)
plt.plot(steps)
plt.ylabel('Scores')
plt.xlabel('Runs')
plt.legend(['Average Rewards', 'Steps'])
plt.show()

# plot loss
plt.plot(actor_losses)
plt.ylabel(' Actor Loss')
plt.xlabel('Steps')
plt.show()

plt.plot(critic_losses)
plt.ylabel(' Critic Loss')
plt.xlabel('Steps')
plt.show()

# Test results
test_reward = test(10, model=True, with_limit=False, render=False)
print("Final test rewards, ", test_reward)