# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:15:31 2020

@author: MY2
"""

import random
import gym
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# create the envirnment
env = gym.make("CartPole-v1")

# Parameters
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

exploration_rate = EXPLORATION_MAX

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

# create a memory buffer that is a queue structure
memory = deque(maxlen=MEMORY_SIZE)

# create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(24, input_shape=(observation_space,), activation="relu"))
model.add(tf.keras.layers.Dense(24, activation="relu"))
model.add(tf.keras.layers.Dense(action_space, activation="linear"))
model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

# function that appends new steps
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# state to action function
def act(state):
    if np.random.rand() < exploration_rate:
        return random.randrange(action_space)
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# Based on memeory queue, sample batches for updating the network
def experience_replay():
    
    global exploration_rate
    
    # train only if memeory size has reached beyond batch size
    if len(memory) < BATCH_SIZE:
        return
    
    # training based on the memory
    batch = random.sample(memory, BATCH_SIZE)
    
    # for each step in the batch, update the network with new q_update
    for state, action, reward, state_next, terminal in batch:
        q_update = reward
        if not terminal:
            q_update = (reward + GAMMA * np.amax(model.predict(state_next)[0]))
        q_values = model.predict(state)
        q_values[0][action] = q_update
        model.fit(state, q_values, verbose=0)
    exploration_rate *= EXPLORATION_DECAY
    exploration_rate = max(EXPLORATION_MIN, exploration_rate)
    
   
run = 0
N = 20
# run for N episodes
for i in range(0,N):
    run += 1
    state = env.reset()
    terminal = False
    state = np.reshape(state, [1, observation_space])
    step = 0
    
    while not terminal:
        step += 1
        #env.render()
        action = act(state)
        state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -reward
        state_next = np.reshape(state_next, [1, observation_space])
        remember(state, action, reward, state_next, terminal)
        state = state_next
        if terminal:
            print("Run: " + str(run) + ", exploration: " + str(exploration_rate) + ", score: " + str(step))
#                score_logger.add_score(step, run)
        experience_replay()
    
print('Done Training!')        


def test(model, n_episodes):
    ''' 
    function to test the result of the q_table for n_episodes and return average
    rewards
    '''
    
    # store average rewards
    avg_rewards = 0
    
    for i in range(1, n_episodes+1):

        state = env.reset()
        done = False 
        total_rewards = 0
        
        # until done 
        while not done:
            
            # take an action in the max q_table
            action = np.argmax(model.predict(np.array([state]))[0])
            state, reward, done, info = env.step(action)
            
            # acculmulate rewards
            total_rewards += reward
        
        avg_rewards = avg_rewards + 1/(i) * (total_rewards - avg_rewards)
          
    return avg_rewards

# run test
n_episodes = 50
avg_rewards = test(model, n_episodes)
print("After " + str(n_episodes) + " episodes, the average score is " + str(avg_rewards))