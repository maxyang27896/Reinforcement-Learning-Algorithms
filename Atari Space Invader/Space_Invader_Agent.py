# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 23:53:01 2020

@author: MY2
"""


# Import packages
import numpy as np
import gym
import time
from collections import deque
import matplotlib.pyplot as plt

from skimage import transform 
from skimage.color import rgb2gray 

from PER import Memory
from DoubleNetworkModel import Model

class Agent:
    
    def __init__(self):
        self.env = gym.make("SpaceInvaders-v0")
        self.observation_size = self.env.observation_space
        self.action_size = self.env.action_space.n
        self.frame_size = [84, 84]
        self.stack_size = 4
        self.input_shape = [*self.frame_size, self.stack_size]
        
        MEMORY_SIZE = 30000  
        self.memory = Memory(MEMORY_SIZE)
        self.INITIAL_MEMORY_SIZE = 10000

        self.EXPLORATION_RATE = 1
        self.EXPLORATION_DECAY = 0.9999
        self.EXPLORATION_MIN = 0.01
        self.UPDATE_MODEL_STEP = 10000
        self.TRAINING_FREQUENCY = 4
        
        self.model = Model(self.input_shape, self.action_size)
        self.model.update_model()   

    def preprocess_frame(self, frame):
        ''' 
        FUNCTION: pre-process the frames for training 
        - grey scale
        - crop the edges
        - normalise the pixel values
        - resize the fram 
        '''
        gray = rgb2gray(frame)
        cropped_frame = gray[8:-12,4:-12]
        normalized_frame = cropped_frame/255.0
        preprocessed_frame = transform.resize(normalized_frame, self.frame_size)
        
        return preprocessed_frame 
    
    
    def stack_frames(self, stacked_frames, new_frame, is_new_episode):
        '''
        FUNCTION: Given a stacked frame, append a new frame to this stack    
        '''
        # Preprocess frame before stacking
        frame = self.preprocess_frame(new_frame)
        
        # if new episode make copies of frame and stack into np arrays
        if is_new_episode:
            stacked_frames = deque([np.zeros(self.frame_size, dtype = np.int) for i in range(0,self.stack_size)], maxlen=self.stack_size)       
            for _ in range(0, self.stack_size):
                stacked_frames.append(frame)
            stacked_states = np.stack(stacked_frames, axis = 2)
        # else append the frame to the queue
        else:
            stacked_frames.append(frame)
            stacked_states = np.stack(stacked_frames, axis = 2)
            
        return stacked_states, stacked_frames
    
    
    def test(self, n_episodes, model = None, memory = None, render=False, clip_reward=False):
        ''' 
        Play a game to test environment 
        '''
        avg_rewards = 0
        steps = []
        
        for i in range(1, n_episodes+1):
            state = self.env.reset()
            stacked_frames = deque([np.zeros(self.frame_size, dtype = np.int) for i in range(0,self.stack_size)], maxlen=self.stack_size)
            state, stacked_frames = self.stack_frames(stacked_frames, state, True)
            done = False 
            total_reward = 0
            step = 0
    
            while not done:
                if render:
                    self.env.render()
                    time.sleep(0.01)
                if model:
                    action = self.model.act(state, 0)
                else:
                    action = np.random.randint(self.model.action_space)
                state_next, reward, done, info = self.env.step(action)
                if clip_reward:
                    reward = np.sign(reward)
                if done:
                    state_next = np.zeros(self.frame_size, dtype=np.int)
                state_next, stacked_frames = self.stack_frames(stacked_frames, state_next, False)
                if memory:
                    memory.store((state, action, reward, state_next, done))
                state = state_next
                total_reward += reward
                step += 1
            
            if render:
                self.env.close()
            
            avg_rewards = avg_rewards + 1/(i) * (total_reward - avg_rewards)
            steps.append(step)
        
        print("The average rewards for {} runs is {}".format(n_episodes, avg_rewards))
            
        return steps, avg_rewards
    
    def initialise_memory(self):
        print("Start filling memory")
        
        while self.memory.memory_tree.capacity_filled  < self.INITIAL_MEMORY_SIZE:
            steps, total_reward = self.test(1, model = None, memory = self.memory, clip_reward=True)  
            
        print("Memory filled! The memory length is", self.memory.memory_tree.capacity_filled)
        
    def restore_and_test(self):
        self.model.load_model()
        self.test(1, model = self.model.DQNetwork, render=True)
        
        
    def run(self):
        self.initialise_memory()
        total_steps = 0
        losses = []
        rewards = []
        steps = []
        start_time = time.time()
        total_epsiodes = 5 
        for i in range(0, total_epsiodes):
            
            # Make a new episode and observe the first state
            state = self.env.reset()
            stacked_frames = deque([np.zeros(self.frame_size, dtype = np.int) for i in range(0,self.stack_size)], maxlen=self.stack_size)
            state, stacked_frames = self.stack_frames(stacked_frames, state, True)
            
            # Set step to 0
            episode_rewards = 0
            episode_steps = 0
            done = False
            
            while not done:
                episode_steps += 1
                total_steps += 1
                
                # Take a step
                action = self.model.act(state, self.EXPLORATION_RATE)
                next_state, reward, done, _ = self.env.step(action)
               
                # accumulate rewards 
                reward = np.sign(reward)
                episode_rewards += reward
                
                # Tasks when done 
                if done:
                    next_state = np.zeros(self.frame_size, dtype=np.int)
                    next_state, stacked_frames = self.stack_frames(stacked_frames, next_state, False)
                    print("Episode {}, exploration rate: {:.4f}, final rewards: {}, final loss is {:.4f}, Time elapsed: {:.4f}"\
                          .format(i+1, self.EXPLORATION_RATE, episode_rewards, loss, time.time() - start_time))
                    # self.test(1, model = self.model.DQNetwork, render=False)
                    start_time = time.time()
                else:
                    next_state, stacked_frames = self.stack_frames(stacked_frames, next_state, False)
                self.memory.store((state, action, reward, next_state, done))
                state = next_state       
                
                # Update target model and save model every UPDATE_STEP 
                if (total_steps % self.UPDATE_MODEL_STEP == 0):
                    self.model.update_model()
                    self.model.save_model()
                    
                ### LEARNING PROCEDURE ###
                if total_steps % self.TRAINING_FREQUENCY == 0:
                    tree_idx, IS_weights, batch = self.memory.sample(self.model.BATCH_SIZE)
                    loss, abs_TD_error = self.model.DQN_train(batch, IS_weights, tree_idx, total_steps)
                    losses.append(loss)
                    # Update the sample priority of batch
                    self.memory.update_batch(tree_idx, abs_TD_error)
                    # Reduce the exploreation every step
                    self.EXPLORATION_RATE *= self.EXPLORATION_DECAY
                    self.EXPLORATION_RATE = max(self.EXPLORATION_MIN, self.EXPLORATION_RATE)
                ### LEARNING PROCEDURE ###
        
            # Append values at the end of an episode
            steps.append(episode_steps)
            rewards.append(episode_rewards)
        
        # Save model at the end of training 
        print("Training Done")
        self.model.save_model() 
        
        # Save plot 
        plt.plot(rewards)
        plt.ylabel('Rewards')
        plt.xlabel('Episodes')
        plt.savefig('rewards.png')
        
if __name__ == "__main__":
    agent = Agent()
    agent.run()
    # agent.restore_and_test()
