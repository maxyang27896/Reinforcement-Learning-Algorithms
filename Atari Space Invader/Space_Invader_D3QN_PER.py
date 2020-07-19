# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 23:53:01 2020

@author: MY2
"""


# Import packages
import numpy as np
import gym
import time
import os
import tensorflow as tf
from collections import deque

from skimage import transform 
from skimage.color import rgb2gray 

from PER import Memory

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
             
class Model:
    
    def __init__(self, input_shape, action_space):
        self.input_shape = input_shape
        self.action_space = action_space
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 64
        self.GAMMA = 0.9

        self.DQNetwork = DQNetworks(input_shape, action_space, self.LEARNING_RATE, "DQN")
        self.TargetDQNetwork = DQNetworks(input_shape, action_space, self.LEARNING_RATE, "TargetDQN")
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def act(self, state, exploration_rate):
        '''
        FUNCTION: Take an action based on the state
        RETURN: An action based on the exploration rate
        '''
        if np.random.rand() < exploration_rate:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(self.sess.run(self.DQNetwork.output, 
                                        feed_dict = {self.DQNetwork.inputs_: state[tf.newaxis,...]})[0])
        return action

    def DQN_train(self, batch, IS_weights, tree_idx, total_steps):
        '''
        FUNCTION: to train the model based on batch of experiences
        RRETURN: loss after single batch training 
        '''
        # Extract each parameter from batch
        state_batch = np.zeros([self.BATCH_SIZE, *self.input_shape])
        action_batch = np.zeros([self.BATCH_SIZE, self.action_space], dtype = int)
        reward_batch = np.zeros(self.BATCH_SIZE)
        state_next_batch = np.zeros([self.BATCH_SIZE, *self.input_shape])
        done_batch = np.zeros(self.BATCH_SIZE)
        for i in range(0,len(batch)):
            state_batch[i] = batch[i][0]
            action_batch[i][batch[i][1]] = 1
            reward_batch[i] = batch[i][2]
            state_next_batch[i] = batch[i][3]
            done_batch[i] = batch[i][4]
        # Get predicted Q value batch
        Q_next_state = self.sess.run(self.DQNetwork.output, 
                                feed_dict = {self.DQNetwork.inputs_: state_next_batch})
        Target_Q_next_state = self.sess.run(self.TargetDQNetwork.output, 
                                       feed_dict = {self.TargetDQNetwork.inputs_: state_next_batch})
        # Calculate target Q batch
        target_Q_batch = np.zeros(self.BATCH_SIZE)
        for i in range(0, len(batch)):
            if done_batch[i]:
                target_Q_batch[i] = reward_batch[i]
            else:
                best_action = np.argmax(Q_next_state[i])
                target_Q_batch[i] = reward_batch[i] + self.GAMMA * Target_Q_next_state[i][best_action]
        # Fit the data 
        loss, abs_TD_error, _ = self.sess.run([self.DQNetwork.loss, self.DQNetwork.abs_TD_error, self.DQNetwork.optimizer],
                                         feed_dict={self.DQNetwork.inputs_: state_batch,
                                                    self.DQNetwork.target_Q: target_Q_batch,
                                                    self.DQNetwork.actions_: action_batch,
                                                    self.DQNetwork.IS_weights: IS_weights})
   
        return loss, abs_TD_error
        
        
    def update_model(self):
        '''
        FUNCTION: update the target model
        '''
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQN")
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetDQN")
        updated_weights = [to_var.assign(from_var) for from_var, to_var in zip(from_vars, to_vars)]
        sess.run(updated_weights())
        print("Target network updated!")
    
    def save_model(self, path):
        '''
        FUNCTION: save main DQN model to path
        '''
        self.DQNetwork.save(path) 
        print("DQNetwork saved!")
    
    def load_model(self, path):
        '''
        FUNCTION: save main DQN model to path
        '''
        self.DQNetwork = tf.keras.models.load_model(path)
        print("Succesfully loaded network!")



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

        self.EXPLORATION_RATE = 1
        self.EXPLORATION_DECAY = 0.9999
        self.EXPLORATION_MIN = 0.01
        self.MAX_STEP = 50000
        self.UPDATE_MODEL_STEP = 5000
        self.path =  "./Models/model.h5"
        
        
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
    
    
    def test(self, n_episodes, model = None, memory = None, render=False):
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
                if model:
                    action = self.model.act(state, 0)
                else:
                    action = np.random.randint(self.model.action_space)
                state_next, reward, done, info = self.env.step(action)
                state_next, stacked_frames = self.stack_frames(stacked_frames, state_next, False)
                if memory:
                    memory.store((state, action, reward, state_next, done))
                state = state_next
                total_reward += reward
                step += 1
                if step > self.MAX_STEP:
                    done = True
            
            if render:
                self.env.close()
            
            avg_rewards = avg_rewards + 1/(i) * (total_reward - avg_rewards)
            steps.append(step)
        
        print("The average rewards for {} runs is {}".format(n_episodes, avg_rewards))
            
        return steps, avg_rewards


    def run(self):
        
        total_steps = 0
        losses = []
        rewards = []
        steps = []
        start_time = time.time()
        
        
        N = 5 
        for i in range(0, N):
            
            # Initialise training 
            state = self.env.reset()
            stacked_frames = deque([np.zeros(self.frame_size, dtype = np.int) for i in range(0,self.stack_size)], maxlen=self.stack_size)
            state, stacked_frames = self.stack_frames(stacked_frames, state, True)
            total_reward = 0
            step = 0
            done = False
            
            while not done:
                
                # Take a step
                action = self.model.act(state, self.EXPLORATION_RATE)
                state_next, reward, done, _ = self.env.step(action)
                
                # Store the experience 
                state_next, stacked_frames = self.stack_frames(stacked_frames, state_next, False)
                self.memory.store((state, action, reward, state_next, done))
        
                # Forward to next state
                state = state_next
                
                # accumulate rewards
                total_reward += reward
                step += 1
                total_steps += 1
                
                # stops the episode by set max step
                if step > self.MAX_STEP:
                    done = True
                    print("Episode Done")
                    
                # Update model every UPDATE_STEP (5000)
                if total_steps % self.UPDATE_MODEL_STEP == 0:
                    self.model.update_model()     
            
                ### LEARNING PROCEDURE ###
                if self.memory.memory_tree.capacity_filled >= self.model.BATCH_SIZE:
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
            steps.append(step)
            rewards.append(total_reward)
            self.test(1, model = self.model)
            
            # print information at the end of the episode
            print("Episode {}, exploration rate: {:.4f}, final rewards: {}, final loss is {:.4f}, Time elapsed: {:.4f}"\
                  .format(i+1, self.EXPLORATION_RATE, total_reward, loss, time.time() - start_time))
            start_time = time.time()
            
        # save model
        # self.model.DQNetwork.save(self.path) 
    

if __name__ == "__main__":
    agent = Agent()
    agent.run()
