# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:13:57 2020

@author: MY2

Asychronous Advantage Actor Critic with variable N step return. 

"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import threading

# Clear any previous graph
tf.reset_default_graph()

# Define network            
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
            
            # Calculate the advantage with N step discounted rewards
            self.advantage = self.discounted_rewards_ - self.value
            
            # Actor Loss
            self.neg_log_prob = - tf.log(tf.reduce_sum(tf.multiply(self.action_distribution, self.actions_), axis=1, keep_dims=True) + 1e-10)
            self.actor_loss = self.neg_log_prob * tf.stop_gradient(self.advantage)
            
            # Critic Loss 
            self.critic_loss  = 0.5 * tf.square(self.advantage)
            
            # Entropy Loss
            self.entropy = 0.01 * tf.reduce_sum(self.action_distribution * tf.log(self.action_distribution + 1e-10), axis=1, keep_dims=True)
            
            # Totol loss
            self.loss_total = tf.reduce_mean(self.actor_loss + self.critic_loss + self.entropy)
            
            # Optimiser
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=.99).minimize(self.loss_total)
            
            
# Define each agent for training in their subsequent environments
class Agent(threading.Thread):
    def __init__(self, model, sess, thread_index, A3C_brain):
        threading.Thread.__init__(self)
        self.thread_index = thread_index

        # The brain class contains all the shared resources between each thread
        self.A3C_brain = A3C_brain
        
        # define model
        self.model = model
        self.sess = sess
        
        self.env = gym.make("CartPole-v1")
        self.memory_mb = []
        self.R = 0
        
    def run(self):
        
        # Initialise loss to print
        loss = 0
        
        # run for N episodes
        while self.A3C_brain.GLOBAL_EPISODE_NUM  < A3C_brain.MAX_TRAIN_STEPS or self.A3C_brain.GLOBAL_STOP_SIGNAL == True:
            
            # Initialise episode
            state = self.env.reset()
            terminal = False
            step = 0
            
            
            while not terminal:
                step += 1
                
                # Take a step
                action = self.act(state)
                state_next, reward, terminal, info = self.env.step(action)
                reward = reward if not terminal else -reward
                
                # store rewards, states and actions
                action_ = np.zeros(self.A3C_brain.action_size)
                action_[action] = 1 
                
                # Store experience to memory minibatch
                self.memory_mb.append((state, action_, reward, state_next, terminal))
                self.R = (self.R + reward * self.A3C_brain.GAMMA ** self.A3C_brain.N_STEP_RETURN)/self.A3C_brain.GAMMA   
                
                # Get N step discounted rewards and clear memory if its terminal state
                if terminal:
                    while len(self.memory_mb) > 0:
                        s, a, _, _, _ = self.memory_mb[0]
                        _, _, _, s_next, done = self.memory_mb[-1]
                        self.train_push(s, a, self.R, s_next, done)
                        self.R = (self.R - self.memory_mb[0][2]) / self.A3C_brain.GAMMA
                        self.memory_mb.pop(0)
                    self.R = 0
                    
                    # Print training progress if end of episode
                    with lock:
                        print("Episode: {}, Score: {}, loss: {:0.4f}, thread: {}".format(self.A3C_brain.GLOBAL_EPISODE_NUM, step, loss, self.thread_index))
                        self.A3C_brain.GLOBAL_EPISODE_NUM += 1
                        self.A3C_brain.steps.append(step)
                        self.A3C_brain.losses.append(loss)
                        
                # N step discounted rewards
                if len(self.memory_mb) >= self.A3C_brain.N_STEP_RETURN:
                    s, a, _, _, _ = self.memory_mb[0]
                    _, _, _, s_next, done = self.memory_mb[-1]
                    self.train_push(s, a, self.R, s_next, done)
                    self.R = self.R - self.memory_mb[0][2]
                    self.memory_mb.pop(0)                 
                    
                # Advance next step
                state = state_next
                
                # If training batch has been filled, then train and clear memoory
                with lock:
                    if len(self.A3C_brain.train_mb) >= self.A3C_brain.BATCH_SIZE:
                        loss = self.model_train()
                        
        
        # Print after the thread has finished training
        print("Thread: {} stopped".format(self.thread_index))
        
    def train_push(self, s, a, r, s_next, done):
        '''
        Function to push the each experience to the A3C_brain training batch 
        '''
        self.A3C_brain.train_mb.append((s, a, r, s_next, done))
        
    def model_train(self):
        '''
        Function: To train the global A3C_brain model when the training batch has been filled
        '''
        # Extract each parameter from training batch memory
        states_mb = np.zeros([self.A3C_brain.BATCH_SIZE, self.A3C_brain.input_shape])
        actions_mb = np.zeros([self.A3C_brain.BATCH_SIZE, self.A3C_brain.action_size], dtype = int)
        rewards_mb = np.zeros(self.A3C_brain.BATCH_SIZE)
        states_next_mb = np.zeros([self.A3C_brain.BATCH_SIZE, self.A3C_brain.input_shape])
        done_mb = np.zeros(self.A3C_brain.BATCH_SIZE)
        for i in range(0, self.A3C_brain.BATCH_SIZE):
            states_mb[i] = self.A3C_brain.train_mb[i][0]
            actions_mb[i] = self.A3C_brain.train_mb[i][1]
            rewards_mb[i] = self.A3C_brain.train_mb[i][2]
            states_next_mb[i] = self.A3C_brain.train_mb[i][3]
            done_mb[i] = self.A3C_brain.train_mb[i][4]
        
        # Clear training batch memory 
        self.A3C_brain.train_mb = []
        
        # Train on N step discounted rewards
        discounted_rewards_mb = []
        value = self.sess.run([self.model.value], feed_dict={self.model.inputs_: np.array(states_next_mb)})[0]
        for i in range(0, self.A3C_brain.BATCH_SIZE):
            if done_mb[i]:
                discounted_rewards_mb.append(np.array([rewards_mb[i]]))
            else:
                discounted_rewards_mb.append(rewards_mb[i] + self.A3C_brain.GAMMA ** self.A3C_brain.N_STEP_RETURN * value[i])
        
        _, loss = self.sess.run([self.model.optimizer, self.model.loss_total],
                                        feed_dict={self.model.inputs_: states_mb,
                                                    self.model.actions_: actions_mb,
                                                    self.model.discounted_rewards_: discounted_rewards_mb})
        
        return loss

    def act(self, state, greedy=False):
        '''
        Function to return action based on probability distribution
        '''
    
        action_distribution = self.sess.run(self.model.action_distribution, feed_dict={self.model.inputs_: state[tf.newaxis,...]})
        if greedy:
            action = np.argmax(action_distribution[0])
        else:
            action = np.random.choice(self.A3C_brain.action_size, p = action_distribution[0])
        return action

    def predict(self, state):
        return self.sess.run(self.model.action_distribution, feed_dict={self.model.inputs_: state[tf.newaxis,...]})
    
    
    def test(self, n_episodes, model=True, with_limit=False, render=False):
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
                    action = self.act(state, greedy=True)
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


# The training procedure 
class A3C_brain:
    def __init__(self, input_shape, action_size):
        
        # Global parameters
        self.input_shape = input_shape
        self.action_size = action_size
        self.LEARNING_RATE = 0.005
        self.GAMMA = 0.95
        self.N_STEP_RETURN = 4
        self.BATCH_SIZE = 32
        self.THREADS = 8
        self.MAX_TRAIN_STEPS = 5000
        
        self.GLOBAL_EPISODE_NUM = 0
        self.GLOBAL_STOP_SIGNAL = False
        
        self.train_mb = []
        self.steps = []
        self.losses = [] 
        
        # Initiate Model
        self.model = ACNetworks(input_shape, action_size, self.LEARNING_RATE, "A3CNetwork")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer()) 
        

    def train(self):
        agents = [Agent(self.model, self.sess, index, self) for index in range(self.THREADS)]
        
        for agent in agents:
            agent.start()
            
        # time.sleep(30)
        # self.stop()
            
        for agent in agents:
            agent.join()
            
        print("Training Stopped")
    
    def test_model(self):
        '''
        Test the trained model in an environment

        '''
        agent = Agent(self.model, self.sess, 0, self)
        test_reward = agent.test(10, model=True, with_limit=False, render=False)
        print("Final test rewards, ", test_reward)
    
    def stop(self):
        self.GLOBAL_STOP_SIGNAL = True
        print("Global stop activated")
        
    def plot_result(self):
    
        # plot rewards
        plt.plot(self.steps)
        plt.ylabel('Scores')
        plt.xlabel('Runs')
        plt.legend(['Average Rewards'])
        plt.show()
        
        plt.plot(self.losses)
        plt.ylabel(' Total Loss')
        plt.xlabel('Steps')
        plt.show()

if __name__ == "__main__":
    
    # create global lock
    lock = threading.Lock() 
    
    # Get environment variables
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Train brain
    A3C_brain = A3C_brain(observation_space, action_size)
    A3C_brain.train()
    A3C_brain.test_model()
    A3C_brain.plot_result()