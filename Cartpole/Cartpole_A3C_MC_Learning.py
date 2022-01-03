# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:47:42 2020

@author: MY2
"""
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import threading

# Clear any previous graph
tf.reset_default_graph()

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
            

# Define each agent for training in their subsequent environments
class Agent(threading.Thread):
    def __init__(self, model, sess, thread_index, gamma, global_class):
        threading.Thread.__init__(self)
        
        self.global_class = global_class
        
        self.model = model
        self.sess = sess
        self.thread_index = thread_index
        
        self.env = gym.make("CartPole-v1")
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        
        self.GAMMA = gamma
        
        
    def run(self):
        
        # run for N episodes
        while self.global_class.global_episode  < 500 or self.global_class.gobal_stop_signal == True:
            
            # Initialise episode
            state = self.env.reset()
            terminal = False
            step = 0
            
            # For storing 
            states_mb = []
            actions_mb = []
            episode_rewards = []
            
            while not terminal:
                step += 1
                
                # Take a step
                action = self.act(state)
                state_next, reward, terminal, info = self.env.step(action)
                reward = reward if not terminal else -reward
                
                # store rewards, states and actions
                episode_rewards.append(reward)
                states_mb.append(state)
                action_ = np.zeros(self.action_space)
                action_[action] = 1
                actions_mb.append(action_)
                 
                '''Monte Carlo  train'''
                if terminal:
                    discounted_rewards_mb = self.get_discounted_rewards(episode_rewards)
                    
                    actions_mb = np.array(actions_mb)
                    states_mb = np.array(states_mb)
                    discounted_rewards_mb = np.expand_dims(discounted_rewards_mb, axis=-1)
                    
                    # Fit the data on lock
                    with lock:
                        _, actor_loss, _, critic_loss = self.sess.run([self.model.actor_optimiser, self.model.actor_loss, self.model.critic_optimiser, self.model.critic_loss],
                                                                feed_dict={self.model.inputs_: states_mb,
                                                                            self.model.actions_: actions_mb,
                                                                            self.model.discounted_rewards_: discounted_rewards_mb})
                                        
                        print("Episode: {}, Score: {}, actor loss: {:0.2f}, critic loss: {:0.2f}, thread: {}".format(self.global_class.global_episode, step, actor_loss, critic_loss, self.thread_index))
                    
                        self.global_class.rewards.append(sum(episode_rewards))
                        self.global_class.steps.append(step)
                        self.global_class.actor_losses.append(actor_loss)
                        self.global_class.critic_losses.append(critic_loss)
                        self.global_class.global_episode += 1
   
                
                state = state_next
        
        print("Thread: {} stopped".format(self.thread_index))
        
        
        
    def get_discounted_rewards(self, episode_rewards):
        '''
        Function returns discounte rewards at the end of an episode
    
        '''
        discounted_rewards_mb = np.zeros(len(episode_rewards))
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.GAMMA +  episode_rewards[i]
            discounted_rewards_mb[i] = cumulative
            
        # Normalise the discounted rewards
        mean = np.mean(discounted_rewards_mb)
        std = np.std(discounted_rewards_mb)
        discounted_rewards_mb = (discounted_rewards_mb - mean) / (std)
        return discounted_rewards_mb

    def act(self, state, greedy=False):
        '''
        Function to return action based on probability distribution
        '''
    
        action_distribution = self.sess.run(self.model.action_distribution, feed_dict={self.model.inputs_: state[tf.newaxis,...]})
        if greedy:
            action = np.argmax(action_distribution[0])
        else:
            action = np.random.choice(self.action_space, p = action_distribution[0])
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
        
        self.LEARNING_RATE = 0.005
        self.gamma = 0.95
        self.threads = 8
        
        self.model = ACNetworks(input_shape, action_size, self.LEARNING_RATE, "ACNetwork")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer()) 
        
        self.global_episode = 0
        self.gobal_stop_signal = False
        
        self.rewards = []
        self.steps = []
        self.actor_losses = []
        self.critic_losses = []

    def train(self):
        agents = [Agent(self.model, self.sess, i, self.gamma, self) for i in range(self.threads)]
        
        for agent in agents:
            agent.start()
            
        # time.sleep(30)
        # self.stop()
            
        for agent in agents:
            agent.join()
            
        print("Training Stopped")
        
        # state = np.random.rand(4)
        # for agent in agents:
        #     print(agent.predict(state))
        
        # Test final model
        test_reward = agent.test(10, model=True, with_limit=False, render=False)
        print("Final test rewards, ", test_reward)
        
    def stop(self):
        self.gobal_stop_signal = True
        print("Global stop activated")
        
    def plot_result(self):
    
        # plot rewards
        plt.plot(self.rewards)
        plt.plot(self.steps)
        plt.ylabel('Scores')
        plt.xlabel('Runs')
        plt.legend(['Average Rewards'])
        plt.show()
        
        # plot loss
        plt.plot(self.actor_losses)
        plt.ylabel(' Actor Loss')
        plt.xlabel('Steps')
        plt.show()
        
        plt.plot(self.critic_losses)
        plt.ylabel(' Critic Loss')
        plt.xlabel('Steps')
        plt.show()
        
if __name__ == "__main__":
    
    lock = threading.Lock() 
    
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    A3C_brain = A3C_brain(observation_space, action_space)
    A3C_brain.train()
    A3C_brain.plot_result()