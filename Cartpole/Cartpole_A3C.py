# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:54:37 2020

@author: MY2

Advantage Actor Critic N-Step return without concurrency
"""


import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time


# create the envirnment
env = gym.make("CartPole-v1")
# env.seed(1)
# np.random.seed(1)
# tf.set_random_seed(1)

# Get state spaces
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

GAMMA = 0.95
LEARNING_RATE = 0.005


class ACNetworks:
     def __init__(self, state_size, action_size, learning_rate_actor, learning_rate_critic, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
    
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, 4], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 2], name="actions")
            self.discounted_rewards_ = tf.placeholder(tf.float32, [None,1], name="discounted_rewards")

            self.fc_shared = tf.layers.dense(inputs = self.inputs_,
                                                units = 24,
                                                activation=tf.nn.relu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name = "fc_shared")
            
            with tf.variable_scope('fc_shared', reuse=True):
                self.fc_shared_w = tf.get_variable('kernel')
                self.fc_shared_b = tf.get_variable('bias')
            
            # Actor stream
            self.fc_actor = tf.layers.dense(inputs = self.fc_shared,
                                                units = 24,
                                                activation= tf.nn.relu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name = "fc_actor")
            
            with tf.variable_scope('fc_actor', reuse=True):
                self.fc_actor_w = tf.get_variable('kernel')
                self.fc_actor_b = tf.get_variable('bias')
            
            # Output action probabilities           
            self.action_distribution = tf.layers.dense(inputs = self.fc_actor,
                                                units = self.action_size,
                                                activation= tf.nn.softmax,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            
            # Value Stream 
            self.fc_value = tf.layers.dense(inputs = self.fc_shared,
                                    units = 24,
                                    activation= tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name = "fc_value")
            
            with tf.variable_scope('fc_value', reuse=True):
                self.fc_value_w = tf.get_variable('kernel')
                self.fc_value_b = tf.get_variable('bias')
            
            # output value
            self.value = tf.layers.dense(inputs = self.fc_value,
                                    units = 1,
                                    activation= None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            # Calculate the advantage with N step discounted rewards
            self.advantage = self.discounted_rewards_ - self.value
            
            # Actor optimisation
            self.prob = tf.reduce_sum(tf.multiply(self.action_distribution, self.actions_), axis=1, keep_dims=True)
            self.neg_log_prob = -tf.log(self.prob)
            self.actor_loss = tf.reduce_mean(self.neg_log_prob * tf.stop_gradient(self.advantage))
            self.actor_optimiser = tf.train.AdamOptimizer(0.001).minimize(self.actor_loss)
            
            # Critic optimsation
            self.critic_loss = tf.reduce_mean(tf.square(self.advantage))
            self.critic_optimiser = tf.train.AdamOptimizer(0.001).minimize(self.critic_loss)
            
         
            
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

            
def act(state, greedy=False):
    action_distribution = sess.run(ACNetwork.action_distribution, feed_dict={ACNetwork.inputs_: state[tf.newaxis,...]})
    if greedy:
        action = np.argmax(action_distribution[0])
    else:
        action = np.random.choice(action_space, p = action_distribution[0])
    return action


def train_push(s, a, r, s_next, done):
    states_mb.append(s)
    actions_mb.append(a)
    discounted_rewards_mb.append(r)
    states_next_mb.append(s_next)
    done_mb.append(done)
    


# Create the graph
tf.reset_default_graph()
ACNetwork = ACNetworks(observation_space, action_space, 0.005, 0.001, "ACNetwork")    
sess = tf.Session()
sess.run(tf.global_variables_initializer()) 

# For logging
log_path = r"C:\Projects\Personal Projects\Saved_Sessions\Cartpole\tensorboard\1"
writer = tf.summary.FileWriter(log_path)
tf.summary.scalar("actor_loss", ACNetwork.actor_loss)
tf.summary.scalar("critic_loss", ACNetwork.critic_loss)
tf.summary.histogram("value", ACNetwork.value)
tf.summary.histogram("advantage", ACNetwork.advantage)
tf.summary.histogram("action_probablity", ACNetwork.action_distribution)
tf.summary.histogram("fc_shared/Weights", ACNetwork.fc_shared_w)
tf.summary.histogram("fc_shared/Bias", ACNetwork.fc_shared_b)
tf.summary.histogram("fc_shared/Activation", ACNetwork.fc_shared)
# tf.summary.histogram("fc_shared/Gradient", ACNetwork.fc_shared_grad)
tf.summary.histogram("fc_value/Weights", ACNetwork.fc_value_w)
tf.summary.histogram("fc_value/Bias", ACNetwork.fc_value_b)
tf.summary.histogram("fc_value/Activation", ACNetwork.fc_value)
# tf.summary.histogram("fc_value/Gradient", ACNetwork.fc_value_grad)
tf.summary.histogram("fc_actor/Weights", ACNetwork.fc_actor_w)
tf.summary.histogram("fc_actor/Bias", ACNetwork.fc_actor_b)
tf.summary.histogram("fc_actor/Activation", ACNetwork.fc_actor)
# tf.summary.histogram("fc_actor/Gradient", ACNetwork.fc_actor_grad)
write_op = tf.summary.merge_all()
        

# run for N episodes
N_STEP_TRAIN = 4
BATCH_SIZE =  16
actor_loss = 0
R = 0
memory = []

N = 5000
rewards = []
steps = []
actor_losses = []
critic_losses = []
gl_norms = []
actions_list = []

for episode in range(0, N):
    
    # Initialise episode
    state = env.reset()
    terminal = False
    step = 0
    
    # For storing 
    states_mb = []
    actions_mb = []
    discounted_rewards_mb = []
    states_next_mb = []
    done_mb = []
    
    episode_rewards = 0
    
    while not terminal:
        step += 1
        # Take a step
        action = act(state)
        actions_list.append(action)
        state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -reward
        
        # store values
        episode_rewards += reward
        action_ = np.zeros(action_space)
        action_[action] = 1
        
        # store to memory
        memory.append((state, action_, reward, state_next, terminal))
        R = (R + reward * GAMMA ** N_STEP_TRAIN)/GAMMA  
        
        # Get N step discounted rewards and clear memory if its terminal state
        if terminal:
            while len(memory) > 0:
                s, a, _, _, _ = memory[0]
                _, _, _, s_next, done = memory[-1]
                train_push(s, a, R, s_next, done)
                R = (R - memory[0][2]) / GAMMA
                memory.pop(0)
            R = 0
        
        # N step discounted rewards
        if len(memory) >= N_STEP_TRAIN:
            s, a, _, _, _ = memory[0]
            _, _, _, s_next, done = memory[-1]
            train_push(s, a, R, s_next, done)
            R = R - memory[0][2]
            memory.pop(0)        
            
        state = state_next
        
        # Train model if training batch size filled
        if len(states_mb) >= BATCH_SIZE:
            target_mb = []
            value = sess.run([ACNetwork.value], feed_dict={ACNetwork.inputs_: np.array(states_next_mb)})[0]
            for i in range(0, len(states_next_mb)):
                if done_mb[i]:
                    target_mb.append(np.array([discounted_rewards_mb[i]]))
                else:
                    target_mb.append(discounted_rewards_mb[i] + GAMMA ** N_STEP_TRAIN * value[i])
                    
            _, actor_loss, _, critic_loss = sess.run([ACNetwork.actor_optimiser, ACNetwork.actor_loss, ACNetwork.critic_optimiser, ACNetwork.critic_loss],
                                                    feed_dict={ACNetwork.inputs_: np.array(states_mb),
                                                                ACNetwork.actions_: np.array(actions_mb),
                                                                ACNetwork.discounted_rewards_: np.array(target_mb)})
            
            # summary = sess.run(write_op, feed_dict={ACNetwork.inputs_: np.array(states_mb),
            #                                         ACNetwork.actions_: np.array(actions_mb),
            #                                         ACNetwork.discounted_rewards_: np.array(target_mb)})
            # writer.add_summary(summary, episode)
            # writer.flush()

            # Empty training batch
            states_mb = []
            actions_mb = []
            discounted_rewards_mb = []
            states_next_mb = []
            done_mb = []
                
        if terminal and abs(actor_loss):
            print("Episode: {}, Score: {}, actor loss: {}, critic loss: {}".format(episode, step, actor_loss, critic_loss))
            rewards.append(episode_rewards)
            steps.append(step)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
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