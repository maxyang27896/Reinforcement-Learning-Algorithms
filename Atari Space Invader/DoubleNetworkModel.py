# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:31:32 2020

@author: MY2
"""
import shutil
import os
import tensorflow as tf
import numpy as np
from D3QNetwork import DQNetworks

class Model:
    
    def __init__(self, input_shape, action_space):
        self.input_shape = input_shape
        self.action_space = action_space
        self.LOG_FREQUENCY = 20
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 64
        self.GAMMA = 0.9

        self.DQNetwork = DQNetworks(input_shape, action_space, self.LEARNING_RATE, "DQN")
        self.TargetDQNetwork = DQNetworks(input_shape, action_space, self.LEARNING_RATE, "TargetDQN")
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer()) 
        
        self.saver = tf.train.Saver()
        self.path =  "./Models/model.ckpt"
        
        # For logging
        log_path = "./tensorboard/dddqn/1"
        if os.path.isdir(log_path):
            shutil.rmtree(log_path)
        self.writer = tf.summary.FileWriter(log_path)
        tf.summary.scalar("Loss", self.DQNetwork.loss)
        tf.summary.scalar("Mean_Target", self.DQNetwork.mean_target)
        tf.summary.scalar("Mean_Predict", self.DQNetwork.max_Q_mean_pred)
        tf.summary.histogram("Conv1/Weights", self.DQNetwork.conv1w)
        tf.summary.histogram("Conv1/Bias", self.DQNetwork.conv1b)
        tf.summary.histogram("Conv1/Activation", self.DQNetwork.conv1_out)
        tf.summary.histogram("Conv2/Weights", self.DQNetwork.conv2w)
        tf.summary.histogram("Conv2/Bias", self.DQNetwork.conv2b)
        tf.summary.histogram("Conv2/Activation", self.DQNetwork.conv2_out)
        tf.summary.histogram("Conv3/Weights", self.DQNetwork.conv3w)
        tf.summary.histogram("Conv3/Bias", self.DQNetwork.conv3b)
        tf.summary.histogram("Conv3/Activation", self.DQNetwork.conv3_out)
        tf.summary.histogram("Value/Weights", self.DQNetwork.value_fcw)
        tf.summary.histogram("Value/Bias", self.DQNetwork.value_fcb)
        tf.summary.histogram("Advantage/Weights", self.DQNetwork.advantage_fcw)
        tf.summary.histogram("Advantage/Bias", self.DQNetwork.advantage_fcb)
        tf.summary.histogram("Q_values/Q_predict", self.DQNetwork.output)
        tf.summary.histogram("Q_values/Q_target", self.DQNetwork.target_Q)
        tf.summary.histogram("Q_values/Q_predict_max", self.DQNetwork.max_Q_pred)
        tf.summary.histogram("Q_values/arg_Q_predict_max", self.DQNetwork.argmax_Q_pred)
        self.write_op = tf.summary.merge_all()
        
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
        states_mb = np.zeros([self.BATCH_SIZE, *self.input_shape])
        actions_mb = np.zeros([self.BATCH_SIZE, self.action_space], dtype = int)
        rewards_mb = np.zeros(self.BATCH_SIZE)
        next_states_mb = np.zeros([self.BATCH_SIZE,*self.input_shape])
        dones_mb = np.zeros(self.BATCH_SIZE)
        for i in range(0,len(batch)):
            states_mb[i] = batch[i][0]
            actions_mb[i][batch[i][1]] = 1
            rewards_mb[i] = batch[i][2]
            next_states_mb[i] = batch[i][3]
            dones_mb[i] = batch[i][4]
    
        targets_mb = np.zeros(self.BATCH_SIZE)
    
        # Model predict the Q-values
        q_next_state = self.sess.run(self.DQNetwork.output, feed_dict = {self.DQNetwork.inputs_: next_states_mb})
        q_target_next_state = self.sess.run(self.TargetDQNetwork.output, feed_dict = {self.TargetDQNetwork.inputs_: next_states_mb})
    
        # Calculate target Q 
        for i in range(0, len(batch)):
            terminal = dones_mb[i]
            if terminal:
                targets_mb[i] = rewards_mb[i]
            else:
                action = np.argmax(q_next_state[i])
                targets_mb[i] = rewards_mb[i] + self.GAMMA * q_target_next_state[i][action]
    
        # Fit the data 
        _, loss, absolute_errors = self.sess.run([self.DQNetwork.optimizer, self.DQNetwork.loss, self.DQNetwork.abs_TD_error],
                            feed_dict={self.DQNetwork.inputs_: states_mb,
                                       self.DQNetwork.target_Q: targets_mb,
                                       self.DQNetwork.actions_: actions_mb,
                                       self.DQNetwork.IS_weights: IS_weights})
        
            # Write TF Summaries
        if total_steps % self.LOG_FREQUENCY == 0:
            summary = self.sess.run(self.write_op, feed_dict={self.DQNetwork.inputs_: states_mb,
                                                        self.DQNetwork.target_Q: targets_mb,
                                                        self.DQNetwork.actions_: actions_mb,
                                                        self.DQNetwork.IS_weights: IS_weights})
            self.writer.add_summary(summary, total_steps)
            self.writer.flush()

        return loss, absolute_errors
        
    def update_model(self):
        '''
        FUNCTION: update the target model
        '''
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQN")
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetDQN")
        updated_weights = [to_var.assign(from_var) for from_var, to_var in zip(from_vars, to_vars)]
        self.sess.run(updated_weights)
        print("Target network updated!")
    
    def save_model(self):
        '''
        FUNCTION: save main DQN model to path
        '''
        self.saver.save(self.sess, self.path)
        print("DQNetwork saved!")
    
    def load_model(self):
        '''
        FUNCTION: save main DQN model to path
        '''
        self.saver.restore(self.sess, self.path)
        print("Succesfully loaded network!")
