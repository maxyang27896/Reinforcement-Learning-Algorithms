# Reinforcement Learning Algorithms
This repository contains a series of reinforcement learning algorithms applied to different environments in OpenAI Gym aimed at training agents to play games optimally.

## Content
* [Taxi with TD Q table learning](#taxi-v3)
* Frazen lake with Monte Carlo Q table learning 
* Cartpole with TD Q table learning
* Cartpole with Deep Q Learning 
* Space invader with Dueling Double Deep Q Network and Prioritised Experience Replay

## Frozen Lake
Inside this grid environment, the objective is to move the agent from a fixed starting location to aa fixed destination. There are holes in the grid where it would cause the agent to lose the game. There are 16 states in this 4x4 grid containing information of the location of the agent and the actions are (up, down, left, right). This is nondeterministic environment where each forward action has a probability of 33% of achieving the desired effect. 

<img src="./img/Frozen_Lake.png" width="250">

A Monte carlo reinformcement learning algorithm was implemented to learn a optimal Q-table to solve the frozen lake problem. The agent was able to reach the destination for ~60% of the time with the learned derived from the learned Q-table. 

## Cartpole
The objective of the Cartpole problem is to balance a inverted pendulum on a cart, given the input states [position, speed, angle, angular velocity] and the control is (left, right). 

<img src="/img/cartpole.jpg" width="250">

Two methods have been applied:
* Q-table learning - An obervation wrapper was applied to discretise the continous state space and then a Q-table learning algorithm was applied. 
* Deep Q Learning - The input state was kept as continous inputs and a deep Q network (DQN) was used to map state to action. The objective of the neural network is to approximate the Q values of each action at each state that the network sees and the agent chooses the action with the highest Q values as the optimal policy. 

## Space Invader
The objective is to train a deep Q network to play the Atari space invader game and obtain the highest score as possible. The input state is a stacked frames of size 4 frame that has been processed and the output consists of 6 actions of the game. A convolutional neural network has been used to model the state to action function. 

<img src="/img/poster.jpg" width="150"> <img src="/img/dqn.jpg" width="600">

* Double DQN - uses an additional target network that predicts the Q values of the next state which is frozen during training. This allows the main network to train stably without the target network being altered with it. The target network is then updated every N iterations of training. 
* Dueling DQN - a special type of neural network architecture that produces two streams to model the values of being in a state and the value of each action provided that the agent is in that state. This helps the agent to learn quicker by undertanding good states and bad states to be in before considering the actions to take. 
* Priority experience replay - a memory tree was implemented that samples preivous experiences based on priority value which is calculated by TD error between the predicted Q value and actual Q value. The prioritises experiences that resulted in poor model prediction therefore those that the model has the most to learn from. 
