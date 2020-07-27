# Reinforcement Learning Algorithms
This repository contains a series of reinforcement learning algorithms applied to different environments in OpenAI Gym aimed at training agents to play games optimally.

## Taxi V3
Inside this grid environment, the objective is for the taxi to pick up the passenger from a location and drop him to the desired destination. There are 500 possible states in this 5x5 grid containing information of location of taxi and location of passenger and 6 actions for the taxi (up, down, left, right, pickup and dropoff).

<img src="./img/taxi.png" width="400" height="790">

A Q-table learning from the bellmen equation was implemented to find the optimal action in this taxi environment. The output is a learned Q-table for the optimal action to take for every possible state that the taxi and passenger can be in. 

## Frozen Lake
Inside this grid environment, the objective is to move the agent from a fixed starting location to aa fixed destination. There are holes in the grid where it would cause the agent to lose the game. There are 16 states in this 4x4 grid containing information of the location of the agent and the actions are (up, down, left, right). This is nondeterministic environment where each forward action has a probability of 33% of achieving the desired effect. 
![frozenlake](./img/Frozen_Lake.png)
A Monte carlo reinformcement learning algorithm was implemented to learn a optimal Q-table to solve the frozen lake problem. The agent was able to reach the destination for ~60% of the time with the learned derived from the learned Q-table. 

## Cartpole
The objective of the Cartpole problem is to balance a inverted pendulum on a cart, given the input states [position, speed, angle, angular velocity] and the control is (left, right). 
![Cartpole](./img/cartpole.jpg)
* Q-table learning - An obervation wrapper was applied to discretise the continous state space and then a Q-table learning algorithm was applied. 
* Deep Q Learning - The input state was kept as continous inputs and a neural network (Deep Q Network) was used to model state to action function. The DQN was optimised for maximum rewards. 

## Space Invader
