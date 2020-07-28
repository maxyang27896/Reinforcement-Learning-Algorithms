## Cartpole
The objective of the Cartpole problem is to balance a inverted pendulum on a cart, given the input states [position, speed, angle, angular velocity] and the control is (left, right). 

<img src="/img/cartpole.jpg" width="250">

Methods have been applied:
* Q-table learning - An obervation wrapper was applied to discretise the continous state space and then a Q-table learning algorithm was applied. 
* Deep Q Learning - The input state was kept as continous inputs and a deep Q network (DQN) was used to map state to action. The objective of the neural network is to approximate the Q values of each action at each state that the network sees and the agent chooses the action with the highest Q values as the optimal policy. 
