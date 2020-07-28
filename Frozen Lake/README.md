## Frozen Lake
Inside this grid environment, the objective is to move the agent from a fixed starting location to a fixed destination. There are holes in the grid where it would cause the agent to lose the game. There are 16 states in this 4x4 grid containing information of the location of the agent and the actions are (up, down, left, right). This is non-deterministic environment where each forward action has a probability of 33% of achieving the desired effect. 

<img src="/img/Frozen_Lake.png" width="250">

A Monte carlo reinformcement learning algorithm was implemented to learn a optimal Q-table to solve the frozen lake problem. The agent was able to reach the destination for ~60% of the time with the learned derived from the learned Q-table. 
