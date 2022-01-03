## Space Invader
The objective is to train a deep Q network to play the Atari space invader game and obtain the highest score as possible. The input state is a stacked frames of size 4 frame that has been processed and the output consists of 6 actions of the game. A convolutional neural network has been used to model the state to action function. 

<img src="/img/poster.jpg" width="150"> <img src="/img/dqn.jpg" width="600">

* Double DQN - uses an additional target network that predicts the Q values of the next state which is frozen during training. This allows the main network to train stably without the target network being altered with it. The target network is then updated every N iterations of training. 
* Dueling DQN - a special type of neural network architecture that produces two streams to model the values of being in a state and the value of each action provided that the agent is in that state. This helps the agent to learn quicker by undertanding good states and bad states to be in before considering the actions to take. 
* Priority experience replay - a memory tree was implemented that samples preivous experiences based on priority value which is calculated by TD error between the predicted Q value and actual Q value. The prioritises experiences that resulted in poor model prediction therefore those that the model has the most to learn from. 