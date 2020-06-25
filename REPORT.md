# Summary

In this project I implemeted and trained two RL agents to play tennis. 
If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play and get as many rewards as possible.

# Implementation 
## Algorithm
This project utilised the `DDPG` (Deep Deterministic Policy Gradient) architecture outlined in the 
[DDPG-Bipedal Udacity project repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal).

The extended `DDPG` algorithm `MADDPG` is implemented in the [maddpg.py](maddpg.py) file. Learning of continuous actions requires an actor and a 
critic model. The models for `Actor` and `Critic` are implemented in the [networks.py](networks.py).

The agents were trained using shared actor and critic networks, as well as a shared replay buffer.
The actor model learns to predict an action vector while the critic model 
learns Q values for state-action pairs. The Replay Buffer is implemented in the [replay_buffer.py](replay_buffer.py).
It also distinguishes between online and target models for both actor and critic, similar to 
fixed Q-targets and double DQN technique. Online models are updated by minimizing loses while target models are updated
through soft update, i.e. online model parameters values are partially transferred to target models. This helps to avoid 
overestimation of Q-values and makes the training more stable.

The core of MADDPG algorithm is implemented in the Agent class of the [maddpg.py](maddpg.py). An important aspect is the noise added to the actions to allow exploration of the the action space. The 
noise is generated through the `Ornstein–Uhlenbeck` process, which is a stochastic process that is both Gaussian and Markov, 
drifting towards the mean in long-term. This code of `Ornstein–Uhlenbeck` process can be found n the [noise.py](noise.py). The noise was added only in the first 300 episodes.


## Architectures

### Actor Network 

1. State input (33 units * 2)
2. Hidden layer (256 units) with ReLU activation and batch normalization
3. Hidden layer (256 units) with ReLU activation and batch normalization
4. Action output (4 units) with tanh activation

### Critic Network 

1. State input (33 units * 2)
2. Hidden layer (256 nodes) with ReLU activation and batch normalization
3. Action input (4 units)
4. Hidden layer with inputs from layers 2 and 3 (128 nodes) with ReLU activation and batch normalization
5. Q-value output (1 node)

### Hyperparameters

Almost all the hyperparameters are defined in the [constants.py](constants.py):

 Hyperparameter | Value | Description |
|---|---:|---|
| Replay buffer size | 1e6 | Maximum size of experience replay buffer |
| Replay batch size | 256 | Number of experiences sampled in one batch |
| Actor hidden units | 256, 256 | Number of units in hidden layers of the actor model |
| Critic hidden units | 256, 256 | Number of units in hidden layers of the critic model |
| Actor learning rate | 1e-3 | Controls parameters update of the online actor model |
| Critic learning rate | 1e-3 | Controls parameters update of the online critic model |
| Target update mix | 1e-3 | Controls parameters update of the target actor and critic models |
| Update every N steps | 10 | The target model will be soft update every N steps.
| Discount factor | 0.99 | Discount rate for future rewards |
| Ornstein-Uhlenbeck, mu | 0 | Mean of the stochastic  process|
| Ornstein-Uhlenbeck, theta | 0.15 | Parameter of the stochastic process |
| Ornstein-Uhlenbeck, sigma | 0.2 | Standard deviation of the stochastic process |
| print every | 200 | How often to print average score during training |
| consecutive episodes | 100 | How many episodes at least SOLVED_SCORE must be achieved to solve the RL task
| solved score | 0.5 | Average score needed to solve the RL task |
| stop noise after episode | 300 | Stop adding noise after defined number of periods |
| Max episodes | 8000 | Maximum number of episodes to train |
| Max steps | 2e3 | Maximum number of timesteps per episode |


## Results

![results](images/scores.png)

```
Episode 200	Average Score: 0.01 best_score 0.20000000298023224
Episode 400	Average Score: 0.02 best_score 0.60000000894069676
Episode 600	Average Score: 0.05 best_score 0.6000000089406967
Episode 800	Average Score: 0.06 best_score 0.6000000089406967
Episode 1000	Average Score: 0.10 best_score 0.7000000104308128
Episode 1200	Average Score: 0.27 best_score 2.2000000327825546
Episode 1271	Average Score: 0.50 best_score 5.2000000774860386
Environment solved in 1271 episodes!	Average Score: 0.50

Total time took for training: 8.693374188741048 min.

```
## Possible extensions

- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Asynchronous Actor Critic](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
