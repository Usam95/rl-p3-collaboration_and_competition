


import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from constants import *

import importlib
import sys
importlib.reload(sys.modules['constants'])

from networks import *
from noise import OrnsteinUhlenbeck
from replay_buffer import ReplayBuffer 

class Agent(): 
    """ Interacts with and learns from the environment. """ 
    
    def __init__(self, state_size, action_size, fc1_units, fc2_units, num_agents):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(SEED)
        self.num_agents = num_agents
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, fc1_units, fc2_units).to(device)
        self.actor_target = Actor(state_size, action_size, fc1_units, fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, fc1_units, fc2_units ).to(device)
        self.critic_target = Critic(state_size, action_size, fc1_units, fc2_units ).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OrnsteinUhlenbeck((num_agents, action_size), SEED)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, SEED, device)
    
    def step(self, step, state, action, reward, next_state, done, agent_number):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward        
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory and if step > 10
        if (len(self.memory) > BATCH_SIZE) and (step % N_TIME_STEPS == 0):
            for n in range(N_LEARN_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, agent_number)

    def act(self, states, add_noise=True):
        """Returns actions for both agents as per current policy, given their respective states."""
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            # get action for each agent and concatenate them
            for agent_num, state in enumerate(states):
                action = self.actor_local(state).cpu().data.numpy()
                actions[agent_num, :] = action
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()
        

    def learn(self, experiences, gamma, agent_number):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
            
        # Construct next actions vector relative to the agent
        if agent_number == 0:
            actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        else:
            actions_next = torch.cat((actions[:,:2], actions_next), dim=1)
        # Compute Q targets for current states (y_i)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
       
        
        if agent_number == 0:
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def get_actions(agent_0, agent_1, states, add_noise):
    '''gets actions for each agent and then combines them into one array'''
    action_0 = agent_0.act(states, add_noise)    # agent 0 chooses an action
    action_1 = agent_1.act(states, add_noise)    # agent 1 chooses an action
    return np.concatenate((action_0, action_1), axis=0).flatten()
            
            
def store(agent_0, agent_1):
        torch.save(agent_0.actor_local.state_dict(), 'actor_0.pth')
        torch.save(agent_0.critic_local.state_dict(), 'critic_0.pth')
        torch.save(agent_1.actor_local.state_dict(), 'actor_1.pth')
        torch.save(agent_1.critic_local.state_dict(), 'critic_1.pth')
        
def load(agent_0, agent_1):
        checkpoints = ['checkpoints/actor_0.pth', 'checkpoints/critic_0.pth', 
                        'checkpoints/actor_1.pth', 'checkpoints/critic_1.pth']
        
        for path in checkpoints: 
            if not os.path.isfile(path):
                print("no checkpoints found for Actor and Critic...")
                return -1
        print("=> loading checkpoints for Actor and Critic... ")
        agent_0.actor_local.load_state_dict(checkpoints[0])
        agent_0.critic_local.load_state_dict(checkpoints[1])
        agent_1.actor_local.load_state_dict(checkpoints[2])
        agent_1.critic_local.load_state_dict(checkpoints[3])
        print("done !")
        return 0
        