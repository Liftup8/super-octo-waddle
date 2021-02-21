import numpy as np
import random

from model import QNetwork
from replay_buffer import ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import operator

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.999  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_NN_EVERY = 1  # how often to update the network

# prioritized experience replay
UPDATE_MEM_EVERY = 20  # how often to update the priorities
UPDATE_MEM_PAR_EVERY = 3000  # how often to update the hyperparameters
EXPERIENCES_PER_SAMPLING = math.ceil(BATCH_SIZE * UPDATE_MEM_EVERY /
                                     UPDATE_NN_EVERY)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, seed, compute_weights=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.compute_weights = compute_weights
        
        # Algorithms to enable during training
        self.PrioritizedReplayBuffer = True # Use False to enable uniform sampling
        self.HardTargetUpdate = True # Use False to enable soft target update

        # building the policy and target Q-networks for the agent, such that the target Q-network is kept frozen to avoid the training instability issues
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size,
                                       seed).to(device)  # main policy network
        self.qnetwork_target = QNetwork(state_size, action_size,
                                        seed).to(device)  # target network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        # Replay memory
        # building the experience replay memory used to avoid training instability issues
        # Below: PER
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE,
                                   EXPERIENCES_PER_SAMPLING, seed,
                                   compute_weights)
                                   
        # Below: Uniform by method defined in this script
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_NN_EVERY steps)
        self.t_step_nn = 0
        # Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
        self.t_step_mem_par = 0
        # Initialize time step (for updating every UPDATE_MEM_EVERY steps)
        self.t_step_mem = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_NN_EVERY time steps.
        self.t_step_nn = (self.t_step_nn + 1) % UPDATE_NN_EVERY
        self.t_step_mem = (self.t_step_mem + 1) % UPDATE_MEM_EVERY
        self.t_step_mem_par = (self.t_step_mem_par + 1) % UPDATE_MEM_PAR_EVERY
        if self.t_step_mem_par == 0:
            self.memory.update_parameters()
        if self.t_step_nn == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.experience_count > EXPERIENCES_PER_SAMPLING:
                sampling = self.memory.sample()
                self.learn(sampling, GAMMA)
        if self.t_step_mem == 0:
            self.memory.update_memory_sampling()

    def act(self, state, eps=0.):
        """A function to select an action based on the Epsilon greedy policy. Epislon percent of times the agent will select a random
        action while 1-Epsilon percent of the time the agent will select the action with the highest Q value as predicted by the
        neural network.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # here we calculate action values (Q values)
        self.qnetwork_local.eval() # model deactivate norm, dropout etc. layers as it is expected
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # model.train() sets the modules in the network in training mode

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.cpu().numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, sampling, gamma):
        """Update value parameters using given batch of experience tuples.
        Function for training the neural network. The function will update the weights of the newtwork

        Params
        ======
            sampling (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indices = sampling

        # Target (absolute) Q values from target Q network
        q_target = self.qnetwork_target(next_states).detach().max(
            1)[0].unsqueeze(1)
        # Predictions from local Q network
        expected_values = rewards + gamma * q_target * (1 - dones)
        output = self.qnetwork_local(states).gather(1, actions)
        # computing the loss
        loss = F.mse_loss(output,
                          expected_values)  # Loss Function: Mean Square Error
        if self.compute_weights:
            with torch.no_grad():
                weight = sum(np.multiply(weights, loss.data.cpu().numpy()))
            loss *= weight
        # Minimizing the loss by optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # ------------------- update priorities ------------------- #
        delta = abs(expected_values - output.detach()).cpu().numpy()
        #print("delta", delta)
        self.memory.update_priorities(delta, indices)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)
    
    # def hard_update(self):
      # """ This hard_update method performs direct update of target network
      # weight update from local network weights instantly"""
      
      # Write the algorithm here
      
    def load_models(self, policy_net_filename, target_net_filename):
      """ Function to load the parameters of the policy and target models """
      print('Loading model...')
      self.qnetwork_local.load_model(policy_net_filename)
      self.qnetwork_target.load_model(target_net_filename)
      
# class ReplayBuffer:
    # """Fixed-size buffer to store experience tuples."""

    # def __init__(self, action_size, buffer_size, batch_size, seed):
        # """Initialize a ReplayBuffer object.

        # Params
        # ======
            # action_size (int): dimension of each action
            # buffer_size (int): maximum size of buffer
            # batch_size (int): size of each training batch
            # seed (int): random seed
        # """
        # self.action_size = action_size
        # self.memory = deque(maxlen=buffer_size)
        # self.batch_size = batch_size
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # self.seed = random.seed(seed)

    # def add(self, state, action, reward, next_state, done):
        # """Add a new experience to memory."""
        # e = self.experience(state, action, reward, next_state, done)
        # self.memory.append(e)

    # def sample(self):
        # """Randomly sample a batch of experiences from memory."""
        # experiences = random.sample(self.memory, k=self.batch_size)

        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            # device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            # device)

        # return (states, actions, rewards, next_states, dones)

    # def __len__(self):
        # """Return the current size of internal memory."""
        # return len(self.memory)
  
