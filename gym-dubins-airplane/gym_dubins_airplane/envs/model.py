<<<<<<< HEAD
# Deep Q Network Architecture
# 3 hidden layers with ReLu activation and output layer is linear

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        # to add initialization method: (xavier for example)
        # torch.nn.init.xavier_normal_(self.linear2.weight)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, action_size)

    # In the forward function, you define how your model is going to be run, from input to output
    # The forward method is called from the __call__ function of nn.Module,
    # so that when we run model(input), the forward method is called.
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
    
    def load_model(self, filename):
      """
      Function to load model parameters
      """
      self.load_state_dict(torch.load(filename))
  
=======
# Deep Q Network Architecture
# 2 hidden layers with ReLu activation and output layer is linear

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)

    # In the forward function, you define how your model is going to be run, from input to output
    # The forward method is called from the __call__ function of nn.Module,
    # so that when we run model(input), the forward method is called.
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
>>>>>>> 6819adb566d3adb52b4ba0d843df0a5e09f4af63
