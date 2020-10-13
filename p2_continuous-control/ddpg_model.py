import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model"""
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn3 = nn.BatchNorm1d(action_size)
        
        self.tanh = nn.Tanh()
        # initialise the weights
        self.reset_parameters()
        
        
    def reset_parameters(self):
        """
        All layers but the final layer are initilaised from uniform
        distributions [-1/sqrt(f) , 1/sqrt(f)] where f is the fan-in of the layer
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        
        """
        The final layer is initialised from uniform distribution
        [-3*10^-3, 3*10^-3]
        """
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
        
    def forward(self, states):
        # Actor network maps states to action probabilities 
        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)

        states = self.bn(states)
        x = F.relu(self.bn1(self.fc1(states)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.tanh(self.bn3(self.fc3(x)))
    
    
class Critic(nn.Module):
    """Critic (Value) Model"""
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc3_units=128):
        super(Critic, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)
        
        self.fc4 = nn.Linear(fc3_units, 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        All layers but the final layer are initilaised from uniform
        distributions [-1/sqrt(f) , 1/sqrt(f)] where f is the fan-in of the layer
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        """
        The final layer is initialised from uniform distribution
        [-3*10^-3, 3*10^-3]
        """
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, states, actions):
        # Critic network maps (state, action) pairs to Q-values
        # For Batch Normalisation
        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)
        
        states = self.bn(states)
        xs = F.leaky_relu(self.bn1(self.fc1(self.bn(states))))
        x = torch.cat([xs, actions], dim=1)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        return self.fc4(x)
    