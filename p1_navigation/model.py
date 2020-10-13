import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc3_units=256):
        """Initialize parameters and build model.
           INPUT
           - state_size (int): Dimension of each state
           - action_size (int): Dimension of each action
           - seed (int): Random seed
           - fc1_units (int): Number of nodes in first hidden layer
           - fc2_units (int): Number of nodes in second hidden layer
           - fc3_units (int): Number of nodes in third hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.action_layer = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(F.dropout(self.fc1(state)))
        x = F.relu(F.dropout(self.fc2(x)))
        x = F.relu(F.dropout(self.fc3(x)))
        q_values = self.action_layer(x)
        return q_values
