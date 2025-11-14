# Neural network class for SAC critic (Q-function) network

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, env):
        """
        Initialize neural network

        Args:
            env (gym.env): environment
        """
        super().__init__()

        # Concatenate observation and action space
        input_dim = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)

        # Define neural network structure 
        self.fc1 = nn.Linear(input_dim, 256)    # Input layer
        self.fc2 = nn.Linear(256, 256)          # Hidden layer
        self.fc3 = nn.Linear(256, 1)            # Output layer

    def forward(self, x, a):
        """
        Forward pass through the neural network

        Args:
            x (torch.Tensor): observations with shape (batch_size, obs_dim)
            a (torch.Tensor): actions with shape (batch_size, action_dim)
        
        Returns:
            torch.Tensor: estimated Q-values with shape (batch_size, 1)
        """
        # Concatenate observations and actions
        x = torch.cat([x, a], 1)

        # Pass through the neural network
        x = F.relu(self.fc1(x)) # Input layer
        x = F.relu(self.fc2(x)) # Hidden layer
        x = self.fc3(x)         # Output layer

        return x