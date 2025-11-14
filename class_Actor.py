# Neural network class for SAC actor (action) network

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX = 2     # Max log standard deviation
LOG_STD_MIN = -5    # Min log standard deviation

class Actor(nn.Module):
    def __init__(self, env):
        """
        Initialize neural network

        Args:
            env (gym.env): environment
        """
        super().__init__()

        # Define neural network structure
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)  # Input layer
        self.fc2 = nn.Linear(256, 256)                                                  # Hidden layer
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))           # Output layer for mean
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))         # Output layer for log standard deviation

    def forward(self, x):
        """
        Forward pass through the neural network

        Args:
            x (torch.Tensor): observations with shape (batch_size, obs_dim)
        
        Returns:
            torch.Tensor: estimated action space mean with shape (batch_size, 1)
            torch.Tensor: estimated action log standard deviation wtih shape (batch_size, 1)
        """
        # Pass through the neural network
        x = F.relu(self.fc1(x))     # Input layer
        x = F.relu(self.fc2(x))     # Hidden layer
        mean = self.fc_mean(x)      # Output layer for mean
        log_std = self.fc_logstd(x) # Output layer for log standard deviation

        # Clamp log_std between LOG_STD_MAX and LOG_STD_MIN
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        """
        Sample action from the actor

        Args:
            x (torch.Tensor): observations with shape (batch_size, obs_dim)
        
        Returns:
            action (torch.Tensor): sampled action with shape (batch_size, action_dim)
            log_prob (torch.Tensor): log probability of the sampled action with shape (batch_size, 1)
            mean (torch.Tensor): deterministic action (mean of the policy) with shape (batch_size, action_dim)
        """
        # Create action distribution
        mean, log_std = self(x)                         # Call actor neural network
        std = log_std.exp()                             # Convert log standard deviation to standard deviation
        normal = torch.distributions.Normal(mean, std)  # Define action distribution

        # Get an action
        x_t = normal.rsample()      # Sample an action from the action distribution
        action = torch.tanh(x_t)    # Squash action to [-1, 1] range

        # Get probability of sampling the action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # Get mean of action distribution
        mean = torch.tanh(mean) # Scale mean to [-1, 1]

        return action, log_prob, mean