import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        """
        Create critic neural network
        (Critic estimates expected Q-value for taking an action)

        Args:
            obs_dim (int): observation array dimension
            act_dim (int): action array dimension
            hidden_dim (int): size of neural network hidden layer
        """
        super().__init__()
        
        # Define neural network architectures
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),   # Input layer
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),          # Hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)                    # Output layer
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),   # Input layer
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),          # Hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)                    # Output layer
        )

    def forward(self, obs, act):
        """
        Forward pass to compute expected Q-value for a set of state-action pairs

        Args:
            obs (torch.Tensor): set of observations
            act (torch.Tensor): set of actions

        Returns:
            q1 (torch.Tensor): Q-value estimate from the first critic network
            q2 (torch.Tensor): Q-value estimate from the second critic network
        """
        # Concatenate observations and actions
        x = torch.cat([obs, act], dim=-1)

        # Pass through the critic neural networks
        return self.q1(x), self.q2(x)