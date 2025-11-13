import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        """
        Create actor neural network
        (Actor provides an action for a given observation)

        Args:
            obs_dim (int): observation array dimension
            act_dim (int): action array dimension
            hidden_dim (int): size of neural network hidden layer
        """
        super().__init__()

        # Define neural network architecture
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),     # Input layer
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Hidden layer
            nn.ReLU(),
        )

        # Output layers
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        """
        Forward pass to compute action distribution parameters

        Args:
            obs (torch.Tensor): set of observations

        Returns:
            mean (torch.Tensor): mean of the Gaussian distribution
            std (torch.Tensor): standard deviation of the Gaussian
        """
        # Pass observations through the internal layers
        x = self.net(obs)

        # Pass through the output layers
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)

        # Convert log_std to std
        std = log_std.exp()

        return mean, std

    def sample(self, obs):
        """
        Sample action from the policy

        Args:
            obs (torch.Tensor): set of observations

        Returns:
            action (torch.Tensor): sampled action
            log_prob (torch.Tensor): log of probability of the action
        """
        # Compute mean and std given a set of observations
        mean, std = self(obs)

        # Draw a sample from the distribution
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  

        # Squash action to be wtihin the [-1,1] range
        action = torch.tanh(x_t)

        # Output the log of the probability of taking action "action" given the observations
        log_prob = normal.log_prob(x_t).sum(axis=-1) - torch.log(1 - action.pow(2) + 1e-6).sum(axis=-1)

        return action, log_prob