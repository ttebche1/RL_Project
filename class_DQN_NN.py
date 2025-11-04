# Feedforward neural network for DQN

# Import dependencies
import torch.nn as nn
import torch.nn.functional as F

class DQN_NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Define neural network structure

        Args:
            input_dim (int): dimension of input
            output_dim (int): # possible actions
            hidden_dim (int): # neurons in hidden layers
        """
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the neural network

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor with q-values for each action
        """
        # Process input through input layer with ReLU activation
        x = F.relu(self.input_layer(x))

        # Process through hidden layer with ReLU activation
        x = F.relu(self.hidden_layer(x))

        # Process through output layer with no activation
        return self.output_layer(x)