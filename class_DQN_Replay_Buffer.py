# Replay buffer for DQN training

# Import dependencies
from collections import deque
import numpy as np
import random

class DQN_Replay_Buffer:
    def __init__(self, capacity):
        """
        Initialize the replay buffer with a given capacity
        
        Args:
            capacity (int): Maximum number of experiences to store in the buffer
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer

        Args:
            state (np.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.array): Next state after action
            done (bool): Whether the episode has ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the replay buffer

        Args:
            batch_size (int): Number of experiences to sample

        Returns:
            tuple: Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        """
        Return the current size of the replay buffer

        Returns:
            int: Number of experiences currently stored in the buffer
        """
        return len(self.buffer)