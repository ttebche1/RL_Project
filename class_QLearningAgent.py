# Class for Q-Learning Agent

"""Q-Learning Algorithm:

* Model-free reinforcement learning algorithm
* Q-Value Q(s,a):
    - Represents the expected future reward of taking action a in state s
    - Updated as follows: Q(s,a) <- Q(s,a) + alpha[r+gamma*max_{a'} of {Q(s',a')-Q(s,a)}]
        . Q(s,a) -> current Q-value for taking action a in state s
        . r -> immediate reward for taking action a from state s
        . gamma -> discount factor; determines how much future rewards are considered
        . max_{a'} of {Q(s',a')-Q(s,a)} -> max Q-value for the nest state s', corresponding to the best action a' the agent can take from state s'
        . alpha -> learning rate; determines how much the Q-value is updated based on new info
    - Exploration: agent sometimes takes random actions to explore the environment; controlled by epsilon
    - Exploitation: agent uses current knowledge (Q-values) to choose best action; more liekly to exploit as epsilon decays
* Algorithm:
    - Initialize:
        . Q-values for all state-action pairs = zero
        . epsilon - initial exploration value
        . learning parameters (learning rate, discount factor, etc.)
    - Each episode:
        . Reset environment to initial state
        . Either explore (random action) or exploit (take best-known action); decision made with probability epsilon, where epsilon = 1 means explore 100% of the time
        . Take the action
        . Observe the next state s', reward r, and whether the episode has terminated
        . Update Q-value using equation above
        . Decay epsilon to reduce exploration over time
        . Repeat until convergence or a predefined number of episodes
    - Compute a Q value for each state-action pair
        . In seafloor mapping, states are every possible location the robot could be
        . The action for each state is every possible location the robot could move to from that state
"""

# Import dependencies
from collections import defaultdict # Automatically creates default values for dictionary keys
from class_GridWorldEnv import Actions
import gymnasium as gym
import numpy as np

# Create a class for the agent we want to train
class QAgent:
    # Initialize agent
    # env, learning_rate, initial epsilon, epsilon_decay, final_epsilon, and discount_factor are inputs to the __init__() function
    def __init__(
        self,
        env: gym.Env,   # env is of type gym.Env class
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
    ):
        """
        Initialize a Reinforcement Learning agent
            - Define environment
            - Intiialize Q-learning variables (learning rate, epsilon)
            - Initialize q-value dictionary to an array of 8 (size of action space) zeros
                As the code runs, an array will be made for each possible observation (x-y coordinate pair)
                If there is no array for an observation, the default values will be 8 zeros

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        # Set environment
        self.env = env

        # For each state, an agent can take 8 actions
        # Create an empty dictionary with size of number of possible actions
        # Action examples: move right, left, up, down, etc.
        # Q-value represents expected future reward of taking an action based on the current state
        # The dictionary will hold the q-values for each possible action given some state
        # If the state hasn't been visited before, the default values will be 0
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        # For each state (location and all cells mapped flag), an agent can take 4 actions
        # Initialize all states with zeros for each action
        grid_size = self.env.env.env.get_grid_size()
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(2):
                    # Initialize a state (x, y, z) with zeros for all actions
                    self.q_values[(x, y, z)] = np.zeros(len(Actions))
        
        # Set forbidden actions to negative infinity
        for x in range(grid_size):
            for z in range(2):
                # If in the top row, agent can't move up
                self.q_values[(x,0,z)][Actions.UP.value] = -np.inf
            
                # If in the bottom row, agent can't move down
                self.q_values[(x,grid_size-1,z)][Actions.DOWN.value] = -np.inf
            
                # If in the left-most column, agent can't move left
                self.q_values[(0,x,z)][Actions.LEFT.value] = -np.inf

                # If in the right-most column, agent can't move right
                self.q_values[(grid_size-1,x,z)][Actions.RIGHT.value] = -np.inf

        # Set Q-learning parameters
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Initialize empty array to hold training error
        self.training_error = []

    def get_action(self, obs: tuple[int, int]) -> int:
        """
        If random number is < epsilon, take a random action to ensure exploration (probability = epsilon)
        Else, return the best action (probability = 1-epsilon)

        Args:
            self: for self.epsilon, which sets the probability of exploring vs. exploiting
            obs: dictionary containing agent's location and number of mapped cells

        Return:
            integer representing the action to take (e.g. move right = 0)
        """
        # Generate a random number between 1 and 0. If it is < epsilon, explore
        if np.random.random() < self.epsilon:
            while(True):
                # Pick a random action
                action = self.env.action_space.sample()

                # If action is within grid bounds, use action. Else, select a new random action
                if (self.check_bounds(obs,action)):
                    return action

        # If the random number is > epsilon, exploit
        else:
            # Get indices of actions sorted by the Q-values in descending order (max to min)
            # self.q_values[obs]: return learned q-values for each possible action given the current state
            q_vals = self.q_values[obs]
            sorted_actions = np.argsort(self.q_values[obs])[::-1]

            # Find largest action within grid bounds (use best value agent has learned so far for this state)
            for action in sorted_actions:
                if self.check_bounds(obs, action):
                    return action

    def update(
        self,
        prev_obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        obs: tuple[int, int, bool],
    ):
        """
        Updates the Q-value of the previous state-action pair
            - Computes the actual q-value of the previous state-action
            - Computes the error between the expected and actual q-values of the previous state-action
            - Updates the q-value of the previous state-action pair based on this error

        At this point, the agent would have taken the step chosen in this turn
        """
        # If not terminated, find max q-value given the current state (agent location)
        next_max_q_value = ((not terminated) * np.max(self.q_values[obs]))

        # Find q-value observed from previous state-action pair: reward from action + next q-value for new state
        # Discount_factor weights the importance of the next q-value vs. reward from most recent action
        prev_observed_q_value = reward + self.discount_factor * next_max_q_value

        # Find difference between actual q-value from most recent action and q-value in the table
        temporal_difference_error = prev_observed_q_value - self.q_values[prev_obs][action]

        # Update q-value from previous state-action pair by adding temporal difference error to it
        # Learning rate controls how much we change previous q-values at any given timestep
        self.q_values[prev_obs][action] = self.q_values[prev_obs][action] + self.lr * temporal_difference_error

        # Add temporal_difference to training_error array
        self.training_error.append(temporal_difference_error)

    def decay_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def get_q_values(self):
        """Return learned Q-values"""
        return self.q_values

    def check_bounds(self, obs: tuple[int, int], action: int) -> bool:
        """
        Check if the desired action would move the agent outside of the grid bounds

        Args:
            obs: agent's current location (before action is taken)
            action: desired action
        
        Return:
            bool=True if action is within bounds, else bool=False
        """

        # Map the action (e.g. right) to the direction we walk in (e.g. [1, 0])
        direction = self.env.env.env.get_direction_for_one_action(action)

        # Compute agent x- and y-coordinates if it takes this action
        new_x = obs[0] + direction[0]
        new_y = obs[1] + direction[1]

        # Check if action is within grid bounds
        grid_size = self.env.env.env.get_grid_size()
        if (0 <= new_x <= grid_size-1 and 0<= new_y <= grid_size-1):
            return True
        else: 
            return False