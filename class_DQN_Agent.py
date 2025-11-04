# DQN Agent

# Import dependencies
from class_DQN_NN import DQN_NN
from class_DQN_Replay_Buffer import DQN_Replay_Buffer
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

class DQN_Agent:
    def __init__(self, env, learning_rate, discount_factor,
                 initial_epsilon, final_epsilon, epsilon_decay,
                 buffer_capacity, batch_size, target_update):
        """
        Initialize DQN agent

        Args:
            env (gym.Env): gymnasium environment
            learning_rate (float): learning rate for optimizer
            discount_factor (float): discount factor for future rewards
            initial_epsilon (float): starting value for epsilon in epsilon-greedy policy
            final_epsilon (float): final value for epsilon after decay
            epsilon_decay (float): amount to decay epsilon each episode
            buffer_capacity (int): capacity of replay buffer
            batch_size (int): batch size for training
            target_update (int): number of steps between target network updates
        """
        if not torch.cuda.is_available():
            print("CUDA is not available. Running on CPU instead.")
            self.device = "cpu"
        else:
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            self.device = "cuda"

        # Set gymnasium environment
        self.env = env
        env.reset()

        # Set input dimension to hold normalized (x,y) coordinates and all_cells_mapped flag
        self.input_dim = 3

        # Set number of actions (left, right, up, down)
        self.n_actions = env.action_space.n

        # Create policy and target networks
        # Policy network is used for selecting actions
        # Target network is used for stable q-value updates (copy of the policy network that is updated less frequently))
        self.HIDDEN_DIM = 64
        self.policy_net = DQN_NN(self.input_dim, self.n_actions, self.HIDDEN_DIM).to(self.device)
        self.target_net = DQN_NN(self.input_dim, self.n_actions, self.HIDDEN_DIM).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())               # Make a copy of the policy network
        self.target_update = target_update
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate) # Initialize optimizer

        # Initialize replay buffer
        self.replay = DQN_Replay_Buffer(buffer_capacity)
        self.batch_size = batch_size

        # Initialize discount factor for future rewards
        self.discount_factor = discount_factor

        # Initialize epsilon for epsilon-greedy policy
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        # Initialize array to hold loss history for training error visualization
        self.loss_history = []

        # Initialize step counter
        self.total_steps = 0

    def _normalize_state(self, obs):
        """
        Normalize state observation
        
        Args:
            obs (np.array): raw observation from environment
            
        Returns:
            np.array: normalized state in the range of [0,1]
        """
        # Get grid size from the environment
        grid_size = self.env.unwrapped.get_grid_size()

        if grid_size is None or grid_size <= 1:
            print("ERROR: Environment does not have attribute 'grid_size'. Defaulting to 2.")
            grid_size = 2

        # Return normalized state
        return np.array(obs, dtype=np.float32) / (grid_size - 1)

    def select_action(self, obs):
        """
        Select action using epsilon-greedy policy

        Args:
            obs (np.array): current observation from environment
        
        Returns:
            int: selected action
        """
        # Normalize state
        state = self._normalize_state(obs)

        # Uniformly sample an action with probability epsilon
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        # Otherwise select action with highest q-value from policy network
        else:
            # Convert state from numpy array to torch tensor
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

            # Disable gradient calculation for action selection
            with torch.no_grad():
                # Obtain q-values from policy network for all actions given state_t
                q_vals = self.policy_net(state_t)

                # Pick action with highest q-value
                return int(torch.argmax(q_vals, dim=1).item())

    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Store transition in replay buffer

        Args:
            obs (np.array): current observation from environment
            action (int): action taken
            reward (float): reward received
            next_obs (np.array): next observation from environment
            done (bool): whether the episode has terminated
        """
        # Normalize current and next states
        state = self._normalize_state(obs)
        next_state = self._normalize_state(next_obs)

        # Store transition in replay buffer
        self.replay.push(state, action, reward, next_state, done)

    def train(self):
        """
        Sample a batch from replay buffer and train the NNs

        Returns:
            float: loss value from the learning step
        """
        # Do not train if there are not enough samples in the replay buffer
        if len(self.replay) < self.batch_size:
            return None
        
        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        # Convert to PyTorch tensors and move to correct device
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.uint8)).float().unsqueeze(1).to(self.device)

        # Obtain Q-values from policy NN for sampled states
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # Obtain target Q-values from target NN for sampled next states
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (1 - dones_t) * (self.discount_factor * next_q)

        # Compute MSE loss between current Q-values and target Q-values
        loss = F.mse_loss(q_values, target_q)

        # Optimize policy NN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss value for training error visualization
        self.loss_history.append(loss.item())

        # Increment step counter
        self.total_steps += 1

        # Update target NN periodically by copying weights from policy NN
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Return loss value
        return loss.item()

    def decay_epsilon(self):
        """
        Decay epsilon value after each episode for epsilon-greedy policy
        """
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay
            if self.epsilon < self.final_epsilon:
                self.epsilon = self.final_epsilon

    """ Return a copy of the training error history """
    @property
    def training_error(self):
        return self.loss_history.copy()
