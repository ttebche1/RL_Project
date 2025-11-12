# Class for environment that enables an agent to search for a static target

from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pygame

class MultiAgentStaticTargetSearchEnv(MultiAgentEnv):
    def __init__(self, env_params, render_mode=None):
        """
        Initialize environment

        Args:
            env_params (dict): Parameters for the static target search environment
            render_mode: mode for rendering the environment; can be None or "human"
        """
        # Set random seed
        np.random.seed(None)

        # Set given parameters
        self._num_agents = env_params["num_agents"]
        self._size = env_params["env_size"]                                 # Distance from origin in all four directions                             
        self._target_radius = env_params["target_radius"] / self._size      # Radius for "found" condition, normalized
        self._max_step_size = env_params["max_step_size"] / self._size      # Maximum step size in meters, normalized
        self._max_steps_per_episode = env_params["max_steps_per_episode"]   # Maximum steps per episode
        self._dist_noise_std = env_params["dist_noise_std"] / self._size    # Standard deviation of Gaussian noise added to distance measurements, normalized      
        self._render_mode = render_mode

        # Initialize agents
        self._agent_ids = [f"agent_{i}" for i in range(self._num_agents)]    # Agent ID
        self._starting_location = np.array([0.0, 0.0], dtype=np.float32)    # Agent starting location = center

        # Initialize observation space (per agent)
        # x coordinate
        # y coordinate
        # distance to target
        # x coordinate at least measured distance
        # y coordinate at last measured distance
        # change in distance to target since last measurement
        # for each other agent:
        #   x coordinate
        #   y coordinate
        #   distance ot target
        #   change in distance to target
        self.observation_space = spaces.Box(
            low = np.array([-1.0, -1.0, 0.0, -1.0, -1.0, -self._max_step_size] +            # agent
                        [-1.0, -1.0, 0.0, -self._max_step_size] * (self._num_agents - 1),   # all other agents
                        dtype=np.float32),
            high = np.array([1.0, 1.0, 2.83, 1.0, 1.0, self._max_step_size] +               # agent
                            [1.0, 1.0, 2.83, self._max_step_size] * (self._num_agents - 1), # all other agents
                            dtype=np.float32),
            dtype = np.float32
        )

        # Initialize action space (per agent)
        # delta_x in [-1. 1]
        # delta_y in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Set render mode
        self._window_size = 512  
        self._window = None
        self._clock = None   

        # Reset environment
        self.reset()  
    
    def _get_obs(self):
        """
        Return all agents' observations
        
        Return:
            observations (dict): Dictionary mapping agent_id to observation numpy array
                Each observation array includes:
                    x coordinate
                    y coordinate
                    distance to target
                    x coordinate at least measured distance
                    y coordinate at last measured distance
                    change in distance to target since last measurement
                    for each other agent:
                        x coordinate
                        y coordinate
                        distance ot target
                        change in distance to target
        """
        # Initialize observation dictionary
        observations = {}
    
        # Get observation array for all agents
        for agent_id in self._agent_ids:
            # Observation array for agent with agent_id
            obs = [
                self._agent_locations[agent_id][0],          # self x
                self._agent_locations[agent_id][1],          # self y
                self._dist_to_target[agent_id],              # self distance to target
                self._prev_agent_locations[agent_id][0],     # prev self x
                self._prev_agent_locations[agent_id][1],     # prev self y
                self._dist_change[agent_id],                 # self distance change
            ]
            
            # Add information about all other agents to this agent's observation
            for other_agent_id in self._agent_ids:
                if other_agent_id != agent_id:
                    obs.extend([
                        self._agent_locations[other_agent_id][0],      # other agent x
                        self._agent_locations[other_agent_id][1],      # other agent y
                        self._dist_to_target[other_agent_id],          # other agent distance to target
                        self._dist_change[other_agent_id],             # other agent distance change
                    ])
            
            # Add observation array for this agent to the observation dictionary
            observations[agent_id] = np.array(obs, dtype=np.float32)
        
        return observations

    def reset(self, *, seed=None, options=None):
        """
        Initiate a new episode for an environment

        Args:
            seed: random seed for environment (unused)
            options: additional options for resetting environment (unused)

        Return:
            observation (numpy array)
            info: none
        """
        # Set the seed of the reset function in gym.Env (parent class)
        super().reset(seed=seed)

        # Initialize targte location
        self._target_location = np.random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32) # Random target location

        # Initialize agent locations
        self._agent_locations = {agent_id: self._starting_location.copy() for agent_id in self._agent_ids}
        self._prev_agent_locations = {agent_id: self._starting_location.copy() for agent_id in self._agent_ids}
        distance_to_target = self._get_distance_to_target(self._agent_ids[0])
        self._dist_to_target = {agent_id: distance_to_target for agent_id in self._agent_ids}
        self._dist_change = {agent_id: 0.0 for agent_id in self._agent_ids}

        # Initialize step count
        self._step_count = 0

        # Re-render environment if in human render mode
        if self._render_mode == "human":
            self._render_frame()

        # Return observation and information
        return self._get_obs(), {}
    
    def step(self, actions: np.ndarray):
        """
        Take actions for all agents

        Args:
            actions (dict): Dictionary mapping agent_id to action [delta_x, delta_y]

        Return:
            observations (dict): Dictionary of observations for all agents
            rewards (dict): Dictionary of rewards for all agents  
            terminated (dict): Dictionary of terminated flags for all agents + "__all__"
            truncated (dict): Dictionary of truncated flags for all agents + "__all__"
            info (dict): none
        """
        # Initialize dictionaries
        rewards = {}
        terminateds = {}
        truncateds = {}

        # Store previous locations and distances for all agents
        self._prev_agent_locations = self._agent_locations.copy()
        prev_distances = self._dist_to_target.copy()

        # Process each agent's action
        for agent_id, action in actions.items():
            # Ensure action is within action space
            action = np.clip(action, self.action_space.low, self.action_space.high)

            # Compute new location
            new_location = self._agent_locations[agent_id] + action * self._max_step_size

            # Check if new location is in bounds
            if np.any(new_location < -1.0) or np.any(new_location > 1.0):
                rewards[agent_id] = -1.0    # If out of bounds, remain in the same place and give a penalty
                terminateds[agent_id] = False
            else:
                # If in bounds, move agent
                self._agent_locations[agent_id] = new_location.copy()

                # Update distance to target
                self._dist_to_target[agent_id] = self._get_distance_to_target(agent_id)
                self._dist_change[agent_id] = prev_distances[agent_id] - self._dist_to_target[agent_id]

                # Check termination
                terminateds[agent_id] = bool(self._dist_to_target[agent_id] <= self._target_radius)

                # Update reward
                if terminateds[agent_id]:
                    rewards[agent_id] = 10.0
                else:
                    rewards[agent_id] = float(-self._dist_to_target[agent_id])

        # Truncate if max steps reached
        self._step_count += 1
        episode_truncated = self._step_count >= self._max_steps_per_episode
        truncateds = {"__all__": episode_truncated} # Set global flag
        for agent_id in self._agent_ids:            # Set individual agent flags
            truncateds[agent_id] = episode_truncated

        # Check if any agent found the target
        episode_terminated = any(terminateds.values())
        terminateds["__all__"] = episode_terminated # Set global flag
        for agent_id in self._agent_ids:            # Set individual agent flags
            terminateds[agent_id] = episode_terminated

        # Re-render environment if in human render mode
        if self._render_mode == "human":
            self._render_frame()

        return self._get_obs(), rewards, terminateds, truncateds, {}
    
    def _get_distance_to_target(self, agent_id):
        """ 
        Computes distance to target with noise added to the sensor measurement

        Args:
            agent_id (int): agent ID

        Return:
            dist_to_target (float): distance to target for the given agent
        """
        dist_to_target = np.linalg.norm(self._agent_locations[agent_id] - self._target_location) # Compute distance to target
        dist_to_target += np.random.normal(0.01 * dist_to_target, self._dist_noise_std)         # Add noise to distance measurement
        dist_to_target = max(0.0, dist_to_target)                                               # Verify distance is never negative

        return dist_to_target
    
    def _env_to_screen(self, location):
        """
        Convert environment coordinates to pygame screen coordinates
        
        Args:
            location (numpy array): [x, y] in environment coordinates
            
        Return:
            screen_location (numpy array): [x_pix, y_pix] in screen coordinates
        """
        # Shift and scale from [-1,1] -> [0,1] for screen
        x_scaled = (location[0] + 1.0) / 2.0
        y_scaled = (location[1] + 1.0) / 2.0

        # Convert to pixel coordinates
        x_pix = int(x_scaled * self._window_size)
        y_pix = int((1.0 - y_scaled) * self._window_size)  # flip y for pygame
        return np.array([x_pix, y_pix])

    def _render_frame(self):
        """Render the next frame"""

        # Initialize window if it hasn't been created yet
        if self._window is None:
            pygame.init()           
            pygame.display.init()   
            self._window = pygame.display.set_mode((self._window_size, self._window_size))

        # Initialize clock if it hasn't been created yet
        if self._clock is None:
            self._clock = pygame.time.Clock()

        # Create white canvas of size window_size x window_size pixels
        canvas = pygame.Surface((self._window_size, self._window_size))
        canvas.fill((255, 255, 255))

        # Draw target
        target_center = tuple(self._env_to_screen(self._target_location))       # Convert coordinates
        circle_radius = max(8, int(self._window_size * 0.015))                  # Set target dot radius
        pygame.draw.circle(canvas, (255, 0, 0), target_center, circle_radius)   # Draw target as a red dot

        # Draw target found radius
        pixels_per_unit = self._window_size / 2  
        radius_pix = int(self._target_radius * pixels_per_unit)  
        if radius_pix > 0:
            pygame.draw.circle(canvas, (255, 0, 0), target_center, radius_pix, width=1)

        # Draw all agents
        for i, agent_id in enumerate(self._agent_ids):
            agent_center = tuple(self._env_to_screen(self._agent_locations[agent_id]))  # Convert coordinates
            pygame.draw.circle(canvas, (0, 0, 255), agent_center, circle_radius)        # Draw agent as a blue dot

        # Copy canvas to visible window
        self._window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # Update at 4 frames per second
        self._clock.tick(4)
        
    def close(self):
        """Close pygame resources if the window has been initialized and is active"""

        if self._window is not None:
            pygame.display.quit()  
            pygame.quit()
            self._window = None
            self._clock = None