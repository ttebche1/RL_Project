# Class for environment that enables an agent to search for a static target

from collections import deque
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pygame

class SingleAgentStaticTargetSearchEnv(gym.Env):
    def __init__(self, env_params, render_mode=None):
        """
        Initialize environment

        Args:
            env_params (dict): Parameters for the static target search environment
            render_mode: mode for rendering the environment; can be None or "human"
        """
        # Set random seed
        np.random.seed(None)

        # Initialize parameters
        self._size = env_params["env_size"]                                 # Distance from origin in all four directions                             
        self._target_radius = env_params["target_radius"] / self._size      # Radius for "found" condition, normalized
        self._max_step_size = env_params["max_step_size"] / self._size      # Maximum step size in meters, normalized
        self._max_steps_per_episode = env_params["max_steps_per_episode"]   # Maximum steps per episode
        self._dist_noise_std = env_params["dist_noise_std"] / self._size    # Standard deviation of Gaussian noise added to distance measurements, normalized      
        self._starting_location = np.array([0.0, 0.0], dtype=np.float32)    # Agent starting location

        # Initialize observation space: 
        # agent's x coordinate
        # agent's y coordinate 
        # distance to target
        # agent's x coordinate at least measured distance
        # agent's y coordinate at last measured distance
        # change in distance to target since last measurement
        # agent's x velocity
        # agent's y velocity
        # agent's x acceleration
        # agent's y acceleration
        self.observation_space = spaces.Box(
            low = np.array([-1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -self._max_step_size, -self._max_step_size, 
                            -self._max_step_size, -2*self._max_step_size, -2*self._max_step_size], dtype=np.float32),
            high = np.array([1.0, 1.0, 2.83, 1.0, 1.0, 1.0, 1.0, self._max_step_size, self._max_step_size, 
                             self._max_step_size, 2*self._max_step_size, 2*self._max_step_size], dtype=np.float32),
            dtype = np.float32
        )

        # Initialize a stacked buffer
        self.stack_size = 4
        self.obs_buffer = deque(maxlen=self.stack_size)

        # Initialize action space: delta_x, delta_y in [-1, 1], will be scaled by self._max_step
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Set render mode
        self.render_mode = render_mode  
        self._window_size = 512  
        self._window = None
        self._clock = None    

        # Reset environment
        self.reset() 
    
    def _get_obs(self):
        """
        Return agent's observation
        
        Return:
            observation (numpy array):
                agent's x coordinate
                agent's y coordiante
                distance to target
                agent's x coordinate at previous distance
                agent's y at previous distance
                change in distance
                agent's x velocity
                agent's y velocity
                agent's x acceleration
                agent's y acceleration
        """
        return np.array([
            self._agent_location[0],
            self._agent_location[1],
            self._dist_to_target,
            self._dist_to_target_vec[0],
            self._dist_to_target_vec[1],
            self._prev_agent_location[0],
            self._prev_agent_location[1],
            self._dist_change,
            self._velocity[0],
            self._velocity[1],
            self._acceleration[0],
            self._acceleration[1]],
        dtype=np.float32)
    
    def _get_info(self):
        return {}

    def reset(self, *, seed=None, options=None):
        """
        Initiate a new episode for an environment

        Args:
            seed: random seed for environment (unused)
            options: additional options for resetting environment (unused)

        Return:
            observation (numpy array): [agent x, agent y, last measured distance to target]
            info: none
        """
        # Set the seed of the reset function in gym.Env (parent class)
        super().reset(seed=seed)

        # Initialize locations
        self._agent_location = self._starting_location.copy()                                       # Center
        self._prev_agent_location = self._starting_location.copy()
        self._target_location = np.random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32) # Random target location

        # Initialize distances
        self._dist_to_target = self._compute_dist_to_target()
        self._dist_to_target_vec = (self._agent_location-self._target_location) / 2
        self._dist_change = 0.0

        # Initialize current
        self._current = (np.random.uniform(-1/3, 1/3, size=(2,)).astype(np.float32)) * self._max_step_size

        # Initialize velocity and acceleration
        self._velocity = np.array([0.0, 0.0], dtype=np.float32) 
        self._prev_velocity = np.array([0.0, 0.0], dtype=np.float32) 
        self._acceleration = np.array([0.0, 0.0], dtype=np.float32)   

        # Initialize step count
        self._step_count = 0

        # Re-render environment if in human render mode
        if self.render_mode == "human":
            self._render_frame()

        # Return observation and information
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray):
        """
        Take an action in the environment

        Args:
            action (numpy array): [delta_x, delta_y] in [-1, 1]

        Return:
            observation (numpy array): [agent_x, agent_y, distance_to_target]   
            reward (float): reward received after taking action
            terminated (bool): whether episode has ended
            truncated (bool): whether episode was truncated (set to False)
            info: none
        """
        # Ensure action is within action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Compute new location
        new_location = self._agent_location + action * self._max_step_size + self._current

        # Check if new location is in bounds
        if np.any(new_location < -1.0) or np.any(new_location > 1.0):
            reward = -1.0   # If out of bounds, remain in the same place and give a penalty
            terminated = False
        else:
            # If in bounds, move agent
            self._prev_agent_location = self._agent_location.copy()
            self._agent_location = new_location.copy()

            # Update distance to target
            prev_dist_to_target = self._dist_to_target.copy()
            self._dist_to_target = self._compute_dist_to_target()
            self._dist_to_target_vec = (self._agent_location - self._dist_to_target_vec) / 2
            self._dist_change = prev_dist_to_target - self._dist_to_target

            # Update velocity and acceleration
            self._prev_velocity = self._velocity.copy()
            self._velocity = self._agent_location - self._prev_agent_location
            self._acceleration = self._velocity - self._prev_velocity

            # Terminal if within target radius
            terminated = bool(self._dist_to_target <= self._target_radius)

            # Update reward
            if terminated:
                reward = 10.0
            else:
                reward = float(-self._dist_to_target)
        
        # Truncate if max steps reached
        self._step_count += 1
        truncated = self._step_count >= self._max_steps_per_episode

        # Re-render environment if in human render mode
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _compute_dist_to_target(self):
        """
        Computes distance from agent to target

        Return:
            dist_to_target (float): distance from agent to target, normalized
        """
        dist_to_target = np.linalg.norm(self._agent_location - self._target_location)   # Compute distance
        dist_to_target += np.random.normal(0.01 * dist_to_target, self._dist_noise_std) # Add noise
        dist_to_target = max(0.0, dist_to_target)                                       # Remove negative distances

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

        # Convert coordinates (flip y)
        target_center = tuple(self._env_to_screen(self._target_location))
        agent_center = tuple(self._env_to_screen(self._agent_location))
       
        # Draw agent and target with a minimum visible radius
        circle_radius = max(8, int(self._window_size * 0.015))  # slightly larger
        pygame.draw.circle(canvas, (0, 0, 255), agent_center, circle_radius)  # agent: blue
        pygame.draw.circle(canvas, (255, 0, 0), target_center, circle_radius)  # target: red

        # Draw target radius scaled to window size
        pixels_per_unit = self._window_size / 2  
        radius_pix = int(self._target_radius * pixels_per_unit)  
        if radius_pix > 0:
            pygame.draw.circle(canvas, (255, 0, 0), target_center, radius_pix, width=1)

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