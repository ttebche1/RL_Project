# Class for environment that enables an agent to search for a static target

from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pygame

class static_target_search_env(gym.Env):
    # Define environment settings (render mode and framerate)
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        """
        Initialize environment

        Args:
            render_mode: mode for rendering the environment; can be None or "human"
        """
        # Width and length of square-shaped environment
        # 1414 x 1414 allows distance between agent and target to never exceed ~2km
        self._size = 1414.0

        # Initialize agent location in top-left corner
        self._starting_location = np.array([0.0, self._size-1], dtype=np.float32)
        self._agent_location = self._starting_location.copy()

        # Initialize target location in bottom-right corner
        self._target_location = np.array([self._size-1, 0.0], dtype=np.float32)

        # Radius for "found" condition
        self._target_radius = 100.0

        # Initialize observation space: agent_x, agent_y, distance_to_target
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize action space: delta_x, delta_y in [-1, 1], will be scaled by self._max_step
        self._max_step_size = 10.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Set max number of steps per episode
        self._step_count = 0
        self._max_steps_per_episode = 200

        # Set render mode
        self.render_mode = render_mode  
        self._window_size = 512  
        self._window = None
        self._clock = None     
    
    def _get_obs(self):
        """
        Return agent's location and distance to target
        
        Return:
            observation (numpy array): [agent_x, agent_y, distance_to_target]
        """
        return np.array([
            self._agent_location[0]/self._size,
            self._agent_location[1]/self._size,
            np.linalg.norm(self._agent_location-self._target_location)/self._size],
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
            observation (numpy array): [agent_x, agent_y, distance_to_target]
            info: none
        """
        # Set the seed of the reset function in gym.Env (parent class)
        super().reset(seed=seed)

        # Initialize agent location to starting location
        self._agent_location = self._starting_location.copy()

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

        # Scale action by max_step and update agent location
        self._agent_location += action * self._max_step_size

        # Compute distance to target
        dist_to_target = np.linalg.norm(self._agent_location-self._target_location)

        # Terminal if within target radius
        terminated = bool(dist_to_target <= self._target_radius)

        # Truncate if max steps reached
        self._step_count += 1
        truncated = self._step_count >= self._max_steps_per_episode

        # Set reward
        reward = float(-dist_to_target/self._size) 
        if terminated:  
            reward += 10.0

        # Re-render environment if in human render mode
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _env_to_screen(self, location):
        """
        Convert environment coordinates to pygame screen coordinates
        
        Args:
            location (numpy array): [x, y] in environment coordinates
            
        Return:
            screen_location (numpy array): [x_pix, y_pix] in screen coordinates
        """
        x_pix = int(location[0] / self._size * self._window_size)
        y_pix = int((self._size - location[1]) / self._size * self._window_size)  # flip y
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
        radius_pix = int(self._target_radius / self._size * self._window_size)
        if radius_pix > 0:
            pygame.draw.circle(canvas, (255, 0, 0), target_center, radius_pix, width=1)

        # Copy canvas to visible window
        self._window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # Update at the specified framerate
        self._clock.tick(self.metadata["render_fps"])
        
    def close(self):
        """Close pygame resources if the window has been initialized and is active"""

        if self._window is not None:
            pygame.display.quit()  
            pygame.quit()
            self._window = None
            self._clock = None