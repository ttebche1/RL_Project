# Class for environment that enables an agent to search for a static target

from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pygame

class SingleAgentStaticTargetSearchEnv(gym.Env):
    def __init__(self, **kwargs):
        """
        Initialize environment

        Args:
            env_params (dict): Parameters for the static target search environment
            render_mode: mode for rendering the environment; can be None or "human"
        """
        # Set random seed
        np.random.seed(None)

        # Initialize parameters
        self.env_size = kwargs.get("env_size")                                  # Distance from origin in all four directions                                   
        self.target_radius = kwargs.get("target_radius") / self.env_size        # Radius for "found" condition, normalized
        self.max_step_size = kwargs.get("max_step_size") / self.env_size        # Max step size in meters, normalized
        self.max_steps_per_episode = kwargs.get("max_steps_per_episode", 200)   # Max steps per episode
        self.dist_noise_std = kwargs.get("dist_noise_std", 1) / self.env_size   # Standard deviation of Gaussian noise added to distance measurements, normalized  
        self.starting_location = np.array([0.0, 0.0], dtype=np.float32)         # Agent starting location

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
            low = np.array([-1.0, -1.0, 0.0, -1.0, -1.0, -self.max_step_size, -self.max_step_size, 
                            -self.max_step_size, -2*self.max_step_size, -2*self.max_step_size], dtype=np.float32),
            high = np.array([1.0, 1.0, 2.83, 1.0, 1.0, self.max_step_size, self.max_step_size, 
                             self.max_step_size, 2*self.max_step_size, 2*self.max_step_size], dtype=np.float32),
            dtype = np.float32
        )

        # Initialize action space: delta_x, delta_y in [-1, 1], will be scaled by self._max_step
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Set render mode
        self.render_mode = kwargs.get("render_mode")   
        self.window_size = 512  
        self.window = None
        self.clock = None    

        # Reset environment
        self.reset() 
    
    def get_obs(self):
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
            self.agent_location[0],
            self.agent_location[1],
            self.dist_to_target,
            self.prev_agent_location[0],
            self.prev_agent_location[1],
            self.dist_change,
            self.velocity[0],
            self.velocity[1],
            self.acceleration[0],
            self.acceleration[1]],
        dtype=np.float32)
    
    def get_info(self):
        info = {}
        if self.step_count >= self.max_steps_per_episode:
            info["final_observation"] = self.get_obs()
        return info

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
        self.agent_location = self.starting_location.copy()                                       # Center
        self.prev_agent_location = self.starting_location.copy()
        self.target_location = np.random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32) # Random target location

        # Initialize distances
        self.dist_to_target = self.compute_dist_to_target()
        self.dist_change = 0.0

        # Initialize current
        self.current = (np.random.uniform(-1/3, 1/3, size=(2,)).astype(np.float32)) * self.max_step_size

        # Initialize velocity and acceleration
        self.velocity = np.array([0.0, 0.0], dtype=np.float32) 
        self.prev_velocity = np.array([0.0, 0.0], dtype=np.float32) 
        self.acceleration = np.array([0.0, 0.0], dtype=np.float32)   

        # Initialize step count
        self.step_count = 0

        # Re-render environment if in human render mode
        if self.render_mode == "human":
            self.render_frame()

        # Return observation and information
        return self.get_obs(), self.get_info()
    
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
        new_location = self.agent_location + action * self.max_step_size + self.current

        # Check if new location is in bounds
        if np.any(new_location < -1.0) or np.any(new_location > 1.0):
            reward = -1.0   # If out of bounds, remain in the same place and give a penalty
            terminated = False
        else:
            # If in bounds, move agent
            self.prev_agent_location = self.agent_location.copy()
            self.agent_location = new_location.copy()

            # Update distance to target
            prev_dist_to_target = self.dist_to_target.copy()
            self.dist_to_target = self.compute_dist_to_target()
            self.dist_change = prev_dist_to_target - self.dist_to_target

            # Update velocity and acceleration
            self.prev_velocity = self.velocity.copy()
            self.velocity = self.agent_location - self.prev_agent_location
            self.acceleration = self.velocity - self.prev_velocity

            # Terminal if within target radius
            terminated = bool(self.dist_to_target <= self.target_radius)

            # Update reward
            if terminated:
                reward = 10.0
            else:
                reward = float(-self.dist_to_target)
        
        # Truncate if max steps reached
        self.step_count += 1
        truncated = self.step_count >= self.max_steps_per_episode

        # Re-render environment if in human render mode
        if self.render_mode == "human":
            self.render_frame()

        return self.get_obs(), reward, terminated, truncated, self.get_info()
    
    def compute_dist_to_target(self):
        """
        Computes distance from agent to target

        Return:
            dist_to_target (float): distance from agent to target, normalized
        """
        dist_to_target = np.linalg.norm(self.agent_location - self.target_location)     # Compute distance
        dist_to_target += np.random.normal(0.01 * dist_to_target, self.dist_noise_std)  # Add noise
        dist_to_target = max(0.0, dist_to_target)                                       # Remove negative distances

        return dist_to_target
    
    def env_to_screen(self, location):
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
        x_pix = int(x_scaled * self.window_size)
        y_pix = int((1.0 - y_scaled) * self.window_size)  # flip y for pygame
        return np.array([x_pix, y_pix])

    def render_frame(self):
        """Render the next frame"""

        # Initialize window if it hasn't been created yet
        if self.window is None:
            pygame.init()           
            pygame.display.init()   
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        # Initialize clock if it hasn't been created yet
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Create white canvas of size window_size x window_size pixels
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Convert coordinates (flip y)
        target_center = tuple(self.env_to_screen(self.target_location))
        agent_center = tuple(self.env_to_screen(self.agent_location))
       
        # Draw agent and target with a minimum visible radius
        circle_radius = max(8, int(self.window_size * 0.015))  # slightly larger
        pygame.draw.circle(canvas, (0, 0, 255), agent_center, circle_radius)  # agent: blue
        pygame.draw.circle(canvas, (255, 0, 0), target_center, circle_radius)  # target: red

        # Draw target radius scaled to window size
        pixels_per_unit = self.window_size / 2  
        radius_pix = int(self.target_radius * pixels_per_unit)  
        if radius_pix > 0:
            pygame.draw.circle(canvas, (255, 0, 0), target_center, radius_pix, width=1)

        # Copy canvas to visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # Update at 4 frames per second
        self.clock.tick(4)
        
    def close(self):
        """Close pygame resources if the window has been initialized and is active"""

        if self.window is not None:
            pygame.display.quit()  
            pygame.quit()
            self.window = None
            self.clock = None