# Class for environment that enables an agent to search for a static target

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
        self.size = env_params["env_size"]                                          # Distance from origin in all four directions                             
        self.target_radius = env_params["target_radius"] / self.size                # Radius for "found" condition, normalized
        self.max_steps_per_episode = env_params["max_steps_per_episode"]            # Maximum steps per episode
        self.dist_noise_std = env_params["dist_noise_std"] / self.size              # Standard deviation of Gaussian noise added to distance measurements, normalized    
        self.vel_mag = env_params["velocity"] / self.size                      # Agent velocity magnitude, normalized
        self.angular_gain = (env_params["velocity"]) / env_params["turning_radius"] # Angular gain in rad/s, normalized 
        self.dt = env_params["dt"]                                                  # Timestep in seconds
        self.current_scale = env_params["max_current_fract"]                        # Max current = this fraction of agent velocity      

        # Initialize observation space: 
        # agent's x coordinate
        # agent's y coordinate 
        # distance to target
        # agent's distance to target in x direction
        # agent's distance to target in y direction
        # agent's x coordinate at least measured distance
        # agent's y coordinate at last measured distance
        # agent's x velocity
        # agent's y velocity
        self.observation_space = spaces.Box(
            low = np.array([-1.0, -1.0, 0.0, -2.0, -2.0, -1.0, -1.0, -self.vel_mag, -self.vel_mag], dtype=np.float32),
            high = np.array([1.0, 1.0, 2.83, 2.0, 2.0, 1.0, 1.0, self.vel_mag, self.vel_mag], dtype=np.float32),
            dtype = np.float32
        )

        # Initialize action space: yaw (heading angle) in [-1, 1], will be scaled by angular gain
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Set render mode
        self.render_mode = render_mode  
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
                agent's x velocity (not accounting for current)
                agent's y velocity (not accounting for current)
        """
        return np.array([
            self.agent_loc_vec[0],
            self.agent_loc_vec[1],
            self.dist_to_target_mag,
            self.dist_to_target_vec[0],
            self.dist_to_target_vec[1],
            self.prev_agent_loc_vec[0],
            self.prev_agent_loc_vec[1],
            self.vel_vec[0],
            self.vel_vec[1]],
        dtype=np.float32)
    
    def get_info(self):
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

        # Initialize agent
        self.agent_loc_vec = np.array([0.0, 0.0], dtype=np.float32)   # Center
        self.prev_agent_loc_vec = np.array([0.0, 0.0], dtype=np.float32)
        self.vel_vec = np.array([0.0, 0.0], dtype=np.float32)  
        self.yaw = 0 

        # Initialize target
        self.target_loc_vec = np.random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)   # Random location

        # Initialize distances
        self.dist_to_target_mag = self.compute_dist_to_target()
        self.dist_to_target_vec = self.agent_loc_vec - self.target_loc_vec

        # Initialize current
        current_mag = np.random.uniform(0, self.vel_mag * self.current_scale)
        current_angle = np.random.uniform(0, 2*np.pi)
        self.current_vec = np.array([
            current_mag * np.cos(current_angle),
            current_mag * np.sin(current_angle)],
        dtype=np.float32)

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
        self.yaw += action[0] * self.angular_gain * self.dt # In radians
        vel_vec = np.array([
            self.vel_mag * np.cos(self.yaw),
            self.vel_mag * np.sin(self.yaw)
        ])
        new_agent_loc_vec = self.agent_loc_vec + vel_vec * self.dt + self.current_vec * self.dt

        # Check if new location is in bounds
        if np.any(new_agent_loc_vec < -1.0) or np.any(new_agent_loc_vec > 1.0):
            reward = -1.0   # If out of bounds, remain in the same place and give a penalty
            terminated = False
        else:
            # If in bounds, move agent
            self.prev_agent_loc_vec = self.agent_loc_vec.copy()
            self.agent_loc_vec = new_agent_loc_vec.copy()

            # Update velocity
            self.vel_vec = self.agent_loc_vec - self.prev_agent_loc_vec

            # Update distance to target
            self.dist_to_target_mag = self.compute_dist_to_target()
            self.dist_to_target_vec = self.agent_loc_vec - self.target_loc_vec

            # Terminal if within target radius
            terminated = bool(self.dist_to_target_mag <= self.target_radius)

            # Update reward
            if terminated:
                reward = 10.0
            else:
                # Compute yaw error
                #target_yaw = np.arctan2(self.dist_to_target_vec[1], self.dist_to_target_vec[0])
                #yaw_error = abs(self.yaw - target_yaw)
                #yaw_error = min(yaw_error, 2*np.pi - yaw_error)  # Wrap around
    
                #reward = float(-self.dist_to_target_mag / 2.83 - yaw_error / np.pi)
                reward = float(-self.dist_to_target_mag)
        
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
        dist_to_target = np.linalg.norm(self.agent_loc_vec - self.target_loc_vec)       # Compute distance
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
        # Shift from [-1,1] -> [0,1] for screen
        x = (location[0] + 1.0) / 2.0
        y = (location[1] + 1.0) / 2.0

        # Convert to pixel coordinates
        x_pix = int(x * self.window_size)
        y_pix = int((1.0 - y) * self.window_size)   # Flip y for pygame

        return np.array([x_pix, y_pix])
    
    def draw_current_arrows(self, canvas):
        """
        Draw a background of arrows representing current

        Args:
            canvas (pygame.Surface): canvas being drawn on
        """
        # Visual parameters
        arrow_scale = 3.0       # Length of arrow
        min_arrow_length = 15   # In pixels   
        head_length = 8         # Length of arrowhead (in pixels)
        head_angle = np.pi / 6  # Angle of arrowhead wings from main shaft (30 degrees)
        grid_size = 10          # Grid size for arrows
        color = (200, 200, 200) # Color of arrows

        # Initialize grid
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        
        for gx in x:
            for gy in y:
                # Get current point on grid
                point = np.array([gx, gy])
                point = self.env_to_screen(point)

                # Scale current arrow by pixels
                pixels_per_env_unit = self.window_size / 2                          # Get number of pixels per env unit
                arrow_vec = self.current_vec * arrow_scale * pixels_per_env_unit    # Get scaled arrow vector

                # Verify arrow isn't smaller than the minimum length
                arrow_length = np.linalg.norm(arrow_vec)    # Get arrow length
                if 0 < arrow_length < min_arrow_length:     # Resize arrow to min length if it's smaller
                    arrow_vec = arrow_vec / arrow_length * min_arrow_length

                # Get arrow end point
                end_point = (point[0] + arrow_vec[0], point[1] - arrow_vec[1])

                # Draw arrow if it has a meaningful length
                if np.linalg.norm(arrow_vec) > 2:
                    # Draw arrow line
                    pygame.draw.line(canvas, color, point, end_point, 3)

                    # Calculate direction angle of arrow
                    dx = end_point[0] - point[0]
                    dy = end_point[1] - point[1]
                    angle = np.arctan2(dy, dx)
                    
                    # Calculate the two corner points of the arrowhead
                    left = (
                        end_point[0] - head_length * np.cos(angle - head_angle),
                        end_point[1] - head_length * np.sin(angle - head_angle)
                    )
                    right = (
                        end_point[0] - head_length * np.cos(angle + head_angle),
                        end_point[1] - head_length * np.sin(angle + head_angle)
                    )
                    
                    # Draw arrowhead
                    pygame.draw.polygon(canvas, color, [end_point, left, right])

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
        target_center = tuple(self.env_to_screen(self.target_loc_vec))
        agent_center = tuple(self.env_to_screen(self.agent_loc_vec))

        # Draw current
        self.draw_current_arrows(canvas)
       
        # Draw agent and target with a minimum visible radius
        circle_radius = max(8, int(self.window_size * 0.015)) 
        pygame.draw.circle(canvas, (0, 0, 255), agent_center, circle_radius)    # Agent: blue
        pygame.draw.circle(canvas, (255, 0, 0), target_center, circle_radius)   # Target: red

        # Draw target radius scaled to window size
        pixels_per_unit = self.window_size / 2  
        radius_pix = int(self.target_radius * pixels_per_unit)  
        if radius_pix > 0:
            pygame.draw.circle(canvas, (255, 0, 0), target_center, radius_pix, width=1)

        # Copy canvas to visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # Update at a set frames per second
        self.clock.tick(5)
        
    def close(self):
        """Close pygame resources if the window has been initialized and is active"""

        if self.window is not None:
            pygame.display.quit()  
            pygame.quit()
            self.window = None
            self.clock = None