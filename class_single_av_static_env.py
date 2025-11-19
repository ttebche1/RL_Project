# Class for environment that enables an agent to search for a static target

from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pygame

class SingleAVStaticEnv(gym.Env):
    def __init__(self, env_params):
        """
        Initialize environment

        Args:
            env_params (dict): Parameters for the static target search environment
        """
        # Set random seed
        np.random.seed(None)

        # Initialize parameters
        inv_size = 1 / env_params["env_size"]                                       # Inverse of distance from origin in all four directions                             
        self.target_radius = env_params["target_radius"] * inv_size                 # Radius for "found" condition, normalized
        self.max_steps_per_episode = env_params["max_steps_per_episode"]            # Maximum steps per episode
        self.vel_mag = env_params["velocity"] * inv_size                            # Agent velocity magnitude, normalized
        self.angular_gain = (env_params["velocity"]) / env_params["turning_radius"] # Angular gain in rad/s, normalized 
        self.dt = env_params["dt"]                                                  # Timestep in seconds
        self.current_scale = env_params["max_current_fract"]                        # Max current = this fraction of agent velocity      
        self.dist_noise_std = env_params["dist_noise_std"] * inv_size               # Standard deviation of Gaussian noise added to distance measurements, normalized    
        self.action_noise_std = env_params["action_noise_std"]                      # Action noise
        self.is_auv = env_params["is_auv"]                                          # Whether the agent is an AUV (True) or ASV (False)
        power_coeff = env_params["rho"] * env_params["drag_coeff"] * \
            env_params["area"] / (2 * env_params["eta"])                            # Power coefficient
        power_used = power_coeff * self.vel_mag**3 + self.hotel_power
        self.energy_used = power_used * self.dt

        if self.is_auv:
            self.dvl_noise_std = 0.01 * self.vel_mag                                # DVL noise standard deviation = 1% of velocity magnitude   

        # Initialize observation space: 
        # agent's x coordinate
        # agent's y coordinate 
        # distance to target
        # agent's distance to target in x direction
        # agent's distance to target in y direction
        # agent's x coordinate at least measured distance
        # agent's y coordinate at last measured distance
        # agent's x velocity (AUV) OR change in x distance to target (ASV)
        # agent's y velocity (AUV) OR change in y distance to target (ASV)
        if self.is_auv:
            bound = np.sqrt(2) * self.vel_mag * (1 + self.current_scale) 
        else:
            bound = self.vel_mag

        self.observation_space = spaces.Box( 
            low = np.array([-1.0, -1.0, 0.0, -2.0, -2.0, -1.0, -1.0, -bound, -bound], dtype=np.float32), 
            high = np.array([1.0, 1.0, 2.83, 2.0, 2.0, 1.0, 1.0, bound, bound], dtype=np.float32),
            dtype = np.float32
        )

        # Initialize action space: yaw (heading angle) in [-1, 1], will be scaled by angular gain
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Pre-allocate arrays for speed
        self.obs = np.zeros(self.observation_space.shape, dtype=np.float32) 
        self.true_agent_loc_vec = np.zeros(2, dtype=np.float32)    
        self.dr_agent_loc_vec = np.zeros(2, dtype=np.float32)        
        self.meas_agent_loc_vec = np.zeros(2, dtype=np.float32) 
        self.true_dist_to_target_vec = np.zeros(2, dtype=np.float32)
        self.current_vec = np.zeros(2, dtype=np.float32) 
        self.temp_vel_vec = np.zeros(2, dtype=np.float32)
        self.dvl_vel_vec = np.zeros(2, dtype=np.float32)                # Used if AUV
        self.d_true_dist_to_target_vec = np.zeros(2, dtype=np.float32)  # Used if ASV

        # Set render mode
        self.render_mode = env_params["render_mode"]  
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
                agent's x coordinate; assumed via dead-reckoning (AUV) OR true position (ASV)
                agent's y coordiante; assumed via dead-reckoning (AUV) OR true position (ASV)
                distance to target magnitude, measured
                distance to target in x direction, true
                distance to target in y direction, true
                agent's x coordinate at last measured distance
                agent's y at last measured distance
                agent's x velocity measured with dvl (AUV) OR change in x distance to target (ASV)
                agent's y velocitymeasured with dvl (AUV) OR change in y distance to target (ASV)
        """        
        self.obs[2] = self.meas_dist_to_target_mag
        self.obs[3:5] = self.true_dist_to_target_vec
        self.obs[5:7] = self.meas_agent_loc_vec

        if self.is_auv:
            self.obs[0:2] = self.dr_agent_loc_vec
            self.obs[7:9] = self.dvl_vel_vec
        else:
            self.obs[0:2] = self.true_agent_loc_vec
            self.obs[7:9] = self.d_true_dist_to_target_vec
    
        return self.obs
    
    def get_info(self):
        """
        Return auxiliary information about the environment

        Return:
            info (dict): dictionary containing cumulative energy used
        """
        return {"e": self.cum_energy_used}


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
        self.true_agent_loc_vec[:] = 0.0    # Center
        self.dr_agent_loc_vec[:] = 0.0      
        self.meas_agent_loc_vec[:] = 0.0
        self.yaw = 0 
        self.dvl_vel_vec[:] = 0.0

        # Initialize target
        self.true_target_loc_vec = np.random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)   # Random location

        # Initialize distances
        np.subtract(self.true_agent_loc_vec, self.true_target_loc_vec, out=self.true_dist_to_target_vec)
        true_dist_to_target_mag = np.linalg.norm(self.true_dist_to_target_vec)
        if true_dist_to_target_mag < 1.0 and np.random.rand() > 0.1:
            self.meas_dist_to_target_mag = self.compute_dist_to_target()
        else:  
            self.meas_dist_to_target_mag = 2.83
        self.d_true_dist_to_target_vec[:] = 0.0
        
        # Initialize current
        current_mag = np.random.uniform(0, self.vel_mag * self.current_scale)
        current_angle = np.random.uniform(0, 2 * np.pi)
        self.current_vec[0] = current_mag * np.cos(current_angle)
        self.current_vec[1] = current_mag * np.sin(current_angle)

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
        reward = 0.0

        # Ensure action is within action space
        action += np.random.normal(0, self.action_noise_std, action.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Compute and take new location
        self.yaw += action[0] * self.angular_gain * self.dt # In radians
        self.temp_vel_vec[0] = self.vel_mag * np.cos(self.yaw)
        self.temp_vel_vec[1] = self.vel_mag * np.sin(self.yaw)
        prev_true_agent_loc_vec = self.true_agent_loc_vec.copy()
        self.true_agent_loc_vec = prev_true_agent_loc_vec + self.temp_vel_vec * self.dt + self.current_vec * self.dt

        # Compute energy used
        self.cum_energy_used += self.energy_used

        # Penalize agent if outside of bounds
        if np.any(self.true_agent_loc_vec < -1.0) or np.any(self.true_agent_loc_vec > 1.0):
            reward = -10.0

        if self.is_auv: # Update velocity
            np.subtract(self.true_agent_loc_vec, prev_true_agent_loc_vec, out=self.dvl_vel_vec)
            self.dvl_vel_vec /= self.dt
            self.dvl_vel_vec += np.random.normal(0, self.dvl_noise_std, 2)
        else:           # Update distance to target 90% of the time (10% dropped distance measurements)
            prev_true_dist_to_target_vec = self.true_dist_to_target_vec.copy()
        
        np.subtract(self.true_agent_loc_vec, self.true_target_loc_vec, out=self.true_dist_to_target_vec)
        true_dist_to_target_mag = np.linalg.norm(self.true_dist_to_target_vec)
        if true_dist_to_target_mag < 1.0 and np.random.rand() > 0.1:
            self.meas_dist_to_target_mag = self.compute_dist_to_target()
            self.meas_agent_loc_vec = self.dr_agent_loc_vec.copy() 
        
        # Update based on AUV or ASV
        if self.is_auv:
            self.dr_agent_loc_vec += self.temp_vel_vec * self.dt
        else:
            np.subtract(self.true_dist_to_target_vec, prev_true_dist_to_target_vec, out=self.d_true_dist_to_target_vec)

        # Terminal if within target radius
        terminated = bool(true_dist_to_target_mag <= self.target_radius)

        # Update reward
        if terminated:
            reward = 10.0
        else:
            reward = -true_dist_to_target_mag
        
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
        dist_to_target = np.linalg.norm(self.true_dist_to_target_vec)
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
        target_center = tuple(self.env_to_screen(self.true_target_loc_vec))
        agent_center = tuple(self.env_to_screen(self.true_agent_loc_vec))

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