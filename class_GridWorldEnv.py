# Class for Grid World Environment
# Create for Environment Creation Tutorial

from enum import Enum
from gymnasium import spaces
from typing import Dict
import gymnasium as gym
import numpy as np
import pygame

class Actions(Enum):
    """Provide actions with an integer key"""
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

# Inherit structure of environment from gymnasium.Env
class GridWorldEnv(gym.Env):
    # Contains metadata for a gymnasium environment specifying rendering configurations for how the environment should display visual output
    # render_modes -> defines type of rendering an environment supports
    #       . human -> render the environment in a way that is meant to be viewed by a human
    # render_fps -> frames per second at which the environment should be rendered
    #       . 4 -> environment rendered at 4 frames per second
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size, render_mode=None):
        """Initialize square grid environment
            - Define grid dimensions
            - Define/quantify actions (e.g. left=0, right=1)
            - Map actions to grid (e.g. right=[1,0])
            - Define size of action space (# of actions)
            - Initialize agent location to [0, 0]
            - Initialize empty list of mapped cells
            - Define observation space as agent's current x-y coordinates and all cells mapped flag
            - Initialize step count to 0

            Args:
                render_mode: type of rendering
                    None: default; no display
                    human: render for human view
        """

        # Width and length of grid (i.e. size x size cells)
        self._size = grid_size

        # Size of PyGame window (i.e. window_size x window_size pixels)
        self._window_size = 512 

        # Set start/end location to top left of grid, or [0,0]
        # dtype -> specifies type of values the array will hold
        self._starting_location = np.array([0, 0], dtype=np.int64)

        # Initialize agent location
        self._agent_location = self._starting_location

        # Initialize array that contains x- and y-coordinates of mapped cells
        self._mapped_cells = np.array([self._starting_location], dtype=np.int64)

        # Initialize flag indicating if all cells have been mapped
        self._all_cells_mapped = False

        # Define possible values in observation space
        # Observations: info that the agent receives from the environment
        # Agent's x- and y-coordinates, and flag indicating if all cells have been mapped
        self.observation_space = spaces.Tuple(
            (
                # Agent's location - range of x-coordinate values
                gym.spaces.Discrete(self._size),

                # Agent's location - range of y-coordinate values
                gym.spaces.Discrete(self._size), 

                # Flag indicating if all cells have been mapped
                gym.spaces.Discrete(2)
            )
        )

        # 4 actions: right, up, left, down
        self.action_space = gym.spaces.Discrete(4)

        # Dictionary maps actions to directions on the grid
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),          # right
            Actions.UP.value: np.array([0, -1]),            # up
            Actions.LEFT.value: np.array([-1, 0]),          # left
            Actions.DOWN.value: np.array([0, 1]),           # down
        }

        # Initialize count of number of steps taken
        self._step_count = 0

        # If render_mode is None or one of the valid modes listed in self.metadata["render_modes"]
        # If it is not valid, the program will be stopped and will provide an error message
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        # Set render mode
        self.render_mode = render_mode   

        # Initially, no window created
        self._window = None

        # Initially, no clock is set to control the framerate
        self._clock = None     
    
    def _get_obs(self) -> tuple[int, int]:
        """Return agent's location as a tuple"""
        return (self._agent_location[0], self._agent_location[1], self._all_cells_mapped)
    
    def _get_info(self) -> dict[int, int]:
        """Return number of mapped cells and step count as a dictionary"""
        return {
            "mapped_cells": self._mapped_cells.shape[0],
            "step_count": self._step_count
         }

    def reset(self, seed=None, options=None):
        """Initiate a new episode for an environment
            - Reset list of mapped cells to empty coordinate pair [~,~]
            - Reset step count to 0
            - Set seed of reset function
            - Initialize agent's location to [0, 0]
                Observation should be [0, 0]
            - Reset all cells mapped flag to 0
            
            Args:
                seed: seed for the random number generator; set to None for non-reproducable values
                options: options for resetting the environment
            
            Return:
                observation: agent's location (should be [0, 0])
                info: none
        """
        # Set the seed of the reset function
        # This sets the seed of the reset function in gym.Env (parent class)
        super().reset(seed=seed)

        # Initialize agent location to top left of grid, or [0,0]
        # dtype -> specifies type of values the array will hold
        self._agent_location = np.array([0, 0], dtype=np.int64)

        # Reset the mapped cells to an empty set
        self._mapped_cells = np.array([self._agent_location], dtype=np.int64)

        # Reset all cells mapped flag
        self._all_cells_mapped = False
        
        # Reset count of number of steps taken to 0
        self._step_count = 0
        
        # Get observation info, to return
        observation = self._get_obs()

        # Get info, to return
        info = self._get_info()

        # If in human mode, render the frame (graphic)
        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action: int):
        """Takes the action passed into step()
            - Terminate episode if agent has mapped all cells
            - Take next step
            - Update list of mapped cells
            - Set reward to number of mapped cells

            Args:
                action: integer representing whether the agent goes up, down, left, or right
            
            Return:
                observation: agent's location
                reward: number of mapped cells
                terminated: end game iff agent maps all cells
                truncated: false
                info: none
        """
        # Check if all cells have been mapped
        if (self._mapped_cells.shape[0] == self._size**2):
            self._all_cells_mapped = True

        # An environment is completed if the agent has mapped all the cells and returns to the starting location
        terminated = self._all_cells_mapped

        # No truncation
        truncated = False 

        if not terminated and not truncated:
            # Map the action (e.g. right) to the direction we walk in (e.g. [1, 0])
            direction = self._action_to_direction[action]

            # Move agent
            self._agent_location = self._agent_location + direction

        # Reward an agent for mapping a new cell
        if not np.any(np.all(self._mapped_cells == self._agent_location, axis=1)):
            # Append current location to mapped cells
            self._mapped_cells = np.vstack([self._mapped_cells, self._agent_location])
            reward = 10
        # Penalize agent for remapping a previously mapped cell
        elif not self._all_cells_mapped:
            reward = -5
        # Reward an agent for mapping all cells
        else:
            reward = 25
        
        # Increment step counter
        self._step_count += 1

        # Get observation to return
        observation = self._get_obs()

        # Get info, to return
        info = self._get_info()

        # If in human mode, render next frame (graphic)
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    def _render_frame(self):
        """Render the next frame (graphic)"""

        # If window hasn't been initialized yet...
        if self._window is None:
            # Initialize pygame modules
            pygame.init()           

            # Initialize pygame display
            pygame.display.init()   

            # Create pygame window and set its size
            self._window = pygame.display.set_mode(
                (self._window_size, self._window_size)
            )

        # If clock hasn't been initialized yet...
        # Clock helps control game timing and frame rates
        if self._clock is None:
            self._clock = pygame.time.Clock()

        # Create the 2D graphic (canvas) for the game, of size window_size x window_size pixels
        canvas = pygame.Surface((self._window_size, self._window_size))

        # Fill the window with white
        canvas.fill((255, 255, 255))

        # Calculate the size of a single cell in pixels
        # Size of single cell = total number of pixes / number of cells
        pix_square_size = self._window_size / self._size
       
        # Color mapped cells in green
        for i in self._mapped_cells:
            pygame.draw.rect(canvas, (255, 255, 0), pygame.Rect(pix_square_size*i, (pix_square_size, pix_square_size)))
        
        # Draw agent on canvas
        # pygame.draw.circle(canvas, color [blue], center position of circle, radius of circle)
        pygame.draw.circle(canvas, (0, 0, 255), (self._agent_location+0.5)*pix_square_size, pix_square_size/3)

        # Draw gridlines on canvas
        for x in range(self._size + 1):
            # Draw horizontal lines
            # pygame.draw.line(canvas, color, (starting x- and y- coordinates), (ending x- and y-coordinates))
            # color: greyscale represented by an integer from black (0) to white (255)
            pygame.draw.line(canvas, 0, (0, pix_square_size*x), (self._window_size, pix_square_size*x), width=3)
            
            # Draw vertical lines
            pygame.draw.line(canvas, 0, (pix_square_size*x, 0), (pix_square_size*x, self._window_size), width=3)

        # Copy our drawings from canvas to the visible window
        self._window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self._clock.tick(self.metadata["render_fps"])
        
    def close(self):
        """Close any pygame resources used by the environment"""
        
        # If the window has been initiaized and is active...
        if self._window is not None:
            # Shut down pygame display
            pygame.display.quit()  

            # Shut down pygame library
            pygame.quit()
    
    def get_direction_for_one_action(self, action: int) -> np.array:
        """Return direction mapping for the given action"""
        return self._action_to_direction[action]
    
    def get_grid_size(self) -> int:
        """Return grid size"""
        return self._size