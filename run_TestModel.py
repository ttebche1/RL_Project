# Test robot's movement in a 5x5 grid using the learned q-values

# Import dependencies
from class_GridWorldEnv import GridWorldEnv # Custom environment
from class_QLearningAgent import QAgent     # Q-learning agent
from matplotlib import pyplot as plt
import gymnasium as gym                     # Reinforcement learning module
import dill                                 # For loading learned q-values        
import numpy as np

if __name__ == "__main__":
    # Load learned q-values
    q_values = dill.load(open (r"q_values.pkl","rb"))
    
    # Initialize variable to hold total number of mapped cells at the end of each episode
    total_mapped_cells = []

    # Register environment
    gym.register(id="gymnasium_env/GridWorld-v0", entry_point=GridWorldEnv)

    # Make environment
    # Set render_mode to None for no display, "human" for display
    env = gym.make("gymnasium_env/GridWorld-v0", grid_size=5, render_mode="human")

    # Reset environment
    obs, info = env.reset()

    # Keep looping until one episode is done
    episode_done = False
    prev_mapped_cell_count = -1
    while not episode_done:
            # Get action from learned q-values
            action = int(np.argmax(q_values[obs]))

            # env.step(action): applies chosen action to the environment
            # observation: agent's location
            # reward: 10 points for a new mapped cell, -5 points for a previously mapped cell
            # terminated: boolean value indicating if episode ended 
            # truncated: set to false
            # info: number of mapped cells, step count
            obs, reward, terminated, truncated, info = env.step(action)

            # Check if the mapped cell count has changed
            if info["mapped_cells"] != prev_mapped_cell_count:
                # Print number of mapped cells
                print("Number of mapped cells: ", info["mapped_cells"])

                # Update the previous count
                prev_mapped_cell_count = info["mapped_cells"]

            # Keep looping until game ends
            episode_done = terminated
        
    # Close environment
    env.close()