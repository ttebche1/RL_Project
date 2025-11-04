import numpy as np
import time
from class_Static_Target_Search_Env import Static_Target_Search_Environment 

# Create environment with human rendering
env = Static_Target_Search_Environment(render_mode="human")

# Number of episodes to test
num_episodes = 1

for episode in range(num_episodes):
    # Reset environment
    obs, info = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        # Sample a random action from the action space
        action = env.action_space.sample()
        
        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)

        # Print observation and reward (optional)
        print(f"Obs: {obs}, Reward: {reward:.3f}, Terminated: {terminated}, Truncated: {truncated}")

        # Slow down visualization
        time.sleep(0.05)

env.close()
