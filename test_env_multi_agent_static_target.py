# Test script with coordinated agent movements

from class_multi_agent_static_target_search_env import MultiAgentStaticTargetSearchEnv
import numpy as np

def test_coordinated_movement():
    # Initialize environment
    env_params = {
        "num_agents": 2,
        "env_size": 1000,
        "target_radius": 300.0,
        "max_step_size": 20.0,  # Larger steps for faster testing
        "max_steps_per_episode": 100,
        "dist_noise_std": 1.0,
    }
    
    # Create environment
    env = MultiAgentStaticTargetSearchEnv(env_params, render_mode="human")
    obs, info = env.reset()
    step_count = 0
    max_steps = 50  # Test for max 50 steps for quick visualization

    # Test script for the graphics of the multi-agent static target search environment

from class_multi_agent_static_target_search_env import MultiAgentStaticTargetSearchEnv
import numpy as np

if __name__ == "__main__":
    # user settings
    env_params = {
        "num_agents": 2,               # Number of agents
        "env_size": 1000,              # Width and length of the environment in meters
        "target_radius": 300.0,        # Radius for "found" condition in meters
        "max_step_size": 10.0,         # Maximum step size in meters
        "max_steps_per_episode": 200,  # Max steps per episode
        "dist_noise_std": 0.5,         # Standard deviation of Gaussian noise added to distance measurements (meters)
    }

    # Initialize environment
    env = MultiAgentStaticTargetSearchEnv(env_params, render_mode="human")
    obs, info = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        # Sample random actions for both agents
        actions = {
            "agent_0": env.action_space.sample(),
            "agent_1": env.action_space.sample()
        }

        # Take a step in the environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Check if episode should end
        done = terminated["__all__"]
        truncated = truncated["__all__"]

    # Close environment
    env.close()