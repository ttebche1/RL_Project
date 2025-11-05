# Test script for the parameters of the static target search environment

from class_static_target_search_env import static_target_search_env
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    # Initialize environment
    env_params = {
        "env_size": 2000.0,             # Width and length of the environment in meters; 1414 x 1414 = ~2km max distance
        "target_radius": 300.0,         # Radius for "found" condition in meters
        "max_step_size": 10.0,          # Maximum step size in meters
        "max_steps_per_episode": 200,   # Max steps per episode
        "dist_noise_std": 0.5,          # Standard deviation of Gaussian noise added to distance measurements (meters)
        "dist_noise_bias": 0.0          # Constant bias added to distance measurements (meters)
    }
    env = static_target_search_env(env_params)

    # Check environment
    check_env(env)
    print("Environment passed the check successfully.")