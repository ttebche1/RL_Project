# Test script for the parameters of the static target search environment

from class_static_target_search_env import static_target_search_env
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    # Initialize environment
    env = static_target_search_env(env_size=100.0, target_radius=5.0, max_step_size=10.0,
                                   max_steps_per_episode=100, dist_noise_std=1.0, dist_noise_bias=0.5)

    # Check environment
    check_env(env)
    print("Environment passed the check successfully.")