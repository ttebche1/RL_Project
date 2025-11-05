# Test script for the parameters of the static target search environment

from class_static_target_search_env import static_target_search_env
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    # Initialize environment
    env = static_target_search_env()

    # Check environment
    check_env(env)
    print("Environment passed the check successfully.")