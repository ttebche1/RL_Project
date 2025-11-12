# Test script for the graphics of the static target search environment

from class_single_agent_static_target_search_env import single_agent_static_target_search_env

if __name__ == "__main__":
    # Initialize environment
    env_params = {
        "env_size": 1000,             # Width and length of the environment in meters; 1414 x 1414 = ~2km max distance
        "target_radius": 300.0,         # Radius for "found" condition in meters
        "max_step_size": 10.0,          # Maximum step size in meters
        "max_steps_per_episode": 200,   # Max steps per episode
        "dist_noise_std": 0.5,          # Standard deviation of Gaussian noise added to distance measurements (meters)
        "dist_noise_bias": 0.0          # Constant bias added to distance measurements (meters)
    }
    env = single_agent_static_target_search_env(env_params, render_mode="human")
    obs, info = env.reset()
    done = False
    truncated = False

    # Initialize counts
    cum_reward = 0.0
    step_count = 0

    while not done and not truncated:
        # Sample a random action from the action space
        action = env.action_space.sample()

        # Take a step in the environment
        obs, reward, done, truncated, info = env.step(action)

        # Increment counts
        cum_reward += reward
        step_count += 1

    env.close()
