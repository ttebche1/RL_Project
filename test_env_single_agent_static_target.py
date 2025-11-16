# Test script for the graphics of the static target search environment

from class_single_agent_static_target_search_env import SingleAgentStaticTargetSearchEnv as gym_env

if __name__ == "__main__":
    # Initialize environment
    env_params = {
        "env_size": 1000.0,             # Distance from the origin in all four directions in meters
        "target_radius": 300.0,         # Radius for "found" condition in meters
        "max_steps_per_episode": 200,   # Max steps per episode
        "velocity": 1.0,                # Agent velocity in m/s
        "turning_radius": 300,          # Agent turning radius in meters
        "max_current_fract": 0.5,       # Max current = this fraction of agent velocity 
        "dt": 30,                       # Time step in seconds
        "dist_noise_std": 1.0,          # Standard deviation of Gaussian noise added to distance measurements in meters
        "action_noise_std": 0.1        # Action noise 
    }
    env = gym_env(env_params, render_mode="human")
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
