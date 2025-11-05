# Test script for the graphics of the static target search environment

from class_static_target_search_env import static_target_search_env

if __name__ == "__main__":
    # Initialize environment
    env = static_target_search_env(render_mode="human")
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
