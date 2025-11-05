# Test script for Static Target Search Environment

from class_Static_Target_Search_Env import Static_Target_Search_Environment

if __name__ == "__main__":
    # Initialize environment
    env = Static_Target_Search_Environment(render_mode="human")
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
