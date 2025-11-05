# Test trained SAC model on Static Target Search Environment

import gymnasium as gym
from class_Static_Target_Search_Env import Static_Target_Search_Environment
from stable_baselines3 import SAC

if __name__ == "__main__":
    # Load trained model
    model = SAC.load("sac_static_target_search")

    # Create environment with visual rendering
    env = Static_Target_Search_Environment(render_mode="human")
    obs, info = env.reset()

    # Run one episode
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episode finished. Total reward: {total_reward}")

    # Close environment
    env.close()
