# Test trained SAC model on Static Target Search Environment

from class_single_av_static_env import SingleAVStaticEnv as av_env
from stable_baselines3 import SAC
import json

if __name__ == "__main__":
    # Load trained model and environmental parameters
    model = SAC.load("sac_static_target_search")
    with open("sac_env_params.json", "r") as f:
        env_params = json.load(f)
    env_params["render_mode"] = "human"  # Enable visual rendering for testing

    # Create environment with visual rendering
    env = av_env(env_params)
    obs, info = env.reset()

    # Run one episode
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episode finished. Total reward: {total_reward}. Total energy: {info['e']/1000} kJ")

    # Close environment
    env.close()
