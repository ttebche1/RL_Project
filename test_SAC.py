# Test trained SAC model on Static Target Search Environment

from class_single_asv_static_env import SingleASVStaticEnv as asv_env
from class_single_auv_static_env import SingleAUVStaticEnv as auv_env
from stable_baselines3 import SAC
import json

if __name__ == "__main__":
    # Load trained model and environmental parameters
    model = SAC.load("sac_static_target_search")
    with open("sac_env_params.json", "r") as f:
        env_params = json.load(f)

    # Create environment with visual rendering
    env = auv_env(env_params, render_mode="human")
    obs, info = env.reset()

    # Run one episode
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episode finished. Total reward: {total_reward}")

    # Close environment
    env.close()
