# Train SAC DRL model on Static Target Search Environment

import gymnasium as gym
from class_Progress_Bar import Progress_Bar_Callback
from class_Static_Target_Search_Env import Static_Target_Search_Environment 
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == "__main__":
    # Create vectorized environments
    num_envs = 8
    env_fns = [lambda: Static_Target_Search_Environment(render_mode=None) for _ in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Create environment
    env = Static_Target_Search_Environment(render_mode=None)  # turn off rendering for training

    # Initialize SAC agent
    model = SAC(
        "MlpPolicy",                # fully-connected neural network
        vec_env,
        verbose = 0,                # 0=no display, 1=minimal info, 2=debug
        device = "cuda",
        batch_size = 32,            # samples used from the buffer per gradient update
        buffer_size = int(1e6),     # past experiences to store
        learning_starts = 10000,    # exploration timesteps before training starts
        tau = 0.01,                 # target network update rate (slow updates)
        gamma = 0.99,               # discount factor for future rewards (heavily considers future rewards)
        train_freq = 30,            # how often to update the NNs
        gradient_steps = 20,        # how many gradient steps to take during each update
        learning_rate=1e-3,         # how fast the NNs update
        ent_coef="auto",            # automatically adjust weight of entropy in the loss function 
        seed=3
    )

    # Train agent
    total_timesteps = 100_000   # Total timesteps to train the agent
    progress_callback = Progress_Bar_Callback(total_timesteps=total_timesteps)
    model.learn(
        total_timesteps=total_timesteps, 
        callback=progress_callback
    )

    # Save trained model
    model.save("sac_static_target_search")

    # Close environments
    vec_env.close()
