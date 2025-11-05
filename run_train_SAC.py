# Train SAC DRL model on the static target search environment

# TO DO:
# Add graphs to plot the training metrics
# # In no particular order:
# - Add currents
# - Add moving target
# - Add dropped comms
# - Randomize starting location of agent
# - Randomize static target location
# - Add multiple agents
# - Limited power
# - Update model to angle-based
# - use their reward function

from class_static_target_search_env import static_target_search_env 
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import pandas as pd
import shutil

def create_vec_env(num_envs):
    """
    Create a vectorized environment with Monitor logging.
    
    Args:
        num_envs (int): Number of parallel environments.
        log_dir (str): Folder to save Monitor CSV logs.
        render_mode: Pass-through render_mode for your environment.
    
    Returns:
        DummyVecEnv: Vectorized environment with Monitor logging.
    """
    # Create directory for training results
    log_dir = "monitor_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create environments with training result logs
    env_fns = [
        lambda i=i: Monitor(
            static_target_search_env(render_mode=None),
            filename=os.path.join(log_dir, f"env_{i}.csv")
        )
        for i in range(num_envs)
    ]

    # Create and return vectorized environment
    return DummyVecEnv(env_fns)

if __name__ == "__main__":
    # User parameters
    num_envs = 12                   # Number of parallel environments
    batch_size = 32                 # Number of samples used from the buffer per gradient update
    buffer_size = int(1e6)          # Number of past experiences to store
    learning_starts = 10000         # Number of exploration timesteps to collect before training starts
    tau = 0.01                      # Target network update rate (slow updates)
    gamma = 0.99                    # Discount factor for future rewards (heavily considers future rewards)
    train_freq = 20                 # How often to update the NNs
    gradient_steps = 5              # How many gradient steps to take during each update
    learning_rate=1e-4              # How fast the NNs update
    target_update_interval = 3000   # How often to update the target NN
    total_timesteps = int(2e6)      # Total timesteps to train the agent

    # Create vectorized environments with training result logs
    vec_env = create_vec_env(num_envs=num_envs)

    # Initialize SAC agent
    # "MlpPolicy" = fully-connected neural network
    # verbose: 0=no output, 1=minimal info, 2=debug info
    # device: "cuda"=GPU, "cpu"=CPU
    # ent_coef="auto": automatically adjust weight of entropy in the loss function
    # seed: set random seed for reproducibility
    model = SAC("MlpPolicy", vec_env, verbose=0,                    
        device="cuda", batch_size=batch_size, buffer_size=buffer_size,         
        learning_starts=learning_starts, tau=tau, gamma=gamma,                   
        train_freq=train_freq, gradient_steps=gradient_steps,             
        learning_rate=learning_rate, target_update_interval=target_update_interval,
        ent_coef="auto", seed = 3
    )

    # Train agent
    model.learn(total_timesteps = total_timesteps, progress_bar = True)

    # Save trained model
    model.save("sac_static_target_search")

    # Close environments
    vec_env.close()