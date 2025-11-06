# Train SAC DRL model on the static target search environment

# TO DO:
# - Add random noise to distance measurements
# - Add random target locations
# - Use their reward function
#
# Once it's working again:
# - If distance between agent and target greater than 0.9 (normalized to 1km), agent does not receive range measurement
# - Update model to angle-based
# - Add currents
#
# In no particular order:
# - Add moving target
# - Add multiple agents
# - Add 3D environment (depth)
# - Get closer to the target than 300m
# - Add a larger search space than 2km

from class_static_target_search_env import static_target_search_env 
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import json

def create_vec_env(num_envs, env_params):
    """
    Create a vectorized environment with training result logging
    
    Args:
        env_params (dict): Parameters for the static target search environment
    
    Returns:
        DummyVecEnv: Vectorized environment with Monitor logging.
    """
    def make_env(i):
        def _init():
            env = static_target_search_env(env_params)
            if i == 0:
                # Only log first environment directly to a CSV in the current directory
                return Monitor(env, filename=f"training")
            else:
                # Skip Monitor wrapper for other environments
                return env
        return _init

    env_fns = [make_env(i) for i in range(num_envs)]
    return DummyVecEnv(env_fns)

if __name__ == "__main__":
    # User parameters
    num_envs = 32 #8                        # Number of parallel environments
    batch_size = 32                         # Number of samples used from the buffer per gradient update
    buffer_size = int(1e6)                  # Number of past experiences to store
    learning_starts = 10000                 # Number of exploration timesteps to collect before training starts
    tau = 0.01                              # Target network update rate (slow updates)
    gamma = 0.99                            # Discount factor for future rewards (heavily considers future rewards)
    train_freq = 20                         # How often to update the NNs
    gradient_steps = 5                      # How many gradient steps to take during each update
    learning_rate = 3e-4 #1e4               # How fast the NNs update
    target_update_interval = 3000           # How often to update the target NN
    total_timesteps = int(2e6)              # Total timesteps to train the agent
    env_params = {
        "env_size": 1000.0,                 # Distance from the origin in all four directions in meters
        "target_radius": 300.0,             # Radius for "found" condition in meters
        "max_step_size": 30.0,              # Maximum step size in meters
        "max_steps_per_episode": 200,       # Max steps per episode
        "dist_noise_std": 1,                # Standard deviation of Gaussian noise added to distance measurements in meters
    }

    # Create vectorized environments with training result logs
    vec_env = create_vec_env(num_envs=num_envs, env_params=env_params)

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
                ent_coef="auto", seed=3)
    
    # Train agent
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save trained model and environmental parameters
    model.save("sac_static_target_search")
    with open("env_params.json", "w") as f:
        json.dump(env_params, f)

    # Close environments
    vec_env.close()