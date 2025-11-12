# Train SAC DRL model on the static target search environment
#
# Multi-agents:
# - Add multiple agents w/ full observability
# - Remove full observability, but they know each other's distance to target
# - Add dropped comms: 10% drop rate, and If distance between agent and target greater than 0.9 (normalized to 1km), agent does not receive range measurement
# - Try no comms
# - Add a larger search space than 2km
#
# Speed:
# - Does prompting it with a non-ML search algorithm help?
# - Curriculum training 
# - Automatically tune hyperparameters
#
# Stochasticity:
# - Add currents; must improve observability (LSTM)
# - Add particle filter for target estimation; add moving target w/ trailing
#
# Later:
# - Update model to angle-based
# - Add baseline comparison
# - Add 3D environment (depth)
# - Add complex comms things such as doppler
# - Turn it into a whale problem!

from class_single_agent_static_target_search_env import single_agent_static_target_search_env 
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import json
import time

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
            env = single_agent_static_target_search_env(env_params)
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
    # Record start time
    start_time = time.perf_counter()

    # User parameters
    num_envs = 16 #32       # Number of parallel environments
    batch_size = 512        # Number of samples used from the buffer per gradient update
    buffer_size = 200000    # Number of past experiences to store
    learning_starts = 5000  # Number of exploration timesteps to collect before training starts
    tau = 0.01              # Target network update rate (slow updates)
    gamma = 0.99            # Discount factor for future rewards (heavily considers future rewards)
    train_freq = 4 #20      # How often to update the NNs
    gradient_steps = 4 #5   # How many gradient steps to take during each update
    learning_rate = 3e-4 #1e-4 # How fast the NNs update
    #target_update_interval = 500 #3000 # How often to update the target NN
    total_timesteps = int(2e6) #int(1.5e6) # Total timesteps to train the agent
    env_params = {
        "env_size": 1000.0,             # Distance from the origin in all four directions in meters
        "target_radius": 300.0,         # Radius for "found" condition in meters
        "max_step_size": 30.0,          # Maximum step size in meters
        "max_steps_per_episode": 200,   # Max steps per episode
        "dist_noise_std": 1,            # Standard deviation of Gaussian noise added to distance measurements in meters
    }

    # Create vectorized environments with training result logs
    vec_env = create_vec_env(num_envs=num_envs, env_params=env_params)

    model = SAC("MlpPolicy", vec_env, verbose=1,
                device="cuda", batch_size=batch_size, buffer_size=buffer_size,         
                learning_starts=learning_starts, tau=tau, gamma=gamma,                   
                train_freq=train_freq, gradient_steps=gradient_steps,             
                learning_rate=learning_rate, #target_update_interval=target_update_interval,
                ent_coef="auto", seed=None)
    
    # Train agent
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save trained model and environmental parameters
    model.save("sac_static_target_search")

    with open("env_params.json", "w") as f:
        json.dump(env_params, f)

    # Close environments
    vec_env.close()

    # Display end time
    end_time = time.perf_counter()
    print(f"Runtime: {end_time - start_time:.4f} seconds")