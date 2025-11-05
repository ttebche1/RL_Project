# Train SAC DRL model on Static Target Search Environment

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
# - switch to sbx
# - use their reward function

from class_Static_Target_Search_Env import Static_Target_Search_Environment 
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == "__main__":
    # Create vectorized environments
    num_envs = 12
    env_fns = [lambda: Static_Target_Search_Environment(render_mode=None) for _ in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Initialize SAC agent
    model = SAC(
        "MlpPolicy",                    # Fully-connected neural network
        vec_env,
        verbose = 0,                    # 0=no display, 1=minimal info, 2=debug
        device = "cuda",
        batch_size = 32,                # Samples used from the buffer per gradient update
        buffer_size = int(1e6),         # Past experiences to store
        learning_starts = 10000,        # Exploration timesteps before training starts
        tau = 0.01,                     # Target network update rate (slow updates)
        gamma = 0.99,                   # Discount factor for future rewards (heavily considers future rewards)
        train_freq = 20,                # How often to update the NNs
        gradient_steps = 5,             # How many gradient steps to take during each update
        learning_rate = 1e-4,           # How fast the NNs update
        target_update_interval = 3000,  # How often to update the target NN
        ent_coef = "auto",              # Automatically adjust weight of entropy in the loss function 
        seed = 3
    )

    # Train agent
    model.learn(
        total_timesteps = int(2e6), # Total timesteps to train the agent
        progress_bar = True
    )

    # Save trained model
    model.save("sac_static_target_search")

    # Close environments
    vec_env.close()