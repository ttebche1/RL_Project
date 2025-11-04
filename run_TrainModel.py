# Train robot to map a 5x5 grid environment using Q-learning

# Import dependencies
from class_DQN_Agent import DQN_Agent       # DQN agent
from class_Static_Target_Search_Env import GridWorldEnv # Custom environment
from matplotlib import pyplot as plt
from tqdm import tqdm                       # Creates progress bars for loops
import gymnasium as gym                     # Reinforcement learning module
import numpy as np
import sys                                  # Allow program to exit before end
import torch

if __name__ == "__main__":
    # Set hyperparameters
    N_EPISODES = 5000
    INITIAL_EPSILON = 1.0
    FINAL_EPSILON = 0.05
    EPSILON_DECAY = INITIAL_EPSILON / (N_EPISODES * 0.6)
    DISCOUNT_FACTOR = 0.99
    LEARNING_RATE = 1e-3    
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 20000
    TARGET_UPDATE = 500

    # Initialize variables I want to plot at the end of testing
    total_reward = []

    # Register environment
    gym.register(id="gymnasium_env/GridWorld-v0", entry_point=GridWorldEnv)

    # Make environment
    # Set render_mode to None for no display, "human" for display
    env = gym.make("gymnasium_env/GridWorld-v0", grid_size=5, render_mode=None)

    # Create a DQN agent
    agent = DQN_Agent(
        env=env, 
        learning_rate=LEARNING_RATE, 
        discount_factor=DISCOUNT_FACTOR,
        initial_epsilon=INITIAL_EPSILON, 
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON, 
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE, 
        target_update=TARGET_UPDATE
    )

    # Train agent
    for episode in tqdm(range(N_EPISODES)):
        # Initialize episode
        obs, info = env.reset()
        episode_done = False
        cum_reward = 0.0

        # Do episode
        while not episode_done:
            # Select action
            action = agent.select_action(obs)

            # Take action
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store transition
            agent.store_transition(obs, action, reward, next_obs, terminated)

            # Train agent
            loss = agent.train()

            # Update variables
            cum_reward += reward
            episode_done = terminated or truncated
            obs = next_obs

        # Store cumulative reward for this episode
        total_reward.append(cum_reward)

        # Decay epsilon value
        agent.decay_epsilon()
    
    # Save learned policy weights
    save_path = "trained_policy.pth"
    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"Saved trained policy to {save_path}")

    # Visualize the episode rewards and training error in one figure
    fig, axs = plt.subplots(2, 2)

    # Plot reward vs. episode
    axs[1,0].plot(np.arange(1,N_EPISODES+1), total_reward)
    axs[1,0].set_title("Reward vs. Episode")
    axs[1,0].set_xlabel("Episode")
    axs[1,0].set_ylabel("Reward")

    # Plot training error vs. episode (smoothed)
    if len(agent.training_error) > 0:
        smooth = np.convolve(agent.training_error, np.ones(100)/100, mode='valid')
        axs[1,1].plot(smooth)
    axs[1,1].set_title("Training Error vs. Episode")
    axs[1,1].set_xlabel("Episode")
    axs[1,1].set_ylabel("Training Error")

    plt.tight_layout()
    plt.show()

    # Close environment
    env.close()