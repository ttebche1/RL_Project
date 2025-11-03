# Train robot to map a 5x5 grid environment using Q-learning

# Import dependencies
from class_GridWorldEnv import GridWorldEnv # Custom environment
from class_QLearningAgent import QAgent     # Q-learning agent
from matplotlib import pyplot as plt
from tqdm import tqdm                       # Creates progress bars for loops
#import dill                                 # To save q-values
import gymnasium as gym                     # Reinforcement learning module
import numpy as np
import sys                                  # Allow program to exit before end

if __name__ == "__main__":
    # Save q-values?
    save = 0
    
    # Set Q-learning algorithm hyperparameters
    learning_rate = 0.001
    n_episodes = 20000     
    start_epsilon = 1
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.05
    discount_factor = 0.99

    # Initialize variables I want to plot at the end of testing
    total_mapped_cells = []
    total_step_count = []
    total_reward = []

    # Register environment
    gym.register(id="gymnasium_env/GridWorld-v0", entry_point=GridWorldEnv)

    # Make environment
    # Set render_mode to None for no display, "human" for display
    env = gym.make("gymnasium_env/GridWorld-v0", grid_size=5, render_mode=None)

    # Create a q-learning agent
    agent = QAgent(env=env, learning_rate=learning_rate, initial_epsilon=start_epsilon, epsilon_decay=epsilon_decay,final_epsilon=final_epsilon,discount_factor=discount_factor)

    # Train agent
    for episode in tqdm(range(n_episodes)):
        # Reinitialize the environment to its starting state
        # obs: initial state of environment after reset (should be [0,0])
        # info: provides additional metadata on environment (should be empty)
        obs, info = env.reset()

        # Keep looping until one episode is done
        episode_done = False
        count = 0
        while not episode_done:
            # Get next action - either random, or based on previously learned values
            # action: stores action chosen
            action = agent.get_action(obs)

            # env.step(action): applies chosen action to the environment
            # observation: agent's location
            # reward: 10 points for a new mapped cell, -5 points for a previously mapped cell
            # terminated: boolean value indicating if episode ended 
            # truncated: set to false
            # info: number of mapped cells, step count
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Update agent
            agent.update(obs, action, reward, terminated, next_obs)

            # Update if the environment is done
            episode_done = terminated or truncated

            # Update current observation for agent.get_action(obs)
            obs = next_obs
        
        # Store the total reward for this episode in the return queue
        total_mapped_cells.append(info["mapped_cells"])
        total_step_count.append(info["step_count"])
        total_reward.append(reward)
  
        # Decay epsilon value
        agent.decay_epsilon()
    
    # Visualize the episode rewards, episode length, and training error in one figure
    fig, axs = plt.subplots(2, 2)

    # Plot # mapped cells vs. episode
    axs[0,0].plot(np.arange(1,n_episodes+1), total_mapped_cells)
    axs[0,0].set_title("# Mapped Cells vs. Episode")
    axs[0,0].set_xlabel("Episode")
    axs[0,0].set_ylabel("# Mapped Cells")

    # Plot step count vs. episode
    axs[0,1].plot(np.arange(1,n_episodes+1), total_step_count)
    axs[0,1].set_title("# Steps vs. Episode")
    axs[0,1].set_xlabel("Episode")
    axs[0,1].set_ylabel("# Steps")

    # Plot step count vs. episode
    axs[1,0].plot(np.arange(1,n_episodes+1), total_reward)
    axs[1,0].set_title("Reward vs. Episode")
    axs[1,0].set_xlabel("Episode")
    axs[1,0].set_ylabel("Reward")

    # Plot training error vs. episode 
    #axs[1].plot(np.arange(1,n_episodes+1), agent.training_error)
    axs[1,1].plot(np.convolve(agent.training_error, np.ones(100)))
    axs[1,1].set_title("Training Error vs. Episode")
    axs[1,1].set_xlabel("Episode")
    axs[1,1].set_ylabel("Training Error vs. Episode")

    plt.tight_layout()
    plt.show()

    # Save learned q-values
    if save == 1:
        q_values = agent.get_q_values();
        dill_file = open("q_values.pkl", "wb")
        dill_file.write(dill.dumps(q_values))
        dill_file.close()

    # Close environment
    env.close()