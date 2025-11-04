# Test robot's movement in a 5x5 grid using the learned q-values

# Import dependencies
from class_Static_Target_Search_Env import GridWorldEnv # Custom environment
from class_DQN_Agent import DQN_Agent       # Q-learning agent
import gymnasium as gym                     # Reinforcement learning module
import torch

if __name__ == "__main__":
    # Make environment with human renderring to visualize agent movement
    gym.register(id="gymnasium_env/GridWorld-v0", entry_point=GridWorldEnv)
    env = gym.make("gymnasium_env/GridWorld-v0", grid_size=5, render_mode="human")
    obs, _ = env.reset()

    # Create agent with evaluation settings
    agent = DQN_Agent(
        env=env,
        learning_rate=0.001,    # Doesn't matter for evaluation
        discount_factor=0.99,
        initial_epsilon=0,      # No exploration during testing
        final_epsilon=0,
        epsilon_decay=0,
        buffer_capacity=1,
        batch_size=1,
        target_update=1
    )

    # Load saved policy
    agent.policy_net.load_state_dict(torch.load("trained_policy.pth"))
    agent.policy_net.eval()  # Set to evaluation mode

    # Keep looping until one episode is done
    episode_done = False
    while not episode_done:
            # Get action from learned q-values
            action = agent.select_action(obs)

            # Take action in environment
            obs, reward, terminated, truncated, _ = env.step(action)

            # Keep looping until game ends
            episode_done = terminated
        
    # Close environment
    env.close()