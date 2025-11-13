import json
import pandas as pd
import matplotlib.pyplot as plt

def load_training_log(csv_file):
    """
    Load the training CSV and check required columns ('r', 'l', 't')

    Args:
        csv_file (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(csv_file, skiprows=1)

    # Check required columns
    for col in ['r', 'l', 't']:
        if col not in df.columns:
            raise ValueError(f"CSV does not contain required column '{col}'.")

    return df

def plot_training_results(df, reward_window, episode_window, success_window, max_steps):
    """
    Plot training results for a single environment

    Args:
        df (pd.DataFrame): Dataframe containing training log
        reward_window (int): Window size for smoothing rewards
        episode_window (int): Window size for smoothing episode lengths
        success_window (int): Window size for smoothing success rates
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # 1. Smoothed Reward per episode
    df['reward_smooth'] = df['r'].rolling(reward_window).mean()
    axes[0].plot(df['reward_smooth'], color='blue')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title(f"Smoothed Reward per Episode")
    axes[0].grid(True)

    # 2. Smoothed Episode length per episode
    df['episode_smooth'] = df['l'].rolling(episode_window).mean()
    axes[1].plot(df['episode_smooth'], color='green')
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title(f"Smoothed Episode Length per Episode")
    axes[1].grid(True)

    # 3. Success rate
    df['success'] = (df['l'] < max_steps).astype(int)    
    df['success_rate'] = df['success'].rolling(success_window).mean() * 100
    axes[2].plot(df['success_rate'], color='purple')
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Success Rate (%)")
    axes[2].set_title(f"Smoothed Success Rate")
    axes[2].set_ylim(0, 100)
    axes[2].grid(True)


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # User settings
    csv_file = "training.monitor.csv"
    reward_window = 75
    episode_window = 75
    success_window = 75

    # Load data
    df = load_training_log(csv_file)
    with open("sac_env_params.json", "r") as f:
        env_params = json.load(f)

    # Plot training results overlayed
    plot_training_results(df, reward_window, episode_window, success_window, env_params["max_steps_per_episode"])
