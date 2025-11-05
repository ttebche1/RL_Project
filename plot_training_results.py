import pandas as pd
import matplotlib.pyplot as plt

def load_training_log(csv_file):
    """
    Load the training CSV and check required columns ('r', 'l', 't', 'env_id')

    Args:
        csv_file (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(csv_file)

    # Check required columns
    for col in ['r', 'l', 't', 'env_id']:
        if col not in df.columns:
            raise ValueError(f"CSV does not contain required column '{col}'.")

    return df

def plot_training_results(df, reward_window=50, episode_window=50):
    """
    Plot training results, overlaying each environment based on 'env_id'.

    Args:
        df (pd.DataFrame): Dataframe containing training log
        reward_window (int): Window size for smoothing rewards
        episode_window (int): Window size for smoothing episode lengths
    """
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # 1. Reward per episode (overlay by env_id)
    for env_id, env_df in df.groupby('env_id'):
        env_df = env_df.reset_index(drop=True)
        env_df['reward_smooth'] = env_df['r'].rolling(reward_window).mean()
        axes[0].plot(env_df['reward_smooth'], alpha=0.7, label=f"{env_id}")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title(f"Smoothed Reward per Episode (Overlayed Environments)")
    axes[0].grid(True)

    # 2. Episode length per episode (overlay by env_id)
    for env_id, env_df in df.groupby('env_id'):
        env_df = env_df.reset_index(drop=True)
        env_df['episode_smooth'] = env_df['l'].rolling(episode_window).mean()
        axes[1].plot(env_df['episode_smooth'], alpha=0.7)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title(f"Smoothed Episode Length per Episode (Overlayed Environments)")
    axes[1].grid(True)

    # 3. Reward over wall-clock time (overlay by env_id)
    for env_id, env_df in df.groupby('env_id'):
        axes[2].plot(env_df['t'], env_df['r'], alpha=0.7)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Reward")
    axes[2].set_title("Reward vs Time (Overlayed Environments)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # User settings
    csv_file = "training_log.csv" # Combined CSV
    reward_window = 50
    episode_window = 50

    # Load CSV
    df = load_training_log(csv_file)

    # Plot training results overlayed
    plot_training_results(df, reward_window, episode_window)
