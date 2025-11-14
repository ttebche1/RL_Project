# Set simulation parameters

from dataclasses import dataclass, field
import gymnasium as gym
import os

# Register environment
gym.envs.registration.register(
    id="StaticSearch-v0",
    entry_point="class_single_agent_static_target_search_env:SingleAgentStaticTargetSearchEnv"
)

@dataclass
class Args:
    seed: int = 1    
    torch_deterministic: bool = True
    cuda: bool = True

    # Algorithm specific arguments
    env_id: str = "StaticSearch-v0"     # Environment
    total_timesteps: int = 500000       # Total timesteps
    num_envs: int = 16                  # Number of parallel environments
    buffer_size: int = 200000           # Replay buffer size
    gamma: float = 0.99                 # Discount factor
    tau: float = 0.01                   # Target smoothing coefficient
    batch_size: int = 512               # Batch size of sample from replay buffer
    learning_starts: int = 5000         # Timestep to start learning
    policy_lr: float = 3e-4             # Learning rate of policy neural network
    q_lr: float = 3e-3                  # Learning rate of Q network neural network
    policy_frequency: int = 4           # Frequency of training policy
    target_network_frequency: int = 1   # Frequency of updating target network
    alpha: float = 0.2                  # Entropy regularization coefficient
    autotune: bool = True               # Automatically tune entropy coefficient

    # Environmental parameters
    env_params: dict = field(default_factory=lambda: {
        "env_size": 1000.0,
        "target_radius": 300.0,
        "max_step_size": 30.0,
        "max_steps_per_episode": 200,
        "dist_noise_std": 1,
    })
