import torch
import numpy as np
from tqdm import trange
from class_SACAgent import SACAgent
from class_ReplayBuffer import ReplayBuffer
from class_single_agent_static_target_search_env import SingleAgentStaticTargetSearchEnv
import json
import time
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

def make_env(env_params, seed=None):
    def _init():
        env = SingleAgentStaticTargetSearchEnv(env_params)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init

if __name__ == "__main__":
    # ----- Environment parameters -----
    env_params = {
        "env_size": 1000.0,
        "target_radius": 300.0,
        "max_step_size": 30.0,
        "max_steps_per_episode": 200,
        "dist_noise_std": 1
    }

    num_envs = 8  # parallel environments
    env_fns = [make_env(env_params, seed=i) for i in range(num_envs)]
    vec_env = AsyncVectorEnv(env_fns)
    total_steps = 2_000_000
    batch_size = 512
    start_steps = 5000
    device = "cuda"
    gradient_updates_per_step = 1  # Match SB3 default
    target_update_interval = 1  # Update target network every step

    dummy_env = SingleAgentStaticTargetSearchEnv(env_params)
    obs_dims = dummy_env.observation_space.shape[0]
    act_dims = dummy_env.action_space.shape[0]

    # ----- Initialize agent and replay buffer -----
    agent = SACAgent(obs_dims, act_dims, device=device)
    replay_buffer = ReplayBuffer(obs_dims, act_dims, buf_size=200_000)

    # Pre-allocate tensors for data transfer
    obs_tensor = torch.empty((num_envs, obs_dims), dtype=torch.float32, device=device)
    actions_tensor = torch.empty((num_envs, act_dims), dtype=torch.float32, device=device)

    # ----- Initialize state -----
    obs, _ = vec_env.reset()  # returns (obs_array, info)
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    ep_rewards_history = []
    ep_lengths_history = []
    success_history = []

    # Training statistics
    critic_losses = []
    actor_losses = []
    
    # Pre-allocate numpy arrays for batch operations
    all_obs = np.zeros((num_envs, obs_dims), dtype=np.float32)
    all_actions = np.zeros((num_envs, act_dims), dtype=np.float32)
    all_rewards = np.zeros(num_envs, dtype=np.float32)
    all_next_obs = np.zeros((num_envs, obs_dims), dtype=np.float32)
    all_dones = np.zeros(num_envs, dtype=np.bool_)

    start_time = time.time()
    pbar = trange(total_steps, desc="Training SAC")
    
    # Warmup the environment
    print("Warming up environments...")
    for _ in range(10):
        vec_env.step(np.random.uniform(-1, 1, size=(num_envs, act_dims)))

    step_count = 0
    update_step = 0

    for t in pbar:
        step_count += num_envs
        
        # Convert obs to tensor on GPU (more efficient transfer)
        obs_tensor.copy_(torch.as_tensor(obs, dtype=torch.float32, device=device))

        # Select actions - use reparameterization during training
        with torch.no_grad():
            if step_count < start_steps:
                # Random actions for exploration at start
                actions_tensor = torch.rand((num_envs, act_dims), device=device) * 2 - 1
            else:
                actions_tensor, _ = agent.actor.sample(obs_tensor)
        
        actions = actions_tensor.cpu().numpy()

        # Step environments
        next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        done = np.logical_or(terminated, truncated)

        # Batch store in replay buffer (more efficient)
        replay_buffer.store_batch(obs, actions, rewards, next_obs, done)

        # Update episode statistics
        episode_rewards += rewards
        episode_lengths += 1

        # Handle completed episodes
        done_indices = np.where(done)[0]
        for i in done_indices:
            ep_rewards_history.append(episode_rewards[i])
            ep_lengths_history.append(episode_lengths[i])
            success_history.append(1 if rewards[i] > 0 else 0)
            episode_rewards[i] = 0
            episode_lengths[i] = 0

        obs = next_obs

        # ----- Update agent -----
        if replay_buffer.num_trans > start_steps:
            # Multiple gradient updates per environment step (like SB3)
            for _ in range(gradient_updates_per_step):
                batch = replay_buffer.sample_batch(batch_size)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                critic_loss, actor_loss = agent.update(batch)
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
                update_step += 1

            # Update progress bar with smoothed statistics
            if len(ep_rewards_history) > 0:
                window = min(100, len(ep_rewards_history))
                recent_rewards = ep_rewards_history[-window:]
                recent_lengths = ep_lengths_history[-window:]
                recent_success = success_history[-window:] if success_history else []
                
                pbar.set_postfix({
                    "EpRew": np.mean(recent_rewards),
                    "Len": np.mean(recent_lengths),
                    "Critic": np.mean(critic_losses[-10:]) if critic_losses else 0,
                    "Actor": np.mean(actor_losses[-10:]) if actor_losses else 0,
                    "Success": np.mean(recent_success) if recent_success else 0,
                    "Buffer": replay_buffer.num_trans
                })

    # ----- Save model and training statistics -----
    torch.save(agent.actor.state_dict(), "sac_actor.pth")
    torch.save(agent.critic.state_dict(), "sac_critic.pth")

    # Save training history
    training_stats = {
        "episode_rewards": ep_rewards_history,
        "episode_lengths": ep_lengths_history,
        "success_rate": success_history,
        "critic_loss": critic_losses,
        "actor_loss": actor_losses,
        "env_params": env_params,
        "training_time": time.time() - start_time,
        "total_steps": total_steps
    }
    
    with open("sac_training_stats.json", "w") as f:
        json.dump(training_stats, f)

    with open("sac_env_params.json", "w") as f:
        json.dump(env_params, f)

    print(f"Training finished in {time.time() - start_time:.2f} seconds")
    print(f"Final buffer size: {replay_buffer.num_trans}")
    print(f"Average success rate: {np.mean(success_history) if success_history else 0:.3f}")