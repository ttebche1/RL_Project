from buffers import ReplayBuffer
from class_Actor import Actor
from class_Args import Args
from class_Critic import Critic
from tqdm import trange
import gymnasium as gym
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro

def make_env(env_id: str, seed: int, env_params: dict):
    """
    Make environment

    Args:
        env_id (str): gym environment ID
        seed (int): random seed for environment and action space
        env_params (dict): environmental parameters

    
        Returns:
            Callable: a function that creates the environment when called
    """
    def thunk():
        env = gym.make(env_id, **env_params)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

def initialize_sim(args: Args):
    """
    Initialize training

    Args:
        args (Args): experiment parameters
    
    Returns:
        envs (gym.vector.SyncVectorEnv): vectorized environments
        actor (Actor): policy network
        q1, q2 (Critic): critic networks
        q1_target, q2_target (Critic): target networks
        actor_optimizer, q_optimizer (torch.optim.Optimizer): optimizers for policy network and Q networks
        a_optimizer (torch.optim.Optimizer or None): optimizer for entropy coefficient
        replay_buff (ReplayBuffer): replay buffer for storing experiences
        target_entropy (float or None): target entropy for automatic tuning
        log_alpha (float): log of entropy coefficient
        device (torch.device): torch device (CPU or GPU)
    """
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Create environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, env_params=args.env_params) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Initialize neural networks
    actor = Actor(envs).to(device)
    q1 = Critic(envs).to(device)
    q2 = Critic(envs).to(device)
    q1_target = Critic(envs).to(device)
    q2_target = Critic(envs).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    q_optimizer = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Initialize entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # Initialize observation space
    envs.single_observation_space.dtype = np.float32
    replay_buff = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    
    return envs, actor, q1, q2, q1_target, q2_target, actor_optimizer, \
        q_optimizer, a_optimizer, replay_buff, target_entropy, alpha, log_alpha, device

if __name__ == "__main__":
    # Get simulation arguments
    args = tyro.cli(Args)

    # Initialize simulation
    envs, actor, q1, q2, q1_target, q2_target, actor_optimizer, q_optimizer, \
        a_optimizer, replay_buff, target_entropy, alpha, log_alpha, device = initialize_sim(args)

    # Start training
    start_time = time.time()            # Record start time
    obs, _ = envs.reset(seed=args.seed) # Reset environment

    pbar = trange(args.total_timesteps)
    for global_step in pbar: # Iterate through timesteps
        # Take an action
        if global_step < args.learning_starts:  # Randomly sample actions until learning starts
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # Take a step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        replay_buff.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = replay_buff.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = q1_target(data.next_observations, next_state_actions)
                qf2_next_target = q2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = q1(data.observations, data.actions).view(-1)
            qf2_a_values = q2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = q1(data.observations, pi)
                    qf2_pi = q2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(q1.parameters(), q1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(q2.parameters(), q2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            # Print reward
            if "final_info" in infos:
                # infos["final_info"] is a list of length num_envs
                finished_returns = [
                    info["episode"]["r"] for info in infos["final_info"] if info is not None
                ]
                if finished_returns:
                    # Show the last finished episode's return, or mean if you want
                    pbar.set_postfix(episodic_return=f"{finished_returns[-1]:.2f}")

    envs.close()
