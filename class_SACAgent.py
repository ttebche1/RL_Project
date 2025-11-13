from class_Actor import Actor
from class_Critic import Critic
import torch
import torch.nn.functional as F
import torch.optim as optim

class SACAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, device="cuda"):
        """
        Initialize SAC agent

        Args:
            obs_dim (int): observation dimension
            act_dim (int): action dimension
            lr (float): neural network learning rate
            gamma (float): discount factor (impact of future rewards)
            tau (float) 
            alpha (float)
            device (string): device to run on (cpu or gpu)
        """
        # Set device
        self.device = device

        # Initialize neural networks
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = Critic(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())    # Copy weights of critic network

        # Initialize neural network optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # Initialize SAC parameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def update(self, batch):
        """
        Update SAC agent

        Args:
            batch: batch of data
        """
        device = self.device
        obs = batch['obs'].to(device)
        act = batch['act'].to(device)
        rew = batch['rew'].to(device)
        next_obs = batch['next_obs'].to(device)
        done = batch['done'].to(device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob.unsqueeze(1)
            target = rew.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.gamma * q_next


        # Compute Q-values for current state
        q1, q2 = self.critic(obs, act)

        # Compute difference between current and target Q-values
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # Update critic
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Sample actions given observations
        action, log_prob = self.actor.sample(obs)

        # Compute Q-value for the action
        q1_pi, q2_pi = self.critic(obs, action)
        q_pi = torch.min(q1_pi, q2_pi)

        # Compute actor loss
        actor_loss = (self.alpha * log_prob - q_pi).mean()

        # Update actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            mean, _ = self.actor(obs)
            action = torch.tanh(mean).cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(obs)
            action = action.cpu().detach().numpy()[0]
        return action