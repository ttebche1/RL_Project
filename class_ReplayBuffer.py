import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, buf_size):
        self.obs_buf = np.zeros((buf_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((buf_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((buf_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buf_size, dtype=np.float32)
        self.done_buf = np.zeros(buf_size, dtype=np.float32)

        self.buf_size = buf_size
        self.ptr = 0
        self.num_trans = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.buf_size
        self.num_trans = min(self.num_trans + 1, self.buf_size)
    
    def store_batch(self, obs_batch, act_batch, rew_batch, next_obs_batch, done_batch):
        """
        Store a batch of transitions at once (vectorized, no Python loop).

        Args:
            obs_batch (np.ndarray or torch.Tensor): shape (batch_size, obs_dim)
            act_batch (np.ndarray or torch.Tensor): shape (batch_size, act_dim)
            rew_batch (np.ndarray or torch.Tensor): shape (batch_size,)
            next_obs_batch (np.ndarray or torch.Tensor): shape (batch_size, obs_dim)
            done_batch (np.ndarray or torch.Tensor): shape (batch_size,)
        """
        batch_size = obs_batch.shape[0]

        # Convert to tensors on correct device
        device="cpu"
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        act_tensor = torch.as_tensor(act_batch, dtype=torch.float32, device=device)
        rew_tensor = torch.as_tensor(rew_batch, dtype=torch.float32, device=device)
        next_obs_tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=device)
        done_tensor = torch.as_tensor(done_batch, dtype=torch.float32, device=device)

        # Compute end index in circular buffer
        end = self.ptr + batch_size

        if end <= self.buf_size:
            # No wrap-around
            self.obs_buf[self.ptr:end] = obs_tensor
            self.act_buf[self.ptr:end] = act_tensor
            self.rew_buf[self.ptr:end] = rew_tensor
            self.next_obs_buf[self.ptr:end] = next_obs_tensor
            self.done_buf[self.ptr:end] = done_tensor
        else:
            # Wrap-around
            first_part = self.buf_size - self.ptr
            second_part = batch_size - first_part
            self.obs_buf[self.ptr:] = obs_tensor[:first_part]
            self.obs_buf[:second_part] = obs_tensor[first_part:]
            self.act_buf[self.ptr:] = act_tensor[:first_part]
            self.act_buf[:second_part] = act_tensor[first_part:]
            self.rew_buf[self.ptr:] = rew_tensor[:first_part]
            self.rew_buf[:second_part] = rew_tensor[first_part:]
            self.next_obs_buf[self.ptr:] = next_obs_tensor[:first_part]
            self.next_obs_buf[:second_part] = next_obs_tensor[first_part:]
            self.done_buf[self.ptr:] = done_tensor[:first_part]
            self.done_buf[:second_part] = done_tensor[first_part:]

        # Update pointer and current size
        self.ptr = (self.ptr + batch_size) % self.buf_size
        self.num_trans = min(self.num_trans + batch_size, self.buf_size)


    def sample_batch(self, batch_size, device="cpu"):
        idxs = np.random.randint(0, self.num_trans, size=batch_size)

        batch = dict(
            obs=torch.tensor(self.obs_buf[idxs], device=device),
            act=torch.tensor(self.act_buf[idxs], device=device),
            rew=torch.tensor(self.rew_buf[idxs], device=device),
            next_obs=torch.tensor(self.next_obs_buf[idxs], device=device),
            done=torch.tensor(self.done_buf[idxs], device=device)
        )
        return batch
