

"""Replay buffer: stores past transitions and samples random mini-batches."""

import numpy as np


class ReplayBuffer:

    def __init__(self, capacity, obs_dim, n_actions):
        self.capacity = capacity
        self.position = 0  # where the next transition is written
        self.size = 0       # how many transitions are stored

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_masks = np.zeros((capacity, n_actions), dtype=bool)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, obs, action, reward, next_obs, next_mask, done):
        """Store one transition. When full, the oldest one is overwritten."""
        i = self.position
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.next_masks[i] = next_mask
        self.dones[i] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        """Random mini-batch of stored transitions."""
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "next_masks": self.next_masks[idx],
            "dones": self.dones[idx],
        }

    def __len__(self):
        return self.size