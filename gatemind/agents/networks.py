

"""Neural networks that approximate Q(s, a)."""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Takes a state and outputs one Q-value per action (gate)."""

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.layers(obs)


def masked_q_values(q_values, mask):
    """Set the Q-value of invalid actions to -infinity so they are never chosen."""
    return q_values.masked_fill(~mask, float("-inf"))