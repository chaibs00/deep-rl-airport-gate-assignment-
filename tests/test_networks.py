"""
Tests for the QNetwork class.
    python -m pytest tests/test_networks.py
"""

import torch

from gatemind.agents.networks import QNetwork, masked_q_values


def test_output_has_one_value_per_action():
    net = QNetwork(obs_dim=52, n_actions=24)
    obs = torch.rand(8, 52)          # a batch of 8 states
    assert net(obs).shape == (8, 24)


def test_masked_actions_are_never_chosen():
    q = torch.tensor([[5.0, 1.0, 3.0]])
    mask = torch.tensor([[False, True, True]])  # action 0 is invalid
    assert masked_q_values(q, mask).argmax(dim=1).item() == 2



def test_dueling_output_has_one_value_per_action():
    from gatemind.agents.networks import DuelingQNetwork
    net = DuelingQNetwork(obs_dim=52, n_actions=24)
    assert net(torch.rand(8, 52)).shape == (8, 24)