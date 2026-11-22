"""
Tests for the ReplayBuffer class.
    python -m pytest tests/test_replay_buffer.py
"""

import numpy as np

from gatemind.agents.replay_buffer import ReplayBuffer


def make_buffer(capacity=5):
    return ReplayBuffer(capacity=capacity, obs_dim=3, n_actions=4)


def push_n(buffer, n):
    for i in range(n):
        buffer.push(
            obs=np.full(3, i), action=i % 4, reward=float(i),
            next_obs=np.full(3, i + 1), next_mask=np.ones(4, dtype=bool), done=False,
        )


def test_size_grows_until_capacity():
    buffer = make_buffer(capacity=5)
    push_n(buffer, 3)
    assert len(buffer) == 3
    push_n(buffer, 10)
    assert len(buffer) == 5


def test_oldest_transitions_are_overwritten():
    buffer = make_buffer(capacity=5)
    push_n(buffer, 7)  # rewards 0..6, only 2..6 should remain
    assert sorted(buffer.rewards.tolist()) == [2, 3, 4, 5, 6]


def test_sample_shapes():
    buffer = make_buffer(capacity=10)
    push_n(buffer, 10)
    batch = buffer.sample(4, np.random.default_rng(0))
    assert batch["obs"].shape == (4, 3)
    assert batch["actions"].shape == (4,)
    assert batch["next_masks"].shape == (4, 4)