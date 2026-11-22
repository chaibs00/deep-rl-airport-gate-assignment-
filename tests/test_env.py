"""
Tests for the GateAssignmentEnv.
    python -m pytest tests/test_env.py

"""

import numpy as np
import pytest

from gatemind.environement.gate_env import GateAssignmentEnv


def make_env():
    return GateAssignmentEnv()


def test_reset_returns_valid_observation():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.all(obs >= 0) and np.all(obs <= 1)
    assert info["action_mask"].shape == (env.action_space.n,)


def test_remote_always_in_mask():
    env = make_env()
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        mask = info["action_mask"]
        assert mask[env.airport.remote_index]
        action = env.airport.remote_index
        obs, reward, done, _, info = env.step(action)


def test_full_day_with_random_valid_actions():
    env = make_env()
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=1)
    steps, done = 0, False
    while not done:
        valid = np.flatnonzero(info["action_mask"])
        obs, reward, done, _, info = env.step(int(rng.choice(valid)))
        steps += 1
    assert steps == 100
    assert len(env.assignments) == 100


def test_invalid_action_raises():
    env = make_env()
    env.reset(seed=0)
    env.airport.assign(0, departure_time=10_000)  # B1 busy all day
    with pytest.raises(ValueError):
        env.step(0)


def test_reward_values():
    env = make_env()
    env.reset(seed=0)
    flight = env.flights[0]
    remote = env.airport.remote_index
    assert env._reward(flight, remote) == -10.0

    # Best possible gate: D (closest) for a wide-body -> 10 - 3 * 3/10 - 0 = 9.1
    flight.size = env.airport.sizes["wide"]
    d1 = [g.name for g in env.airport.gates].index("D1")
    assert env._reward(flight, d1) == pytest.approx(9.1)

    # Regional at a wide gate -> extra penalty of 2 size levels
    flight.size = env.airport.sizes["regional"]
    assert env._reward(flight, d1) == pytest.approx(7.1)