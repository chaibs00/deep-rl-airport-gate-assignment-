"""
Tests for the DQN agent.
    python -m pytest tests/test_dqn_agent.py
"""

import numpy as np
import yaml

from gatemind.agents.dqn_agent import DQNAgent
from gatemind.environement.gate_env import GateAssignmentEnv


def make_agent_and_env(**overrides):
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)["dqn"]
    config.update(overrides)
    env = GateAssignmentEnv()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config, seed=0)
    return agent, env


def play_one_day(agent, env, seed=0):
    obs, info = env.reset(seed=seed)
    done, losses = False, []
    while not done:
        action = agent.act(obs, info["action_mask"])
        assert info["action_mask"][action]
        next_obs, reward, done, _, info = env.step(action)
        agent.remember(obs, action, reward, next_obs, info["action_mask"], done)
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
        obs = next_obs
    return losses


def test_actions_are_always_valid():
    agent, env = make_agent_and_env()
    play_one_day(agent, env)
    agent.training = False
    play_one_day(agent, env, seed=1)


def test_epsilon_decreases():
    agent, _ = make_agent_and_env(epsilon_decay_steps=100)
    start = agent.epsilon
    agent.steps = 50
    assert agent.epsilon < start
    agent.steps = 1000
    assert np.isclose(agent.epsilon, agent.cfg["epsilon_end"])


def test_learning_produces_finite_losses():
    agent, env = make_agent_and_env(learning_starts=64)
    losses = play_one_day(agent, env)
    assert len(losses) > 0
    assert all(np.isfinite(losses))


def test_double_and_dueling_variants_learn():
    for options in [{"double": True}, {"double": True, "dueling": True}]:
        agent, env = make_agent_and_env(learning_starts=64, **options)
        losses = play_one_day(agent, env)
        assert len(losses) > 0
        assert all(np.isfinite(losses))