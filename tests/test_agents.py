"""
Tests for the agents.
    python -m pytest tests/test_agents.py
"""



import numpy as np

from gatemind.agents.greedy_agent import GreedyAgent
from gatemind.agents.random_agent import RandomAgent
from gatemind.environement.gate_env import GateAssignmentEnv
from gatemind.utils.runner import run_episode


def test_agents_only_choose_valid_actions():
    env = GateAssignmentEnv()
    for agent in [RandomAgent(seed=0), GreedyAgent(env)]:
        obs, info = env.reset(seed=0)
        done = False
        while not done:
            action = agent.act(obs, info["action_mask"])
            assert info["action_mask"][action]
            obs, _, done, _, info = env.step(action)


def test_greedy_beats_random_on_average():
    env = GateAssignmentEnv()
    seeds = range(10)
    random_scores = [run_episode(env, RandomAgent(seed=s), s)["reward"] for s in seeds]
    greedy_scores = [run_episode(env, GreedyAgent(env), s)["reward"] for s in seeds]
    assert np.mean(greedy_scores) > np.mean(random_scores)