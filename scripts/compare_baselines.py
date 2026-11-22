"""
    Compares the random and greedy agents on the same 20 simulated days.

    python -m scripts.compare_baselines
"""

import numpy as np

from gatemind.agents.greedy_agent import GreedyAgent
from gatemind.agents.random_agent import RandomAgent
from gatemind.environement.gate_env import GateAssignmentEnv
from gatemind.utils.runner import run_episode

env = GateAssignmentEnv()
seeds = range(1000, 1020)  # test days

agents = {
    "Random": lambda s: RandomAgent(seed=s),
    "Greedy": lambda s: GreedyAgent(env),
}

print(f"{'Agent':8s} | {'Reward':>14s} | {'Remote stands':>14s}")
for name, make_agent in agents.items():
    results = [run_episode(env, make_agent(s), s) for s in seeds]
    rewards = [r["reward"] for r in results]
    remotes = [r["remote"] for r in results]
    print(f"{name:8s} | {np.mean(rewards):7.1f} ({np.std(rewards):4.1f}) | "
          f"{np.mean(remotes):7.1f} ({np.std(remotes):4.1f})")