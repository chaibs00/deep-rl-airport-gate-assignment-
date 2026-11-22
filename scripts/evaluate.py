"""Evaluates all agents on the same unseen test days and builds the results table."""

import csv
import os

import numpy as np
import yaml

from gatemind.agents.dqn_agent import DQNAgent
from gatemind.agents.greedy_agent import GreedyAgent
from gatemind.agents.random_agent import RandomAgent
from gatemind.environement.gate_env import GateAssignmentEnv
from gatemind.utils.runner import run_episode

METRICS = ["contact_rate", "remote", "walk", "size_mismatch", "reward"]


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def average_over_days(env, agent, days):
    """Plays every test day and averages each metric."""
    results = [run_episode(env, agent, day) for day in days]
    return {m: float(np.mean([r[m] for r in results])) for m in METRICS}


def load_dqn(env, config, name, seed):
    """Rebuilds a trained agent and loads its saved weights."""
    path = os.path.join(config["training"]["results_dir"], "models", f"{name}_seed{seed}.pt")
    if not os.path.exists(path):
        return None
    agent_cfg = {**config["dqn"], **config["agents"][name]}
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, agent_cfg, seed=seed)
    agent.load(path)
    agent.training = False  # no exploration
    return agent


def main():
    config = load_config()
    e_cfg = config["evaluation"]
    env = GateAssignmentEnv()
    days = range(e_cfg["test_seed_start"], e_cfg["test_seed_start"] + e_cfg["test_days"])

    # Each agent has one or more "runs": one per seed. Metrics are averaged over
    # test days within a run, then mean and std are taken across runs.
    runs = {
        "Random": [average_over_days(env, RandomAgent(seed=s), days) for s in range(e_cfg["random_runs"])],
        "Greedy": [average_over_days(env, GreedyAgent(env), days)],
    }

    labels = {"dqn": "DQN", "double_dqn": "Double DQN", "dueling_dqn": "Dueling Double DQN"}
    for name in config["agents"]:
        agent_runs = []
        for seed in config["training"]["seeds"]:
            agent = load_dqn(env, config, name, seed)
            if agent is None:
                print(f"Missing model: {name} seed {seed} (skipped)")
                continue
            agent_runs.append(average_over_days(env, agent, days))
        if agent_runs:
            runs[labels.get(name, name)] = agent_runs

    # Build the table: mean (std) across runs.
    rows = []
    for agent_name, agent_runs in runs.items():
        row = {"agent": agent_name, "runs": len(agent_runs)}
        for m in METRICS:
            values = [r[m] for r in agent_runs]
            row[m] = round(float(np.mean(values)), 2)
            row[f"{m}_std"] = round(float(np.std(values)), 2)
        rows.append(row)

    print(f"\nResults on {len(days)} unseen test days (mean, std across runs)\n")
    print(f"{'Agent':20s} | {'Contact %':>13s} | {'Remote':>11s} | {'Walk (min)':>11s} | "
          f"{'Mismatch %':>12s} | {'Reward':>14s}")
    print("-" * 97)
    for r in rows:
        print(f"{r['agent']:20s} | "
              f"{r['contact_rate']:6.1f} ({r['contact_rate_std']:4.1f}) | "
              f"{r['remote']:4.1f} ({r['remote_std']:3.1f}) | "
              f"{r['walk']:4.2f} ({r['walk_std']:4.2f}) | "
              f"{r['size_mismatch']:5.1f} ({r['size_mismatch_std']:4.1f}) | "
              f"{r['reward']:6.1f} ({r['reward_std']:4.1f})")

    out_dir = os.path.join(config["training"]["results_dir"], "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {out_path}")