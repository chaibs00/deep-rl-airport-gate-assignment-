"""Generates the learning curves and the Gantt chart comparison."""

import os

import numpy as np
import yaml

from gatemind.agents.greedy_agent import GreedyAgent
from gatemind.environement.gate_env import GateAssignmentEnv
from gatemind.utils.plotting import plot_gantt, plot_learning_curves
from gatemind.utils.runner import run_episode

LABELS = {"dqn": "DQN", "double_dqn": "Double DQN", "dueling_dqn": "Dueling Double DQN"}


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    t_cfg, p_cfg = config["training"], config["plots"]
    env = GateAssignmentEnv()

    figures_dir = os.path.join(t_cfg["results_dir"], "figures")
    logs_dir = os.path.join(t_cfg["results_dir"], "logs")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Learning curves, with greedy on the same validation days as reference.
    val_days = range(t_cfg["val_seed_start"], t_cfg["val_seed_start"] + t_cfg["val_days"])
    greedy_val = np.mean([run_episode(env, GreedyAgent(env), d)["reward"] for d in val_days])
    if os.path.isdir(logs_dir) and os.listdir(logs_dir):
        path = os.path.join(figures_dir, "1_learning_curves.png")
        plot_learning_curves(logs_dir, list(config["agents"]), LABELS, greedy_val, path)
        print(f"Saved {path}")
    else:
        print("No training logs found: skipping learning curves.")

    # 2. Gantt charts: greedy vs a trained agent on the same test day.
    day = p_cfg["gantt_day"]
    run_episode(env, GreedyAgent(env), day)
    schedules = {"Greedy": list(env.assignments)}

    name, seed = p_cfg["gantt_agent"], p_cfg["gantt_seed"]
    model_path = os.path.join(t_cfg["results_dir"], "models", f"{name}_seed{seed}.pt")
    if os.path.exists(model_path):
        from scripts.evaluate import load_dqn
        agent = load_dqn(env, config, name, seed)
        run_episode(env, agent, day)
        schedules[LABELS.get(name, name)] = list(env.assignments)
    else:
        print(f"No model {model_path}: Gantt chart shows greedy only.")

    gate_names = [g.name for g in env.airport.gates]
    path = os.path.join(figures_dir, "2_gantt_comparison.png")
    plot_gantt(schedules, gate_names, path)
    print(f"Saved {path}")