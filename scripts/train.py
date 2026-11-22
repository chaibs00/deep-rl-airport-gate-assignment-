"""
Training script for the DQN agent.
Trains a DQN agent for each seed and saves the models and training logs.

"""


import csv
import os
import time

import numpy as np
import yaml

from gatemind.agents.dqn_agent import DQNAgent
from gatemind.agents.greedy_agent import GreedyAgent
from gatemind.environement.gate_env import GateAssignmentEnv
from gatemind.utils.runner import run_episode


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(env, agent, days):
    """Average reward over the given days, without exploration."""
    agent.training = False
    rewards = [run_episode(env, agent, day)["reward"] for day in days]
    agent.training = True
    return float(np.mean(rewards))


def train_one_seed(config, run_seed, name="dqn"):
    t_cfg = config["training"]
    env = GateAssignmentEnv()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config["dqn"], seed=run_seed)

    val_days = range(t_cfg["val_seed_start"], t_cfg["val_seed_start"] + t_cfg["val_days"])
    models_dir = os.path.join(t_cfg["results_dir"], "models")
    logs_dir = os.path.join(t_cfg["results_dir"], "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    log = []
    start = time.time()

    for episode in range(1, t_cfg["episodes"] + 1):
        # A new day every episode. Training days never overlap validation or test days.
        day_seed = 100_000 + run_seed * 10_000 + episode
        obs, info = env.reset(seed=day_seed)
        done, total_reward, losses = False, 0.0, []

        while not done:
            action = agent.act(obs, info["action_mask"])
            next_obs, reward, done, _, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, info["action_mask"], done)

            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            total_reward += reward

        remote = sum(1 for _, gate in env.assignments if gate == "REMOTE")
        row = {
            "episode": episode,
            "reward": round(total_reward, 2),
            "remote": remote,
            "epsilon": round(agent.epsilon, 3),
            "loss": round(float(np.mean(losses)), 4) if losses else "",
            "val_reward": "",
        }

        if episode % t_cfg["eval_every"] == 0:
            row["val_reward"] = round(evaluate(env, agent, val_days), 2)
            print(
                f"  [{name} seed {run_seed}] episode {episode:4d} | "
                f"train reward {total_reward:6.1f} | val reward {row['val_reward']:6.1f} | "
                f"epsilon {agent.epsilon:.2f} | {time.time() - start:.0f}s"
            )

        log.append(row)

    agent.save(os.path.join(models_dir, f"{name}_seed{run_seed}.pt"))

    with open(os.path.join(logs_dir, f"{name}_seed{run_seed}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        writer.writeheader()
        writer.writerows(log)


def main(episodes=None, seeds=None):
    config = load_config()
    if episodes is not None:
        config["training"]["episodes"] = episodes
    if seeds is not None:
        config["training"]["seeds"] = list(range(seeds))

    # Reference: greedy agent on the same validation days.
    t_cfg = config["training"]
    env = GateAssignmentEnv()
    val_days = range(t_cfg["val_seed_start"], t_cfg["val_seed_start"] + t_cfg["val_days"])
    greedy = np.mean([run_episode(env, GreedyAgent(env), d)["reward"] for d in val_days])
    print(f"Greedy val reward (reference): {greedy:.1f}\n")

    for run_seed in config["training"]["seeds"]:
        print(f"Training dqn, seed {run_seed}")
        train_one_seed(config, run_seed)

    print(f"\nDone. Models in {t_cfg['results_dir']}/models, logs in {t_cfg['results_dir']}/logs")