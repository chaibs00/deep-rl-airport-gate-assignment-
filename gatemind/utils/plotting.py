"""Figures: learning curves and Gantt charts of gate schedules."""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # save figures to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

CATEGORY_COLORS = {"regional": "#9ecae1", "narrow": "#3182bd", "wide": "#e6550d"}


def read_log(path):
    """Reads a training log and returns (episodes, validation rewards)."""
    episodes, rewards = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["val_reward"] != "":
                episodes.append(int(row["episode"]))
                rewards.append(float(row["val_reward"]))
    return np.array(episodes), np.array(rewards)


def plot_learning_curves(logs_dir, agents, labels, greedy_reward, out_path):
    """Validation reward over training: mean across seeds, with a band for the std."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in agents:
        files = sorted(f for f in os.listdir(logs_dir) if f.startswith(f"{name}_seed"))
        if not files:
            continue
        curves = [read_log(os.path.join(logs_dir, f)) for f in files]
        episodes = curves[0][0]
        rewards = np.array([r for _, r in curves])
        mean, std = rewards.mean(axis=0), rewards.std(axis=0)
        ax.plot(episodes, mean, label=labels.get(name, name))
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2)

    ax.axhline(greedy_reward, color="black", linestyle="--", label="Greedy")
    ax.set_xlabel("Training episode (simulated day)")
    ax.set_ylabel("Validation reward per day")
    ax.set_title("Learning curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_schedule(ax, assignments, gate_names, title):
    """One horizontal bar per flight: row = gate, x = time on the ground."""
    rows = {name: i for i, name in enumerate(gate_names)}
    for flight, gate_name in assignments:
        ax.barh(
            rows[gate_name],
            (flight.departure - flight.arrival) / 60,
            left=flight.arrival / 60,
            height=0.7,
            color=CATEGORY_COLORS[flight.category],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_yticks(range(len(gate_names)))
    ax.set_yticklabels(gate_names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(6, 26)
    ax.set_xticks(range(6, 27, 2))
    ax.set_xticklabels([f"{h % 24:02d}:00" for h in range(6, 27, 2)], fontsize=8)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)


def plot_gantt(schedules, gate_names, out_path):
    """Side-by-side Gantt charts. `schedules` = {title: assignments}."""
    fig, axes = plt.subplots(1, len(schedules), figsize=(7 * len(schedules), 7), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (title, assignments) in zip(axes, schedules.items()):
        _draw_schedule(ax, assignments, gate_names, title)

    legend = [Patch(color=c, label=cat.capitalize()) for cat, c in CATEGORY_COLORS.items()]
    fig.legend(handles=legend, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)