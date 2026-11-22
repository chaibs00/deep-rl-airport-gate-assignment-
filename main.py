"""Entry point for GateMind.

Usage:
    python main.py train                         # full training (config values)
    python main.py train --episodes 50 --seeds 1 # quick test run
    python main.py evaluate
    python main.py plot

For more information, run:
    python main.py --help
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="GateMind: DQN for airport gate assignment")
    parser.add_argument(
        "command",
        choices=["train", "evaluate", "plot"],
        help="Which part of the project to run",
    )
    parser.add_argument("--episodes", type=int, help="Override the number of training episodes")
    parser.add_argument("--seeds", type=int, help="Override the number of training seeds")
    parser.add_argument(
        "--agent",
        default="all",
        choices=["all", "dqn", "double_dqn", "dueling_dqn"],
        help="Which learning agent to train",
    )
    args = parser.parse_args()

    if args.command == "train":
        from scripts.train import main as train
        train(episodes=args.episodes, seeds=args.seeds, agent=args.agent)
    elif args.command == "evaluate":
        from scripts.evaluate import main as evaluate
        evaluate()
    elif args.command == "plot":
        print("Plotting is not implemented yet (step 10).")


if __name__ == "__main__":
    main()