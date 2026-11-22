"""Entry point for GateMind.

Usage:
    python main.py train
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
    args = parser.parse_args()

    if args.command == "train":
        print("Training is not implemented yet (step 7).")
    elif args.command == "evaluate":
        print("Evaluation is not implemented yet (step 9).")
    elif args.command == "plot":
        print("Plotting is not implemented yet (step 10).")


if __name__ == "__main__":
    main()