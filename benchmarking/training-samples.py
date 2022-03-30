from argparse import ArgumentParser
from pathlib import Path

from cnre.data import create_training_samples

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--tasks",
        default=[
            "two_moons",
            "slcp_distractors",
            "slcp",
            "gaussian_mixture",
            "gaussian_linear_uniform",
        ],
    )
    parser.add_argument("--num_simulations", type=int, default=100_000)
    args = parser.parse_args()

    for task in args.tasks:
        create_training_samples(task, args.num_simulations, args.root)
