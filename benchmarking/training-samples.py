from argparse import ArgumentParser
from pathlib import Path

from cnre.data.presampled import create_training_samples

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
    parser.add_argument("--extra_theta_factor", type=int, default=0)
    args = parser.parse_args()

    for task in args.tasks:
        create_training_samples(
            task_name=task,
            num_simulations=args.num_simulations,
            training_samples_root=args.root,
            extra_theta_factor=args.extra_theta_factor,
        )
