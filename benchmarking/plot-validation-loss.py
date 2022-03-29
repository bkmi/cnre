from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
from sbibm.utils.io import get_tensor_from_csv


def main(args: Namespace) -> None:
    validation_losses = get_tensor_from_csv(args.path)
    plt.plot(validation_losses.squeeze().numpy())
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    main(args)
