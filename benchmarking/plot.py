from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import sbi.analysis as analysis
from sbibm.utils.io import get_tensor_from_csv


def main(args: Namespace) -> None:
    files = list(args.root.glob("*"))
    path_validation_loss = filter(
        lambda p: "validation_losses" in p.name, files
    ).__next__()
    path_posterior_samples = filter(
        lambda p: "posterior_samples" in p.name, files
    ).__next__()

    if args.loss:
        validation_losses = get_tensor_from_csv(path_validation_loss)
        fig, ax = plt.subplots()
        ax.plot(validation_losses.squeeze().numpy())
        plt.show()

    if args.posterior:
        posterior_samples = get_tensor_from_csv(path_posterior_samples)
        name = f"ref"
        fig, _ = analysis.pairplot(
            posterior_samples,
            # figsize=(6,6),
            # points=true_theta.cpu().numpy(),
            title=name,
            # limits=limits,
        )
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--loss", action="store_true")
    parser.add_argument("--posterior", action="store_true")
    args = parser.parse_args()
    main(args)
