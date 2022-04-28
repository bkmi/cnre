from pathlib import Path

import sbi.analysis as analysis
import sbibm
from sbibm.utils.io import get_tensor_from_csv


def get_training_samples_id(path: str):
    if path == "training-data/extra":
        print("warning!")
        return 10
    else:
        directory_name = path.split("/")[-1]
        training_samples_id = directory_name.split("_")[-1]
        return int(training_samples_id)


def get_relative_path(
    reported_path: str,
    prefix: str = "multirun/",
    prefix_to_remove: str = "/home/ben/sci/cnre/benchmarking/multirun/",
):
    return prefix + reported_path[len(prefix_to_remove) :]


def get_validation_losses(root):
    root = Path(root)
    return get_tensor_from_csv(root / "validation_losses.csv.bz2")


def get_posterior_samples(root: str):
    root = Path(root)
    return get_tensor_from_csv(root / "posterior_samples.csv.bz2")


def plot_posterior(samples, true_theta=None, name=None, limits=None):
    points = None if true_theta is None else true_theta.cpu().numpy()
    fig, _ = analysis.pairplot(
        samples.cpu().numpy(),
        figsize=(6, 6),
        points=points,
        title=name,
        limits=limits,
    )


def get_reference_samples_from_row(row):
    task_name = row["task"]
    num_observation = row["num_observation"]
    task = sbibm.get_task(task_name)
    return task.get_reference_posterior_samples(num_observation)


def get_true_theta_from_row(row):
    task_name = row["task"]
    num_observation = row["num_observation"]
    task = sbibm.get_task(task_name)
    return task.get_true_parameters(num_observation)
