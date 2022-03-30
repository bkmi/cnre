from pathlib import Path
from typing import Tuple, Union

import sbibm
import torch
from sbi.inference import simulate_for_sbi
from sbibm.utils.io import get_tensor_from_csv, save_tensor_to_csv


def get_training_samples_paths(
    task_name: str, num_simulations: int, training_samples_root: Union[str, Path]
) -> Tuple[Path, Path]:
    training_samples_root = Path(training_samples_root)
    theta_path = training_samples_root / f"{task_name}-{num_simulations:09d}-theta.csv"
    x_path = training_samples_root / f"{task_name}-{num_simulations:09d}-x.csv"
    return theta_path, x_path


def create_training_samples(
    task_name: str,
    num_simulations: int,
    training_samples_root: str,
    simulation_batch_size: int = 5_000,
):
    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    Path(training_samples_root).mkdir(exist_ok=True)
    theta_path, x_path = get_training_samples_paths(
        task.name, num_simulations, training_samples_root
    )

    theta, x = simulate_for_sbi(
        simulator,
        proposal=prior,
        num_simulations=num_simulations,
        simulation_batch_size=simulation_batch_size,
    )

    save_tensor_to_csv(theta_path, theta)
    save_tensor_to_csv(x_path, x)
    return None


def load_training_samples(
    task_name: str,
    num_simulations: int,
    training_samples_root: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    theta_path, x_path = get_training_samples_paths(
        task_name, num_simulations, training_samples_root
    )
    return get_tensor_from_csv(theta_path), get_tensor_from_csv(x_path)
