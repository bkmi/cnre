from math import ceil
from typing import Any, Callable, Iterator

import sbi.inference as inference
import sbibm
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset


class JointSampler(IterableDataset):
    def __init__(self, simulator: Callable, proposal: Any, batch_size: int) -> None:
        super().__init__()
        self.simulator = simulator
        self.proposal = proposal
        self.batch_size = batch_size

    def sample(self):
        while True:
            theta, x = inference.simulate_for_sbi(
                self.simulator,
                self.proposal,
                num_simulations=self.batch_size,
                simulation_batch_size=self.batch_size,
            )
            yield theta, x

    def __iter__(self) -> Iterator:
        return self.sample()


def get_endless_train_loader_and_new_valid_loader(
    dataset: IterableDataset,
    num_validation_examples: int,
):
    # Create training and validation loaders using a subset sampler.
    # Intentionally use dicts to define the default dataloader args
    # Then, use dataloader_kwargs to override (or add to) any of these defaults
    # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
    train_loader_kwargs = {
        "batch_size": None,
        "batch_sampler": None,
    }

    train_loader = DataLoader(dataset, **train_loader_kwargs)
    for theta, _ in train_loader:
        training_batch_size = theta.size(0)
        break

    niters = ceil(num_validation_examples / training_batch_size)
    assert niters >= 0
    thetas, xs = [], []
    counter = 1
    for theta, x in train_loader:
        thetas.append(theta)
        xs.append(x)
        counter += 1
        if counter > niters:
            break
    theta = torch.concat(thetas, dim=0)[:num_validation_examples]
    x = torch.concat(xs, dim=0)[:num_validation_examples]
    val_dataset = TensorDataset(theta, x)

    val_loader_kwargs = {
        "batch_size": min(training_batch_size, num_validation_examples),
        "shuffle": False,
        "drop_last": True,
    }
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_loader_kwargs)

    return train_loader, val_loader


if __name__ == "__main__":
    task = sbibm.get_task("slcp")
    simulator = task.get_simulator()
    prior = task.get_prior_dist()
    batch_size = 512
    dataset = JointSampler(simulator, prior, batch_size)
    loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
    )

    for theta, x in loader:
        print(theta.shape, x.shape)
        break
