from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch
from sbi import inference
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from cnre.data.benchmark import get_dataloaders
from cnre.data.joint import JointSampler, get_endless_train_loader_and_new_valid_loader
from cnre.data.prior import PriorSampler


@dataclass
class AlgorithmOutput:
    posterior_samples: Sequence[torch.Tensor]
    num_simulations: int
    validation_loss: Sequence[float] = field(default_factory=lambda: [float("nan")])
    avg_log_ratio: float = field(default=float("nan"))
    state_dicts: Dict[int, Dict] = field(default_factory=dict)
    log_prob_true_parameters: Sequence[float] = field(
        default_factory=lambda: [float("nan")] * 10
    )


def get_cheap_joint_dataloaders(
    self,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    dataset = JointSampler(self.simulator, self.prior, self.training_batch_size)
    (train_loader, valid_loader,) = get_endless_train_loader_and_new_valid_loader(
        dataset,
        self.num_validation_examples,
    )
    extra_train_loader = None
    extra_val_loader = None
    return train_loader, valid_loader, extra_train_loader, extra_val_loader


def get_cheap_prior_dataloaders(
    self,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    theta, x = inference.simulate_for_sbi(
        self.simulator,
        self.prior,
        num_simulations=self.num_simulations,
        simulation_batch_size=self.simulation_batch_size,
    )
    dataset = TensorDataset(theta, x)
    train_loader, valid_loader = get_dataloaders(
        dataset, self.training_batch_size, self.validation_fraction
    )

    try:
        prior_sampler = PriorSampler(
            self.prior, (self.num_atoms - 1) * self.training_batch_size
        )
    except AttributeError:
        prior_sampler = PriorSampler(self.prior, self.K * self.training_batch_size)
    prior_sample_loader = DataLoader(prior_sampler, batch_size=None, batch_sampler=None)
    return train_loader, valid_loader, prior_sample_loader, prior_sample_loader


def get_benchmark_dataloaders(self) -> Tuple[DataLoader, DataLoader, None, None]:
    theta, x = inference.simulate_for_sbi(
        self.simulator,
        self.prior,
        num_simulations=self.num_simulations,
        simulation_batch_size=self.simulation_batch_size,
    )
    dataset = TensorDataset(theta, x)
    train_loader, valid_loader = get_dataloaders(
        dataset, self.training_batch_size, self.validation_fraction
    )
    return train_loader, valid_loader, None, None
