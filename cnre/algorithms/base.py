from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from sbi.utils.get_nn_models import classifier_nn
from sbibm.algorithms.sbi.utils import wrap_prior_dist, wrap_simulator_fn
from sbibm.tasks.task import Task
from torch.utils.data import DataLoader

from cnre.algorithms.utils import AlgorithmOutput


class AlgBase(ABC):
    def __init__(
        self,
        task: Task,
        num_posterior_samples: int,
        max_num_epochs: int,
        learning_rate: float = 5e-4,
        neural_net: str = "resnet",
        hidden_features: int = 50,
        num_blocks: int = 2,
        use_batch_norm: bool = True,
        training_batch_size: int = 10000,
        automatic_transforms_enabled: bool = True,
        sample_with: str = "mcmc",
        mcmc_method: str = "slice_np_vectorized",
        mcmc_parameters: Dict[str, Any] = {
            "num_chains": 100,
            "thin": 10,
            "warmup_steps": 100,
            "init_strategy": "sir",
        },
        z_score_x: bool = True,
        z_score_theta: bool = True,
        state_dict_saving_rate: Optional[int] = None,
    ) -> None:
        self.task = task
        self.prior = task.get_prior_dist()
        self.simulator = task.get_simulator()

        self.transforms = task._get_transforms(automatic_transforms_enabled)[
            "parameters"
        ]
        if automatic_transforms_enabled:
            self.prior = wrap_prior_dist(self.prior, self.transforms)
            self.simulator = wrap_simulator_fn(self.simulator, self.transforms)

        self.get_classifier = classifier_nn(
            model=neural_net.lower(),
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            use_batch_norm=use_batch_norm,
            z_score_x=z_score_x,
            z_score_theta=z_score_theta,
        )

        self.get_optimizer = torch.optim.Adam
        self.num_posterior_samples = num_posterior_samples
        self.max_num_epochs = max_num_epochs
        self.learning_rate = learning_rate
        self.training_batch_size = training_batch_size
        self.state_dict_saving_rate = state_dict_saving_rate
        self.sample_with = sample_with
        self.mcmc_method = mcmc_method
        self.mcmc_parameters = mcmc_parameters

    @abstractmethod
    def get_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        raise NotImplementedError()

    @abstractmethod
    def train(
        self,
        classifier: torch.nn.Module,
        optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        extra_train_loader: DataLoader,
        extra_val_loader: DataLoader,
    ) -> Dict:
        raise NotImplementedError()

    @abstractmethod
    def run(self) -> AlgorithmOutput:
        raise NotImplementedError()
