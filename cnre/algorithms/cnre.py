from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from sbi import inference
from sbi.utils.get_nn_models import classifier_nn
from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task
from torch.utils.data import DataLoader

from cnre.algorithms.utils import AlgorithmOutput
from cnre.data.joint import JointSampler, get_endless_train_loader_and_new_valid_loader
from cnre.data.prior import PriorSampler
from cnre.experiments import expected_log_ratio, get_sbi_posterior, train


class CNREBase(ABC):
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
        num_atoms: int = 10,
        automatic_transforms_enabled: bool = True,
        sample_with: str = "mcmc",
        mcmc_method: str = "slice_np_vectorized",
        mcmc_parameters: Dict[str, Any] = {
            "num_chains": 100,
            "thin": 10,
            "warmup_steps": 100,
            "init_strategy": "sir",
            "sir_batch_size": 1000,
            "sir_num_batches": 100,
            "warmup_steps": 25,
        },
        z_score_x: bool = True,
        z_score_theta: bool = True,
        state_dict_saving_rate: Optional[int] = None,
        gamma: float = 1.0,
        reuse: bool = False,
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
        self.num_atoms = num_atoms
        self.gamma = gamma
        self.reuse = reuse
        self.state_dict_saving_rate = state_dict_saving_rate
        self.sample_with = sample_with
        self.mcmc_method = mcmc_method
        self.mcmc_parameters = mcmc_parameters

        self.max_steps_per_epoch = None

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

    def run(self):
        (
            train_loader,
            val_loader,
            extra_train_loader,
            extra_val_loader,
        ) = self.get_dataloaders()

        for theta, x in train_loader:
            classifier = self.get_classifier(theta, x)
            break
        optimizer = self.get_optimizer(classifier.parameters(), lr=self.learning_rate)

        results = self.train(
            classifier,
            optimizer,
            train_loader,
            val_loader,
            extra_train_loader,
            extra_val_loader,
        )

        classifier.load_state_dict(results["best_network_state_dict"])

        avg_log_ratio = expected_log_ratio(val_loader, classifier)

        posterior = get_sbi_posterior(
            ratio_estimator=classifier,
            prior=self.prior,
            sample_with=self.sample_with,
            mcmc_method=self.mcmc_method,
            mcmc_parameters=self.mcmc_parameters,
            rejection_sampling_parameters={},
            enable_transform=False,
        )

        posterior = wrap_posterior(posterior, self.transforms)
        observations = [
            self.task.get_observation(num_observation)
            for num_observation in range(1, 11)
        ]
        samples = [
            posterior.sample((self.num_posterior_samples,), x=observation).detach()
            for observation in observations
        ]

        return AlgorithmOutput(
            posterior_samples=samples,
            num_simulations=self.simulator.num_simulations,
            validation_loss=results["valid_losses"],
            avg_log_ratio=avg_log_ratio,
            state_dicts=results["state_dicts"],
        )


class CNREInfinite(CNREBase):
    def __init__(
        self, max_steps_per_epoch: int, num_validation_examples: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_steps_per_epoch = max_steps_per_epoch
        self.num_validation_examples = num_validation_examples

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, None, None]:
        dataset = JointSampler(self.simulator, self.prior, self.training_batch_size)
        (train_loader, valid_loader,) = get_endless_train_loader_and_new_valid_loader(
            dataset,
            self.num_validation_examples,
        )
        extra_train_loader = None
        extra_val_loader = None
        return train_loader, valid_loader, extra_train_loader, extra_val_loader

    def train(
        self,
        classifier,
        optimizer,
        train_loader,
        val_loader,
        extra_train_loader,
        extra_val_loader,
    ):
        return train(
            classifier,
            optimizer,
            self.max_num_epochs,
            train_loader,
            val_loader,
            extra_train_loader,
            extra_val_loader,
            num_atoms=self.num_atoms,
            gamma=self.gamma,
            reuse=self.reuse,
            max_steps_per_epoch=self.max_steps_per_epoch,
            state_dict_saving_rate=self.state_dict_saving_rate,
        )


class CNRECheapPrior(CNREBase):
    def __init__(
        self, num_simulations: int, simulation_batch_size: int, *args, kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_simulations = num_simulations
        self.simulation_batch_size = simulation_batch_size
        raise NotImplementedError()  # TODO

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        theta, x = inference.simulate_for_sbi(
            self.simulator,
            self.prior,
            num_simulations=self.num_simulations_per_round,
            simulation_batch_size=self.simulation_batch_size,
        )
        # train_loader, valid_loader = get_dataloaders(
        #     dataset, training_batch_size, validation_fraction
        # )

        prior_sampler = PriorSampler(self.prior, self.training_batch_size)
        prior_sample_loader = DataLoader(
            prior_sampler, batch_size=None, batch_sampler=None
        )
        return train_loader, valid_loader, prior_sample_loader
