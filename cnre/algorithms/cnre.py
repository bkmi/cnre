from abc import ABC
from typing import Any, Dict, Optional, Tuple

import torch
from joblib import Parallel, delayed, parallel_backend
from sbibm.algorithms.sbi.utils import wrap_posterior
from sbibm.tasks.task import Task
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnre.algorithms.base import AlgBase
from cnre.algorithms.utils import (
    AlgorithmOutput,
    get_benchmark_dataloaders,
    get_cheap_joint_dataloaders,
    get_cheap_prior_dataloaders,
)
from cnre.experiments import (
    expected_log_ratio,
    get_sbi_posterior,
    loss_cheap_prior,
    train,
)


class CNREBase(AlgBase, ABC):
    def __init__(
        self,
        task: Task,
        num_posterior_samples: int,
        max_num_epochs: int,
        learning_rate: float = 0.0005,
        neural_net: str = "resnet",
        hidden_features: int = 50,
        num_blocks: int = 2,
        use_batch_norm: bool = True,
        training_batch_size: int = 10000,
        K: int = 9,
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
        gamma: float = 1.0,
        reuse: bool = False,
        val_K: Optional[int] = None,
        val_gamma: Optional[int] = None,
    ) -> None:
        super().__init__(
            task,
            num_posterior_samples,
            max_num_epochs,
            learning_rate,
            neural_net,
            hidden_features,
            num_blocks,
            use_batch_norm,
            training_batch_size,
            automatic_transforms_enabled,
            sample_with,
            mcmc_method,
            mcmc_parameters,
            z_score_x,
            z_score_theta,
            state_dict_saving_rate,
        )
        self.K = K
        self.gamma = gamma
        self.val_K = val_K
        self.val_gamma = val_gamma
        self.reuse = reuse

    def run(self) -> AlgorithmOutput:
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

        # if n_jobs is None:
        if self.sample_with == "rejection":
            samples = [
                posterior.sample((self.num_posterior_samples,), x=observation).detach()
                for observation in observations
            ]
        elif (
            self.sample_with == "mcmc"
        ):  # this is a hack, it would be better as a parameter of the function TODO
            samples = [
                posterior.sample((self.num_posterior_samples,), x=observation).detach()
                for observation in observations
            ]
            # with parallel_backend("loky"):
            #     samples = Parallel(n_jobs=torch.get_num_threads())(
            #         delayed(posterior.sample)(
            #             (self.num_posterior_samples,),
            #             observation
            #         )
            #         for observation in tqdm(
            #             observations, leave=False, desc="sampling"
            #         )
            #     )
        else:
            raise NotImplementedError()

        return AlgorithmOutput(
            posterior_samples=samples,
            num_simulations=self.simulator.num_simulations,
            validation_loss=results["valid_losses"],
            avg_log_ratio=avg_log_ratio,
            state_dicts=results["state_dicts"],
            avg_log_ratios=results["avg_log_ratios"],
            unnormalized_klds=results["unnormalized_klds"],
        )


class CNRECheapJoint(CNREBase):
    def __init__(
        self, max_steps_per_epoch: int, num_validation_examples: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_steps_per_epoch = max_steps_per_epoch
        self.num_validation_examples = num_validation_examples

    def get_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        return get_cheap_joint_dataloaders(self)

    def train(
        self,
        classifier,
        optimizer,
        train_loader,
        val_loader,
        extra_train_loader,
        extra_val_loader,
    ) -> Dict:
        return train(
            classifier,
            optimizer,
            self.max_num_epochs,
            train_loader,
            val_loader,
            extra_train_loader,
            extra_val_loader,
            K=self.K,
            gamma=self.gamma,
            reuse=self.reuse,
            max_steps_per_epoch=self.max_steps_per_epoch,
            state_dict_saving_rate=self.state_dict_saving_rate,
            val_K=self.val_K,
            val_gamma=self.val_gamma,
        )


class CNRECheapPrior(CNREBase):
    def __init__(
        self,
        num_simulations: int,
        simulation_batch_size: int,
        validation_fraction: float,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_simulations = num_simulations
        self.simulation_batch_size = simulation_batch_size
        self.validation_fraction = validation_fraction

    def get_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        return get_cheap_prior_dataloaders(self)

    def train(
        self,
        classifier,
        optimizer,
        train_loader,
        val_loader,
        extra_train_loader,
        extra_val_loader,
    ) -> Dict:
        return train(
            classifier,
            optimizer,
            self.max_num_epochs,
            train_loader,
            val_loader,
            extra_train_loader,
            extra_val_loader,
            K=self.K,
            gamma=self.gamma,
            reuse=self.reuse,
            max_steps_per_epoch=None,
            state_dict_saving_rate=self.state_dict_saving_rate,
            loss=loss_cheap_prior,
            val_K=self.val_K,
            val_gamma=self.val_gamma,
        )


class CNREBenchmark(CNREBase):
    def __init__(
        self,
        num_simulations: int,
        simulation_batch_size: int,
        validation_fraction: float,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_simulations = num_simulations
        self.simulation_batch_size = simulation_batch_size
        self.validation_fraction = validation_fraction

    def get_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        return get_benchmark_dataloaders(self)

    def train(
        self,
        classifier,
        optimizer,
        train_loader,
        val_loader,
        extra_train_loader,
        extra_val_loader,
    ) -> Dict:
        return train(
            classifier,
            optimizer,
            self.max_num_epochs,
            train_loader,
            val_loader,
            extra_train_loader,
            extra_val_loader,
            K=self.K,
            gamma=self.gamma,
            reuse=self.reuse,
            max_steps_per_epoch=None,
            state_dict_saving_rate=self.state_dict_saving_rate,
            val_K=self.val_K,
            val_gamma=self.val_gamma,
        )
