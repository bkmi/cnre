import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from sbi import inference as inference
from sbi import utils as utils
from sbi.utils import clamp_and_warn, x_shape_from_simulation
from sbi.utils.get_nn_models import classifier_nn
from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task
from sbibm.utils.torch import get_default_device
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader

import cnre.data.joint
from cnre.algorithms.base import AlgBase
from cnre.algorithms.utils import AlgorithmOutput
from cnre.experiments import expected_log_ratio


class SNRE_B_INF(inference.SNRE_B):
    def train(
        self,
        max_steps_per_epoch: int,
        train_loader,
        val_loader,
        get_optimizer: Callable,
        num_atoms: int = 10,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        show_train_summary: bool = False,
        state_dict_saving_rate: int = 100,
    ) -> nn.Module:
        self._state_dicts = {}

        clipped_batch_size = min(training_batch_size, val_loader.batch_size)  # type: ignore

        num_atoms = int(
            clamp_and_warn(
                "num_atoms", num_atoms, min_val=2, max_val=clipped_batch_size
            )
        )

        for theta, x in train_loader:
            break

        if self._neural_net is None:
            self._neural_net = self._build_neural_net(theta, x)
            self._x_shape = x_shape_from_simulation(x)
        self._neural_net.to(self._device)
        self.optimizer = get_optimizer(self._neural_net.parameters(), lr=learning_rate)
        self.epoch, self._val_log_prob = 0, float("-Inf")

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):

            # Train for a single epoch.
            self._neural_net.train()
            train_log_probs_sum = 0
            counter = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )

                train_losses = self._loss(theta_batch, x_batch, num_atoms)
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(),
                        max_norm=clip_max_norm,
                    )
                self.optimizer.step()
                counter += 1
                if counter >= max_steps_per_epoch:
                    break

            self.epoch += 1

            train_log_prob_average = train_log_probs_sum / (
                max_steps_per_epoch * clipped_batch_size  # type: ignore
            )
            self._summary["train_log_probs"].append(train_log_prob_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_log_prob_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    val_losses = self._loss(theta_batch, x_batch, num_atoms)
                    val_log_prob_sum -= val_losses.sum().item()
                self._val_log_prob = val_log_prob_sum / (
                    len(val_loader) * clipped_batch_size  # type: ignore
                )
                # Log validation log prob for every epoch.
                self._summary["validation_log_probs"].append(self._val_log_prob)

                if (
                    state_dict_saving_rate is not None
                    and self.epoch % state_dict_saving_rate == 0
                ):
                    self._state_dicts[self.epoch] = deepcopy(
                        self._neural_net.state_dict()
                    )

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs"].append(self.epoch)
        self._summary["best_validation_log_probs"].append(self._best_val_log_prob)

        # # Update TensorBoard and summary dict.
        # self._summarize(
        #     round_=self._round,
        #     x_o=None,
        #     theta_bank=theta,
        #     x_bank=x,
        # )

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)


class NREBase(AlgBase, ABC):
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
        },
        z_score_x: bool = True,
        z_score_theta: bool = True,
        state_dict_saving_rate: Optional[int] = None,
        variant: str = "B",
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
            num_atoms,
            automatic_transforms_enabled,
            sample_with,
            mcmc_method,
            mcmc_parameters,
            z_score_x,
            z_score_theta,
            state_dict_saving_rate,
        )
        self.variant = variant
        if self.variant == "A":
            inference_class = inference.SNRE_A
            self.inference_method_kwargs = {}
            raise NotImplementedError()
        elif self.variant == "B":
            inference_class = SNRE_B_INF
            self.inference_method_kwargs = {"num_atoms": self.num_atoms}
        else:
            raise NotImplementedError

        device_string = (
            f"{get_default_device().type}:{get_default_device().index}"
            if get_default_device().type == "cuda"
            else f"{get_default_device().type}"
        )
        self.inference_method = inference_class(
            classifier=self.get_classifier,
            prior=self.prior,
            device=device_string,
        )

    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        extra_train_loader: DataLoader,
        extra_val_loader: DataLoader,
    ) -> nn.Module:
        raise NotImplementedError()

    def run(self) -> AlgorithmOutput:
        (
            train_loader,
            val_loader,
            extra_train_loader,
            extra_val_loader,
        ) = self.get_dataloaders()

        density_estimator = self.train(
            train_loader,
            val_loader,
            extra_train_loader,
            extra_val_loader,
        )

        density_estimator.load_state_dict(self.inference_method._best_model_state_dict)

        avg_log_ratio = expected_log_ratio(val_loader, density_estimator)

        posterior = self.inference_method.build_posterior(
            density_estimator,
            sample_with=self.sample_with,
            mcmc_method=self.mcmc_method,
            mcmc_parameters=self.mcmc_parameters,
            enable_transform=False,  # NOTE: Disable `sbi` auto-transforms, since `sbibm` does its own
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
            validation_loss=[
                -i for i in self.inference_method._summary["validation_log_probs"]
            ],
            avg_log_ratio=avg_log_ratio,
            state_dicts=self.inference_method._state_dicts,
        )


class NREInfinite(NREBase):
    def __init__(
        self,
        task: Task,
        num_posterior_samples: int,
        max_num_epochs: int,
        max_steps_per_epoch: int,
        num_validation_examples: int,
        learning_rate: float = 0.0005,
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
        },
        z_score_x: bool = True,
        z_score_theta: bool = True,
        state_dict_saving_rate: Optional[int] = None,
        variant: str = "B",
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
            num_atoms,
            automatic_transforms_enabled,
            sample_with,
            mcmc_method,
            mcmc_parameters,
            z_score_x,
            z_score_theta,
            state_dict_saving_rate,
            variant,
        )
        self.max_steps_per_epoch = max_steps_per_epoch
        self.num_validation_examples = num_validation_examples

    def get_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        dataset = cnre.data.joint.JointSampler(
            self.simulator, self.prior, self.training_batch_size
        )
        (
            train_loader,
            valid_loader,
        ) = cnre.data.joint.get_endless_train_loader_and_new_valid_loader(
            dataset,
            self.num_validation_examples,
        )
        return train_loader, valid_loader, None, None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *args,
        **kwargs,
    ) -> Dict:
        return self.inference_method.train(
            self.max_steps_per_epoch,
            train_loader,
            val_loader=val_loader,
            get_optimizer=self.get_optimizer,
            training_batch_size=self.training_batch_size,
            learning_rate=self.learning_rate,
            max_num_epochs=self.max_num_epochs,
            stop_after_epochs=2**30 - 1,
            state_dict_saving_rate=self.state_dict_saving_rate,
            **self.inference_method_kwargs,
        )