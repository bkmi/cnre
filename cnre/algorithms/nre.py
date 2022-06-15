from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from sbi import inference as inference
from sbi import utils as utils
from sbi.utils import clamp_and_warn, x_shape_from_simulation
from sbibm.algorithms.sbi.utils import wrap_posterior
from sbibm.tasks.task import Task
from sbibm.utils.torch import get_default_device
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from cnre import classifier_logits_cheap_prior, expected_log_ratio
from cnre.algorithms.base import AlgBase
from cnre.algorithms.utils import (
    AlgorithmOutput,
    get_benchmark_dataloaders,
    get_cheap_joint_dataloaders,
    get_cheap_prior_dataloaders,
)
from cnre.metrics import (
    log_normalizing_constant,
    mutual_information_0,
    mutual_information_1,
    unnormalized_kld,
)


class OurSNRE_B(inference.SNRE_B):
    def _train_loop(
        self,
        max_steps_per_epoch: Optional[int],
        train_loader: DataLoader,
        val_loader: DataLoader,
        extra_train_loader: Optional[DataLoader],
        extra_val_loader: Optional[DataLoader],
        get_optimizer: Callable,
        num_atoms: int = 10,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        show_train_summary: bool = False,
        state_dict_saving_rate: int = 100,
        val_num_atoms: Optional[int] = None,
        num_theta_for_mutual_information: Optional[int] = None,
    ):
        self._state_dicts = {}
        self._avg_log_ratios = []
        self._unnormalized_klds = []
        self._avg_log_zs = []
        self._mutual_information_0s = []
        self._mutual_information_1s = []

        clipped_batch_size = min(training_batch_size, val_loader.batch_size)  # type: ignore

        num_atoms = int(
            clamp_and_warn(
                "num_atoms", num_atoms, min_val=2, max_val=clipped_batch_size
            )
        )

        val_num_atoms = num_atoms if val_num_atoms is None else val_num_atoms

        for theta, x in train_loader:
            break

        if self._neural_net is None:
            self._neural_net = self._build_neural_net(theta, x)
            self._x_shape = x_shape_from_simulation(x)
        self._neural_net.to(self._device)
        self.optimizer = get_optimizer(self._neural_net.parameters(), lr=learning_rate)
        self.epoch, self._val_log_prob = 0, float("-Inf")

        max_steps_per_epoch = (
            max_steps_per_epoch
            if max_steps_per_epoch is not None
            else len(train_loader)
        )
        M = (
            val_loader.batch_size - 1
            if num_theta_for_mutual_information is None
            else num_theta_for_mutual_information
        )

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):

            # Train for a single epoch.
            self._neural_net.train()
            train_log_probs_sum = 0
            counter = 0
            for batch, extra_theta in zip(train_loader, extra_train_loader):
                self.optimizer.zero_grad()
                theta_batch, x_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                )
                extra_theta = (
                    extra_theta[0].to(self._device)
                    if isinstance(extra_theta, list)
                    else extra_theta
                )
                extra_theta = extra_theta.to(self._device)
                train_losses = self._loss(theta_batch, x_batch, num_atoms, extra_theta)
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
            avg_log_ratio = 0
            lnz = 0
            mi0 = 0
            mi1 = 0
            kld = 0
            with torch.no_grad():
                for batch, extra_theta in zip(val_loader, extra_val_loader):
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    extra_theta = (
                        extra_theta[0].to(self._device)
                        if isinstance(extra_theta, list)
                        else extra_theta
                    )
                    extra_theta = extra_theta.to(self._device)
                    val_losses = self._loss(
                        theta_batch, x_batch, val_num_atoms, extra_theta
                    )
                    _lnr = self._neural_net([theta_batch, x_batch]).squeeze()
                    _lnz = log_normalizing_constant(
                        self._neural_net, theta_batch, x_batch, M
                    )
                    val_log_prob_sum -= val_losses.sum().item()
                    avg_log_ratio += _lnr.detach().cpu().mean().numpy()
                    lnz += _lnz.detach().cpu().mean().numpy()
                    mi0 += (
                        mutual_information_0(_lnr, _lnz).detach().cpu().mean().numpy()
                    )
                    mi1 += (
                        mutual_information_1(_lnr, _lnz).detach().cpu().mean().numpy()
                    )
                    kld += unnormalized_kld(_lnr).detach().cpu().mean().numpy()
                self._val_log_prob = val_log_prob_sum / (
                    len(val_loader) * clipped_batch_size  # type: ignore
                )
                self._avg_log_zs.append(lnz / len(val_loader))
                self._mutual_information_0s.append(mi0 / len(val_loader))
                self._mutual_information_1s.append(mi1 / len(val_loader))
                self._unnormalized_klds.append(kld / len(val_loader))
                self._avg_log_ratios.append(avg_log_ratio / len(val_loader))
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

    def _loss(
        self, theta: torch.Tensor, x: torch.Tensor, num_atoms: int, extra_theta: None
    ) -> torch.Tensor:
        return super()._loss(theta, x, num_atoms)

    @abstractmethod
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
        val_num_atoms: Optional[int] = None,
    ) -> nn.Module:
        raise NotImplementedError()


class SNRE_B_CheapPrior(OurSNRE_B):
    def _loss(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        num_atoms: int,
        extra_theta: torch.Tensor,
    ) -> torch.Tensor:
        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]
        K = num_atoms - 1
        extra_theta = extra_theta[: batch_size * K]
        logits = classifier_logits_cheap_prior(
            self._neural_net, theta, x, K, extra_theta
        )

        # For 1-out-of-`num_atoms` classification each datapoint consists
        # of `num_atoms` points, with one of them being the correct one.
        # We have a batch of `batch_size` such datapoints.
        logits = logits.reshape(batch_size, num_atoms)

        # Index 0 is the theta-x-pair sampled from the joint p(theta,x) and hence the
        # "correct" one for the 1-out-of-N classification.
        log_prob = logits[:, 0] - torch.logsumexp(logits, dim=-1)

        return -torch.mean(log_prob)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        extra_train_loader: DataLoader,
        extra_val_loader: DataLoader,
        get_optimizer: Callable,
        num_atoms: int = 10,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        show_train_summary: bool = False,
        state_dict_saving_rate: int = 100,
        val_num_atoms: Optional[int] = None,
        num_theta_for_mutual_information: Optional[int] = None,
    ) -> nn.Module:
        return self._train_loop(
            max_steps_per_epoch=None,
            train_loader=train_loader,
            val_loader=val_loader,
            extra_train_loader=extra_train_loader,
            extra_val_loader=extra_val_loader,
            get_optimizer=get_optimizer,
            num_atoms=num_atoms,
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            stop_after_epochs=stop_after_epochs,
            max_num_epochs=max_num_epochs,
            clip_max_norm=clip_max_norm,
            show_train_summary=show_train_summary,
            state_dict_saving_rate=state_dict_saving_rate,
            val_num_atoms=val_num_atoms,
            num_theta_for_mutual_information=num_theta_for_mutual_information,
        )


class SNRE_B_CheapJoint(OurSNRE_B):
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
        val_num_atoms: Optional[int] = None,
        num_theta_for_mutual_information: Optional[int] = None,
    ) -> nn.Module:
        dataset = TensorDataset(
            torch.zeros((training_batch_size * max_steps_per_epoch, 0))
        )
        extra_loader = DataLoader(dataset, batch_size=training_batch_size)
        return self._train_loop(
            max_steps_per_epoch=max_steps_per_epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            extra_train_loader=extra_loader,
            extra_val_loader=extra_loader,
            get_optimizer=get_optimizer,
            num_atoms=num_atoms,
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            stop_after_epochs=stop_after_epochs,
            max_num_epochs=max_num_epochs,
            clip_max_norm=clip_max_norm,
            show_train_summary=show_train_summary,
            state_dict_saving_rate=state_dict_saving_rate,
            val_num_atoms=val_num_atoms,
            num_theta_for_mutual_information=num_theta_for_mutual_information,
        )


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
        },
        z_score_x: bool = True,
        z_score_theta: bool = True,
        state_dict_saving_rate: Optional[int] = None,
        val_num_atoms: Optional[int] = None,
        num_theta_for_mutual_information: Optional[int] = None,
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
        self.num_atoms = num_atoms
        self.val_num_atoms = val_num_atoms
        self.num_theta_for_mutual_information = num_theta_for_mutual_information
        self.inference_method_kwargs = {
            "num_atoms": self.num_atoms,
            "val_num_atoms": self.val_num_atoms,
            "num_theta_for_mutual_information": self.num_theta_for_mutual_information,
        }
        self.device_string = (
            f"{get_default_device().type}:{get_default_device().index}"
            if get_default_device().type == "cuda"
            else f"{get_default_device().type}"
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
            avg_log_zs=self.inference_method._avg_log_zs,
            state_dicts=self.inference_method._state_dicts,
            avg_log_ratios=self.inference_method._avg_log_ratios,
            unnormalized_klds=self.inference_method._unnormalized_klds,
            mutual_information_0s=self.inference_method._mutual_information_0s,
            mutual_information_1s=self.inference_method._mutual_information_1s,
        )


class NRECheapJoint(NREBase):
    def __init__(
        self,
        max_steps_per_epoch: int,
        num_validation_examples: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_steps_per_epoch = max_steps_per_epoch
        self.num_validation_examples = num_validation_examples
        inference_class = SNRE_B_CheapJoint
        self.inference_method = inference_class(
            classifier=self.get_classifier,
            prior=self.prior,
            device=self.device_string,
        )

    def get_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        return get_cheap_joint_dataloaders(self)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *args,
        **kwargs,
    ) -> nn.Module:
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


class NRECheapPrior(NREBase):
    def __init__(
        self,
        num_simulations: int,
        simulation_batch_size: int,
        validation_fraction: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_simulations = num_simulations
        self.simulation_batch_size = simulation_batch_size
        self.validation_fraction = validation_fraction
        inference_class = SNRE_B_CheapPrior
        self.inference_method = inference_class(
            classifier=self.get_classifier,
            prior=self.prior,
            device=self.device_string,
        )

    def get_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        return get_cheap_prior_dataloaders(self)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        extra_train_loader: DataLoader,
        extra_val_loader: DataLoader,
    ) -> nn.Module:
        return self.inference_method.train(
            train_loader,
            val_loader,
            extra_train_loader,
            extra_val_loader,
            get_optimizer=self.get_optimizer,
            training_batch_size=self.training_batch_size,
            learning_rate=self.learning_rate,
            max_num_epochs=self.max_num_epochs,
            stop_after_epochs=2**30 - 1,
            state_dict_saving_rate=self.state_dict_saving_rate,
            **self.inference_method_kwargs,
        )


class NREBenchmark(NREBase):
    def __init__(
        self,
        num_simulations: int,
        simulation_batch_size: int,
        validation_fraction: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_simulations = num_simulations
        self.simulation_batch_size = simulation_batch_size
        self.validation_fraction = validation_fraction
        inference_class = SNRE_B_CheapJoint
        self.inference_method = inference_class(
            classifier=self.get_classifier,
            prior=self.prior,
            device=self.device_string,
        )

    def get_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        return get_benchmark_dataloaders(self)

    def train(
        self,
        train_loader,
        val_loader,
        *args,
        **kwargs,
    ) -> Dict:
        return self.inference_method.train(
            len(train_loader),
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
