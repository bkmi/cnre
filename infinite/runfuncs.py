import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import hydra
import numpy as np
import sbi
import sbibm
import scipy
import scipy.integrate
import scipy.stats
import torch
import torch.distributions
import torch.nn as nn
from sbi import inference as inference
from sbi import utils as utils
from sbi.utils import clamp_and_warn, x_shape_from_simulation
from sbi.utils.get_nn_models import classifier_nn
from sbi.utils.sbiutils import clamp_and_warn
from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils import data
from tqdm import tqdm

import cnre
import cnre.data
import cnre.joint


def run_cnre(
    task: Task,
    num_samples: int,
    max_steps_per_epoch: int,
    num_validation_examples: int,
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
    },
    z_score_x: bool = True,
    z_score_theta: bool = True,
    max_num_epochs: Optional[int] = None,
    state_dict_saving_rate: Optional[int] = None,
    gamma: float = 1.0,
    reuse: bool = True,
) -> Dict:
    """Runs infinite CNRE

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_simulations: Simulation budget
        num_rounds: Number of rounds
        neural_net: Neural network to use, one of linear / mlp / resnet
        hidden_features: Number of hidden features in network
        simulation_batch_size: Batch size for simulator
        training_batch_size: Batch size for training network
        num_atoms: Number of atoms, -1 means same as `training_batch_size`
        automatic_transforms_enabled: Whether to enable automatic transforms
        mcmc_method: MCMC method
        mcmc_parameters: MCMC parameters
        z_score_x: Whether to z-score x
        z_score_theta: Whether to z-score theta
        max_num_epochs: Maximum number of epochs

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    log = logging.getLogger(__name__)

    if max_num_epochs is None:
        raise ValueError()

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    if automatic_transforms_enabled:
        prior = wrap_prior_dist(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)

    get_classifier = classifier_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        use_batch_norm=use_batch_norm,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )

    mcmc_parameters["warmup_steps"] = 25

    dataset = cnre.joint.JointSampler(simulator, prior, training_batch_size)
    (
        train_loader,
        valid_loader,
    ) = cnre.joint.get_endless_train_loader_and_new_valid_loader(
        dataset,
        num_validation_examples,
    )
    for theta, x in train_loader:
        classifier = get_classifier(theta, x)
        break
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    results = cnre.train(
        classifier,
        optimizer,
        max_num_epochs,
        train_loader,
        valid_loader,
        extra_train_loader=None,
        extra_val_loader=None,
        num_atoms=num_atoms,
        gamma=gamma,
        reuse=reuse,
        max_steps_per_epoch=max_steps_per_epoch,
        state_dict_saving_rate=state_dict_saving_rate,
    )

    classifier.load_state_dict(results["best_network_state_dict"])

    avg_log_ratio = cnre.expected_log_ratio(valid_loader, classifier)

    posterior = cnre.get_sbi_posterior(
        ratio_estimator=classifier,
        prior=prior,
        sample_with=sample_with,
        mcmc_method=mcmc_method,
        mcmc_parameters=mcmc_parameters,
        rejection_sampling_parameters={},
        enable_transform=False,
    )

    posterior = wrap_posterior(posterior, transforms)
    observations = [
        task.get_observation(num_observation) for num_observation in range(1, 11)
    ]
    samples = [
        posterior.sample((num_samples,), x=observation).detach()
        for observation in observations
    ]

    return {
        "posterior_samples": samples,
        "num_simulations": simulator.num_simulations,
        "validation_loss": results["valid_losses"],
        "avg_log_ratio": avg_log_ratio,
        "state_dicts": results["state_dicts"],
    }


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


def run_nre(
    task: Task,
    num_samples: int,
    max_steps_per_epoch: int,
    num_validation_examples: int,
    learning_rate: float = 5e-4,
    num_rounds: int = 1,
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
    variant: str = "B",
    max_num_epochs: Optional[int] = None,
    state_dict_saving_rate: Optional[int] = None,
) -> Dict:
    """Runs infinite (S)NRE

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`
        num_simulations: Simulation budget
        num_rounds: Number of rounds
        neural_net: Neural network to use, one of linear / mlp / resnet
        hidden_features: Number of hidden features in network
        simulation_batch_size: Batch size for simulator
        training_batch_size: Batch size for training network
        num_atoms: Number of atoms, -1 means same as `training_batch_size`
        automatic_transforms_enabled: Whether to enable automatic transforms
        mcmc_method: MCMC method
        mcmc_parameters: MCMC parameters
        z_score_x: Whether to z-score x
        z_score_theta: Whether to z-score theta
        variant: Can be used to switch between SNRE-A (AALR) and -B (SRE)
        max_num_epochs: Maximum number of epochs

    Returns:
        Samples from posterior, number of simulator calls, log probability of true params if computable
    """
    log = logging.getLogger(__name__)

    if num_rounds == 1:
        log.info(f"Running NRE")
        # num_simulations_per_round = num_simulations
    else:
        raise NotImplementedError()
        log.info(f"Running SNRE")
        num_simulations_per_round = math.floor(num_simulations / num_rounds)

    # if simulation_batch_size > num_simulations_per_round:
    #     simulation_batch_size = num_simulations_per_round
    #     log.warn("Reduced simulation_batch_size to num_simulation_per_round")

    # if training_batch_size > num_simulations_per_round:
    #     training_batch_size = num_simulations_per_round
    #     log.warn("Reduced training_batch_size to num_simulation_per_round")

    prior = task.get_prior_dist()
    # if observation is None:
    #     observation = task.get_observation(num_observation)

    simulator = task.get_simulator()

    transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]
    if automatic_transforms_enabled:
        prior = wrap_prior_dist(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)

    classifier = classifier_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        use_batch_norm=use_batch_norm,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )
    if variant == "A":
        inference_class = inference.SNRE_A
        inference_method_kwargs = {}
        raise NotImplementedError()
    elif variant == "B":
        inference_class = SNRE_B_INF
        inference_method_kwargs = {"num_atoms": num_atoms}
    else:
        raise NotImplementedError

    inference_method = inference_class(classifier=classifier, prior=prior)

    posteriors = []
    proposal = prior
    mcmc_parameters["warmup_steps"] = 25
    # mcmc_parameters["enable_transform"] = False  # NOTE: Disable `sbi` auto-transforms, since `sbibm` does its own

    dataset = cnre.joint.JointSampler(simulator, proposal, training_batch_size)
    (
        train_loader,
        valid_loader,
    ) = cnre.joint.get_endless_train_loader_and_new_valid_loader(
        dataset,
        num_validation_examples,
    )

    for r in range(num_rounds):
        density_estimator = inference_method.train(
            max_steps_per_epoch,
            train_loader,
            val_loader=valid_loader,
            get_optimizer=torch.optim.Adam,
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            max_num_epochs=max_num_epochs,
            stop_after_epochs=2**30 - 1,
            state_dict_saving_rate=state_dict_saving_rate,
            **inference_method_kwargs,
        )
        # if r > 1:
        #     mcmc_parameters["init_strategy"] = "latest_sample"

    density_estimator.load_state_dict(inference_method._best_model_state_dict)

    avg_log_ratio = cnre.expected_log_ratio(valid_loader, density_estimator)

    posterior = inference_method.build_posterior(
        density_estimator,
        sample_with=sample_with,
        mcmc_method=mcmc_method,
        mcmc_parameters=mcmc_parameters,
        enable_transform=False,  # NOTE: Disable `sbi` auto-transforms, since `sbibm` does its own
    )
    # Copy hyperparameters, e.g., mcmc_init_samples for "latest_sample" strategy.
    # if r > 0:
    #     posterior.copy_hyperparameters_from(posteriors[-1])
    # proposal = posterior.set_default_x(observation)
    # posteriors.append(posterior)

    posterior = wrap_posterior(posterior, transforms)
    observations = [
        task.get_observation(num_observation) for num_observation in range(1, 11)
    ]
    samples = [
        posterior.sample((num_samples,), x=observation).detach()
        for observation in observations
    ]

    return {
        "posterior_samples": samples,
        "num_simulations": simulator.num_simulations,
        "validation_loss": [
            -i for i in inference_method._summary["validation_log_probs"]
        ],
        "avg_log_ratio": avg_log_ratio,
        "state_dicts": inference_method._state_dicts,
    }


if __name__ == "__main__":
    task = sbibm.get_task("two_moons")
    num_samples = 1_000
    max_steps_per_epoch = 100
    training_batch_size = 512
    num_validation_examples = training_batch_size * 8
    max_num_epochs = 10
    run_cnre(
        task=task,
        num_samples=num_samples,
        max_steps_per_epoch=max_steps_per_epoch,
        num_validation_examples=num_validation_examples,
        training_batch_size=training_batch_size,
        max_num_epochs=max_num_epochs,
        sample_with="rejection",
    )
