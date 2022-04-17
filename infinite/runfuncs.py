import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import sbi
import sbibm
import scipy
import scipy.integrate
import scipy.stats
import torch
import torch.distributions
from sbi import inference as inference
from sbi.utils.get_nn_models import classifier_nn
from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
from sbibm.tasks.task import Task
from tqdm import tqdm

import cnre
import cnre.data
import cnre.joint


def run_cnre(
    task: Task,
    num_samples: int,
    max_steps_per_epoch: int,
    num_validation_examples: int,
    neural_net: str = "resnet",
    hidden_features: int = 50,
    num_blocks: int = 2,
    use_batch_norm: bool = True,
    simulation_batch_size: int = 1000,
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
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs CNRE

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

    learning_rate: float = 5e-4

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

    dataset = cnre.joint.JointSampler(simulator, prior, simulation_batch_size)
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


def run_nre():
    raise NotImplementedError()


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
