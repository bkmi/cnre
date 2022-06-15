from abc import ABC
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from joblib import Parallel, delayed, parallel_backend
from sbibm.algorithms.sbi.utils import wrap_posterior
from sbibm.tasks.task import Task
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm, trange

from cnre import expected_log_ratio, get_sbi_posterior, loss_cheap_prior
from cnre.algorithms.base import AlgBase
from cnre.algorithms.cnre import CNREBase
from cnre.algorithms.utils import (
    AlgorithmOutput,
    get_benchmark_dataloaders,
    get_cheap_joint_dataloaders,
    get_cheap_prior_dataloaders,
    iterate_over_two_dataloaders,
)
from cnre.loss import loss as cnre_loss
from cnre.metrics import (
    log_normalizing_constant,
    mutual_information_0,
    mutual_information_1,
    unnormalized_kld,
)


def train(
    classifier: torch.nn.Module,
    optimizer,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    extra_train_loader: Optional[DataLoader] = None,
    extra_val_loader: Optional[DataLoader] = None,
    clip_max_norm: Optional[float] = 5.0,
    max_steps_per_epoch: Optional[int] = None,
    state_dict_saving_rate: Optional[int] = None,
    log_z: Callable = log_normalizing_constant,
    val_K: int = 1,
    val_gamma: float = 1,
    num_theta_for_mutual_information: Optional[int] = None,
):
    best_network_state_dict = None
    min_mi0 = float("-Inf")

    # catch infinite training loaders
    try:
        len_train_loader = len(train_loader)
    except TypeError:
        len_train_loader = float("Inf")

    if max_steps_per_epoch is None:
        max_steps_per_epoch = float("Inf")

    num_training_steps = min(max_steps_per_epoch, len_train_loader)
    M = (
        val_loader.batch_size - 1
        if num_theta_for_mutual_information is None
        else num_theta_for_mutual_information
    )

    state_dicts = {}
    train_mi1_losses = []
    valid_classifier_losses = []
    avg_log_ratios = []
    avg_log_zs = []
    mutual_information_0s = []
    mutual_information_1s = []
    unnormalized_klds = []
    for epoch in trange(epochs, leave=False):
        # Training
        classifier.train()
        optimizer.zero_grad()
        _train_mi1_loss = 0
        for i, ((theta, x), (extra_theta,)) in enumerate(
            iterate_over_two_dataloaders(train_loader, extra_train_loader)
        ):
            _lnr = classifier([theta, x]).squeeze()
            _lnz = log_z(classifier, theta, x, M)
            _mi1_loss = mutual_information_1(_lnr, _lnz).neg()
            _mi1_loss.backward()
            if clip_max_norm is not None:
                clip_grad_norm_(
                    classifier.parameters(),
                    max_norm=clip_max_norm,
                )
            optimizer.step()
            _train_mi1_loss += _mi1_loss.detach().cpu().mean().numpy()
            if i >= num_training_steps:
                break
        train_mi1_losses.append(_train_mi1_loss / num_training_steps)

        # Evaluation
        classifier.eval()
        optimizer.zero_grad()
        with torch.no_grad():
            valid_loss = 0
            avg_log_ratio = 0
            lnz = 0
            mi0 = 0
            mi1 = 0
            kld = 0
            for (theta, x), (extra_theta,) in iterate_over_two_dataloaders(
                val_loader, extra_val_loader
            ):
                _loss = cnre_loss(
                    classifier,
                    theta,
                    x,
                    K=val_K,
                    gamma=val_gamma,
                    extra_theta=extra_theta,
                )
                _lnr = classifier([theta, x]).squeeze()
                _lnz = log_normalizing_constant(classifier, theta, x, M)
                valid_loss += _loss.detach().cpu().mean().numpy()
                avg_log_ratio += _lnr.detach().cpu().mean().numpy()
                lnz += _lnz.detach().cpu().mean().numpy()
                mi0 += mutual_information_0(_lnr, _lnz).detach().cpu().mean().numpy()
                mi1 += mutual_information_1(_lnr, _lnz).detach().cpu().mean().numpy()
                kld += unnormalized_kld(_lnr).detach().cpu().mean().numpy()
            valid_classifier_losses.append(valid_loss / len(val_loader))
            avg_log_ratios.append(avg_log_ratio / len(val_loader))
            avg_log_zs.append(lnz / len(val_loader))
            mutual_information_0s.append(mi0 / len(val_loader))
            mutual_information_1s.append(mi1 / len(val_loader))
            unnormalized_klds.append(kld / len(val_loader))
            if epoch == 0 or min_mi0 > -mi0:
                min_mi0 = mi0
                best_network_state_dict = deepcopy(classifier.state_dict())
            if (
                state_dict_saving_rate is not None
                and epoch % state_dict_saving_rate == 0
            ):
                state_dicts[epoch] = deepcopy(classifier.state_dict())

    # Avoid keeping the gradients in the resulting network, which can
    # cause memory leakage when benchmarking.
    classifier.zero_grad(set_to_none=True)

    return dict(
        train_losses=train_mi1_losses,
        valid_losses=valid_classifier_losses,
        avg_log_ratios=avg_log_ratios,
        avg_log_zs=avg_log_zs,
        mutual_information_0s=mutual_information_0s,
        mutual_information_1s=mutual_information_1s,
        unnormalized_klds=unnormalized_klds,
        best_network_state_dict=best_network_state_dict,
        last_state_dict=deepcopy(classifier.state_dict()),
        state_dicts=state_dicts,
    )


class InfoRatioBase(AlgBase, ABC):
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
        val_K: int = 1,
        val_gamma: float = 1.0,
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
        self.val_K = val_K
        self.val_gamma = val_gamma
        self.num_theta_for_mutual_information = num_theta_for_mutual_information

    def run(self) -> AlgorithmOutput:
        return CNREBase.run(self)


class InfoRatioBenchmark(InfoRatioBase):
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
            classifier=classifier,
            optimizer=optimizer,
            epochs=self.max_num_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            extra_train_loader=extra_train_loader,
            extra_val_loader=extra_val_loader,
            max_steps_per_epoch=None,
            state_dict_saving_rate=self.state_dict_saving_rate,
            log_z=log_normalizing_constant,
            val_K=self.val_K,
            val_gamma=self.val_gamma,
            num_theta_for_mutual_information=self.num_theta_for_mutual_information,
        )
