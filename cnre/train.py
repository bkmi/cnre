from copy import deepcopy
from typing import Callable, Optional, Tuple

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
from tqdm import trange

from cnre.loss import loss
from cnre.metrics import (
    log_normalizing_constant,
    mutual_information_0,
    mutual_information_1,
    unnormalized_kld,
)


def iterate_over_two_dataloaders(
    dl_small: DataLoader, dl_large: Optional[DataLoader]
) -> Tuple:
    if dl_large is None:
        for data_small in dl_small:
            yield data_small, [None]
    elif isinstance(dl_large.dataset, IterableDataset):
        # for the infinite large case
        for data_small, data_large in zip(dl_small, dl_large):
            yield data_small, [data_large]
    else:
        assert len(dl_small) <= len(dl_large)
        dl_iterator_small = iter(dl_small)
        for data_large in dl_large:
            try:
                data_small = next(dl_iterator_small)
            except StopIteration:
                dl_iterator_small = iter(dl_small)
                data_small = next(dl_iterator_small)
            yield data_small, data_large


def train(
    classifier: torch.nn.Module,
    optimizer,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    extra_train_loader: Optional[DataLoader] = None,
    extra_val_loader: Optional[DataLoader] = None,
    clip_max_norm: Optional[float] = 5.0,
    K: int = 1,
    gamma: float = 1.0,
    reuse: bool = False,
    max_steps_per_epoch: Optional[int] = None,
    state_dict_saving_rate: Optional[int] = None,
    loss: Callable = loss,
    val_K: Optional[int] = None,
    val_gamma: Optional[float] = None,
    num_theta_for_mutual_information: Optional[int] = None,
):
    best_network_state_dict = None
    min_loss = float("-Inf")

    val_K = K if val_K is None else val_K
    val_gamma = gamma if val_gamma is None else val_gamma

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
    train_losses = []
    valid_losses = []
    avg_log_ratios = []
    avg_log_zs = []
    mutual_information_0s = []
    mutual_information_1s = []
    unnormalized_klds = []
    for epoch in trange(epochs, leave=False):
        # Training
        classifier.train()
        optimizer.zero_grad()
        train_loss = 0
        for i, ((theta, x), (extra_theta,)) in enumerate(
            iterate_over_two_dataloaders(train_loader, extra_train_loader)
        ):
            _loss = loss(
                classifier,
                theta,
                x,
                K,
                gamma=gamma,
                reuse=reuse,
                extra_theta=extra_theta,
            )
            _loss.backward()
            if clip_max_norm is not None:
                clip_grad_norm_(
                    classifier.parameters(),
                    max_norm=clip_max_norm,
                )
            optimizer.step()
            train_loss += _loss.detach().cpu().mean().numpy()
            if i >= num_training_steps:
                break
        train_losses.append(train_loss / num_training_steps)

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
                _loss = loss(
                    classifier,
                    theta,
                    x,
                    val_K,
                    gamma=val_gamma,
                    reuse=reuse,
                    extra_theta=extra_theta,
                )
                _lnr = classifier([theta, x])
                _lnz = log_normalizing_constant(classifier, theta, x, M)
                valid_loss += _loss.detach().cpu().mean().numpy()
                avg_log_ratio += _lnr.detach().cpu().mean().numpy()
                lnz += _lnz.detach().cpu().mean().numpy()
                mi0 += mutual_information_0(_lnr, _lnz).detach().cpu().mean().numpy()
                mi1 += mutual_information_1(_lnr, _lnz).detach().cpu().mean().numpy()
                kld += unnormalized_kld(_lnr).detach().cpu().mean().numpy()
            valid_losses.append(valid_loss / len(val_loader))
            avg_log_ratios.append(avg_log_ratio / len(val_loader))
            avg_log_zs.append(lnz / len(val_loader))
            mutual_information_0s.append(mi0 / len(val_loader))
            mutual_information_1s.append(mi1 / len(val_loader))
            unnormalized_klds.append(kld / len(val_loader))
            if epoch == 0 or min_loss > valid_loss:
                min_loss = valid_loss
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
        train_losses=train_losses,
        valid_losses=valid_losses,
        avg_log_ratios=avg_log_ratios,
        avg_log_zs=avg_log_zs,
        mutual_information_0s=mutual_information_0s,
        mutual_information_1s=mutual_information_1s,
        unnormalized_klds=unnormalized_klds,
        best_network_state_dict=best_network_state_dict,
        last_state_dict=deepcopy(classifier.state_dict()),
        state_dicts=state_dicts,
    )


if __name__ == "__main__":
    pass
