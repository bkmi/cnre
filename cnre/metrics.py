from typing import Optional

import torch
from sbi.utils import repeat_rows
from torch.utils.data.dataloader import DataLoader


def expected_log_ratio(
    loader: DataLoader,
    classifier: torch.nn.Module,
) -> torch.Tensor:
    """This expects a loader with the same batch_size for every batch."""
    avg_log_ratio = 0
    for theta, x in loader:
        log_ratio = classifier([theta, x])
        _avg_log_ratio = log_ratio.mean()
        avg_log_ratio += _avg_log_ratio.cpu().item()
    return avg_log_ratio / len(loader)


def log_normalizing_constant(
    classifier: torch.nn.Module,
    theta: torch.Tensor,
    x: torch.Tensor,
    M: int,
    extra_theta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if extra_theta is None:
        extra_theta = torch.zeros(
            (0, theta.shape[1]), dtype=theta.dtype, device=theta.device
        )

    batch_size = theta.shape[0]
    extra_batch_size = extra_theta.shape[0]
    repeated_x = repeat_rows(x, M)

    # Choose `M` thetas from the rest of the batch for each x.
    probs = torch.cat(
        [(1 - torch.eye(batch_size)), torch.ones(batch_size, extra_batch_size)], dim=-1
    ) / (batch_size + extra_batch_size - 1)

    choices = torch.multinomial(probs, num_samples=M, replacement=False)

    contrasting_theta = torch.cat([theta, extra_theta], dim=0)[choices].reshape(
        batch_size * M, -1
    )
    log_ratio = classifier([contrasting_theta, repeated_x]).reshape(batch_size, M)
    return log_ratio.exp().mean(dim=1).log()


def log_normalizing_constant_cheap_prior(
    classifier: torch.nn.Module,
    extra_theta: torch.Tensor,
    x: torch.Tensor,
    M: int,
) -> torch.Tensor:
    batch_size = x.shape[0]
    repeated_x = repeat_rows(x, M)
    extra_theta = extra_theta[: batch_size * M]
    extra_theta = extra_theta[torch.randperm(batch_size * M), ...].reshape(
        batch_size * M, -1
    )
    log_ratio = classifier([extra_theta, repeated_x]).reshape(batch_size, M)
    return log_ratio.exp().mean(dim=1).log()


def mutual_information_0(log_ratio: torch.Tensor, log_z: torch.Tensor) -> torch.Tensor:
    return torch.mean(log_ratio - log_z)


def mutual_information_1(log_ratio: torch.Tensor, log_z: torch.Tensor) -> torch.Tensor:
    return torch.mean(log_ratio - log_z.exp() - 1)


def estimate_mutual_information(
    loader: DataLoader,
    classifier: torch.nn.Module,
    M: int,
    version: int = 0,
) -> torch.Tensor:
    """This expects a loader with the same batch_size for every batch."""
    mutual_information = 0
    for theta, x in loader:
        log_ratio = classifier([theta, x])
        log_z = log_normalizing_constant(classifier, theta, x, M)
        if version == 0:
            _mutual_information = mutual_information_0(log_ratio, log_z)
        elif version == 1:
            _mutual_information = mutual_information_1(log_ratio, log_z)
        else:
            raise NotImplementedError("must choose 0 or 1.")
        mutual_information += _mutual_information.cpu().item()
    return mutual_information / len(loader)


def estimate_mutual_information_cheap_prior(
    loader: DataLoader,
    classifier: torch.nn.Module,
    M: int,
    version: int = 0,
) -> torch.Tensor:
    """This expects a loader with the same batch_size for every batch."""
    raise NotImplementedError()
