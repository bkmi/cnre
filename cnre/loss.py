from typing import Optional

import torch
import torch.nn
import torch.nn.functional
from sbi.utils import repeat_rows


def get_pmarginal_pjoint(K: int, gamma: float) -> float:
    """let the joint class to be equally likely across K options."""
    assert K >= 1
    p_joint = gamma / (1 + gamma * K)
    p_marginal = 1 / (1 + gamma * K)
    raise NotImplementedError("there is a bug in this computation. refer to https://github.com/mackelab/sbi/blob/main/sbi/inference/snre/snre_c.py")
    return p_marginal, p_joint


def classifier_logits(
    classifier: torch.nn.Module,
    theta: torch.Tensor,
    x: torch.Tensor,
    K: int,
    extra_theta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return logits obtained through classifier forward pass.

    The logits are obtained from atomic sets of (theta,x) pairs.
    """
    if extra_theta is None:
        extra_theta = torch.zeros(
            (0, theta.shape[1]), dtype=theta.dtype, device=theta.device
        )

    batch_size = theta.shape[0]
    extra_batch_size = extra_theta.shape[0]
    repeated_x = repeat_rows(x, K + 1)

    # Choose `K - 1` thetas from the rest of the batch for each x.
    probs = torch.cat(
        [(1 - torch.eye(batch_size)), torch.ones(batch_size, extra_batch_size)], dim=-1
    ) / (batch_size + extra_batch_size - 1)

    choices = torch.multinomial(probs, num_samples=K, replacement=False)

    contrasting_theta = torch.cat([theta, extra_theta], dim=0)[choices]

    atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
        batch_size * (K + 1), -1
    )

    return classifier([atomic_theta, repeated_x])


def classifier_logits_cheap_prior(
    classifier: torch.nn.Module,
    theta: torch.Tensor,
    x: torch.Tensor,
    K: int,
    extra_theta: torch.Tensor,
) -> torch.Tensor:
    """Return logits obtained through classifier forward pass.

    The logits are obtained from atomic sets of (theta,x) pairs.
    """
    batch_size = theta.shape[0]
    repeated_x = repeat_rows(x, K + 1)
    extra_theta = extra_theta[: batch_size * K]
    extra_theta = extra_theta[torch.randperm(batch_size * K), ...]
    extra_theta = extra_theta.reshape(batch_size, K, theta.shape[-1])
    atomic_theta = torch.cat((theta[:, None, :], extra_theta), dim=1).reshape(
        batch_size * (K + 1), -1
    )
    return classifier([atomic_theta, repeated_x])


def loss_bce(
    classifier: torch.nn.Module,
    theta: torch.Tensor,
    x: torch.Tensor,
    K: int = 1,
) -> torch.Tensor:
    """Returns the binary cross-entropy loss for the trained classifier.

    The classifier takes as input a $(\theta,x)$ pair. It is trained to predict 1
    if the pair was sampled from the joint $p(\theta,x)$, and to predict 0 if the
    pair was sampled from the marginals $p(\theta)p(x)$.
    """

    assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
    batch_size = theta.shape[0]

    logits = classifier_logits(classifier, theta, x, K).squeeze()
    likelihood = torch.sigmoid(logits).squeeze()

    # Alternating pairs where there is one sampled from the joint and one
    # sampled from the marginals. The first element is sampled from the
    # joint p(theta, x) and is labelled 1. The second element is sampled
    # from the marginals p(theta)p(x) and is labelled 0. And so on.
    labels = torch.ones(2 * batch_size, device=logits.device)  # two atoms
    labels[1::2] = 0.0

    # Binary cross entropy to learn the likelihood (AALR-specific)
    return torch.nn.BCELoss()(likelihood, labels)
    # return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")


def compute_loss_on_logits_marginal_and_joint(
    logits_marginal: torch.Tensor,
    logits_joint: torch.tensor,
    batch_size: int,
    K: int,
    gamma: float,
) -> torch.Tensor:
    dtype = logits_marginal.dtype
    device = logits_marginal.device

    # For 1-out-of-`K + 1` classification each datapoint consists
    # of `K + 1` points, with one of them being the correct one.
    # We have a batch of `batch_size` such datapoints.
    logits_marginal = logits_marginal.reshape(batch_size, K + 1)
    logits_joint = logits_joint.reshape(batch_size, K + 1)

    # Index 0 is the theta-x-pair sampled from the joint p(theta,x) and hence the
    # "correct" one for the 1-out-of-N classification.
    # We remove the jointly drawn sample from the logits_marginal
    logits_marginal = logits_marginal[:, 1:]
    # ... and retain it in the logits_joint.
    logits_joint = logits_joint[:, :-1]

    # To use logsumexp, we extend the denominator logits with loggamma
    loggamma = torch.tensor(gamma, dtype=dtype, device=device).log()
    logK = torch.tensor(K, dtype=dtype, device=device).log()
    denominator_marginal = torch.concat(
        [loggamma + logits_marginal, logK.expand((batch_size, 1))],
        dim=-1,
    )
    denominator_joint = torch.concat(
        [loggamma + logits_joint, logK.expand((batch_size, 1))],
        dim=-1,
    )

    # Index 0 is the theta-x-pair sampled from the joint p(theta,x) and hence the
    # "correct" one for the 1-out-of-N classification.
    log_prob_marginal = logK - torch.logsumexp(denominator_marginal, dim=-1)
    log_prob_joint = (
        loggamma + logits_joint[:, 0] - torch.logsumexp(denominator_joint, dim=-1)
    )

    # relative weights
    pm, pj = get_pmarginal_pjoint(K, gamma)
    return -torch.mean(pm * log_prob_marginal + pj * K * log_prob_joint)
    # return -(pm * log_prob_marginal + pj * K * log_prob_joint)


def loss(
    classifier: torch.nn.Module,
    theta: torch.Tensor,
    x: torch.Tensor,
    K: int,
    gamma: float,
    reuse: bool = False,
    extra_theta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """K = num_atoms + 1 because it's num_atoms joint samples and one marginal sample."""
    assert K >= 1
    assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
    batch_size = theta.shape[0]
    if reuse:
        logits_marginal = classifier_logits(classifier, theta, x, K, extra_theta)
        logits_joint = torch.clone(logits_marginal)
    else:
        logits_marginal = classifier_logits(classifier, theta, x, K, extra_theta)
        logits_joint = classifier_logits(classifier, theta, x, K, extra_theta)
    return compute_loss_on_logits_marginal_and_joint(
        logits_marginal, logits_joint, batch_size, K, gamma
    )


def loss_cheap_prior(
    classifier: torch.nn.Module,
    theta: torch.Tensor,
    x: torch.Tensor,
    K: int,
    gamma: float,
    extra_theta: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    assert K >= 1
    assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
    batch_size = theta.shape[0]
    logits_marginal = classifier_logits_cheap_prior(
        classifier, theta, x, K, extra_theta
    )
    logits_joint = classifier_logits_cheap_prior(classifier, theta, x, K, extra_theta)
    return compute_loss_on_logits_marginal_and_joint(
        logits_marginal, logits_joint, batch_size, K, gamma
    )
