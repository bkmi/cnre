from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn
from sbi.inference.posteriors import MCMCPosterior, RejectionPosterior
from sbi.inference.potentials import ratio_estimator_based_potential
from sbi.utils import repeat_rows
from torch.distributions import Distribution
from tqdm import trange


def classifier_logits(
    classifier: torch.nn.Module,
    theta: torch.Tensor,
    x: torch.Tensor,
    num_atoms: int,
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
    repeated_x = repeat_rows(x, num_atoms)

    # Choose `1` or `num_atoms - 1` thetas from the rest of the batch for each x.
    probs = torch.cat(
        [(1 - torch.eye(batch_size)), torch.ones(batch_size, extra_batch_size)], dim=-1
    ) / (batch_size + extra_batch_size - 1)

    choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)

    contrasting_theta = torch.cat([theta, extra_theta], dim=0)[choices]

    atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
        batch_size * num_atoms, -1
    )

    return classifier([atomic_theta, repeated_x])


def loss_bce(
    classifier: torch.nn.Module,
    theta: torch.Tensor,
    x: torch.Tensor,
    num_atoms: int = 2,
) -> torch.Tensor:
    """Returns the binary cross-entropy loss for the trained classifier.

    The classifier takes as input a $(\theta,x)$ pair. It is trained to predict 1
    if the pair was sampled from the joint $p(\theta,x)$, and to predict 0 if the
    pair was sampled from the marginals $p(\theta)p(x)$.
    """

    assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
    batch_size = theta.shape[0]

    logits = classifier_logits(classifier, theta, x, num_atoms).squeeze()
    likelihood = torch.sigmoid(logits).squeeze()

    # Alternating pairs where there is one sampled from the joint and one
    # sampled from the marginals. The first element is sampled from the
    # joint p(theta, x) and is labelled 1. The second element is sampled
    # from the marginals p(theta)p(x) and is labelled 0. And so on.
    labels = torch.ones(2 * batch_size, device=logits.device)  # two atoms
    labels[1::2] = 0.0

    # Binary cross entropy to learn the likelihood (AALR-specific)
    return torch.nn.BCELoss()(likelihood, labels)


def loss(
    classifier: torch.nn.Module,
    theta: torch.Tensor,
    x: torch.Tensor,
    num_atoms: int,
    alpha: float = 1.0,
    reuse: bool = True,
) -> torch.Tensor:
    """num_atoms should include the always marginal draw, i.e. num_atoms = K + 1 with K possible jointly drawn samples and 1 marginal sample"""
    assert num_atoms >= 2
    assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
    batch_size = theta.shape[0]
    if reuse:
        logits_marginal = classifier_logits(classifier, theta, x, num_atoms)
        logits_joint = torch.clone(logits_marginal)
    else:
        logits_marginal = classifier_logits(classifier, theta, x, num_atoms)
        logits_joint = classifier_logits(classifier, theta, x, num_atoms)

    dtype = logits_marginal.dtype
    device = logits_marginal.device

    # For 1-out-of-`num_atoms` classification each datapoint consists
    # of `num_atoms` points, with one of them being the correct one.
    # We have a batch of `batch_size` such datapoints.
    logits_marginal = logits_marginal.reshape(batch_size, num_atoms)
    logits_joint = logits_joint.reshape(batch_size, num_atoms)

    # Index 0 is the theta-x-pair sampled from the joint p(theta,x) and hence the
    # "correct" one for the 1-out-of-N classification.
    # We remove the jointly drawn sample from the logits_marginal
    logits_marginal = logits_marginal[:, 1:]
    # ... and retain it in the logits_joint.
    logits_joint = logits_joint[:, :-1]

    # To use logsumexp, we extend the denominator logits with logalpha
    logalpha = (
        torch.tensor(alpha, dtype=dtype, device=device).log().expand(batch_size, 1)
    )
    denominator_marginal = torch.concat(
        [logits_marginal, logalpha],
        dim=-1,
    )
    denominator_joint = torch.concat(
        [logits_joint, logalpha],
        dim=-1,
    )

    # Index 0 is the theta-x-pair sampled from the joint p(theta,x) and hence the
    # "correct" one for the 1-out-of-N classification.
    log_prob_marginal = -torch.logsumexp(denominator_marginal, dim=-1)
    log_prob_joint = logits_joint[:, 0] - torch.logsumexp(denominator_joint, dim=-1)
    return -torch.mean(log_prob_marginal + (num_atoms - 1) * log_prob_joint)


class Parabola(object):
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def g(self, theta: torch.Tensor) -> torch.Tensor:
        return theta**2 - 1.5

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Normal(x, self.scale).log_prob(theta)

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Normal(self.g(theta), self.scale).sample()


class Gaussian(object):
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def g(self, theta: torch.Tensor) -> torch.Tensor:
        return theta

    def log_likelihood(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Normal(x, self.scale).log_prob(theta)

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Normal(self.g(theta), self.scale).sample()


def get_prior_marginal_to_joint_ratio(num_atoms: int) -> float:
    """let the marginal class be 50% likely and the joint class to be equally likely across num_atoms."""
    assert num_atoms >= 1.0
    return 0.5 + (0.5 / (num_atoms - 1))


def train(
    classifier: torch.nn.Module,
    optimizer,
    epochs: int,
    train_loader,
    val_loader,
    num_atoms: int = 2,
    reuse: bool = False,
):
    best_network_state_dict = None
    min_loss = float("-Inf")

    train_losses = []
    valid_losses = []
    for epoch in trange(epochs, leave=False):
        # Training
        classifier.train()
        train_loss = 0
        for x, theta in train_loader:
            _loss = loss(classifier, theta, x, num_atoms, reuse=reuse)
            _loss.backward()
            optimizer.step()
            train_loss += _loss.detach().cpu().mean().numpy()
        train_losses.append(train_loss / len(train_loader))

        # Evaluation
        classifier.eval()
        with torch.no_grad():
            valid_loss = 0
            for x, theta in val_loader:
                _loss = loss(classifier, theta, x, num_atoms, reuse=reuse)
                valid_loss += _loss.detach().cpu().mean().numpy()
            valid_losses.append(valid_loss / len(val_loader))
            if epoch == 0 or min_loss > valid_loss:
                min_loss = valid_loss
                best_network_state_dict = deepcopy(classifier.state_dict())
    return dict(
        train_losses=train_losses,
        valid_losses=valid_losses,
        best_network_state_dict=best_network_state_dict,
        last_state_dict=deepcopy(classifier.state_dict()),
    )


def get_sbi_posterior(
    ratio_estimator: torch.nn.Module,
    x_shape: Tuple[int, ...],
    prior: Optional[Distribution] = None,
    sample_with: str = "rejection",
    mcmc_method: str = "slice_np",
    mcmc_parameters: Dict[str, Any] = {},
    rejection_sampling_parameters: Dict[str, Any] = {},
):
    """Try it.

    Args:
        density_estimator: The density estimator that the posterior is based on.
            If `None`, use the latest neural density estimator that was trained.
        prior: Prior distribution.
        sample_with: Method to use for sampling from the posterior. Must be one of
            [`mcmc` | `rejection` | `vi`].
        mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
            `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
            implementation of slice sampling; select `hmc`, `nuts` or `slice` for
            Pyro-based sampling.
        vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`]. Note
            that some of the methods admit a `mode seeking` property (e.g. rKL)
            whereas some admit a `mass covering` one (e.g fKL).
        mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
        vi_parameters: Additional kwargs passed to `VIPosterior`.
        rejection_sampling_parameters: Additional kwargs passed to
            `RejectionPosterior`.
    """
    device = next(ratio_estimator.parameters()).device.type
    potential_fn, theta_transform = ratio_estimator_based_potential(
        ratio_estimator=ratio_estimator, prior=prior, x_o=None
    )

    if sample_with == "mcmc":
        posterior = MCMCPosterior(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            proposal=prior,
            method=mcmc_method,
            device=device,
            x_shape=x_shape,
            **mcmc_parameters,
        )
    elif sample_with == "rejection":
        posterior = RejectionPosterior(
            potential_fn=potential_fn,
            proposal=prior,
            device=device,
            x_shape=x_shape,
            **rejection_sampling_parameters,
        )
    else:
        raise NotImplementedError

    return posterior


if __name__ == "__main__":
    bs = 100
    dt = 5
    dx = 10

    c = torch.nn.Linear(dt + dx, 1)
    cc = lambda x: c(torch.cat(x, dim=-1))
    t = torch.rand(bs, dt)
    x = torch.rand(bs, dx)
    et = torch.rand(bs, dt)

    cl = classifier_logits(cc, t, x, 2, None)
