from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn
import torch.nn.functional
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.potentials.ratio_based_potential import RatioBasedPotential
from sbi.utils import mcmc_transform, repeat_rows
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
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
    # return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")


def loss(
    classifier: torch.nn.Module,
    theta: torch.Tensor,
    x: torch.Tensor,
    num_atoms: int,
    gamma: float,
    reuse: bool,
    extra_theta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """num_atoms should include the always marginal draw, i.e. num_atoms = K + 1 with K possible jointly drawn samples and 1 marginal sample

    loss = 2 * loss_bce, due to the mean computation over more items in loss_bce
    """
    assert num_atoms >= 2
    assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
    batch_size = theta.shape[0]
    if reuse:
        logits_marginal = classifier_logits(
            classifier, theta, x, num_atoms, extra_theta
        )
        logits_joint = torch.clone(logits_marginal)
    else:
        logits_marginal = classifier_logits(
            classifier, theta, x, num_atoms, extra_theta
        )
        logits_joint = classifier_logits(classifier, theta, x, num_atoms, extra_theta)

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

    # To use logsumexp, we extend the denominator logits with loggamma
    loggamma = torch.tensor(gamma, dtype=dtype, device=device).log()
    logK = torch.tensor(num_atoms - 1, dtype=dtype, device=device).log()
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
    pm, pj = get_pmarginal_pjoint(num_atoms, gamma)
    return -torch.mean(pm * log_prob_marginal + pj * logK.exp() * log_prob_joint)
    # return -(pm * log_prob_marginal + pj * logK.exp() * log_prob_joint)


def expected_log_ratio(
    loader,
    classifier,
):
    avg_log_ratio = 0
    for theta, x in loader:
        log_ratio = classifier([theta, x])
        _avg_log_ratio = log_ratio.mean()
        avg_log_ratio += _avg_log_ratio.cpu().item()
    return avg_log_ratio / len(loader)


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


def get_pmarginal_pjoint(num_atoms: int, gamma: float) -> float:
    """let the joint class to be equally likely across num_atoms."""
    assert num_atoms >= 2
    K = num_atoms - 1
    p_joint = gamma / (1 + gamma * K)
    p_marginal = 1 / (1 + gamma * K)
    return p_marginal, p_joint


def iterate_over_two_dataloaders(
    dl_small: DataLoader, dl_large: Optional[DataLoader]
) -> Tuple:
    if dl_large is None:
        for data_small in dl_small:
            yield data_small, [None]
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
    num_atoms: int = 2,
    gamma: float = 1.0,
    reuse: bool = True,
    max_steps_per_epoch: Optional[int] = None,
    state_dict_saving_rate: Optional[int] = None,
):
    best_network_state_dict = None
    min_loss = float("-Inf")

    # catch infinite training loaders
    try:
        num_training_steps = min(max_steps_per_epoch, len(train_loader))
    except TypeError:
        num_training_steps = max_steps_per_epoch

    state_dicts = {}
    train_losses = []
    valid_losses = []
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
                num_atoms,
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
            if i >= max_steps_per_epoch:
                break
        train_losses.append(train_loss / num_training_steps)

        # Evaluation
        classifier.eval()
        optimizer.zero_grad()
        with torch.no_grad():
            valid_loss = 0
            for (theta, x), (extra_theta,) in iterate_over_two_dataloaders(
                val_loader, extra_val_loader
            ):
                _loss = loss(
                    classifier,
                    theta,
                    x,
                    num_atoms,
                    gamma=gamma,
                    reuse=reuse,
                    extra_theta=extra_theta,
                )
                valid_loss += _loss.detach().cpu().mean().numpy()
            valid_losses.append(valid_loss / len(val_loader))
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
        best_network_state_dict=best_network_state_dict,
        last_state_dict=deepcopy(classifier.state_dict()),
        state_dicts=state_dicts,
    )


def get_sbi_posterior(
    ratio_estimator: torch.nn.Module,
    prior: Optional[Distribution] = None,
    sample_with: str = "rejection",
    mcmc_method: str = "slice_np",
    mcmc_parameters: Dict[str, Any] = {},
    rejection_sampling_parameters: Dict[str, Any] = {},
    enable_transform: bool = True,
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
    potential_fn = RatioBasedPotential(ratio_estimator, prior, x_o=None, device=device)
    theta_transform = mcmc_transform(
        prior, device=device, enable_transform=enable_transform
    )

    if sample_with == "mcmc":
        posterior = MCMCPosterior(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            proposal=prior,
            method=mcmc_method,
            device=device,
            **mcmc_parameters,
        )
    elif sample_with == "rejection":
        posterior = RejectionPosterior(
            potential_fn=potential_fn,
            proposal=prior,
            device=device,
            **rejection_sampling_parameters,
        )
    else:
        raise NotImplementedError

    return posterior


def get_dataloaders(
    dataset: torch.utils.data.TensorDataset,
    training_batch_size: int = 50,
    validation_fraction: float = 0.1,
    dataloader_kwargs: Optional[dict] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return dataloaders for training and validation.

    Args:
        dataset: holding all theta and x, optionally masks.
        training_batch_size: training arg of inference methods.
        resume_training: Whether the current call is resuming training so that no
            new training and validation indices into the dataset have to be created.
        dataloader_kwargs: Additional or updated kwargs to be passed to the training
            and validation dataloaders (like, e.g., a collate_fn).

    Returns:
        Tuple of dataloaders for training and validation.

    """

    # Get total number of training examples.
    num_examples = len(dataset)

    # Select random train and validation splits from (theta, x) pairs.
    num_training_examples = int((1 - validation_fraction) * num_examples)
    num_validation_examples = num_examples - num_training_examples

    permuted_indices = torch.randperm(num_examples)
    train_indices, val_indices = (
        permuted_indices[:num_training_examples],
        permuted_indices[num_training_examples:],
    )

    # Create training and validation loaders using a subset sampler.
    # Intentionally use dicts to define the default dataloader args
    # Then, use dataloader_kwargs to override (or add to) any of these defaults
    # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
    train_loader_kwargs = {
        "batch_size": min(training_batch_size, num_training_examples),
        "drop_last": True,
        "sampler": torch.utils.data.SubsetRandomSampler(train_indices.tolist()),
    }
    val_loader_kwargs = {
        "batch_size": min(training_batch_size, num_validation_examples),
        "shuffle": False,
        "drop_last": True,
        "sampler": torch.utils.data.SubsetRandomSampler(val_indices.tolist()),
    }
    if dataloader_kwargs is not None:
        train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
        val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset, **train_loader_kwargs)
    val_loader = torch.utils.data.DataLoader(dataset, **val_loader_kwargs)

    return train_loader, val_loader


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
