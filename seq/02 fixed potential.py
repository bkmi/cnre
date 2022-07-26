from copy import deepcopy
from functools import cache, partial
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyro
import sbi
import sbi.analysis as analysis
import sbibm
import scipy
import scipy.integrate
import scipy.stats
import torch
import torch.distributions
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from pyro import distributions as pdist
from sbi.inference import SNRE_A, SNRE_B, infer, prepare_for_sbi, simulate_for_sbi
from sbi.inference.posteriors import MCMCPosterior
from sbi.inference.potentials.likelihood_based_potential import LikelihoodBasedPotential
from sbi.inference.potentials.ratio_based_potential import (
    ratio_estimator_based_potential,
)
from sbi.utils.get_nn_models import classifier_nn
from sbibm.algorithms.sbi.utils import wrap_posterior
from tqdm.notebook import tqdm, trange

import cnre

# num_observation = 1

# task = sbibm.get_task("gaussian_linear")
# prior = task.get_prior_dist()
# # likelihood = task._get_log_prob_fn(num_observation)
# ref_posterior = task._get_reference_posterior(num_observation)
# theta_o = task.get_true_parameters(num_observation)
# x_o = task.get_observation(num_observation)
# print(type(ref_posterior))


class Ratio(torch.nn.Module):
    def __init__(self, posterior, d, device):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(10, 10))
        self.map = torch.randn(d, d)
        self.net.to(device)
        self.posterior = posterior

    def forward(self, w: list):
        """Inputs to (s)nre classifier must be a list containing raw theta and x."""
        # return -torch.sum(w[0] ** 2, dim=-1)
        theta, x = w
        assert isinstance(theta, torch.Tensor)
        assert len(theta.shape) == 2

        return self.posterior.log_prob(theta)
        return prior.log_prob(theta).unsqueeze(-1)
        # return (theta).pow(2).neg().sum(
        #     dim=-1,
        #     keepdims=True
        # )


def create_chain(
    starting_theta: torch.Tensor,
    ratio_estimator: torch.nn.Module,
    prior,
    # potential_fn: Callable,
    transform: Optional[dict] = None,
    thin: int = 1,
    num_workers: int = 1,
    device: Optional[str] = None,
    x_shape: Optional[tuple] = None,
):
    """starting_theta [num_starting_pts, theta_dim] - we try to make this an array of samples drawn from the prior which then start each chain."""
    if transform is not None:
        starting_theta = transform["parameters"](starting_theta)

    potential_fn, theta_transform = ratio_estimator_based_potential(
        ratio_estimator,
        prior,
        x_o=None,
        enable_transform=False,  # this is false with sbibm
    )
    posterior = MCMCPosterior(
        potential_fn=potential_fn,
        proposal=starting_theta,
        theta_transform=theta_transform,
        # method="slice_np_vectorized",
        method="hmc",
        thin=thin,
        warmup_steps=0,
        num_chains=starting_theta.shape[0],
        init_strategy="delta",
        num_workers=num_workers,
        device=device,
        x_shape=x_shape,
    )
    if transform is not None:
        return wrap_posterior(posterior, transform["parameters"])
    else:
        return posterior


def main():
    n = 2
    d = 2

    dim = 2
    prior_loc = 2 * torch.ones(dim)
    prior_cov = 0.1 * torch.eye(dim)
    prior = pdist.MultivariateNormal(loc=prior_loc, covariance_matrix=prior_cov)
    theta_o = torch.zeros((1, dim))
    x_o = torch.zeros((1, dim))

    posti_loc = torch.zeros((1, dim)).squeeze()
    posti_cov = 0.1 * torch.eye(dim)
    ref_posterior = pdist.MultivariateNormal(loc=posti_loc, covariance_matrix=posti_cov)

    starting_theta = prior.sample((n,)) + torch.tensor([[10, 10], [-10, -10]])

    ratio_fixed = Ratio(ref_posterior, d, "cpu")

    p = create_chain(starting_theta, ratio_fixed, prior)

    # print(starting_theta)

    samples = p.sample((40,), x=x_o)

    print(samples)

    print(torch.isclose(samples, starting_theta[:1, ...].expand(*samples.shape)))


## Now I'm having issues with pickling... and in the end I will get a sampler which doesn't do random walk. I should probably just rewrite one.

if __name__ == "__main__":
    main()
