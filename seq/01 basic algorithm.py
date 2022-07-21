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

task = sbibm.get_task("slcp")
prior = task.get_prior_dist()
simulator = task.get_simulator()
x_o = task.get_observation(1)
theta_o = task.get_true_parameters(1)
transform = task._get_transforms()


def create_chain(
    starting_theta: torch.Tensor,
    ratio_estimator: torch.nn.Module,
    # potential_fn: Callable,
    transform: Optional[dict] = None,
    thin: int = 10,
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
        method="slice_np_vectorized",
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
        return wrap_posterior(posterior, transform)


n = 3
d = theta_o.shape[1]
dd = x_o.shape[1]
starting_theta = torch.rand(n, d)

inference = SNRE_A(prior)
theta, x = simulate_for_sbi(simulator, prior, num_simulations=500)
ratio_estimator = inference.append_simulations(theta, x).train()


class Ratio(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(10, 10))
        self.map = torch.randn(dd, d)
        self.net.to(device)

    def forward(self, w: list):
        """Inputs to (s)nre classifier must be a list containing raw theta and x."""
        # return -torch.sum(w[0] ** 2, dim=-1)
        theta, x = w
        assert isinstance(theta, torch.Tensor)
        assert len(theta.shape) == 2

        # print(self.map)
        # print(theta)
        # print(x)
        t = torch.einsum("ij,bj->bi", self.map, theta)
        return (t - x).pow(2).neg().sum(dim=-1, keepdims=True)


ratio_fixed = Ratio("cpu")

# p = create_chain(starting_theta, ratio_estimator, transform)
p = create_chain(starting_theta, ratio_fixed, transform)

# p_built = inference.build_posterior(
#     mcmc_method="slice_np_vectorized",
#     mcmc_parameters={"num_chains": n},
#     enable_transform=False
# )
# p_built = wrap_posterior(p_built, transform)

samples = p.sample((10,), x=x_o)

print(samples)

# This runs and I'll leave it there. this is how you can write a sampler.

exit()
