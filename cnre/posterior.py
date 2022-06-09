from typing import Any, Dict, Optional

import torch
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.potentials.ratio_based_potential import RatioBasedPotential
from sbi.utils import mcmc_transform
from torch.distributions import Distribution


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
