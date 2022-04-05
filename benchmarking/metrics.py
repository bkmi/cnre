import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sbibm
import scipy
import scipy.integrate
import scipy.stats
import torch
import torch.distributions
from omegaconf import DictConfig, OmegaConf
from sbibm.utils.io import get_float_from_csv, get_tensor_from_csv
from tqdm import tqdm


def compute_metrics_df(
    task_name: str,
    num_observation: int,
    path_samples: str,
    path_runtime: str,
    path_predictive_samples: str,
    path_log_prob_true_parameters: str,
    path_avg_log_ratio: str,
    log: logging.Logger = logging.getLogger(__name__),
) -> pd.DataFrame:
    """Compute all metrics, returns dataframe

    Args:
        task_name: Task
        num_observation: Observation
        path_samples: Path to posterior samples
        path_runtime: Path to runtime file
        path_predictive_samples: Path to predictive samples
        path_log_prob_true_parameters: Path to NLTP
        path_avg_log_ratio
        log: Logger

    Returns:
        Dataframe with results
    """
    log.info(f"Compute all metrics")

    # Load task
    task = sbibm.get_task(task_name)

    # Load samples
    reference_posterior_samples = task.get_reference_posterior_samples(num_observation)[
        : task.num_posterior_samples, :
    ]
    algorithm_posterior_samples = get_tensor_from_csv(path_samples)[
        : task.num_posterior_samples, :
    ]
    assert reference_posterior_samples.shape[0] == task.num_posterior_samples
    assert algorithm_posterior_samples.shape[0] == task.num_posterior_samples
    log.info(
        f"Loaded {task.num_posterior_samples} samples from reference and algorithm"
    )

    # Load posterior predictive samples
    predictive_samples = get_tensor_from_csv(path_predictive_samples)[
        : task.num_posterior_samples, :
    ]
    assert predictive_samples.shape[0] == task.num_posterior_samples

    # Load observation
    observation = task.get_observation(num_observation=num_observation)  # noqa

    # Get runtime info
    runtime_sec = torch.tensor(get_float_from_csv(path_runtime))  # noqa

    # Get log prob true parameters
    log_prob_true_parameters = torch.tensor(
        get_float_from_csv(path_log_prob_true_parameters)
    )  # noqa

    avg_log_ratio = torch.tensor(get_float_from_csv(path_avg_log_ratio))

    # Names of all metrics as keys, values are calls that are passed to eval
    # NOTE: Originally, we computed a large number of metrics, as reflected in the
    # dictionary below. Ultimately, we used 10k samples and z-scoring for C2ST but
    # not for MMD. If you were to adapt this code for your own pipeline of experiments,
    # the entries for C2ST_Z, MMD and RT would probably suffice (and save compute).
    _METRICS_ = {
        #
        # 10k samples
        #
        "C2ST": "metrics.c2st(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=False)",
        "C2ST_Z": "metrics.c2st(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=True)",
        "MMD": "metrics.mmd(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=False)",
        "MMD_Z": "metrics.mmd(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=True)",
        # "KSD_GAUSS": "metrics.ksd(task=task, num_observation=num_observation, samples=algorithm_posterior_samples, sig2=float(torch.median(torch.pdist(reference_posterior_samples))**2), log=False)",
        "MEDDIST": "metrics.median_distance(predictive_samples, observation)",
        #
        # 1K samples
        #
        # "C2ST_1K": "metrics.c2st(X=reference_posterior_samples[:1000,:], Y=algorithm_posterior_samples[:1000,:], z_score=False)",
        # "C2ST_1K_Z": "metrics.c2st(X=reference_posterior_samples[:1000,:], Y=algorithm_posterior_samples[:1000, :], z_score=True)",
        # "MMD_1K": "metrics.mmd(X=reference_posterior_samples[:1000,:], Y=algorithm_posterior_samples[:1000, :], z_score=False)",
        # "MMD_1K_Z": "metrics.mmd(X=reference_posterior_samples[:1000,:], Y=algorithm_posterior_samples[:1000, :], z_score=True)",
        # "KSD_GAUSS_1K": "metrics.ksd(task=task, num_observation=num_observation, samples=algorithm_posterior_samples[:1000, :], sig2=float(torch.median(torch.pdist(reference_posterior_samples))**2), log=False)",
        # "MEDDIST_1K": "metrics.median_distance(predictive_samples[:1000,:], observation)",
        #
        # Not based on samples
        #
        "NLTP": "-1. * log_prob_true_parameters",
        "AVG_LOG_RATIO": "avg_log_ratio",
        "RT": "runtime_sec",
    }

    import sbibm.metrics as metrics  # noqa

    metrics_dict = {}
    for metric, eval_cmd in _METRICS_.items():
        log.info(f"Computing {metric}")
        try:
            metrics_dict[metric] = eval(eval_cmd).cpu().numpy().astype(np.float32)
            log.info(f"{metric}: {metrics_dict[metric]}")
        except:
            metrics_dict[metric] = float("nan")

    return pd.DataFrame(metrics_dict)


def root_to_metrics_df(
    root: Path, log: logging.Logger, overwrite: bool
) -> pd.DataFrame:
    metrics_path = root / "metrics.csv"
    if not metrics_path.exists() or overwrite:
        pass
    else:
        log.info(f"{metrics_path} already exists.")
        return (root, "exists")

    try:
        cfg = OmegaConf.to_container(OmegaConf.load(str(root / "run.yaml")))
        path_samples = root / "posterior_samples.csv.bz2"
        path_runtime = root / "runtime.csv"
        path_predictive_samples = root / "predictive_samples.csv.bz2"
        path_log_prob_true_parameters = root / "log_prob_true_parameters.csv"

        df_metrics = compute_metrics_df(
            cfg["task"]["name"],
            cfg["task"]["num_observation"],
            path_samples,
            path_runtime,
            path_predictive_samples,
            path_log_prob_true_parameters,
            log=log,
        )
        df_metrics.to_csv(metrics_path, index=False)
        log.info(f"Metrics:\n{df_metrics.transpose().to_string(header=False)}")
        return df_metrics
    except FileNotFoundError as error:
        log.exception(error)
        return (root, error)


def main(roots: list, overwrite: bool) -> None:
    log = logging.getLogger(__name__)
    roots = [Path(root) for root in roots]

    results = []
    for root in tqdm(roots, leave=False):
        results.append(root_to_metrics_df(root, log=log, overwrite=overwrite))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(**vars(args))
