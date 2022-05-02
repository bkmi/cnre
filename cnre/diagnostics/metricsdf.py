import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import sbibm
import sbibm.metrics as metrics  # noqa # save this
import torch
import torch.distributions
from joblib import Parallel, delayed, parallel_backend
from omegaconf import OmegaConf
from sbibm.tasks.task import Task
from sbibm.utils.io import get_float_from_csv, get_tensor_from_csv
from toolz import merge, valmap
from tqdm import tqdm, trange


def get_ref_and_alg_samples(
    task: Task,
    num_observation: int,
    path_posterior_samples_root: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reference_posterior_samples = task.get_reference_posterior_samples(num_observation)[
        : task.num_posterior_samples, :
    ]
    path_posterior_samples_root = Path(path_posterior_samples_root)
    algorithm_posterior_samples = get_tensor_from_csv(
        path_posterior_samples_root / f"{num_observation:02d}.csv.bz2"
    )[: task.num_posterior_samples, :]
    assert reference_posterior_samples.shape[0] == task.num_posterior_samples
    assert algorithm_posterior_samples.shape[0] == task.num_posterior_samples
    return reference_posterior_samples, algorithm_posterior_samples


def get_pred_samples(
    task: Task,
    num_observation: int,
    path_predictive_samples_root: str,
) -> torch.Tensor:
    path_predictive_samples_root = Path(path_predictive_samples_root)
    predictive_samples = get_tensor_from_csv(
        path_predictive_samples_root / f"{num_observation:02d}.csv.bz2"
    )[: task.num_posterior_samples, :]
    assert predictive_samples.shape[0] == task.num_posterior_samples
    return predictive_samples


def get_log_prob_true_parameters(
    path_log_prob_true_parameters_root: str,
    num_observation: int,
) -> torch.Tensor:
    return torch.tensor(
        get_float_from_csv(
            path_log_prob_true_parameters_root / f"{num_observation:02d}.csv"
        )
    )  # noqa


def get_metrics(
    task: Task,
    num_observation: int,
    path_posterior_samples_root: str,
    path_predictive_samples_root: str,
    path_log_prob_true_parameters_root: str,
    log: logging.Logger = logging.getLogger(__name__),
):
    # Load samples and things which depend on num_observation
    reference_posterior_samples, algorithm_posterior_samples = get_ref_and_alg_samples(
        task, num_observation, path_posterior_samples_root
    )
    predictive_samples = get_pred_samples(
        task, num_observation, path_predictive_samples_root
    )
    observation = task.get_observation(num_observation)  # noqa
    log_prob_true_parameters = get_log_prob_true_parameters(
        path_log_prob_true_parameters_root, num_observation
    )  # noqa

    # compute a metrics dict

    # Names of all metrics as keys, values are calls that are passed to eval
    # NOTE: Originally, we computed a large number of metrics, as reflected in the
    # dictionary below. Ultimately, we used 10k samples and z-scoring for C2ST but
    # not for MMD. If you were to adapt this code for your own pipeline of experiments,
    # the entries for C2ST_Z, MMD and RT would probably suffice (and save compute).
    _METRICS_ = {
        #
        # 10k samples
        #
        # "C2ST": "metrics.c2st(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=False)",
        f"C2ST_Z-{num_observation:02d}": "metrics.c2st(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=True)",
        f"MMD-{num_observation:02d}": "metrics.mmd(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=False)",
        # "MMD_Z": "metrics.mmd(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=True)",
        f"MEDDIST-{num_observation:02d}": "metrics.median_distance(predictive_samples, observation)",
        #
        # Not sample based
        #
        f"NLTP-{num_observation:02d}": "-1. * log_prob_true_parameters",
    }

    metrics_dict = {}

    for metric, eval_cmd in _METRICS_.items():
        log.info(f"Computing {metric}")
        try:
            metrics_dict[metric] = eval(eval_cmd).cpu().numpy().astype(np.float32)
            log.info(f"{metric}: {metrics_dict[metric]}")
        except:
            metrics_dict[metric] = float("nan")

    return metrics_dict


def compute_metrics_df(
    task_name: Task,
    path_posterior_samples_root: str,
    path_runtime: str,
    path_predictive_samples_root: str,
    path_log_prob_true_parameters_root: str,
    path_avg_log_ratio: str,
    n_jobs: int,
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

    # Load things which are independent of num_observation
    avg_log_ratio = torch.tensor(get_float_from_csv(path_avg_log_ratio))
    runtime_sec = torch.tensor(get_float_from_csv(path_runtime))  # noqa

    # Add them to the metrics dict
    metrics_dict = {
        "AVG_LOG_RATIO": avg_log_ratio,
        "RT": runtime_sec,
    }
    metrics_dict = valmap(lambda x: x.cpu().numpy().astype(np.float32), metrics_dict)

    if n_jobs == 0:
        metrics_dict_by_num_obs = []
        for num_observation in trange(
            1, 11, leave=False, desc="sequentially in num_observation"
        ):
            metrics_dict_by_num_obs.append(
                get_metrics(
                    task,
                    num_observation,
                    path_posterior_samples_root,
                    path_predictive_samples_root,
                    path_log_prob_true_parameters_root,
                    log,
                )
            )
    else:
        with parallel_backend("loky"):
            metrics_dict_by_num_obs = Parallel(n_jobs=n_jobs)(
                delayed(get_metrics)(
                    task,
                    num_observation,
                    path_posterior_samples_root,
                    path_predictive_samples_root,
                    path_log_prob_true_parameters_root,
                    log,
                )
                for num_observation in trange(
                    1, 11, leave=False, desc="sequentially in num_observation"
                )
            )

    metrics_dict = merge(metrics_dict, *metrics_dict_by_num_obs)

    # # Load samples and things which depend on num_observation
    # reference_posterior_samples, algorithm_posterior_samples = get_ref_and_alg_samples(task, num_observation, path_posterior_samples_root, log)
    # predictive_samples = get_pred_samples(task, num_observation, path_predictive_samples_root)
    # observation = task.get_observation(num_observation)  # noqa
    # log_prob_true_parameters = get_log_prob_true_parameters(path_log_prob_true_parameters_root, num_observation)  # noqa

    # # Add them to the metrics dict

    # # Names of all metrics as keys, values are calls that are passed to eval
    # # NOTE: Originally, we computed a large number of metrics, as reflected in the
    # # dictionary below. Ultimately, we used 10k samples and z-scoring for C2ST but
    # # not for MMD. If you were to adapt this code for your own pipeline of experiments,
    # # the entries for C2ST_Z, MMD and RT would probably suffice (and save compute).
    # _METRICS_ = {
    #     #
    #     # 10k samples
    #     #
    #     # "C2ST": "metrics.c2st(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=False)",
    #     f"C2ST_Z-{num_observation:02d}": "metrics.c2st(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=True)",
    #     f"MMD-{num_observation:02d}": "metrics.mmd(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=False)",
    #     # "MMD_Z": "metrics.mmd(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=True)",
    #     f"MEDDIST-{num_observation:02d}": "metrics.median_distance(predictive_samples, observation)",
    #     #
    #     # Not sample based
    #     #
    #     f"NLTP-{num_observation:02d}": "-1. * log_prob_true_parameters",
    # }

    # for metric, eval_cmd in _METRICS_.items():
    #     log.info(f"Computing {metric}")
    #     try:
    #         metrics_dict[metric] = eval(eval_cmd).cpu().numpy().astype(np.float32)
    #         log.info(f"{metric}: {metrics_dict[metric]}")
    #     except:
    #         metrics_dict[metric] = float("nan")

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
