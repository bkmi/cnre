import importlib
import logging
import random
import socket
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import sbi
import sbi.analysis as analysis
import sbibm
import scipy
import scipy.integrate
import scipy.stats
import torch
import torch.distributions
import yaml
from omegaconf import DictConfig, OmegaConf
from sbi.inference import prepare_for_sbi
from sbi.utils.get_nn_models import classifier_nn
from sbibm.utils.debug import pdb_hook
from sbibm.utils.io import (
    get_float_from_csv,
    get_tensor_from_csv,
    save_float_to_csv,
    save_tensor_to_csv,
)

import cnre


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"sbi version: {sbi.__version__}")
    log.info(f"sbibm version: {sbibm.__version__}")
    log.info(f"cnre version: {cnre.__version__}")
    log.info(f"Hostname: {socket.gethostname()}")
    if cfg.seed is None:
        log.info("Seed not specified, generating random seed for replicability")
        cfg.seed = int(torch.randint(low=1, high=2**32 - 1, size=(1,))[0])
        log.info(f"Random seed: {cfg.seed}")
    save_config(cfg)

    # Seeding
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Devices
    gpu = True if cfg.device != "cpu" else False
    if gpu:
        torch.cuda.set_device(0)
        torch.set_default_tensor_type(
            "torch.cuda.FloatTensor" if gpu else "torch.FloatTensor"
        )

    # Paths
    path_samples = "posterior_samples.csv.bz2"
    path_runtime = "runtime.csv"
    path_log_prob_true_parameters = "log_prob_true_parameters.csv"
    path_num_simulations_simulator = "num_simulations_simulator.csv"
    path_predictive_samples = "predictive_samples.csv.bz2"
    path_validation_losses = "validation_losses.csv.bz2"

    # Run
    task = sbibm.get_task(cfg.task.name)
    t0 = time.time()
    parts = cfg.algorithm.run.split(".")
    module_name = ".".join(["sbibm", "algorithms"] + parts[:-1])
    run_fn = getattr(importlib.import_module(module_name), parts[-1])
    algorithm_params = cfg.algorithm.params if "params" in cfg.algorithm else {}
    log.info("Start run")
    outputs = run_fn(
        task,
        num_observation=cfg.task.num_observation,
        num_samples=task.num_posterior_samples,
        num_simulations=cfg.task.num_simulations,
        **algorithm_params,
    )
    runtime = time.time() - t0
    log.info("Finished run")

    # Store outputs
    if type(outputs) == torch.Tensor:
        samples = outputs
        num_simulations_simulator = float("nan")
        log_prob_true_parameters = float("nan")
        validation_losses = [float("nan")]
    elif type(outputs) == tuple and len(outputs) == 3:
        samples = outputs[0]
        num_simulations_simulator = float(outputs[1])
        log_prob_true_parameters = (
            float(outputs[2]) if outputs[2] is not None else float("nan")
        )
        validation_losses = [float("nan")]
    elif type(outputs) == tuple and len(outputs) == 4:
        samples = outputs[0]
        num_simulations_simulator = float(outputs[1])
        log_prob_true_parameters = (
            float(outputs[2]) if outputs[2] is not None else float("nan")
        )
        validation_losses = outputs[3]
    else:
        raise NotImplementedError
    validation_losses = torch.tensor(outputs[3], dtype=torch.float32)

    save_tensor_to_csv(
        path_validation_losses, validation_losses, columns=["validation_loss"]
    )
    save_tensor_to_csv(path_samples, samples, columns=task.get_labels_parameters())
    save_float_to_csv(path_runtime, runtime)
    save_float_to_csv(path_num_simulations_simulator, num_simulations_simulator)
    save_float_to_csv(path_log_prob_true_parameters, log_prob_true_parameters)

    # Predictive samples
    log.info("Draw posterior predictive samples")
    simulator = task.get_simulator()
    predictive_samples = []
    batch_size = 1_000
    for idx in range(int(samples.shape[0] / batch_size)):
        try:
            predictive_samples.append(
                simulator(samples[(idx * batch_size) : ((idx + 1) * batch_size), :])
            )
        except:
            predictive_samples.append(
                float("nan") * torch.ones((batch_size, task.dim_data))
            )
    predictive_samples = torch.cat(predictive_samples, dim=0)
    save_tensor_to_csv(
        path_predictive_samples, predictive_samples, task.get_labels_data()
    )

    # Compute metrics
    if cfg.compute_metrics:
        df_metrics = compute_metrics_df(
            task_name=cfg.task.name,
            num_observation=cfg.task.num_observation,
            path_samples=path_samples,
            path_runtime=path_runtime,
            path_predictive_samples=path_predictive_samples,
            path_log_prob_true_parameters=path_log_prob_true_parameters,
            log=log,
        )
        df_metrics.to_csv("metrics.csv", index=False)
        log.info(f"Metrics:\n{df_metrics.transpose().to_string(header=False)}")


def save_config(cfg: DictConfig, filename: str = "run.yaml") -> None:
    """Saves config as yaml

    Args:
        cfg: Config to store
        filename: Filename
    """
    with open(filename, "w") as fh:
        yaml.dump(
            OmegaConf.to_container(cfg, resolve=True), fh, default_flow_style=False
        )


def compute_metrics_df(
    task_name: str,
    num_observation: int,
    path_samples: str,
    path_runtime: str,
    path_predictive_samples: str,
    path_log_prob_true_parameters: str,
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


def cli():
    if "--debug" in sys.argv:
        sys.excepthook = pdb_hook
    main()


if __name__ == "__main__":
    cli()
