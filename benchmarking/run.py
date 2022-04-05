import importlib
import logging
import random
import socket
import sys
import time

import hydra
import numpy as np
import sbi
import sbibm
import scipy
import scipy.integrate
import scipy.stats
import torch
import torch.distributions
import yaml
from metrics import compute_metrics_df
from omegaconf import DictConfig, OmegaConf
from sbibm.utils.debug import pdb_hook
from sbibm.utils.io import save_float_to_csv, save_tensor_to_csv

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

    # Threading
    if cfg.num_cores is not None:
        log.info(f"Setting num_threads and num_interop_threads to {cfg.num_cores}")
        torch.set_num_threads(cfg.num_cores)
        torch.set_num_interop_threads(cfg.num_cores)
    log.info(f"num_threads: {torch.get_num_threads()}")
    log.info(f"num_interop_threads: {torch.get_num_interop_threads()}")

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
    path_avg_log_ratio = "avg_log_ratio.csv"

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
        training_samples_root=cfg.training_samples_root,
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
        avg_log_ratio = float("nan")
    elif type(outputs) == tuple and len(outputs) == 4:
        samples = outputs[0]
        num_simulations_simulator = float(outputs[1])
        log_prob_true_parameters = (
            float(outputs[2]) if outputs[2] is not None else float("nan")
        )
        validation_losses = [float("nan")]
        avg_log_ratio = outputs[3]
    elif type(outputs) == tuple and len(outputs) == 5:
        samples = outputs[0]
        num_simulations_simulator = float(outputs[1])
        log_prob_true_parameters = (
            float(outputs[2]) if outputs[2] is not None else float("nan")
        )
        validation_losses = outputs[3]
        avg_log_ratio = outputs[4]
    else:
        raise NotImplementedError
    validation_losses = torch.tensor(validation_losses, dtype=torch.float32)

    save_tensor_to_csv(
        path_validation_losses, validation_losses, columns=["validation_loss"]
    )
    save_tensor_to_csv(path_samples, samples, columns=task.get_labels_parameters())
    save_float_to_csv(path_avg_log_ratio, avg_log_ratio)
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
            path_avg_log_ratio=path_avg_log_ratio,
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


def cli():
    if "--debug" in sys.argv:
        sys.excepthook = pdb_hook
    main()


if __name__ == "__main__":
    cli()
