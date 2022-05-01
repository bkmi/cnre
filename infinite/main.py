import importlib
import logging
import os
import random
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import hydra
import numpy as np
import sbi
import sbibm
import torch
import torch.distributions
import yaml
from matplotlib.pyplot import isinteractive
from metrics import compute_metrics_df
from omegaconf import DictConfig, OmegaConf
from sbi import inference as inference
from sbibm.utils.debug import pdb_hook
from sbibm.utils.io import save_float_to_csv, save_tensor_to_csv

import cnre
from cnre.algorithms.utils import AlgorithmOutput


@dataclass
class Output:
    posterior_samples: List[torch.Tensor]
    num_simulations: int
    validation_loss: List[float] = field(default_factory=lambda: [float("nan")])
    avg_log_ratio: float = field(default=float("nan"))
    state_dicts: dict[int, dict] = field(default_factory=dict)
    log_prob_true_parameters: List[float] = field(
        default_factory=lambda: [float("nan")] * 10
    )


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
        log.info(f"number of available gpus: {torch.cuda.device_count()}")
        # log.info(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        try:
            device_id = int(cfg.device[-1])
        except:
            device_id = 0
        torch.cuda.set_device(device_id)
        torch.set_default_tensor_type(
            "torch.cuda.FloatTensor" if gpu else "torch.FloatTensor"
        )
    log.info(
        f"using device {torch.device(device_id)} named {torch.cuda.get_device_name()}"
    )

    # Run
    task = sbibm.get_task(cfg.task.name)
    t0 = time.time()
    parts = cfg.algorithm.run.split(".")
    module_name = ".".join(parts[:-1])
    run_fn = getattr(importlib.import_module(module_name), parts[-1])
    algorithm_params = cfg.algorithm.params if "params" in cfg.algorithm else {}
    log.info("Start run")
    obj = run_fn(
        task=task,
        num_posterior_samples=task.num_posterior_samples,
        max_num_epochs=cfg.max_num_epochs,
        max_steps_per_epoch=cfg.max_steps_per_epoch,
        **algorithm_params,
    )
    # TODO, make this consistent
    if isinstance(obj, AlgorithmOutput):
        output = obj
    else:
        output = obj.run()
    runtime = time.time() - t0
    log.info("Finished run")

    # Paths
    path_runtime = "runtime.csv"
    path_num_simulations_simulator = "num_simulations_simulator.csv"
    path_validation_loss = "validation_loss.csv.bz2"
    path_avg_log_ratio = "avg_log_ratio.csv"
    path_posterior_samples_root = Path("posterior_samples")
    path_log_prob_true_parameters_root = Path("log_prob_true_parameters")
    path_predictive_samples_root = Path("predictive_samples")
    path_state_dicts_root = Path("state_dicts")

    path_posterior_samples_root.mkdir()
    path_log_prob_true_parameters_root.mkdir()
    path_predictive_samples_root.mkdir()
    path_state_dicts_root.mkdir()

    save_tensor_to_csv(
        path_validation_loss,
        torch.tensor(output.validation_loss, dtype=torch.float32),
        columns=["validation_loss"],
    )
    save_float_to_csv(path_avg_log_ratio, output.avg_log_ratio)
    save_float_to_csv(path_runtime, runtime)
    save_float_to_csv(path_num_simulations_simulator, output.num_simulations)
    for i, samples in enumerate(output.posterior_samples, start=1):
        save_tensor_to_csv(
            path_posterior_samples_root / f"{i:02d}.csv.bz2",
            samples,
            columns=task.get_labels_parameters(),
        )
    for i, log_prob in enumerate(output.log_prob_true_parameters, start=1):
        save_float_to_csv(path_log_prob_true_parameters_root / f"{i:02d}.csv", log_prob)
    for epoch, state_dict in output.state_dicts.items():
        torch.save(state_dict, path_state_dicts_root / f"{epoch:04d}.pt")

    # Predictive samples
    log.info("Draw posterior predictive samples")
    simulator = task.get_simulator()
    for i, samples in enumerate(output.posterior_samples, start=1):
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
            path_predictive_samples_root / f"{i:02d}.csv.bz2",
            predictive_samples,
            task.get_labels_data(),
        )

    # Compute metrics
    torch.set_default_tensor_type("torch.FloatTensor")

    if cfg.compute_metrics:
        df_metrics = compute_metrics_df(
            task_name=cfg.task.name,
            path_posterior_samples_root=path_posterior_samples_root,
            path_runtime=path_runtime,
            path_predictive_samples_root=path_predictive_samples_root,
            path_log_prob_true_parameters_root=path_log_prob_true_parameters_root,
            path_avg_log_ratio=path_avg_log_ratio,
            n_jobs=cfg.num_cores,
            log=log,
        )
        df_metrics.to_csv("metrics.csv", index=False)
        log.info(f"Metrics:\n{df_metrics.transpose().to_string(header=False)}")

    log.info(f"entire time {time.time() - t0}")


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
