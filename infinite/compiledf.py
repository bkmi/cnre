import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from sbibm.utils.io import get_float_from_csv
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def compile_df(
    root: str,
    glob: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compile dataframe for further analyses
    `root` is the path to a folder over which to recursively loop. All information
    is compiled into a big dataframe and returned for further analyses.
    Args:
        root: Base path to use
    Returns:
        Dataframe with results
    """
    df = []

    basepaths = [p.parent for p in Path(root).expanduser().rglob(glob)]

    for i, path_base in tqdm(enumerate(basepaths)):
        path_metrics = path_base / "metrics.csv"

        row = {}

        # Read hydra config
        path_cfg = path_metrics.parent / "run.yaml"
        if path_cfg.exists():
            cfg = OmegaConf.to_container(OmegaConf.load(str(path_cfg)))
        else:
            continue

        # Config file
        try:
            row["task"] = cfg["task"]["name"]
        except:
            print(f"path {path_base} has no task name")
            continue
        # row["num_simulations"] = cfg["task"]["num_simulations"]
        # row["num_observation"] = cfg["task"]["num_observation"]
        row["seed"] = cfg["seed"]
        # row["max_num_epochs"] = cfg["algorithm"]["params"]["max_num_epochs"]

        if cfg["algorithm"]["name"].lower() in ["nre", "snre"]:
            row["algorithm"] = cfg["algorithm"]["name"] + "-B"
        else:
            row["algorithm"] = cfg["algorithm"]["name"]

        # these are paired
        try:
            row["num_atoms"] = cfg["algorithm"]["params"]["num_atoms"]
        except KeyError:
            row["num_atoms"] = cfg["algorithm"]["params"]["K"]

        try:
            row["K"] = cfg["algorithm"]["params"]["K"]
        except KeyError:
            row["K"] = cfg["algorithm"]["params"]["num_atoms"]

        try:
            row["gamma"] = cfg["algorithm"]["params"]["gamma"]
        except KeyError:
            row["gamma"] = float("nan")

        try:
            row["extra_theta_factor"] = cfg["algorithm"]["params"]["extra_theta_factor"]
        except KeyError:
            row["extra_theta_factor"] = 0

        try:
            row["reuse"] = cfg["algorithm"]["params"]["reuse"]
        except KeyError:
            row["extra_theta_factor"] = False

        row["hidden_features"] = cfg["algorithm"]["params"]["hidden_features"]
        row["num_blocks"] = cfg["algorithm"]["params"]["num_blocks"]

        # if cfg["training_samples_root"] is not None:
        #     row["training_samples_root"] = "/".join(
        #         cfg["training_samples_root"].split("/")[-3:-1]
        #     )
        # else:
        #     row["training_samples_root"] = ""

        # Metrics df
        if path_metrics.exists():
            metrics_df = pd.read_csv(path_metrics)
            for metric_name, metric_value in metrics_df.items():
                row[metric_name] = metric_value[0]
        else:
            print(f"path {path_base} has no metrics")
            continue

        # NLTP can be properly computed for NPE as part of the algorithm
        # SNPE's estimation of NLTP via rejection rates may introduce additional errors
        path_log_prob_true_parameters = (
            path_metrics.parent / "log_prob_true_parameters.csv"
        )
        row["NLTP"] = float("nan")
        if row["algorithm"][:3] == "NPE":
            if path_log_prob_true_parameters.exists():
                row["NLTP"] = -1.0 * get_float_from_csv(path_log_prob_true_parameters)

        # Runtime
        # While almost all runs were executed on AWS hardware under the same conditions,
        # this was not the case for 100% of the runs. To prevent uneven comparison,
        # the file `runtime.csv` was deleted for those runs where this was not the case.
        # If `runtime.csv` is absent from a run, RT will be set to NaN accordingly.
        path_runtime = path_metrics.parent / "runtime.csv"
        if not path_runtime.exists():
            row["RT"] = float("nan")
        else:
            row["RT"] = get_float_from_csv(path_runtime)

        # Runtime to minutes
        row["RT"] = row["RT"] / 60.0

        # Num simulations simulator
        path_num_simulations_simulator = (
            path_metrics.parent / "num_simulations_simulator.csv"
        )
        if path_num_simulations_simulator.exists():
            row["num_simulations_simulator"] = get_float_from_csv(
                path_num_simulations_simulator
            )

        # Path and folder
        row["path"] = str((path_base).absolute())
        row["folder"] = row["path"].split("/")[-1]

        # # Exclude from df if there are no posterior samples
        # if not Path(f"{row['path']}/posterior_samples.csv.bz2").exists():
        #     continue

        df.append(row)

    df = pd.DataFrame(df)
    # if len(df) > 0:
    #     df["num_observation"] = df["num_observation"].astype("category")

    print(f"{len(basepaths) - len(df)} were missed")
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--glob", type=str, default="run.yaml")
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    df = compile_df(args.root, args.glob)

    if args.save is not None:
        df.to_csv(args.save)
