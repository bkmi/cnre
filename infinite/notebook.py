from typing import Sequence, Tuple

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

nre_gamma_default = float("Inf")


def prepare_df_for_plot(df: pd.DataFrame, expr: str = None) -> pd.DataFrame:
    df = df.reset_index()
    if expr is not None:
        df = df[df["expr"] == expr]
    df["gamma"] = df["gamma"].replace([nre_gamma_default], 1.0)

    df_copy = df.reset_index().copy()
    df_copy["task"] = "average"
    df = pd.concat([df, df_copy], ignore_index=True)
    return df


def get_metrics(files: Sequence[str], expr: str = None) -> pd.DataFrame:
    df = pd.concat(
        [pd.read_csv(file, index_col=0) for file in files], ignore_index=True
    )
    df["expr"] = expr
    return df


def wide_to_long(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataFrameGroupBy]:
    # groupby drops NaNs fix this here.
    df["gamma"] = df["gamma"].fillna(nre_gamma_default)

    df_wide = pd.wide_to_long(
        df,
        ["C2ST_Z"],
        i=[
            "task",
            "algorithm",
            "num_contrastive_parameters",
            "gamma",
            "num_blocks",
            "hidden_features",
            "seed",
        ],
        j="num_observation",
        sep="-",
    )

    # groupby drops NaNs
    grp = df_wide.groupby(
        [
            "task",
            "algorithm",
            "num_contrastive_parameters",
            "gamma",
            "num_blocks",
            "hidden_features",
        ]
    )
    grp.count()
    return df_wide, grp
