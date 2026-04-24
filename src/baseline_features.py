# src/baseline_features.py

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")


def add_slot_15min_feature(df: pd.DataFrame) -> pd.DataFrame:
    _validate_datetime_index(df)
    out = df.copy().sort_index()
    out["slot_15min"] = out.index.hour * 4 + out.index.minute // 15
    return out


def build_slot_baseline_table(
    df_train: pd.DataFrame,
    *,
    target_col: str,
    use_median: bool = True,
) -> pd.DataFrame:
    _validate_datetime_index(df_train)

    if target_col not in df_train.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe")

    d = df_train.copy().sort_index()
    d["dayofweek"] = d.index.dayofweek
    d["slot_15min"] = d.index.hour * 4 + d.index.minute // 15

    grouped = d.groupby(["dayofweek", "slot_15min"])[target_col]

    if use_median:
        baseline = grouped.median()
    else:
        baseline = grouped.mean()

    baseline = baseline.rename(f"{target_col}_baseline").reset_index()
    return baseline


def apply_slot_baseline(
    df: pd.DataFrame,
    *,
    baseline_table: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    _validate_datetime_index(df)

    baseline_col = f"{target_col}_baseline"

    d = df.copy().sort_index()
    d["dayofweek"] = d.index.dayofweek
    d["slot_15min"] = d.index.hour * 4 + d.index.minute // 15

    d = d.reset_index().rename(columns={d.index.name or "index": "Time"})
    d = d.merge(
        baseline_table,
        on=["dayofweek", "slot_15min"],
        how="left",
    )
    d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
    d = d.set_index("Time").sort_index()

    return d


def add_residual_target(
    df: pd.DataFrame,
    *,
    target_col: str,
) -> pd.DataFrame:
    _validate_datetime_index(df)

    baseline_col = f"{target_col}_baseline"
    residual_col = f"{target_col}_residual"

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe")
    if baseline_col not in df.columns:
        raise ValueError(f"baseline column '{baseline_col}' not found in dataframe")

    out = df.copy().sort_index()
    out[residual_col] = out[target_col] - out[baseline_col]
    return out


def build_baseline_only_prediction(
    df: pd.DataFrame,
    *,
    target_col: str,
) -> pd.DataFrame:
    _validate_datetime_index(df)

    baseline_col = f"{target_col}_baseline"

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe")
    if baseline_col not in df.columns:
        raise ValueError(f"baseline column '{baseline_col}' not found in dataframe")

    out = pd.DataFrame(index=df.index.copy())
    out["y_true"] = df[target_col]
    out["y_pred"] = df[baseline_col]
    return out