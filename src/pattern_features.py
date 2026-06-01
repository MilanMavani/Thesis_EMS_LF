from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")


def add_ramp_pattern_features(
    df: pd.DataFrame,
    target_cols: list[str],
    *,
    diff_lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    active_threshold: float = 0.5,
    ramp_threshold: float = 1.0,
    since_ramp_cap: int | None = None,
) -> pd.DataFrame:

    out = df.copy().sort_index()

    if diff_lags is None:
        diff_lags = [1, 2, 4]
    if rolling_windows is None:
        rolling_windows = [4, 12, 24, 96]

    target_cols = [c for c in target_cols if c in out.columns]

    for target_col in target_cols:
        p = out[target_col]

        # Change / ramp magnitude
        for lag in diff_lags:
            out[f"{target_col}_diff_{lag}"] = p.diff(lag)
            out[f"{target_col}_abs_diff_{lag}"] = p.diff(lag).abs()

        # Current operating state at the forecast issue timestamp
        out[f"{target_col}_active"] = (p.abs() > active_threshold).astype(int)
        out[f"{target_col}_near_zero"] = (p.abs() <= active_threshold).astype(int)

        # Ramp event flags
        diff_1 = p.diff(1)
        out[f"{target_col}_ramp_up"] = (diff_1 > ramp_threshold).astype(int)
        out[f"{target_col}_ramp_down"] = (diff_1 < -ramp_threshold).astype(int)
        out[f"{target_col}_ramp_any"] = (
            (out[f"{target_col}_ramp_up"] == 1) |
            (out[f"{target_col}_ramp_down"] == 1)
        ).astype(int)

        # Time since last ramp. At each timestamp this is known from history/current observation.
        ramp_any = out[f"{target_col}_ramp_any"] == 1
        ramp_group = ramp_any.cumsum()
        since_ramp = out.groupby(ramp_group).cumcount()
        since_ramp = since_ramp.where(ramp_group > 0, np.nan)
        if since_ramp_cap is not None:
            since_ramp = since_ramp.clip(upper=since_ramp_cap)
        out[f"{target_col}_steps_since_ramp"] = since_ramp

        pattern_base_cols = [
            f"{target_col}_active",
            f"{target_col}_near_zero",
            f"{target_col}_ramp_up",
            f"{target_col}_ramp_down",
            f"{target_col}_ramp_any",
            f"{target_col}_steps_since_ramp",
        ]
        pattern_base_cols += [f"{target_col}_abs_diff_{lag}" for lag in diff_lags]

        for col in pattern_base_cols:
            shifted = out[col].shift(1)
            for win in rolling_windows:
                out[f"{col}_roll_sum_{win}"] = shifted.rolling(win).sum()
                out[f"{col}_roll_mean_{win}"] = shifted.rolling(win).mean()

    return out


def add_historical_baseline_residual_features(
    df: pd.DataFrame,
    target_cols: list[str],
    *,
    steps_per_day: int = 96,
    steps_per_week: int | None = None,
    rolling_days: int = 7,
    add_same_slot_baseline: bool = True,
) -> pd.DataFrame:

    out = df.copy().sort_index()

    if steps_per_week is None:
        steps_per_week = steps_per_day * 7

    target_cols = [c for c in target_cols if c in out.columns]

    for target_col in target_cols:
        p = out[target_col]

        prev_day_same_time = p.shift(steps_per_day)
        prev_week_same_time = p.shift(steps_per_week)

        prev_day_mean = p.shift(1).rolling(steps_per_day).mean()
        prev_week_mean = p.shift(1).rolling(steps_per_week).mean()
        rolling_hist_mean = p.shift(1).rolling(steps_per_day * rolling_days).mean()

        out[f"{target_col}_prev_day_same_time"] = prev_day_same_time
        out[f"{target_col}_prev_week_same_time"] = prev_week_same_time
        out[f"{target_col}_prev_day_mean"] = prev_day_mean
        out[f"{target_col}_prev_week_mean"] = prev_week_mean
        out[f"{target_col}_rolling_{rolling_days}d_mean"] = rolling_hist_mean

        out[f"{target_col}_dev_from_prev_day_mean"] = p - prev_day_mean
        out[f"{target_col}_dev_from_prev_week_mean"] = p - prev_week_mean
        out[f"{target_col}_dev_from_rolling_{rolling_days}d_mean"] = p - rolling_hist_mean

        if add_same_slot_baseline:
            out[f"{target_col}_dev_from_prev_day_same_time"] = p - prev_day_same_time
            out[f"{target_col}_dev_from_prev_week_same_time"] = p - prev_week_same_time

    return out


def get_ramp_pattern_feature_columns(
    target_col: str,
    *,
    diff_lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> list[str]:
  
    if diff_lags is None:
        diff_lags = [1, 2, 4]
    if rolling_windows is None:
        rolling_windows = [4, 12, 24, 96]

    cols: list[str] = []

    for lag in diff_lags:
        cols += [
            f"{target_col}_diff_{lag}",
            f"{target_col}_abs_diff_{lag}",
        ]

    cols += [
        f"{target_col}_active",
        f"{target_col}_near_zero",
        f"{target_col}_ramp_up",
        f"{target_col}_ramp_down",
        f"{target_col}_ramp_any",
        f"{target_col}_steps_since_ramp",
    ]

    pattern_base_cols = [
        f"{target_col}_active",
        f"{target_col}_near_zero",
        f"{target_col}_ramp_up",
        f"{target_col}_ramp_down",
        f"{target_col}_ramp_any",
        f"{target_col}_steps_since_ramp",
    ]
    pattern_base_cols += [f"{target_col}_abs_diff_{lag}" for lag in diff_lags]

    for col in pattern_base_cols:
        for win in rolling_windows:
            cols.append(f"{col}_roll_sum_{win}")
            cols.append(f"{col}_roll_mean_{win}")

    return cols


def get_historical_baseline_residual_feature_columns(
    target_col: str,
    *,
    rolling_days: int = 7,
    add_same_slot_baseline: bool = True,
) -> list[str]:

    cols = [
        f"{target_col}_prev_day_same_time",
        f"{target_col}_prev_week_same_time",
        f"{target_col}_prev_day_mean",
        f"{target_col}_prev_week_mean",
        f"{target_col}_rolling_{rolling_days}d_mean",
        f"{target_col}_dev_from_prev_day_mean",
        f"{target_col}_dev_from_prev_week_mean",
        f"{target_col}_dev_from_rolling_{rolling_days}d_mean",
    ]

    if add_same_slot_baseline:
        cols += [
            f"{target_col}_dev_from_prev_day_same_time",
            f"{target_col}_dev_from_prev_week_same_time",
        ]

    return cols


def get_pattern_feature_columns(
    target_col: str,
    *,
    diff_lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    rolling_days: int = 7,
) -> list[str]:

    return (
        get_ramp_pattern_feature_columns(
            target_col,
            diff_lags=diff_lags,
            rolling_windows=rolling_windows,
        )
        + get_historical_baseline_residual_feature_columns(
            target_col,
            rolling_days=rolling_days,
        )
    )
