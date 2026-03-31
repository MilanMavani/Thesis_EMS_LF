from __future__ import annotations

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    load_cols: list[str],
    *,
    lags: list[int],
) -> pd.DataFrame:
    out = df.copy().sort_index()
    load_cols = [c for c in load_cols if c in out.columns]

    for col in load_cols:
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)

    return out


def add_rolling_features(
    df: pd.DataFrame,
    load_cols: list[str],
    *,
    rolling_windows: list[int],
    add_std_windows: list[int] | None = None,
) -> pd.DataFrame:
    out = df.copy().sort_index()
    load_cols = [c for c in load_cols if c in out.columns]

    if add_std_windows is None:
        add_std_windows = []

    for col in load_cols:
        shifted = out[col].shift(1)

        for win in rolling_windows:
            out[f"{col}_roll_mean_{win}"] = shifted.rolling(win).mean()

        for win in add_std_windows:
            out[f"{col}_roll_std_{win}"] = shifted.rolling(win).std()

    return out


def add_trend_features(
    df: pd.DataFrame,
    load_cols: list[str],
    *,
    trend_pairs: list[tuple[int, int]],
) -> pd.DataFrame:
    out = df.copy().sort_index()
    load_cols = [c for c in load_cols if c in out.columns]

    for col in load_cols:
        for short_lag, long_lag in trend_pairs:
            out[f"{col}_trend_{short_lag}_{long_lag}"] = out[col].shift(short_lag) - out[col].shift(long_lag)

    return out