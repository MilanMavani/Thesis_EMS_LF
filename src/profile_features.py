from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")


def add_day_ahead_history_features(
    df: pd.DataFrame,
    *,
    target_col: str,
    freq_minutes: int = 15,
) -> pd.DataFrame:
    """
    Add historical daily summary features for day-ahead forecasting.

    Features added:
    - yesterday_mean
    - yesterday_max
    - yesterday_midday_max
    - yesterday_peak_time_step
    - lastweek_same_day_peak

    Assumes:
    - DatetimeIndex
    - regular time series
    - target_col exists
    """
    _validate_datetime_index(df)

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe")

    out = df.copy().sort_index()
    y = pd.to_numeric(out[target_col], errors="coerce")

    # calendar helpers
    out["_date"] = out.index.normalize()
    out["_hour"] = out.index.hour
    out["_minute"] = out.index.minute

    # step within day for 15-min data: 0..95
    steps_per_day = int(24 * 60 / freq_minutes)
    out["_step_in_day"] = ((out.index.hour * 60 + out.index.minute) // freq_minutes).astype(int)

    # -----------------------------
    # Previous-day aggregate stats
    # -----------------------------
    daily_mean = y.groupby(out["_date"]).mean()
    daily_max = y.groupby(out["_date"]).max()

    # Midday window: 11:00 to 14:00
    midday_mask = (out["_hour"] >= 11) & (out["_hour"] < 14)
    midday_max = y[midday_mask].groupby(out.loc[midday_mask, "_date"]).max()

    # Peak timing: step index of daily max
    # returns step within day of the first occurrence of max
    peak_idx = y.groupby(out["_date"]).idxmax()

    peak_step_rows = []
    for day, ts in peak_idx.items():
        if pd.isna(ts):
            peak_step_rows.append((day, np.nan))
        else:
            step = ((ts.hour * 60 + ts.minute) // freq_minutes)
            peak_step_rows.append((day, float(step)))

    daily_peak_step = pd.Series(
        data=[v for _, v in peak_step_rows],
        index=[k for k, _ in peak_step_rows],
        name="daily_peak_step",
    )

    daily_feature_df = pd.DataFrame({
        "daily_mean": daily_mean,
        "daily_max": daily_max,
        "daily_midday_max": midday_max,
        "daily_peak_step": daily_peak_step,
    })

    # Shift by 1 day to become "yesterday_*"
    daily_feature_df["yesterday_mean"] = daily_feature_df["daily_mean"].shift(1)
    daily_feature_df["yesterday_max"] = daily_feature_df["daily_max"].shift(1)
    daily_feature_df["yesterday_midday_max"] = daily_feature_df["daily_midday_max"].shift(1)
    daily_feature_df["yesterday_peak_time_step"] = daily_feature_df["daily_peak_step"].shift(1)

    # Last-week same-day peak = daily max from 7 days before
    daily_feature_df["lastweek_same_day_peak"] = daily_feature_df["daily_max"].shift(7)

    # Join daily features back to every timestamp of that day
    keep_cols = [
        "yesterday_mean",
        "yesterday_max",
        "yesterday_midday_max",
        "yesterday_peak_time_step",
        "lastweek_same_day_peak",
    ]

    out = out.join(daily_feature_df[keep_cols], on="_date")

    # cleanup
    out = out.drop(columns=["_date", "_hour", "_minute", "_step_in_day"])

    return out


def get_day_ahead_history_feature_columns() -> list[str]:
    return [
        "yesterday_mean",
        "yesterday_max",
        "yesterday_midday_max",
        "yesterday_peak_time_step",
        "lastweek_same_day_peak",
    ]