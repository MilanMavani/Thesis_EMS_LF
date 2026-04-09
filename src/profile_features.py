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
    midday_start_hour: int = 11,
    midday_end_hour: int = 14,
) -> pd.DataFrame:
    _validate_datetime_index(df)

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe")

    out = df.copy().sort_index()
    y = pd.to_numeric(out[target_col], errors="coerce")

    date_key = out.index.normalize()
    hour = out.index.hour
    midday_mask = (hour >= midday_start_hour) & (hour < midday_end_hour)

    daily_mean = y.groupby(date_key).mean()
    daily_max = y.groupby(date_key).max()
    daily_midday_max = y[midday_mask].groupby(date_key[midday_mask]).max()

    # Safe peak time calculation: returns NaN for all-NaN days
    def _safe_peak_time_step(day_series: pd.Series) -> float:
        day_series = day_series.dropna()
        if day_series.empty:
            return np.nan

        peak_ts = day_series.idxmax()
        return float((peak_ts.hour * 60 + peak_ts.minute) // freq_minutes)

    daily_peak_time_step = y.groupby(date_key).apply(_safe_peak_time_step)

    feature_df = pd.DataFrame({
        f"{target_col}_yesterday_mean": daily_mean.shift(1),
        f"{target_col}_yesterday_max": daily_max.shift(1),
        f"{target_col}_yesterday_midday_max": daily_midday_max.shift(1),
        f"{target_col}_yesterday_peak_time_step": daily_peak_time_step.shift(1),
        f"{target_col}_lastweek_same_day_peak": daily_max.shift(7),
    })

    out["_date_key_tmp"] = date_key
    out = out.join(feature_df, on="_date_key_tmp")
    out = out.drop(columns=["_date_key_tmp"])

    return out