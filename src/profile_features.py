from __future__ import annotations

import numpy as np
import pandas as pd


def add_day_ahead_history_features(
    df: pd.DataFrame,
    *,
    target_col: str,
    freq_minutes: int = 30,
    min_history: int = 3,
) -> pd.DataFrame:
    """
    Add past-only historical pattern features for day-ahead/profile forecasting.

    Features created:
    - target slot historical mean / p75 / p90
    - target day-of-week + slot historical mean / p75 / p90
    - same-slot last 4 weeks mean / max / p90

    Important:
    These features are calculated using only past data, so they avoid leakage.
    """

    out = df.copy().sort_index()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    if target_col not in out.columns:
        raise ValueError(f"target_col not found in dataframe: {target_col}")

    y = pd.to_numeric(out[target_col], errors="coerce")

    slot_col = f"{target_col}_slot"
    dow_col = f"{target_col}_dow"

    out[slot_col] = (out.index.hour * 60 + out.index.minute) // freq_minutes
    out[dow_col] = out.index.dayofweek

    slot_mean = []
    slot_p75 = []
    slot_p90 = []

    dow_slot_mean = []
    dow_slot_p75 = []
    dow_slot_p90 = []

    same_slot_last4w_mean = []
    same_slot_last4w_max = []
    same_slot_last4w_p90 = []

    for ts in out.index:
        slot = out.loc[ts, slot_col]
        dow = out.loc[ts, dow_col]

        past_mask = out.index < ts
        past = out.loc[past_mask, [target_col, slot_col, dow_col]].copy()
        past[target_col] = pd.to_numeric(past[target_col], errors="coerce")

        # -------------------------------------------------
        # Same time-slot pattern across previous days
        # -------------------------------------------------
        hist_slot = past.loc[past[slot_col] == slot, target_col].dropna()

        if len(hist_slot) >= min_history:
            slot_mean.append(float(hist_slot.mean()))
            slot_p75.append(float(hist_slot.quantile(0.75)))
            slot_p90.append(float(hist_slot.quantile(0.90)))
        else:
            slot_mean.append(np.nan)
            slot_p75.append(np.nan)
            slot_p90.append(np.nan)

        # -------------------------------------------------
        # Same day-of-week + same time-slot pattern
        # -------------------------------------------------
        hist_dow_slot = past.loc[
            (past[dow_col] == dow) & (past[slot_col] == slot),
            target_col,
        ].dropna()

        if len(hist_dow_slot) >= min_history:
            dow_slot_mean.append(float(hist_dow_slot.mean()))
            dow_slot_p75.append(float(hist_dow_slot.quantile(0.75)))
            dow_slot_p90.append(float(hist_dow_slot.quantile(0.90)))
        else:
            dow_slot_mean.append(np.nan)
            dow_slot_p75.append(np.nan)
            dow_slot_p90.append(np.nan)

        # -------------------------------------------------
        # Same slot from previous 4 weeks
        # -------------------------------------------------
        vals = []

        for k in range(1, 5):
            lag_time = ts - pd.Timedelta(days=7 * k)

            if lag_time in out.index:
                val = y.loc[lag_time]

                if pd.notna(val):
                    vals.append(float(val))

        if len(vals) >= 1:
            same_slot_last4w_mean.append(float(np.mean(vals)))
            same_slot_last4w_max.append(float(np.max(vals)))
            same_slot_last4w_p90.append(float(np.quantile(vals, 0.90)))
        else:
            same_slot_last4w_mean.append(np.nan)
            same_slot_last4w_max.append(np.nan)
            same_slot_last4w_p90.append(np.nan)

    prefix = target_col

    out[f"{prefix}_slot_mean_past"] = slot_mean
    out[f"{prefix}_slot_p75_past"] = slot_p75
    out[f"{prefix}_slot_p90_past"] = slot_p90

    out[f"{prefix}_dow_slot_mean_past"] = dow_slot_mean
    out[f"{prefix}_dow_slot_p75_past"] = dow_slot_p75
    out[f"{prefix}_dow_slot_p90_past"] = dow_slot_p90

    out[f"{prefix}_same_slot_last4w_mean"] = same_slot_last4w_mean
    out[f"{prefix}_same_slot_last4w_max"] = same_slot_last4w_max
    out[f"{prefix}_same_slot_last4w_p90"] = same_slot_last4w_p90

    # These helper columns are optional.
    # Keep them if you want to use them as features.
    # Remove them if you already have generic slot/day features elsewhere.
    out = out.drop(columns=[slot_col, dow_col], errors="ignore")

    return out


def get_day_ahead_history_feature_columns(target_col: str) -> list[str]:
    """
    Return the feature names created by add_day_ahead_history_features().
    """
    return [
        f"{target_col}_slot_mean_past",
        f"{target_col}_slot_p75_past",
        f"{target_col}_slot_p90_past",
        f"{target_col}_dow_slot_mean_past",
        f"{target_col}_dow_slot_p75_past",
        f"{target_col}_dow_slot_p90_past",
        f"{target_col}_same_slot_last4w_mean",
        f"{target_col}_same_slot_last4w_max",
        f"{target_col}_same_slot_last4w_p90",
    ]


def add_lag_imputation_flags(
    df: pd.DataFrame,
    *,
    target_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """
    Add lagged imputation flags.

    Example:
    If target_col = BU_TotActPwr_Academy and lag = 48,
    this creates:
    BU_TotActPwr_Academy_lag_48_was_imputed

    For 30-minute data:
    - lag 48 = same time yesterday
    - lag 96 = same time two days ago
    - lag 336 = same time last week

    For 15-minute data:
    - lag 96 = same time yesterday
    - lag 192 = same time two days ago
    - lag 672 = same time last week
    """

    out = df.copy().sort_index()

    base_flag = f"{target_col}_was_imputed"

    if base_flag not in out.columns:
        raise ValueError(
            f"Missing base imputation flag column: {base_flag}. "
            "Run imputation with flags first."
        )

    for lag in lags:
        out[f"{target_col}_lag_{lag}_was_imputed"] = (
            out[base_flag]
            .shift(lag)
            .fillna(0)
            .astype(int)
        )

    return out