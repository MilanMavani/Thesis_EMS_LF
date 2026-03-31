from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(
    df: pd.DataFrame,
    *,
    include_extended: bool = False,
) -> pd.DataFrame:
    out = df.copy().sort_index()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")

    idx = out.index

    out["dayofweek"] = idx.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["hour"] = idx.hour
    out["month"] = idx.month
    out["dayofmonth"] = idx.day
    out["minute_of_day"] = idx.hour * 60 + idx.minute

    out["sin_tod"] = np.sin(2 * np.pi * out["minute_of_day"] / 1440)
    out["cos_tod"] = np.cos(2 * np.pi * out["minute_of_day"] / 1440)

    out["sin_dow"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["cos_dow"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    out["is_business_hours"] = (
        (idx.hour >= 8) & (idx.hour < 18) & (out["dayofweek"] < 5)
    ).astype(int)

    if include_extended:
        out["is_midday_peak_window"] = ((idx.hour >= 11) & (idx.hour < 14)).astype(int)
        out["is_midday_weekday_peak"] = (
            (idx.hour >= 11) & (idx.hour < 14) & (idx.dayofweek < 5)
        ).astype(int)
        out["slot_5min"] = idx.hour * 12 + idx.minute // 5

    return out


def get_calendar_feature_columns(*, include_extended: bool = False) -> list[str]:
    cols = [
        "dayofweek",
        "is_weekend",
        "hour",
        "month",
        "dayofmonth",
        "minute_of_day",
        "sin_tod",
        "cos_tod",
        "sin_dow",
        "cos_dow",
        "is_business_hours",
    ]

    if include_extended:
        cols += [
            "is_midday_peak_window",
            "is_midday_weekday_peak",
            "slot_5min",
        ]

    return cols