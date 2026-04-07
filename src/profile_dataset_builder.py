from __future__ import annotations

import pandas as pd


def _validate_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")


def build_day_ahead_profile_dataset(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    horizon_steps: int = 96,
    issue_hour: int = 23,
    issue_minute: int = 45,
    drop_feature_nan: bool = False,
    drop_target_nan: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    _validate_datetime_index(df)

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    data = df.sort_index().copy()

    # Keep only forecast issue timestamps, e.g. 23:45 each day
    issue_mask = (
        (data.index.hour == issue_hour) &
        (data.index.minute == issue_minute)
    )
    issue_df = data.loc[issue_mask, feature_cols].copy()

    # Build multi-step future targets from full dataframe
    target_frames = []
    target_names = []

    for step in range(1, horizon_steps + 1):
        col_name = f"y_tplus_{step:03d}"
        shifted = data[target_col].shift(-step)
        aligned = shifted.reindex(issue_df.index)
        target_frames.append(aligned.rename(col_name))
        target_names.append(col_name)

    Y = pd.concat(target_frames, axis=1)
    X = issue_df.copy()

    # Drop incomplete rows near the end or rows with missing targets
    if drop_target_nan:
        valid_target_mask = Y.notna().all(axis=1)
        X = X.loc[valid_target_mask]
        Y = Y.loc[valid_target_mask]

    # Optionally drop rows with missing features
    if drop_feature_nan:
        valid_feature_mask = X.notna().all(axis=1)
        X = X.loc[valid_feature_mask]
        Y = Y.loc[valid_feature_mask]

    return X, Y