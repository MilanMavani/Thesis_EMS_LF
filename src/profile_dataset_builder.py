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
    """
    Build a day-ahead profile forecasting dataset.

    Each row in X corresponds to one forecast issue timestamp
    (default: 23:45), and each row in Y contains the next 96 target values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with DatetimeIndex and all required feature columns.
    target_col : str
        Name of the target column.
    feature_cols : list[str]
        Feature columns used as model input.
    horizon_steps : int, default=96
        Number of future 15-minute steps to predict.
    issue_hour : int, default=23
        Hour of forecast issue time.
    issue_minute : int, default=45
        Minute of forecast issue time.
    drop_feature_nan : bool, default=False
        If True, rows with NaNs in feature columns are removed.
    drop_target_nan : bool, default=True
        If True, rows with NaNs in future targets are removed.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with one row per issue time.
    Y : pd.DataFrame
        Multi-output target matrix with columns y_tplus_001 ... y_tplus_096.
    """
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