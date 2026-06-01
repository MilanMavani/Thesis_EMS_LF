from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_daily_load_profiles(
    df: pd.DataFrame,
    target_col: str,
    *,
    freq: str = "15min",
    expected_steps: int | None = None,
    min_valid_fraction: float = 0.90,
) -> pd.DataFrame:
    """
    Convert a time-series load column into daily wide profiles.

    Rows = dates
    Columns = time slots inside the day
    Example for 15min data: 96 columns per day
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in dataframe.")

    d = df[[target_col]].copy().sort_index()
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce")

    if expected_steps is None:
        expected_steps = int(pd.Timedelta(days=1) / pd.to_timedelta(freq))

    d["date"] = d.index.date
    d["slot"] = ((d.index.hour * 60 + d.index.minute) / (pd.to_timedelta(freq).seconds / 60)).astype(int)

    profiles = d.pivot_table(
        index="date",
        columns="slot",
        values=target_col,
        aggfunc="mean",
    )

    # Force expected slot columns
    profiles = profiles.reindex(columns=range(expected_steps))

    # Keep only days with enough valid values
    valid_fraction = profiles.notna().mean(axis=1)
    profiles = profiles.loc[valid_fraction >= min_valid_fraction].copy()

    # Fill remaining small gaps inside daily profiles
    profiles = profiles.interpolate(axis=1, limit_direction="both")

    profiles.index = pd.to_datetime(profiles.index)

    return profiles


def add_daily_pattern_cluster_feature(
    df: pd.DataFrame,
    target_col: str,
    *,
    freq: str = "15min",
    n_clusters: int = 4,
    min_valid_fraction: float = 0.90,
    random_state: int = 42,
    normalize_daily_shape: bool = True,
    feature_name: str = "previous_day_pattern_cluster",
) -> tuple[pd.DataFrame, pd.DataFrame, KMeans, StandardScaler]:
    """
    Add previous-day daily pattern cluster as a feature.

    The clustering is performed on historical daily profiles.
    The cluster label of day D is assigned to timestamps of day D+1.
    This avoids future leakage because the model only sees yesterday's pattern.
    """
    out = df.copy().sort_index()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    expected_steps = int(pd.Timedelta(days=1) / pd.to_timedelta(freq))

    profiles = build_daily_load_profiles(
        out,
        target_col=target_col,
        freq=freq,
        expected_steps=expected_steps,
        min_valid_fraction=min_valid_fraction,
    )

    if len(profiles) < n_clusters:
        raise ValueError(
            f"Not enough valid daily profiles for {n_clusters} clusters. "
            f"Only {len(profiles)} valid days found."
        )

    X = profiles.copy()

    if normalize_daily_shape:
        # Shape-based clustering:
        # each day is normalized, so clustering focuses more on pattern shape
        # than only absolute load magnitude.
        daily_mean = X.mean(axis=1)
        daily_std = X.std(axis=1).replace(0, np.nan)
        X = X.sub(daily_mean, axis=0).div(daily_std, axis=0)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
    )

    clusters = kmeans.fit_predict(X_scaled)

    cluster_df = pd.DataFrame(
        {
            "date": profiles.index,
            "pattern_cluster": clusters,
            "daily_mean": profiles.mean(axis=1).values,
            "daily_max": profiles.max(axis=1).values,
            "daily_min": profiles.min(axis=1).values,
            "daily_std": profiles.std(axis=1).values,
            "daily_energy_kWh": profiles.sum(axis=1).values * (pd.to_timedelta(freq) / pd.Timedelta(hours=1)),
        }
    )

    # Assign yesterday's cluster to today's timestamps
    cluster_df["feature_date"] = cluster_df["date"] + pd.Timedelta(days=1)

    mapping = cluster_df.set_index("feature_date")["pattern_cluster"]

    out["_date_for_cluster"] = out.index.normalize()
    out[feature_name] = out["_date_for_cluster"].map(mapping)

    # Fill early missing value safely using past/available cluster mode
    if out[feature_name].notna().any():
        mode_cluster = int(out[feature_name].mode().iloc[0])
        out[feature_name] = out[feature_name].fillna(mode_cluster)
    else:
        out[feature_name] = 0

    out[feature_name] = out[feature_name].astype(int)
    out = out.drop(columns=["_date_for_cluster"])

    return out, cluster_df, kmeans, scaler


def summarize_pattern_clusters(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact summary of each pattern cluster.
    """
    summary = (
        cluster_df
        .groupby("pattern_cluster")
        .agg(
            days=("date", "count"),
            avg_daily_energy_kWh=("daily_energy_kWh", "mean"),
            avg_daily_mean_kW=("daily_mean", "mean"),
            avg_daily_max_kW=("daily_max", "mean"),
            avg_daily_std_kW=("daily_std", "mean"),
        )
        .reset_index()
        .sort_values("pattern_cluster")
    )

    return summary