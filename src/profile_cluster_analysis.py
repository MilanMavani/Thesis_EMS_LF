from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


def load_behaviour_summary(
    df: pd.DataFrame,
    target_col: str,
    low_quantile: float = 0.25,
    high_quantile: float = 0.90,
) -> pd.DataFrame:
    s = pd.to_numeric(df[target_col], errors="coerce").dropna()

    low_thr = s.quantile(low_quantile)
    high_thr = s.quantile(high_quantile)

    return pd.DataFrame([{
        "target_col": target_col,
        "n_samples": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "min": s.min(),
        "max": s.max(),
        "q25": s.quantile(0.25),
        "q75": s.quantile(0.75),
        "q90": s.quantile(0.90),
        "q95": s.quantile(0.95),
        "coefficient_of_variation": s.std() / s.mean() if s.mean() != 0 else np.nan,
        "zero_or_near_zero_%": (s.abs() < 0.01).mean() * 100,
        "low_load_%": (s <= low_thr).mean() * 100,
        "high_load_%": (s >= high_thr).mean() * 100,
        "low_load_threshold": low_thr,
        "high_load_threshold": high_thr,
    }])


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    out = df.copy()
    out["date"] = out.index.date
    out["hour"] = out.index.hour
    out["minute"] = out.index.minute
    out["minute_of_day"] = out.index.hour * 60 + out.index.minute
    out["dayofweek_num"] = out.index.dayofweek
    out["dayofweek"] = out.index.day_name()
    out["is_weekend"] = out["dayofweek_num"].isin([5, 6]).astype(int)
    return out


def make_daily_profile_matrix(
    df: pd.DataFrame,
    target_col: str,
    freq: str = "30min",
    min_daily_coverage: float = 0.90,
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    d = df[[target_col]].copy().sort_index()
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
    d = d.resample(freq).mean()

    expected_steps = int(pd.Timedelta("1D") / pd.to_timedelta(freq))

    daily_profiles = (
        d[target_col]
        .groupby(d.index.date)
        .apply(lambda x: x.reset_index(drop=True))
        .unstack()
    )

    daily_profiles = daily_profiles.loc[:, :expected_steps - 1]

    coverage = daily_profiles.notna().mean(axis=1)
    daily_profiles = daily_profiles[coverage >= min_daily_coverage].copy()

    daily_profiles = daily_profiles.interpolate(axis=1, limit_direction="both")
    daily_profiles.index = pd.to_datetime(daily_profiles.index)

    return daily_profiles


def scale_daily_profiles(
    daily_profiles: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler]:
    X = daily_profiles.to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def evaluate_cluster_numbers(
    daily_profiles: pd.DataFrame,
    cluster_range=range(2, 8),
    random_state: int = 42,
) -> pd.DataFrame:
    X_scaled, _ = scale_daily_profiles(daily_profiles)

    rows = []

    for k in cluster_range:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
        )

        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)

        rows.append({
            "n_clusters": k,
            "silhouette_score": sil,
        })

    return pd.DataFrame(rows)


def cluster_daily_profiles(
    daily_profiles: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    X_scaled, scaler = scale_daily_profiles(daily_profiles)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
    )

    labels = model.fit_predict(X_scaled)

    clustered = daily_profiles.copy()
    clustered["pattern_cluster"] = labels
    clustered["date"] = clustered.index
    clustered["dayofweek"] = clustered.index.day_name()
    clustered["is_weekend"] = clustered.index.dayofweek.isin([5, 6]).astype(int)

    return clustered, model, scaler


def get_profile_columns(clustered_profiles: pd.DataFrame) -> list:
    return [
        c for c in clustered_profiles.columns
        if isinstance(c, int)
    ]


def summarize_clusters(
    clustered_profiles: pd.DataFrame,
    freq: str = "30min",
) -> pd.DataFrame:
    profile_cols = get_profile_columns(clustered_profiles)
    step_hours = pd.to_timedelta(freq).total_seconds() / 3600

    rows = []

    for cl in sorted(clustered_profiles["pattern_cluster"].unique()):
        temp = clustered_profiles[clustered_profiles["pattern_cluster"] == cl]
        values = temp[profile_cols].to_numpy(dtype=float).ravel()

        daily_energy = temp[profile_cols].sum(axis=1) * step_hours

        rows.append({
            "pattern_cluster": cl,
            "n_days": len(temp),
            "mean_load_kW": np.nanmean(values),
            "median_load_kW": np.nanmedian(values),
            "max_load_kW": np.nanmax(values),
            "std_load_kW": np.nanstd(values),
            "daily_energy_kWh_mean": daily_energy.mean(),
            "daily_energy_kWh_median": daily_energy.median(),
            "daily_energy_kWh_std": daily_energy.std(),
            "daily_energy_kWh_cv": daily_energy.std() / daily_energy.mean()
            if daily_energy.mean() != 0 else np.nan,
        })

    return pd.DataFrame(rows)


def calculate_cluster_homogeneity(
    clustered_profiles: pd.DataFrame,
    freq: str = "30min",
) -> pd.DataFrame:
    profile_cols = get_profile_columns(clustered_profiles)
    step_hours = pd.to_timedelta(freq).total_seconds() / 3600

    rows = []

    for cl in sorted(clustered_profiles["pattern_cluster"].unique()):
        temp = clustered_profiles[clustered_profiles["pattern_cluster"] == cl]
        X = temp[profile_cols].to_numpy(dtype=float)

        centroid = X.mean(axis=0)

        distances = pairwise_distances(
            X,
            centroid.reshape(1, -1),
            metric="euclidean",
        ).ravel()

        profile_std = X.std(axis=0)
        daily_energy = X.sum(axis=1) * step_hours

        rows.append({
            "pattern_cluster": cl,
            "n_days": len(temp),
            "mean_distance_to_centroid": distances.mean(),
            "median_distance_to_centroid": np.median(distances),
            "max_distance_to_centroid": distances.max(),
            "mean_profile_std_kW": profile_std.mean(),
            "max_profile_std_kW": profile_std.max(),
            "daily_energy_kWh_mean": daily_energy.mean(),
            "daily_energy_kWh_std": daily_energy.std(),
            "daily_energy_kWh_cv": daily_energy.std() / daily_energy.mean()
            if daily_energy.mean() != 0 else np.nan,
        })

    return pd.DataFrame(rows)


def add_distance_to_centroid(
    clustered_profiles: pd.DataFrame,
) -> pd.DataFrame:
    profile_cols = get_profile_columns(clustered_profiles)

    out = clustered_profiles.copy()
    out["distance_to_cluster_centroid"] = np.nan

    for cl in sorted(out["pattern_cluster"].unique()):
        idx = out["pattern_cluster"] == cl
        X = out.loc[idx, profile_cols].to_numpy(dtype=float)

        centroid = X.mean(axis=0)

        distances = pairwise_distances(
            X,
            centroid.reshape(1, -1),
            metric="euclidean",
        ).ravel()

        out.loc[idx, "distance_to_cluster_centroid"] = distances

    return out


def get_farthest_days(
    clustered_profiles_with_distance: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    required_col = "distance_to_cluster_centroid"

    if required_col not in clustered_profiles_with_distance.columns:
        raise ValueError("Run add_distance_to_centroid() first.")

    return (
        clustered_profiles_with_distance
        .sort_values(
            ["pattern_cluster", required_col],
            ascending=[True, False],
        )
        .groupby("pattern_cluster")
        .head(top_n)
        [["date", "dayofweek", "is_weekend", "pattern_cluster", required_col]]
        .reset_index(drop=True)
    )


def cluster_distribution_by_weekday(
    clustered_profiles: pd.DataFrame,
) -> pd.DataFrame:
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]

    table = pd.crosstab(
        clustered_profiles["dayofweek"],
        clustered_profiles["pattern_cluster"],
        normalize="index",
    ) * 100

    return table.reindex(weekday_order).round(2)


def plot_average_daily_profile(
    df: pd.DataFrame,
    target_col: str,
) -> None:
    d = add_time_columns(df)

    profile = (
        d.groupby("minute_of_day")[target_col]
        .mean()
    )

    plt.figure(figsize=(14, 5))
    plt.plot(profile.index / 60, profile.values)
    plt.xlabel("Hour of day")
    plt.ylabel(f"{target_col} [kW]")
    plt.title(f"Average daily load profile - {target_col}")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_weekday_profiles(
    df: pd.DataFrame,
    target_col: str,
) -> None:
    d = add_time_columns(df)

    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]

    profile = (
        d.groupby(["dayofweek", "minute_of_day"])[target_col]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(15, 6))

    for day in weekday_order:
        temp = profile[profile["dayofweek"] == day]
        if not temp.empty:
            plt.plot(
                temp["minute_of_day"] / 60,
                temp[target_col],
                label=day,
            )

    plt.xlabel("Hour of day")
    plt.ylabel(f"{target_col} [kW]")
    plt.title(f"Average weekday profiles - {target_col}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_cluster_profiles(
    clustered_profiles: pd.DataFrame,
    target_col: str,
    freq: str = "30min",
    show_std_band: bool = True,
) -> None:
    profile_cols = get_profile_columns(clustered_profiles)

    step_minutes = int(pd.to_timedelta(freq).total_seconds() / 60)
    x_hours = np.arange(len(profile_cols)) * step_minutes / 60

    plt.figure(figsize=(15, 6))

    for cl in sorted(clustered_profiles["pattern_cluster"].unique()):
        temp = clustered_profiles[clustered_profiles["pattern_cluster"] == cl]

        mean_profile = temp[profile_cols].mean(axis=0)
        std_profile = temp[profile_cols].std(axis=0)

        plt.plot(
            x_hours,
            mean_profile.values,
            label=f"Cluster {cl} | n={len(temp)} days",
        )

        if show_std_band:
            plt.fill_between(
                x_hours,
                mean_profile.values - std_profile.values,
                mean_profile.values + std_profile.values,
                alpha=0.15,
            )

    plt.xlabel("Hour of day")
    plt.ylabel(f"{target_col} [kW]")
    plt.title(f"Average daily pattern clusters - {target_col}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_daily_profile_heatmap(
    daily_profiles: pd.DataFrame,
    target_col: str,
    freq: str = "30min",
) -> None:
    plt.figure(figsize=(15, 8))

    plt.imshow(
        daily_profiles.values,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(label=f"{target_col} [kW]")
    plt.xlabel("Time step of day")
    plt.ylabel("Day")
    plt.title(f"Daily load profile heatmap - {target_col}")

    step_minutes = int(pd.to_timedelta(freq).total_seconds() / 60)
    steps_per_hour = int(60 / step_minutes)

    xticks = np.arange(0, daily_profiles.shape[1], steps_per_hour * 2)
    xtick_labels = [str(int(x / steps_per_hour)) for x in xticks]

    plt.xticks(xticks, xtick_labels)
    plt.show()