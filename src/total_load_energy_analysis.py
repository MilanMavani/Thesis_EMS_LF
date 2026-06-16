from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


RESOLUTION_ORDER = {"15min": 0, "30min": 1, "60min": 2, "1h": 2}

DT_HOURS_MAP = {
    "15min": 0.25,
    "30min": 0.50,
    "60min": 1.00,
    "1h": 1.00,
}

FREQ_MAP = {
    "15min": "15min",
    "30min": "30min",
    "60min": "60min",
    "1h": "60min",
}


def sort_horizon_columns(cols: list[str]) -> list[str]:
    def horizon_number(col: str) -> int:
        base = col.replace("_true", "").replace("_pred", "")
        nums = [p for p in base.split("_") if p.isdigit()]
        return int(nums[-1]) if nums else 999999

    return sorted(cols, key=horizon_number)


def load_best_prediction_registry(
    registry_path: str | Path,
    target_cols: list[str],
    resolutions: list[str],
) -> pd.DataFrame:
    registry = pd.read_csv(registry_path)

    required_cols = {"target_col", "resolution", "model_name", "prediction_path"}
    missing = required_cols - set(registry.columns)
    if missing:
        raise ValueError(f"Registry is missing required columns: {missing}")

    registry = registry[
        registry["target_col"].isin(target_cols)
        & registry["resolution"].isin(resolutions)
    ].copy()

    registry["prediction_path"] = registry["prediction_path"].astype(str)
    registry["resolution_order"] = registry["resolution"].map(RESOLUTION_ORDER)

    return (
        registry
        .sort_values(["resolution_order", "target_col"])
        .drop(columns="resolution_order")
        .reset_index(drop=True)
    )


def read_profile_prediction_file_long(
    row: pd.Series,
    *,
    dt_hours_map: dict[str, float] | None = None,
    freq_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    dt_hours_map = dt_hours_map or DT_HOURS_MAP
    freq_map = freq_map or FREQ_MAP

    file_path = Path(row["prediction_path"])
    target_col = row["target_col"]
    resolution = row["resolution"]
    model_name = row["model_name"]

    if not file_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {file_path}")

    dt_hours = dt_hours_map[resolution]
    freq = freq_map[resolution]

    df = pd.read_csv(file_path)
    df = df.rename(columns={df.columns[0]: "issue_time"})

    df["issue_time"] = pd.to_datetime(df["issue_time"], errors="coerce")
    df = df.dropna(subset=["issue_time"]).copy()

    true_cols = sort_horizon_columns([c for c in df.columns if c.endswith("_true")])
    pred_cols = [c.replace("_true", "_pred") for c in true_cols]

    missing_pred_cols = [c for c in pred_cols if c not in df.columns]
    if missing_pred_cols:
        raise ValueError(f"Missing prediction columns in {file_path}: {missing_pred_cols}")

    rows = []

    for _, r in df.iterrows():
        issue_time = r["issue_time"]
        forecast_date = (issue_time + pd.Timedelta(days=1)).date()

        y_true = pd.to_numeric(r[true_cols], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(r[pred_cols], errors="coerce").to_numpy(dtype=float)

        y_constant = np.full_like(y_true, y_true[0], dtype=float)

        valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_constant))

        y_true = y_true[valid]
        y_pred = y_pred[valid]
        y_constant = y_constant[valid]

        horizon_times = pd.date_range(
            start=issue_time + pd.Timedelta(hours=dt_hours),
            periods=len(y_true),
            freq=freq,
        )

        temp = pd.DataFrame({
            "target_col": target_col,
            "resolution": resolution,
            "model_name": model_name,
            "issue_time": issue_time,
            "forecast_date": forecast_date,
            "time": horizon_times,
            "actual_load_kW": y_true,
            "ml_predicted_load_kW": y_pred,
            "constant_load_kW": y_constant,
        })

        rows.append(temp)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def build_all_load_predictions_long(registry: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for _, row in registry.iterrows():
        frames.append(read_profile_prediction_file_long(row))

    if not frames:
        raise ValueError("No prediction files were loaded.")

    return pd.concat(frames, ignore_index=True)


def get_common_dates_per_resolution(
    prediction_long: pd.DataFrame,
    target_cols: list[str],
    resolutions: list[str],
) -> dict[str, set]:
    valid_dates = {}

    for resolution in resolutions:
        temp = prediction_long[prediction_long["resolution"] == resolution]

        load_count_by_date = (
            temp.groupby("forecast_date")["target_col"]
            .nunique()
        )

        common_dates = load_count_by_date[
            load_count_by_date == len(target_cols)
        ].index

        valid_dates[resolution] = set(common_dates)

    return valid_dates


def build_total_load_by_resolution(
    prediction_long: pd.DataFrame,
    target_cols: list[str],
    resolutions: list[str],
    *,
    use_only_complete_load_days: bool = True,
) -> pd.DataFrame:
    valid_dates = get_common_dates_per_resolution(
        prediction_long,
        target_cols=target_cols,
        resolutions=resolutions,
    )

    frames = []

    for resolution in resolutions:
        temp = prediction_long[
            prediction_long["resolution"].eq(resolution)
        ].copy()

        if use_only_complete_load_days:
            temp = temp[temp["forecast_date"].isin(valid_dates[resolution])].copy()

        grouped = (
            temp
            .groupby(["resolution", "forecast_date", "time"], as_index=False)
            .agg(
                total_actual_load_kW=("actual_load_kW", "sum"),
                total_ml_predicted_load_kW=("ml_predicted_load_kW", "sum"),
                total_constant_load_kW=("constant_load_kW", "sum"),
                n_loads=("target_col", "nunique"),
            )
        )

        grouped = grouped[grouped["n_loads"] == len(target_cols)].copy()
        frames.append(grouped)

    if not frames:
        raise ValueError("No total load data was created.")

    out = pd.concat(frames, ignore_index=True)
    out["resolution_order"] = out["resolution"].map(RESOLUTION_ORDER)

    return (
        out
        .sort_values(["resolution_order", "forecast_date", "time"])
        .drop(columns="resolution_order")
        .reset_index(drop=True)
    )


def calculate_daily_energy_error(
    total_load_by_resolution: pd.DataFrame,
    resolutions: list[str],
    *,
    dt_hours_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    dt_hours_map = dt_hours_map or DT_HOURS_MAP
    rows = []

    for resolution in resolutions:
        dt_hours = dt_hours_map[resolution]

        temp = total_load_by_resolution[
            total_load_by_resolution["resolution"].eq(resolution)
        ].copy()

        daily = (
            temp
            .groupby(["resolution", "forecast_date"], as_index=False)
            .agg(
                actual_energy_kWh=("total_actual_load_kW", lambda x: np.sum(x * dt_hours)),
                ml_predicted_energy_kWh=("total_ml_predicted_load_kW", lambda x: np.sum(x * dt_hours)),
                constant_energy_kWh=("total_constant_load_kW", lambda x: np.sum(x * dt_hours)),
            )
        )

        daily["ml_signed_energy_error_kWh"] = (
            daily["actual_energy_kWh"] - daily["ml_predicted_energy_kWh"]
        )

        daily["constant_signed_energy_error_kWh"] = (
            daily["actual_energy_kWh"] - daily["constant_energy_kWh"]
        )

        daily["ml_absolute_energy_error_kWh"] = daily["ml_signed_energy_error_kWh"].abs()
        daily["constant_absolute_energy_error_kWh"] = daily["constant_signed_energy_error_kWh"].abs()

        daily["ml_absolute_energy_error_%"] = (
            daily["ml_absolute_energy_error_kWh"] / daily["actual_energy_kWh"] * 100
        )

        daily["constant_absolute_energy_error_%"] = (
            daily["constant_absolute_energy_error_kWh"] / daily["actual_energy_kWh"] * 100
        )

        daily["ml_improvement_vs_constant_kWh"] = (
            daily["constant_absolute_energy_error_kWh"]
            - daily["ml_absolute_energy_error_kWh"]
        )

        daily["ml_improvement_vs_constant_%"] = np.where(
            daily["constant_absolute_energy_error_kWh"] != 0,
            daily["ml_improvement_vs_constant_kWh"]
            / daily["constant_absolute_energy_error_kWh"] * 100,
            np.nan,
        )

        rows.append(daily)

    out = pd.concat(rows, ignore_index=True)
    out["resolution_order"] = out["resolution"].map(RESOLUTION_ORDER)

    return (
        out
        .sort_values(["forecast_date", "resolution_order"])
        .drop(columns="resolution_order")
        .reset_index(drop=True)
    )


def summarize_energy_error(energy_error_df: pd.DataFrame, *, common_days: bool = False) -> pd.DataFrame:
    n_col = "n_common_days" if common_days else "n_days"

    summary = (
        energy_error_df
        .groupby("resolution", as_index=False)
        .agg(
            **{
                n_col: ("forecast_date", "nunique"),
                "mean_actual_energy_kWh": ("actual_energy_kWh", "mean"),
                "mean_ml_predicted_energy_kWh": ("ml_predicted_energy_kWh", "mean"),
                "mean_constant_energy_kWh": ("constant_energy_kWh", "mean"),
                "mean_ml_signed_energy_error_kWh": ("ml_signed_energy_error_kWh", "mean"),
                "mean_constant_signed_energy_error_kWh": ("constant_signed_energy_error_kWh", "mean"),
                "mean_ml_absolute_energy_error_kWh": ("ml_absolute_energy_error_kWh", "mean"),
                "mean_ml_absolute_energy_error_percent": ("ml_absolute_energy_error_%", "mean"),
                "mean_constant_absolute_energy_error_kWh": ("constant_absolute_energy_error_kWh", "mean"),
                "mean_constant_absolute_energy_error_percent": ("constant_absolute_energy_error_%", "mean"),
                "mean_ml_improvement_vs_constant_kWh": ("ml_improvement_vs_constant_kWh", "mean"),
                "mean_ml_improvement_vs_constant_percent": ("ml_improvement_vs_constant_%", "mean"),
            }
        )
    )

    summary["resolution_order"] = summary["resolution"].map(RESOLUTION_ORDER)

    return (
        summary
        .sort_values("resolution_order")
        .drop(columns="resolution_order")
        .reset_index(drop=True)
    )


def filter_common_dates_across_resolutions(
    energy_error_df: pd.DataFrame,
    required_resolutions: set[str],
) -> tuple[pd.DataFrame, list]:
    available = (
        energy_error_df
        .groupby("forecast_date")["resolution"]
        .apply(set)
    )

    common_dates = available[
        available.apply(lambda x: required_resolutions.issubset(x))
    ].index.tolist()

    out = energy_error_df[
        energy_error_df["forecast_date"].isin(common_dates)
        & energy_error_df["resolution"].isin(required_resolutions)
    ].copy()

    return out.reset_index(drop=True), common_dates


def add_residual_columns(total_load_df: pd.DataFrame) -> pd.DataFrame:
    out = total_load_df.copy()

    out["ml_residual_kW"] = (
        out["total_actual_load_kW"] - out["total_ml_predicted_load_kW"]
    )

    out["constant_residual_kW"] = (
        out["total_actual_load_kW"] - out["total_constant_load_kW"]
    )

    return out


def forecast_bias_variance_summary(total_load_df: pd.DataFrame) -> pd.DataFrame:
    df = add_residual_columns(total_load_df)

    rows = []

    for resolution, g in df.groupby("resolution"):
        actual = g["total_actual_load_kW"].to_numpy(dtype=float)
        pred = g["total_ml_predicted_load_kW"].to_numpy(dtype=float)
        residual = actual - pred

        mse = mean_squared_error(actual, pred)
        bias = float(np.mean(residual))
        residual_var = float(np.var(residual, ddof=0))

        rows.append({
            "resolution": resolution,
            "n_samples": len(g),
            "mean_actual_load_kW": float(np.mean(actual)),
            "actual_std_kW": float(np.std(actual, ddof=0)),
            "actual_cv": float(np.std(actual, ddof=0) / np.mean(actual)) if np.mean(actual) != 0 else np.nan,
            "forecast_bias_kW": bias,
            "abs_forecast_bias_kW": abs(bias),
            "residual_std_kW": float(np.std(residual, ddof=0)),
            "residual_variance_kW2": residual_var,
            "MAE_kW": mean_absolute_error(actual, pred),
            "RMSE_kW": np.sqrt(mse),
            "bias_share_of_MSE_%": (bias ** 2 / mse * 100) if mse != 0 else np.nan,
            "variance_share_of_MSE_%": (residual_var / mse * 100) if mse != 0 else np.nan,
        })

    out = pd.DataFrame(rows)
    out["resolution_order"] = out["resolution"].map(RESOLUTION_ORDER)

    return (
        out
        .sort_values("resolution_order")
        .drop(columns="resolution_order")
        .reset_index(drop=True)
    )


def diagnose_train_test_gap(metrics_df: pd.DataFrame) -> pd.DataFrame:
    required = {"resolution", "model_name", "train_MAE", "test_MAE"}
    missing = required - set(metrics_df.columns)

    if missing:
        raise ValueError(f"Missing columns for train-test diagnosis: {missing}")

    out = metrics_df.copy()

    out["MAE_gap_test_train"] = out["test_MAE"] - out["train_MAE"]

    out["MAE_gap_percent"] = np.where(
        out["train_MAE"] != 0,
        out["MAE_gap_test_train"] / out["train_MAE"] * 100,
        np.nan,
    )

    conditions = [
        (out["train_MAE"] > 0.8 * out["test_MAE"]),
        (out["MAE_gap_percent"] > 50),
    ]

    choices = [
        "Possible high bias / underfitting",
        "Possible high variance / overfitting",
    ]

    out["bias_variance_diagnosis"] = np.select(
        conditions,
        choices,
        default="Reasonable generalization",
    )

    return out


def add_bar_labels(ax, fmt: str = "{:.2f}") -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=9)


def plot_energy_error_comparison(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df.copy()
    plot_df["resolution_order"] = plot_df["resolution"].map(RESOLUTION_ORDER)
    plot_df = plot_df.sort_values("resolution_order").reset_index(drop=True)

    n_col = "n_common_days" if "n_common_days" in plot_df.columns else "n_days"
    n_days = plot_df[n_col].iloc[0]

    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.bar(
        x - width / 2,
        plot_df["mean_ml_absolute_energy_error_kWh"],
        width,
        label="ML forecast",
        color="tab:orange",
    )

    ax.bar(
        x + width / 2,
        plot_df["mean_constant_absolute_energy_error_kWh"],
        width,
        label="Constant baseline",
        color="tab:green",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["resolution"])
    ax.set_ylabel("Mean absolute energy error [kWh]")
    ax.set_xlabel("Forecast resolution")
    ax.set_title(f"Total Load Energy Error Comparison over Common Days (n = {n_days} days)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    add_bar_labels(ax)
    plt.tight_layout()
    plt.show()


def plot_energy_for_day(energy_error_df: pd.DataFrame, selected_day: str) -> None:
    day = pd.to_datetime(selected_day).date()

    plot_df = energy_error_df[
        energy_error_df["forecast_date"].eq(day)
    ].copy()

    if plot_df.empty:
        print(f"No energy data found for {selected_day}")
        return

    plot_df["resolution_order"] = plot_df["resolution"].map(RESOLUTION_ORDER)
    plot_df = plot_df.sort_values("resolution_order").reset_index(drop=True)

    x = np.arange(len(plot_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.bar(x - width, plot_df["actual_energy_kWh"], width, label="Actual energy", color="tab:blue")
    ax.bar(x, plot_df["ml_predicted_energy_kWh"], width, label="ML predicted energy", color="tab:orange")
    ax.bar(x + width, plot_df["constant_energy_kWh"], width, label="Constant baseline energy", color="tab:green")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["resolution"])
    ax.set_ylabel("Energy [kWh]")
    ax.set_xlabel("Forecast resolution")
    ax.set_title(f"Total Load Energy Comparison for {selected_day}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    add_bar_labels(ax)
    plt.tight_layout()
    plt.show()


def plot_residuals_for_day(
    total_load_df: pd.DataFrame,
    selected_day: str,
    *,
    include_actual: bool = False,
    resolutions: list[str] | None = None,
) -> None:
    resolutions = resolutions or ["15min", "30min", "60min"]
    day = pd.to_datetime(selected_day).date()

    df = add_residual_columns(total_load_df)
    df = df[df["forecast_date"].eq(day)].copy()

    for resolution in resolutions:
        temp = df[df["resolution"].eq(resolution)].copy()

        if temp.empty:
            print(f"No data found for {selected_day} and {resolution}")
            continue

        plt.figure(figsize=(14, 5))

        if include_actual:
            plt.plot(
                temp["time"],
                temp["total_actual_load_kW"],
                label="Actual total load",
                color="tab:blue",
                linewidth=2.5,
            )

        plt.plot(
            temp["time"],
            temp["ml_residual_kW"],
            label="ML residual: Actual - ML prediction",
            color="tab:orange",
            linewidth=2,
        )

        plt.plot(
            temp["time"],
            temp["constant_residual_kW"],
            label="Constant residual: Actual - Constant baseline",
            color="tab:green",
            linewidth=2,
            linestyle="--",
        )

        plt.axhline(0, color="black", linestyle=":", linewidth=1)

        title = "Total Load and Residual Comparison" if include_actual else "Total Load Residual Comparison"
        plt.title(f"{title} | {resolution} | {selected_day}")
        plt.xlabel("Time")
        plt.ylabel("Power / Residual [kW]" if include_actual else "Residual [kW]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_bias_variance_summary(bias_variance_df: pd.DataFrame) -> None:
    plot_df = bias_variance_df.copy()
    plot_df["resolution_order"] = plot_df["resolution"].map(RESOLUTION_ORDER)
    plot_df = plot_df.sort_values("resolution_order").reset_index(drop=True)

    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.bar(
        x - width / 2,
        plot_df["abs_forecast_bias_kW"],
        width,
        label="Absolute forecast bias",
    )

    ax.bar(
        x + width / 2,
        plot_df["residual_std_kW"],
        width,
        label="Residual standard deviation",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["resolution"])
    ax.set_ylabel("Error component [kW]")
    ax.set_xlabel("Forecast resolution")
    ax.set_title("Forecast Bias and Residual Variance Check")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    add_bar_labels(ax)
    plt.tight_layout()
    plt.show()