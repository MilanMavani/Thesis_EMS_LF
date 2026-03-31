import json
import uuid
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import (
    EXPERIMENT_LOG_FILE,
    MODELS_DIR,
    PREDICTIONS_DIR,
    FEATURE_IMPORTANCE_DIR,
    LOGS_DIR,
    FORECAST_FIGURES_DIR,
)
from src.column_metadata import get_plot_label, get_unit


def generate_run_id():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    return f"run_{timestamp}_{short_id}"


def save_predictions(run_id, model_name, target_col, y_true, y_pred, index):
    df_pred = pd.DataFrame({
        "Time": index,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    file_path = PREDICTIONS_DIR / f"{run_id}_{model_name}_{target_col}_predictions.csv"
    df_pred.to_csv(file_path, index=False)
    return file_path


def save_model(run_id, model_name, target_col, model):
    file_path = MODELS_DIR / f"{run_id}_{model_name}_{target_col}.pkl"
    joblib.dump(model, file_path)
    return file_path


def save_feature_importance(run_id, model_name, target_col, model, feature_cols):
    file_path = FEATURE_IMPORTANCE_DIR / f"{run_id}_{model_name}_{target_col}_feature_importance.csv"

    importance_df = None

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

    elif hasattr(model, "coef_"):
        coef = model.coef_
        if len(getattr(coef, "shape", [])) > 1:
            coef = coef.ravel()
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": coef,
        }).sort_values("importance", ascending=False)

    if importance_df is not None:
        importance_df.to_csv(file_path, index=False)
        return file_path

    return None


def save_model_params(run_id, model_name, target_col, model):
    file_path = LOGS_DIR / f"{run_id}_{model_name}_{target_col}_params.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(model.get_params(), f, indent=2, default=str)
    return file_path


def save_forecast_diagnostic_plot(
    run_id,
    model_name,
    target_col,
    predictions_df,
    dataset_name=None,
    *,
    zoom_start=None,
    zoom_end=None,
):
    df_plot = predictions_df.copy()

    if "Time" in df_plot.columns:
        df_plot["Time"] = pd.to_datetime(df_plot["Time"], errors="coerce")
        df_plot = df_plot.dropna(subset=["Time"]).set_index("Time")

    if not isinstance(df_plot.index, pd.DatetimeIndex):
        raise ValueError("predictions_df must have a DatetimeIndex or a 'Time' column.")

    required_cols = {"y_true", "y_pred"}
    missing_cols = required_cols - set(df_plot.columns)
    if missing_cols:
        raise ValueError(f"predictions_df is missing required columns: {missing_cols}")

    df_plot = df_plot.sort_index().copy()
    df_plot["residual"] = df_plot["y_true"] - df_plot["y_pred"]
    df_plot["abs_error"] = df_plot["residual"].abs()
    df_plot["ape_percent"] = np.where(
        df_plot["y_true"] != 0,
        df_plot["abs_error"] / df_plot["y_true"] * 100,
        np.nan,
    )

    plot_label = get_plot_label(target_col)
    unit = get_unit(target_col)
    dataset_text = f" [{dataset_name}]" if dataset_name else ""
    safe_dataset_name = str(dataset_name).replace(" ", "_") if dataset_name else "unknown_dataset"

    fig = plt.figure(figsize=(16, 16))

    # 1. Actual vs predicted
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(df_plot.index, df_plot["y_true"], label="Actual")
    ax1.plot(df_plot.index, df_plot["y_pred"], label="Predicted")
    ax1.set_title(f"{model_name} - {target_col}{dataset_text}")
    ax1.set_ylabel(plot_label)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals over time
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(df_plot.index, df_plot["residual"])
    ax2.axhline(0, linestyle="--")
    ax2.set_title(f"{model_name} - {target_col}{dataset_text} - Residuals Over Time")
    ax2.set_ylabel(f"Residual [{unit}]" if unit else "Residual")
    ax2.grid(True, alpha=0.3)

    # 3. Zoomed window
    ax3 = plt.subplot(4, 1, 3)
    df_zoom = df_plot.copy()
    if zoom_start is not None:
        df_zoom = df_zoom[df_zoom.index >= pd.to_datetime(zoom_start)]
    if zoom_end is not None:
        df_zoom = df_zoom[df_zoom.index <= pd.to_datetime(zoom_end)]

    ax3.plot(df_zoom.index, df_zoom["y_true"], label="Actual")
    ax3.plot(df_zoom.index, df_zoom["y_pred"], label="Predicted")
    ax3.set_title(f"{model_name} - {target_col}{dataset_text} - Zoomed Window")
    ax3.set_ylabel(plot_label)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Scatter
    ax4 = plt.subplot(4, 1, 4)
    ax4.scatter(df_plot["y_true"], df_plot["y_pred"], alpha=0.5)
    min_val = min(df_plot["y_true"].min(), df_plot["y_pred"].min())
    max_val = max(df_plot["y_true"].max(), df_plot["y_pred"].max())
    ax4.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax4.set_title(f"{model_name} - {target_col}{dataset_text} - Actual vs Predicted Scatter")
    ax4.set_xlabel(f"Actual [{unit}]" if unit else "Actual")
    ax4.set_ylabel(f"Predicted [{unit}]" if unit else "Predicted")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    file_path = (
        FORECAST_FIGURES_DIR
        / f"{run_id}_{safe_dataset_name}_{model_name}_{target_col}_diagnostics.png"
    )
    fig.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return file_path


def append_experiment_log(record: dict):
    df_record = pd.DataFrame([record])

    if EXPERIMENT_LOG_FILE.exists():
        df_existing = pd.read_csv(EXPERIMENT_LOG_FILE)
        df_all = pd.concat([df_existing, df_record], ignore_index=True)
    else:
        df_all = df_record

    df_all.to_csv(EXPERIMENT_LOG_FILE, index=False)