from __future__ import annotations

import json
import uuid
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from src.config import (
    PROFILE_EXPERIMENT_LOG_FILE,
    PROFILE_PREDICTIONS_DIR,
    PROFILE_HORIZON_METRICS_DIR,
    PROFILE_MODELS_DIR,
    PROFILE_LOGS_DIR,
    PROFILE_FIGURES_DIR,
)


def generate_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    return f"run_{timestamp}_{short_id}"


def save_profile_predictions(
    run_id: str,
    model_name: str,
    target_col: str,
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
) -> Path:
    pred_df = pd.DataFrame(y_pred, index=y_true.index, columns=y_true.columns)

    true_df = y_true.copy()
    true_df.columns = [f"{c}_true" for c in true_df.columns]
    pred_df.columns = [f"{c}_pred" for c in pred_df.columns]

    out = pd.concat([true_df, pred_df], axis=1)

    file_path = PROFILE_PREDICTIONS_DIR / f"{run_id}_{model_name}_{target_col}_profile_predictions.csv"
    out.to_csv(file_path, index=True)
    return file_path


def save_profile_horizon_metrics(
    run_id: str,
    model_name: str,
    target_col: str,
    horizon_df: pd.DataFrame,
) -> Path:
    file_path = PROFILE_HORIZON_METRICS_DIR / f"{run_id}_{model_name}_{target_col}_horizon_metrics.csv"
    horizon_df.to_csv(file_path, index=False)
    return file_path


def save_profile_model(
    run_id: str,
    model_name: str,
    target_col: str,
    model,
) -> Path:
    file_path = PROFILE_MODELS_DIR / f"{run_id}_{model_name}_{target_col}.pkl"
    joblib.dump(model, file_path)
    return file_path


def save_profile_model_params(
    run_id: str,
    model_name: str,
    target_col: str,
    model,
) -> Path:
    file_path = PROFILE_LOGS_DIR / f"{run_id}_{model_name}_{target_col}_params.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(model.get_params(), f, indent=2, default=str)
    return file_path


def save_profile_plot(
    run_id: str,
    model_name: str,
    target_col: str,
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    dataset_name: str | None = None,
    sample_day_index: int = 0,
) -> Path:
    pred_df = pd.DataFrame(y_pred, index=y_true.index, columns=y_true.columns)

    if len(y_true) == 0:
        raise ValueError("No test rows available for plotting.")

    sample_day_index = min(sample_day_index, len(y_true) - 1)

    x = np.arange(1, y_true.shape[1] + 1)
    true_row = y_true.iloc[sample_day_index].astype(float).values
    pred_row = pred_df.iloc[sample_day_index].astype(float).values
    residuals = true_row - pred_row
    issue_time = y_true.index[sample_day_index]

    fig = plt.figure(figsize=(14, 12))

    # 1. Actual vs Predicted
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(x, true_row, label="Actual")
    ax1.plot(x, pred_row, label="Predicted")
    ax1.set_title(
        f"{model_name} - {target_col} - Day-ahead profile\n"
        f"Issue time: {issue_time} [{dataset_name}]"
    )
    ax1.set_xlabel("Horizon step (15-min ahead)")
    ax1.set_ylabel(target_col)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Residuals
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(x, residuals)
    ax2.axhline(0, linestyle="--")
    ax2.set_title("Residuals by Horizon Step")
    ax2.set_xlabel("Horizon step (15-min ahead)")
    ax2.set_ylabel("Residual")
    ax2.grid(True, alpha=0.3)

    # 3. Scatter plot
    ax3 = plt.subplot(3, 1, 3)
    ax3.scatter(true_row, pred_row, alpha=0.6)
    min_val = min(np.nanmin(true_row), np.nanmin(pred_row))
    max_val = max(np.nanmax(true_row), np.nanmax(pred_row))
    ax3.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax3.set_title("Actual vs Predicted Scatter")
    ax3.set_xlabel("Actual")
    ax3.set_ylabel("Predicted")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    file_path = PROFILE_FIGURES_DIR / f"{run_id}_{model_name}_{target_col}_profile_plot.png"
    plt.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return file_path


def append_profile_experiment_log(record: dict) -> None:
    df_record = pd.DataFrame([record])

    if PROFILE_EXPERIMENT_LOG_FILE.exists():
        df_existing = pd.read_csv(PROFILE_EXPERIMENT_LOG_FILE)
        df_all = pd.concat([df_existing, df_record], ignore_index=True)
    else:
        df_all = df_record

    df_all.to_csv(PROFILE_EXPERIMENT_LOG_FILE, index=False)