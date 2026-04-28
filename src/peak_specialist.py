# src/peak_specialist.py

from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.config import RANDOM_STATE
from src.splitter import time_based_split
from src.profile_dataset_builder import build_day_ahead_profile_dataset
from src.profile_metrics import evaluate_profile_global, evaluate_profile_by_horizon


def get_horizon_steps_for_time_window(
    *,
    start_hour: int,
    end_hour: int,
    samples_per_hour: int = 4,
) -> list[int]:
    """
    Convert a clock-time window into 1-based horizon steps.

    Assumption:
    y_tplus_001 corresponds to 00:00,
    y_tplus_002 corresponds to 00:15,
    ...
    for a day-ahead forecast issued at 23:45.
    """
    start_step = start_hour * samples_per_hour + 1
    end_step = end_hour * samples_per_hour

    return list(range(start_step, end_step + 1))


def select_horizon_columns(Y: pd.DataFrame, horizon_steps: list[int]) -> pd.DataFrame:
    cols = [f"y_tplus_{step:03d}" for step in horizon_steps]

    missing = [c for c in cols if c not in Y.columns]
    if missing:
        raise ValueError(f"Missing horizon columns: {missing}")

    return Y[cols].copy()


def get_peak_xgboost_model(random_state: int = RANDOM_STATE):
    """
    Regularized XGBoost model for small-data peak-window forecasting.
    """
    base_model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )

    return MultiOutputRegressor(base_model)


def run_peak_specialist_experiment(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    dataset_name: str,
    output_dir: str | Path,
    peak_start_hour: int = 10,
    peak_end_hour: int = 18,
    horizon_steps: int = 96,
    issue_hour: int = 23,
    issue_minute: int = 45,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    drop_feature_nan: bool = False,
):
    """
    Train and evaluate a peak-window-only specialist model.

    The model uses the same X features as the full-day model,
    but the target Y contains only peak-window horizon columns.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    df = df.sort_index().copy()

    X, Y_full = build_day_ahead_profile_dataset(
        df,
        target_col=target_col,
        feature_cols=feature_cols,
        horizon_steps=horizon_steps,
        issue_hour=issue_hour,
        issue_minute=issue_minute,
        drop_feature_nan=drop_feature_nan,
        drop_target_nan=True,
    )

    peak_steps = get_horizon_steps_for_time_window(
        start_hour=peak_start_hour,
        end_hour=peak_end_hour,
        samples_per_hour=4,
    )

    Y_peak = select_horizon_columns(Y_full, peak_steps)

    dataset = pd.concat([X, Y_peak], axis=1)

    train_df, val_df, test_df = time_based_split(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    y_cols = list(Y_peak.columns)

    X_train = train_df[feature_cols].copy()
    Y_train = train_df[y_cols].copy()

    X_val = val_df[feature_cols].copy()
    Y_val = val_df[y_cols].copy()

    X_test = test_df[feature_cols].copy()
    Y_test = test_df[y_cols].copy()

    model = get_peak_xgboost_model(random_state=RANDOM_STATE)

    start_time = datetime.now()
    model.fit(X_train, Y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = evaluate_profile_global(Y_val, val_pred)
    test_metrics = evaluate_profile_global(Y_test, test_pred)

    horizon_metrics = evaluate_profile_by_horizon(Y_test, test_pred)

    pred_df = pd.DataFrame(
        test_pred,
        index=Y_test.index,
        columns=[f"{c}_pred" for c in y_cols],
    )

    true_df = Y_test.copy()
    true_df.columns = [f"{c}_true" for c in y_cols]

    out_pred = pd.concat([true_df, pred_df], axis=1)

    pred_path = predictions_dir / f"peak_specialist_{target_col}_{peak_start_hour}_{peak_end_hour}_predictions.csv"
    horizon_path = metrics_dir / f"peak_specialist_{target_col}_{peak_start_hour}_{peak_end_hour}_horizon_metrics.csv"
    summary_path = metrics_dir / f"peak_specialist_{target_col}_{peak_start_hour}_{peak_end_hour}_summary.csv"

    out_pred.to_csv(pred_path, index=True)
    horizon_metrics.to_csv(horizon_path, index=False)

    summary = {
        "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_type": "peak_specialist_profile_forecasting",
        "dataset_name": dataset_name,
        "target": target_col,
        "model_name": "peak_specialist_xgboost",
        "peak_start_hour": peak_start_hour,
        "peak_end_hour": peak_end_hour,
        "peak_horizon_start": min(peak_steps),
        "peak_horizon_end": max(peak_steps),
        "n_peak_horizons": len(peak_steps),
        "n_features": len(feature_cols),
        "n_train_days": len(X_train),
        "n_val_days": len(X_val),
        "n_test_days": len(X_test),
        "train_start": str(X_train.index.min()),
        "train_end": str(X_train.index.max()),
        "val_start": str(X_val.index.min()),
        "val_end": str(X_val.index.max()),
        "test_start": str(X_test.index.min()),
        "test_end": str(X_test.index.max()),
        "val_MAE": val_metrics["MAE"],
        "val_RMSE": val_metrics["RMSE"],
        "val_MAPE": val_metrics["MAPE"],
        "val_sMAPE": val_metrics["sMAPE"],
        "val_R2": val_metrics["R2"],
        "test_MAE": test_metrics["MAE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_MAPE": test_metrics["MAPE"],
        "test_sMAPE": test_metrics["sMAPE"],
        "test_R2": test_metrics["R2"],
        "prediction_path": str(pred_path),
        "horizon_metrics_path": str(horizon_path),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_path, index=False)

    return summary_df, horizon_metrics, out_pred, model