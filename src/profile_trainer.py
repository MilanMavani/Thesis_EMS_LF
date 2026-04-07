from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE
from src.splitter import time_based_split
from src.profile_dataset_builder import build_day_ahead_profile_dataset
from src.profile_model_registry import get_profile_models
from src.profile_metrics import evaluate_profile_global, evaluate_profile_by_horizon


def _ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_profile_predictions(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    save_path: Path,
) -> None:
 
    pred_df = pd.DataFrame(
        y_pred,
        index=y_true.index,
        columns=y_true.columns,
    )

    true_df = y_true.copy()
    true_df.columns = [f"{c}_true" for c in true_df.columns]
    pred_df.columns = [f"{c}_pred" for c in pred_df.columns]

    out = pd.concat([true_df, pred_df], axis=1)
    out.to_csv(save_path, index=True)


def run_profile_training_experiment(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    dataset_name: str,
    output_dir: str | Path,
    horizon_steps: int = 96,
    issue_hour: int = 23,
    issue_minute: int = 45,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    selected_models: list[str] | None = None,
    drop_feature_nan: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    df = df.sort_index().copy()

    X, Y = build_day_ahead_profile_dataset(
        df,
        target_col=target_col,
        feature_cols=feature_cols,
        horizon_steps=horizon_steps,
        issue_hour=issue_hour,
        issue_minute=issue_minute,
        drop_feature_nan=drop_feature_nan,
        drop_target_nan=True,
    )

    dataset = pd.concat([X, Y], axis=1)

    train_df, val_df, test_df = time_based_split(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    y_cols = list(Y.columns)

    X_train = train_df[feature_cols].copy()
    Y_train = train_df[y_cols].copy()

    X_val = val_df[feature_cols].copy()
    Y_val = val_df[y_cols].copy()

    X_test = test_df[feature_cols].copy()
    Y_test = test_df[y_cols].copy()

    models = get_profile_models(random_state=RANDOM_STATE)

    if selected_models is not None:
        models = {k: v for k, v in models.items() if k in selected_models}

    if not models:
        raise ValueError("No models selected after filtering.")

    output_dir = _ensure_output_dir(output_dir)
    predictions_dir = _ensure_output_dir(output_dir / "predictions")
    horizon_dir = _ensure_output_dir(output_dir / "horizon_metrics")

    results = []
    horizon_results = {}

    for model_name, model in models.items():
        print(f"Training {model_name} for day-ahead profile forecasting: {target_col}")

        start_time = datetime.now()

        model.fit(X_train, Y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_global = evaluate_profile_global(Y_val, val_pred)
        test_global = evaluate_profile_global(Y_test, test_pred)

        horizon_df = evaluate_profile_by_horizon(Y_test, test_pred)
        horizon_results[model_name] = horizon_df

        pred_path = predictions_dir / f"{model_name}_{target_col}_profile_predictions.csv"
        _save_profile_predictions(Y_test, test_pred, pred_path)

        horizon_path = horizon_dir / f"{model_name}_{target_col}_horizon_metrics.csv"
        horizon_df.to_csv(horizon_path, index=False)

        record = {
            "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_type": "day_ahead_profile_forecasting",
            "dataset_name": dataset_name,
            "target": target_col,
            "model_name": model_name,
            "model_params": json.dumps(model.get_params(), default=str),
            "n_features": len(feature_cols),
            "n_horizons": horizon_steps,
            "n_train_days": len(X_train),
            "n_val_days": len(X_val),
            "n_test_days": len(X_test),
            "train_start": str(X_train.index.min()),
            "train_end": str(X_train.index.max()),
            "val_start": str(X_val.index.min()),
            "val_end": str(X_val.index.max()),
            "test_start": str(X_test.index.min()),
            "test_end": str(X_test.index.max()),
            "val_MAE": val_global["MAE"],
            "val_RMSE": val_global["RMSE"],
            "val_MAPE": val_global["MAPE"],
            "val_sMAPE": val_global["sMAPE"],
            "val_R2": val_global["R2"],
            "test_MAE": test_global["MAE"],
            "test_RMSE": test_global["RMSE"],
            "test_MAPE": test_global["MAPE"],
            "test_sMAPE": test_global["sMAPE"],
            "test_R2": test_global["R2"],
            "prediction_path": str(pred_path),
            "horizon_metrics_path": str(horizon_path),
        }

        results.append(record)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / f"profile_results_{target_col}.csv", index=False)

    return results_df, horizon_results