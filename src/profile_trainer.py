from __future__ import annotations

from datetime import datetime
import json
import pandas as pd

from src.config import RANDOM_STATE
from src.splitter import time_based_split
from src.profile_dataset_builder import build_day_ahead_profile_dataset
from src.profile_model_registry import (
    get_profile_dense_models,
    get_profile_nan_friendly_models,
)
from src.profile_metrics import evaluate_profile_global, evaluate_profile_by_horizon
from src.profile_tracker import (
    generate_run_id,
    save_profile_predictions,
    save_profile_horizon_metrics,
    save_profile_model,
    save_profile_model_params,
    save_profile_plot,
    append_profile_experiment_log,
)


def run_profile_training_experiment(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    dataset_name: str,
    feature_set_name: str = "default_features",
    horizon_steps: int = 96,
    issue_hour: int = 23,
    issue_minute: int = 45,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    selected_models: list[str] | None = None,
    drop_feature_nan: bool = False,
    data_mode: str = "auto",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Train and evaluate multi-output day-ahead profile forecasting models.

    data_mode:
        - "auto"  -> detect if feature NaNs exist, then switch to "NaNs" or "dense"
        - "dense" -> use dense models, drop rows with feature NaNs
        - "NaNs"  -> use NaN-friendly models, keep feature NaNs
    """
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

    if data_mode == "auto":
        has_x_nans = (
            X_train.isna().any().any()
            or X_val.isna().any().any()
            or X_test.isna().any().any()
        )
        data_mode = "NaNs" if has_x_nans else "dense"

    if data_mode == "dense":
        train_mask = X_train.notna().all(axis=1)
        val_mask = X_val.notna().all(axis=1)
        test_mask = X_test.notna().all(axis=1)

        X_train = X_train.loc[train_mask]
        Y_train = Y_train.loc[train_mask]

        X_val = X_val.loc[val_mask]
        Y_val = Y_val.loc[val_mask]

        X_test = X_test.loc[test_mask]
        Y_test = Y_test.loc[test_mask]

        models = get_profile_dense_models(random_state=RANDOM_STATE)

    elif data_mode == "NaNs":
        models = get_profile_nan_friendly_models(random_state=RANDOM_STATE)

    else:
        raise ValueError("data_mode must be one of: 'auto', 'dense', 'NaNs'")

    if selected_models is not None:
        models = {k: v for k, v in models.items() if k in selected_models}

    if not models:
        raise ValueError("No models selected after filtering.")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            "One of train/val/test sets is empty after preprocessing. "
            "Check split ratios, issue time filtering, and NaN handling."
        )

    results = []
    horizon_results: dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        print(f"Training {model_name} for day-ahead profile forecasting: {target_col} [{data_mode}]")

        run_id = generate_run_id()
        start_time = datetime.now()

        model.fit(X_train, Y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_global = evaluate_profile_global(Y_val, val_pred)
        test_global = evaluate_profile_global(Y_test, test_pred)

        horizon_df = evaluate_profile_by_horizon(Y_test, test_pred)
        horizon_results[model_name] = horizon_df

        pred_path = save_profile_predictions(
            run_id=run_id,
            model_name=model_name,
            target_col=target_col,
            y_true=Y_test,
            y_pred=test_pred,
        )

        horizon_path = save_profile_horizon_metrics(
            run_id=run_id,
            model_name=model_name,
            target_col=target_col,
            horizon_df=horizon_df,
        )

        model_path = save_profile_model(
            run_id=run_id,
            model_name=model_name,
            target_col=target_col,
            model=model,
        )

        params_path = save_profile_model_params(
            run_id=run_id,
            model_name=model_name,
            target_col=target_col,
            model=model,
        )

        plot_path = save_profile_plot(
            run_id=run_id,
            model_name=model_name,
            target_col=target_col,
            y_true=Y_test,
            y_pred=test_pred,
            dataset_name=dataset_name,
            feature_set_name=feature_set_name,
            sample_day_index=0,
        )

        record = {
            "run_id": run_id,
            "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_type": "day_ahead_profile_forecasting",
            "dataset_name": dataset_name,
            "feature_set_name": feature_set_name,
            "data_mode": data_mode,
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
            "model_path": str(model_path),
            "prediction_path": str(pred_path),
            "horizon_metrics_path": str(horizon_path),
            "plot_path": str(plot_path),
            "params_path": str(params_path),
        }

        append_profile_experiment_log(record)
        results.append(record)

    results_df = pd.DataFrame(results)
    return results_df, horizon_results