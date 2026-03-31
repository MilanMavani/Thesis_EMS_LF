import pandas as pd
from datetime import datetime
import json

from src.config import RANDOM_STATE
from src.splitter import time_based_split, extract_xy
from src.metrics import evaluate_regression
from src.model_registry import get_nan_friendly_models, get_dense_models
from src.tracker import (
    generate_run_id,
    save_predictions,
    save_model,
    save_feature_importance,
    append_experiment_log,
    save_model_params,
    save_forecast_diagnostic_plot,
)


def infer_dataframe_frequency(index: pd.DatetimeIndex) -> str:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")

    if len(index) < 3:
        return "unknown"

    inferred = pd.infer_freq(index)

    if inferred is not None:
        inferred = inferred.lower()
        freq_map = {
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "h": "1h",
            "60min": "1h",
        }
        return freq_map.get(inferred, inferred)

    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return "unknown"

    median_diff = diffs.median()

    if median_diff == pd.Timedelta(minutes=5):
        return "5min"
    elif median_diff == pd.Timedelta(minutes=15):
        return "15min"
    elif median_diff == pd.Timedelta(minutes=30):
        return "30min"
    elif median_diff == pd.Timedelta(hours=1):
        return "1h"
    else:
        return str(median_diff)


def run_training_experiment(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    dataset_name: str,
    frequency: str | None = None,
    feature_set_name: str = "default_features",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    selected_models: list[str] | None = None,
    data_mode: str = "auto",
):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    df = df.sort_index().copy()

    detected_frequency = infer_dataframe_frequency(df.index)
    if frequency is None:
        frequency = detected_frequency

    df = df.dropna(subset=[target_col])

    if data_mode == "auto":
        has_x_nans = df[feature_cols].isna().any().any()
        data_mode = "nan_friendly" if has_x_nans else "dense"

    train_df, val_df, test_df = time_based_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    if data_mode == "dense":
        train_df = train_df.dropna(subset=feature_cols)
        val_df = val_df.dropna(subset=feature_cols)
        test_df = test_df.dropna(subset=feature_cols)
        models = get_dense_models(random_state=RANDOM_STATE)
    elif data_mode == "nan_friendly":
        models = get_nan_friendly_models(random_state=RANDOM_STATE)
    else:
        raise ValueError("data_mode must be one of: 'auto', 'dense', 'nan_friendly'")

    X_train, y_train = extract_xy(train_df, target_col, feature_cols)
    X_val, y_val = extract_xy(val_df, target_col, feature_cols)
    X_test, y_test = extract_xy(test_df, target_col, feature_cols)

    if selected_models is not None:
        models = {name: model for name, model in models.items() if name in selected_models}

    if not models:
        raise ValueError("No models selected after filtering. Check selected_models and data_mode.")

    results = []

    for model_name, model in models.items():
        print(f"Training {model_name} for target: {target_col} [{data_mode}]")

        run_id = generate_run_id()
        start_time = datetime.now()

        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_metrics = evaluate_regression(y_val, val_pred)
        test_metrics = evaluate_regression(y_test, test_pred)

        pred_path = save_predictions(
            run_id=run_id,
            model_name=model_name,
            target_col=target_col,
            y_true=y_test.values,
            y_pred=test_pred,
            index=X_test.index,
        )
        pred_df = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": test_pred,
        }, index=X_test.index)

        plot_path = save_forecast_diagnostic_plot(
            run_id=run_id,
            model_name=model_name,
            target_col=target_col,
            predictions_df=pred_df,
            dataset_name=dataset_name,
        )
        model_path = save_model(
            run_id=run_id,
            model_name=model_name,
            target_col=target_col,
            model=model,
        )

        fi_path = save_feature_importance(
            run_id=run_id,
            model_name=model_name,
            target_col=target_col,
            model=model,
            feature_cols=feature_cols,
        )

        params_path = save_model_params(
            run_id=run_id,
            model_name=model_name,
            target_col=target_col,
            model=model,
        )

        record = {
            "run_id": run_id,
            "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "target": target_col,
            "dataset_name": dataset_name,
            "frequency": frequency,
            "detected_frequency": detected_frequency,
            "feature_set_name": feature_set_name,
            "data_mode": data_mode,
            "model_name": model_name,
            "model_params": json.dumps(model.get_params(), default=str),
            "n_features": len(feature_cols),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "train_start": str(train_df.index.min()),
            "train_end": str(train_df.index.max()),
            "val_start": str(val_df.index.min()),
            "val_end": str(val_df.index.max()),
            "test_start": str(test_df.index.min()),
            "test_end": str(test_df.index.max()),
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
            "model_path": str(model_path),
            "prediction_path": str(pred_path),
            "diagnostic_plot_path": str(plot_path),
            "feature_importance_path": str(fi_path) if fi_path else None,
            "params_path": str(params_path),
        }

        append_experiment_log(record)
        results.append(record)

    return pd.DataFrame(results)