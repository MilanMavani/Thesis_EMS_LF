from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def evaluate_profile_global(
    y_true: pd.DataFrame | np.ndarray,
    y_pred: pd.DataFrame | np.ndarray,
) -> dict[str, float]:
    """
    Evaluate profile forecast globally by flattening all horizons.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_flat = y_true.ravel()
    pred_flat = y_pred.ravel()

    rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))

    return {
        "MAE": mean_absolute_error(true_flat, pred_flat),
        "RMSE": rmse,
        "MAPE": mape(true_flat, pred_flat),
        "sMAPE": smape(true_flat, pred_flat),
        "R2": r2_score(true_flat, pred_flat),
    }


def evaluate_profile_by_horizon(
    y_true: pd.DataFrame | np.ndarray,
    y_pred: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate error separately for each horizon step.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_horizons = y_true.shape[1]
    rows = []

    for h in range(n_horizons):
        yt = y_true[:, h]
        yp = y_pred[:, h]

        rows.append({
            "horizon_step": h + 1,
            "MAE": mean_absolute_error(yt, yp),
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "MAPE": mape(yt, yp),
            "sMAPE": smape(yt, yp),
            "R2": r2_score(yt, yp),
        })

    return pd.DataFrame(rows)