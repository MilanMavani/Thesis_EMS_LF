from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.metrics import evaluate_regression
from src.tracker import (
    generate_run_id,
    save_predictions,
    append_experiment_log,
    save_forecast_diagnostic_plot,
)


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Scaling
# =========================================================

class StandardScaler1D:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_


# =========================================================
# Sequence builder
# =========================================================

def build_gru_sequences(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    lookback: int = 96,
    horizon: int = 1,
    dropna: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build supervised sequences for GRU.

    X shape: (n_samples, lookback, n_features)
    y shape: (n_samples,)
    times: timestamp of prediction target
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    if target_col not in df.columns:
        raise ValueError(f"Missing target_col: {target_col}")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    data = df.sort_index().copy()

    X_list = []
    y_list = []
    t_list = []

    values_x = data[feature_cols].values
    values_y = data[target_col].values
    times = data.index

    max_i = len(data) - horizon + 1

    for end_idx in range(lookback, max_i):
        x_seq = values_x[end_idx - lookback:end_idx]
        y_val = values_y[end_idx + horizon - 1]
        y_time = times[end_idx + horizon - 1]

        if dropna and (np.isnan(x_seq).any() or np.isnan(y_val)):
            continue

        X_list.append(x_seq)
        y_list.append(y_val)
        t_list.append(y_time)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    t = pd.DatetimeIndex(t_list)

    return X, y, t


# =========================================================
# Chronological split for sequences
# =========================================================

def split_sequences_timewise(
    X: np.ndarray,
    y: np.ndarray,
    times: pd.DatetimeIndex,
    *,
    train_ratio: float = 0.80,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
):
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (
        X[:train_end], y[:train_end], times[:train_end],
        X[train_end:val_end], y[train_end:val_end], times[train_end:val_end],
        X[val_end:], y[val_end:], times[val_end:],
    )


# =========================================================
# Dataset
# =========================================================

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# Model
# =========================================================

class GRURegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        y = self.head(last_hidden).squeeze(-1)
        return y


# =========================================================
# Training helpers
# =========================================================

@dataclass
class GRUConfig:
    lookback: int = 96
    horizon: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    batch_size: int = 256
    learning_rate: float = 0.01
    max_epochs: int = 50
    patience: int = 8
    train_ratio: float = 0.80
    val_ratio: float = 0.15
    test_ratio: float = 0.05
    seed: int = 42


def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())

    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    return y_true, y_pred


def train_one_model(
    model,
    train_loader,
    val_loader,
    *,
    device,
    learning_rate: float,
    max_epochs: int,
    patience: int,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return pd.DataFrame(history)


# =========================================================
# Main GRU experiment
# =========================================================

def run_gru_training_experiment(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    dataset_name: str,
    feature_set_name: str = "gru_features_v1",
    config: GRUConfig | None = None,
):
    if config is None:
        config = GRUConfig()

    set_seed(config.seed)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    df = df.sort_index().copy()
    df = df.dropna(subset=[target_col])

    X, y, times = build_gru_sequences(
        df,
        target_col=target_col,
        feature_cols=feature_cols,
        lookback=config.lookback,
        horizon=config.horizon,
        dropna=True,
    )

    (
        X_train, y_train, t_train,
        X_val, y_val, t_val,
        X_test, y_test, t_test,
    ) = split_sequences_timewise(
        X, y, times,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    # scale features using train set only
    n_features = X_train.shape[-1]
    x_scaler = StandardScaler1D().fit(X_train.reshape(-1, n_features))

    X_train_scaled = x_scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_scaled = x_scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = x_scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    # scale target using train set only
    y_scaler = StandardScaler1D().fit(y_train.reshape(-1, 1))
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    train_ds = SequenceDataset(X_train_scaled, y_train_scaled)
    val_ds = SequenceDataset(X_val_scaled, y_val_scaled)
    test_ds = SequenceDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRURegressor(
        input_size=len(feature_cols),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    start_time = datetime.now()
    run_id = generate_run_id()
    model_name = "gru"

    history_df = train_one_model(
        model,
        train_loader,
        val_loader,
        device=device,
        learning_rate=config.learning_rate,
        max_epochs=config.max_epochs,
        patience=config.patience,
    )

    y_val_true_scaled, y_val_pred_scaled = evaluate_model(model, val_loader, device)
    y_test_true_scaled, y_test_pred_scaled = evaluate_model(model, test_loader, device)

    y_val_true = y_scaler.inverse_transform(y_val_true_scaled.reshape(-1, 1)).reshape(-1)
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).reshape(-1)

    y_test_true = y_scaler.inverse_transform(y_test_true_scaled.reshape(-1, 1)).reshape(-1)
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).reshape(-1)

    val_metrics = evaluate_regression(y_val_true, y_val_pred)
    test_metrics = evaluate_regression(y_test_true, y_test_pred)

    pred_path = save_predictions(
        run_id=run_id,
        model_name=model_name,
        target_col=target_col,
        y_true=y_test_true,
        y_pred=y_test_pred,
        index=t_test,
    )

    pred_df = pd.DataFrame(
        {"y_true": y_test_true, "y_pred": y_test_pred},
        index=t_test,
    )

    plot_path = save_forecast_diagnostic_plot(
        run_id=run_id,
        model_name=model_name,
        target_col=target_col,
        predictions_df=pred_df,
        dataset_name=dataset_name,
    )

    record = {
        "run_id": run_id,
        "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": target_col,
        "dataset_name": dataset_name,
        "feature_set_name": feature_set_name,
        "data_mode": "sequence_dense",
        "model_name": model_name,
        "model_params": json.dumps(vars(config), default=str),
        "n_features": len(feature_cols),
        "lookback": config.lookback,
        "horizon": config.horizon,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "train_start": str(t_train.min()) if len(t_train) else None,
        "train_end": str(t_train.max()) if len(t_train) else None,
        "val_start": str(t_val.min()) if len(t_val) else None,
        "val_end": str(t_val.max()) if len(t_val) else None,
        "test_start": str(t_test.min()) if len(t_test) else None,
        "test_end": str(t_test.max()) if len(t_test) else None,
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
        "diagnostic_plot_path": str(plot_path),
    }

    append_experiment_log(record)

    return pd.DataFrame([record]), history_df, pred_df