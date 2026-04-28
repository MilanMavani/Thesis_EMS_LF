# src/emd_features.py

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import CubicSpline
except ImportError:
    CubicSpline = None


def _find_local_extrema(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima and minima indices.
    """
    dx1 = x[1:-1] - x[:-2]
    dx2 = x[1:-1] - x[2:]

    maxima = np.where((dx1 > 0) & (dx2 > 0))[0] + 1
    minima = np.where((dx1 < 0) & (dx2 < 0))[0] + 1

    return maxima, minima


def _envelope(x: np.ndarray, extrema_idx: np.ndarray) -> np.ndarray:
    """
    Build spline envelope through extrema points.
    Endpoints are included to stabilize interpolation.
    """
    n = len(x)

    if CubicSpline is None:
        raise ImportError("scipy is required for local EMD. Install with: pip install scipy")

    if len(extrema_idx) < 2:
        return np.full(n, np.nan)

    idx = np.unique(np.concatenate(([0], extrema_idx, [n - 1])))
    vals = x[idx]

    if len(idx) < 4:
        return np.interp(np.arange(n), idx, vals)

    spline = CubicSpline(idx, vals, bc_type="natural")
    return spline(np.arange(n))


def _is_imf(h: np.ndarray) -> bool:
    """
    Basic IMF stopping criterion:
    number of extrema and zero crossings should differ by at most 1.
    """
    maxima, minima = _find_local_extrema(h)
    n_extrema = len(maxima) + len(minima)

    zero_crossings = np.sum(h[:-1] * h[1:] < 0)

    return abs(n_extrema - zero_crossings) <= 1


def simple_emd(
    x: np.ndarray,
    *,
    max_imfs: int = 4,
    max_siftings: int = 10,
    stop_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple Empirical Mode Decomposition implementation.

    Returns
    -------
    imfs : np.ndarray
        Array with shape [n_imfs, n_samples]
    residual : np.ndarray
        Remaining residual signal
    """
    x = np.asarray(x, dtype=float)

    if np.isnan(x).any():
        raise ValueError("Input signal contains NaNs.")

    residual = x.copy()
    imfs = []

    for _ in range(max_imfs):
        maxima, minima = _find_local_extrema(residual)

        if len(maxima) + len(minima) < 3:
            break

        h = residual.copy()

        for _ in range(max_siftings):
            maxima, minima = _find_local_extrema(h)

            if len(maxima) < 2 or len(minima) < 2:
                break

            upper = _envelope(h, maxima)
            lower = _envelope(h, minima)

            if np.isnan(upper).any() or np.isnan(lower).any():
                break

            mean_env = (upper + lower) / 2.0
            h_new = h - mean_env

            denom = np.sum(h ** 2) + 1e-12
            sift_change = np.sum((h - h_new) ** 2) / denom

            h = h_new

            if _is_imf(h) and sift_change < stop_std:
                break

        imfs.append(h)
        residual = residual - h

        maxima, minima = _find_local_extrema(residual)
        if len(maxima) + len(minima) < 3:
            break

    if len(imfs) == 0:
        return np.empty((0, len(x))), residual

    return np.vstack(imfs), residual


def _safe_series_to_array(x: pd.Series) -> np.ndarray | None:
    """
    Convert history window to clean numeric array.
    """
    x = pd.to_numeric(x, errors="coerce")

    if x.notna().sum() < max(20, int(len(x) * 0.5)):
        return None

    x = x.interpolate(method="time", limit_direction="both")
    x = x.ffill().bfill()

    values = x.to_numpy(dtype=float)

    if np.isnan(values).any():
        return None

    if np.nanstd(values) == 0:
        return None

    return values


def _emd_features_from_values(
    values: np.ndarray,
    *,
    max_imfs: int = 4,
    prefix: str,
) -> dict[str, float]:
    """
    Extract compact EMD features from a past-only signal window.
    """
    imfs, residual = simple_emd(
        values,
        max_imfs=max_imfs,
        max_siftings=10,
        stop_std=0.05,
    )

    feats = {}

    total_energy = float(np.sum(values ** 2))
    if total_energy <= 0:
        total_energy = np.nan

    n_imfs = imfs.shape[0]
    feats[f"{prefix}_emd_n_imfs"] = float(n_imfs)

    used_imf_energy = 0.0
    high_freq_energy = 0.0

    for k in range(max_imfs):
        if k < n_imfs:
            imf = imfs[k]
            energy = float(np.sum(imf ** 2))
            used_imf_energy += energy

            if k < 2:
                high_freq_energy += energy

            feats[f"{prefix}_emd_imf{k+1}_last"] = float(imf[-1])
            feats[f"{prefix}_emd_imf{k+1}_mean"] = float(np.mean(imf))
            feats[f"{prefix}_emd_imf{k+1}_std"] = float(np.std(imf))
            feats[f"{prefix}_emd_imf{k+1}_energy_ratio"] = (
                energy / total_energy if total_energy and not np.isnan(total_energy) else np.nan
            )
            feats[f"{prefix}_emd_imf{k+1}_amplitude"] = float(np.max(imf) - np.min(imf))
        else:
            feats[f"{prefix}_emd_imf{k+1}_last"] = np.nan
            feats[f"{prefix}_emd_imf{k+1}_mean"] = np.nan
            feats[f"{prefix}_emd_imf{k+1}_std"] = np.nan
            feats[f"{prefix}_emd_imf{k+1}_energy_ratio"] = np.nan
            feats[f"{prefix}_emd_imf{k+1}_amplitude"] = np.nan

    feats[f"{prefix}_emd_residual_last"] = float(residual[-1])
    feats[f"{prefix}_emd_residual_mean"] = float(np.mean(residual))
    feats[f"{prefix}_emd_residual_std"] = float(np.std(residual))
    feats[f"{prefix}_emd_residual_trend"] = float(residual[-1] - residual[0])

    feats[f"{prefix}_emd_high_freq_energy_ratio"] = (
        high_freq_energy / total_energy if total_energy and not np.isnan(total_energy) else np.nan
    )

    feats[f"{prefix}_emd_used_imf_energy_ratio"] = (
        used_imf_energy / total_energy if total_energy and not np.isnan(total_energy) else np.nan
    )

    return feats


def add_emd_features_at_issue_times(
    df: pd.DataFrame,
    *,
    target_col: str,
    history_steps: int = 96 * 7,
    issue_hour: int = 23,
    issue_minute: int = 45,
    max_imfs: int = 4,
    min_history_steps: int = 96,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add leakage-safe EMD features for day-ahead profile forecasting.

    EMD is calculated only using past data up to the forecast issue time.
    Features are filled only at issue timestamps, e.g. 23:45.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in dataframe.")

    out = df.sort_index().copy()
    prefix = target_col

    issue_mask = (
        (out.index.hour == issue_hour)
        & (out.index.minute == issue_minute)
    )

    issue_times = out.index[issue_mask]
    feature_rows = []

    for ts in issue_times:
        hist = out.loc[:ts, target_col].tail(history_steps)

        if len(hist) < min_history_steps:
            feature_rows.append(pd.Series(name=ts, dtype=float))
            continue

        values = _safe_series_to_array(hist)

        if values is None:
            feature_rows.append(pd.Series(name=ts, dtype=float))
            continue

        try:
            feats = _emd_features_from_values(
                values,
                max_imfs=max_imfs,
                prefix=prefix,
            )
        except Exception:
            feats = {}

        feature_rows.append(pd.Series(feats, name=ts))

    emd_df = pd.DataFrame(feature_rows)

    if emd_df.empty:
        return out, []

    emd_feature_cols = list(emd_df.columns)

    for c in emd_feature_cols:
        out[c] = np.nan

    out.loc[emd_df.index, emd_feature_cols] = emd_df[emd_feature_cols]

    return out, emd_feature_cols