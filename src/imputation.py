from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_time_indexed_csv(path: str | Path, *, time_col: str = "Time") -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).copy()
    df = df.set_index(time_col).sort_index()
    return df


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)

    out = pd.concat([missing_count, missing_pct], axis=1)
    out.columns = ["missing_count", "missing_pct"]
    out = out.sort_values("missing_count", ascending=False)
    return out


def impute_loads_by_gap_categories_safe(
    df: pd.DataFrame,
    load_cols: list[str],
    *,
    freq_minutes: int = 5,
    short_gap_hours: float = 2.0,
    medium_gap_hours: float = 24.0,
    min_history: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy().sort_index()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")
    if out.index.has_duplicates:
        raise ValueError("DatetimeIndex has duplicates. Resolve duplicates before imputing.")

    load_cols = [c for c in load_cols if c in out.columns]
    out[load_cols] = out[load_cols].apply(pd.to_numeric, errors="coerce")

    samples_per_hour = int(round(60 / freq_minutes))
    short_thr = int(round(short_gap_hours * samples_per_hour))
    medium_thr = int(round(medium_gap_hours * samples_per_hour))

    out["_dow"] = out.index.dayofweek
    out["_mod"] = out.index.hour * 60 + out.index.minute

    report_rows = []

    for col in load_cols:
        s = out[col]

        # Cat-1: short gaps
        is_na = s.isna()
        run_id = is_na.ne(is_na.shift()).cumsum()
        run_len = is_na.groupby(run_id).transform("sum")

        long_mask = is_na & (run_len > short_thr)

        s_interp = s.interpolate(method="time", limit=short_thr, limit_direction="forward")
        s_interp.loc[long_mask] = np.nan
        out.loc[:, col] = s_interp

        # Recompute after Cat-1
        s = out[col]
        is_na = s.isna()
        run_id = is_na.ne(is_na.shift()).cumsum()
        run_len = is_na.groupby(run_id).transform("sum")

        nan_run_lengths = run_len[is_na].groupby(run_id[is_na]).first()

        report_rows.append({
            "column": col,
            "NaNs_after_cat1": int(is_na.sum()),
            "NaN_runs_after_cat1": int(nan_run_lengths.shape[0]),
            "min_run": int(nan_run_lengths.min()) if nan_run_lengths.shape[0] else 0,
            "max_run": int(nan_run_lengths.max()) if nan_run_lengths.shape[0] else 0,
            "runs_cat2_(2h_to_24h)": int(((nan_run_lengths > short_thr) & (nan_run_lengths <= medium_thr)).sum()) if nan_run_lengths.shape[0] else 0,
            "runs_cat3_(>24h)": int((nan_run_lengths > medium_thr).sum()) if nan_run_lengths.shape[0] else 0,
        })

        # Cat-2 / Cat-3: strictly past-only
        nan_idx = out.index[out[col].isna()]

        for ts in nan_idx:
            L = int(run_len.loc[ts])
            dow = int(out.loc[ts, "_dow"])
            mod = int(out.loc[ts, "_mod"])

            past = out.loc[:ts].iloc[:-1]

            if (L > short_thr) and (L <= medium_thr):
                cands = past.loc[(past["_dow"] == dow) & (past["_mod"] == mod), col].dropna()
                if len(cands) >= min_history:
                    out.loc[ts, col] = float(cands.mean())
                    continue

            cands2 = past.loc[past["_mod"] == mod, col].dropna()
            if len(cands2) >= min_history:
                out.loc[ts, col] = float(cands2.median())
                continue

            cands3 = past[col].dropna()
            if len(cands3) > 0:
                out.loc[ts, col] = float(cands3.median())

    out = out.drop(columns=["_dow", "_mod"])
    report = pd.DataFrame(report_rows)
    return out, report


def apply_imputed_columns(
    df_original: pd.DataFrame,
    df_imputed: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    out = df_original.copy()
    cols = [c for c in cols if c in out.columns and c in df_imputed.columns]
    out.loc[:, cols] = df_imputed.loc[:, cols]
    return out