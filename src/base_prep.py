from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_time_indexed_csv(path: str | Path, *, time_col: str = "Time") -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).copy()
    df = df.set_index(time_col)
    return df


def ensure_clean_datetime_index(
    df: pd.DataFrame,
    *,
    keep: str = "first",
) -> pd.DataFrame:
    out = df.copy()

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep=keep)].copy()

    return out


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)

    out = pd.concat([missing_count, missing_pct], axis=1)
    out.columns = ["missing_count", "missing_pct"]
    out = out.sort_values("missing_count", ascending=False)
    return out