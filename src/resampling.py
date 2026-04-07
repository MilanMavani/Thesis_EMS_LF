from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_time_indexed_csv(path: str | Path, *, time_col: str = "Time") -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).copy()
    df = df.set_index(time_col).sort_index()
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


def build_resample_rule_map(
    df: pd.DataFrame,
    *,
    mean_cols: list[str] | None = None,
    last_cols: list[str] | None = None,
    max_cols: list[str] | None = None,
    min_cols: list[str] | None = None,
    sum_cols: list[str] | None = None,
    default_numeric_rule: str = "mean",
) -> dict[str, str]:

    out: dict[str, str] = {}

    mean_cols = mean_cols or []
    last_cols = last_cols or []
    max_cols = max_cols or []
    min_cols = min_cols or []
    sum_cols = sum_cols or []

    for c in df.columns:
        if c in mean_cols:
            out[c] = "mean"
        elif c in last_cols:
            out[c] = "last"
        elif c in max_cols:
            out[c] = "max"
        elif c in min_cols:
            out[c] = "min"
        elif c in sum_cols:
            out[c] = "sum"
        else:
            # default for remaining columns
            out[c] = default_numeric_rule

    return out


def resample_timeseries(
    df: pd.DataFrame,
    *,
    target_freq: str,
    agg_map: dict[str, str],
    label: str = "left",
    closed: str = "left",
    drop_all_nan_rows: bool = False,
) -> pd.DataFrame:
  
    out = ensure_clean_datetime_index(df)

    missing_cols = [c for c in agg_map if c not in out.columns]
    if missing_cols:
        raise KeyError(f"Aggregation map contains columns not in dataframe: {missing_cols}")

    out = out.resample(target_freq, label=label, closed=closed).agg(agg_map)

    if drop_all_nan_rows:
        out = out.dropna(how="all")

    return out


def infer_default_rule_groups(df: pd.DataFrame) -> dict[str, list[str]]:
  
    mean_cols = []
    last_cols = []

    for c in df.columns:
        c_low = c.lower()

        # likely state / categorical / status-like
        if (
            "unitstate" in c_low
            or "status" in c_low
            or "state" in c_low
            or "mode" in c_low
        ):
            last_cols.append(c)
        else:
            mean_cols.append(c)

    return {
        "mean_cols": mean_cols,
        "last_cols": last_cols,
        "max_cols": [],
        "min_cols": [],
        "sum_cols": [],
    }


def summarize_resampled_dataframe(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    original_freq: str,
    target_freq: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "metric": [
                "rows_before",
                "rows_after",
                "cols_before",
                "cols_after",
                "start_before",
                "end_before",
                "start_after",
                "end_after",
                "original_freq",
                "target_freq",
            ],
            "value": [
                len(df_before),
                len(df_after),
                df_before.shape[1],
                df_after.shape[1],
                str(df_before.index.min()) if len(df_before) else None,
                str(df_before.index.max()) if len(df_before) else None,
                str(df_after.index.min()) if len(df_after) else None,
                str(df_after.index.max()) if len(df_after) else None,
                original_freq,
                target_freq,
            ],
        }
    )
    return out


def save_resampled_csv(
    df: pd.DataFrame,
    save_path: str | Path,
    *,
    date_format: str = "%Y-%m-%d %H:%M:%S",
    float_format: str = "%.5f",
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        save_path,
        index=True,
        date_format=date_format,
        float_format=float_format,
    )
    return save_path


def save_resampled_parquet(
    df: pd.DataFrame,
    save_path: str | Path,
    *,
    engine: str = "pyarrow",
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(save_path, engine=engine)
    return save_path


def convert_and_save_resolution(
    df: pd.DataFrame,
    *,
    target_freq: str,
    agg_map: dict[str, str],
    csv_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    label: str = "left",
    closed: str = "left",
    drop_all_nan_rows: bool = False,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    df_res = resample_timeseries(
        df,
        target_freq=target_freq,
        agg_map=agg_map,
        label=label,
        closed=closed,
        drop_all_nan_rows=drop_all_nan_rows,
    )

    saved: dict[str, Path] = {}

    if csv_path is not None:
        saved["csv"] = save_resampled_csv(df_res, csv_path)

    if parquet_path is not None:
        saved["parquet"] = save_resampled_parquet(df_res, parquet_path)

    return df_res, saved


def convert_multiple_resolutions(
    df: pd.DataFrame,
    *,
    target_freqs: list[str],
    agg_map: dict[str, str],
    output_dir: str | Path | None = None,
    file_stem: str = "lf_data",
    save_csv: bool = True,
    save_parquet: bool = False,
    label: str = "left",
    closed: str = "left",
    drop_all_nan_rows: bool = False,
) -> dict[str, dict]:

    results: dict[str, dict] = {}
    output_dir = Path(output_dir) if output_dir is not None else None

    for freq in target_freqs:
        csv_path = None
        parquet_path = None

        safe_freq = freq.replace("min", "m").replace("h", "h")

        if output_dir is not None:
            if save_csv:
                csv_path = output_dir / f"{file_stem}_{safe_freq}.csv"
            if save_parquet:
                parquet_path = output_dir / f"{file_stem}_{safe_freq}.parquet"

        df_res, saved = convert_and_save_resolution(
            df,
            target_freq=freq,
            agg_map=agg_map,
            csv_path=csv_path,
            parquet_path=parquet_path,
            label=label,
            closed=closed,
            drop_all_nan_rows=drop_all_nan_rows,
        )

        results[freq] = {"df": df_res, **saved}

    return results