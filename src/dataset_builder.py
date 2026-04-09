from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_target_df(
    df: pd.DataFrame,
    target_col: str,
    support_cols: list[str],
    *,
    calendar_cols: list[str] | None = None,
    include_calendar: bool = True,
    include_lags: bool = True,
    include_rolls: bool = True,
    include_trends: bool = True,
    include_history: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    if calendar_cols is None:
        calendar_cols = []

    lag_cols = [c for c in out.columns if c.startswith(f"{target_col}_lag_")]
    roll_cols = [c for c in out.columns if c.startswith(f"{target_col}_roll_")]
    trend_cols = [c for c in out.columns if c.startswith(f"{target_col}_trend_")]
    history_cols = [c for c in out.columns if c.startswith(f"{target_col}_yesterday_")]
    history_cols += [c for c in out.columns if c.startswith(f"{target_col}_lastweek_")]

    selected_cols = [target_col] + [c for c in support_cols if c in out.columns]

    if include_calendar:
        selected_cols += [c for c in calendar_cols if c in out.columns]
    if include_lags:
        selected_cols += lag_cols
    if include_rolls:
        selected_cols += roll_cols
    if include_trends:
        selected_cols += trend_cols
    if include_history:
        selected_cols += history_cols

    selected_cols = [c for c in selected_cols if c in out.columns]
    return out[selected_cols].copy()


def clean_target_df(
    df_target: pd.DataFrame,
    target_col: str,
    *,
    drop_startup: bool = True,
    startup_rows: int = 288,
    drop_target_nan: bool = True,
) -> pd.DataFrame:
    out = df_target.copy()

    if drop_startup:
        out = out.iloc[startup_rows:].copy()

    if drop_target_nan:
        out = out[out[target_col].notna()].copy()

    return out


def build_multiple_target_datasets(
    df: pd.DataFrame,
    *,
    target_cols: list[str],
    support_cols: list[str],
    calendar_cols: list[str] | None = None,
    include_calendar: bool = True,
    include_lags: bool = True,
    include_rolls: bool = True,
    include_trends: bool = True,
    include_history: bool = True,
    drop_startup: bool = True,
    startup_rows: int = 288,
    drop_target_nan: bool = True,
) -> dict[str, pd.DataFrame]:
    datasets = {}

    for target_col in target_cols:
        df_target = build_target_df(
            df,
            target_col=target_col,
            support_cols=support_cols,
            calendar_cols=calendar_cols,
            include_calendar=include_calendar,
            include_lags=include_lags,
            include_rolls=include_rolls,
            include_trends=include_trends,
            include_history=include_history,
        )

        df_target = clean_target_df(
            df_target,
            target_col=target_col,
            drop_startup=drop_startup,
            startup_rows=startup_rows,
            drop_target_nan=drop_target_nan,
        )

        datasets[target_col] = df_target

    return datasets


def save_target_datasets_parquet(
    datasets: dict[str, pd.DataFrame],
    save_dir: str | Path,
    *,
    engine: str = "pyarrow",
) -> dict[str, Path]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    for target_col, df_target in datasets.items():
        file_path = save_dir / f"df_{target_col}.parquet"
        df_target.to_parquet(file_path, engine=engine)
        saved[target_col] = file_path

    return saved