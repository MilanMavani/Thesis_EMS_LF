from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# Load helper
# =========================================================

def load_time_indexed_csv(
    path: str | Path,
    *,
    time_col: str = "Time",
) -> pd.DataFrame:
    """
    Load a CSV file, parse the Time column, set it as DatetimeIndex,
    and sort chronologically.
    """
    df = pd.read_csv(path, sep=",")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).copy()

    df = df.set_index(time_col)
    df = df.sort_index()

    return df


# =========================================================
# Missing summary helper
# =========================================================

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return missing count and missing percentage for every column.
    """
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)

    out = pd.concat([missing_count, missing_pct], axis=1)
    out.columns = ["missing_count", "missing_pct"]
    out = out.sort_values("missing_count", ascending=False)

    return out


# =========================================================
# Imputation with flags
# =========================================================

def impute_loads_by_gap_categories_safe_with_flags(
    df: pd.DataFrame,
    load_cols: list[str],
    *,
    freq_minutes: int = 5,
    short_gap_hours: float = 2.0,
    medium_gap_hours: float = 24.0,
    min_history: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    out = df.copy().sort_index()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    if out.index.has_duplicates:
        raise ValueError("DatetimeIndex has duplicates. Resolve duplicates before imputing.")

    load_cols = [c for c in load_cols if c in out.columns]

    if not load_cols:
        raise ValueError("None of the requested load_cols exist in df.")

    out[load_cols] = out[load_cols].apply(pd.to_numeric, errors="coerce")

    original_na = out[load_cols].isna().copy()

    impute_flags = pd.DataFrame(
        0,
        index=out.index,
        columns=[f"{c}_was_imputed" for c in load_cols],
    )

    impute_methods = pd.DataFrame(
        "",
        index=out.index,
        columns=[f"{c}_impute_method" for c in load_cols],
    )

    samples_per_hour = int(round(60 / freq_minutes))
    short_thr = int(round(short_gap_hours * samples_per_hour))
    medium_thr = int(round(medium_gap_hours * samples_per_hour))

    out["_dow"] = out.index.dayofweek
    out["_mod"] = out.index.hour * 60 + out.index.minute

    report_rows = []

    for col in load_cols:
        flag_col = f"{col}_was_imputed"
        method_col = f"{col}_impute_method"

        s_original = out[col].copy()

        # Original NaN runs
        is_na_original = s_original.isna()
        run_id_original = is_na_original.ne(is_na_original.shift()).cumsum()
        run_len_original = is_na_original.groupby(run_id_original).transform("sum")

        short_gap_mask = is_na_original & (run_len_original <= short_thr)
        long_gap_mask = is_na_original & (run_len_original > short_thr)

        # -------------------------------------------------
        # Cat-1: short gaps by time interpolation
        # -------------------------------------------------
        s_interp = s_original.interpolate(
            method="time",
            limit=short_thr,
            limit_direction="forward",
        )

        # Do not let interpolation fill long gaps
        s_interp.loc[long_gap_mask] = np.nan

        filled_by_cat1 = short_gap_mask & s_interp.notna()

        out.loc[:, col] = s_interp
        impute_flags.loc[filled_by_cat1, flag_col] = 1
        impute_methods.loc[filled_by_cat1, method_col] = "cat1_short_time_interp"

        # -------------------------------------------------
        # Recompute remaining NaNs after Cat-1
        # -------------------------------------------------
        s = out[col]
        is_na = s.isna()
        run_id = is_na.ne(is_na.shift()).cumsum()
        run_len = is_na.groupby(run_id).transform("sum")

        nan_run_lengths = run_len[is_na].groupby(run_id[is_na]).first()

        report_rows.append({
            "column": col,
            "NaNs_original": int(is_na_original.sum()),
            "NaNs_after_cat1": int(is_na.sum()),
            "NaN_runs_after_cat1": int(nan_run_lengths.shape[0]),
            "min_run": int(nan_run_lengths.min()) if nan_run_lengths.shape[0] else 0,
            "max_run": int(nan_run_lengths.max()) if nan_run_lengths.shape[0] else 0,
            "runs_cat2_(2h_to_24h)": int(
                ((nan_run_lengths > short_thr) & (nan_run_lengths <= medium_thr)).sum()
            ) if nan_run_lengths.shape[0] else 0,
            "runs_cat3_(>24h)": int(
                (nan_run_lengths > medium_thr).sum()
            ) if nan_run_lengths.shape[0] else 0,
        })

        # -------------------------------------------------
        # Cat-2 and Cat-3: strictly past-only filling
        # -------------------------------------------------
        nan_idx = out.index[out[col].isna()]

        for ts in nan_idx:
            L = int(run_len.loc[ts])
            dow = int(out.loc[ts, "_dow"])
            mod = int(out.loc[ts, "_mod"])

            # Important: only past data, no leakage
            past = out.loc[:ts].iloc[:-1]

            # Cat-2:
            # medium gap: same day-of-week + same minute-of-day mean
            if (L > short_thr) and (L <= medium_thr):
                cands = past.loc[
                    (past["_dow"] == dow) & (past["_mod"] == mod),
                    col,
                ].dropna()

                if len(cands) >= min_history:
                    out.loc[ts, col] = float(cands.mean())
                    impute_flags.loc[ts, flag_col] = 1
                    impute_methods.loc[ts, method_col] = "cat2_same_dow_mod_mean"
                    continue

            # Cat-3 fallback A:
            # same minute-of-day median
            cands2 = past.loc[past["_mod"] == mod, col].dropna()

            if len(cands2) >= min_history:
                out.loc[ts, col] = float(cands2.median())
                impute_flags.loc[ts, flag_col] = 1
                impute_methods.loc[ts, method_col] = "cat3_same_mod_median"
                continue

            # Cat-3 fallback B:
            # global past median
            cands3 = past[col].dropna()

            if len(cands3) > 0:
                out.loc[ts, col] = float(cands3.median())
                impute_flags.loc[ts, flag_col] = 1
                impute_methods.loc[ts, method_col] = "cat3_past_global_median"

    out = out.drop(columns=["_dow", "_mod"])

    # Final safety check:
    # Flag only points that were originally NaN and are now filled.
    for col in load_cols:
        flag_col = f"{col}_was_imputed"
        method_col = f"{col}_impute_method"

        valid_imputed = original_na[col] & out[col].notna()

        impute_flags[flag_col] = valid_imputed.astype(int)
        impute_methods.loc[~valid_imputed, method_col] = ""

    report = pd.DataFrame(report_rows)

    return out, report, impute_flags, impute_methods


# =========================================================
# Apply imputed values helper
# =========================================================

def apply_imputed_columns(
    df_original: pd.DataFrame,
    df_imputed: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """
    Replace selected columns in the original dataframe with imputed versions.
    """
    out = df_original.copy()

    cols = [c for c in cols if c in out.columns and c in df_imputed.columns]
    out.loc[:, cols] = df_imputed.loc[:, cols]

    return out