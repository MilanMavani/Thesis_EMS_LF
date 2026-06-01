from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# Default column groups / configs
# =========================================================

PV_SENSOR_COLS = [
    "PV_WS_Radiation",
    "PV_WS_AirTemp",
    "PV_WS_RelHum",
    "PV_WS_RelAirPre",
]

BESS_VALIDATE_COLS = [
    "BA_PwrAtChrLimTot",
    "BA_PwrAtDisLimTot",
    "BA_TotActPwr_BESS_AC_Panel2",
    "BA_TotActPwr_BESS_AC_Panel1",
    "BA_PwrAt",
    "BA_Soc",
]

BUILDING_DETECT_COLS = [
    "BU_TotActPwr_Tech_Room",
    "BU_TotActPwr_SDB_EL_Substation",
    "BU_TotActPwr_Academy",
    "BU_PwrReq",
]

BUILDING_VALIDATE_COLS = BUILDING_DETECT_COLS.copy()

ELX_VALIDATE_COLS = ["EL_TotActPwr_NitrogenUnit"]

DEFAULT_DROP_COLS = [
    "EL_AuxPwr",
    "BA_PwrAt",
    "BA_Unitstate",
    "BA_PwrAtChrLimTot",
    "BA_PwrAtDisLimTot",
    "EL_TotActPwr_NitrogenUnit",
    "PV_WS_RelAirPre",
    "PV_Unitstate",
    "BU_PwrReq",
    "ELX_TotActPwr_Pump_Room",
    "BU_Unitstate",
]

DEFAULT_PRE_DROP_COLS = [
    "TA_TotActPwr_Comp1",
    "TA_TotActPwr_Comp2",
    "TA_TotActPwr_Comp3",
    "BA_TotActPwr_Chiller",
    "BU_TotActPwr_UPS2",
    "BU_TotActPwr_UPS1",
    "ELX_AuxPowCons",
    "EL_TotActPwr_Ely_BoP",
]


# =========================================================
# Load / prepare helpers
# =========================================================

def load_time_indexed_csv(path: str | Path, *, time_col: str = "Time") -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).copy()
    df = df.set_index(time_col).sort_index()
    return df


def drop_columns_if_exist(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.drop(columns=cols, errors="ignore").copy()


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.isna()].sort_index()
    return d


def coerce_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def apply_cleaned_columns(
    df_raw: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    out = df_raw.copy()
    cols_existing = [c for c in cols if c in out.columns and c in df_cleaned.columns]
    out.loc[df_cleaned.index, cols_existing] = df_cleaned.loc[df_cleaned.index, cols_existing]
    return out


# =========================================================
# Event / stuck detection helpers
# =========================================================

def keep_long_runs(mask: pd.Series, *, min_samples: int) -> pd.Series:
    m = mask.fillna(False).astype(bool)
    if min_samples <= 1:
        return m
    g = (m != m.shift()).cumsum()
    run_len = m.groupby(g).transform("sum")
    return m & (run_len >= min_samples)


def run_intervals(mask: pd.Series) -> pd.DataFrame:
    m = mask.fillna(False).astype(bool)
    if not m.any():
        return pd.DataFrame(columns=["start", "end", "samples"])

    idx = m.index
    g = (m != m.shift()).cumsum()
    true_groups = g[m]

    starts = idx[m].to_series().groupby(true_groups).first()
    ends = idx[m].to_series().groupby(true_groups).last()
    lens = m.groupby(g).sum()
    lens = lens.loc[lens.index.isin(true_groups.unique())].astype(int)

    out = pd.DataFrame({
        "start": starts.values,
        "end": ends.values,
        "samples": lens.values,
    })
    return out.sort_values("start").reset_index(drop=True)


def merge_intervals(intervals: pd.DataFrame, *, max_gap: pd.Timedelta) -> pd.DataFrame:
    if intervals is None or intervals.empty:
        return pd.DataFrame(columns=["start", "end", "samples"])

    itv = intervals.sort_values("start").reset_index(drop=True).copy()

    merged = []
    cur_s = itv.loc[0, "start"]
    cur_e = itv.loc[0, "end"]
    cur_samples = int(itv.loc[0, "samples"])

    for i in range(1, len(itv)):
        s, e = itv.loc[i, "start"], itv.loc[i, "end"]
        if (s - cur_e) <= max_gap:
            cur_e = max(cur_e, e)
            cur_samples += int(itv.loc[i, "samples"])
        else:
            merged.append((cur_s, cur_e, cur_samples))
            cur_s, cur_e, cur_samples = s, e, int(itv.loc[i, "samples"])

    merged.append((cur_s, cur_e, cur_samples))
    return pd.DataFrame(merged, columns=["start", "end", "samples"]).reset_index(drop=True)


def max_consecutive_true(mask: pd.Series) -> int:
    m = mask.fillna(False).astype(bool)
    if not m.any():
        return 0
    g = (m != m.shift()).cumsum()
    lens = m.groupby(g).sum()
    first = m.groupby(g).first()
    true_lens = lens[first]
    return int(true_lens.max()) if len(true_lens) else 0


def stuck_decision_graded(
    seg: pd.Series,
    *,
    step: pd.Timedelta,
    zero_tol: float = 0.0,
    high_thr: float = 0.95,
    med_thr: float = 0.85,
    probable_min_hours: float = 2.0,
    change_rate_thr: float = 0.01,
    min_non_nan: int = 10,
) -> tuple[str, str, dict]:
    x = pd.to_numeric(seg, errors="coerce")
    n_total = int(len(x))
    n = int(x.notna().sum())

    metrics = {
        "n_total": n_total,
        "n_non_nan": n,
        "zero_ratio": np.nan,
        "dominant_ratio": np.nan,
        "change_rate": np.nan,
        "unique_count": np.nan,
    }

    if n < min_non_nan:
        return "no", "too_few_non_nan", metrics

    xs = x.dropna()

    z = xs.abs().le(zero_tol)
    metrics["zero_ratio"] = float(z.mean())

    rounded = xs.round(6)
    vc = rounded.value_counts(dropna=True)
    dominant_ratio = float(vc.iloc[0] / len(rounded)) if len(vc) else np.nan
    metrics["dominant_ratio"] = dominant_ratio
    metrics["unique_count"] = int(vc.size)

    dif = xs.diff().abs()
    change_rate = float((dif.gt(zero_tol)).mean())
    metrics["change_rate"] = change_rate

    dur_hours = (len(xs) * step) / pd.Timedelta(hours=1)

    if dominant_ratio >= high_thr:
        return "definite", "dominant_value_high", metrics
    if dominant_ratio >= med_thr:
        return "probable", "dominant_value_med", metrics
    if (dur_hours >= probable_min_hours) and (change_rate <= change_rate_thr):
        return "probable", "low_change_rate_long", metrics

    return "no", "normal_variation", metrics


def events_from_intervals(intervals: pd.DataFrame, group: str, event_type: str) -> pd.DataFrame:
    if intervals is None or intervals.empty:
        return pd.DataFrame(columns=["event_id", "group", "event_type", "start", "end", "minutes"])

    out = intervals.copy()
    out["group"] = group
    out["event_type"] = event_type
    out["minutes"] = ((out["end"] - out["start"]) / pd.Timedelta(minutes=1)).astype(float)
    out = out.loc[:, ["group", "event_type", "start", "end", "minutes"]]
    out = out.sort_values(["start", "group"]).reset_index(drop=True)
    out.insert(0, "event_id", np.arange(1, len(out) + 1))
    return out


def detect_consensus_zero_intervals(
    d: pd.DataFrame,
    *,
    cols: list[str],
    freq: str,
    min_minutes: int,
    zero_tol: float,
    min_cols_fraction: float,
) -> tuple[pd.DataFrame, dict]:
    cols_used = [c for c in cols if c in d.columns]
    if not cols_used:
        return pd.DataFrame(columns=["start", "end", "samples"]), {"cols_used": []}

    step = pd.to_timedelta(freq)
    min_samples = int(np.ceil(pd.Timedelta(minutes=min_minutes) / step))

    X = d[cols_used].apply(pd.to_numeric, errors="coerce")
    Z = X.abs().le(zero_tol)
    base_mask = Z.mean(axis=1).ge(min_cols_fraction)

    event_mask = keep_long_runs(base_mask, min_samples=min_samples)
    intervals_raw = run_intervals(event_mask)

    debug = {
        "cols_used": cols_used,
        "min_cols_fraction": float(min_cols_fraction),
        "min_minutes": int(min_minutes),
        "freq": freq,
        "min_samples": int(min_samples),
        "base_hits": int(base_mask.sum()),
        "event_hits_after_duration_filter": int(event_mask.sum()),
        "num_intervals_raw": int(len(intervals_raw)),
        "zero_tol": float(zero_tol),
    }
    return intervals_raw, debug


def detect_single_zero_intervals(
    d: pd.DataFrame,
    *,
    col: str,
    freq: str,
    min_minutes: int,
    zero_tol: float,
) -> tuple[pd.DataFrame, dict]:
    if col not in d.columns:
        return pd.DataFrame(columns=["start", "end", "samples"]), {"col_used": None}

    step = pd.to_timedelta(freq)
    min_samples = int(np.ceil(pd.Timedelta(minutes=min_minutes) / step))

    x = pd.to_numeric(d[col], errors="coerce")
    base_mask = x.abs().le(zero_tol)

    event_mask = keep_long_runs(base_mask, min_samples=min_samples)
    intervals_raw = run_intervals(event_mask)

    debug = {
        "col_used": col,
        "min_minutes": int(min_minutes),
        "freq": freq,
        "min_samples": int(min_samples),
        "base_hits": int(base_mask.sum()),
        "event_hits_after_duration_filter": int(event_mask.sum()),
        "num_intervals_raw": int(len(intervals_raw)),
        "zero_tol": float(zero_tol),
    }
    return intervals_raw, debug


def pv_pressure_priority_qc_mask(
    d: pd.DataFrame,
    *,
    pressure_col: str,
    other_cols: list[str],
    p_lo: float,
    p_hi: float,
    zero_tol: float,
    min_other_fraction: float = 1.0,
) -> pd.Series:
    if pressure_col not in d.columns:
        return pd.Series(False, index=d.index)

    p = pd.to_numeric(d[pressure_col], errors="coerce")
    p_bad = p.lt(p_lo) | p.gt(p_hi)

    others = [c for c in other_cols if c in d.columns and c != pressure_col]
    if not others:
        return pd.Series(False, index=d.index)

    Xo = d[others].apply(pd.to_numeric, errors="coerce")
    others_zero = Xo.abs().le(zero_tol).mean(axis=1).ge(min_other_fraction)

    return p_bad & others_zero


# =========================================================
# Event-based cleaning pipeline
# =========================================================

def full_comm_loss_maintenance_pipeline(
    df: pd.DataFrame,
    *,
    freq: str = "5min",
    merge_gap_minutes: int = 30,

    bess_detect_chr: str = "BA_PwrAtChrLimTot",
    bess_detect_dis: str = "BA_PwrAtDisLimTot",
    bess_tol_zero_limits: float = 0.0,
    bess_min_minutes: int = 30,
    bess_cols_validate_nanify: list[str],

    pv_sensor_cols: list[str],
    pv_min_minutes: int = 30,
    pv_zero_tol: float = 0.0,
    pv_min_cols_fraction: float = 0.75,
    pv_pressure_col: str = "PV_WS_RelAirPre",
    pv_pressure_lo: float = 800.0,
    pv_pressure_hi: float = 1100.0,
    pv_pressure_other_fraction: float = 1.0,

    building_cols_detect: list[str],
    building_cols_validate_nanify: list[str],
    building_min_minutes: int = 30,
    building_zero_tol: float = 0.0,
    building_min_cols_fraction: float = 0.75,

    elx_detect_col: str = "EL_TotActPwr_NitrogenUnit",
    elx_cols_validate_nanify: list[str] | None = None,
    elx_min_minutes: int = 30,
    elx_zero_tol: float = 0.0,

    zero_tol_signals: float = 0.0,
    high_thr: float = 0.95,
    med_thr: float = 0.85,
    probable_min_hours: float = 2.0,
    change_rate_thr: float = 0.01,
    min_non_nan: int = 10,
    nanify_levels: tuple[str, ...] = ("definite", "probable"),

    zero_run_min_minutes: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict]:
    d0 = ensure_datetime_index(df).copy()
    step = pd.to_timedelta(freq)
    merge_gap = pd.Timedelta(minutes=int(merge_gap_minutes))
    zero_run_min_samples = int(np.ceil(pd.Timedelta(minutes=int(zero_run_min_minutes)) / step))

    if elx_cols_validate_nanify is None:
        elx_cols_validate_nanify = [elx_detect_col]

    numeric_cols = sorted(set(
        [bess_detect_chr, bess_detect_dis]
        + list(bess_cols_validate_nanify)
        + list(pv_sensor_cols)
        + list(building_cols_detect)
        + list(building_cols_validate_nanify)
        + list(elx_cols_validate_nanify)
        + [elx_detect_col]
    ))
    numeric_cols = [c for c in numeric_cols if c in d0.columns]
    d0 = coerce_numeric_columns(d0, numeric_cols)

    cleaned = d0.copy()

    # BESS detect
    if bess_detect_chr not in d0.columns or bess_detect_dis not in d0.columns:
        bess_raw = pd.DataFrame(columns=["start", "end", "samples"])
        bess_debug = {"missing_cols": [c for c in [bess_detect_chr, bess_detect_dis] if c not in d0.columns]}
    else:
        chr_v = d0[bess_detect_chr]
        dis_v = d0[bess_detect_dis]
        base_mask = chr_v.abs().le(bess_tol_zero_limits) & dis_v.abs().le(bess_tol_zero_limits)

        min_samples = int(np.ceil(pd.Timedelta(minutes=bess_min_minutes) / step))
        event_mask = keep_long_runs(base_mask, min_samples=min_samples)
        bess_raw = run_intervals(event_mask)

        bess_debug = {
            "detect_cols": [bess_detect_chr, bess_detect_dis],
            "min_minutes": int(bess_min_minutes),
            "min_samples": int(min_samples),
            "base_hits": int(base_mask.sum()),
            "num_intervals_raw": int(len(bess_raw)),
            "tol_zero_limits": float(bess_tol_zero_limits),
        }

    pv_raw, pv_debug = detect_consensus_zero_intervals(
        d0,
        cols=pv_sensor_cols,
        freq=freq,
        min_minutes=pv_min_minutes,
        zero_tol=pv_zero_tol,
        min_cols_fraction=pv_min_cols_fraction,
    )

    building_raw, building_debug = detect_consensus_zero_intervals(
        d0,
        cols=building_cols_detect,
        freq=freq,
        min_minutes=building_min_minutes,
        zero_tol=building_zero_tol,
        min_cols_fraction=building_min_cols_fraction,
    )

    elx_raw, elx_debug = detect_single_zero_intervals(
        d0,
        col=elx_detect_col,
        freq=freq,
        min_minutes=elx_min_minutes,
        zero_tol=elx_zero_tol,
    )

    bess_merged = merge_intervals(bess_raw, max_gap=merge_gap)
    pv_merged = merge_intervals(pv_raw, max_gap=merge_gap)
    building_merged = merge_intervals(building_raw, max_gap=merge_gap)
    elx_merged = merge_intervals(elx_raw, max_gap=merge_gap)

    reports: dict[str, pd.DataFrame] = {}

    # PV cleaning
    pv_cols_existing = [c for c in pv_sensor_cols if c in cleaned.columns]

    pv_priority_mask = pv_pressure_priority_qc_mask(
        cleaned,
        pressure_col=pv_pressure_col,
        other_cols=pv_sensor_cols,
        p_lo=pv_pressure_lo,
        p_hi=pv_pressure_hi,
        zero_tol=pv_zero_tol,
        min_other_fraction=pv_pressure_other_fraction,
    )

    if pv_cols_existing:
        cleaned.loc[pv_priority_mask, pv_cols_existing] = np.nan

    if pv_cols_existing and len(pv_merged):
        for _, r in pv_merged.iterrows():
            cleaned.loc[r["start"]:r["end"], pv_cols_existing] = np.nan

    pv_debug["priority_qc_hits"] = int(pv_priority_mask.sum())

    # graded cleaning helper
    def _graded_interval_clean(
        cleaned_df: pd.DataFrame,
        intervals: pd.DataFrame,
        cols: list[str],
        *,
        group_name: str,
        event_type: str,
    ) -> pd.DataFrame:
        rows = []
        cols_existing = [c for c in cols if c in cleaned_df.columns]

        for interval_id, r in intervals.reset_index(drop=True).iterrows():
            a, b = r["start"], r["end"]
            block = cleaned_df.loc[a:b, cols_existing] if cols_existing else pd.DataFrame(index=cleaned_df.loc[a:b].index)

            for col in cols_existing:
                seg = block[col]

                stuck_level, reason, metrics = stuck_decision_graded(
                    seg,
                    step=step,
                    zero_tol=zero_tol_signals,
                    high_thr=high_thr,
                    med_thr=med_thr,
                    probable_min_hours=probable_min_hours,
                    change_rate_thr=change_rate_thr,
                    min_non_nan=min_non_nan,
                )

                nz = seg.abs().le(zero_tol_signals)
                max_zero_run = max_consecutive_true(nz)

                if (stuck_level == "no") and (max_zero_run >= zero_run_min_samples):
                    stuck_level = "probable"
                    reason = "long_consecutive_zero_run"

                rows.append({
                    "group": group_name,
                    "event_type": event_type,
                    "interval_id": int(interval_id),
                    "start": a,
                    "end": b,
                    "column": col,
                    "stuck_level": stuck_level,
                    "reason": reason,
                    "max_consecutive_zero_samples": int(max_zero_run),
                    **metrics,
                })

                if stuck_level in nanify_levels:
                    cleaned_df.loc[a:b, col] = np.nan

        return pd.DataFrame(rows)

    reports["bess"] = _graded_interval_clean(
        cleaned, bess_merged, bess_cols_validate_nanify,
        group_name="BESS", event_type="bess_limits_zero"
    )
    reports["building"] = _graded_interval_clean(
        cleaned, building_merged, building_cols_validate_nanify,
        group_name="BUILDING", event_type="building_consensus_zero"
    )
    reports["elx"] = _graded_interval_clean(
        cleaned, elx_merged, elx_cols_validate_nanify,
        group_name="ELX", event_type="elx_single_zero"
    )

    events_all = pd.concat([
        events_from_intervals(bess_merged, "BESS", "bess_limits_zero"),
        events_from_intervals(pv_merged, "PV_WEATHER", "pv_sensors_zero_long"),
        events_from_intervals(building_merged, "BUILDING", "building_consensus_zero"),
        events_from_intervals(elx_merged, "ELX", "elx_single_zero"),
    ], ignore_index=True).sort_values(["start", "group"]).reset_index(drop=True)

    debug = {
        "freq": freq,
        "merge_gap_minutes": int(merge_gap_minutes),
        "zero_run_min_minutes": int(zero_run_min_minutes),
        "zero_run_min_samples": int(zero_run_min_samples),
        "bess": {**bess_debug, "num_intervals_merged": int(len(bess_merged))},
        "pv": {**pv_debug, "num_intervals_merged": int(len(pv_merged))},
        "building": {**building_debug, "num_intervals_merged": int(len(building_merged))},
        "elx": {**elx_debug, "num_intervals_merged": int(len(elx_merged))},
        "nanify_levels": list(nanify_levels),
        "events_total": int(len(events_all)),
    }

    return cleaned, events_all, reports, debug


# =========================================================
# Spike detection / spike cleaning
# =========================================================

def neighbor_spike_mask_1pt(x: pd.Series, *, jump_thr: float, neighbor_close_thr: float) -> pd.Series:
    prev = x.shift(1)
    nxt = x.shift(-1)

    ok_neighbors = (prev - nxt).abs().le(neighbor_close_thr)
    big_jump = (x - prev).abs().ge(jump_thr) & (x - nxt).abs().ge(jump_thr)

    mask = ok_neighbors & big_jump
    mask = mask & x.notna() & prev.notna() & nxt.notna()
    return mask.fillna(False)


def neighbor_spike_mask_2pt(x: pd.Series, *, jump_thr: float, outer_close_thr: float) -> pd.Series:
    xm1 = x.shift(1)
    x0 = x
    xp1 = x.shift(-1)
    xp2 = x.shift(-2)

    ok_outer = (xm1 - xp2).abs().le(outer_close_thr)

    big_jump_pair = (
        (x0 - xm1).abs().ge(jump_thr) & (x0 - xp2).abs().ge(jump_thr) &
        (xp1 - xm1).abs().ge(jump_thr) & (xp1 - xp2).abs().ge(jump_thr)
    )

    core = ok_outer & big_jump_pair
    core = core & xm1.notna() & x0.notna() & xp1.notna() & xp2.notna()
    core = core.fillna(False)

    return (core | core.shift(1, fill_value=False)).fillna(False)


def estimate_neighbor_spike_thresholds(
    x: pd.Series,
    *,
    q_jump: float = 0.995,
    q_neighbor: float = 0.90,
    min_jump: float = 0.0,
) -> tuple[float, float]:
    ramps = x.diff().abs().dropna()
    if ramps.empty:
        return float(min_jump), 0.0

    jump_thr = float(max(ramps.quantile(q_jump), min_jump))
    neigh = (x.shift(1) - x.shift(-1)).abs().dropna()
    neighbor_thr = float(neigh.quantile(q_neighbor)) if not neigh.empty else 0.0

    return jump_thr, neighbor_thr


def neighbor_spike_filter_df(
    df: pd.DataFrame,
    *,
    cols: list[str],
    auto_thresholds: bool = True,
    q_jump: float = 0.995,
    q_neighbor: float = 0.90,
    min_jump: float = 0.0,
    jump_thr: float | None = None,
    neighbor_close_thr: float | None = None,
    outer_close_thr: float | None = None,
    detect_1pt: bool = True,
    detect_2pt: bool = True,
    replace: str = "nan",
    interp_method: str = "time",
    interp_limit_direction: str = "both",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if replace not in ("nan", "interp"):
        raise ValueError("replace must be 'nan' or 'interp'")

    d = df.copy()
    used_cols = [c for c in cols if c in d.columns]

    if not used_cols:
        empty_mask = pd.DataFrame(index=d.index)
        empty_summary = pd.DataFrame(columns=[
            "column", "rows", "non_nan_before", "spikes",
            "spikes_%_rows", "spikes_%_non_nan",
            "jump_thr", "neighbor_close_thr", "outer_close_thr"
        ])
        return d, empty_mask, empty_summary

    if replace == "interp" and interp_method == "time":
        d = ensure_datetime_index(d)

    mask_df = pd.DataFrame(False, index=d.index, columns=used_cols)
    rows = []

    for c in used_cols:
        x = pd.to_numeric(d[c], errors="coerce")

        if auto_thresholds:
            jt, nt = estimate_neighbor_spike_thresholds(
                x, q_jump=q_jump, q_neighbor=q_neighbor, min_jump=min_jump
            )
            ot = nt if outer_close_thr is None else float(outer_close_thr)
        else:
            if jump_thr is None or neighbor_close_thr is None:
                raise ValueError("auto_thresholds=False requires jump_thr and neighbor_close_thr.")
            jt = float(jump_thr)
            nt = float(neighbor_close_thr)
            ot = float(outer_close_thr) if outer_close_thr is not None else nt

        m = pd.Series(False, index=d.index)
        if detect_1pt:
            m |= neighbor_spike_mask_1pt(x, jump_thr=jt, neighbor_close_thr=nt)
        if detect_2pt:
            m |= neighbor_spike_mask_2pt(x, jump_thr=jt, outer_close_thr=ot)

        m = m.fillna(False)
        mask_df[c] = m

        d.loc[m, c] = np.nan
        if replace == "interp":
            if interp_method == "time":
                d[c] = d[c].interpolate(method="time", limit_direction=interp_limit_direction)
            elif interp_method == "linear":
                d[c] = d[c].interpolate(method="linear", limit_direction=interp_limit_direction)
            else:
                raise ValueError("interp_method must be 'time' or 'linear'")
            d[c] = d[c].ffill().bfill()

        n_total = int(len(x))
        n_spikes = int(m.sum())
        n_non_nan = int(x.notna().sum())

        rows.append({
            "column": c,
            "rows": n_total,
            "non_nan_before": n_non_nan,
            "spikes": n_spikes,
            "spikes_%_rows": (n_spikes / n_total * 100.0) if n_total else np.nan,
            "spikes_%_non_nan": (n_spikes / n_non_nan * 100.0) if n_non_nan else np.nan,
            "jump_thr": jt,
            "neighbor_close_thr": nt,
            "outer_close_thr": ot,
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("spikes_%_non_nan", ascending=False).reset_index(drop=True)

    return d, mask_df, summary


def spike_summary_table(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    spike_mask_df: pd.DataFrame,
    spike_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cols = [c for c in spike_mask_df.columns if c in df_before.columns and c in df_after.columns]
    rows = []

    for c in cols:
        raw = pd.to_numeric(df_before[c], errors="coerce")
        clean = pd.to_numeric(df_after[c], errors="coerce")
        m = spike_mask_df[c].fillna(False).astype(bool)

        n_rows = int(len(raw))
        n_non_nan_before = int(raw.notna().sum())
        n_spikes = int(m.sum())
        new_nans = int((raw.notna() & clean.isna()).sum())

        rows.append({
            "column": c,
            "rows": n_rows,
            "non_nan_before": n_non_nan_before,
            "spikes_detected": n_spikes,
            "new_nans_introduced": new_nans,
            "spikes_%_of_rows": (n_spikes / n_rows * 100.0) if n_rows else np.nan,
            "spikes_%_of_non_nan": (n_spikes / n_non_nan_before * 100.0) if n_non_nan_before else np.nan,
        })

    out = pd.DataFrame(rows)

    if spike_summary is not None and not spike_summary.empty and "column" in spike_summary.columns:
        keep = [c for c in ["column", "jump_thr", "neighbor_close_thr", "outer_close_thr"] if c in spike_summary.columns]
        out = out.merge(spike_summary[keep], on="column", how="left")

    if not out.empty:
        out = out.sort_values("spikes_%_of_non_nan", ascending=False).reset_index(drop=True)

    return out


def run_spike_stage(
    df_in: pd.DataFrame,
    *,
    spike_cols: list[str],
    q_jump: float = 0.995,
    q_neighbor: float = 0.90,
    replace: str = "nan",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_before = df_in.copy()
    df_after, spike_mask, spike_summary = neighbor_spike_filter_df(
        df_before,
        cols=spike_cols,
        auto_thresholds=True,
        q_jump=q_jump,
        q_neighbor=q_neighbor,
        detect_1pt=True,
        detect_2pt=True,
        replace=replace,
    )
    spike_table = spike_summary_table(df_before, df_after, spike_mask, spike_summary)
    return df_after, spike_mask, spike_summary, spike_table


# =========================================================
# Additional rule-based cleaning
# =========================================================

def clean_soc_bounds(
    df: pd.DataFrame,
    *,
    soc_col: str = "BA_Soc",
    low: float = 1.0,
    high: float = 100.0,
) -> tuple[pd.DataFrame, pd.Series]:
    out = df.copy()
    if soc_col not in out.columns:
        return out, pd.Series(False, index=out.index)

    out[soc_col] = pd.to_numeric(out[soc_col], errors="coerce")
    invalid_mask = (out[soc_col] < low) | (out[soc_col] > high)
    out.loc[invalid_mask, soc_col] = np.nan
    return out, invalid_mask


# =========================================================
# Summary helpers
# =========================================================

def summary_removed_percent_by_group(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    *,
    pv_cols: list[str],
    bess_cols: list[str],
    building_cols: list[str],
    elx_cols: list[str],
) -> pd.DataFrame:
    groups = {
        "PV_WEATHER": pv_cols,
        "BESS": bess_cols,
        "BUILDING": building_cols,
        "ELX": elx_cols,
    }

    out = []
    for g, cols in groups.items():
        cols = [c for c in cols if c in df_raw.columns and c in df_clean.columns]
        if not cols:
            continue

        raw = df_raw[cols].apply(pd.to_numeric, errors="coerce")
        clean = df_clean[cols].apply(pd.to_numeric, errors="coerce")

        denom = int(raw.notna().sum().sum())
        removed = int((raw.notna() & clean.isna()).sum().sum())

        out.append({
            "group": g,
            "cols": len(cols),
            "raw_non_nan_points": denom,
            "new_nan_points_added": removed,
            "removed_%": (removed / denom * 100.0) if denom else np.nan,
        })

    return pd.DataFrame(out).sort_values("removed_%", ascending=False).reset_index(drop=True)


# =========================================================
# High-level cleaning pipeline
# =========================================================

def run_cleaning_pipeline(
    df: pd.DataFrame,
    *,
    pre_drop_cols: list[str] | None = None,
    pv_sensor_cols: list[str] | None = None,
    bess_validate_cols: list[str] | None = None,
    building_detect_cols: list[str] | None = None,
    building_validate_cols: list[str] | None = None,
    elx_validate_cols: list[str] | None = None,
    spike_cols: list[str] | None = None,
    final_drop_cols: list[str] | None = None,
) -> dict:
    d0 = ensure_datetime_index(df)

    if pre_drop_cols is None:
        pre_drop_cols = DEFAULT_PRE_DROP_COLS
    if pv_sensor_cols is None:
        pv_sensor_cols = PV_SENSOR_COLS
    if bess_validate_cols is None:
        bess_validate_cols = BESS_VALIDATE_COLS
    if building_detect_cols is None:
        building_detect_cols = BUILDING_DETECT_COLS
    if building_validate_cols is None:
        building_validate_cols = BUILDING_VALIDATE_COLS
    if elx_validate_cols is None:
        elx_validate_cols = ELX_VALIDATE_COLS
    if final_drop_cols is None:
        final_drop_cols = DEFAULT_DROP_COLS

    d1 = drop_columns_if_exist(d0, pre_drop_cols)

    df_event_clean, events_all, reports, debug = full_comm_loss_maintenance_pipeline(
        d1,
        freq="5min",
        merge_gap_minutes=30,

        bess_cols_validate_nanify=bess_validate_cols,
        bess_tol_zero_limits=0.0,
        bess_min_minutes=30,

        pv_sensor_cols=pv_sensor_cols,
        pv_min_minutes=30,
        pv_zero_tol=0.0,
        pv_min_cols_fraction=0.75,
        pv_pressure_col="PV_WS_RelAirPre",
        pv_pressure_lo=800.0,
        pv_pressure_hi=1100.0,
        pv_pressure_other_fraction=1.0,

        building_cols_detect=building_detect_cols,
        building_cols_validate_nanify=building_validate_cols,
        building_min_minutes=30,
        building_zero_tol=0.0,
        building_min_cols_fraction=0.75,

        elx_detect_col="EL_TotActPwr_NitrogenUnit",
        elx_cols_validate_nanify=elx_validate_cols,
        elx_min_minutes=30,
        elx_zero_tol=0.0,

        zero_tol_signals=0.0,
        high_thr=0.95,
        med_thr=0.85,
        probable_min_hours=2.0,
        change_rate_thr=0.01,
        min_non_nan=10,
        nanify_levels=("definite", "probable"),
        zero_run_min_minutes=60,
    )

    cleaned_cols = sorted(set(
        pv_sensor_cols + bess_validate_cols + building_validate_cols + elx_validate_cols
    ))
    cleaned_cols = [c for c in cleaned_cols if c in d1.columns and c in df_event_clean.columns]
    d2 = apply_cleaned_columns(d1, df_event_clean, cleaned_cols)

    if spike_cols is None:
        spike_cols = [
            "PV_WS_Radiation",
            "PV_WS_AirTemp",
            "PV_WS_RelHum",
            "BA_Soc",
            "BA_TotActPwr_BESS_AC_Panel2",
            "BA_TotActPwr_BESS_AC_Panel1",
            "BU_TotActPwr_Tech_Room",
            "BU_TotActPwr_SDB_EL_Substation",
            "BU_TotActPwr_Academy",
            "BU_PwrReq",
            "EL_TotActPwr_NitrogenUnit",
        ]

    spike_cols = [c for c in spike_cols if c in d2.columns]

    d3, spike_mask, spike_summary, spike_table = run_spike_stage(
        d2,
        spike_cols=spike_cols,
        q_jump=0.995,
        q_neighbor=0.90,
        replace="nan",
    )

    d4, soc_invalid_mask = clean_soc_bounds(d3, soc_col="BA_Soc", low=1.0, high=100.0)

    summary_df = summary_removed_percent_by_group(
        d1,
        d4,
        pv_cols=pv_sensor_cols,
        bess_cols=bess_validate_cols,
        building_cols=building_validate_cols,
        elx_cols=elx_validate_cols,
    )

    df_final = drop_columns_if_exist(d4, final_drop_cols)

    return {
        "df_input": d0,
        "df_after_pre_drop": d1,
        "df_event_clean": d2,
        "df_after_spike": d3,
        "df_final": df_final,
        "events_all": events_all,
        "reports": reports,
        "debug": debug,
        "spike_mask": spike_mask,
        "spike_summary": spike_summary,
        "spike_table": spike_table,
        "summary_removed": summary_df,
        "soc_invalid_mask": soc_invalid_mask,
        "cleaned_cols": cleaned_cols,
        "spike_cols": spike_cols,
    }


def save_cleaned_csv(
    df: pd.DataFrame,
    save_path: str | Path,
    *,
    float_format: str = "%.5f",
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        save_path,
        index=True,
        float_format=float_format,
        date_format=date_format,
    )