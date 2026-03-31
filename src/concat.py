from pathlib import Path
import pandas as pd
import csv
import sys


def find_header_row(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.strip().startswith("Time"):
                return i
    raise ValueError(f"Header line starting with 'Time' not found in: {path}")


def parse_time_mixed(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()

    t1 = pd.to_datetime(s, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    t2 = pd.to_datetime(s, format="%m/%d/%Y %H:%M", errors="coerce")
    t3 = pd.to_datetime(s, format="%m/%d/%Y %H:%M:%S", errors="coerce")
    t4 = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    t5 = pd.to_datetime(s, format="%Y-%m-%d %H:%M", errors="coerce")

    out = t1.fillna(t2).fillna(t3).fillna(t4).fillna(t5)

    if out.isna().any():
        out = out.fillna(pd.to_datetime(s, errors="coerce"))

    return out


def robust_read_csv_first_two_cols(csv_path: Path, sep: str = ",") -> pd.DataFrame:
    header_row = find_header_row(csv_path)

    try:
        df = pd.read_csv(
            csv_path,
            sep=sep,
            header=header_row,
            engine="c",
            usecols=[0, 1],
            encoding="utf-8",
        )
        return df
    except Exception:
        csv.field_size_limit(min(sys.maxsize, 10_000_000))
        df = pd.read_csv(
            csv_path,
            sep=sep,
            header=header_row,
            engine="python",
            usecols=[0, 1],
            encoding="utf-8",
            on_bad_lines="skip",
        )
        return df


def read_year(signal_folder: Path, year: str) -> tuple[pd.DataFrame, str]:
    csv_path = signal_folder / f"{year}.csv"
    df = robust_read_csv_first_two_cols(csv_path, sep=",")

    df.columns = [c.strip() for c in df.columns]

    time_col = df.columns[0]
    sig_col = df.columns[1]

    df = df.rename(columns={time_col: "Time"})
    df["Time"] = parse_time_mixed(df["Time"])
    df[sig_col] = pd.to_numeric(df[sig_col], errors="coerce")

    df = df.dropna(subset=["Time"]).sort_values("Time").drop_duplicates("Time", keep="last")

    return df[["Time", sig_col]], sig_col


def read_signal(signal_folder: Path, years: list[str] | None = None, save_per_signal: bool = False) -> tuple[pd.DataFrame, str] | None:
    if years is None:
        years = ["2025", "2026"]

    dfs = []
    sig_name = None

    for year in years:
        csv_path = signal_folder / f"{year}.csv"
        if csv_path.exists():
            df_year, this_sig_name = read_year(signal_folder, year)

            if sig_name is None:
                sig_name = this_sig_name
            elif sig_name != this_sig_name:
                df_year = df_year.rename(columns={this_sig_name: sig_name})

            dfs.append(df_year)

    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True).sort_values("Time")
    df_all = df_all.drop_duplicates("Time", keep="last")

    if save_per_signal:
        out_path = signal_folder / f"{sig_name}_combined.csv"
        df_all.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Saved per-signal: {out_path}")

    return df_all, sig_name


def build_combined_dataset(
    root: Path,
    out_path: Path | None = None,
    years: list[str] | None = None,
    save_per_signal: bool = False,
) -> pd.DataFrame:
    signal_folders = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    )

    combined = None
    used_names = set()
    skipped = []

    for i, folder in enumerate(signal_folders, start=1):
        try:
            res = read_signal(folder, years=years, save_per_signal=save_per_signal)

            if res is None:
                skipped.append(folder.name)
                continue

            df_sig, sig_name = res

            final_name = sig_name
            if final_name in used_names:
                final_name = f"{sig_name}__{folder.name}"
                df_sig = df_sig.rename(columns={sig_name: final_name})

            used_names.add(final_name)

            if combined is None:
                combined = df_sig
            else:
                combined = combined.merge(df_sig, on="Time", how="outer")

            print(f"Processed {i}/{len(signal_folders)}: {folder.name} -> {final_name}")

        except Exception as e:
            skipped.append(folder.name)
            print(f"Skipped {folder.name} because: {e}")

    if combined is None:
        raise RuntimeError("No signal folders were successfully processed.")

    combined = combined.sort_values("Time")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False, encoding="utf-8")
        print("\nDONE :)")
        print(f"Saved big file: {out_path}")

    print(f"Final shape: {combined.shape}")
    if skipped:
        print(f"Skipped folders: {skipped}")

    return combined