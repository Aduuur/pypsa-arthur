#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_demand_csv_coverage_2025_2050.py

Geht alle Ordner /home/endata/PycharmProjects/pypsa-ee/resources/demand_20XX (2025-2050) durch
und prüft electricity_demand_non_hist.csv auf:
- Existenz der Datei
- Zeitindex vollständig (jede Stunde des Jahres; leap years berücksichtigt)
- Spalten vorhanden (CH,DE,ES,FR,GB,IT,PL)
- pro Land: fehlende Stunden / NaNs / nicht-numerische Werte / Duplikate

Speichert "fehlerhafte Zeitreihen" (pro Jahr+Land mit Problemen) als CSV in einem Output-Ordner
und schreibt eine Summary-CSV + druckt eine kompakte Zusammenfassung.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = Path("/home/endata/PycharmProjects/pypsa-ee/resources")
YEAR_START = 2025
YEAR_END = 2050

DEMAND_DIR_FMT = "demand_{year}"
FILENAME = "electricity_demand_non_hist.csv"  # exakt wie in deinen Ordnern (case-sensitive)

COUNTRIES = ["CH", "DE", "ES", "FR", "GB", "IT", "PL"]

OUT_DIR = BASE_DIR / "_demand_checks_2025_2050"
BAD_SERIES_DIR = OUT_DIR / "bad_series"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BAD_SERIES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================
def is_leap_year(year: int) -> bool:
    return pd.Timestamp(year=year, month=12, day=31).is_leap_year


def expected_hourly_index(year: int) -> pd.DatetimeIndex:
    # hourly grid for full year in UTC-naive timestamps
    start = pd.Timestamp(year=year, month=1, day=1, hour=0)
    end = pd.Timestamp(year=year, month=12, day=31, hour=23)
    return pd.date_range(start=start, end=end, freq="h")


def load_csv_timeindex(csv_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Robust load: find time column 'Date' etc. else first column.
    Returns df indexed by time and name of time column used.
    """
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("CSV is empty")

    time_col = None
    for cand in ["Date", "date", "time", "Time", "timestamp", "Timestamp"]:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        time_col = df.columns[0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    n_bad_time = int(df[time_col].isna().sum())
    if n_bad_time > 0:
        # keep rows with valid timestamps; report later via metric
        df = df.dropna(subset=[time_col])

    df = df.set_index(time_col)
    df.index.name = "time"
    df = df.sort_index()

    return df, time_col


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


# =============================================================================
# CHECK LOGIC
# =============================================================================
def check_year_file(year: int, csv_path: Path) -> Dict:
    """
    Checks the whole CSV file and each country series.
    Returns a dict row for summary.
    Also writes bad series csvs for any country+year with issues.
    """
    row: Dict = {
        "year": year,
        "csv_path": str(csv_path),
        "file_exists": csv_path.exists(),
        "file_empty": None,
        "time_col": None,
        "n_rows_raw": None,
        "n_unique_timestamps": None,
        "n_duplicate_timestamps": None,
        "n_expected_hours": len(expected_hourly_index(year)),
        "n_missing_hours_any": None,
        "missing_hours_any": None,  # stringified list (truncated)
        "missing_countries": None,
        "countries_ok": 0,
        "countries_bad": 0,
        "bad_countries_list": None,
        "notes": None,
    }

    if not csv_path.exists():
        row["file_empty"] = None
        row["notes"] = "missing_file"
        return row

    try:
        df, time_col = load_csv_timeindex(csv_path)
        row["time_col"] = time_col
        row["n_rows_raw"] = int(df.shape[0])
        row["file_empty"] = bool(df.empty)
        if df.empty:
            row["notes"] = "empty_after_time_parse"
            return row
    except Exception as e:
        row["notes"] = f"read_error:{type(e).__name__}:{e}"
        return row

    # Duplicates in timestamp index
    n_dups = int(df.index.duplicated(keep=False).sum())
    row["n_duplicate_timestamps"] = n_dups
    row["n_unique_timestamps"] = int(df.index.nunique())

    # Expected full-year hourly grid
    expected_idx = expected_hourly_index(year)
    present_idx = pd.DatetimeIndex(df.index.unique()).sort_values()

    missing_idx = expected_idx.difference(present_idx)
    row["n_missing_hours_any"] = int(len(missing_idx))
    if len(missing_idx) > 0:
        # store a truncated representation
        sample = list(map(str, missing_idx[:25]))
        suffix = "" if len(missing_idx) <= 25 else f"... (+{len(missing_idx)-25} more)"
        row["missing_hours_any"] = ";".join(sample) + suffix
    else:
        row["missing_hours_any"] = ""

    # Countries presence
    missing_cols = [c for c in COUNTRIES if c not in df.columns]
    row["missing_countries"] = ",".join(missing_cols) if missing_cols else ""

    bad_countries: List[str] = []
    ok_countries = 0

    # Align to expected grid for per-country checks
    # If duplicates exist, aggregate duplicates by mean (but mark as problem)
    df_aligned = df.copy()
    if n_dups > 0:
        # groupby index and mean for numeric cols; keep everything numeric-friendly
        df_aligned = df_aligned.groupby(df_aligned.index).mean(numeric_only=True)

    # reindex to full year grid
    df_aligned = df_aligned.reindex(expected_idx)

    # Per-country checks
    for c in COUNTRIES:
        if c not in df.columns:
            bad_countries.append(c)
            # save a stub "bad series" file with reason
            out = pd.DataFrame(index=expected_idx)
            out.index.name = "time"
            out["reason"] = "missing_column"
            out_path = BAD_SERIES_DIR / f"bad_{year}_{c}_missing_column.csv"
            out.to_csv(out_path)
            continue

        s_raw = df[c]
        s = df_aligned[c] if c in df_aligned.columns else df_aligned.get(c)

        # numeric coercion
        s_num = coerce_numeric_series(s)

        n_total = int(len(expected_idx))
        n_missing_time = int(s.isna().sum())  # due to missing timestamps or missing values
        n_nan_numeric = int(s_num.isna().sum())  # includes non-numeric cells too

        mask_present = ~pd.isna(s)
        mask_non_numeric = mask_present & pd.isna(s_num)
        n_non_numeric = int(mask_non_numeric.sum())

        # additional diagnostics
        # detect if index doesn't start/end correctly in raw
        raw_start = str(pd.DatetimeIndex(s_raw.index).min()) if len(s_raw.index) else ""
        raw_end = str(pd.DatetimeIndex(s_raw.index).max()) if len(s_raw.index) else ""

        has_problem = False
        reasons: List[str] = []

        if row["n_missing_hours_any"] and row["n_missing_hours_any"] > 0:
            has_problem = True
            reasons.append("missing_timestamps_in_file")

        if n_dups > 0:
            has_problem = True
            reasons.append("duplicate_timestamps_in_file")

        # per series: missing or non-numeric
        if n_missing_time > 0:
            has_problem = True
            reasons.append(f"missing_values_or_timestamps:{n_missing_time}")

        if n_non_numeric > 0:
            has_problem = True
            reasons.append(f"non_numeric_values:{n_non_numeric}")

        # sanity: should be all present and numeric
        if not has_problem:
            ok_countries += 1
            continue

        bad_countries.append(c)

        # Write bad series file with detailed columns
        bad_df = pd.DataFrame(index=expected_idx)
        bad_df.index.name = "time"
        bad_df["value_raw_aligned"] = s.values
        bad_df["value_numeric"] = s_num.values
        bad_df["is_missing"] = pd.isna(s).values
        bad_df["is_non_numeric"] = ((~pd.isna(s)) & pd.isna(s_num)).values

        # Mark missing timestamps explicitly (same for all cols)
        if len(missing_idx) > 0:
            miss_mask = bad_df.index.isin(missing_idx)
            bad_df["missing_timestamp_in_file"] = miss_mask
        else:
            bad_df["missing_timestamp_in_file"] = False

        bad_df.attrs["reasons"] = ", ".join(reasons)

        out_path = BAD_SERIES_DIR / f"bad_{year}_{c}.csv"
        bad_df.to_csv(out_path)

    row["countries_ok"] = int(ok_countries)
    row["countries_bad"] = int(len(set(bad_countries)))
    row["bad_countries_list"] = ",".join(sorted(set(bad_countries)))

    return row


def main() -> None:
    rows: List[Dict] = []
    total_files = 0
    missing_files = 0

    for year in range(YEAR_START, YEAR_END + 1):
        csv_path = BASE_DIR / DEMAND_DIR_FMT.format(year=year) / FILENAME
        total_files += 1
        if not csv_path.exists():
            missing_files += 1
        rows.append(check_year_file(year, csv_path))

    summary = pd.DataFrame(rows).sort_values("year")

    # Overall stats
    n_years = summary.shape[0]
    n_ok_years = int(((summary["file_exists"] == True) &
                      (summary["countries_bad"] == 0) &
                      (summary["n_missing_hours_any"] == 0) &
                      ((summary["n_duplicate_timestamps"].fillna(0) == 0))).sum())

    # Save summary
    summary_path = OUT_DIR / "summary_2025_2050.csv"
    summary.to_csv(summary_path, index=False)

    # Print compact report
    print("=" * 80)
    print("DEMAND CHECK REPORT (2025-2050)")
    print(f"Base dir: {BASE_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print("-" * 80)
    print(f"Years checked: {n_years}")
    print(f"Missing files: {missing_files}")
    print(f"Fully OK years (file exists, no missing hours, no dups, all countries ok): {n_ok_years}")
    print(f"Summary CSV: {summary_path}")
    print(f"Bad series dir: {BAD_SERIES_DIR}")
    print("=" * 80)

    # Show problematic years overview
    problems = summary[
        (summary["file_exists"] != True)
        | (summary["countries_bad"].fillna(0) > 0)
        | (summary["n_missing_hours_any"].fillna(0) > 0)
        | (summary["n_duplicate_timestamps"].fillna(0) > 0)
        | (summary["missing_countries"].fillna("") != "")
        | (summary["notes"].fillna("") != "")
    ].copy()

    if problems.empty:
        print("✅ No problems detected.")
        return

    print("⚠️ Problem years overview (first 50 rows):")
    cols_show = [
        "year",
        "file_exists",
        "notes",
        "missing_countries",
        "n_missing_hours_any",
        "n_duplicate_timestamps",
        "countries_ok",
        "countries_bad",
        "bad_countries_list",
    ]
    print(problems[cols_show].head(50).to_string(index=False))

    # Also save problem-only view
    problems_path = OUT_DIR / "problems_only_2025_2050.csv"
    problems.to_csv(problems_path, index=False)
    print("-" * 80)
    print(f"Problems-only CSV: {problems_path}")


if __name__ == "__main__":
    main()
