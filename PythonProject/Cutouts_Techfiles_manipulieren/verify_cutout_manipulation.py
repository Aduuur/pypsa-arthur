#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import atlite


# =============================================================================
# KONFIGURATION
# =============================================================================

# -------------------------------------------------------------------------
# Basis-Cutout
# -------------------------------------------------------------------------
BASE_CUTOUT_PATH = "/home/endata/cutouts/cutout_mCNRM-CERFACS-CM5_rcp45_2028.nc"

# -------------------------------------------------------------------------
# Ordner mit den manipulierten Cutouts
# Erwartetes Dateimuster:
#   <base_stem>__<case_name>.nc
# z. B.
#   cutout_mCNRM-CERFACS-CM5_rcp45_2028__stress_01_from_2031_12_06.nc
# -------------------------------------------------------------------------
MANIPULATED_DIR = "/home/endata/cutouts_manipulated/"

# -------------------------------------------------------------------------
# Ausgabeordner für Verifikationsergebnisse
# -------------------------------------------------------------------------
VERIFY_OUTPUT_DIR = "/home/endata/cutouts_manipulated_verification/"
os.makedirs(VERIFY_OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Ereignislänge und Wiederholungen
# -------------------------------------------------------------------------
EVENT_DURATION = pd.Timedelta(weeks=1)
REPETITIONS = 1

# -------------------------------------------------------------------------
# Numerische Toleranzen
# -------------------------------------------------------------------------
ABS_TOL = 1e-6
REL_TOL = 1e-8

# -------------------------------------------------------------------------
# Wenn True:
# - außerhalb des Ereignisfensters muss das manipulierte Cutout exakt dem
#   Basis-Cutout entsprechen
# -------------------------------------------------------------------------
CHECK_OUTSIDE_EVENT_EQUALS_BASE = True

# -------------------------------------------------------------------------
# Deine Stressfälle
# -------------------------------------------------------------------------
STRESS_CASES: List[Dict[str, Any]] = [
    {
        "name": "stress_01_from_2031_12_06",
        "source_cutout_path": "/home/endata/cutouts/cutout_mCNRM-CERFACS-CM5_rcp45_2031.nc",
        "source_week_start_str": "2031-12-06",
    },
    {
        "name": "stress_02_from_2030_11_23",
        "source_cutout_path": "/home/endata/cutouts/cutout_mCNRM-CERFACS-CM5_rcp45_2030.nc",
        "source_week_start_str": "2030-11-23",
    },
    {
        "name": "stress_03_from_2035_11_22",
        "source_cutout_path": "/home/endata/cutouts/cutout_mCNRM-CERFACS-CM5_rcp45_2035.nc",
        "source_week_start_str": "2035-11-22",
    },
    {
        "name": "stress_04_from_2040_12_12",
        "source_cutout_path": "/home/endata/cutouts/cutout_mCNRM-CERFACS-CM5_rcp45_2040.nc",
        "source_week_start_str": "2040-12-12",
    },
    {
        "name": "stress_05_from_2033_11_21",
        "source_cutout_path": "/home/endata/cutouts/cutout_mCNRM-CERFACS-CM5_rcp45_2033.nc",
        "source_week_start_str": "2033-11-21",
    },
    {
        "name": "stress_06_from_2047_01_07",
        "source_cutout_path": "/home/endata/cutouts/cutout_mCNRM-CERFACS-CM5_rcp45_2047.nc",
        "source_week_start_str": "2047-01-07",
    },
    {
        "name": "stress_07_from_2041_11_21",
        "source_cutout_path": "/home/endata/cutouts/cutout_mCNRM-CERFACS-CM5_rcp45_2041.nc",
        "source_week_start_str": "2041-11-21",
    },
    {
        "name": "stress_08_from_2047_11_23",
        "source_cutout_path": "//home/endata/cutouts/cutout_mCNRM-CERFACS-CM5_rcp45_2047.nc",
        "source_week_start_str": "2047-11-23",
    },
    {
        "name": "stress_09_from_2039_12_13",
        "source_cutout_path": "/home/endata/cutouts/cutout_mCNRM-CERFACS-CM5_rcp45_2039.nc",
        "source_week_start_str": "2039-12-13",
    },
    {
        "name": "stress_10_from_2041_01_30",
        "source_cutout_path": "/home/endata/cutouts/cutout_mCNRM-CERFACS-CM5_rcp45_2041.nc",
        "source_week_start_str": "2041-01-30",
    },
]


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================
def normalize_path(p: str) -> str:
    return os.path.normpath(p)


def build_manipulated_cutout_path(base_cutout_path: str, case_name: str, manipulated_dir: str) -> str:
    base_name = Path(base_cutout_path).name
    return os.path.join(manipulated_dir, f"{base_name}__{case_name}.nc")


def infer_target_year_from_cutout(ds_target: xr.Dataset) -> int:
    if ds_target.sizes.get("time", 0) == 0:
        raise ValueError("Basis-Cutout hat keine Zeitschritte.")
    return pd.to_datetime(ds_target.time.values[0]).year


def get_time_step(ds: xr.Dataset) -> pd.Timedelta:
    times = pd.to_datetime(ds.time.values)
    if len(times) < 2:
        raise ValueError("Zu wenige Zeitschritte zur Bestimmung der Auflösung.")
    diffs = pd.Series(times[1:] - times[:-1])
    step = diffs.mode().iloc[0]
    if not isinstance(step, pd.Timedelta):
        step = pd.Timedelta(step)
    return step


def map_source_start_to_target_year(source_start_date: pd.Timestamp, target_year: int) -> pd.Timestamp:
    try:
        return source_start_date.replace(year=target_year)
    except ValueError as e:
        raise ValueError(
            f"Kann Quellzeitpunkt {source_start_date} nicht auf Zieljahr {target_year} abbilden."
        ) from e


def get_time_dependent_vars(ds: xr.Dataset) -> List[str]:
    return [v for v, da in ds.data_vars.items() if "time" in da.dims]


def get_static_vars(ds: xr.Dataset) -> List[str]:
    return [v for v, da in ds.data_vars.items() if "time" not in da.dims]


def compare_dataarrays(
    da_a: xr.DataArray,
    da_b: xr.DataArray,
    abs_tol: float,
    rel_tol: float,
) -> Dict[str, Any]:
    """
    Vergleicht zwei DataArrays robust und liefert Diagnosekennzahlen.
    """
    if tuple(da_a.dims) != tuple(da_b.dims):
        return {
            "equal": False,
            "reason": f"dims mismatch: {da_a.dims} vs {da_b.dims}",
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "n_diff": None,
            "n_total": None,
        }

    a = da_a.values
    b = da_b.values

    if a.shape != b.shape:
        return {
            "equal": False,
            "reason": f"shape mismatch: {a.shape} vs {b.shape}",
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "n_diff": None,
            "n_total": None,
        }

    finite_mask = np.isfinite(a) & np.isfinite(b)
    nan_mask_equal = np.isnan(a) == np.isnan(b)

    if not np.all(nan_mask_equal):
        return {
            "equal": False,
            "reason": "NaN pattern differs",
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "n_diff": int(np.size(nan_mask_equal) - np.count_nonzero(nan_mask_equal)),
            "n_total": int(np.size(nan_mask_equal)),
        }

    diff = np.abs(a - b)
    allowed = abs_tol + rel_tol * np.abs(b)

    # NaN-Stellen ausblenden
    valid = finite_mask
    diff_valid = diff[valid]
    allowed_valid = allowed[valid]

    if diff_valid.size == 0:
        return {
            "equal": True,
            "reason": "all values NaN but pattern equal",
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "n_diff": 0,
            "n_total": int(np.size(a)),
        }

    neq_mask = diff_valid > allowed_valid
    n_diff = int(np.count_nonzero(neq_mask))

    return {
        "equal": n_diff == 0,
        "reason": "ok" if n_diff == 0 else "values differ",
        "max_abs_diff": float(np.max(diff_valid)),
        "mean_abs_diff": float(np.mean(diff_valid)),
        "n_diff": n_diff,
        "n_total": int(diff_valid.size),
    }


def validate_case_metadata(case: Dict[str, Any]) -> List[str]:
    """
    Prüft offensichtliche Inkonsistenzen im Fall-Metadatenblock.
    """
    warnings: List[str] = []

    case_name = case["name"]
    source_cutout_path = normalize_path(case["source_cutout_path"])
    source_week_start_str = case["source_week_start_str"]

    if "//" in case["source_cutout_path"] and not case["source_cutout_path"].startswith("http"):
        warnings.append("source_cutout_path enthält doppelten Slash; wird per normpath normalisiert.")

    try:
        parsed_date = pd.to_datetime(source_week_start_str)
    except Exception as e:
        warnings.append(f"source_week_start_str konnte nicht geparst werden: {e}")
        return warnings

    # Prüfe, ob das im Namen codierte Datum grob zum Datumsstring passt
    # z. B. stress_10_from_2041_01_30 vs 2041-01-3
    if "_from_" in case_name:
        suffix = case_name.split("_from_")[-1]
        try:
            name_date = pd.to_datetime(suffix.replace("_", "-"))
            if name_date.date() != parsed_date.date():
                warnings.append(
                    f"Namens-/Datums-Mismatch: case_name deutet auf {name_date.date()} hin, "
                    f"source_week_start_str aber auf {parsed_date.date()}."
                )
        except Exception:
            warnings.append("Datum im case_name konnte nicht geprüft werden.")

    # Prüfe, ob das Jahr im Pfad grob passt
    src_path_name = Path(source_cutout_path).name
    years_in_name = [tok for tok in src_path_name.replace(".", "_").split("_") if tok.isdigit() and len(tok) == 4]
    if years_in_name:
        try:
            path_year = int(years_in_name[-1])
            if path_year != parsed_date.year:
                warnings.append(
                    f"Pfad-/Datums-Mismatch: source_cutout_path deutet auf Jahr {path_year}, "
                    f"source_week_start_str aber auf Jahr {parsed_date.year}."
                )
        except Exception:
            pass

    return warnings


def verify_single_case(
    base_cutout_path: str,
    manipulated_cutout_path: str,
    source_cutout_path: str,
    source_week_start_str: str,
    repetitions: int,
    event_duration: pd.Timedelta,
    abs_tol: float,
    rel_tol: float,
    check_outside_equals_base: bool,
) -> Dict[str, Any]:
    """
    Verifiziert genau einen Fall.
    """
    result: Dict[str, Any] = {
        "status": "FAIL",
        "message": "",
        "base_cutout_path": base_cutout_path,
        "manipulated_cutout_path": manipulated_cutout_path,
        "source_cutout_path": source_cutout_path,
        "source_week_start_str": source_week_start_str,
        "checks": {},
        "warnings": [],
    }

    base_cutout_path = normalize_path(base_cutout_path)
    manipulated_cutout_path = normalize_path(manipulated_cutout_path)
    source_cutout_path = normalize_path(source_cutout_path)

    if not os.path.exists(base_cutout_path):
        result["message"] = f"Basis-Cutout nicht gefunden: {base_cutout_path}"
        return result

    if not os.path.exists(source_cutout_path):
        result["message"] = f"Quell-Cutout nicht gefunden: {source_cutout_path}"
        return result

    if not os.path.exists(manipulated_cutout_path):
        result["message"] = f"Manipuliertes Cutout nicht gefunden: {manipulated_cutout_path}"
        return result

    try:
        source_start_date = pd.to_datetime(source_week_start_str)
    except Exception as e:
        result["message"] = f"source_week_start_str ungültig: {e}"
        return result

    base_cutout = atlite.Cutout(path=base_cutout_path)
    src_cutout = atlite.Cutout(path=source_cutout_path)
    manip_cutout = atlite.Cutout(path=manipulated_cutout_path)

    ds_base = base_cutout.data
    ds_source = src_cutout.data
    ds_manip = manip_cutout.data

    try:
        target_year = infer_target_year_from_cutout(ds_base)
        time_step = get_time_step(ds_base)

        source_end_date = source_start_date + event_duration - time_step
        target_start_date = map_source_start_to_target_year(source_start_date, target_year)
        target_end_date = target_start_date + (event_duration * repetitions) - time_step

        result["derived_target_start"] = str(target_start_date)
        result["derived_target_end"] = str(target_end_date)
        result["time_step"] = str(time_step)

        # -------------------------------------------------------------
        # 1) Strukturcheck
        # -------------------------------------------------------------
        checks = {}

        checks["same_time_length_as_base"] = bool(len(ds_manip.time) == len(ds_base.time))
        checks["same_data_vars_as_base"] = bool(set(ds_manip.data_vars) == set(ds_base.data_vars))
        checks["same_x_size_as_base"] = bool(ds_manip.sizes.get("x") == ds_base.sizes.get("x"))
        checks["same_y_size_as_base"] = bool(ds_manip.sizes.get("y") == ds_base.sizes.get("y"))

        # -------------------------------------------------------------
        # 2) Ereignisfenster extrahieren
        # -------------------------------------------------------------
        time_vars = get_time_dependent_vars(ds_base)

        source_event = ds_source[time_vars].sel(time=slice(source_start_date, source_end_date))
        if source_event.sizes.get("time", 0) == 0:
            result["message"] = (
                f"Quellereignis leer: {source_start_date} bis {source_end_date}"
            )
            result["checks"] = checks
            return result

        expected_event = xr.concat([source_event] * repetitions, dim="time")

        target_event_time = ds_base.sel(time=slice(target_start_date, target_end_date)).time
        if len(target_event_time) == 0:
            result["message"] = (
                f"Zielereignisfenster im Basis-Cutout leer: {target_start_date} bis {target_end_date}"
            )
            result["checks"] = checks
            return result

        if len(expected_event.time) != len(target_event_time):
            result["message"] = (
                f"Längenproblem im Ereignisfenster: expected={len(expected_event.time)}, "
                f"target={len(target_event_time)}"
            )
            result["checks"] = checks
            return result

        expected_event = expected_event.assign_coords(time=target_event_time)
        actual_event = ds_manip[time_vars].sel(time=slice(target_start_date, target_end_date))

        # -------------------------------------------------------------
        # 3) Vergleich innerhalb des Ereignisses
        # -------------------------------------------------------------
        per_var_event_checks = {}
        event_all_ok = True

        for var in time_vars:
            cmp_res = compare_dataarrays(
                actual_event[var],
                expected_event[var],
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
            per_var_event_checks[var] = cmp_res
            if not cmp_res["equal"]:
                event_all_ok = False

        checks["event_window_matches_source"] = event_all_ok
        result["per_var_event_checks"] = per_var_event_checks

        # -------------------------------------------------------------
        # 4) Vergleich außerhalb des Ereignisses
        # -------------------------------------------------------------
        if check_outside_equals_base:
            before_actual = ds_manip[time_vars].sel(time=slice(None, target_start_date - time_step))
            before_base = ds_base[time_vars].sel(time=slice(None, target_start_date - time_step))

            after_actual = ds_manip[time_vars].sel(time=slice(target_end_date + time_step, None))
            after_base = ds_base[time_vars].sel(time=slice(target_end_date + time_step, None))

            per_var_outside_checks = {}
            outside_all_ok = True

            for var in time_vars:
                cmp_before = compare_dataarrays(
                    before_actual[var],
                    before_base[var],
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                )
                cmp_after = compare_dataarrays(
                    after_actual[var],
                    after_base[var],
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                )

                var_ok = bool(cmp_before["equal"] and cmp_after["equal"])
                if not var_ok:
                    outside_all_ok = False

                per_var_outside_checks[var] = {
                    "equal": var_ok,
                    "before": cmp_before,
                    "after": cmp_after,
                }

            checks["outside_event_equals_base"] = outside_all_ok
            result["per_var_outside_checks"] = per_var_outside_checks

        # -------------------------------------------------------------
        # 5) Gesamtergebnis
        # -------------------------------------------------------------
        result["checks"] = checks

        all_checks_ok = all(bool(v) for v in checks.values())
        result["status"] = "PASS" if all_checks_ok else "FAIL"
        result["message"] = "Alle Prüfungen bestanden." if all_checks_ok else "Mindestens eine Prüfung fehlgeschlagen."

        return result

    except Exception as e:
        result["message"] = f"Exception während Verifikation: {e}"
        return result


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("--- Verifikation manipulierter Cutouts gestartet ---")
    print(f"Basis-Cutout:      {BASE_CUTOUT_PATH}")
    print(f"Manipulated dir:   {MANIPULATED_DIR}")
    print(f"Output dir:        {VERIFY_OUTPUT_DIR}")
    print(f"Anzahl Stressfälle:{len(STRESS_CASES)}")
    print(f"Repetitions:       {REPETITIONS}")
    print(f"Event duration:    {EVENT_DURATION}")

    all_results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for i, case in enumerate(STRESS_CASES, start=1):
        print("\n" + "#" * 100)
        print(f"Fall {i}/{len(STRESS_CASES)}: {case['name']}")
        print("#" * 100)

        case_name = case["name"]
        source_cutout_path = case["source_cutout_path"]
        source_week_start_str = case["source_week_start_str"]

        warnings = validate_case_metadata(case)
        for w in warnings:
            print(f"[WARN] {w}")

        manipulated_cutout_path = build_manipulated_cutout_path(
            base_cutout_path=BASE_CUTOUT_PATH,
            case_name=case_name,
            manipulated_dir=MANIPULATED_DIR,
        )

        result = verify_single_case(
            base_cutout_path=BASE_CUTOUT_PATH,
            manipulated_cutout_path=manipulated_cutout_path,
            source_cutout_path=source_cutout_path,
            source_week_start_str=source_week_start_str,
            repetitions=REPETITIONS,
            event_duration=EVENT_DURATION,
            abs_tol=ABS_TOL,
            rel_tol=REL_TOL,
            check_outside_equals_base=CHECK_OUTSIDE_EVENT_EQUALS_BASE,
        )

        result["case_name"] = case_name
        result["warnings"] = warnings
        all_results.append(result)

        checks = result.get("checks", {})
        summary_row = {
            "case_name": case_name,
            "status": result.get("status"),
            "message": result.get("message"),
            "source_cutout_path": normalize_path(source_cutout_path),
            "manipulated_cutout_path": normalize_path(manipulated_cutout_path),
            "source_week_start_str": source_week_start_str,
            "derived_target_start": result.get("derived_target_start"),
            "derived_target_end": result.get("derived_target_end"),
            "same_time_length_as_base": checks.get("same_time_length_as_base"),
            "same_data_vars_as_base": checks.get("same_data_vars_as_base"),
            "same_x_size_as_base": checks.get("same_x_size_as_base"),
            "same_y_size_as_base": checks.get("same_y_size_as_base"),
            "event_window_matches_source": checks.get("event_window_matches_source"),
            "outside_event_equals_base": checks.get("outside_event_equals_base"),
            "warnings": " | ".join(warnings) if warnings else "",
        }
        summary_rows.append(summary_row)

        print(f"[{result['status']}] {case_name}: {result['message']}")

    # -------------------------------------------------------------------------
    # Outputs schreiben
    # -------------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(VERIFY_OUTPUT_DIR, "verification_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    details_json = os.path.join(VERIFY_OUTPUT_DIR, "verification_details.json")
    with open(details_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    n_pass = int((summary_df["status"] == "PASS").sum())
    n_fail = int((summary_df["status"] == "FAIL").sum())

    print("\n" + "=" * 100)
    print("VERIFIKATION ABGESCHLOSSEN")
    print("=" * 100)
    print(f"PASS: {n_pass}")
    print(f"FAIL: {n_fail}")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Details JSON: {details_json}")

    if n_fail > 0:
        print("\nFehlgeschlagene Fälle:")
        failed = summary_df.loc[summary_df["status"] == "FAIL", ["case_name", "message"]]
        for _, row in failed.iterrows():
            print(f"  - {row['case_name']}: {row['message']}")


if __name__ == "__main__":
    main()