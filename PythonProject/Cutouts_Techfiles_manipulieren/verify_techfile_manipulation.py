#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# KONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Allgemeine Techfile-Konfiguration
# -----------------------------------------------------------------------------
MODEL_NAME = "mCNRM-CERFACS-CM5"
SCENARIO = "rcp45"   # möglich: "rcp26", "rcp45", "rcp85"
TARGET_YEAR = 2028   # fixes Basisjahr, das manipuliert wurde

# Wurzelordner der Original-Techfiles
TECHFILES_BASE_DIR = "/home/endata/techfiles"

# Ordner der manipulierten Techfiles
OUTPUT_DIR = "/home/endata/techfiles_manipulated/d"

# Ausgabeordner für Verifikationsergebnisse
VERIFY_OUTPUT_DIR = "/home/endata/techfiles_manipulated/d_verification"
os.makedirs(VERIFY_OUTPUT_DIR, exist_ok=True)

# Techfile-Typen
TECHFILE_TYPES = [
    "wind",
    "wind_offshore",
    "pv",
]

# Wie oft wurde die Quellwoche wiederholt?
REPETITIONS = 1

# Zeitdimension
TIME_DIM = "time"

# Falls nur bestimmte zeitabhängige Variablen geprüft werden sollen:
# None => alle zeitabhängigen Variablen
TIME_VARS_TO_CHECK: Optional[List[str]] = None

# Ereignislänge
EVENT_DURATION = pd.Timedelta(weeks=1)

# Toleranzen
ABS_TOL = 1e-6
REL_TOL = 1e-8

# Zusätzliche strenge Prüfung:
# außerhalb des Ereignisfensters muss alles identisch zum Basis-Techfile sein
CHECK_OUTSIDE_EVENT_EQUALS_BASE = True


# -----------------------------------------------------------------------------
# Die 10 Stressfälle
# -----------------------------------------------------------------------------
STRESS_CASES: List[Dict[str, Any]] = [
    {
        "name": "stress_01",
        "source_year": 2031,
        "source_week_start_str": "2031-12-06",
    },
    {
        "name": "stress_02",
        "source_year": 2030,
        "source_week_start_str": "2030-11-23",
    },
    {
        "name": "stress_03",
        "source_year": 2035,
        "source_week_start_str": "2035-11-22",
    },
    {
        "name": "stress_04",
        "source_year": 2040,
        "source_week_start_str": "2040-12-12",
    },
    {
        "name": "stress_05",
        "source_year": 2033,
        "source_week_start_str": "2033-11-21",
    },
    {
        "name": "stress_06",
        "source_year": 2047,
        "source_week_start_str": "2047-01-07",
    },
    {
        "name": "stress_07",
        "source_year": 2041,
        "source_week_start_str": "2041-11-21",
    },
    {
        "name": "stress_08",
        "source_year": 2047,
        "source_week_start_str": "2047-11-23",
    },
    {
        "name": "stress_09",
        "source_year": 2039,
        "source_week_start_str": "2039-12-13",
    },
    {
        "name": "stress_10",
        "source_year": 2041,
        "source_week_start_str": "2041-01-30",
    },
]


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================
def validate_scenario_name(scenario: str) -> None:
    allowed = {"rcp26", "rcp45", "rcp85"}
    if scenario not in allowed:
        raise ValueError(
            f"Ungültiges SCENARIO='{scenario}'. Erlaubt sind: {sorted(allowed)}"
        )


def build_techfile_path(
    techfile_type: str,
    scenario: str,
    year: int,
    base_dir: str,
    model_name: str,
) -> str:
    filename = f"{techfile_type}_{model_name}_{scenario}_{year}_notAgg_pypsa.nc"
    return os.path.join(base_dir, filename)


def build_output_path(
    techfile_type: str,
    scenario: str,
    target_year: int,
    case_name: str,
    output_dir: str,
    model_name: str,
) -> str:
    filename = (
        f"{techfile_type}_{model_name}_{scenario}_{target_year}_notAgg_pypsa"
        f"__{case_name}.nc"
    )
    return os.path.join(output_dir, filename)


def normalize_time_index(ds: xr.Dataset, time_dim: str) -> xr.Dataset:
    if time_dim not in ds.coords and time_dim not in ds.dims:
        raise ValueError(f"Zeitdimension/-koordinate '{time_dim}' nicht im Dataset gefunden.")

    times = pd.to_datetime(ds[time_dim].values)
    try:
        if getattr(times, "tz", None) is not None:
            times = times.tz_localize(None)
    except Exception:
        pass

    return ds.assign_coords({time_dim: times})


def infer_time_dependent_vars(ds: xr.Dataset, time_dim: str) -> List[str]:
    return [v for v, da in ds.data_vars.items() if time_dim in da.dims]


def infer_static_vars(ds: xr.Dataset, time_dim: str) -> List[str]:
    return [v for v, da in ds.data_vars.items() if time_dim not in da.dims]


def validate_datasets_compatibility(
    ds_target: xr.Dataset,
    ds_source: xr.Dataset,
    time_dim: str,
    time_vars_to_check: Optional[List[str]] = None,
) -> List[str]:
    if time_dim not in ds_target.dims:
        raise ValueError(f"Ziel-Dataset hat keine Zeitdimension '{time_dim}'.")
    if time_dim not in ds_source.dims:
        raise ValueError(f"Quell-Dataset hat keine Zeitdimension '{time_dim}'.")

    target_non_time_dims = {d: s for d, s in ds_target.sizes.items() if d != time_dim}
    source_non_time_dims = {d: s for d, s in ds_source.sizes.items() if d != time_dim}

    if target_non_time_dims != source_non_time_dims:
        raise ValueError(
            "Nicht-zeitliche Dimensionen stimmen nicht überein.\n"
            f"target: {target_non_time_dims}\n"
            f"source: {source_non_time_dims}"
        )

    target_time_vars = set(infer_time_dependent_vars(ds_target, time_dim))
    source_time_vars = set(infer_time_dependent_vars(ds_source, time_dim))

    if time_vars_to_check is None:
        if target_time_vars != source_time_vars:
            raise ValueError(
                "Zeitabhängige Variablen unterscheiden sich.\n"
                f"Nur in target: {sorted(target_time_vars - source_time_vars)}\n"
                f"Nur in source: {sorted(source_time_vars - target_time_vars)}"
            )
        chosen = sorted(target_time_vars)
    else:
        chosen = list(time_vars_to_check)
        missing_target = [v for v in chosen if v not in target_time_vars]
        missing_source = [v for v in chosen if v not in source_time_vars]
        if missing_target:
            raise ValueError(f"Diese Variablen fehlen im Ziel-Dataset: {missing_target}")
        if missing_source:
            raise ValueError(f"Diese Variablen fehlen im Quell-Dataset: {missing_source}")

    return chosen


def estimate_time_step(ds: xr.Dataset, time_dim: str) -> pd.Timedelta:
    times = pd.to_datetime(ds[time_dim].values)
    if len(times) < 2:
        raise ValueError("Zu wenige Zeitpunkte zur Bestimmung der Zeitauflösung.")

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
            f"Kann source_week_start_str={source_start_date} nicht auf Zieljahr {target_year} abbilden "
            f"(z. B. 29. Februar in Nicht-Schaltjahr)."
        ) from e


def compare_dataarrays(
    da_a: xr.DataArray,
    da_b: xr.DataArray,
    abs_tol: float,
    rel_tol: float,
) -> Dict[str, Any]:
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

    finite_mask = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(a - b)
    allowed = abs_tol + rel_tol * np.abs(b)

    diff_valid = diff[finite_mask]
    allowed_valid = allowed[finite_mask]

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


def verify_single_techfile(
    base_techfile_path: str,
    source_techfile_path: str,
    manipulated_techfile_path: str,
    source_week_start_str: str,
    target_year: int,
    repetitions: int,
    event_duration: pd.Timedelta,
    time_dim: str,
    time_vars_to_check: Optional[List[str]],
    abs_tol: float,
    rel_tol: float,
    check_outside_equals_base: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "FAIL",
        "message": "",
        "checks": {},
    }

    if not os.path.exists(base_techfile_path):
        result["message"] = f"Basis-Techfile nicht gefunden: {base_techfile_path}"
        return result

    if not os.path.exists(source_techfile_path):
        result["message"] = f"Quell-Techfile nicht gefunden: {source_techfile_path}"
        return result

    if not os.path.exists(manipulated_techfile_path):
        result["message"] = f"Manipuliertes Techfile nicht gefunden: {manipulated_techfile_path}"
        return result

    ds_base = xr.open_dataset(base_techfile_path)
    ds_source = xr.open_dataset(source_techfile_path)
    ds_manip = xr.open_dataset(manipulated_techfile_path)

    try:
        ds_base = normalize_time_index(ds_base, time_dim)
        ds_source = normalize_time_index(ds_source, time_dim)
        ds_manip = normalize_time_index(ds_manip, time_dim)

        time_vars = validate_datasets_compatibility(
            ds_target=ds_base,
            ds_source=ds_source,
            time_dim=time_dim,
            time_vars_to_check=time_vars_to_check,
        )

        checks: Dict[str, Any] = {}
        checks["same_time_length_as_base"] = bool(len(ds_manip[time_dim]) == len(ds_base[time_dim]))
        checks["same_data_vars_as_base"] = bool(set(ds_manip.data_vars) == set(ds_base.data_vars))

        base_non_time_dims = {d: s for d, s in ds_base.sizes.items() if d != time_dim}
        manip_non_time_dims = {d: s for d, s in ds_manip.sizes.items() if d != time_dim}
        checks["same_non_time_dims_as_base"] = bool(base_non_time_dims == manip_non_time_dims)

        source_start_date = pd.to_datetime(source_week_start_str)
        target_start_date = map_source_start_to_target_year(source_start_date, target_year)

        time_step = estimate_time_step(ds_base, time_dim)

        source_end_date = source_start_date + event_duration - time_step
        target_end_date = target_start_date + (event_duration * repetitions) - time_step

        result["derived_target_start"] = str(target_start_date)
        result["derived_target_end"] = str(target_end_date)
        result["time_step"] = str(time_step)

        source_event = ds_source[time_vars].sel({time_dim: slice(source_start_date, source_end_date)})
        if source_event.sizes.get(time_dim, 0) == 0:
            result["message"] = f"Quellereignis leer: {source_start_date} bis {source_end_date}"
            result["checks"] = checks
            return result

        expected_event = xr.concat([source_event] * repetitions, dim=time_dim)

        target_time_index = ds_base.sel({time_dim: slice(target_start_date, target_end_date)})[time_dim]
        if len(target_time_index) == 0:
            result["message"] = f"Zielereignisfenster leer: {target_start_date} bis {target_end_date}"
            result["checks"] = checks
            return result

        if len(expected_event[time_dim]) != len(target_time_index):
            result["message"] = (
                f"Längenproblem im Ereignisfenster: expected={len(expected_event[time_dim])}, "
                f"target={len(target_time_index)}"
            )
            result["checks"] = checks
            return result

        expected_event = expected_event.assign_coords({time_dim: target_time_index})
        actual_event = ds_manip[time_vars].sel({time_dim: slice(target_start_date, target_end_date)})

        per_var_event_checks: Dict[str, Any] = {}
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

        if check_outside_equals_base:
            before_actual = ds_manip[time_vars].sel({time_dim: slice(None, target_start_date - time_step)})
            before_base = ds_base[time_vars].sel({time_dim: slice(None, target_start_date - time_step)})

            after_actual = ds_manip[time_vars].sel({time_dim: slice(target_end_date + time_step, None)})
            after_base = ds_base[time_vars].sel({time_dim: slice(target_end_date + time_step, None)})

            per_var_outside_checks: Dict[str, Any] = {}
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

        result["checks"] = checks
        all_checks_ok = all(bool(v) for v in checks.values())
        result["status"] = "PASS" if all_checks_ok else "FAIL"
        result["message"] = "Alle Prüfungen bestanden." if all_checks_ok else "Mindestens eine Prüfung fehlgeschlagen."

        return result

    except Exception as e:
        result["message"] = f"Exception während Verifikation: {e}"
        return result

    finally:
        ds_base.close()
        ds_source.close()
        ds_manip.close()


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    validate_scenario_name(SCENARIO)

    print("--- Verifikation manipulierter Techfiles gestartet ---")
    print(f"MODEL_NAME:          {MODEL_NAME}")
    print(f"SCENARIO:            {SCENARIO}")
    print(f"TARGET_YEAR:         {TARGET_YEAR}")
    print(f"TECHFILES_BASE_DIR:  {TECHFILES_BASE_DIR}")
    print(f"OUTPUT_DIR:          {OUTPUT_DIR}")
    print(f"VERIFY_OUTPUT_DIR:   {VERIFY_OUTPUT_DIR}")
    print(f"TECHFILE_TYPES:      {TECHFILE_TYPES}")
    print(f"REPETITIONS:         {REPETITIONS}")
    print(f"Anzahl Stress-Fälle: {len(STRESS_CASES)}")

    summary_rows: List[Dict[str, Any]] = []
    detail_results: List[Dict[str, Any]] = []

    for i, case in enumerate(STRESS_CASES, start=1):
        case_name = case["name"]
        source_year = int(case["source_year"])
        source_week_start_str = case["source_week_start_str"]

        print("\n" + "#" * 100)
        print(f"Fall {i}/{len(STRESS_CASES)}: {case_name}")
        print("#" * 100)

        for techfile_type in TECHFILE_TYPES:
            print(f"\n--- Prüfe {techfile_type} | {case_name} ---")

            base_techfile_path = build_techfile_path(
                techfile_type=techfile_type,
                scenario=SCENARIO,
                year=TARGET_YEAR,
                base_dir=TECHFILES_BASE_DIR,
                model_name=MODEL_NAME,
            )

            source_techfile_path = build_techfile_path(
                techfile_type=techfile_type,
                scenario=SCENARIO,
                year=source_year,
                base_dir=TECHFILES_BASE_DIR,
                model_name=MODEL_NAME,
            )

            manipulated_techfile_path = build_output_path(
                techfile_type=techfile_type,
                scenario=SCENARIO,
                target_year=TARGET_YEAR,
                case_name=case_name,
                output_dir=OUTPUT_DIR,
                model_name=MODEL_NAME,
            )

            verify_res = verify_single_techfile(
                base_techfile_path=base_techfile_path,
                source_techfile_path=source_techfile_path,
                manipulated_techfile_path=manipulated_techfile_path,
                source_week_start_str=source_week_start_str,
                target_year=TARGET_YEAR,
                repetitions=REPETITIONS,
                event_duration=EVENT_DURATION,
                time_dim=TIME_DIM,
                time_vars_to_check=TIME_VARS_TO_CHECK,
                abs_tol=ABS_TOL,
                rel_tol=REL_TOL,
                check_outside_equals_base=CHECK_OUTSIDE_EVENT_EQUALS_BASE,
            )

            verify_res["case_name"] = case_name
            verify_res["techfile_type"] = techfile_type
            verify_res["base_techfile_path"] = base_techfile_path
            verify_res["source_techfile_path"] = source_techfile_path
            verify_res["manipulated_techfile_path"] = manipulated_techfile_path
            verify_res["source_year"] = source_year
            verify_res["source_week_start_str"] = source_week_start_str
            detail_results.append(verify_res)

            checks = verify_res.get("checks", {})
            summary_rows.append({
                "case_name": case_name,
                "techfile_type": techfile_type,
                "status": verify_res.get("status"),
                "message": verify_res.get("message"),
                "source_year": source_year,
                "source_week_start_str": source_week_start_str,
                "derived_target_start": verify_res.get("derived_target_start"),
                "derived_target_end": verify_res.get("derived_target_end"),
                "same_time_length_as_base": checks.get("same_time_length_as_base"),
                "same_data_vars_as_base": checks.get("same_data_vars_as_base"),
                "same_non_time_dims_as_base": checks.get("same_non_time_dims_as_base"),
                "event_window_matches_source": checks.get("event_window_matches_source"),
                "outside_event_equals_base": checks.get("outside_event_equals_base"),
                "base_techfile_path": base_techfile_path,
                "source_techfile_path": source_techfile_path,
                "manipulated_techfile_path": manipulated_techfile_path,
            })

            print(f"[{verify_res['status']}] {techfile_type} | {case_name}: {verify_res['message']}")

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = os.path.join(VERIFY_OUTPUT_DIR, "verification_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    details_json = os.path.join(VERIFY_OUTPUT_DIR, "verification_details.json")
    with open(details_json, "w", encoding="utf-8") as f:
        json.dump(detail_results, f, indent=2, ensure_ascii=False)

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
        print("\nFehlgeschlagene Prüfungen:")
        failed = summary_df.loc[summary_df["status"] == "FAIL", ["case_name", "techfile_type", "message"]]
        for _, row in failed.iterrows():
            print(f"  - {row['case_name']} | {row['techfile_type']}: {row['message']}")


if __name__ == "__main__":
    main()