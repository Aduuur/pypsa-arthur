#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import xarray as xr
import pandas as pd


# =============================================================================
# KONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Allgemeine Techfile-Konfiguration
# -----------------------------------------------------------------------------
MODEL_NAME = "mCNRM-CERFACS-CM5"
SCENARIO = "rcp45"   # möglich: "rcp26", "rcp45", "rcp85"
TARGET_YEAR = 2028   # fixes Basisjahr, das manipuliert werden soll

# Wurzelordner der Techfiles
TECHFILES_BASE_DIR = "/home/endata/techfiles"

# Ausgabeordner
OUTPUT_DIR = "/home/endata/techfiles_manipulated/d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Techfile-Typen, die pro Fall alle manipuliert werden sollen
TECHFILE_TYPES = [
    "wind",
    "wind_offshore",
    "pv",
]

# Wie oft wird die Quellwoche wiederholt?
REPETITIONS = 1

# Zeitdimension
TIME_DIM = "time"

# Falls du nur bestimmte zeitabhängige Variablen ersetzen willst:
# None => alle zeitabhängigen Variablen
TIME_VARS_TO_REPLACE: Optional[List[str]] = None


# -----------------------------------------------------------------------------
# Die 10 Stressfälle
#
# Jeder Fall beschreibt:
# - name: Name des Falls
# - source_year: aus welchem Wetterjahr die Dunkelflaute stammt
# - source_week_start_str: Startzeitpunkt der Quellwoche
#
# Die Quellwoche wird automatisch auf denselben Monat/Tag/Uhrzeitpunkt
# im TARGET_YEAR gemappt.
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
    """
    Baut Dateinamen im Stil:
      wind_mCNRM-CERFACS-CM5_rcp45_2010_notAgg_pypsa.nc
      wind_offshore_mCNRM-CERFACS-CM5_rcp45_2010_notAgg_pypsa.nc
      pv_mCNRM-CERFACS-CM5_rcp45_2010_notAgg_pypsa.nc
    """
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
    time_vars_to_replace: Optional[List[str]] = None,
) -> List[str]:
    """
    Prüft strukturelle Kompatibilität und gibt die zu manipulierenden Variablen zurück.
    """
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

    if time_vars_to_replace is None:
        if target_time_vars != source_time_vars:
            raise ValueError(
                "Zeitabhängige Variablen unterscheiden sich.\n"
                f"Nur in target: {sorted(target_time_vars - source_time_vars)}\n"
                f"Nur in source: {sorted(source_time_vars - target_time_vars)}"
            )
        chosen = sorted(target_time_vars)
    else:
        chosen = list(time_vars_to_replace)
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
    """
    Überträgt Monat/Tag/Uhrzeit aus der Quelle auf das Zieljahr.
    Beispiel:
      2031-12-06 00:00 -> 2010-12-06 00:00
    """
    try:
        return source_start_date.replace(year=target_year)
    except ValueError as e:
        raise ValueError(
            f"Kann source_week_start_str={source_start_date} nicht auf Zieljahr {target_year} abbilden "
            f"(z. B. 29. Februar in Nicht-Schaltjahr)."
        ) from e


def select_source_period(
    ds_source: xr.Dataset,
    time_vars: List[str],
    time_dim: str,
    source_start: pd.Timestamp,
    duration: pd.Timedelta,
    time_step: pd.Timedelta,
) -> xr.Dataset:
    source_end = source_start + duration - time_step
    source_period = ds_source[time_vars].sel({time_dim: slice(source_start, source_end)})

    if source_period.sizes.get(time_dim, 0) == 0:
        raise ValueError(
            f"Der gewählte Quellzeitraum ist leer: {source_start} bis {source_end}"
        )

    return source_period


def manipulate_single_techfile(
    base_techfile_path: str,
    source_techfile_path: str,
    source_week_start_str: str,
    target_year: int,
    repetitions: int,
    output_path: str,
    time_dim: str = "time",
    time_vars_to_replace: Optional[List[str]] = None,
) -> None:
    """
    Manipuliert genau eine Techfile-Datei.

    Der Quellzeitraum wird nicht an einen festen Zielstart gelegt,
    sondern an dieselbe Kalenderposition im target_year:
      Quelle: 2031-12-06 00:00
      Ziel:   2010-12-06 00:00
    """
    print("\n" + "-" * 90)
    print(f"Manipuliere Techfile: {Path(base_techfile_path).name}")
    print("-" * 90)

    print(f"[1/6] Lade Basis-Techfile:  {base_techfile_path}")
    ds_target = xr.open_dataset(base_techfile_path)
    ds_target = normalize_time_index(ds_target, time_dim)

    print(f"[1/6] Lade Quell-Techfile:  {source_techfile_path}")
    ds_source = xr.open_dataset(source_techfile_path)
    ds_source = normalize_time_index(ds_source, time_dim)

    try:
        print("[2/6] Prüfe strukturelle Kompatibilität ...")
        time_vars = validate_datasets_compatibility(
            ds_target=ds_target,
            ds_source=ds_source,
            time_dim=time_dim,
            time_vars_to_replace=time_vars_to_replace,
        )
        static_vars = infer_static_vars(ds_target, time_dim)

        print(f"      Zeitabhängige Variablen für Austausch: {len(time_vars)}")
        print(f"      Statische Variablen:                  {len(static_vars)}")

        print("[3/6] Definiere Quell- und Zielzeiträume ...")
        source_start_date = pd.to_datetime(source_week_start_str)
        target_start_date = map_source_start_to_target_year(source_start_date, target_year)

        time_step = estimate_time_step(ds_target, time_dim)
        week_duration = pd.Timedelta(weeks=1)
        target_total_duration = week_duration * repetitions

        source_end_date = source_start_date + week_duration - time_step
        target_end_date = target_start_date + target_total_duration - time_step

        print(f"      Zeitauflösung erkannt: {time_step}")
        print(f"      Quellwoche:            {source_start_date} -> {source_end_date}")
        print(f"      Zielperiode:           {target_start_date} -> {target_end_date}")
        print(f"      Zieljahr:              {target_year}")

        print("[4/6] Erzeuge Austauschblock ...")
        source_week_ds = select_source_period(
            ds_source=ds_source,
            time_vars=time_vars,
            time_dim=time_dim,
            source_start=source_start_date,
            duration=week_duration,
            time_step=time_step,
        )

        replacement_block_ds = xr.concat([source_week_ds] * repetitions, dim=time_dim)

        target_time_index = ds_target.sel(
            {time_dim: slice(target_start_date, target_end_date)}
        )[time_dim]

        if len(target_time_index) == 0:
            raise ValueError(
                f"Die Zielperiode ist leer. Abgeleitet wurde target_start_date={target_start_date}"
            )

        if len(replacement_block_ds[time_dim]) != len(target_time_index):
            raise ValueError(
                "Zeitdimensionen stimmen nicht überein:\n"
                f"replacement_block_ds[{time_dim}] = {len(replacement_block_ds[time_dim])}\n"
                f"target_time_index                = {len(target_time_index)}"
            )

        replacement_block_ds = replacement_block_ds.assign_coords({time_dim: target_time_index})
        print("      Austauschblock erfolgreich auf Ziel-Zeitachse gelegt.")

        print("[5/6] Setze neues manipuliertes Dataset zusammen ...")
        before_ds = ds_target.sel({time_dim: slice(None, target_start_date - time_step)})
        after_ds = ds_target.sel({time_dim: slice(target_end_date + time_step, None)})

        manipulated_time_vars_ds = xr.concat(
            [
                before_ds[time_vars],
                replacement_block_ds,
                after_ds[time_vars],
            ],
            dim=time_dim,
        )

        static_vars_ds = ds_target[static_vars] if static_vars else xr.Dataset()
        final_manipulated_ds = xr.merge([manipulated_time_vars_ds, static_vars_ds])

        final_manipulated_ds.attrs = ds_target.attrs.copy()
        for v in final_manipulated_ds.data_vars:
            if v in ds_target.data_vars:
                final_manipulated_ds[v].attrs = ds_target[v].attrs.copy()

        for coord in ds_target.coords:
            if coord in final_manipulated_ds.coords and coord != time_dim:
                final_manipulated_ds = final_manipulated_ds.assign_coords(
                    {coord: ds_target.coords[coord]}
                )

        print("[6/6] Prüfe Ergebnis und speichere ...")
        if len(final_manipulated_ds[time_dim]) != len(ds_target[time_dim]):
            raise ValueError(
                "FEHLER: Länge des manipulierten Datasets stimmt nicht mit dem Original überein."
            )

        if set(final_manipulated_ds.data_vars) != set(ds_target.data_vars):
            raise ValueError(
                "FEHLER: Datenvariablen im manipulierten Dataset stimmen nicht mit dem Original überein."
            )

        target_non_time_dims = {d: s for d, s in ds_target.sizes.items() if d != time_dim}
        final_non_time_dims = {d: s for d, s in final_manipulated_ds.sizes.items() if d != time_dim}
        if target_non_time_dims != final_non_time_dims:
            raise ValueError(
                "FEHLER: Nicht-zeitliche Dimensionen haben sich verändert.\n"
                f"target: {target_non_time_dims}\n"
                f"final:  {final_non_time_dims}"
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        final_manipulated_ds.to_netcdf(output_path)

        print(f"[OK] Gespeichert: {output_path}")

        print("      Teste Reload der neuen Datei ...")
        test_ds = xr.open_dataset(output_path)
        test_ds.close()
        print("      Reload erfolgreich.")

    finally:
        ds_target.close()
        ds_source.close()


def manipulate_case_all_techfiles(
    case: Dict[str, Any],
    techfile_types: List[str],
    scenario: str,
    target_year: int,
    base_dir: str,
    output_dir: str,
    model_name: str,
    repetitions: int,
    time_dim: str = "time",
    time_vars_to_replace: Optional[List[str]] = None,
) -> List[str]:
    """
    Manipuliert für einen Stressfall alle Techfile-Typen:
      wind, wind_offshore, pv

    Der Stresszeitraum wird auf dieselbe Kalenderposition im target_year gelegt.
    """
    case_name = case["name"]
    source_year = int(case["source_year"])
    source_week_start_str = case["source_week_start_str"]

    created_files: List[str] = []

    print("\n" + "=" * 100)
    print(f"STARTE STRESSFALL: {case_name}")
    print(f"source_year = {source_year}")
    print(f"source_week_start_str = {source_week_start_str}")
    print(f"mapped target_year = {target_year}")
    print("=" * 100)

    for techfile_type in techfile_types:
        base_techfile_path = build_techfile_path(
            techfile_type=techfile_type,
            scenario=scenario,
            year=target_year,
            base_dir=base_dir,
            model_name=model_name,
        )

        source_techfile_path = build_techfile_path(
            techfile_type=techfile_type,
            scenario=scenario,
            year=source_year,
            base_dir=base_dir,
            model_name=model_name,
        )

        output_path = build_output_path(
            techfile_type=techfile_type,
            scenario=scenario,
            target_year=target_year,
            case_name=case_name,
            output_dir=output_dir,
            model_name=model_name,
        )

        if not os.path.exists(base_techfile_path):
            raise FileNotFoundError(f"Basis-Techfile nicht gefunden: {base_techfile_path}")
        if not os.path.exists(source_techfile_path):
            raise FileNotFoundError(f"Quell-Techfile nicht gefunden: {source_techfile_path}")

        manipulate_single_techfile(
            base_techfile_path=base_techfile_path,
            source_techfile_path=source_techfile_path,
            source_week_start_str=source_week_start_str,
            target_year=target_year,
            repetitions=repetitions,
            output_path=output_path,
            time_dim=time_dim,
            time_vars_to_replace=time_vars_to_replace,
        )

        created_files.append(output_path)

    return created_files


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    validate_scenario_name(SCENARIO)

    print("--- Batch-Erzeugung manipulierter Techfiles gestartet ---")
    print(f"MODEL_NAME:          {MODEL_NAME}")
    print(f"SCENARIO:            {SCENARIO}")
    print(f"TARGET_YEAR:         {TARGET_YEAR}")
    print(f"TECHFILES_BASE_DIR:  {TECHFILES_BASE_DIR}")
    print(f"OUTPUT_DIR:          {OUTPUT_DIR}")
    print(f"TECHFILE_TYPES:      {TECHFILE_TYPES}")
    print(f"REPETITIONS:         {REPETITIONS}")
    print(f"Anzahl Stress-Fälle: {len(STRESS_CASES)}")

    if len(STRESS_CASES) != 10:
        print(f"[WARN] Aktuell sind {len(STRESS_CASES)} Fälle definiert, nicht genau 10.")

    all_created_files: List[str] = []
    failed_cases: List[str] = []

    for i, case in enumerate(STRESS_CASES, start=1):
        print("\n" + "#" * 100)
        print(f"Fall {i}/{len(STRESS_CASES)}: {case['name']}")
        print("#" * 100)

        try:
            created_files = manipulate_case_all_techfiles(
                case=case,
                techfile_types=TECHFILE_TYPES,
                scenario=SCENARIO,
                target_year=TARGET_YEAR,
                base_dir=TECHFILES_BASE_DIR,
                output_dir=OUTPUT_DIR,
                model_name=MODEL_NAME,
                repetitions=REPETITIONS,
                time_dim=TIME_DIM,
                time_vars_to_replace=TIME_VARS_TO_REPLACE,
            )
            all_created_files.extend(created_files)

        except Exception as exc:
            failed_cases.append(case["name"])
            print(f"[FEHLER] Fall '{case['name']}' fehlgeschlagen: {exc}")

    print("\n" + "=" * 100)
    print("BATCH-LAUF ABGESCHLOSSEN")
    print("=" * 100)
    print(f"Erfolgreich erzeugte Dateien: {len(all_created_files)}")
    print(f"Fehlgeschlagene Stress-Fälle: {len(failed_cases)}")

    if all_created_files:
        print("\nErzeugte Dateien:")
        for fp in all_created_files:
            print(f"  - {fp}")

    if failed_cases:
        print("\nFehlgeschlagene Fälle:")
        for name in failed_cases:
            print(f"  - {name}")


if __name__ == "__main__":
    main()