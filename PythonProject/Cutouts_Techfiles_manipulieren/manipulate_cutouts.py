#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

import atlite
import xarray as xr
import pandas as pd


# =============================================================================
# KONFIGURATION
# =============================================================================

# -------------------------------------------------------------------------
# 1) Basis-Cutout (dieses bleibt immer gleich und wird pro Fall neu geladen)
# -------------------------------------------------------------------------
BASE_CUTOUT_PATH = "/home/endata/cutouts/cutout_mCNRM-CERFACS-CM5_rcp45_2028.nc"

# -------------------------------------------------------------------------
# 2) Ausgabeordner für alle manipulierten Cutouts
# -------------------------------------------------------------------------
OUTPUT_DIR = "/home/endata/cutouts_manipulated/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# 3) Wie oft soll die Quellwoche wiederholt werden?
#    Beispiel: 3 => 1 Woche wird zu 3 Wochen gestreckt
# -------------------------------------------------------------------------
REPETITIONS = 1

# -------------------------------------------------------------------------
# 4) Liste der 10 Dunkelflauten-Fälle
#
#    Für jeden Fall:
#    - name: Name für die Ausgabedatei
#    - source_cutout_path: Quell-Cutout
#    - source_week_start_str: Startdatum der zu kopierenden Woche im Quell-Cutout
#
#    WICHTIG:
#    Passe diese Liste an deine echten 10 Ereignisse an.
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
def validate_datasets_compatibility(ds_target: xr.Dataset, ds_source: xr.Dataset) -> None:
    """
    Prüft grob, ob Ziel- und Quelldataset strukturell kompatibel sind.
    """
    required_spatial_dims = ["x", "y"]

    for dim in required_spatial_dims:
        if dim not in ds_target.dims:
            raise ValueError(f"Ziel-Dataset hat keine erwartete Dimension '{dim}'.")
        if dim not in ds_source.dims:
            raise ValueError(f"Quell-Dataset hat keine erwartete Dimension '{dim}'.")

        if ds_target.sizes[dim] != ds_source.sizes[dim]:
            raise ValueError(
                f"Dimension '{dim}' ungleich: "
                f"target={ds_target.sizes[dim]}, source={ds_source.sizes[dim]}"
            )

    if "time" not in ds_target.dims:
        raise ValueError("Ziel-Dataset hat keine 'time'-Dimension.")
    if "time" not in ds_source.dims:
        raise ValueError("Quell-Dataset hat keine 'time'-Dimension.")

    target_time_vars = {v for v, da in ds_target.data_vars.items() if "time" in da.dims}
    source_time_vars = {v for v, da in ds_source.data_vars.items() if "time" in da.dims}

    if target_time_vars != source_time_vars:
        raise ValueError(
            "Zeitabhängige Variablen unterscheiden sich.\n"
            f"Nur in target: {sorted(target_time_vars - source_time_vars)}\n"
            f"Nur in source: {sorted(source_time_vars - target_time_vars)}"
        )


def get_time_dependent_vars(ds: xr.Dataset) -> List[str]:
    """
    Liefert alle Datenvariablen mit time-Dimension.
    """
    return [v for v, da in ds.data_vars.items() if "time" in da.dims]


def get_static_vars(ds: xr.Dataset) -> List[str]:
    """
    Liefert alle Datenvariablen ohne time-Dimension.
    """
    return [v for v, da in ds.data_vars.items() if "time" not in da.dims]


def build_output_path(base_cutout_path: str, case_name: str, output_dir: str) -> str:
    """
    Erzeugt einen sinnvollen Dateinamen für das manipulierte Cutout.
    """
    base_name = Path(base_cutout_path).name
    return os.path.join(output_dir, f"{base_name}__{case_name}.nc")


def infer_target_year_from_cutout(ds_target: xr.Dataset) -> int:
    """
    Leitet das Zieljahr aus der Zeitachse des Basis-Cutouts ab.
    """
    if ds_target.sizes.get("time", 0) == 0:
        raise ValueError("Basis-Cutout hat keine Zeitschritte.")
    return pd.to_datetime(ds_target.time.values[0]).year


def map_source_start_to_target_year(source_start_date: pd.Timestamp, target_year: int) -> pd.Timestamp:
    """
    Überträgt Monat/Tag/Uhrzeit aus der Quelle auf das Zieljahr.
    Beispiel:
      1985-01-14 03:00 -> 2028-01-14 03:00
    """
    try:
        return source_start_date.replace(year=target_year)
    except ValueError as e:
        raise ValueError(
            f"Kann source_week_start_str={source_start_date} nicht auf Zieljahr {target_year} abbilden "
            f"(z. B. 29. Februar in Nicht-Schaltjahr)."
        ) from e


def manipulate_cutout_with_source_period(
    base_cutout_path: str,
    source_cutout_path: str,
    source_week_start_str: str,
    repetitions: int,
    output_path: str,
) -> None:
    """
    Erstellt EIN neues manipuliertes Cutout:
    - lädt das feste Basis-Cutout
    - lädt das Quell-Cutout
    - kopiert 1 Woche aus der Quelle
    - wiederholt diese 'repetitions'-mal
    - setzt sie im Basis-Cutout an derselben Kalenderposition ein
      (Monat/Tag/Uhrzeit wie Quelle, aber im Zieljahr des Basis-Cutouts)
    - speichert das Ergebnis als neue NetCDF
    """

    print("\n" + "=" * 90)
    print(f"Starte Fall -> {Path(output_path).name}")
    print("=" * 90)

    # ---------------------------------------------------------------------
    # 1) Cutouts laden
    # ---------------------------------------------------------------------
    print(f"[1/6] Lade Basis-Cutout:  {base_cutout_path}")
    base_cutout = atlite.Cutout(path=base_cutout_path)
    ds_target = base_cutout.data

    print(f"[1/6] Lade Quell-Cutout:  {source_cutout_path}")
    source_cutout = atlite.Cutout(path=source_cutout_path)
    ds_source = source_cutout.data

    # ---------------------------------------------------------------------
    # 2) Kompatibilität prüfen
    # ---------------------------------------------------------------------
    print("[2/6] Prüfe strukturelle Kompatibilität ...")
    validate_datasets_compatibility(ds_target, ds_source)

    time_dependent_vars = get_time_dependent_vars(ds_target)
    static_vars = get_static_vars(ds_target)

    print(f"      Zeitabhängige Variablen: {len(time_dependent_vars)}")
    print(f"      Statische Variablen:     {len(static_vars)}")

    # ---------------------------------------------------------------------
    # 3) Zeiträume definieren
    # ---------------------------------------------------------------------
    print("[3/6] Definiere Quell- und Zielzeiträume ...")
    week_duration = pd.Timedelta(weeks=1)

    source_start_date = pd.to_datetime(source_week_start_str)
    source_end_date = source_start_date + week_duration - pd.Timedelta(hours=1)

    target_year = infer_target_year_from_cutout(ds_target)
    target_start_date = map_source_start_to_target_year(source_start_date, target_year)
    target_end_date = target_start_date + (week_duration * repetitions) - pd.Timedelta(hours=1)

    print(f"      Zieljahr aus Basis-Cutout: {target_year}")
    print(f"      Quellwoche:                {source_start_date}  ->  {source_end_date}")
    print(f"      Zielperiode:              {target_start_date}  ->  {target_end_date}")

    # ---------------------------------------------------------------------
    # 4) Austauschblock bauen
    # ---------------------------------------------------------------------
    print("[4/6] Erzeuge Austauschblock ...")

    source_week_ds = ds_source[time_dependent_vars].sel(
        time=slice(source_start_date, source_end_date)
    )

    if source_week_ds.sizes.get("time", 0) == 0:
        raise ValueError(
            f"Die Quellwoche ist leer. Prüfe source_week_start_str={source_week_start_str} "
            f"für Cutout {source_cutout_path}"
        )

    replacement_block_ds = xr.concat([source_week_ds] * repetitions, dim="time")

    target_time_index = ds_target.sel(
        time=slice(target_start_date, target_end_date)
    ).time

    if len(target_time_index) == 0:
        raise ValueError(
            f"Die Zielperiode ist leer. "
            f"Abgeleitet wurde target_start_date={target_start_date} im Basis-Cutout."
        )

    if len(replacement_block_ds.time) != len(target_time_index):
        raise ValueError(
            "Zeitdimensionen stimmen nicht überein:\n"
            f"replacement_block_ds.time = {len(replacement_block_ds.time)}\n"
            f"target_time_index         = {len(target_time_index)}"
        )

    replacement_block_ds = replacement_block_ds.assign_coords(time=target_time_index)
    print("      Austauschblock erfolgreich auf Ziel-Zeitachse gelegt.")

    # ---------------------------------------------------------------------
    # 5) Basis-Dataset neu zusammensetzen
    # ---------------------------------------------------------------------
    print("[5/6] Setze neues manipuliertes Dataset zusammen ...")

    before_ds = ds_target.sel(time=slice(None, target_start_date - pd.Timedelta(hours=1)))
    after_ds = ds_target.sel(time=slice(target_end_date + pd.Timedelta(hours=1), None))

    manipulated_time_vars_ds = xr.concat(
        [
            before_ds[time_dependent_vars],
            replacement_block_ds,
            after_ds[time_dependent_vars],
        ],
        dim="time",
    )

    static_vars_ds = ds_target[static_vars]
    final_manipulated_ds = xr.merge([manipulated_time_vars_ds, static_vars_ds])

    # ursprüngliche Attributes grob erhalten
    final_manipulated_ds.attrs = ds_target.attrs.copy()
    for v in final_manipulated_ds.data_vars:
        if v in ds_target.data_vars:
            final_manipulated_ds[v].attrs = ds_target[v].attrs.copy()

    # ---------------------------------------------------------------------
    # 6) Konsistenzcheck + Speichern
    # ---------------------------------------------------------------------
    print("[6/6] Prüfe Ergebnis und speichere ...")

    if len(final_manipulated_ds.time) != len(ds_target.time):
        raise ValueError(
            "FEHLER: Länge des manipulierten Datasets stimmt nicht mit dem Original überein."
        )

    if set(final_manipulated_ds.data_vars) != set(ds_target.data_vars):
        raise ValueError(
            "FEHLER: Datenvariablen im manipulierten Dataset stimmen nicht mit dem Original überein."
        )

    if final_manipulated_ds.sizes["x"] != ds_target.sizes["x"] or final_manipulated_ds.sizes["y"] != ds_target.sizes["y"]:
        raise ValueError("FEHLER: Räumliche Dimensionen haben sich verändert.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_manipulated_ds.to_netcdf(output_path)

    print(f"[OK] Gespeichert: {output_path}")

    # optionaler Test-Reload
    print("      Teste Reload des neuen Cutouts ...")
    test_cutout = atlite.Cutout(path=output_path)
    _ = test_cutout.data
    print("      Reload erfolgreich.")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("--- Batch-Erzeugung manipulierter Atlite-Cutouts gestartet ---")
    print(f"Basis-Cutout: {BASE_CUTOUT_PATH}")
    print(f"Ausgabeordner: {OUTPUT_DIR}")
    print(f"Anzahl Stress-Fälle: {len(STRESS_CASES)}")

    if len(STRESS_CASES) != 10:
        print(
            f"[WARN] Du hast aktuell {len(STRESS_CASES)} Fälle definiert, "
            "nicht genau 10."
        )

    created_files: List[str] = []
    failed_cases: List[str] = []

    for i, case in enumerate(STRESS_CASES, start=1):
        print("\n" + "#" * 100)
        print(f"Fall {i}/{len(STRESS_CASES)}: {case['name']}")
        print("#" * 100)

        try:
            source_cutout_path = case["source_cutout_path"]
            source_week_start_str = case["source_week_start_str"]
            case_name = case["name"]

            output_path = build_output_path(
                base_cutout_path=BASE_CUTOUT_PATH,
                case_name=case_name,
                output_dir=OUTPUT_DIR,
            )

            manipulate_cutout_with_source_period(
                base_cutout_path=BASE_CUTOUT_PATH,
                source_cutout_path=source_cutout_path,
                source_week_start_str=source_week_start_str,
                repetitions=REPETITIONS,
                output_path=output_path,
            )

            created_files.append(output_path)

        except Exception as exc:
            failed_cases.append(case["name"])
            print(f"[FEHLER] Fall '{case['name']}' fehlgeschlagen: {exc}")

    print("\n" + "=" * 100)
    print("BATCH-LAUF ABGESCHLOSSEN")
    print("=" * 100)
    print(f"Erfolgreich erzeugte Cutouts: {len(created_files)}")
    print(f"Fehlgeschlagene Fälle:        {len(failed_cases)}")

    if created_files:
        print("\nErzeugte Dateien:")
        for fp in created_files:
            print(f"  - {fp}")

    if failed_cases:
        print("\nFehlgeschlagene Fälle:")
        for name in failed_cases:
            print(f"  - {name}")


if __name__ == "__main__":
    main()