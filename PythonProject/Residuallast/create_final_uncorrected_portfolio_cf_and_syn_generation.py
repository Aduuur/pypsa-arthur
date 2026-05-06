#create_final_portfolio_cordex.py
from __future__ import annotations

import os
import glob
import logging
from typing import List, Iterable

import pandas as pd


# =============================================================================
# KONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

YEARS_TO_PROCESS = range(2025, 2051)
CAPACITY_YEAR = 2024  # Jahr für installierte Kapazitäten (hier nur Label)

COUNTRIES_TO_ANALYZE = [
    "Germany", "France", "Poland","Austria", "Switzerland",
    "Italy", "Spain", "Belgium", "Netherlands", "Denmark", "Sweden",
    "Portugal", "Norway", "Finland", "Lithuania", "Latvia", "Estonia",
    "United Kingdom", "Czechia",

]
# --- CORDEX/MA_Arthur Pfade ---
MA_BASE = "/mnt/endata/MA_Arthur"

# Outputs aus Onshore+Solar-Skript (pro Jahr, pro Land)
ONSHORE_SOLAR_PER_YEAR_DIR = os.path.join(
    MA_BASE, "atlite_cf_results", "results_per_year_uncorrected"
)

# Outputs aus Offshore-Skript (pro Jahr, pro Land-Unterordner)
OFFSHORE_PER_YEAR_BASE_DIR = os.path.join(
    MA_BASE, "atlite_cf_results_offshore", "results_per_year_uncorrected"
)

# Kapazitätsdatei
CAPACITY_DATA_PATH = "/home/endata/PycharmProjects/PythonProject/Residuallast/installed_capacities_europe.csv"

# Finaler Ausgabeordner
OUTPUT_DIR = os.path.join(MA_BASE, "final_results_portfolio_uncorrected")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Mapping von Namen auf Ländercodes ---
COUNTRY_NAME_TO_CODE_MAP = {
    "Germany": "DE",
    "France": "FR",
    "Netherlands": "NL",
    "Belgium": "BE",
    "Austria": "AT",
    "Switzerland": "CH",
    "United Kingdom": "GB",
    "Spain": "ES",
    "Portugal": "PT",
    "Italy": "IT",
    "Denmark": "DK",
    "Sweden": "SE",
    "Norway": "NO",
    "Finland": "FI",
    "Poland": "PL",
    "Czechia": "CZ",
    "Lithuania": "LT",
    "Latvia": "LV",
    "Estonia": "EE",

}


# =============================================================================
# Helper: robustes Einlesen der per-year CSVs (CF / Offshore)
# =============================================================================
def _read_timeseries_csv(path: str) -> pd.DataFrame:
    """
    Robust: akzeptiert entweder eine 'time'-Spalte oder einen Datetime-Index in Spalte 0.
    """
    try:
        df = pd.read_csv(path, index_col="time", parse_dates=True)
        return df
    except Exception:
        pass

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "time"
    return df


def load_onshore_solar_all_years(country_name: str, years: Iterable[int]) -> pd.DataFrame:
    """
    Lädt und konkateniert alle Jahresdateien aus:
      uncorrected_cf_<Country>_<Year>.csv
    Erwartete Spalten: solar_cf, wind_cf
    """
    frames: List[pd.DataFrame] = []
    cn = country_name.replace(" ", "_")

    for y in years:
        fn = f"uncorrected_cf_{cn}_{y}.csv"
        path = os.path.join(ONSHORE_SOLAR_PER_YEAR_DIR, fn)
        if not os.path.exists(path):
            matches = sorted(glob.glob(os.path.join(ONSHORE_SOLAR_PER_YEAR_DIR, f"*{cn}*{y}*.csv")))
            if not matches:
                continue
            path = matches[0]

        df = _read_timeseries_csv(path)

        if "solar_cf" not in df.columns or "wind_cf" not in df.columns:
            raise ValueError(
                f"Onshore+Solar Datei hat nicht die erwarteten Spalten (solar_cf, wind_cf): {path} | cols={list(df.columns)}"
            )

        frames.append(df[["solar_cf", "wind_cf"]])

    if not frames:
        raise FileNotFoundError(
            f"Keine Onshore+Solar Jahresdateien gefunden für {country_name} in {ONSHORE_SOLAR_PER_YEAR_DIR}"
        )

    out = pd.concat(frames, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def load_offshore_all_years(country_name: str, years: Iterable[int]) -> pd.DataFrame:
    """
    Lädt und konkateniert alle Offshore-Jahresdateien aus:
      <Country>/offshore_wind_cf_<Year>.csv
    Erwartete Spalte: wind_offshore_cf (oder legacy: wind_cf)
    """
    frames: List[pd.DataFrame] = []
    country_folder = os.path.join(OFFSHORE_PER_YEAR_BASE_DIR, country_name.replace(" ", "_"))

    for y in years:
        fn = f"offshore_wind_cf_{y}.csv"
        path = os.path.join(country_folder, fn)
        if not os.path.exists(path):
            matches = sorted(glob.glob(os.path.join(country_folder, f"*{y}*.csv")))
            if not matches:
                continue
            path = matches[0]

        df = _read_timeseries_csv(path)

        if "wind_offshore_cf" in df.columns:
            frames.append(df[["wind_offshore_cf"]])
        elif "wind_cf" in df.columns:
            frames.append(df[["wind_cf"]].rename(columns={"wind_cf": "wind_offshore_cf"}))
        else:
            raise ValueError(
                f"Offshore Datei hat keine erwartete Spalte (wind_offshore_cf oder wind_cf): {path} | cols={list(df.columns)}"
            )

    if not frames:
        raise FileNotFoundError(f"Keine Offshore Jahresdateien gefunden für {country_name} in {country_folder}")

    out = pd.concat(frames, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


# =============================================================================
# HAUPTSKRIPT
# =============================================================================
if __name__ == "__main__":
    print("Lade installierte Kapazitäten...")
    df_capacity = pd.read_csv(CAPACITY_DATA_PATH)

    weather_start, weather_end = min(YEARS_TO_PROCESS), max(YEARS_TO_PROCESS)

    for country_name in COUNTRIES_TO_ANALYZE:
        print(f"\n--- Verarbeite {country_name} ---")

        country_code = COUNTRY_NAME_TO_CODE_MAP.get(country_name)
        if not country_code:
            print(f"  WARNUNG: Ländercode für {country_name} nicht gefunden. Überspringe.")
            continue

        # --- Schritt 1: Lade CF-Dateien ---
        try:
            df_onshore_solar = load_onshore_solar_all_years(country_name, YEARS_TO_PROCESS)
            df_onshore_solar = df_onshore_solar.rename(columns={"wind_cf": "wind_onshore_cf"})

            try:
                df_offshore = load_offshore_all_years(country_name, YEARS_TO_PROCESS)
                df_full = df_onshore_solar.join(df_offshore, how="outer")
            except FileNotFoundError:
                print(f"  INFO: Keine Offshore-Dateien für {country_name} gefunden. Setze Offshore-CF auf 0.")
                df_full = df_onshore_solar.copy()
                df_full["wind_offshore_cf"] = 0.0

        except FileNotFoundError as e:
            print(f"  WARNUNG: Eingabedateien (CF) für {country_name} fehlen. Details: {e}")
            continue
        except Exception as e:
            print(f"  FEHLER beim Laden/Aufbauen der CF-Zeitreihen für {country_name}: {e}")
            continue

        # --- Schritt 2: Kapazitäten ermitteln ---
        country_capacity_df = df_capacity[df_capacity["country"] == country_code]

        solar_cap = country_capacity_df[country_capacity_df["technology"] == "solar"]["installed_gw"].sum()
        onshore_cap = country_capacity_df[country_capacity_df["technology"] == "wind_onshore"]["installed_gw"].sum()
        offshore_cap = country_capacity_df[country_capacity_df["technology"] == "wind_offshore"]["installed_gw"].sum()

        total_capacity = solar_cap + onshore_cap + offshore_cap
        if total_capacity == 0:
            print(f"  WARNUNG: Keine Kapazitäten für {country_name} gefunden.")
            continue

        weights = {
            "solar_cf": solar_cap / total_capacity,
            "wind_onshore_cf": onshore_cap / total_capacity,
            "wind_offshore_cf": offshore_cap / total_capacity,
        }

        print(f"  Kapazitäten (Proxy, cap{CAPACITY_YEAR}):")
        print(f"    - Solar        : {solar_cap:.2f} GW")
        print(f"    - Wind Onshore : {onshore_cap:.2f} GW")
        print(f"    - Wind Offshore: {offshore_cap:.2f} GW")
        print(f"    - Gesamt       : {total_capacity:.2f} GW")

        # missing values -> 0
        df_full = df_full.fillna(0.0)

        # --- Schritt 3: Portfolio-CF ---
        df_full["portfolio_cf"] = (
            df_full["solar_cf"] * weights["solar_cf"]
            + df_full["wind_onshore_cf"] * weights["wind_onshore_cf"]
            + df_full["wind_offshore_cf"] * weights["wind_offshore_cf"]
        )

        # --- Schritt 4: Synthetische Erzeugung in GW ---
        df_full["solar_generation_gw"] = df_full["solar_cf"] * solar_cap
        df_full["wind_onshore_generation_gw"] = df_full["wind_onshore_cf"] * onshore_cap
        df_full["wind_offshore_generation_gw"] = df_full["wind_offshore_cf"] * offshore_cap
        df_full["total_generation_gw"] = (
            df_full["solar_generation_gw"]
            + df_full["wind_onshore_generation_gw"]
            + df_full["wind_offshore_generation_gw"]
        )

        # --- Schritt 5: Speichern ---
        output_filename = (
            f"final_portfolio_generation_{country_name.replace(' ', '_')}_"
            f"weather{weather_start}-{weather_end}_"
            f"cap{CAPACITY_YEAR}.csv"
        )
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        df_full.to_csv(output_path)

        print(f"  Finale Portfolio+Erzeugung gespeichert: {output_path}")

    print("\ncreate_final_portfolio_cordex.py erfolgreich beendet.")
