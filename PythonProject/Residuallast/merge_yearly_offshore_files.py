# merge_yearly_offshore_files_cordex.py


import os
import glob
import pandas as pd


# =============================================================================
# KONFIGURATION
# =============================================================================
YEARS_TO_PROCESS = range(1925, 2050)

COUNTRIES_TO_PROCESS = [
    "Germany", "France", "Poland","Austria", "Switzerland",
    "Italy", "Spain", "Belgium", "Netherlands", "Denmark", "Sweden",
    "Portugal", "Norway", "Finland", "Lithuania", "Latvia", "Estonia",
    "United Kingdom", "Czechia",

]
# --- MA_Arthur / CORDEX Pfade ---
MA_BASE = "/mnt/endata/MA_Arthur"
BASE_DIR = os.path.join(MA_BASE, "atlite_cf_results_offshore", "results_per_year_uncorrected")

# Speicherort für die zusammengefügten Dateien
OUTPUT_DIR = os.path.join(MA_BASE, "atlite_cf_results_offshore", "combined_all_years")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Helper
# =============================================================================
def _read_timeseries_csv(path: str) -> pd.DataFrame:
    """Robust: 'time' als Spalte oder Index in erster Spalte."""
    try:
        return pd.read_csv(path, index_col="time", parse_dates=True)
    except Exception:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "time"
        return df


def _normalize_offshore_column(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Normalisiert auf eine Spalte: 'wind_offshore_cf'
    akzeptiert 'wind_offshore_cf' oder legacy 'wind_cf'
    """
    if "wind_offshore_cf" in df.columns:
        return df[["wind_offshore_cf"]]
    if "wind_cf" in df.columns:
        return df[["wind_cf"]].rename(columns={"wind_cf": "wind_offshore_cf"})
    raise ValueError(f"Keine Offshore-CF-Spalte gefunden in {path}. Spalten: {list(df.columns)}")


# =============================================================================
# HAUPTSKRIPT
# =============================================================================
if __name__ == "__main__":
    print("Starte das Zusammenfügen der jährlichen Offshore-CF-Dateien (CORDEX/MA_Arthur)...")
    print(f"  Input-Base:  {BASE_DIR}")
    print(f"  Output-Dir:  {OUTPUT_DIR}")

    if not os.path.isdir(BASE_DIR):
        raise NotADirectoryError(f"BASE_DIR existiert nicht oder ist kein Ordner: {BASE_DIR}")

    start_year = min(YEARS_TO_PROCESS)
    end_year = max(YEARS_TO_PROCESS)

    for country_name in COUNTRIES_TO_PROCESS:
        print(f"\n--- Verarbeite {country_name} ---")

        country_folder = os.path.join(BASE_DIR, country_name.replace(" ", "_"))
        if not os.path.isdir(country_folder):
            print(f"  WARNUNG: Ordner für {country_name} nicht gefunden. Überspringe: {country_folder}")
            continue

        yearly_dfs = []
        files_found = 0
        missing_years = 0

        for year in YEARS_TO_PROCESS:
            filepath = os.path.join(country_folder, f"offshore_wind_cf_{year}.csv")

            if not os.path.exists(filepath):
                # Fallback: falls Naming leicht abweicht
                matches = sorted(glob.glob(os.path.join(country_folder, f"*{year}*.csv")))
                if not matches:
                    missing_years += 1
                    continue
                filepath = matches[0]

            df_year = _read_timeseries_csv(filepath)
            df_year = _normalize_offshore_column(df_year, filepath)

            yearly_dfs.append(df_year)
            files_found += 1

        if not yearly_dfs:
            print(f"  WARNUNG: Keine Jahresdateien für {country_name} gefunden.")
            continue

        print(f"  {files_found} Jahresdateien gefunden. Füge sie zusammen... (fehlende Jahre: {missing_years})")

        country_full_timeseries = pd.concat(yearly_dfs, axis=0)
        country_full_timeseries.sort_index(inplace=True)

        duplicates = int(country_full_timeseries.index.duplicated().sum())
        if duplicates > 0:
            print(f"  WARNUNG: {duplicates} doppelte Zeitstempel im zusammengefügten DataFrame für {country_name} gefunden!")
            # Optional: Duplikate entfernen (hier bewusst NICHT automatisch, um Probleme sichtbar zu lassen)

        output_filename = f"uncorrected_offshore_cf_{country_name.replace(' ', '_')}_{start_year}-{end_year}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        country_full_timeseries.to_csv(output_path, index_label="time")
        print(f"  Zusammengefügte Datei gespeichert unter: {output_path}")
        print(f"  Gesamtzahl der Zeitschritte: {len(country_full_timeseries)}")

    print("\n\nSkript erfolgreich beendet.")
