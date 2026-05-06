# calc residual load
from __future__ import annotations

import os
import glob
import logging
import argparse
from typing import Iterable, List, Optional, Tuple

import pandas as pd


# =============================================================================
# KONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Länder, für die die Residual-Last berechnet werden soll
COUNTRIES_TO_ANALYZE = [
    "DE", "FR", "PL", "CZ", "AT", "CH", "IT", "ES", "BE", "NL", "DK", "SE",
    "PT", "NO", "FI", "LT", "LV", "EE", "GB",
]

# Demand: Modus
# - "yearly_dir"        => demand_{year}/electricity_demand_non_hist.csv (alte Variante)
# - "rcp45_singlefile"  => demand_{country}_scenDN_ALL_1_ICHECK-rcp-45.csv (neue Variante)
DEFAULT_DEMAND_SOURCE = "yearly_dir"

# Demand: alte Variante (jährliche Ordnerstruktur)
DEMAND_BASE_DIR_YEARLY = "/home/endata/PycharmProjects/pypsa-ee/resources"
DEMAND_DIR_PATTERN = "demand-test-{year}"
DEMAND_FILENAME = "electricity_demand_non_hist.csv"

# Demand: neue Variante (ein File pro Land, lange Zeitreihe)
DEMAND_BASE_DIR_RCP45 = "/mnt/endata/Cordex/Hourly_profiles_rcp4.5-8.5_2011-2100/Demand_rcp45"
DEMAND_RCP45_FILENAME_PATTERN = "demand_{country}_scenDN_ALL_1_ICHEC-rcp45.csv"
DEMAND_RCP45_VALUE_COLUMN = "total"  # <- user requirement

# Generationen aus MA_Arthur
GENERATION_DATA_DIR = "/mnt/endata/MA_Arthur/final_results_portfolio_uncorrected/"

# Zielordner für die Ergebnisdateien
OUTPUT_DIR = os.path.join(GENERATION_DATA_DIR, "residual_load_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Mapping von Ländercodes zu den Dateinamen-Formaten ---
COUNTRY_CODE_TO_FILENAME_MAP = {
    "DE": "Germany",
    "FR": "France",
    "GB": "United_Kingdom",
    "ES": "Spain",
    "IT": "Italy",
    "PL": "Poland",
    "NL": "Netherlands",
    "BE": "Belgium",
    "DK": "Denmark",
    "CZ": "Czechia",
    "AT": "Austria",
    "CH": "Switzerland",
    "EE": "Estonia",
    "SE": "Sweden",
    "PT": "Portugal",
    "NO": "Norway",
    "FI": "Finland",
    "LT": "Lithuania",
    "LV": "Latvia",
}


# =============================================================================
# Helper: robustes Einlesen Generation (time-Spalte oder Index=0)
# =============================================================================
def read_ts_csv(path: str) -> pd.DataFrame:
    """Robust: 'time' als Spalte oder Zeitindex in Spalte 0."""
    try:
        return pd.read_csv(path, index_col="time", parse_dates=True)
    except Exception:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "time"
        return df


# =============================================================================
# Helper: Demand lesen (robust CSV -> DatetimeIndex)
# =============================================================================
def read_demand_csv(path: str) -> pd.DataFrame:
    """
    Robust:
    - akzeptiert explizite Zeitspalten (Date/date/time/Timestamp/...) WENN vorhanden
    - sonst interpretiert die *erste Spalte* als Zeitstempel
    Return: DataFrame mit DatetimeIndex 'time' und numerischen Spalten.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Demand-Datei ist leer: {path}")

    # Kandidaten für Zeitspalte prüfen
    time_col = None
    for cand in ["Date", "date", "time", "Time", "timestamp", "Timestamp", "datetime", "Datetime"]:
        if cand in df.columns:
            time_col = cand
            break

    # Wenn keine Zeitspalte vorhanden: erste Spalte verwenden
    if time_col is None:
        time_col = df.columns[0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col)
    df.index.name = "time"

    # sortieren + duplikate entfernen
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # numerisch erzwingen (robust gegen Strings)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# =============================================================================
# Helper: Jahr im Zeitindex ersetzen (2024 -> Zieljahr), robust ggü. Feb 29
# =============================================================================
def replace_index_year(df: pd.DataFrame, target_year: int, *, assume_source_year: Optional[int] = None) -> pd.DataFrame:
    """
    Ersetzt das Jahr im DatetimeIndex durch target_year (Monat/Tag/Uhrzeit bleiben).
    Typischer Use-Case: Demand-Dateien sind pro Jahr korrekt benannt, tragen aber im Index immer 2024.

    - assume_source_year: wenn gesetzt, wird nur dann remapped, wenn df.index.year == assume_source_year (oder dessen Set).
      Wenn None: remapped immer.

    Feb 29 Handling:
    - Wenn Zieljahr kein Schaltjahr ist und Feb 29 vorkommt, wird auf Feb 28 gemappt (gleiche Uhrzeit).
      (Alternativ könnte man droppen; hier: konservativ behalten.)
    """
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("replace_index_year erwartet einen DatetimeIndex.")

    if assume_source_year is not None:
        years_present = set(int(y) for y in df.index.year.unique())
        if years_present != {int(assume_source_year)}:
            # Nicht der erwartete "alles ist 2024"-Fall -> nichts anfassen
            return df

    idx = df.index
    new_vals = []
    feb29_fixed = 0

    for ts in idx:
        try:
            new_vals.append(ts.replace(year=int(target_year)))
        except ValueError:
            # i.d.R. 29. Feb in Nicht-Schaltjahr
            if ts.month == 2 and ts.day == 29:
                feb29_fixed += 1
                new_vals.append(ts.replace(year=int(target_year), day=28))
            else:
                raise

    new_index = pd.DatetimeIndex(new_vals, name=idx.name)

    out = df.copy()
    out.index = new_index
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]

    if feb29_fixed:
        logger.warning(
            "replace_index_year: %d Timestamp(s) am 29.02. wurden auf 28.02. gemappt (target_year=%s).",
            feb29_fixed, target_year
        )

    return out


# =============================================================================
# Demand-Variante 1: jährliche Ordnerstruktur (wie bisher)
# =============================================================================
def find_demand_file_for_year(base_dir: str, year: int) -> Optional[str]:
    """
    Sucht:
      {base_dir}/demand_20XY/electricity_demand_non_hist.csv
    """
    demand_dir = os.path.join(base_dir, DEMAND_DIR_PATTERN.format(year=year))
    preferred = os.path.join(demand_dir, DEMAND_FILENAME)
    if os.path.exists(preferred):
        return preferred

    # Fallback: falls Naming leicht anders ist
    matches = sorted(glob.glob(os.path.join(demand_dir, "*demand*non_hist*.csv")))
    if matches:
        return matches[0]
    return None


def load_demand_all_years_yearly(country_code: str, years: Iterable[int], base_dir: str) -> pd.DataFrame:
    """
    Lädt Demand-Spalte für country_code (z.B. 'DE') über alle Jahre und konkateniert.
    Erwartet pro Jahr eine Datei mit Spalte country_code.
    Output: DataFrame mit Spalte 'demand_mw' und Index 'time'

    WICHTIG (Fix):
    - In deinen Demand-Dateien steht im Zeitindex immer das Jahr 2024, obwohl die Datei per Namen/Jahresordner eindeutig
      dem Zieljahr zugeordnet ist (2025-2050).
    - Deshalb wird pro Datei das Jahr im Index auf das jeweilige Zieljahr ersetzt.
    """
    frames: List[pd.DataFrame] = []

    for y in years:
        path = find_demand_file_for_year(base_dir, y)
        if path is None:
            logger.warning(
                "Demand-Datei nicht gefunden (year=%s) unter %s/%s",
                y, base_dir, DEMAND_DIR_PATTERN.format(year=y),
            )
            continue

        df = read_demand_csv(path)

        if country_code not in df.columns:
            raise ValueError(
                f"Demand-Datei enthält keine Spalte '{country_code}': {path} | cols={list(df.columns)}"
            )

        # ---- FIX: Jahr im Index auf Zieljahr setzen (typisch: Quelle = 2024) ----
        # Wir remappen NUR, wenn wirklich alles auf 2024 steht, damit "korrekte" Files nicht kaputt gemacht werden.
        df = replace_index_year(df, target_year=int(y), assume_source_year=2024)

        frames.append(df[[country_code]].rename(columns={country_code: "demand_mw"}))

    if not frames:
        raise FileNotFoundError(f"Keine Demand-Dateien gefunden für {country_code} in {base_dir}")

    out = pd.concat(frames, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


# =============================================================================
# Demand-Variante 2: ein RCP45-File pro Land (neu)
# =============================================================================
def find_demand_file_rcp45(country_code: str, base_dir: str) -> str:
    """
    Sucht:
      {base_dir}/demand_{country}_scenDN_ALL_1_ICHECK-rcp-45.csv
    """
    filename = DEMAND_RCP45_FILENAME_PATTERN.format(country=country_code)
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"RCP45-Demand-Datei nicht gefunden: {path}")
    return path


def load_demand_all_years_rcp45(country_code: str, years: Iterable[int], base_dir: str) -> pd.DataFrame:
    """
    Lädt RCP45-Demand aus einem einzelnen Länderfile und schneidet auf 'years' zu.
    Verwendet Spalte 'total' (DEMAND_RCP45_VALUE_COLUMN).
    Output: DataFrame mit Spalte 'demand_mw' und Index 'time'
    """
    path = find_demand_file_rcp45(country_code, base_dir)
    df = read_demand_csv(path)

    if DEMAND_RCP45_VALUE_COLUMN not in df.columns:
        raise ValueError(
            f"RCP45-Demand-Datei enthält keine Spalte '{DEMAND_RCP45_VALUE_COLUMN}': {path} | cols={list(df.columns)}"
        )

    out = df[[DEMAND_RCP45_VALUE_COLUMN]].rename(columns={DEMAND_RCP45_VALUE_COLUMN: "demand_mw"}).copy()

    years_set = set(int(y) for y in years)
    out = out[out.index.year.isin(years_set)]

    if out.empty:
        raise ValueError(
            f"RCP45-Demand hat nach Year-Filter keine Daten für {country_code} (years={sorted(years_set)}). "
            f"Check Zeitachse in {path}."
        )

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def load_demand(country_code: str, years: Iterable[int], demand_source: str) -> pd.DataFrame:
    """
    Wrapper, der je nach demand_source die passende Demand-Ladefunktion verwendet.
    """
    if demand_source == "yearly_dir":
        return load_demand_all_years_yearly(country_code, years, base_dir=DEMAND_BASE_DIR_YEARLY)
    elif demand_source == "rcp45_singlefile":
        return load_demand_all_years_rcp45(country_code, years, base_dir=DEMAND_BASE_DIR_RCP45)
    else:
        raise ValueError(f"Unbekannte demand_source='{demand_source}'. Erlaubt: yearly_dir, rcp45_singlefile")


# =============================================================================
# Helper: Generation-File finden (damit weather/cap Labels nicht hart codiert sind)
# =============================================================================
def find_generation_file(country_name: str) -> Optional[Tuple[str, str, str]]:
    """
    Findet die passende Datei:
      final_portfolio_generation_<Country>_weatherYYYY-YYYY_capYYYY.csv

    Return:
      (filepath, weather_tag, cap_tag)   z.B. (..., "2025-2049", "2024")
    """
    patt = os.path.join(GENERATION_DATA_DIR, f"final_portfolio_generation_{country_name}_weather*_cap*.csv")
    matches = sorted(glob.glob(patt))
    if not matches:
        return None

    # Wenn mehrere: nimm die lexikographisch letzte
    path = matches[-1]
    base = os.path.basename(path)

    # parse weather/cap aus filename
    try:
        weather_part = base.split("_weather", 1)[1]
        weather_tag = weather_part.split("_cap", 1)[0]
        cap_tag = weather_part.split("_cap", 1)[1].replace(".csv", "")
    except Exception:
        weather_tag = "unknown"
        cap_tag = "unknown"

    return path, weather_tag, cap_tag


def infer_years_from_generation_index(df_generation: pd.DataFrame) -> List[int]:
    """Nimmt Jahre aus Generation-Zeitachse."""
    return sorted(set(df_generation.index.year.tolist()))


def unique_countries(seq: List[str]) -> List[str]:
    """Dedupliziert, erhält Reihenfolge."""
    seen = set()
    out = []
    for x in seq:
        x = str(x).strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# =============================================================================
# HAUPTSKRIPT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute residual load from demand and generation time series.")
    parser.add_argument(
        "--demand-source",
        choices=["yearly_dir", "rcp45_singlefile"],
        default=DEFAULT_DEMAND_SOURCE,
        help="Which demand input format to use."
    )
    args = parser.parse_args()

    countries = unique_countries(COUNTRIES_TO_ANALYZE)

    print("--- Starte Berechnung der stündlichen Residual-Last ---")
    print(f"--- Demand-Quelle: {args.demand_source} ---")

    for country_code in countries:
        print(f"\n--- Verarbeite {country_code} ---")

        country_name = COUNTRY_CODE_TO_FILENAME_MAP.get(country_code)
        if not country_name:
            print(f"  WARNUNG: Kein Dateiname für Ländercode {country_code} im Mapping gefunden. Überspringe.")
            continue

        # --- Schritt 1: Lade Erzeugungsdaten (robust: finde Datei automatisch) ---
        found = find_generation_file(country_name)
        if found is None:
            print(f"  WARNUNG: Keine Generation-Datei gefunden für {country_name} in {GENERATION_DATA_DIR}")
            continue

        generation_filepath, weather_tag, cap_tag = found
        df_generation = read_ts_csv(generation_filepath)

        if "total_generation_gw" not in df_generation.columns:
            print(f"  WARNUNG: Spalte 'total_generation_gw' fehlt in {generation_filepath}. Überspringe.")
            continue

        df_generation = df_generation[["total_generation_gw"]]
        print(f"  Generation geladen: {os.path.basename(generation_filepath)}")

        # --- Schritt 2: Lade Demand passend zu den Generation-Jahren ---
        years_needed = infer_years_from_generation_index(df_generation)
        if not years_needed:
            print("  WARNUNG: Konnte keine Jahre aus Generation-Zeitachse ableiten. Überspringe.")
            continue

        try:
            df_demand_country = load_demand(country_code, years_needed, demand_source=args.demand_source)
            # Einheiten: Demand ist MW (laut deiner Annahme/Datei). Generation ist GW -> wir rechnen Generation in MW.
        except Exception as e:
            print(f"  FEHLER beim Laden des Demand für {country_code}: {e}")
            continue

        # --- Schritt 3: Join auf Zeitachse (inner, damit residual sauber ist) ---
        df_combined = df_demand_country.join(df_generation, how="inner")
        if df_combined.empty:
            print("  WARNUNG: Join Demand/Generation ergibt leeren DF (Zeitachsen mismatch). Überspringe.")
            continue

        # --- Schritt 4: Residual Load ---
        df_combined["generation_mw"] = df_combined["total_generation_gw"] * 1000.0
        df_combined["residual_load_mw"] = df_combined["demand_mw"] - df_combined["generation_mw"]

        # --- Schritt 5: Export ---
        df_output = df_combined[["demand_mw", "generation_mw", "residual_load_mw"]]

        output_filename = f"residual_load_{country_name}_weather{weather_tag}_cap{cap_tag}_demand{args.demand_source}.csv"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)

        df_output.to_csv(output_filepath, index_label="time")
        print(f"  Ergebnis gespeichert: {output_filepath}")

        # Join-Qualität kurz berichten
        coverage = len(df_output) / len(df_generation) if len(df_generation) else 0.0
        if coverage < 0.99:
            logger.warning(
                "Zeitachsen-Coverage (%s): %.2f%% (Demand/Generation inner join). Generation rows=%d, joined=%d",
                country_code, 100 * coverage, len(df_generation), len(df_output)
            )

    print("\n\n--- Alle Länder verarbeitet. ---")