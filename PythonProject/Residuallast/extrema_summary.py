import os
import pandas as pd

# =============================================================================
# KONFIGURATION
# =============================================================================


DEMAND_TYPE = "plain"  # <-- HIER EINTRAGEN: "plain" oder "demandrcp45"


# Länder (Codes) die aufsummiert werden
#COUNTRIES_TO_SUM = ["DE", "FR", "PL",  "IT", "ES", "GB"]


COUNTRIES_TO_SUM = [
    "DE", "FR", "PL", "CZ", "AT", "CH", "IT", "ES", "BE", "NL", "DK", "SE",
    "PT", "NO", "FI", "LT", "LV", "EE", "GB"
]

# Zeitfenster in Tagen
TIME_WINDOWS_DAYS = [1, 7, 14, 21, 28, 90, 180, 365]
TOP_N_EVENTS = 30

# Welche Spalte aus den Residual-Load CSVs soll analysiert werden?
VALUE_COLUMN = "residual_load_mw"

# Alles liegt in EINEM Ordner:
INPUT_DIR = "/mnt/endata/MA_Arthur/final_results_portfolio_uncorrected/residual_load_analysis"
SUMMARY_OUTPUT_DIR = INPUT_DIR
os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)

# Output-Datei
OUTPUT_FILENAME_BASE = "extrema_summary_hydro_year"

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
# Validierung DEMAND_TYPE (muss definiert sein)
# =============================================================================
ALLOWED_DEMAND_TYPES = {"plain", "demandrcp45"}
if DEMAND_TYPE not in ALLOWED_DEMAND_TYPES:
    raise ValueError(
        "DEMAND_TYPE muss im Skript gesetzt werden!\n"
        f"Erlaubt: {sorted(ALLOWED_DEMAND_TYPES)}\n"
        "Setze z.B.:\n"
        "  DEMAND_TYPE = 'plain'\n"
        "oder\n"
        "  DEMAND_TYPE = 'demandrcp45'\n"
    )

# =============================================================================
# Helper: Dateiname je nach Demand-Typ (Suffix ja/nein)
# =============================================================================
def build_filename(country_name: str, demand_type: str) -> str:
    """
    demand_type:
      - "plain"       -> residual_load_{country}_weather2025-2050_cap2024.csv
      - "demandrcp45" -> residual_load_{country}_weather2025-2050_cap2024_demandrcp45_singlefile.csv
    """
    base = f"residual_load_{country_name}_weather2025-2050_cap2024_demandyearly_dir"
    if demand_type == "plain":
        return base + ".csv"
    elif demand_type == "demandrcp45":
        return base + "_demandrcp45_singlefile.csv"
    # unreachable durch Validierung oben
    raise ValueError(f"Unbekannter demand_type='{demand_type}'")


# =============================================================================
# HAUPTSKRIPT
# =============================================================================
if __name__ == "__main__":
    print("--- Starte Analyse (Hydrologisches Jahr: Juli-Juni) ---")
    print(f"Input Dir:     {INPUT_DIR}")
    print(f"Value Column:  {VALUE_COLUMN}")
    print(f"Demand Type:   {DEMAND_TYPE}")

    all_country_dfs = []

    for country_code in COUNTRIES_TO_SUM:
        if country_code not in COUNTRY_CODE_TO_FILENAME_MAP:
            print(f"  WARNUNG: Kein Mapping für {country_code}. Überspringe.")
            continue

        country_name = COUNTRY_CODE_TO_FILENAME_MAP[country_code]
        filename = build_filename(country_name, DEMAND_TYPE)
        filepath = os.path.join(INPUT_DIR, filename)

        if not os.path.exists(filepath):
            print(f"  FEHLER: Datei nicht gefunden für {country_code}: {filepath}")
            continue

        try:
            df_country = pd.read_csv(filepath, index_col=0, parse_dates=True)

            if VALUE_COLUMN not in df_country.columns:
                print(
                    f"  FEHLER bei {country_name}: Spalte '{VALUE_COLUMN}' fehlt. "
                    f"Vorhanden: {list(df_country.columns)} | Datei: {filename}"
                )
                continue

            df_country = df_country[[VALUE_COLUMN]].copy()
            df_country.columns = [country_code]
            all_country_dfs.append(df_country)

            print(f"  - {country_name} geladen: {df_country.shape} | {filename}")

        except Exception as e:
            print(f"  FEHLER bei {country_name}: {e}")

    if not all_country_dfs:
        print("Keine Daten geladen. Abbruch.")
        raise SystemExit(1)

    # Zusammenfügen
    df_combined = pd.concat(all_country_dfs, axis=1)

    # Aggregieren (Summe über Länder)
    df_aggregated = pd.DataFrame()
    df_aggregated["series_mw"] = df_combined.sum(axis=1, skipna=True)

    if not isinstance(df_aggregated.index, pd.DatetimeIndex):
        df_aggregated.index = pd.to_datetime(df_aggregated.index)

    print(f"Daten aggregiert. Zeitraum: {df_aggregated.index.min()} bis {df_aggregated.index.max()}")

    if df_aggregated["series_mw"].notna().sum() == 0:
        print(
            "FEHLER: Aggregierte Serie enthält nur NaN. "
            "Wahrscheinlich ist residual_load_mw leer. "
            "Setze VALUE_COLUMN z.B. auf 'generation_mw' zum Testen."
        )
        raise SystemExit(1)

    # --- Analyse der Zeitfenster ---
    all_results_dfs = []

    for days in TIME_WINDOWS_DAYS:
        print(f"  - Analysiere Zeitfenster: {days} Tage...")
        window_hours = days * 24

        if len(df_aggregated) < window_hours:
            print(f"    Zu wenig Daten für {days} Tage.")
            continue

        min_periods = int(window_hours * 0.95)

        rolling_mean = df_aggregated["series_mw"].rolling(
            window=window_hours,
            min_periods=min_periods
        ).mean()

        try:
            # Hydrologisches Jahr (Juli-Juni)
            hydro_year = rolling_mean.index.year - (rolling_mean.index.month < 7).astype(int)

            yearly_max_indices = rolling_mean.groupby(hydro_year).idxmax().dropna()
            yearly_max_values = rolling_mean.loc[yearly_max_indices]

            start_dates = yearly_max_indices - pd.Timedelta(days=days)

            df_extrema = pd.DataFrame({
                "Startdatum": start_dates.dt.date,
                "Value in GW": yearly_max_values.values / 1000.0,
            })

            df_sorted = df_extrema.sort_values(by="Value in GW", ascending=False).head(TOP_N_EVENTS)
            df_sorted.columns = [f"{days}T Start", f"GW ({days}d)"]
            df_sorted.reset_index(drop=True, inplace=True)

            all_results_dfs.append(df_sorted)

        except Exception as e:
            print(f"    FEHLER bei Analyse {days} Tage: {e}")
            continue

    if not all_results_dfs:
        print("Keine Ergebnisse generiert.")
        raise SystemExit(1)

    print("Füge Ergebnisse zusammen...")
    final_table = pd.concat(all_results_dfs, axis=1)

    print("\n" + "=" * 100)
    print(f"TOP {TOP_N_EVENTS} EREIGNISSE (Sortiert nach höchster rolling mean pro Zeitfenster)")
    print("=" * 100)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 2000,
        "display.colheader_justify", "center",
    ):
        print(final_table)

    print("=" * 100 + "\n")

    output_filepath = os.path.join(
        SUMMARY_OUTPUT_DIR,
        f"{OUTPUT_FILENAME_BASE}_{DEMAND_TYPE}.csv"
    )
    final_table.to_csv(output_filepath, index=False)
    print(f"Tabelle erfolgreich gespeichert unter:\n{output_filepath}")
