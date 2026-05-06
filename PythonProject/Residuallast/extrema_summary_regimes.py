# extrema summary for regimes (MIDPOINT regime assignment) + EXTRA: TRANSITION/OVERLAP search
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# KONFIGURATION
# =============================================================================

ANALYSIS_TYPE = "SYNTHETIC_DEMAND"

# -------------------------
# Run-Parameter
# -------------------------
PERSISTENCE_DAYS = 5
K_CLUSTERS = 8
Y0, Y1 = 2025, 2050

RUN_TAG = f"p{PERSISTENCE_DAYS}d_k{K_CLUSTERS}_{Y0}_{Y1}"

# Länder (Codes) die aufsummiert werden
COUNTRIES_TO_SUM = [
    "DE", "FR", "PL", "CZ", "AT", "CH", "IT", "ES", "BE", "NL", "DK", "SE",
    "PT", "NO", "FI", "LT", "LV", "EE", "GB"
]
# Zeitfenster in Tagen
TIME_WINDOWS_DAYS = [1, 7, 14, 21, 28, 90, 180, 365]
TOP_N_EVENTS = 30

VALUE_COLUMN = "residual_load_mw"

MA_BASE = "/mnt/endata/MA_Arthur"
BASE_RESIDUAL_DIR = os.path.join(
    MA_BASE,
    "final_results_portfolio_uncorrected",
    "residual_load_analysis"
)

# --- Regime-Daten ---
REGIME_DIR = os.path.join(
    "/mnt/endata/Cordex/Copernicus_CORDEX_data",
    f"results_grams_strict_p{PERSISTENCE_DAYS}d_k{K_CLUSTERS}_{Y0}_{Y1}",
)
REGIME_PATTERN = r"regime_periods_(\d{4})_3hourly\.csv$"

REGIME_NAME_MAP: Optional[Dict[int, str]] = None

NO_REGIME_ALIASES_STR = {"no_regime", "none", "nan", "no-regime", "no regime", "noregime"}
NO_REGIME_ALIASES_INT = {-1, 99}
NO_REGIME_LABEL = "no_regime"

# Rolling coverage (Datenverfügbarkeit)
ROLLING_MIN_COVERAGE = 0.95

# Regime-Kohärenz im Fenster (Anteil der Steps im Regime) für Pro-Regime-Events
REGIME_WINDOW_COVERAGE = 0.75

# Regime-Zuordnung am Fenstermittelpunkt, optional zusätzlich Kohärenz-Filter
APPLY_REGIME_COVERAGE_FILTER = True  # False => nur Midpoint-Regime zählt, keine Mindest-Frac

# -------------------------
# NEU: Transition/Overlap Suche (Regime-Übergänge)
# -------------------------
ENABLE_TRANSITION_SEARCH = True

# Definition "Overlap/Transition": im Fenster müssen mind. 2 verschiedene Regime substanziell auftreten
# Empfehlung: 0.80/0.20 oder strenger 0.70/0.30
TRANSITION_TOP1_MAX_FRAC = 0.70  # Top-1 Regime-Anteil darf höchstens so groß sein
TRANSITION_TOP2_MIN_FRAC = 0.30  # Top-2 Regime-Anteil muss mindestens so groß sein

# Optional: no_regime in Transition-Analyse erlauben oder ausschließen?
# True  => Übergänge inkl. no_regime (z.B. Regime 1 <-> no_regime)
# False => nur Übergänge zwischen "echten" Regimen (no_regime wird ignoriert)
TRANSITION_INCLUDE_NO_REGIME = True

# Ausgabepräfix für Transition
TRANSITION_OUTPUT_FILENAME_PREFIX = "extrema_summary_transition_overlap_midpoint_synthetic_demand_hydro_year"

if ANALYSIS_TYPE == "SYNTHETIC_DEMAND":
    INPUT_DIR = BASE_RESIDUAL_DIR
    FILENAME_PATTERN = "residual_load_{country_name}_weather2025-2050_cap2024_demandyearly_dir.csv"
    OUTPUT_FILENAME_PREFIX = "extrema_summary_regime_based_midpoint_synthetic_demand_hydro_year"
elif ANALYSIS_TYPE == "DEMAND_2024":
    INPUT_DIR = os.path.join(BASE_RESIDUAL_DIR, "residual_load_analysis_with_baseyear_2024")
    FILENAME_PATTERN = "residual_load_{country_name}_based_on_2024_demand.csv"
    OUTPUT_FILENAME_PREFIX = "extrema_summary_regime_based_midpoint_based_on_2024_demand_hydro_year"
    TRANSITION_OUTPUT_FILENAME_PREFIX = "extrema_summary_transition_overlap_midpoint_based_on_2024_demand_hydro_year"
else:
    raise ValueError("Ungültiger ANALYSIS_TYPE.")

# -------------------------
# Run-spezifischer Output-Ordner
# -------------------------
SUMMARY_OUTPUT_DIR = os.path.join(
    BASE_RESIDUAL_DIR,
    "regime_extrema_runs",
    RUN_TAG,
)
os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)

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
# HELPER: Regime Loading + Normalisierung + Alignment
# =============================================================================

def _parse_datetime_series(s: pd.Series) -> pd.Series:
    """Robustes datetime parsing."""
    return pd.to_datetime(s, errors="coerce", utc=False).dt.tz_localize(None)


def _detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _expand_period_3h_left_closed(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Expandiert [start, end) auf 3h Raster (end-exklusiv).
    Das verhindert Overlap an Regime-Grenzen in der Regime-TS.
    """
    if pd.isna(start) or pd.isna(end) or end <= start:
        return pd.DatetimeIndex([])

    try:
        return pd.date_range(start, end, freq="3h", inclusive="left")
    except TypeError:
        end_adj = end - pd.Timedelta(hours=3)
        if end_adj < start:
            return pd.DatetimeIndex([])
        return pd.date_range(start, end_adj, freq="3h")


def _read_regime_files(regime_dir: str) -> pd.DataFrame:
    """
    Liest alle regime_periods_20xx_3hourly.csv und liefert:
      columns: time, regime

    Unterstützt:
      A) Timeseries: time + regime/cluster/label
      B) Perioden: start/end + regime/cluster/label (expandiert auf 3h Raster, END-EXKLUSIV!)
    """
    p = Path(regime_dir)
    files = sorted([f for f in p.iterdir() if f.is_file() and re.search(REGIME_PATTERN, f.name)])
    if not files:
        raise FileNotFoundError(f"Keine Regime-Dateien gefunden in: {regime_dir}")

    all_ts = []
    for f in files:
        df = pd.read_csv(f)

        time_col = _detect_column(df, ["time", "datetime", "date", "timestamp"])
        regime_col = _detect_column(df, ["regime", "cluster", "label", "wr", "weather_regime", "kmeans_label"])

        if time_col is None and len(df.columns) > 0:
            c0 = df.columns[0]
            parsed0 = pd.to_datetime(df[c0], errors="coerce")
            if parsed0.notna().mean() > 0.9:
                time_col = c0

        start_col = _detect_column(df, ["start", "start_time", "start_datetime", "begin"])
        end_col = _detect_column(df, ["end", "end_time", "end_datetime", "stop", "finish"])

        # A) timeseries
        if time_col is not None and regime_col is not None and (start_col is None or end_col is None):
            ts = pd.DataFrame({
                "time": _parse_datetime_series(df[time_col]),
                "regime": df[regime_col],
            }).dropna(subset=["time"])
            all_ts.append(ts)
            continue

        # B) periods
        if start_col is not None and end_col is not None and regime_col is not None:
            starts = _parse_datetime_series(df[start_col])
            ends = _parse_datetime_series(df[end_col])
            regs = df[regime_col]

            tmp = pd.DataFrame({"start": starts, "end": ends, "reg": regs}).dropna(subset=["start", "end"])
            tmp = tmp.sort_values("start")
            if len(tmp) > 1:
                overlaps = (tmp["start"].iloc[1:].values < tmp["end"].iloc[:-1].values)
                if np.any(overlaps):
                    print(f"  WARNUNG: Overlap in Periodendatei erkannt: {f.name} (start < previous end).")

            rows = []
            for s, e, r in zip(starts, ends, regs):
                idx = _expand_period_3h_left_closed(s, e)
                if len(idx) == 0:
                    continue
                rows.append(pd.DataFrame({"time": idx, "regime": r}))

            if rows:
                all_ts.append(pd.concat(rows, ignore_index=True))
            continue

        raise ValueError(f"Unbekanntes Regime-Dateiformat: {f}")

    regime_ts = pd.concat(all_ts, ignore_index=True)
    regime_ts = regime_ts.dropna(subset=["time"]).sort_values("time")
    regime_ts = regime_ts.drop_duplicates(subset=["time"], keep="last")
    return regime_ts


def _normalize_regime_labels(regime_s: pd.Series) -> Tuple[pd.Series, List[str], str]:
    """
    Normalisiert Regime-Labels.

    WICHTIG:
      - Numerischer Labelwert == K_CLUSTERS wird IMMER als NO_REGIME_LABEL gemappt.
    """
    s = regime_s.copy()
    DYNAMIC_NO_REGIME_INT = int(K_CLUSTERS)

    def _clean_obj(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            t = x.strip()
            return np.nan if t == "" else t
        return x

    s = s.map(_clean_obj)

    def _to_int_if_possible(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float) and float(x).is_integer():
            return int(x)
        if isinstance(x, str):
            try:
                return int(x)
            except Exception:
                return x.lower().strip()
        return x

    s2 = s.map(_to_int_if_possible)

    def _is_no_regime(v) -> bool:
        if pd.isna(v):
            return True
        if isinstance(v, str) and v.lower().strip() in NO_REGIME_ALIASES_STR:
            return True
        if isinstance(v, (int, np.integer)):
            iv = int(v)
            if iv == DYNAMIC_NO_REGIME_INT:
                return True
            if iv in NO_REGIME_ALIASES_INT:
                return True
        return False

    s2 = s2.map(lambda v: NO_REGIME_LABEL if _is_no_regime(v) else v)

    if REGIME_NAME_MAP is not None:
        def _map_reg(v):
            if v == NO_REGIME_LABEL:
                return NO_REGIME_LABEL
            if isinstance(v, (int, np.integer)) and int(v) in REGIME_NAME_MAP:
                return REGIME_NAME_MAP[int(v)]
            return str(v)

        s2 = s2.map(_map_reg).astype("object")
        uniq_main = [u for u in pd.unique(s2.dropna()) if u != NO_REGIME_LABEL]
        uniq_main = list(dict.fromkeys(uniq_main))
        regimes_ordered = uniq_main[:7] + [NO_REGIME_LABEL]
        return s2, regimes_ordered, NO_REGIME_LABEL

    # Ohne Mapping: numerisch bevorzugt
    uniq = pd.unique(s2.dropna())
    uniq_main = [u for u in uniq if u != NO_REGIME_LABEL]

    numeric_main = [u for u in uniq_main if isinstance(u, (int, np.integer))]
    if numeric_main:
        numeric_sorted = sorted(set(int(u) for u in numeric_main))
        regimes_main = [str(u) for u in numeric_sorted][:7]
        s2 = s2.map(lambda v: str(int(v)) if isinstance(v, (int, np.integer)) else v).astype("object")
        regimes_ordered = regimes_main + [NO_REGIME_LABEL]
        return s2, regimes_ordered, NO_REGIME_LABEL

    string_main = sorted(set(str(u) for u in uniq_main))[:7]
    s2 = s2.astype("object")
    regimes_ordered = string_main + [NO_REGIME_LABEL]
    return s2, regimes_ordered, NO_REGIME_LABEL


def _align_regimes_to_index(regime_ts: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.Series:
    """
    Regime-Zeitreihe -> auf target_index per forward fill.
    Erwartung: regime_ts ist Zustandsvariable, die bis zum nächsten Timestamp gilt.
    """
    regime_series = regime_ts.set_index("time")["regime"].sort_index()
    regime_series = regime_series.loc[(regime_series.index <= target_index.max())]
    aligned = regime_series.reindex(target_index, method="ffill")
    aligned = aligned.fillna(NO_REGIME_LABEL).astype("object")
    return aligned


def _infer_step_timedelta(idx: pd.DatetimeIndex) -> pd.Timedelta:
    """
    Ermittelt die zeitliche Schrittweite des Index robust.
    """
    freq = pd.infer_freq(idx)
    if freq is not None:
        try:
            return pd.Timedelta(freq)
        except Exception:
            pass

    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        raise ValueError("Kann Schrittweite nicht bestimmen: Index hat zu wenige Zeitpunkte.")
    step = diffs.median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("Kann Schrittweite nicht bestimmen: median diff ungültig.")
    return step


def _top2_regimes_in_window(
    regime_aligned: pd.Series,
    end_times: pd.DatetimeIndex,
    window_steps: int,
    include_no_regime: bool,
    no_regime_label: str
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Für jede end_time (Fensterende, wie beim rolling) wird im Fenster [end-(W-1)*step, end] (W steps)
    die Top-1 und Top-2 Regime (Label + Anteil) bestimmt.

    Rückgabe:
      top1_label, top1_frac, top2_label, top2_frac  (alles index=end_times)
    """
    if window_steps < 1:
        raise ValueError("window_steps < 1")

    idx = regime_aligned.index
    pos = idx.get_indexer(end_times)
    if (pos < 0).any():
        # end_times sollten aus idx stammen; wenn nicht: reindex/ffill vorher lösen
        raise ValueError("Einige end_times sind nicht im regime_aligned Index enthalten.")

    top1_lab = []
    top1_frac = []
    top2_lab = []
    top2_frac = []

    # Zugriff als numpy-array für Geschwindigkeit
    reg_vals = regime_aligned.astype("object").values

    for p in pos:
        start_p = p - (window_steps - 1)
        if start_p < 0:
            # Fenster nicht vollständig im Index (sollte bei rolling idxmax selten passieren)
            win = reg_vals[0:p + 1]
        else:
            win = reg_vals[start_p:p + 1]

        if not include_no_regime:
            win = np.array([w for w in win if w != no_regime_label], dtype=object)

        if win.size == 0:
            top1_lab.append(np.nan)
            top1_frac.append(np.nan)
            top2_lab.append(np.nan)
            top2_frac.append(np.nan)
            continue

        # counts
        uniq, cnt = np.unique(win, return_counts=True)
        order = np.argsort(cnt)[::-1]
        uniq = uniq[order]
        cnt = cnt[order].astype(float)
        total = cnt.sum() if cnt.sum() > 0 else 1.0

        t1_lab = uniq[0]
        t1_frac = cnt[0] / total

        if len(uniq) > 1:
            t2_lab = uniq[1]
            t2_frac = cnt[1] / total
        else:
            t2_lab = np.nan
            t2_frac = 0.0

        top1_lab.append(str(t1_lab))
        top1_frac.append(float(t1_frac))
        top2_lab.append(str(t2_lab) if not (isinstance(t2_lab, float) and np.isnan(t2_lab)) else np.nan)
        top2_frac.append(float(t2_frac))

    s1 = pd.Series(top1_lab, index=end_times, dtype="object")
    s1f = pd.Series(top1_frac, index=end_times, dtype="float64")
    s2 = pd.Series(top2_lab, index=end_times, dtype="object")
    s2f = pd.Series(top2_frac, index=end_times, dtype="float64")
    return s1, s1f, s2, s2f


# =============================================================================
# HAUPTSKRIPT
# =============================================================================
if __name__ == "__main__":
    print("--- Starte Analyse (Hydrologisches Jahr: Juli-Juni) ---")
    print(f"Input Dir:     {INPUT_DIR}")
    print(f"Value Column:  {VALUE_COLUMN}")
    print(f"Regime Dir:    {REGIME_DIR}")
    print(f"Time windows:  {TIME_WINDOWS_DAYS}")
    print(f"Top N:         {TOP_N_EVENTS}")
    print(f"Data coverage (rolling min):   {ROLLING_MIN_COVERAGE:.2f}")
    print(f"Regime coverage (window frac): {REGIME_WINDOW_COVERAGE:.2f} (nur Pro-Regime)")
    print(f"Apply regime coverage filter:  {APPLY_REGIME_COVERAGE_FILTER}")
    print("Regime assignment:             MIDPOINT\n")
    print(f"Transition search enabled:      {ENABLE_TRANSITION_SEARCH}")
    if ENABLE_TRANSITION_SEARCH:
        print(f"  Transition criteria: top1_frac <= {TRANSITION_TOP1_MAX_FRAC:.2f} "
              f"AND top2_frac >= {TRANSITION_TOP2_MIN_FRAC:.2f}")
        print(f"  Include no_regime in transition: {TRANSITION_INCLUDE_NO_REGIME}")
    print()

    # -------------------------------------------------------------------------
    # 1) Residual Load laden & aggregieren
    # -------------------------------------------------------------------------
    all_country_dfs = []

    for country_code in COUNTRIES_TO_SUM:
        if country_code not in COUNTRY_CODE_TO_FILENAME_MAP:
            print(f"  WARNUNG: Kein Mapping für {country_code}. Überspringe.")
            continue

        country_name = COUNTRY_CODE_TO_FILENAME_MAP[country_code]
        filename = FILENAME_PATTERN.format(country_name=country_name)
        filepath = os.path.join(INPUT_DIR, filename)

        if not os.path.exists(filepath):
            print(f"  FEHLER: Datei nicht gefunden für {country_code}: {filepath}")
            continue

        try:
            df_country = pd.read_csv(filepath, index_col=0, parse_dates=True)

            if VALUE_COLUMN not in df_country.columns:
                print(f"  FEHLER bei {country_name}: Spalte '{VALUE_COLUMN}' fehlt. "
                      f"Vorhanden: {list(df_country.columns)}")
                continue

            df_country = df_country[[VALUE_COLUMN]].copy()
            df_country.columns = [country_code]
            all_country_dfs.append(df_country)

            print(f"  - {country_name} geladen: {df_country.shape}")

        except Exception as e:
            print(f"  FEHLER bei {country_name}: {e}")

    if not all_country_dfs:
        print("Keine Daten geladen. Abbruch.")
        raise SystemExit(1)

    df_combined = pd.concat(all_country_dfs, axis=1)

    df_aggregated = pd.DataFrame()
    df_aggregated["series_mw"] = df_combined.sum(axis=1, skipna=True)

    if not isinstance(df_aggregated.index, pd.DatetimeIndex):
        df_aggregated.index = pd.to_datetime(df_aggregated.index)

    df_aggregated.index = df_aggregated.index.tz_localize(None)

    print(f"\nDaten aggregiert. Zeitraum: {df_aggregated.index.min()} bis {df_aggregated.index.max()}")

    if df_aggregated["series_mw"].notna().sum() == 0:
        print("FEHLER: Aggregierte Serie enthält nur NaN. "
              "Wahrscheinlich ist demand/residual_load noch nicht befüllt. "
              "Setze VALUE_COLUMN z.B. auf 'generation_mw' zum Testen.")
        raise SystemExit(1)

    # Schrittweite bestimmen (z.B. 3h)
    step_td = _infer_step_timedelta(df_aggregated.index)
    print(f"Erkannte Zeitauflösung: {step_td} (Index steps)")

    # -------------------------------------------------------------------------
    # 2) Regime laden + normalisieren + alignen auf Index
    # -------------------------------------------------------------------------
    print("\nLade Regime-Daten...")
    regime_ts_raw = _read_regime_files(REGIME_DIR)
    regime_norm, regimes_ordered, no_regime_label = _normalize_regime_labels(regime_ts_raw["regime"])
    regime_ts_raw["regime"] = regime_norm

    regime_aligned = _align_regimes_to_index(regime_ts_raw, df_aggregated.index)

    counts = regime_aligned.value_counts(dropna=False)
    print("Regime-Verteilung (aligned) [counts]:")
    print(counts.to_string())

    main_labels = [r for r in counts.index.tolist() if r != no_regime_label]
    if len(main_labels) > K_CLUSTERS:
        main_labels = main_labels[:K_CLUSTERS]
    regimes_final = main_labels + [no_regime_label]
    print(f"Regimes im Output: {regimes_final}")

    # -------------------------------------------------------------------------
    # 3) Pro-Regime Ausgabe + Sammeln für Overall-Zusammenfassung
    # -------------------------------------------------------------------------
    per_regime_tables: Dict[str, pd.DataFrame] = {}
    collected_events_long: List[pd.DataFrame] = []

    for regime_label in regimes_final:
        reg_str = str(regime_label)

        print(f"\n{'=' * 100}")
        print(f"Regime: {reg_str}")
        print(f"{'=' * 100}")

        mask_regime = (regime_aligned.astype("object") == reg_str).astype(float)
        all_results_dfs = []

        for days in TIME_WINDOWS_DAYS:
            print(f"  - Analysiere Zeitfenster: {days} Tage...")

            window_td = pd.Timedelta(days=days)
            window_steps = int(round(window_td / step_td))

            if window_steps < 1:
                print(f"    Fenster {days}d ergibt window_steps<1 (step={step_td}). Überspringe.")
                continue

            if len(df_aggregated) < window_steps:
                print(f"    Zu wenig Daten für {days} Tage.")
                continue

            # Rolling mean über Steps
            min_periods_data = int(window_steps * ROLLING_MIN_COVERAGE)
            rolling_mean = df_aggregated["series_mw"].rolling(
                window=window_steps,
                min_periods=min_periods_data
            ).mean()

            # Regime-Frac im Fenster (Diagnose/optional Filter)
            rolling_regime_frac = mask_regime.rolling(window=window_steps, min_periods=window_steps).mean()

            # -------------------------
            # MIDPOINT-Regime Zuordnung
            # -------------------------
            mid_offset_steps = (window_steps - 1) // 2
            mid_offset = mid_offset_steps * step_td

            mid_index = rolling_mean.index - mid_offset

            # regime am midpoint (per reindex auf den existierenden 3h-index)
            regime_at_mid = regime_aligned.reindex(mid_index).astype("object").values
            ok_mid = pd.Series(regime_at_mid, index=rolling_mean.index).astype("object") == reg_str

            if APPLY_REGIME_COVERAGE_FILTER:
                ok_window = ok_mid & (rolling_regime_frac >= REGIME_WINDOW_COVERAGE)
            else:
                ok_window = ok_mid

            rolling_mean_reg = rolling_mean.where(ok_window)

            try:
                # Hydrologisches Jahr: Juli-Juni
                hydro_year = rolling_mean_reg.index.year - (rolling_mean_reg.index.month < 7).astype(int)

                yearly_max_indices = rolling_mean_reg.groupby(hydro_year).idxmax().dropna()
                yearly_max_values = rolling_mean_reg.loc[yearly_max_indices]

                # Fensterstart korrekt in Steps
                start_ts = yearly_max_indices - (window_steps - 1) * step_td

                # Diagnose: regime_frac am Maximum (wie viel vom Fenster liegt im Regime)
                frac_at_max = rolling_regime_frac.reindex(yearly_max_indices).astype(float)

                # Diagnose: midpoint-time und midpoint-regime am Maximum
                mid_at_max = yearly_max_indices - mid_offset
                mid_reg_at_max = regime_aligned.reindex(mid_at_max).astype("object")

                # Long sammeln (für Overall)
                df_long = pd.DataFrame({
                    "event_type": "REGIME",
                    "window_days": days,
                    "regime": reg_str,
                    "end_time": yearly_max_indices,
                    "start_time": start_ts,
                    "mid_time": mid_at_max,
                    "mid_regime": mid_reg_at_max.values,
                    "intensity_mw_mean": yearly_max_values.values.astype(float),
                    "intensity_gw_mean": (yearly_max_values.values.astype(float) / 1000.0),
                    "regime_frac": frac_at_max.values.astype(float),
                    "between_regimes": np.nan,   # für REGIME-Events leer
                    "top1_regime": np.nan,
                    "top2_regime": np.nan,
                    "top1_frac": np.nan,
                    "top2_frac": np.nan,
                })
                collected_events_long.append(df_long)

                # Anzeige/CSV: nach Intensität sortiert, Top N
                df_extrema = pd.DataFrame({
                    "Startdatum": start_ts.dt.date,
                    "Value in GW": yearly_max_values.values / 1000.0,
                    "RegimeFrac": frac_at_max.values,
                })

                df_sorted = df_extrema.sort_values(by="Value in GW", ascending=False).head(TOP_N_EVENTS)
                df_sorted.columns = [f"{days}T Start", f"GW ({days}d)", f"Frac ({days}d)"]
                df_sorted.reset_index(drop=True, inplace=True)

                all_results_dfs.append(df_sorted)

            except Exception as e:
                print(f"    FEHLER bei Analyse {days} Tage: {e}")
                continue

        if not all_results_dfs:
            print("  -> Keine Ergebnisse für dieses Regime (unter den Bedingungen).")
            continue

        final_table = pd.concat(all_results_dfs, axis=1)
        per_regime_tables[reg_str] = final_table

        print("\n" + "-" * 100)
        cov_txt = f" & Fenster-Frac >= {REGIME_WINDOW_COVERAGE:.2f}" if APPLY_REGIME_COVERAGE_FILTER else ""
        print(f"TOP {TOP_N_EVENTS} EREIGNISSE | Regime: {reg_str} "
              f"(Sortiert nach höchster rolling mean; MIDPOINT-Zuordnung{cov_txt})")
        print("-" * 100)

        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 2000,
            "display.colheader_justify", "center",
        ):
            print(final_table)

        print("-" * 100 + "\n")

        safe_reg = re.sub(r"[^A-Za-z0-9_\-]+", "_", reg_str)
        outpath = os.path.join(SUMMARY_OUTPUT_DIR, f"{OUTPUT_FILENAME_PREFIX}__REGIME_{safe_reg}.csv")
        final_table.to_csv(outpath, index=False)
        print(f"Tabelle (Regime={reg_str}) gespeichert unter:\n{outpath}")

    # -------------------------------------------------------------------------
    # 3b) NEU: Transition/Overlap Suche (zusätzlich, unabhängig von Regime/No-Regime)
    # -------------------------------------------------------------------------
    collected_transition_long: List[pd.DataFrame] = []

    if ENABLE_TRANSITION_SEARCH:
        print("\n" + "=" * 100)
        print(f"TRANSITION/OVERLAP: TOP {TOP_N_EVENTS} (je Zeitfenster) an Regime-Übergängen / Mischfenstern")
        print(f"Kriterium: top1_frac <= {TRANSITION_TOP1_MAX_FRAC:.2f} UND top2_frac >= {TRANSITION_TOP2_MIN_FRAC:.2f}")
        print(f"no_regime einbeziehen: {TRANSITION_INCLUDE_NO_REGIME}")
        print("=" * 100)

        for days in TIME_WINDOWS_DAYS:
            print(f"\n  - Transition-Analyse Zeitfenster: {days} Tage...")

            window_td = pd.Timedelta(days=days)
            window_steps = int(round(window_td / step_td))

            if window_steps < 1:
                print(f"    Fenster {days}d ergibt window_steps<1 (step={step_td}). Überspringe.")
                continue

            if len(df_aggregated) < window_steps:
                print(f"    Zu wenig Daten für {days} Tage.")
                continue

            # Rolling mean über Steps (gleicher Ansatz wie oben)
            min_periods_data = int(window_steps * ROLLING_MIN_COVERAGE)
            rolling_mean = df_aggregated["series_mw"].rolling(
                window=window_steps,
                min_periods=min_periods_data
            ).mean()

            # Midpoint-Zeitpunkt (für Reporting)
            mid_offset_steps = (window_steps - 1) // 2
            mid_offset = mid_offset_steps * step_td
            mid_index = rolling_mean.index - mid_offset
            regime_at_mid = regime_aligned.reindex(mid_index).astype("object").values
            mid_regime_series = pd.Series(regime_at_mid, index=rolling_mean.index).astype("object")

            # Top-1/Top-2 Regime im Fenster bestimmen (end_time = rolling index)
            try:
                top1_lab, top1_frac, top2_lab, top2_frac = _top2_regimes_in_window(
                    regime_aligned=regime_aligned,
                    end_times=rolling_mean.index,          # gleiche Indexbasis
                    window_steps=window_steps,
                    include_no_regime=TRANSITION_INCLUDE_NO_REGIME,
                    no_regime_label=no_regime_label
                )
            except Exception as e:
                print(f"    FEHLER bei top2-Regime-Auswertung ({days}d): {e}")
                continue

            # Transition-Kriterium
            ok_transition = (top1_frac <= TRANSITION_TOP1_MAX_FRAC) & (top2_frac >= TRANSITION_TOP2_MIN_FRAC)

            rolling_mean_trans = rolling_mean.where(ok_transition)

            try:
                hydro_year = rolling_mean_trans.index.year - (rolling_mean_trans.index.month < 7).astype(int)

                yearly_max_indices = rolling_mean_trans.groupby(hydro_year).idxmax().dropna()
                yearly_max_values = rolling_mean_trans.loc[yearly_max_indices]

                start_ts = yearly_max_indices - (window_steps - 1) * step_td

                # top1/top2 am Maximum
                t1 = top1_lab.reindex(yearly_max_indices).astype("object")
                t2 = top2_lab.reindex(yearly_max_indices).astype("object")
                t1f = top1_frac.reindex(yearly_max_indices).astype(float)
                t2f = top2_frac.reindex(yearly_max_indices).astype(float)

                between = (t1.fillna("NA").astype(str) + " <-> " + t2.fillna("NA").astype(str)).astype("object")

                mid_at_max = yearly_max_indices - mid_offset
                mid_reg_at_max = mid_regime_series.reindex(yearly_max_indices).astype("object")

                df_long = pd.DataFrame({
                    "event_type": "TRANSITION",
                    "window_days": days,
                    "regime": "TRANSITION",
                    "end_time": yearly_max_indices,
                    "start_time": start_ts,
                    "mid_time": mid_at_max,
                    "mid_regime": mid_reg_at_max.values,
                    "intensity_mw_mean": yearly_max_values.values.astype(float),
                    "intensity_gw_mean": (yearly_max_values.values.astype(float) / 1000.0),
                    "regime_frac": np.nan,  # nicht definiert (keine single-regime kohärenz)
                    "between_regimes": between.values,
                    "top1_regime": t1.values,
                    "top2_regime": t2.values,
                    "top1_frac": t1f.values,
                    "top2_frac": t2f.values,
                })
                collected_transition_long.append(df_long)

                # Print-block: Top30 über alle hydro-years (wie bei Regime-Tabelle: sortiert nach intensity)
                df_extrema = pd.DataFrame({
                    "Startdatum": start_ts.dt.date,
                    "Value in GW": yearly_max_values.values / 1000.0,
                    "Between": between.values,
                    "Top1Frac": t1f.values,
                    "Top2Frac": t2f.values,
                    "MidRegime": mid_reg_at_max.values,
                }).sort_values("Value in GW", ascending=False).head(TOP_N_EVENTS).reset_index(drop=True)

                print("\n" + "-" * 100)
                print(f"TOP {TOP_N_EVENTS} TRANSITION-EVENTS | {days}d (sortiert nach höchster rolling mean)")
                print("-" * 100)
                with pd.option_context(
                    "display.max_rows", None,
                    "display.max_columns", None,
                    "display.width", 2000,
                    "display.colheader_justify", "center",
                ):
                    print(df_extrema)
                print("-" * 100)

                # CSV speichern (wide-ish, pro Fenster)
                outpath = os.path.join(
                    SUMMARY_OUTPUT_DIR,
                    f"{TRANSITION_OUTPUT_FILENAME_PREFIX}__TRANSITION__WINDOW_{days}d.csv"
                )
                df_extrema.to_csv(outpath, index=False)
                print(f"Transition-Tabelle ({days}d) gespeichert unter:\n{outpath}")

            except Exception as e:
                print(f"    FEHLER bei Transition-Analyse {days} Tage: {e}")
                continue

        if collected_transition_long:
            transitions_long = pd.concat(collected_transition_long, ignore_index=True)
            out_long = os.path.join(
                SUMMARY_OUTPUT_DIR,
                f"{TRANSITION_OUTPUT_FILENAME_PREFIX}__TRANSITION_EVENTS_LONG.csv"
            )
            transitions_long.to_csv(out_long, index=False)
            print(f"\nTransition (long) gespeichert unter:\n{out_long}")
        else:
            print("\nKeine Transition-Events gefunden (unter den Kriterien).")

    # -------------------------------------------------------------------------
    # 4) Overall-Ausgabe:
    #    Overall = globales Ranking der Pro-Regime-Events nach Intensität.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"OVERALL: TOP {TOP_N_EVENTS} global schlimmste Dunkelflauten (je Zeitfenster)")
    print("        = Zusammenfassung der Pro-Regime-Events (kein Neuberechnen, kein zusätzlicher Filter)")
    print("=" * 100)

    if not collected_events_long:
        print("Keine Events aus Pro-Regime-Tabellen gesammelt. Abbruch.")
        raise SystemExit(1)

    events_long = pd.concat(collected_events_long, ignore_index=True)

    out_long = os.path.join(SUMMARY_OUTPUT_DIR, f"{OUTPUT_FILENAME_PREFIX}__OVERALL_FROM_REGIME_EVENTS_LONG.csv")
    events_long.to_csv(out_long, index=False)
    print(f"Overall (long) gespeichert unter:\n{out_long}")

    overall_blocks_wide = []
    overall_blocks_print = []

    for days in TIME_WINDOWS_DAYS:
        dfw = events_long.loc[events_long["window_days"] == days].copy()
        if dfw.empty:
            continue

        dfw = dfw.sort_values("intensity_gw_mean", ascending=False).head(TOP_N_EVENTS).reset_index(drop=True)

        df_print = dfw[["start_time", "end_time", "mid_time", "intensity_gw_mean", "regime", "regime_frac"]].copy()
        df_print.columns = [
            f"{days}T StartTS", f"{days}T EndTS", f"{days}T MidTS",
            f"GW ({days}d)", f"Regime ({days}d)", f"Frac ({days}d)"
        ]
        overall_blocks_print.append(df_print)

        df_wide = pd.DataFrame({
            f"{days}T Start": pd.to_datetime(dfw["start_time"]).dt.date,
            f"GW ({days}d)": dfw["intensity_gw_mean"].values,
            f"Regime ({days}d)": dfw["regime"].astype("object").values,
            f"Frac ({days}d)": dfw["regime_frac"].astype(float).values,
        })
        overall_blocks_wide.append(df_wide)

    if not overall_blocks_wide:
        print("Keine Overall-Ergebnisse generiert.")
        raise SystemExit(1)

    overall_table = pd.concat(overall_blocks_wide, axis=1)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 2000,
        "display.colheader_justify", "center",
    ):
        print(pd.concat(overall_blocks_print, axis=1))

    out_overall = os.path.join(
        SUMMARY_OUTPUT_DIR,
        f"{OUTPUT_FILENAME_PREFIX}__OVERALL_GLOBAL_TOP30_FROM_REGIME_EVENTS.csv"
    )
    overall_table.to_csv(out_overall, index=False)
    print(f"\nOverall (wide) gespeichert unter:\n{out_overall}")

    # -------------------------------------------------------------------------
    # 5) NEU: Optionaler Combined-Overall (Regime + Transition) als long/wide
    # -------------------------------------------------------------------------
    if ENABLE_TRANSITION_SEARCH and collected_transition_long:
        print("\n" + "=" * 100)
        print(f"COMBINED OVERALL: TOP {TOP_N_EVENTS} (Regime + Transition) je Zeitfenster")
        print("        = globales Ranking über beide Event-Typen")
        print("=" * 100)

        transitions_long = pd.concat(collected_transition_long, ignore_index=True)
        combined_long = pd.concat([events_long, transitions_long], ignore_index=True)

        out_comb_long = os.path.join(
            SUMMARY_OUTPUT_DIR,
            f"{OUTPUT_FILENAME_PREFIX}__COMBINED_REGIME_PLUS_TRANSITION_LONG.csv"
        )
        combined_long.to_csv(out_comb_long, index=False)
        print(f"Combined (long) gespeichert unter:\n{out_comb_long}")

        comb_blocks_print = []
        comb_blocks_wide = []

        for days in TIME_WINDOWS_DAYS:
            dfc = combined_long.loc[combined_long["window_days"] == days].copy()
            if dfc.empty:
                continue

            dfc = dfc.sort_values("intensity_gw_mean", ascending=False).head(TOP_N_EVENTS).reset_index(drop=True)

            # Print-Block mit Markierung "between_regimes" für Transition
            df_print = dfc[[
                "start_time", "end_time", "mid_time", "intensity_gw_mean",
                "event_type", "regime", "between_regimes", "top1_frac", "top2_frac"
            ]].copy()
            df_print.columns = [
                f"{days}T StartTS", f"{days}T EndTS", f"{days}T MidTS", f"GW ({days}d)",
                f"Type ({days}d)", f"Regime ({days}d)", f"Between ({days}d)", f"Top1Frac ({days}d)", f"Top2Frac ({days}d)"
            ]
            comb_blocks_print.append(df_print)

            # Wide: etwas kompakter
            df_wide = pd.DataFrame({
                f"{days}T Start": pd.to_datetime(dfc["start_time"]).dt.date,
                f"GW ({days}d)": dfc["intensity_gw_mean"].values,
                f"Type ({days}d)": dfc["event_type"].astype("object").values,
                f"Reg/Between ({days}d)": dfc.apply(
                    lambda r: r["between_regimes"] if r["event_type"] == "TRANSITION" else r["regime"], axis=1
                ).astype("object").values,
            })
            comb_blocks_wide.append(df_wide)

        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 2000,
            "display.colheader_justify", "center",
        ):
            print(pd.concat(comb_blocks_print, axis=1))

        out_comb_wide = os.path.join(
            SUMMARY_OUTPUT_DIR,
            f"{OUTPUT_FILENAME_PREFIX}__COMBINED_REGIME_PLUS_TRANSITION_TOP30_WIDE.csv"
        )
        pd.concat(comb_blocks_wide, axis=1).to_csv(out_comb_wide, index=False)
        print(f"\nCombined (wide) gespeichert unter:\n{out_comb_wide}")

    print("\nFertig.")
