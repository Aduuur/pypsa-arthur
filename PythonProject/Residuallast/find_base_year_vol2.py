#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# KONFIGURATION
# =============================================================================
Y0, Y1 = 2025, 2050
YEARS_TO_ANALYZE = range(Y0, Y1 + 1)

DATA_FREQ_HOURS = 3
COVERAGE = 0.95  # Mindestabdeckung pro Rolling-Fenster (Anteil erwarteter Samples)

# Wettervariablen :
# - rsds: surface downwelling shortwave radiation (W m-2)
# - tas:  2m temperature (K)
# - sfcWind: near-surface wind speed (m s-1)
WEATHER_VARS = ["rsds", "tas", "sfcWind"]

# Mehrskalige Persistenz-Fenster (Tage)
ROLL_WINDOWS_DAYS = [2, 7, 14, 28]

# Climatology-Auflösung (für Anomalien): day-of-year × hour
USE_HOURLY_CLIMATOLOGY = True  # True = (doy,hour), False = nur doy

# Gesamt-Score = gewichtete Summe (nach min-max Skalierung auf 0..1; 0=best).
WEIGHT_RMS = 0.70
WEIGHT_Q95 = 0.30

# Gewichtung über Fenster: längere Fenster leicht höher (Persistenz wichtiger)
WINDOW_WEIGHTS: Dict[int, float] = {2: 0.15, 7: 0.25, 14: 0.30, 28: 0.30}

# Gewichtung über Variablen: zunächst gleich (gut begründbar)
VAR_WEIGHTS: Dict[str, float] = {"rsds": 1.0 / 3.0, "tas": 1.0 / 3.0, "sfcWind": 1.0 / 3.0}

# -------------------------
# Externe Extremereignis-CSVs (Residual Load) -> regime-basiert: Exclude / Penalty
# -------------------------
REGIME_EXTREMA_DIR = (
    "/mnt/endata/MA_Arthur/final_results_portfolio_uncorrected/residual_load_analysis/"
    "regime_extrema_runs/p5d_k8_2025_2050"
)

# WICHTIG: deine Dateien haben "__REGIME_0.csv" (doppelter Unterstrich)
REGIME_FILE_PATTERN = "extrema_summary_regime_based_midpoint_synthetic_demand_hydro_year__REGIME_{REGIME_X}.csv"

# WICHTIG: no_regime liegt als "__REGIME_no_regime.csv" vor, also als Suffix "no_regime"
REGIMES: List[str] = [str(i) for i in range(8)] + ["no_regime"]

# user request: take 7d values
EXTREMA_USE_WINDOW = "7d"
EXTREMA_START_COL = "7T Start"
EXTREMA_VALUE_COL = "GW (7d)"

# --- NEU: Exclude und Penalty sauber entkoppelt (damit Penalty auch funktioniert, wenn Exclude aus ist) ---
USE_EXTREMA_EXCLUDE = False          # Exclude aus
EXCLUDE_TOPN_PER_REGIME = 15         # wenn USE_EXTREMA_EXCLUDE=True, dann N=15 pro Regime

USE_EXTREMA_PENALTY = True           # Penalty an
PENALTY_TOPN_PER_REGIME = 15         # N=15 pro Regime für Penalty-Berechnung

EXTREMA_PENALTY_WEIGHT = 0.50        # additiv zur Wetter-Score-Summe (typisch 0.2..1.0)

# Optional: manuelle Ausschlüsse bleiben möglich
EXCLUDED_YEARS_MANUAL: List[int] = [
   2026,2030,2041
]

# Optional: zusätzliche automatische Ausschlüsse nach Wetter-Score (unverändert)
AUTO_EXCLUDE_MODE = True  # None | "top_score_quantile"
AUTO_EXCLUDE_Q = 0.95     # z.B. 0.90 => schlechteste 10% automatisch ausschließen

# -------------------------
# Datenpfad Wetter
# -------------------------
CORDEX_BASE_DIR = "/mnt/endata/Cordex/Copernicus_CORDEX_data"
CORD_DIR_PATTERN = "cordex_{year}"  # Ordner je Jahr

OUTPUT_DIR = "/mnt/endata/MA_Arthur/average_weather_year/weather_anomaly_score"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Hilfsfunktionen: robustes Laden & Aggregation
# =============================================================================
def _find_year_files(year_dir: str) -> List[str]:
    """Findet NetCDF/Zarr Dateien in cordex_YEAR Ordnern."""
    p = Path(year_dir)
    if not p.exists():
        raise FileNotFoundError(f"Jahresordner nicht gefunden: {year_dir}")

    nc = sorted([str(x) for x in p.rglob("*.nc")])
    zarr = sorted([str(x) for x in p.rglob("*.zarr")])

    files = nc + zarr
    if not files:
        raise FileNotFoundError(f"Keine .nc/.zarr Dateien gefunden in: {year_dir}")
    return files


def _open_dataset_for_vars(files: List[str], vars_needed: List[str]) -> xr.Dataset:
    """
    Öffnet Dataset robust und behält nur benötigte Variablen.
    Falls mehrere Dateien vorliegen: open_mfdataset by_coords.
    """
    if all(f.endswith(".zarr") for f in files) and len(files) == 1:
        ds = xr.open_zarr(files[0])
    else:
        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            parallel=False,
            coords="minimal",
            data_vars="minimal",
            compat="override",
        )

    present = set(ds.data_vars)
    keep = [v for v in vars_needed if v in present]
    if not keep:
        raise KeyError(
            f"Keine der benötigten Variablen {vars_needed} in Dataset gefunden. "
            f"Vorhanden: {sorted(list(present))[:50]}..."
        )
    return ds[keep]


def _detect_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    """Detektiert lat/lon Koordinatennamen."""
    candidates_lat = ["lat", "latitude", "y"]
    candidates_lon = ["lon", "longitude", "x"]

    lat_name = None
    lon_name = None
    for c in candidates_lat:
        if c in ds.coords:
            lat_name = c
            break
    for c in candidates_lon:
        if c in ds.coords:
            lon_name = c
            break

    if lat_name is None:
        for c in candidates_lat:
            if c in ds:
                lat_name = c
                break
    if lon_name is None:
        for c in candidates_lon:
            if c in ds:
                lon_name = c
                break

    if lat_name is None or lon_name is None:
        for c in ds.coords:
            if "lat" in c.lower():
                lat_name = c
            if "lon" in c.lower():
                lon_name = c
        if lat_name is None or lon_name is None:
            raise KeyError(f"Konnte lat/lon Koordinaten nicht detektieren. Coords: {list(ds.coords)}")

    return lat_name, lon_name


def _spatial_mean_series(da: xr.DataArray) -> pd.Series:
    """
    Flächengewichtetes räumliches Mittel:
      - Gewichte ~ cos(lat)
      - Funktioniert für 1D lat oder 2D lat
    """
    if "time" not in da.dims:
        raise ValueError("DataArray hat keine 'time'-Dimension.")

    ds = da.to_dataset(name="v")
    lat_name, _ = _detect_lat_lon_names(ds)

    lat = ds[lat_name]
    w = np.cos(np.deg2rad(lat))

    other_dims = [d for d in da.dims if d != "time"]
    if not other_dims:
        s = da.to_series()
        s.index = pd.DatetimeIndex(s.index)
        return s

    m = da.weighted(w).mean(dim=other_dims, skipna=True)
    s = m.to_series()
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def _ensure_datetime_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


def _regularize_to_3hourly(s: pd.Series) -> pd.Series:
    """
    Erzwingt ein sauberes 3h-Raster:
      - sort
      - gruppiert doppelte Timestamps via mean (ignoriert NaNs)
      - resample auf 3h via mean
    """
    s = _ensure_datetime_index(s)
    if s.empty:
        return s
    s = s.groupby(level=0).mean()
    s = s.resample(f"{DATA_FREQ_HOURS}h").mean()
    return s


def _quick_diag(s: pd.Series, label: str) -> None:
    """Kurzdiagnose: NaN-Anteil, dt, Duplicates."""
    s = _ensure_datetime_index(s)
    dt = s.index.to_series().diff().dropna()
    print(f"  [{label}] n={len(s):,}  NaN={s.isna().mean():.1%}")
    if not dt.empty:
        print(f"    median dt = {dt.median()}")
        print("    dt counts (top3):")
        print(dt.value_counts().head(3).to_string())
    print(f"    duplicated timestamps: {int(s.index.duplicated().sum())}")


# =============================================================================
# Climatology + Anomalien
# =============================================================================
def _build_climatology(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Liefert mu, sigma als Mapping über (doy,hour) oder doy.
    """
    s = _ensure_datetime_index(s).dropna()
    if s.empty:
        raise ValueError("Zeitreihe leer nach dropna.")

    idx = s.index
    doy = idx.dayofyear
    if USE_HOURLY_CLIMATOLOGY:
        hour = idx.hour
        key = pd.MultiIndex.from_arrays([doy, hour], names=["doy", "hour"])
    else:
        key = pd.Index(doy, name="doy")

    tmp = pd.DataFrame({"x": s.values}, index=key)
    mu = tmp.groupby(level=list(range(tmp.index.nlevels)))["x"].mean()
    sigma = tmp.groupby(level=list(range(tmp.index.nlevels)))["x"].std(ddof=0).replace(0.0, np.nan)
    return mu, sigma


def _standardized_anomaly(s: pd.Series, mu: pd.Series, sigma: pd.Series) -> pd.Series:
    s = _ensure_datetime_index(s).astype(float)

    idx = s.index
    doy = idx.dayofyear
    if USE_HOURLY_CLIMATOLOGY:
        hour = idx.hour
        key = pd.MultiIndex.from_arrays([doy, hour], names=["doy", "hour"])
    else:
        key = pd.Index(doy, name="doy")

    mu_t = mu.reindex(key).values
    sig_t = sigma.reindex(key).values

    # robust: wenn sigma fehlt/0 -> 1.0; wenn mu fehlt -> NaN (kein stilles "0")
    mu_ok = np.isfinite(mu_t)
    sig_ok = np.isfinite(sig_t) & (sig_t > 0)

    sig_t = np.where(sig_ok, sig_t, 1.0)
    z = (s.values - mu_t) / sig_t
    z = np.where(mu_ok, z, np.nan)

    return pd.Series(z, index=s.index, name=f"{s.name}_anom")


def _rolling_mean_timebased_3hourly(z: pd.Series, window_days: int, coverage: float = COVERAGE) -> pd.Series:
    """
    Rolling mean für 3h Daten:
      - Zeitfenster: f"{window_days}D"
      - min_periods: coverage * expected_samples
    """
    z = _ensure_datetime_index(z)
    expected = int(round((window_days * 24) / DATA_FREQ_HOURS))  # z.B. 7d -> 56
    min_periods = max(1, int(np.floor(coverage * expected)))
    return z.rolling(window=f"{int(window_days)}D", min_periods=min_periods).mean()


# =============================================================================
# Scoring
# =============================================================================
def _minmax01(s: pd.Series) -> pd.Series:
    mn = float(s.min(skipna=True))
    mx = float(s.max(skipna=True))
    if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
        return (s - mn) / (mx - mn)
    return pd.Series(0.0, index=s.index)


def _load_regime_extrema_7d(
    base_dir: str,
    regimes: List[str],
    pattern: str,
) -> pd.DataFrame:
    """
    Lädt alle regime-basierten Extrema-Tabellen und gibt DataFrame mit Spalten:
      - regime (str)
      - start (datetime)
      - value (float)
      - year (int)
    Es werden die 7d-Spalten genutzt: '7T Start' und 'GW (7d)'.
    """
    rows = []
    for reg in regimes:
        fp = os.path.join(base_dir, pattern.format(REGIME_X=reg))
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Regime-Extrema CSV nicht gefunden: {fp}")

        df = pd.read_csv(fp)
        if EXTREMA_START_COL not in df.columns or EXTREMA_VALUE_COL not in df.columns:
            raise KeyError(
                f"Regime-Extrema CSV {fp} hat nicht die erwarteten Spalten "
                f"'{EXTREMA_START_COL}' / '{EXTREMA_VALUE_COL}'. Gefunden: {list(df.columns)}"
            )

        start = pd.to_datetime(df[EXTREMA_START_COL], errors="coerce")
        val = pd.to_numeric(df[EXTREMA_VALUE_COL], errors="coerce")
        d = pd.DataFrame({"regime": reg, "start": start, "value": val}).dropna()
        d["year"] = d["start"].dt.year.astype(int)

        # filter to analysis horizon
        d = d[(d["year"] >= Y0) & (d["year"] <= Y1)]
        rows.append(d)

    out = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(
        columns=["regime", "start", "value", "year"]
    )
    return out.sort_values(["regime", "value"], ascending=[True, False]).reset_index(drop=True)


def _regime_topn_years(extrema_all: pd.DataFrame, topn: int) -> Tuple[pd.DataFrame, List[int]]:
    """
    Nimmt pro regime die Top-N Events nach 'value' und liefert:
      - top_events (DataFrame) nur diese Events
      - excluded_years: Union der Jahre dieser Top-N pro regime
    """
    if extrema_all.empty:
        return extrema_all.copy(), []

    top_frames = []
    years = set()
    for reg, g in extrema_all.groupby("regime"):
        gg = g.sort_values("value", ascending=False).head(topn)
        top_frames.append(gg)
        years.update(gg["year"].astype(int).tolist())

    top_events = pd.concat(top_frames, axis=0, ignore_index=True) if top_frames else extrema_all.iloc[0:0].copy()
    excluded_years = sorted(years)
    return top_events, excluded_years


def _extrema_year_penalty_from_top_events(top_events: pd.DataFrame) -> pd.Series:
    """
    Erzeugt pro Jahr eine Penalty in [0,1] basierend auf regime-basierten Top-N 7d Residual Load Events.
    Robust: pro Jahr das Maximum über alle Regime-TopEvents.
    Jahre ohne TopEvent -> 0.
    """
    idx_years = pd.Index(list(YEARS_TO_ANALYZE), name="year")
    if top_events.empty:
        return pd.Series(0.0, index=idx_years, name="extrema_penalty_7d_sc")

    by = top_events.groupby("year")["value"].max()
    by = by.reindex(idx_years).fillna(0.0)

    mn = float(by.min())
    mx = float(by.max())
    if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
        pen = (by - mn) / (mx - mn)
    else:
        pen = by * 0.0
    pen.name = "extrema_penalty_7d_sc"
    return pen


def compute_year_scores(weather_ts: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    weather_ts: dict var -> Series(time -> value) (aggregiert, 3h)
    Output: pro Jahr Scores (RMS/Q95 je var+window) + total_score

    Änderung A: Rolling wird innerhalb des Jahres gerechnet (kein Leak über Jahresgrenzen).
    """
    anomalies: Dict[str, pd.Series] = {}
    for v, s in weather_ts.items():
        mu, sig = _build_climatology(s)
        anomalies[v] = _standardized_anomaly(s, mu, sig)

    rows = []
    for year in YEARS_TO_ANALYZE:
        row = {"year": int(year)}

        for v in WEATHER_VARS:
            z = anomalies[v]
            zy = z[z.index.year == year].dropna()

            if zy.empty:
                for w in ROLL_WINDOWS_DAYS:
                    row[f"{v}_rms_{w}d"] = np.nan
                    row[f"{v}_q95_{w}d"] = np.nan
                continue

            for w in ROLL_WINDOWS_DAYS:
                # A) Rolling *innerhalb* des Jahres
                rm_y = _rolling_mean_timebased_3hourly(zy, w, coverage=COVERAGE).dropna()
                if rm_y.empty:
                    row[f"{v}_rms_{w}d"] = np.nan
                    row[f"{v}_q95_{w}d"] = np.nan
                else:
                    row[f"{v}_rms_{w}d"] = float(np.sqrt(np.mean(np.square(rm_y.values))))
                    row[f"{v}_q95_{w}d"] = float(np.quantile(np.abs(rm_y.values), 0.95))

        rows.append(row)

    df = pd.DataFrame(rows).set_index("year").sort_index()

    # harter Check: wenn zu viele NaNs, nicht stillschweigend weitermachen
    raw_cols = [c for c in df.columns if ("_rms_" in c or "_q95_" in c)]
    max_nan_share = float(df[raw_cols].isna().mean().max())
    if max_nan_share > 0.05:
        raise RuntimeError(
            f"Zu viele NaNs in Jahresfeatures (max NaN share={max_nan_share:.1%}). "
            "Ursache meist: Zeitindex/Dateiüberlappung oder Resampling. "
            "Prüfe Diagnoseausgaben oben."
        )

    df_sc = pd.DataFrame(index=df.index)
    for v in WEATHER_VARS:
        for w in ROLL_WINDOWS_DAYS:
            rms_col = f"{v}_rms_{w}d"
            q95_col = f"{v}_q95_{w}d"
            df_sc[f"{rms_col}_sc"] = _minmax01(df[rms_col])
            df_sc[f"{q95_col}_sc"] = _minmax01(df[q95_col])

    total = pd.Series(0.0, index=df.index, name="total_score_weather")
    for v in WEATHER_VARS:
        alpha = float(VAR_WEIGHTS.get(v, 0.0))
        for w in ROLL_WINDOWS_DAYS:
            beta = float(WINDOW_WEIGHTS.get(w, 0.0))
            rms_sc = df_sc[f"{v}_rms_{w}d_sc"]
            q95_sc = df_sc[f"{v}_q95_{w}d_sc"]
            total = total + alpha * beta * (WEIGHT_RMS * rms_sc + WEIGHT_Q95 * q95_sc)

    out = df.join(df_sc)
    out["total_score_weather"] = total
    out["total_score"] = out["total_score_weather"]
    return out


# =============================================================================
# Plots
# =============================================================================
def plot_score_bar(df_scores: pd.DataFrame, excluded_years: List[int], best_year: int) -> str:
    df = df_scores.sort_index()
    years = df.index.astype(int).tolist()
    vals = df["total_score"].values

    excluded_set = set(map(int, excluded_years))

    colors = []
    for y in years:
        if y == int(best_year):
            colors.append("red")
        elif y in excluded_set:
            colors.append("lightgray")
        else:
            colors.append("steelblue")

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(years, vals, color=colors, width=0.7, alpha=0.9)
    # ax.set_title("Standard-year score per year (lower=more 'boring/typical') — Weather + Regime-Extrema penalty/exclusion")
    ax.set_xlabel("Year")
    ax.set_ylabel("total_score")
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red", lw=6, label=f"Bestes Jahr: {best_year}"),
        Line2D([0], [0], color="steelblue", lw=6, label="Kandidaten"),
        Line2D([0], [0], color="lightgray", lw=6, label="Ausgeschlossen"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=12, frameon=True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"weather_score_bar_{Y0}_{Y1}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_scaled_feature_violin(df_scores: pd.DataFrame) -> str:
    sc_cols = [c for c in df_scores.columns if c.endswith("_sc")]
    df_plot = df_scores[sc_cols].melt(var_name="feature", value_name="value")

    plt.figure(figsize=(14, 6))
    sns.violinplot(data=df_plot, x="feature", y="value", inner="box")
    plt.title("Scaled feature distributions (0=best, 1=worst)")
    plt.ylim(-0.05, 1.05)
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"weather_features_violin_scaled_{Y0}_{Y1}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ANALYSE: Standardjahr via Wetter-Anomalien (aggregiert, 3-hourly)")
    print("         + (A) rolling innerhalb des Jahres")
    print("         + (B) Regime-Extrema 7d: Top-N pro Regime (0..7 + no_regime) -> Exclude und Penalty entkoppelt")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # [0/4] Regime-Extrema laden (für Exclude + Penalty)
    # -------------------------------------------------------------------------
    print("\n[0/4] Lade regime-basierte Residual-Load-Extrema CSVs (7d) ...")
    extrema_all = _load_regime_extrema_7d(REGIME_EXTREMA_DIR, REGIMES, REGIME_FILE_PATTERN)
    print(f"  Extrema rows total (all regimes, 7d cols): {len(extrema_all):,}")

    if not extrema_all.empty:
        print("  Top 3 events per regime (7d):")
        for reg in REGIMES:
            gg = extrema_all[extrema_all["regime"] == reg].sort_values("value", ascending=False).head(3)
            if gg.empty:
                print(f"    {reg}: <empty>")
            else:
                print(f"    {reg}:")
                print(gg[["start", "value", "year"]].to_string(index=False))

    # --- NEU: Top-Events für Penalty unabhängig von Exclude erzeugen ---
    extrema_excluded_years: List[int] = []
    top_events_penalty = pd.DataFrame(columns=["regime", "start", "value", "year"])

    if USE_EXTREMA_PENALTY and not extrema_all.empty:
        n_pen = int(PENALTY_TOPN_PER_REGIME)
        top_events_penalty, _ = _regime_topn_years(extrema_all, n_pen)
        print(f"\n  Penalty selection: N={n_pen} per regime -> selected events={len(top_events_penalty):,}")

    if USE_EXTREMA_EXCLUDE and not extrema_all.empty:
        n_exc = int(EXCLUDE_TOPN_PER_REGIME)
        _, extrema_excluded_years = _regime_topn_years(extrema_all, n_exc)
        print(f"  Exclude selection: N={n_exc} per regime -> excluded years (union)={extrema_excluded_years}")

    extrema_pen = _extrema_year_penalty_from_top_events(top_events_penalty) if USE_EXTREMA_PENALTY else None
    if extrema_pen is not None:
        print("\n  Extrema penalty (scaled 0..1) summary:")
        print(extrema_pen.describe().to_string())

    # -------------------------------------------------------------------------
    # [1/4] Laden & aggregieren Wetter
    # -------------------------------------------------------------------------
    print("\n[1/4] Lade CORDEX Wetterdaten pro Jahr und bilde aggregierte 3h-Zeitreihen...")
    series_by_var: Dict[str, List[pd.Series]] = {v: [] for v in WEATHER_VARS}

    for y in YEARS_TO_ANALYZE:
        year_dir = os.path.join(CORDEX_BASE_DIR, CORD_DIR_PATTERN.format(year=y))
        files = _find_year_files(year_dir)

        try:
            ds = _open_dataset_for_vars(files, WEATHER_VARS)
        except Exception as e:
            raise RuntimeError(f"Fehler beim Öffnen von {year_dir}: {e}")

        if "time" not in ds.coords and "time" not in ds.dims:
            raise KeyError(f"'time' nicht gefunden in Dataset aus {year_dir}. Coords: {list(ds.coords)}")

        for v in WEATHER_VARS:
            if v not in ds.data_vars:
                continue
            s = _spatial_mean_series(ds[v])
            s = _regularize_to_3hourly(s)
            s.name = v
            series_by_var[v].append(s)

        print(f"  - {y}: geladen (vars={list(ds.data_vars)})")

    weather_ts: Dict[str, pd.Series] = {}
    print("\n[DEBUG] Diagnostik nach Aggregation (vor Merge über Jahre):")
    for v in WEATHER_VARS:
        if not series_by_var[v]:
            raise RuntimeError(f"Für Variable {v} wurde keine Zeitreihe geladen. Prüfe Daten/Variablennamen.")

        s_all = pd.concat(series_by_var[v], axis=0).sort_index()
        s_all = s_all.groupby(level=0).mean()
        s_all = s_all.resample(f"{DATA_FREQ_HOURS}h").mean()
        s_all = s_all[(s_all.index.year >= Y0) & (s_all.index.year <= Y1)]
        weather_ts[v] = s_all

        _quick_diag(s_all, f"{v}-ALL")
        if s_all.isna().mean() > 0.01:
            print(f"  WARNUNG: {v} hat {s_all.isna().mean():.1%} NaNs nach finalem Merge/Resample.")

    print("\nAggregierte Zeitreihen geladen:")
    for v, s in weather_ts.items():
        print(f"  {v:8s}: {s.index.min()} -> {s.index.max()} | n={len(s):,}")

    # -------------------------------------------------------------------------
    # [2/4] Wetter-Score berechnen (A: rolling within-year)
    # -------------------------------------------------------------------------
    print("\n[2/4] Berechne Jahres-Scores aus standardisierten Anomalien + rolling means (within-year) ...")
    df_scores = compute_year_scores(weather_ts)

    # -------------------------------------------------------------------------
    # [2b] Regime-Extrema-Penalty addieren (B)
    # -------------------------------------------------------------------------
    if extrema_pen is not None:
        df_scores = df_scores.join(extrema_pen, how="left")
        df_scores["extrema_penalty_7d_sc"] = df_scores["extrema_penalty_7d_sc"].fillna(0.0)

        df_scores["total_score"] = (
            df_scores["total_score_weather"]
            + EXTREMA_PENALTY_WEIGHT * df_scores["extrema_penalty_7d_sc"]
        )
        print(f"  Extrema penalty aktiviert: total_score = weather + {EXTREMA_PENALTY_WEIGHT} * extrema_penalty_7d_sc")
    else:
        print("  Extrema penalty deaktiviert.")

    # -------------------------------------------------------------------------
    # [3/4] Ausschlüsse anwenden (manuell + optional regime-extrema + optional weather quantile)
    # -------------------------------------------------------------------------
    print("\n[3/4] Wende Ausschlusskriterien an ...")

    excluded_years = sorted(set(int(y) for y in EXCLUDED_YEARS_MANUAL))
    excluded_years = sorted(set(excluded_years).union(extrema_excluded_years))

    if AUTO_EXCLUDE_MODE == "top_score_quantile":
        q = float(df_scores["total_score"].quantile(AUTO_EXCLUDE_Q))
        auto = df_scores[df_scores["total_score"] >= q].index.astype(int).tolist()
        excluded_years = sorted(set(excluded_years).union(auto))
        print(f"  Auto-Exclude aktiv: Mode={AUTO_EXCLUDE_MODE}, Q={AUTO_EXCLUDE_Q:.2f} -> +{len(auto)} Jahre")

    candidates = df_scores[~df_scores.index.astype(int).isin(excluded_years)].copy()
    if candidates.empty:
        raise RuntimeError("Nach Ausschlüssen sind keine Kandidaten übrig. Prüfe Exclude-Settings.")

    best_year = int(candidates.sort_values("total_score").index[0])

    print("\nAusschluss-Setup:")
    print(f"  EXCLUDED_YEARS_MANUAL        = {sorted(set(EXCLUDED_YEARS_MANUAL))}")
    print(f"  REGIME_EXTREMA_DIR           = {REGIME_EXTREMA_DIR}")
    print(f"  REGIMES                      = {REGIMES}")
    print(f"  USE_EXTREMA_PENALTY          = {USE_EXTREMA_PENALTY} (TopN={PENALTY_TOPN_PER_REGIME}, weight={EXTREMA_PENALTY_WEIGHT})")
    print(f"  USE_EXTREMA_EXCLUDE          = {USE_EXTREMA_EXCLUDE} (TopN={EXCLUDE_TOPN_PER_REGIME})")
    print(f"  Extrema excluded years       = {extrema_excluded_years}")
    print(f"  AUTO_EXCLUDE_MODE (weather)  = {AUTO_EXCLUDE_MODE}")
    print(f"  AUTO_EXCLUDE_Q               = {AUTO_EXCLUDE_Q}")
    print(f"  ==> Gesamt excluded years    = {excluded_years}")

    print(f"\nBestes Jahr (min total_score, nach Ausschlüssen): {best_year}")

    print(f"\nTop 10 Kandidaten (total_score) {Y0}-{Y1}:")
    cols_show = ["total_score", "total_score_weather"]
    if "extrema_penalty_7d_sc" in df_scores.columns:
        cols_show.append("extrema_penalty_7d_sc")
    cols_show += [c for c in df_scores.columns if c.endswith("_sc")][:6]
    print(candidates.sort_values("total_score")[cols_show].head(10).to_string(float_format="{:0.6f}".format))

    # -------------------------------------------------------------------------
    # [4/4] Outputs
    # -------------------------------------------------------------------------
    print("\n[4/4] Speichere Outputs (CSV + Plots) ...")

    csv_path = os.path.join(OUTPUT_DIR, f"weather_anomaly_scores_{Y0}_{Y1}.csv")
    df_scores.to_csv(csv_path)
    print("  CSV:", csv_path)

    p_bar = plot_score_bar(df_scores, excluded_years, best_year)
    print("  Bar:", p_bar)

    p_violin = plot_scaled_feature_violin(df_scores)
    print("  Violin:", p_violin)

    print("\n" + "=" * 80)
    print("ANALYSE ABGESCHLOSSEN")
    print("=" * 80)