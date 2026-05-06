#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_residual_load_robust_anomalies.py

Analog zum Generation-Anomalie-Skript, aber für Residual-Last (RL) mit Option B:
=> robuste z-Score-Anomalie pro Land/Monat.

Metrik (pro Land, pro Eventfenster):
  mu_event  = mean(RL[t] über Eventfenster)
  baseline  = RL-Werte im gleichen Monat über CLIM_YEARS
  med_m     = median(baseline)
  mad_m     = median(|baseline - med_m|)
  sigma_rob = 1.4826 * mad_m   (robuste "sigma"-Skalierung)
  z         = (mu_event - med_m) / sigma_rob
Fallbacks bei sigma_rob ~ 0: IQR/1.349, dann std, dann 1.0.

Plots (analog):
A) pro Regime: 1-panel Map (RL robust z-anomaly) für "heaviest event per regime" (inkl. no_regime)
B) Übersicht-Grid: heaviest event je Regime (inkl. no_regime) im 5x2 Layout (5 Regimes pro Spalte, 2 Spalten)
D) pro Regime: Regime-interne Top-10 (5x2 Event-Grid; paginiert falls >10)

Daten:
- Events CSVs wie zuvor:
  tables/heaviest_event_per_regime__{EVENT_WINDOW_DAYS}d.csv
  tables/top10_events_per_regime__{EVENT_WINDOW_DAYS}d.csv

- Residual-load CSVs pro Land unter:
  /mnt/endata/MA_Arthur/final_results_portfolio_uncorrected/residual_load_analysis
  (Dateinamen enthalten englische volle Namen)

Hinweise:
- Shapes: NaturalEarth admin_0_countries; ISO-Spalte wird robust erkannt (ADM0_A3/ISO_A3/SOV_A3/...)
- Missing-data wird grau dargestellt (statt „unsichtbar“)
"""

from __future__ import annotations

import os
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import geopandas as gpd


# =============================================================================
# KONFIGURATION
# =============================================================================

# Run-Tag (steuert Output-Ordnerstruktur)
PERSISTENCE_DAYS = 5
K_CLUSTERS = 8
Y0, Y1 = 2025, 2050
RUN_TAG = f"p{PERSISTENCE_DAYS}d_k{K_CLUSTERS}_{Y0}_{Y1}"

# Event-Definition
EVENT_WINDOW_DAYS = 7

# Climatology / Baseline Jahre (Monatsklima)
CLIM_YEARS = list(range(2025, 2051))  # inclusive 2050

# Residual Load CSVs
MA_BASE = "/mnt/endata/MA_Arthur"
RESIDUAL_DIR = os.path.join(
    MA_BASE,
    "final_results_portfolio_uncorrected",
    "residual_load_analysis",
)
FILENAME_PATTERN = "residual_load_{country_name}_weather2025-2050_cap2024_demandyearly_dir.csv"
VALUE_COLUMN = "residual_load_mw"

COUNTRIES_TO_PLOT = [
    "DE", "FR", "PL", "CZ", "AT", "CH", "IT", "ES", "BE", "NL", "DK", "SE",
    "PT", "NO", "FI", "LT", "LV", "EE", "GB"
]

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

# NO_REGIME
NO_REGIME_LABEL = "no_regime"

# Europa-Extent (lon/lat) für reine Plot-Achsen (PlateCarree-nahe Darstellung)
EUROPE_EXTENT = (-12.0, 35.0, 35.0, 72.0)

# Output
OUT_BASE = os.path.join(
    MA_BASE,
    "final_results_portfolio_uncorrected",
    "regime_residual_load_robust_z__experiments_v2",
)
OUT_ROOT = os.path.join(OUT_BASE, RUN_TAG)

MAPS_DIR = os.path.join(OUT_ROOT, "maps_heaviest_per_regime")
OVERVIEW_DIR = os.path.join(OUT_ROOT, "overview_grids")
TOP10_REGIME_DIR = os.path.join(OUT_ROOT, "top10_per_regime_grids")
TABLES_DIR = os.path.join(
    MA_BASE,
    "final_results_portfolio_uncorrected",
    "regime_dunkelflaute_maps__experiments_v2",
    RUN_TAG,
    "tables",
)

for d in [MAPS_DIR, OVERVIEW_DIR, TOP10_REGIME_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[OK] RUN_TAG: {RUN_TAG}")
print(f"[OK] RESIDUAL_DIR: {RESIDUAL_DIR}")
print(f"[OK] TABLES_DIR: {TABLES_DIR}")
print(f"[OK] OUT_ROOT: {OUT_ROOT}")


# =============================================================================
# Helpers
# =============================================================================

def _parse_datetime_index(idx: pd.Index) -> pd.DatetimeIndex:
    dt = pd.to_datetime(idx, errors="coerce")
    dt = pd.DatetimeIndex(dt).tz_localize(None)
    return dt


def _detect_iso_column(gdf: gpd.GeoDataFrame) -> str:
    candidates = [
        "ADM0_A3", "ISO_A3", "SOV_A3", "ADM0A3", "ISO3", "iso_a3", "sov_a3",
        "adm0_a3", "adm0a3", "iso3", "ISO", "ISO_3"
    ]
    cols = {c.lower(): c for c in gdf.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    raise ValueError(f"Keine ISO-Spalte gefunden. Verfügbare Spalten: {list(gdf.columns)}")


def load_country_residual_series() -> pd.DataFrame:
    """
    Lädt RL-Zeitreihen pro Land (MW) und gibt DataFrame (index=time, columns=ISO2/cc).
    """
    dfs = []
    for cc in COUNTRIES_TO_PLOT:
        if cc not in COUNTRY_CODE_TO_FILENAME_MAP:
            print(f" [WARN] Kein Mapping für {cc} – skip")
            continue

        cname = COUNTRY_CODE_TO_FILENAME_MAP[cc]
        fp = os.path.join(RESIDUAL_DIR, FILENAME_PATTERN.format(country_name=cname))
        if not os.path.exists(fp):
            print(f" [WARN] Datei fehlt: {fp} – skip")
            continue

        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if VALUE_COLUMN not in df.columns:
            print(f" [WARN] {cname}: keine Spalte '{VALUE_COLUMN}' – vorhanden: {list(df.columns)} – skip")
            continue

        s = df[VALUE_COLUMN].copy()
        s.index = _parse_datetime_index(s.index)
        s.name = cc
        dfs.append(s.to_frame())
        print(f" - geladen: {cc} ({cname}) shape={df.shape}")

    if not dfs:
        raise RuntimeError("Keine Residual-Load Dateien geladen.")

    out = pd.concat(dfs, axis=1).sort_index()
    if out.notna().sum().sum() == 0:
        raise RuntimeError("Alle Residual-Load Werte sind NaN.")
    return out


def _robust_sigma_from_baseline(x: np.ndarray) -> float:
    """
    Robuste sigma-Schätzung:
      sigma_rob = 1.4826 * MAD
    Fallbacks:
      - IQR/1.349
      - std
      - 1.0
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad
    if np.isfinite(sigma) and sigma > 1e-12:
        return float(sigma)

    q25, q75 = np.percentile(x, [25.0, 75.0])
    iqr = float(q75 - q25)
    sigma = iqr / 1.349 if np.isfinite(iqr) and iqr > 1e-12 else np.nan
    if np.isfinite(sigma) and sigma > 1e-12:
        return float(sigma)

    sigma = float(np.std(x, ddof=0))
    if np.isfinite(sigma) and sigma > 1e-12:
        return float(sigma)

    return 1.0


def compute_event_country_z_anomalies(
    rl_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    clim_years: List[int],
) -> pd.Series:
    """
    Berechnet pro Land robuste z-Anomalie für ein Eventfenster [start,end] (inklusive),
    mit baseline als "gleicher Monat über CLIM_YEARS".
    """
    start = pd.to_datetime(start).tz_localize(None)
    end = pd.to_datetime(end).tz_localize(None)
    if end < start:
        raise ValueError("end < start")

    # Event mean pro Land
    event_slice = rl_df.loc[(rl_df.index >= start) & (rl_df.index <= end)]
    mu_event = event_slice.mean(axis=0, skipna=True)

    # Baseline: gleicher Monat (start.month) über CLIM_YEARS
    m = int(start.month)
    idx = rl_df.index
    baseline_mask = (idx.month == m) & (idx.year.isin(clim_years))
    baseline = rl_df.loc[baseline_mask]

    z = {}
    for cc in rl_df.columns:
        base = baseline[cc].to_numpy(dtype=float)
        base = base[np.isfinite(base)]
        if base.size == 0 or not np.isfinite(mu_event.get(cc, np.nan)):
            z[cc] = np.nan
            continue

        med = float(np.median(base))
        sigma = _robust_sigma_from_baseline(base)
        z[cc] = float((float(mu_event[cc]) - med) / sigma) if sigma > 0 else np.nan

    return pd.Series(z, name="z_robust")


def _sort_key_regime(r: str) -> Tuple[int, int]:
    if r == NO_REGIME_LABEL:
        return (1, 10**9)
    try:
        return (0, int(r))
    except Exception:
        return (0, 10**8)


def _load_naturalearth_countries():
    import cartopy.io.shapereader as shpreader
    shp_path = shpreader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries",
    )
    gdf = gpd.read_file(shp_path)

    # ISO-Spalte robust finden und auf "ISO3" normieren
    iso_col = _detect_iso_column(gdf)
    gdf = gdf.copy()
    gdf["ISO3"] = gdf[iso_col].astype(str)
    gdf.loc[gdf["ISO3"].isin(["-99", "nan", "None", ""]), "ISO3"] = np.nan

    print(f"[SHAPES] ISO-Spalte erkannt: '{iso_col}' -> umbenannt zu 'ISO3'")
    return gdf


def _iso2_to_iso3_map() -> Dict[str, str]:
    # Minimal, zielgerichtet für unsere Länder
    return {
        "DE": "DEU",
        "FR": "FRA",
        "GB": "GBR",
        "ES": "ESP",
        "IT": "ITA",
        "PL": "POL",
        "NL": "NLD",
        "BE": "BEL",
        "DK": "DNK",
        "CZ": "CZE",
        "AT": "AUT",
        "CH": "CHE",
        "EE": "EST",
        "SE": "SWE",
        "PT": "PRT",
        "NO": "NOR",
        "FI": "FIN",
        "LT": "LTU",
        "LV": "LVA",
    }


def plot_event_map_rl_z(
    countries_gdf: gpd.GeoDataFrame,
    z_by_cc: pd.Series,
    out_png: str,
    *,
    title: str,
    norm: TwoSlopeNorm,
    cmap: str = "RdBu_r",
    dpi: int = 300,
) -> None:
    iso2_to_3 = _iso2_to_iso3_map()

    data = []
    for cc, z in z_by_cc.items():
        iso3 = iso2_to_3.get(cc, None)
        if iso3 is None:
            continue
        data.append((iso3, float(z) if np.isfinite(z) else np.nan))

    dfz = pd.DataFrame(data, columns=["ISO3", "z"])
    g = countries_gdf.merge(dfz, on="ISO3", how="left")

    fig = plt.figure(figsize=(9.5, 8.0), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=20)

    # Missing-data grau
    g.plot(
        ax=ax,
        column="z",
        cmap=cmap,
        norm=norm,
        linewidth=0.35,
        edgecolor="black",
        missing_kwds=dict(color="lightgrey", edgecolor="black", hatch=None, label="Missing"),
    )

    ax.set_xlim(EUROPE_EXTENT[0], EUROPE_EXTENT[1])
    ax.set_ylim(EUROPE_EXTENT[2], EUROPE_EXTENT[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.15)

    # Colorbar (Dummy mappable)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05, extend="both")
    cb.set_label("Residual-Load robuste z-Anomalie (Event-Mittel vs Monatsbaseline)", fontsize=20)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Map gespeichert: {out_png}")


def plot_overview_grid_heaviest_5x2(
    event_rows: List[Dict[str, object]],
    rl_df: pd.DataFrame,
    countries_gdf: gpd.GeoDataFrame,
    out_png: str,
    *,
    title: str,
    norm: TwoSlopeNorm,
    cmap: str = "RdBu_r",
    dpi: int = 300,
) -> None:
    """
    5x2 Layout: 5 Regimes pro Spalte, 2 Spalten (insg. max 10 Panels).
    Reihenfolge: regimes sortiert, gefüllt spaltenweise (oben->unten, links dann rechts).
    """
    if not event_rows:
        print("[WARN] Keine event_rows für Overview – skip.")
        return

    # Sort nach regime (stabil)
    event_rows = sorted(event_rows, key=lambda r: _sort_key_regime(str(r["regime"])))

    n = len(event_rows)
    n_panels = min(10, n)
    nrows = 5
    ncols = 2
    panels = event_rows[:n_panels]

    fig = plt.figure(figsize=(28.0, 8.0), dpi=dpi)
    gs = fig.add_gridspec(nrows=2 + 1, ncols=5, height_ratios=[1.0, 1.0, 0.06], hspace=0.15, wspace=0.05)

    for i, row in enumerate(panels):
        # spaltenweise füllen: erst links 0..4, dann rechts 5..9
        col = i % 5
        rix = i // 5

        reg = str(row["regime"])
        st = pd.to_datetime(row["start"])
        en = pd.to_datetime(row["end"])
        val_gw = float(row.get("value_gw", row.get("val_gw", np.nan)))

        z = compute_event_country_z_anomalies(rl_df, st, en, clim_years=CLIM_YEARS)
        iso2_to_3 = _iso2_to_iso3_map()
        dfz = pd.DataFrame(
            [(iso2_to_3.get(cc, None), float(v) if np.isfinite(v) else np.nan) for cc, v in z.items()],
            columns=["ISO3", "z"],
        ).dropna(subset=["ISO3"])
        g = countries_gdf.merge(dfz, on="ISO3", how="left")

        ax = fig.add_subplot(gs[rix, col])
        g.plot(
            ax=ax,
            column="z",
            cmap=cmap,
            norm=norm,
            linewidth=0.25,
            edgecolor="black",
            missing_kwds=dict(color="lightgrey", edgecolor="black"),
        )
        ax.set_xlim(EUROPE_EXTENT[0], EUROPE_EXTENT[1])
        ax.set_ylim(EUROPE_EXTENT[2], EUROPE_EXTENT[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Reg {reg} | {st:%Y-%m-%d}→{en:%Y-%m-%d} | {val_gw:.1f} GW", fontsize=20)
        ax.grid(True, alpha=0.10)

    # Colorbar unten über beide Spalten
    cax = fig.add_subplot(gs[2, :])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("Residual-Load robuste z-Anomalie (Event-Mittel vs Monatsbaseline)", fontsize=20)

    fig.suptitle(title, fontsize=20, y=0.995)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Overview grid gespeichert: {out_png}")


def plot_topn_grid_5x2_paginated(
    events_df: pd.DataFrame,
    rl_df: pd.DataFrame,
    countries_gdf: gpd.GeoDataFrame,
    out_dir: str,
    *,
    title_prefix: str,
    norm: TwoSlopeNorm,
    cmap: str = "RdBu_r",
    per_page: int = 10,
    dpi: int = 300,
) -> None:
    """
    5x2 Event-Grid, paginiert falls >10.
    events_df muss Spalten: start, end, value_gw, rank (optional), regime (optional) haben.
    """
    if events_df.empty:
        print("[WARN] events_df leer – kein Grid.")
        return

    events_df = events_df.copy()
    if "rank" in events_df.columns:
        events_df = events_df.sort_values("rank")
    else:
        if "value_mw" in events_df.columns:
            events_df = events_df.sort_values("value_mw", ascending=False)

    total = len(events_df)
    pages = int(math.ceil(total / per_page))

    iso2_to_3 = _iso2_to_iso3_map()

    for p in range(pages):
        chunk = events_df.iloc[p * per_page:(p + 1) * per_page]

        fig = plt.figure(figsize=(28.0, 9.6), dpi=dpi)
        gs = fig.add_gridspec(
            nrows=3,                  # 2 Plot-Zeilen + 1 Colorbar-Zeile
            ncols=5,
            height_ratios=[1.0, 1.0, 0.07],
            hspace=0.34,
            wspace=0.05,
        )

        for i, rec in enumerate(chunk.to_dict("records")):
            col = i % 5
            rix = i // 5

            st = pd.to_datetime(rec["start"])
            en = pd.to_datetime(rec["end"])
            val_gw = float(rec.get("value_gw", np.nan))
            rk = rec.get("rank", None)

            z = compute_event_country_z_anomalies(rl_df, st, en, clim_years=CLIM_YEARS)
            dfz = pd.DataFrame(
                [(iso2_to_3.get(cc, None), float(v) if np.isfinite(v) else np.nan) for cc, v in z.items()],
                columns=["ISO3", "z"],
            ).dropna(subset=["ISO3"])
            g = countries_gdf.merge(dfz, on="ISO3", how="left")

            ax = fig.add_subplot(gs[rix, col])
            g.plot(
                ax=ax,
                column="z",
                cmap=cmap,
                norm=norm,
                linewidth=0.25,
                edgecolor="black",
                missing_kwds=dict(color="lightgrey", edgecolor="black"),
            )
            ax.set_xlim(EUROPE_EXTENT[0], EUROPE_EXTENT[1])
            ax.set_ylim(EUROPE_EXTENT[2], EUROPE_EXTENT[3])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.10)

            head = f"#{int(rk)} | " if rk is not None and str(rk) != "nan" else ""
            ax.set_title(
                f"{head}{st:%Y-%m-%d}→{en:%Y-%m-%d}\n{val_gw:.1f} GW",
                fontsize=16,
                pad=10,
            )

        cax = fig.add_subplot(gs[2, :])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
        cb.set_label(
            "Residual-Load robuste z-Anomalie (Event-Mittel vs Monatsbaseline)",
            fontsize=18,
        )
        cb.ax.tick_params(labelsize=14)

        page_suffix = f"__page{p+1:02d}-of-{pages:02d}" if pages > 1 else ""
        out_png = os.path.join(out_dir, f"{title_prefix}{page_suffix}.png")

        fig.suptitle(
            f"{title_prefix.replace('_', ' ')}{page_suffix}",
            fontsize=20,
            y=0.975,
        )
        fig.subplots_adjust(top=0.84, bottom=0.07)

        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Grid gespeichert: {out_png}")

def plot_overview_grid_heaviest_3x3(
    event_rows: List[Dict[str, object]],
    rl_df: pd.DataFrame,
    countries_gdf: gpd.GeoDataFrame,
    out_png: str,
    *,
    title: str,
    norm: TwoSlopeNorm,
    cmap: str = "RdBu_r",
    dpi: int = 300,
) -> None:
    """
    3x3 Layout: max 9 Panels.
    Reihenfolge: regimes sortiert, gefüllt zeilenweise (links->rechts, oben->unten).
    """
    if not event_rows:
        print("[WARN] Keine event_rows für Overview – skip.")
        return

    # Sort nach regime (stabil)
    event_rows = sorted(event_rows, key=lambda r: _sort_key_regime(str(r["regime"])))

    n_panels = min(9, len(event_rows))
    panels = event_rows[:n_panels]

    nrows, ncols = 3, 3

    fig = plt.figure(figsize=(18.0, 16.8), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=nrows + 1,            # +1 für Colorbar
        ncols=ncols,
        height_ratios=[1.0, 1.0, 1.0, 0.07],
        hspace=0.34,
        wspace=0.05,
    )

    iso2_to_3 = _iso2_to_iso3_map()

    for i, row in enumerate(panels):
        rix = i // ncols
        col = i % ncols

        reg = str(row["regime"])
        st = pd.to_datetime(row["start"])
        en = pd.to_datetime(row["end"])
        val_gw = float(row.get("value_gw", row.get("val_gw", np.nan)))

        z = compute_event_country_z_anomalies(rl_df, st, en, clim_years=CLIM_YEARS)

        dfz = pd.DataFrame(
            [(iso2_to_3.get(cc, None), float(v) if np.isfinite(v) else np.nan) for cc, v in z.items()],
            columns=["ISO3", "z"],
        ).dropna(subset=["ISO3"])

        g = countries_gdf.merge(dfz, on="ISO3", how="left")

        ax = fig.add_subplot(gs[rix, col])
        g.plot(
            ax=ax,
            column="z",
            cmap=cmap,
            norm=norm,
            linewidth=0.25,
            edgecolor="black",
            missing_kwds=dict(color="lightgrey", edgecolor="black"),
        )
        ax.set_xlim(EUROPE_EXTENT[0], EUROPE_EXTENT[1])
        ax.set_ylim(EUROPE_EXTENT[2], EUROPE_EXTENT[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.10)

        ax.set_title(
            f"Reg {reg} | {st:%Y-%m-%d}→{en:%Y-%m-%d}\n{val_gw:.1f} GW",
            fontsize=16,
            pad=10,
        )

    # Colorbar als letzte Zeile über alle Spalten
    cax = fig.add_subplot(gs[nrows, :])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(
        "Residual-Load Anomalie (Event-Mittel vs Monatsbaseline)",
        fontsize=18,
    )
    cb.ax.tick_params(labelsize=14)

    fig.suptitle(title, fontsize=20, y=0.975)
    fig.subplots_adjust(top=0.84, bottom=0.07)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Overview grid gespeichert: {out_png}")
# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("--- Lade Residual Load Zeitreihen ---")
    rl = load_country_residual_series()
    print(f"[OK] RL index: {rl.index.min()} bis {rl.index.max()} | cols={len(rl.columns)}")

    print("--- Lade NaturalEarth shapes ---")
    countries = _load_naturalearth_countries()

    # Norm global (symmetrisch) für z: robuste Default-Spanne

    Z_VMAX = 1.0
    z_norm = TwoSlopeNorm(vmin=-Z_VMAX, vcenter=0.0, vmax=Z_VMAX)

    # Event CSVs
    heaviest_fp = os.path.join(TABLES_DIR, f"heaviest_event_per_regime__{EVENT_WINDOW_DAYS}d.csv")
    top10_fp = os.path.join(TABLES_DIR, f"top10_events_per_regime__{EVENT_WINDOW_DAYS}d.csv")

    if not os.path.exists(heaviest_fp):
        raise FileNotFoundError(f"Fehlt: {heaviest_fp}")
    if not os.path.exists(top10_fp):
        raise FileNotFoundError(f"Fehlt: {top10_fp}")

    heaviest = pd.read_csv(heaviest_fp)
    top10 = pd.read_csv(top10_fp)

    # =============================================================================
    # A) pro Regime: 1-panel Map (heaviest event per regime)
    # =============================================================================
    print("\n--- A) Karten: Heftigstes Event je Regime (RL Anomalie) ---")
    for rec in heaviest.to_dict("records"):
        reg = str(rec["regime"])
        st = pd.to_datetime(rec["start"])
        en = pd.to_datetime(rec["end"])
        val_gw = float(rec.get("value_gw", np.nan))

        z = compute_event_country_z_anomalies(rl, st, en, clim_years=CLIM_YEARS)

        safe_reg = re.sub(r"[^A-Za-z0-9_\-]+", "_", reg)
        stamp = f"{st:%Y%m%d}-{en:%Y%m%d}__{EVENT_WINDOW_DAYS}d"
        out_png = os.path.join(MAPS_DIR, f"rl_zmap__regime_{safe_reg}__{stamp}.png")

        title = (
            f""
        )
        plot_event_map_rl_z(
            countries_gdf=countries,
            z_by_cc=z,
            out_png=out_png,
            title=title,
            norm=z_norm,
            cmap="RdBu_r",
            dpi=300,
        )

    # =============================================================================
    # B) Übersicht-Grid: heaviest event je Regime (5x2)
    # =============================================================================
    print("\n--- B) Overview-Grid (5x2): Heftigstes Event je Regime ---")

    out_overview=os.path.join(
            OVERVIEW_DIR,
            f"overview_heaviest_per_regime__RL_robust_z__{EVENT_WINDOW_DAYS}d__3x3.png",
    )
    plot_overview_grid_heaviest_3x3(
        event_rows=heaviest.to_dict("records"),
        rl_df=rl,
        countries_gdf=countries,
        out_png=out_overview,
        title=f"",
        norm=z_norm,
        cmap="RdBu_r",
        dpi=300,
    )

    # =============================================================================
    # D) pro Regime: Regime-interne Top-10 (5x2, paginiert falls >10)
    # =============================================================================
    print("\n--- D) Top-10 pro Regime (5x2, paginiert falls >10) ---")
    for reg, g in top10.groupby("regime"):
        gg = g.copy()
        safe_reg = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(reg))
        title_prefix = f"top10__RL_robust_z__regime_{safe_reg}__{EVENT_WINDOW_DAYS}d__5x2"

        # Output pro Regime in eigenen Unterordner
        reg_dir = os.path.join(TOP10_REGIME_DIR, f"regime_{safe_reg}")
        os.makedirs(reg_dir, exist_ok=True)

        plot_topn_grid_5x2_paginated(
            events_df=gg,
            rl_df=rl,
            countries_gdf=countries,
            out_dir=reg_dir,
            title_prefix=title_prefix,
            norm=z_norm,
            cmap="RdBu_r",
            per_page=10,
            dpi=300,
        )

    print("\nFertig.")


if __name__ == "__main__":
    main()