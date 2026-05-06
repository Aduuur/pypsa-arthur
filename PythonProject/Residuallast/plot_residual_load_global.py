#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

MA_BASE = "/mnt/endata/MA_Arthur"
BASE_RESIDUAL_DIR = os.path.join(MA_BASE, "final_results_portfolio_uncorrected", "residual_load_analysis")

ANALYSIS_TYPE = "SYNTHETIC_DEMAND"
if ANALYSIS_TYPE == "SYNTHETIC_DEMAND":
    EXTREMA_CSV = os.path.join(BASE_RESIDUAL_DIR, "extrema_summary_hydro_year_plain.csv")
elif ANALYSIS_TYPE == "DEMAND_2024":
    EXTREMA_CSV = os.path.join(BASE_RESIDUAL_DIR, "extrema_summary_based_on_2024_demand_hydro_year.csv")
else:
    raise ValueError("Ungültiger ANALYSIS_TYPE.")

EVENT_WINDOW_DAYS = 7
N_EVENTS = 10

# CSV enthält meist nur Datum (ohne Uhrzeit). Standard: 00:00.
START_HOUR = 0

# Residual Load CSVs (pro Land)
RESIDUAL_DIR = BASE_RESIDUAL_DIR
FILENAME_PATTERN = "residual_load_{country_name}_weather2025-2050_cap2024_demandyearly_dir.csv"
VALUE_COLUMN = "residual_load_mw"

COUNTRIES_TO_PLOT = [
    "DE", "FR", "PL", "CZ", "AT", "CH", "IT", "ES", "BE", "NL",
    "DK", "SE", "PT", "NO", "FI", "LT", "LV", "EE", "GB"
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

# ISO2 -> ISO3 (NaturalEarth kompatibel)
ISO2_TO_ISO3: Dict[str, str] = {
    "DE": "DEU",
    "FR": "FRA",
    "PL": "POL",
    "CZ": "CZE",
    "AT": "AUT",
    "CH": "CHE",
    "IT": "ITA",
    "ES": "ESP",
    "BE": "BEL",
    "NL": "NLD",
    "DK": "DNK",
    "SE": "SWE",
    "PT": "PRT",
    "NO": "NOR",
    "FI": "FIN",
    "LT": "LTU",
    "LV": "LVA",
    "EE": "EST",
    "GB": "GBR",
}

# Map extent
EUROPE_EXTENT = (-12.0, 35.0, 35.0, 72.0)

# Climatology mode + Jahre (Monatsbaseline)
CLIM_YEARS = list(range(2025, 2051))  # inclusive 2050

# z-Norm global (symmetrisch)
Z_VMAX = 2.0

# Output
OUT_ROOT = os.path.join(MA_BASE, "final_results_portfolio_uncorrected", "residual_load_z_maps__global_extrema")
OUT_RUN = os.path.join(OUT_ROOT, f"window{EVENT_WINDOW_DAYS}d__top{N_EVENTS}")
MAPS_DIR = os.path.join(OUT_RUN, "maps_single")
OVERVIEW_DIR = os.path.join(OUT_RUN, "overview")
TABLES_DIR = os.path.join(OUT_RUN, "tables")

for d in [MAPS_DIR, OVERVIEW_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[OK] EXTREMA_CSV: {EXTREMA_CSV}")
print(f"[OK] OUT_RUN:    {OUT_RUN}")
print(f"[OK] RESIDUAL_DIR:{RESIDUAL_DIR}")


# =============================================================================
# Helpers: Extremwert-Tabelle -> Eventliste
# =============================================================================

def _find_window_columns(df: pd.DataFrame, days: int) -> Tuple[str, str]:
    start_col = f"{days}T Start"
    val_col = f"GW ({days}d)"
    if start_col not in df.columns or val_col not in df.columns:
        raise KeyError(
            f"Fenster {days}d nicht in CSV gefunden. Erwartet Spalten: '{start_col}', '{val_col}'. "
            f"Vorhanden: {list(df.columns)}"
        )
    return start_col, val_col


def load_top_events_from_extrema_csv(csv_path: str, window_days: int, n_events: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    start_col, val_col = _find_window_columns(df, window_days)
    starts = pd.to_datetime(df[start_col], errors="coerce")
    vals = pd.to_numeric(df[val_col], errors="coerce")

    out = pd.DataFrame(
        {
            "rank": np.arange(1, len(df) + 1),
            "start": starts,
            "value_gw": vals,
        }
    ).dropna(subset=["start", "value_gw"])

    out["start"] = out["start"].dt.tz_localize(None)
    out["start"] = out["start"].dt.floor("D") + pd.Timedelta(hours=START_HOUR)
    out["end"] = out["start"] + pd.Timedelta(days=window_days)

    out = out.sort_values("value_gw", ascending=False).head(n_events).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    out_csv = os.path.join(TABLES_DIR, f"top_events__{window_days}d__top{len(out)}.csv")
    out.to_csv(out_csv, index=False)
    print(f"[OK] Event-Liste gespeichert: {out_csv}")

    return out


# =============================================================================
# Helpers: RL loading + robust z
# =============================================================================

def _parse_datetime_index(idx: pd.Index) -> pd.DatetimeIndex:
    dt = pd.to_datetime(idx, errors="coerce")
    dt = pd.DatetimeIndex(dt).tz_localize(None)
    return dt


def load_country_residual_series() -> pd.DataFrame:
    """
    Lädt RL-Zeitreihen pro Land (MW) und gibt DataFrame (index=time, columns=ISO2).
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
    Pro Land: robuste z-Anomalie für Eventfenster [start,end] (inklusive),
    baseline = gleicher Monat (start.month) über CLIM_YEARS.
    """
    start = pd.to_datetime(start).tz_localize(None)
    end = pd.to_datetime(end).tz_localize(None)
    if end < start:
        raise ValueError("end < start")

    event_slice = rl_df.loc[(rl_df.index >= start) & (rl_df.index <= end)]
    mu_event = event_slice.mean(axis=0, skipna=True)

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


# =============================================================================
# Shapes + ISO helpers
# =============================================================================

def _detect_iso_column(gdf: gpd.GeoDataFrame) -> str:
    candidates = [
        "ADM0_A3", "ISO_A3", "SOV_A3", "ADM0A3", "ISO3", "iso_a3", "sov_a3",
        "adm0_a3", "adm0a3", "ISO", "ISO_3"
    ]
    cols = {c.lower(): c for c in gdf.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    raise ValueError(f"Keine ISO-Spalte gefunden. Verfügbare Spalten: {list(gdf.columns)}")


def load_naturalearth_countries() -> gpd.GeoDataFrame:
    """
    Lädt NaturalEarth admin_0_countries (110m) via cartopy shapereader.
    Funktioniert auch mit GeoPandas >= 1.0 (ohne gpd.datasets).
    """
    import cartopy.io.shapereader as shpreader

    shp_path = shpreader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries",
    )
    gdf = gpd.read_file(shp_path)

    iso_col = _detect_iso_column(gdf)
    gdf = gdf.rename(columns={iso_col: "ISO3"})
    gdf["ISO3"] = gdf["ISO3"].astype(str)
    gdf.loc[gdf["ISO3"].isin(["-99", "nan", "None", ""]), "ISO3"] = np.nan

    return gdf


def iso2_to_iso3(cc2: str) -> str:
    cc2 = str(cc2).upper().strip()
    if cc2 not in ISO2_TO_ISO3:
        raise KeyError(f"Kein ISO3 Mapping für ISO2='{cc2}'. Ergänze ISO2_TO_ISO3.")
    return ISO2_TO_ISO3[cc2]


# =============================================================================
# Plotting
# =============================================================================

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
    data = []
    for cc, z in z_by_cc.items():
        try:
            iso3 = iso2_to_iso3(cc)
        except Exception:
            continue
        data.append((iso3, float(z) if np.isfinite(z) else np.nan))

    dfz = pd.DataFrame(data, columns=["ISO3", "z"])
    g = countries_gdf.merge(dfz, on="ISO3", how="left")

    fig = plt.figure(figsize=(9.5, 8.0), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=20)

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

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05, extend="both")
    cb.set_label("Residual-Load Anomalie (Event-Mittel vs Monatsbaseline)", fontsize=20)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] gespeichert: {out_png}")


def plot_overview_grid_top_events_5x2_paginated(
    events_df: pd.DataFrame,
    rl_df: pd.DataFrame,
    countries_gdf: gpd.GeoDataFrame,
    out_dir: str,
    *,
    title: str,
    norm: TwoSlopeNorm,
    cmap: str = "RdBu_r",
    per_page: int = 10,
    dpi: int = 300,
) -> None:
    """
    Overview-Grid als 2 Zeilen x 5 Spalten (insg. 10 Panels) pro Seite.
    Paginiert falls total > per_page.
    """
    if events_df.empty:
        print("[WARN] events_df leer – kein Overview-Grid.")
        return

    events_df = events_df.copy().sort_values("rank") if "rank" in events_df.columns else events_df.copy()
    total = len(events_df)
    pages = int(math.ceil(total / per_page))

    for p in range(pages):
        chunk = events_df.iloc[p * per_page:(p + 1) * per_page].copy()

        fig = plt.figure(figsize=(28.0, 9.6), dpi=dpi)
        gs = fig.add_gridspec(
            nrows=2 + 1,
            ncols=5,
            height_ratios=[1.0, 1.0, 0.07],
            hspace=0.34,
            wspace=0.04,
        )

        for i, rec in enumerate(chunk.to_dict("records")):
            # 2 Reihen, 5 Spalten: zeilenweise füllen
            col = i % 5
            rix = i // 5  # 0 oder 1

            st = pd.to_datetime(rec["start"]).tz_localize(None)
            en = pd.to_datetime(rec["end"]).tz_localize(None)
            val_gw = float(rec.get("value_gw", np.nan))
            rk = rec.get("rank", None)

            z = compute_event_country_z_anomalies(rl_df, st, en, clim_years=CLIM_YEARS)

            data = []
            for cc, zz in z.items():
                try:
                    iso3 = iso2_to_iso3(cc)
                except Exception:
                    continue
                data.append((iso3, float(zz) if np.isfinite(zz) else np.nan))
            dfz = pd.DataFrame(data, columns=["ISO3", "z"])

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

            head = f"#{int(rk):02d} | " if rk is not None and str(rk) != "nan" else ""
            ax.set_title(
                f"{head}{st:%Y-%m-%d}→{en:%Y-%m-%d}\n{val_gw:.1f} GW",
                fontsize=16,
                pad=10,
            )

        # Colorbar unten über alle 5 Spalten
        cax = fig.add_subplot(gs[2, :])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
        cb.set_label(
            "Residual-Load Anomalie (Event-Mittel vs Monatsbaseline)",
            fontsize=18,
        )
        cb.ax.tick_params(labelsize=14)

        page_suffix = "" if pages == 1 else f"__page{p+1:02d}-of-{pages:02d}"
        out_png = os.path.join(
            out_dir,
            f"overview__top{total}__window{EVENT_WINDOW_DAYS}d__RL_robust_z__2x5{page_suffix}.png",
        )

        fig.suptitle(
            f"{title}{'' if pages == 1 else f' (page {p+1}/{pages})'}",
            fontsize=20,
            y=0.975,
        )
        fig.subplots_adjust(top=0.84, bottom=0.07)

        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Overview-Grid gespeichert: {out_png}")


# =============================================================================
# MAIN
# =============================================================================

def _require_file(path: str, desc: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} fehlt: {path}")


def main() -> None:
    _require_file(EXTREMA_CSV, "Extrema-CSV")

    print("\n--- Lade Top-Events aus Extremwert-Tabelle ---")
    top_df = load_top_events_from_extrema_csv(
        csv_path=EXTREMA_CSV,
        window_days=EVENT_WINDOW_DAYS,
        n_events=N_EVENTS,
    )
    if top_df.empty:
        raise RuntimeError("Keine Events aus CSV extrahiert (prüfe Spalten / Parsing).")

    print(top_df.to_string(index=False))

    print("\n--- Lade Residual Load Zeitreihen ---")
    rl = load_country_residual_series()
    print(f"[OK] RL index: {rl.index.min()} bis {rl.index.max()} | cols={len(rl.columns)}")

    print("\n--- Lade NaturalEarth shapes ---")
    countries = load_naturalearth_countries()

    # automatische robuste Skalierung über alle Events
    all_z_vals = []
    for rec in top_df.to_dict("records"):
        st = pd.to_datetime(rec["start"]).tz_localize(None)
        en = pd.to_datetime(rec["end"]).tz_localize(None)
        z = compute_event_country_z_anomalies(rl, st, en, clim_years=CLIM_YEARS)
        all_z_vals.extend(z[np.isfinite(z)].values)

    if all_z_vals:
        lo = np.percentile(all_z_vals, 2)
        hi = np.percentile(all_z_vals, 98)
        z_vmax_auto = max(abs(lo), abs(hi))
    else:
        z_vmax_auto = Z_VMAX  # fallback auf config

    print(f"[OK] Auto z_vmax = {z_vmax_auto:.2f}")
    z_norm = TwoSlopeNorm(vmin=-z_vmax_auto, vcenter=0.0, vmax=z_vmax_auto)

    # 1) Einzelmaps pro Event
    print("\n--- Plot: Einzelmaps pro Top-Event (RL Anomalie) ---")
    for rec in top_df.to_dict("records"):
        rk = int(rec["rank"])
        st = pd.to_datetime(rec["start"]).tz_localize(None)
        en = pd.to_datetime(rec["end"]).tz_localize(None)
        val_gw = float(rec.get("value_gw", np.nan))

        z = compute_event_country_z_anomalies(rl, st, en, clim_years=CLIM_YEARS)

        stamp = f"{st:%Y%m%d}-{en:%Y%m%d}__{EVENT_WINDOW_DAYS}d"
        outpng = os.path.join(MAPS_DIR, f"rl_z_anom__top{rk:02d}__{stamp}.png")

        title = (
            f""
        )

        plot_event_map_rl_z(
            countries_gdf=countries,
            z_by_cc=z,
            out_png=outpng,
            title=title,
            norm=z_norm,
            cmap="RdBu_r",
            dpi=300,
        )

    # 2) Overview grids (2x5, paginiert)
    print("\n--- Plot: Overview grids (Top-Events) — 2 Zeilen x 5 Spalten ---")
    plot_overview_grid_top_events_5x2_paginated(
        events_df=top_df,
        rl_df=rl,
        countries_gdf=countries,
        out_dir=OVERVIEW_DIR,
        title=f"",
        norm=z_norm,
        cmap="RdBu_r",
        per_page=10,
        dpi=300,
    )

    print("\nFertig.")


if __name__ == "__main__":
    main()