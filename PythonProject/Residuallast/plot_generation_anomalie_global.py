#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_generation_anomalies_global.py

"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    import geopandas as gpd
except Exception:
    gpd = None


# =============================================================================
# KONFIGURATION
# =============================================================================

# --- Input: Extremwert-Tabelle (wie in deinem Vorbildskript) ---
MA_BASE = "/mnt/endata/MA_Arthur"
BASE_RESIDUAL_DIR = os.path.join(MA_BASE, "final_results_portfolio_uncorrected", "residual_load_analysis")

ANALYSIS_TYPE = "SYNTHETIC_DEMAND"
if ANALYSIS_TYPE == "SYNTHETIC_DEMAND":
    EXTREMA_CSV = os.path.join(BASE_RESIDUAL_DIR, "extrema_summary_hydro_year_plain.csv")
elif ANALYSIS_TYPE == "DEMAND_2024":
    EXTREMA_CSV = os.path.join(BASE_RESIDUAL_DIR, "extrema_summary_based_on_2024_demand_hydro_year.csv")
else:
    raise ValueError("Ungültiger ANALYSIS_TYPE.")

# --- Auswahl: Fensterlänge M und Anzahl N ---
EVENT_WINDOW_DAYS = 7
N_EVENTS = 10

# CSV enthält meist nur Datum (ohne Uhrzeit). Standard: 00:00.
START_HOUR = 0

# --- Output ---
OUT_ROOT = os.path.join(MA_BASE, "final_results_portfolio_uncorrected", "generation_anom_maps__global_extrema")
OUT_RUN = os.path.join(OUT_ROOT, f"window{EVENT_WINDOW_DAYS}d__top{N_EVENTS}")
MAPS_DIR = os.path.join(OUT_RUN, "maps_single")
OVERVIEW_DIR = os.path.join(OUT_RUN, "overview")
TABLES_DIR = os.path.join(OUT_RUN, "tables")

for d in [MAPS_DIR, OVERVIEW_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[OK] EXTREMA_CSV: {EXTREMA_CSV}")
print(f"[OK] OUT_RUN:    {OUT_RUN}")

# Portfolio generation base
PORTFOLIO_BASE = "/mnt/endata/MA_Arthur/final_results_portfolio_uncorrected"

# File filter keywords
PORTFOLIO_FILE_KEYWORDS = ["final_portfolio_generation", "generation"]

# Time column candidates
TIME_COL_CANDIDATES = ["time", "datetime", "timestamp", "date"]

# Technology column candidates (case-insensitive)
PV_COL_CANDIDATES = [
    "solar_generation_gw", "pv_generation_gw", "pv_gw", "solar_gw", "solar", "pv",
    "solar_generation", "pv_generation",
    "solar_generation_mw", "pv_generation_mw", "pv_mw", "solar_mw",
]
ON_COL_CANDIDATES = [
    "onshore_generation_gw", "wind_onshore_generation_gw", "onshore_wind_generation_gw",
    "onwind_gw", "onshore", "on_wind_gw", "wind_on_gw", "on_gw",
    "onwind", "wind_onshore", "onshore_wind",
    "onshore_generation_mw", "wind_onshore_generation_mw", "onshore_wind_generation_mw",
    "onwind_mw", "on_mw", "wind_on_mw", "wind_onshore_mw", "onshore_wind_mw",
]
OFF_COL_CANDIDATES = [
    "offshore_generation_gw", "wind_offshore_generation_gw", "offshore_wind_generation_gw",
    "offwind_gw", "offshore", "off_wind_gw", "wind_off_gw", "off_gw",
    "offwind", "wind_offshore", "offshore_wind",
    "offshore_generation_mw", "wind_offshore_generation_gw", "offshore_wind_generation_mw",
    "offwind_mw", "off_mw", "wind_off_mw", "wind_offshore_mw", "offshore_wind_mw",
]

# Countries for plotting (ISO2)
COUNTRIES_TO_PLOT = [
    "DE", "FR", "PL", "CZ", "AT", "CH", "IT", "ES", "BE", "NL",
    "DK", "SE", "PT", "NO", "FI", "LT", "LV", "EE", "GB"
]

# ISO2 -> ISO3 (NaturalEarth compatible)
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

# ISO2 -> tokens in filenames (to map "GB" -> "United_Kingdom", etc.)
ISO2_TO_FILENAME_TOKENS: Dict[str, List[str]] = {
    "DE": ["germany"],
    "FR": ["france"],
    "PL": ["poland"],
    "CZ": ["czechia", "czech_republic"],
    "AT": ["austria"],
    "CH": ["switzerland"],
    "IT": ["italy"],
    "ES": ["spain"],
    "BE": ["belgium"],
    "NL": ["netherlands"],
    "DK": ["denmark"],
    "SE": ["sweden"],
    "PT": ["portugal"],
    "NO": ["norway"],
    "FI": ["finland"],
    "LT": ["lithuania"],
    "LV": ["latvia"],
    "EE": ["estonia"],
    "GB": ["united_kingdom", "uk", "great_britain", "britain", "england"],
}

# Map extent
EUROPE_EXTENT = (-12, 35, 35, 72)

# Climatology mode
CLIM_MODE = "multi_year_month"  # or "same_year_month"
CLIM_YEARS = list(range(2025, 2051))

# Anomaly vmax per tech (None => robust from p2/p98)
ANOM_VMAX_PV: Optional[float] = None
ANOM_VMAX_ON: Optional[float] = None
ANOM_VMAX_OFF: Optional[float] = None

# Colormap
ANOM_CMAP = "RdBu_r"

# DPI
GRID_DPI = 300

# Missing-data styling (NaNs become visible)
MISSING_FACE_RGBA = (0.85, 0.85, 0.85, 1.0)


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

    out = pd.DataFrame({
        "rank": np.arange(1, len(df) + 1),
        "start": starts,
        "value_gw": vals,
    }).dropna(subset=["start", "value_gw"])

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
# Helpers: basic utils (wie im Basis-Skript)
# =============================================================================

def _norm_str(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def iso2_to_iso3(cc2: str) -> str:
    cc2 = str(cc2).upper().strip()
    if cc2 not in ISO2_TO_ISO3:
        raise KeyError(f"Kein ISO3 Mapping für ISO2='{cc2}'. Ergänze ISO2_TO_ISO3.")
    return ISO2_TO_ISO3[cc2]


def _detect_column_case_insensitive(columns: List[str], candidates: List[str]) -> Optional[str]:
    colmap = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in colmap:
            return colmap[cand.lower()]
    return None


def _safe_parse_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
    df = df[~df.index.isna()]
    return df.sort_index()


def _infer_year_hint_from_filename(path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    name = path.name.lower()

    m = re.search(r"(?:weather)?\s*([12]\d{3})\s*[-_]\s*([12]\d{3})", name)
    if m:
        y0 = int(m.group(1))
        y1 = int(m.group(2))
        if 1900 <= y0 <= 2200 and 1900 <= y1 <= 2200 and y0 <= y1:
            return y0, y1, None

    m2 = re.search(r"(?:weather|year)?\s*([12]\d{3})", name)
    if m2:
        y = int(m2.group(1))
        if 1900 <= y <= 2200:
            return None, None, y

    return None, None, None


def _infer_freq_from_index(idx: pd.DatetimeIndex) -> Optional[str]:
    if len(idx) < 3:
        return None
    d = idx.to_series().diff().dropna()
    if d.empty:
        return None
    med = d.median()
    if med == pd.Timedelta(hours=1):
        return "H"
    if med == pd.Timedelta(minutes=30):
        return "30min"
    if med == pd.Timedelta(minutes=15):
        return "15min"
    if med == pd.Timedelta(days=1):
        return "D"
    return None


def _rebuild_index_if_needed(df: pd.DataFrame, src: Path) -> pd.DataFrame:
    df = df.copy()
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df

    years = pd.Index(df.index.year.unique()).sort_values()
    if len(years) == 1 and int(years[0]) == 2024:
        y0, y1, y_single = _infer_year_hint_from_filename(src)
        freq = _infer_freq_from_index(df.index) or "H"

        if y_single is not None:
            target_start = pd.Timestamp(y_single, 1, 1, 0, 0)
        elif y0 is not None:
            target_start = pd.Timestamp(y0, 1, 1, 0, 0)
        else:
            return df

        try:
            new_idx = pd.date_range(start=target_start, periods=len(df), freq=freq)
            df.index = new_idx
            return df
        except Exception:
            return df

    return df


def _read_generation_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = _detect_column_case_insensitive(list(df.columns), TIME_COL_CANDIDATES)
    if tcol is not None:
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=[tcol]).set_index(tcol)
        df = _safe_parse_datetime_index(df)
        df = _rebuild_index_if_needed(df, path)
        return df

    df2 = pd.read_csv(path, index_col=0)
    df2 = _safe_parse_datetime_index(df2)
    df2 = _rebuild_index_if_needed(df2, path)
    return df2


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return _detect_column_case_insensitive(list(df.columns), candidates)


def _maybe_to_gw(s: pd.Series, colname: str) -> pd.Series:
    n = str(colname).lower()
    if ("_mw" in n) or n.endswith("mw") or ("mw" in n and "gw" not in n):
        return s.astype(float) / 1000.0
    return s.astype(float)


def _robust_vmax(values: List[float], p_lo: float = 2.0, p_hi: float = 98.0) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return 1.0
    lo = float(np.percentile(v, p_lo))
    hi = float(np.percentile(v, p_hi))
    vmax = max(abs(lo), abs(hi))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(v))) if np.isfinite(np.nanmax(np.abs(v))) else 1.0
    return float(max(vmax, 1e-6))


# =============================================================================
# Generation file indexing + loading
# =============================================================================

def index_generation_csvs(base_dir: str) -> List[Path]:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"PORTFOLIO_BASE fehlt: {base_dir}")

    files = sorted(list(base.rglob("*.csv")))
    scored: List[Tuple[int, Path]] = []
    for f in files:
        name = f.name.lower()
        score = 0
        for kw in PORTFOLIO_FILE_KEYWORDS:
            if kw.lower() in name:
                score += 2
        if score > 0:
            scored.append((score, f))

    scored.sort(key=lambda x: (-x[0], str(x[1])))
    return [p for _, p in scored]


def _score_file_for_country(path: Path, cc2: str) -> int:
    name = path.name.lower()
    cc2_low = cc2.lower()

    score = 0
    if re.search(rf"(^|[^a-z0-9]){re.escape(cc2_low)}([^a-z0-9]|$)", name):
        score += 5
    if cc2_low in name:
        score += 2

    for tok in ISO2_TO_FILENAME_TOKENS.get(cc2.upper(), []):
        if tok.lower() in name:
            score += 3

    if "final_portfolio_generation" in name:
        score += 4

    if "weather" in name:
        score += 1
    if "cap" in name:
        score += 1

    return score


def map_country_to_file(gen_files: List[Path], cc2: str) -> Optional[Path]:
    cc2 = cc2.upper().strip()
    best: Optional[Path] = None
    best_score = 0
    for f in gen_files:
        s = _score_file_for_country(f, cc2)
        if s > best_score:
            best_score = s
            best = f
    return best if best_score >= 3 else None


def load_country_generation_series(
    cc2: str,
    *,
    gen_files: List[Path],
) -> pd.DataFrame:
    cc2 = cc2.upper().strip()

    f = map_country_to_file(gen_files, cc2)
    if f is not None:
        print(f"[MAP] {cc2} -> {f.name}")

    if f is None:
        for cand in gen_files[:250]:
            try:
                df = _read_generation_csv(cand)
            except Exception:
                continue
            pref = cc2 + "_"
            if any(isinstance(c, str) and c.lower().startswith(pref.lower()) for c in df.columns):
                f = cand
                print(f"[MAP-FALLBACK] {cc2} -> {f.name} (prefixed columns detected)")
                break

    if f is None:
        raise FileNotFoundError(
            f"Konnte keine passende Generation-Datei für {cc2} zuordnen. "
            f"Beispiel-Dateien: {[p.name for p in gen_files[:5]]}"
        )

    df = _read_generation_csv(f)

    pv = _pick_col(df, PV_COL_CANDIDATES)
    on = _pick_col(df, ON_COL_CANDIDATES)
    off = _pick_col(df, OFF_COL_CANDIDATES)
    if pv or on or off:
        out = pd.DataFrame(index=df.index)
        out["pv_gw"] = _maybe_to_gw(df[pv], pv) if pv else np.nan
        out["on_gw"] = _maybe_to_gw(df[on], on) if on else np.nan
        out["off_gw"] = _maybe_to_gw(df[off], off) if off else np.nan
        return out

    pref = cc2 + "_"
    lower_map = {str(c).lower(): c for c in df.columns}

    def find_prefixed(cands: List[str]) -> Optional[str]:
        for cand in cands:
            want = (pref + cand).lower()
            if want in lower_map:
                return lower_map[want]
        return None

    pv2 = find_prefixed(PV_COL_CANDIDATES)
    on2 = find_prefixed(ON_COL_CANDIDATES)
    off2 = find_prefixed(OFF_COL_CANDIDATES)

    if pv2 or on2 or off2:
        out = pd.DataFrame(index=df.index)
        out["pv_gw"] = _maybe_to_gw(df[pv2], pv2) if pv2 else np.nan
        out["on_gw"] = _maybe_to_gw(df[on2], on2) if on2 else np.nan
        out["off_gw"] = _maybe_to_gw(df[off2], off2) if off2 else np.nan
        return out

    raise FileNotFoundError(
        f"Datei '{f.name}' zugeordnet, aber keine PV/ON/OFF Spalten gefunden. "
        f"Spaltenbeispiele: {list(df.columns)[:30]}"
    )


# =============================================================================
# Climatology + anomalies
# =============================================================================

def monthly_climatology(series: pd.Series, month: int, years: List[int]) -> float:
    s = series.copy()
    s = s[s.index.year.isin(years)]
    s = s[s.index.month == month]
    if s.empty:
        return float("nan")
    return float(s.mean())


def event_mean(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    s = series.loc[(series.index >= start) & (series.index <= end)]
    if s.empty:
        return float("nan")
    return float(s.mean())


def compute_country_event_anomalies(
    df_country: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    clim_mode: str,
    clim_years: List[int],
) -> Dict[str, float]:
    month = int(pd.to_datetime(start).month)

    if clim_mode == "same_year_month":
        years = [int(pd.to_datetime(start).year)]
    elif clim_mode == "multi_year_month":
        years = clim_years
    else:
        raise ValueError(f"Unknown clim_mode={clim_mode}")

    out: Dict[str, float] = {}
    for key, col in [("pv", "pv_gw"), ("on", "on_gw"), ("off", "off_gw")]:
        if col not in df_country.columns:
            out[key] = np.nan
            continue
        mu_event = event_mean(df_country[col], start, end)
        mu_clim = monthly_climatology(df_country[col], month=month, years=years)
        out[key] = mu_event - mu_clim
    return out


# =============================================================================
# Shapes
# =============================================================================

def load_country_shapes() -> "gpd.GeoDataFrame":
    if gpd is None:
        raise ImportError("geopandas ist nicht verfügbar. Installiere geopandas (shapely/pyproj/fiona).")

    import cartopy.io.shapereader as shpreader

    shp_path = shpreader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries",
    )
    world = gpd.read_file(shp_path)

    iso_candidates = ["ADM0_A3", "ISO_A3", "iso_a3", "ADM0_A3_US", "SOV_A3", "GU_A3", "SU_A3"]
    iso_col = next((c for c in iso_candidates if c in world.columns), None)
    if iso_col is None:
        for c in world.columns:
            cl = c.lower()
            if "a3" in cl and ("iso" in cl or "adm0" in cl or "sov" in cl):
                iso_col = c
                break
    if iso_col is None:
        raise ValueError(
            "Konnte keine ISO3-Spalte finden. Verfügbare Spalten (Auszug): "
            f"{list(world.columns)[:40]} ... (n={len(world.columns)})"
        )

    world = world.copy()
    world["iso_a3"] = world[iso_col].astype(str)
    world.loc[world["iso_a3"].isin(["-99", "nan", "None", ""]), "iso_a3"] = np.nan

    print(f"[SHAPES] using ISO column: {iso_col}")
    print("[SHAPES] iso_a3 top counts:")
    print(world["iso_a3"].value_counts(dropna=True).head(10))
    return world


def build_base_gdf(world: "gpd.GeoDataFrame", countries_iso2: List[str]) -> "gpd.GeoDataFrame":
    iso3_list = [iso2_to_iso3(cc) for cc in countries_iso2]
    gdf = world[world["iso_a3"].isin(iso3_list)].copy()

    iso3_to_iso2 = {iso2_to_iso3(cc): cc for cc in countries_iso2}
    gdf["cc2"] = gdf["iso_a3"].map(iso3_to_iso2)

    missing = [cc for cc in countries_iso2 if iso2_to_iso3(cc) not in set(gdf["iso_a3"].values)]
    if missing:
        print(f"[WARN] Shapes fehlen für: {missing}")
    return gdf


# =============================================================================
# Plotting utils
# =============================================================================

def _colors_with_missing(vals: np.ndarray, cmap_obj, norm, missing_rgba=MISSING_FACE_RGBA):
    vals = np.asarray(vals, dtype=float)
    colors = cmap_obj(norm(vals))
    nan_mask = ~np.isfinite(vals)
    if nan_mask.any():
        colors[nan_mask] = missing_rgba
    return colors


def plot_three_panel_generation_anom(
    gdf: "gpd.GeoDataFrame",
    col_pv: str,
    col_on: str,
    col_off: str,
    title: str,
    outpath: str,
    *,
    vmax_pv: float,
    vmax_on: float,
    vmax_off: float,
    cmap: str,
    dpi: int,
) -> None:
    proj = ccrs.LambertConformal(central_longitude=10, central_latitude=50)

    fig = plt.figure(figsize=(21, 6), dpi=dpi)
    gs = fig.add_gridspec(1, 3, wspace=0.06)
    clean_title = title.replace("\nMissing-data = grey", "").replace("Missing-data = grey", "")
    fig.suptitle(clean_title, fontsize=18, y=0.99)

    def setup_ax(ax):
        ax.set_extent(EUROPE_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.LAKES, alpha=0.2)
        ax.add_feature(cfeature.RIVERS, alpha=0.15)
        return ax

    norms = [
        TwoSlopeNorm(vmin=-vmax_pv, vcenter=0.0, vmax=vmax_pv),
        TwoSlopeNorm(vmin=-vmax_on, vcenter=0.0, vmax=vmax_on),
        TwoSlopeNorm(vmin=-vmax_off, vcenter=0.0, vmax=vmax_off),
    ]
    cols = [col_pv, col_on, col_off]
    titles = ["Solar generation anomaly [GW]", "Onshore wind anomaly [GW]", "Offshore wind anomaly [GW]"]

    cmap_obj = plt.get_cmap(cmap)

    for j in range(3):
        ax = fig.add_subplot(gs[0, j], projection=proj)
        setup_ax(ax)

        ax.add_feature(cfeature.LAND, alpha=0.12)
        ax.add_feature(cfeature.OCEAN, alpha=0.05)

        v = gdf[cols[j]].to_numpy(dtype=float)
        colors = _colors_with_missing(v, cmap_obj, norms[j])

        for geom, color in zip(gdf.geometry, colors):
            if geom is None or geom.is_empty:
                continue
            ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor=color, edgecolor="black", linewidth=0.25)

        ax.set_title(titles[j], fontsize=18)

        sm = plt.cm.ScalarMappable(norm=norms[j], cmap=cmap_obj)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05, fraction=0.045, extend="both")
        cb.ax.tick_params(labelsize=18)

    fig.text(0.5, 0.01, "Grau = fehlende Daten", ha="center", fontsize=18, alpha=0.75)
    fig.subplots_adjust(top=0.88)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] gespeichert: {outpath}")


def plot_event_grid_5x2_triplets(
    panels: List[Dict[str, object]],
    out_png: str,
    *,
    title: str,
    vmax_pv: float,
    vmax_on: float,
    vmax_off: float,
    cmap: str,
    dpi: int,
    page_label: Optional[str] = None,
) -> None:
    """
    panels: list of dicts with keys:
      - "gdf": GeoDataFrame with columns pv_anom_gw/on_anom_gw/off_anom_gw
      - "label": two-line string drawn inside PV panel of each event
    Layout: 5 rows x 2 cols of EVENTS (10 events).
            Each event has 3 panels (PV/ON/OFF) side-by-side.
    => 5 map rows x 6 cols total. If panels < 10, remaining are blank.

    Colorbar layout:
      Row 5:  map row 5 (index 4) ends here
      Row 6:  spacer
      Row 7:  colorbar PV  (tall)
      Row 8:  spacer
      Row 9:  colorbar ON  (tall)
      Row 10: spacer
      Row 11: colorbar OFF (tall)
    """
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    proj = ccrs.LambertConformal(central_longitude=10, central_latitude=50)
    cmap_obj = plt.get_cmap(cmap)

    norm_pv  = TwoSlopeNorm(vmin=-vmax_pv,  vcenter=0.0, vmax=vmax_pv)
    norm_on  = TwoSlopeNorm(vmin=-vmax_on,  vcenter=0.0, vmax=vmax_on)
    norm_off = TwoSlopeNorm(vmin=-vmax_off, vcenter=0.0, vmax=vmax_off)

    def setup_ax(ax):
        ax.set_extent(EUROPE_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.35)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.25)
        ax.add_feature(cfeature.LAKES,     alpha=0.15)
        ax.add_feature(cfeature.RIVERS,    alpha=0.10)
        ax.add_feature(cfeature.LAND,      alpha=0.12)
        ax.add_feature(cfeature.OCEAN,     alpha=0.05)

    def draw(ax, gg: "gpd.GeoDataFrame", col: str, norm):
        vals = gg[col].to_numpy(dtype=float)
        colors = _colors_with_missing(vals, cmap_obj, norm)
        for geom, color in zip(gg.geometry, colors):
            if geom is None or geom.is_empty:
                continue
            ax.add_geometries(
                [geom],
                crs=ccrs.PlateCarree(),
                facecolor=color,
                edgecolor="black",
                linewidth=0.20,
            )

    head_titles = ["PV anom [GW]", "Onshore anom [GW]", "Offshore anom [GW]"]

    # -------------------------------------------------------------------------
    # Figure + GridSpec
    # 5 map rows  +  3 × (spacer + colorbar) = 6 extra rows  =>  11 rows total
    #
    # height_ratios:
    #   map rows  [0-4] : 1.0 each
    #   spacer    [5]   : 0.10  (breathing room before first colorbar)
    #   colorbar  [6]   : 0.28  (PV  — extra tall so label + ticks never clip)
    #   spacer    [7]   : 0.18  (clear gap between colorbars)
    #   colorbar  [8]   : 0.28  (ON)
    #   spacer    [9]   : 0.18
    #   colorbar  [10]  : 0.28  (OFF)
    # -------------------------------------------------------------------------
    CB_H   = 0.14   # colorbar row height (relative)
    SP_TOP = 0.10   # first spacer (maps → first colorbar)
    SP_MID = 0.18   # spacers between colorbars

    fig = plt.figure(
        figsize=(22.0, 4.0 * 5 + 4.0),   # extra 4 inches vs. original for colorbars
        dpi=dpi,
    )
    gs = fig.add_gridspec(
        nrows=11,
        ncols=6,
        height_ratios=[1.0] * 5 + [SP_TOP, CB_H, SP_MID, CB_H, SP_MID, CB_H],
        hspace=0.08,
        wspace=0.02,
    )

    # --- Event panels (5 map rows × 6 cols) ---
    for i in range(10):
        r    = i // 2         # row index 0-4
        side = i % 2          # 0 = left half, 1 = right half

        if i >= len(panels):
            for j in range(3):
                ax = fig.add_subplot(gs[r, 3 * side + j])
                ax.axis("off")
            continue

        gg    = panels[i]["gdf"]
        label = str(panels[i].get("label", ""))

        for j, (col, norm) in enumerate(
            [("pv_anom_gw", norm_pv), ("on_anom_gw", norm_on), ("off_anom_gw", norm_off)]
        ):
            ax = fig.add_subplot(gs[r, 3 * side + j], projection=proj)
            setup_ax(ax)
            draw(ax, gg, col, norm)

            # Column headers only in the very first row
            if r == 0:
                ax.set_title(head_titles[j], fontsize=18, pad=4)

            # Two-line event label in the PV panel
            if j == 0:
                ax.text(
                    0.01, 0.98,
                    label,
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=18,
                    linespacing=1.4,
                    bbox=dict(
                        boxstyle="round,pad=0.30",
                        fc="white", ec="none",
                        alpha=0.80,
                    ),
                )

    # -------------------------------------------------------------------------
    # Colorbar axes  (rows 6 / 8 / 10 — spacer rows 5 / 7 / 9 stay empty)
    # -------------------------------------------------------------------------
    cax_pv  = fig.add_subplot(gs[6,  :])
    cax_on  = fig.add_subplot(gs[8,  :])
    cax_off = fig.add_subplot(gs[10, :])

    sm_pv  = plt.cm.ScalarMappable(norm=norm_pv,  cmap=cmap_obj); sm_pv.set_array([])
    sm_on  = plt.cm.ScalarMappable(norm=norm_on,  cmap=cmap_obj); sm_on.set_array([])
    sm_off = plt.cm.ScalarMappable(norm=norm_off, cmap=cmap_obj); sm_off.set_array([])

    cb1 = fig.colorbar(sm_pv,  cax=cax_pv,  orientation="horizontal", extend="both")
    cb2 = fig.colorbar(sm_on,  cax=cax_on,  orientation="horizontal", extend="both")
    cb3 = fig.colorbar(sm_off, cax=cax_off, orientation="horizontal", extend="both")

    # Label on top of each colorbar so it sits on its own line above the ticks
    for cb, lbl in [
        (cb1, "PV Anomalie [GW]"),
        (cb2, "Onshore Wind Anomalie [GW]"),
        (cb3, "Offshore Wind Anomalie [GW]"),
    ]:
        cb.set_label(lbl, fontsize=18, labelpad=8)
        cb.ax.xaxis.set_major_locator(MaxNLocator(nbins=7, prune="both"))
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cb.ax.tick_params(labelsize=18, pad=4)

    # Overall title
    t = title
    if page_label:
        t += f"  ({page_label})"
    fig.suptitle(t, fontsize=18, y=0.995)

    fig.subplots_adjust(top=0.975, bottom=0.02, left=0.02, right=0.99)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(f"[OK] Event grid 5x2 (triplets) gespeichert: {out_png}")


# =============================================================================
# MAIN
# =============================================================================

def _require_file(path: str, desc: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} fehlt: {path}")


def main() -> None:
    if gpd is None:
        raise ImportError("geopandas ist nicht verfügbar. Installiere geopandas (inkl. shapely/pyproj/fiona).")

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

    # Shapes
    world = load_country_shapes()
    base_gdf = build_base_gdf(world, COUNTRIES_TO_PLOT)

    # Index generation CSVs
    print("\n--- Indexiere generation CSVs ---")
    gen_files = index_generation_csvs(PORTFOLIO_BASE)
    if not gen_files:
        raise RuntimeError(f"Keine generation CSV candidates unter {PORTFOLIO_BASE} gefunden.")
    print(f"[OK] Generation CSV candidates: {len(gen_files)} (Top 5: {[p.name for p in gen_files[:5]]})")

    # Load generation timeseries per country
    print("\n--- Lade generation Zeitreihen pro Land ---")
    gen_ts: Dict[str, pd.DataFrame] = {}
    for cc in COUNTRIES_TO_PLOT:
        try:
            df = load_country_generation_series(cc, gen_files=gen_files)
            gen_ts[cc] = df
            print(f" - {cc}: {df.shape} cols={list(df.columns)}")
        except Exception as e:
            print(f" [WARN] {cc}: konnte nicht geladen werden: {e}")

    if not gen_ts:
        raise RuntimeError("Keine Generation-Zeitreihen geladen – Check PORTFOLIO_BASE + Dateinamen/Spalten.")

    # Diagnose: Coverage + NaNs pro Land
    print("\n--- Diagnose: Coverage + NaNs pro Land ---")
    for cc, df in gen_ts.items():
        y0i, y1i = int(df.index.min().year), int(df.index.max().year)
        nan_share = df.isna().mean().to_dict()
        print(f"{cc}: years={y0i}-{y1i} | NaN-share={nan_share}")

    # Helper: compute event maps
    def compute_event_map(
        start: pd.Timestamp, end: pd.Timestamp
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        anom_pv: Dict[str, float] = {}
        anom_on: Dict[str, float] = {}
        anom_off: Dict[str, float] = {}
        for cc in COUNTRIES_TO_PLOT:
            if cc not in gen_ts:
                anom_pv[cc] = np.nan
                anom_on[cc] = np.nan
                anom_off[cc] = np.nan
                continue
            a = compute_country_event_anomalies(
                gen_ts[cc], start, end,
                clim_mode=CLIM_MODE,
                clim_years=CLIM_YEARS,
            )
            anom_pv[cc] = a["pv"]
            anom_on[cc] = a["on"]
            anom_off[cc] = a["off"]
        return anom_pv, anom_on, anom_off

    # Pass 1: collect anomalies for vmax scaling across all plotted events
    print("\n--- Pass 1: Sammle Anomalien für Skalierung (vmax) ---")
    pv_vals:  List[float] = []
    on_vals:  List[float] = []
    off_vals: List[float] = []

    event_gdfs: List[Dict[str, object]] = []

    for rec in top_df.to_dict("records"):
        rank   = int(rec["rank"])
        start  = pd.to_datetime(rec["start"])
        end    = pd.to_datetime(rec["end"])
        val_gw = float(rec["value_gw"])

        pv_map, on_map, off_map = compute_event_map(start, end)

        pv_vals.extend( [v for v in pv_map.values()  if np.isfinite(v)])
        on_vals.extend( [v for v in on_map.values()  if np.isfinite(v)])
        off_vals.extend([v for v in off_map.values() if np.isfinite(v)])

        gg = base_gdf.copy()
        gg["pv_anom_gw"]  = gg["cc2"].map(pv_map)
        gg["on_anom_gw"]  = gg["cc2"].map(on_map)
        gg["off_anom_gw"] = gg["cc2"].map(off_map)

        event_gdfs.append({
            "rank":     rank,
            "start":    start,
            "end":      end,
            "value_gw": val_gw,
            "gdf":      gg,
        })

    vmax_pv  = float(ANOM_VMAX_PV)  if ANOM_VMAX_PV  is not None else _robust_vmax(pv_vals)
    vmax_on  = float(ANOM_VMAX_ON)  if ANOM_VMAX_ON  is not None else _robust_vmax(on_vals)
    vmax_off = float(ANOM_VMAX_OFF) if ANOM_VMAX_OFF is not None else _robust_vmax(off_vals)
    print(f"[OK] vmax PV/ON/OFF = {vmax_pv:.3f} / {vmax_on:.3f} / {vmax_off:.3f} GW (symmetrisch)")

    # 1) Einzelmaps pro Event
    print("\n--- Plot: Einzelmaps pro Top-Event (3-panel) ---")
    for ev in event_gdfs:
        rank   = int(ev["rank"])
        start  = pd.to_datetime(ev["start"])
        end    = pd.to_datetime(ev["end"])
        val_gw = float(ev["value_gw"])
        gg     = ev["gdf"]

        stamp = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}__{EVENT_WINDOW_DAYS}d"
        title = (
            f"Top-Event #{rank:02d} | Generation anomalies ({CLIM_MODE})\n"
            f"{start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} | Residual mean: {val_gw:.2f} GW"
        )
        outpng = os.path.join(MAPS_DIR, f"generation_anom__top{rank:02d}__{stamp}.png")

        plot_three_panel_generation_anom(
            gdf=gg,
            col_pv="pv_anom_gw",
            col_on="on_anom_gw",
            col_off="off_anom_gw",
            title=title,
            outpath=outpng,
            vmax_pv=vmax_pv,
            vmax_on=vmax_on,
            vmax_off=vmax_off,
            cmap=ANOM_CMAP,
            dpi=260,
        )

    # 2) Overview-Grid (5×2, pro Event 3 Panels nebeneinander) – paginiert
    print("\n--- Plot: Overview grids (Top-Events) — 5x2 (2 Spalten) ---")
    panels: List[Dict[str, object]] = []
    for ev in event_gdfs:
        rank   = int(ev["rank"])
        start  = pd.to_datetime(ev["start"])
        end    = pd.to_datetime(ev["end"])
        val_gw = float(ev["value_gw"])

        # ---------------------------------------------------------------
        # Two-line label:
        #   Line 1 – rank + residual value
        #   Line 2 – date range
        # ---------------------------------------------------------------
        label = (
            f"#{rank:02d}  |  {val_gw:.1f} GW\n"
            f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"
        )
        panels.append({"gdf": ev["gdf"], "label": label})

    pages = [panels[i:i + 10] for i in range(0, len(panels), 10)]
    for page_i, page_panels in enumerate(pages, start=1):
        page_suffix = "" if len(pages) == 1 else f"__page{page_i:02d}"
        out_overview = os.path.join(
            OVERVIEW_DIR,
            f"overview__top{len(panels)}__window{EVENT_WINDOW_DAYS}d__{CLIM_MODE}{page_suffix}.png",
        )

        plot_event_grid_5x2_triplets(
            panels=page_panels,
            out_png=out_overview,
            title=(
                f""
            ),
            vmax_pv=vmax_pv,
            vmax_on=vmax_on,
            vmax_off=vmax_off,
            cmap=ANOM_CMAP,
            dpi=GRID_DPI,
            page_label=f"Seite {page_i}/{len(pages)}" if len(pages) > 1 else None,
        )

    print("\nFertig.")


if __name__ == "__main__":
    main()