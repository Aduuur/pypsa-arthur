#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
PERSISTENCE_DAYS = 5
K_CLUSTERS = 8
Y0, Y1 = 2025, 2050
EVENT_WINDOW_DAYS = 7
RUN_TAG = f"p{PERSISTENCE_DAYS}d_k{K_CLUSTERS}_{Y0}_{Y1}"
MA_BASE = "/mnt/endata/MA_Arthur"
OUT_BASE = os.path.join(
    MA_BASE,
    "final_results_portfolio_uncorrected",
    "regime_dunkelflaute_maps__experiments_v2",
)
OUT_ROOT = os.path.join(OUT_BASE, RUN_TAG)
TABLES_DIR              = os.path.join(OUT_ROOT, "tables")
IN_HEAVIEST_CSV         = os.path.join(TABLES_DIR, f"heaviest_event_per_regime__{EVENT_WINDOW_DAYS}d.csv")
IN_TOP10_PER_REGIME_CSV = os.path.join(TABLES_DIR, f"top10_events_per_regime__{EVENT_WINDOW_DAYS}d.csv")
GEN_MAPS_DIR           = os.path.join(OUT_ROOT, "generation_anom_maps__heaviest_per_regime")
GRID_DIR               = os.path.join(OUT_ROOT, "generation_anom_grids")
GRID_HEAVIEST_DIR      = os.path.join(GRID_DIR, "overview_heaviest_per_regime")
GRID_TOP10_PER_REGIME_DIR = os.path.join(GRID_DIR, "top10_per_regime")
for d in [GEN_MAPS_DIR, GRID_DIR, GRID_HEAVIEST_DIR, GRID_TOP10_PER_REGIME_DIR]:
    os.makedirs(d, exist_ok=True)
PORTFOLIO_BASE         = "/mnt/endata/MA_Arthur/final_results_portfolio_uncorrected"
PORTFOLIO_FILE_KEYWORDS = ["final_portfolio_generation", "generation"]
TIME_COL_CANDIDATES    = ["time", "datetime", "timestamp", "date"]
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
    "offshore_generation_mw", "wind_offshore_generation_mw", "offshore_wind_generation_mw",
    "offwind_mw", "off_mw", "wind_off_mw", "wind_offshore_mw", "offshore_wind_mw",
]
COUNTRIES_TO_PLOT = [
    "DE", "FR", "PL", "CZ", "AT", "CH", "IT", "ES", "BE", "NL",
    "DK", "SE", "PT", "NO", "FI", "LT", "LV", "EE", "GB"
]
ISO2_TO_ISO3: Dict[str, str] = {
    "DE": "DEU", "FR": "FRA", "PL": "POL", "CZ": "CZE", "AT": "AUT",
    "CH": "CHE", "IT": "ITA", "ES": "ESP", "BE": "BEL", "NL": "NLD",
    "DK": "DNK", "SE": "SWE", "PT": "PRT", "NO": "NOR", "FI": "FIN",
    "LT": "LTU", "LV": "LVA", "EE": "EST", "GB": "GBR",
}
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
EUROPE_EXTENT = (-12, 35, 35, 72)
CLIM_MODE     = "multi_year_month"
CLIM_YEARS    = list(range(2025, 2051))
ANOM_VMAX_PV:  Optional[float] = None
ANOM_VMAX_ON:  Optional[float] = None
ANOM_VMAX_OFF: Optional[float] = None
ANOM_CMAP = "RdBu_r"
GRID_DPI  = 300
MISSING_FACE_RGBA = (0.85, 0.85, 0.85, 1.0)
# =============================================================================
# Helpers: basic utils
# =============================================================================
def _norm_str(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")
def iso2_to_iso3(cc2: str) -> str:
    cc2 = str(cc2).upper().strip()
    if cc2 not in ISO2_TO_ISO3:
        raise KeyError(f"Kein ISO3 Mapping fuer ISO2='{cc2}'.")
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
        y0 = int(m.group(1)); y1 = int(m.group(2))
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
    d   = idx.to_series().diff().dropna()
    med = d.median()
    if med == pd.Timedelta(hours=1):      return "H"
    if med == pd.Timedelta(minutes=30):   return "30min"
    if med == pd.Timedelta(minutes=15):   return "15min"
    if med == pd.Timedelta(days=1):       return "D"
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
            df.index = pd.date_range(start=target_start, periods=len(df), freq=freq)
        except Exception:
            pass
    return df
def _read_generation_csv(path: Path) -> pd.DataFrame:
    df   = pd.read_csv(path)
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
    lo   = float(np.percentile(v, p_lo))
    hi   = float(np.percentile(v, p_hi))
    vmax = max(abs(lo), abs(hi))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(v))) if np.isfinite(np.nanmax(np.abs(v))) else 1.0
    return float(max(vmax, 1e-6))
# =============================================================================
# Generation file indexing + loading
# =============================================================================
def index_generation_csvs(base_dir: str) -> List[Path]:
    base  = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"PORTFOLIO_BASE fehlt: {base_dir}")
    files  = sorted(list(base.rglob("*.csv")))
    scored: List[Tuple[int, Path]] = []
    for f in files:
        name  = f.name.lower()
        score = sum(2 for kw in PORTFOLIO_FILE_KEYWORDS if kw.lower() in name)
        if score > 0:
            scored.append((score, f))
    scored.sort(key=lambda x: (-x[0], str(x[1])))
    return [p for _, p in scored]
def _score_file_for_country(path: Path, cc2: str) -> int:
    name    = path.name.lower()
    cc2_low = cc2.lower()
    score   = 0
    if re.search(rf"(^|[^a-z0-9]){re.escape(cc2_low)}([^a-z0-9]|$)", name): score += 5
    if cc2_low in name:                                                        score += 2
    for tok in ISO2_TO_FILENAME_TOKENS.get(cc2.upper(), []):
        if tok.lower() in name: score += 3
    if "final_portfolio_generation" in name: score += 4
    if "weather" in name: score += 1
    if "cap"     in name: score += 1
    return score
def map_country_to_file(gen_files: List[Path], cc2: str) -> Optional[Path]:
    cc2 = cc2.upper().strip()
    best, best_score = None, 0
    for f in gen_files:
        s = _score_file_for_country(f, cc2)
        if s > best_score:
            best_score = s; best = f
    return best if best_score >= 3 else None
def load_country_generation_series(cc2: str, *, gen_files: List[Path]) -> pd.DataFrame:
    cc2 = cc2.upper().strip()
    f   = map_country_to_file(gen_files, cc2)
    if f is not None:
        print(f"[MAP] {cc2} -> {f.name}")
    if f is None:
        for cand in gen_files[:250]:
            try:
                df   = _read_generation_csv(cand)
            except Exception:
                continue
            pref = cc2 + "_"
            if any(isinstance(c, str) and c.lower().startswith(pref.lower()) for c in df.columns):
                f = cand
                print(f"[MAP-FALLBACK] {cc2} -> {f.name} (prefixed columns detected)")
                break
    if f is None:
        raise FileNotFoundError(f"Konnte keine passende Generation-Datei fuer {cc2} zuordnen.")
    df  = _read_generation_csv(f)
    pv  = _pick_col(df, PV_COL_CANDIDATES)
    on  = _pick_col(df, ON_COL_CANDIDATES)
    off = _pick_col(df, OFF_COL_CANDIDATES)
    if pv or on or off:
        out = pd.DataFrame(index=df.index)
        out["pv_gw"]  = _maybe_to_gw(df[pv],  pv)  if pv  else np.nan
        out["on_gw"]  = _maybe_to_gw(df[on],  on)  if on  else np.nan
        out["off_gw"] = _maybe_to_gw(df[off], off) if off else np.nan
        return out
    pref      = cc2 + "_"
    lower_map = {str(c).lower(): c for c in df.columns}
    def find_prefixed(cands):
        for cand in cands:
            want = (pref + cand).lower()
            if want in lower_map: return lower_map[want]
        return None
    pv2  = find_prefixed(PV_COL_CANDIDATES)
    on2  = find_prefixed(ON_COL_CANDIDATES)
    off2 = find_prefixed(OFF_COL_CANDIDATES)
    if pv2 or on2 or off2:
        out = pd.DataFrame(index=df.index)
        out["pv_gw"]  = _maybe_to_gw(df[pv2],  pv2)  if pv2  else np.nan
        out["on_gw"]  = _maybe_to_gw(df[on2],  on2)  if on2  else np.nan
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
    s = series[series.index.year.isin(years)]
    s = s[s.index.month == month]
    return float("nan") if s.empty else float(s.mean())
def event_mean(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    s = series.loc[(series.index >= start) & (series.index <= end)]
    return float("nan") if s.empty else float(s.mean())
def compute_country_event_anomalies(
    df_country: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    clim_mode: str,
    clim_years: List[int],
) -> Dict[str, float]:
    month = int(pd.to_datetime(start).month)
    years = [int(pd.to_datetime(start).year)] if clim_mode == "same_year_month" else clim_years
    out: Dict[str, float] = {}
    for key, col in [("pv", "pv_gw"), ("on", "on_gw"), ("off", "off_gw")]:
        if col not in df_country.columns:
            out[key] = np.nan; continue
        mu_event = event_mean(df_country[col], start, end)
        mu_clim  = monthly_climatology(df_country[col], month=month, years=years)
        out[key] = mu_event - mu_clim
    return out
# =============================================================================
# Shapes
# =============================================================================
def load_country_shapes() -> "gpd.GeoDataFrame":
    if gpd is None:
        raise ImportError("geopandas ist nicht verfuegbar.")
    import cartopy.io.shapereader as shpreader
    shp_path = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
    world    = gpd.read_file(shp_path)
    iso_candidates = ["ADM0_A3", "ISO_A3", "iso_a3", "ADM0_A3_US", "SOV_A3", "GU_A3", "SU_A3"]
    iso_col = next((c for c in iso_candidates if c in world.columns), None)
    if iso_col is None:
        for c in world.columns:
            cl = c.lower()
            if "a3" in cl and ("iso" in cl or "adm0" in cl or "sov" in cl):
                iso_col = c; break
    if iso_col is None:
        raise ValueError(f"Konnte keine ISO3-Spalte finden. Spalten: {list(world.columns)[:40]}")
    world         = world.copy()
    world["iso_a3"] = world[iso_col].astype(str)
    world.loc[world["iso_a3"].isin(["-99", "nan", "None", ""]), "iso_a3"] = np.nan
    print(f"[SHAPES] using ISO column: {iso_col}")
    return world
def build_base_gdf(world: "gpd.GeoDataFrame", countries_iso2: List[str]) -> "gpd.GeoDataFrame":
    iso3_list   = [iso2_to_iso3(cc) for cc in countries_iso2]
    gdf         = world[world["iso_a3"].isin(iso3_list)].copy()
    iso3_to_iso2 = {iso2_to_iso3(cc): cc for cc in countries_iso2}
    gdf["cc2"]  = gdf["iso_a3"].map(iso3_to_iso2)
    missing = [cc for cc in countries_iso2 if iso2_to_iso3(cc) not in set(gdf["iso_a3"].values)]
    if missing:
        print(f"[WARN] Shapes fehlen fuer: {missing}")
    return gdf
# =============================================================================
# Plotting utils
# =============================================================================
def _colors_with_missing(vals: np.ndarray, cmap_obj, norm, missing_rgba=MISSING_FACE_RGBA):
    vals   = np.asarray(vals, dtype=float)
    colors = cmap_obj(norm(vals))
    nan_mask = ~np.isfinite(vals)
    if nan_mask.any():
        colors[nan_mask] = missing_rgba
    return colors


def _add_colorbars(
    fig: plt.Figure,
    norm_pv:  TwoSlopeNorm,
    norm_on:  TwoSlopeNorm,
    norm_off: TwoSlopeNorm,
    cmap_obj,
) -> None:
    """
    Drei horizontale Colorbars in festen Figure-Koordinaten.
    Layout von oben nach unten: PV (0.14), ON (0.08), OFF (0.02)
    Ticks via set_ticks(linspace) – funktioniert zuverlässig mit TwoSlopeNorm.
    """
    CB_LEFT   = 0.10
    CB_WIDTH  = 0.80
    CB_HEIGHT = 0.022

    cax_pv  = fig.add_axes([CB_LEFT, 0.14, CB_WIDTH, CB_HEIGHT])
    cax_on  = fig.add_axes([CB_LEFT, 0.08, CB_WIDTH, CB_HEIGHT])
    cax_off = fig.add_axes([CB_LEFT, 0.02, CB_WIDTH, CB_HEIGHT])

    for norm, cax, label in [
        (norm_pv,  cax_pv,  "PV-Erzeugung Anomalie vs Monatsklima [GW]"),
        (norm_on,  cax_on,  "Onshore-Wind Anomalie vs Monatsklima [GW]"),
        (norm_off, cax_off, "Offshore-Wind Anomalie vs Monatsklima [GW]"),
    ]:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
        cb.set_label(label, fontsize=18)
        cb.set_ticks(np.linspace(norm.vmin, norm.vmax, 9))
        cb.ax.tick_params(labelsize=18)


# =============================================================================
# plot_three_panel_generation_anom  (Einzelkarten – unverändert außer Fußzeile)
# =============================================================================
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
    proj     = ccrs.LambertConformal(central_longitude=10, central_latitude=50)
    cmap_obj = plt.get_cmap(cmap)
    fig = plt.figure(figsize=(21, 6), dpi=dpi)
    gs  = fig.add_gridspec(1, 3, wspace=0.06)
    clean_title = title.replace("\nMissing-data = grey", "").replace("Missing-data = grey", "")
    fig.suptitle(clean_title, fontsize=18, y=0.99)

    def setup_ax(ax):
        ax.set_extent(EUROPE_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4)
        ax.add_feature(cfeature.LAKES,     alpha=0.2)
        ax.add_feature(cfeature.RIVERS,    alpha=0.15)
        return ax

    norms  = [
        TwoSlopeNorm(vmin=-vmax_pv,  vcenter=0.0, vmax=vmax_pv),
        TwoSlopeNorm(vmin=-vmax_on,  vcenter=0.0, vmax=vmax_on),
        TwoSlopeNorm(vmin=-vmax_off, vcenter=0.0, vmax=vmax_off),
    ]
    cols   = [col_pv, col_on, col_off]
    titles = [
        "Solar generation anomaly [GW]",
        "Onshore wind anomaly [GW]",
        "Offshore wind anomaly [GW]",
    ]
    for j in range(3):
        ax = fig.add_subplot(gs[0, j], projection=proj)
        setup_ax(ax)
        ax.add_feature(cfeature.LAND,  alpha=0.12)
        ax.add_feature(cfeature.OCEAN, alpha=0.05)
        v      = gdf[cols[j]].to_numpy(dtype=float)
        colors = _colors_with_missing(v, cmap_obj, norms[j])
        for geom, color in zip(gdf.geometry, colors):
            if geom is None or geom.is_empty: continue
            ax.add_geometries(
                [geom], crs=ccrs.PlateCarree(),
                facecolor=color, edgecolor="black", linewidth=0.25,
            )
        ax.set_title(titles[j], fontsize=18, pad=4)
        sm = plt.cm.ScalarMappable(norm=norms[j], cmap=cmap_obj)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05, fraction=0.045, extend="both")
        cb.set_ticks(np.linspace(norms[j].vmin, norms[j].vmax, 9))
        cb.ax.tick_params(labelsize=18)
    fig.text(0.5, 0.01, "Grau = fehlende Daten", ha="center", fontsize=18, alpha=0.75)
    fig.subplots_adjust(top=0.88)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] gespeichert: {outpath}")


# =============================================================================
# Shared helpers for grid plots
# =============================================================================
def _sort_key_regime(reg: str) -> Tuple[int, int]:
    if str(reg).lower() in {"no_regime", "none", "nan"}:
        return (1, 10**9)
    try:    return (0, int(reg))
    except: return (0, 10**8)


def _setup_grid_ax(ax):
    ax.set_extent(EUROPE_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.25)
    ax.add_feature(cfeature.LAKES,     alpha=0.15)
    ax.add_feature(cfeature.RIVERS,    alpha=0.10)
    ax.add_feature(cfeature.LAND,      alpha=0.12)
    ax.add_feature(cfeature.OCEAN,     alpha=0.05)


def _draw_country_panel(ax, gdf, col, norm, cmap_obj):
    vals   = gdf[col].to_numpy(dtype=float)
    colors = _colors_with_missing(vals, cmap_obj, norm)
    for geom, color in zip(gdf.geometry, colors):
        if geom is None or geom.is_empty: continue
        ax.add_geometries(
            [geom], crs=ccrs.PlateCarree(),
            facecolor=color, edgecolor="black", linewidth=0.20,
        )


def _make_grid_figure(n_data_rows: int) -> Tuple[plt.Figure, "plt.GridSpec"]:
    """
    Erzeugt Figure + GridSpec für ein 5-Zeilen-×-6-Spalten-Grid.
    bottom=0.22 reserviert Platz für die 3 Colorbars via _add_colorbars().
    """
    proj = ccrs.LambertConformal(central_longitude=10, central_latitude=50)
    fig = plt.figure(figsize=(22.0, 4.5 * n_data_rows + 2.0))
    gs  = fig.add_gridspec(
        nrows=n_data_rows, ncols=6,
        hspace=0.30, wspace=0.02,
        top=0.93, bottom=0.22, left=0.02, right=0.99,
    )
    return fig, gs


# =============================================================================
# plot_regime_grid_5x2
# Layout: links→rechts, oben→unten (slot 0 links, slot 1 rechts, slot 2 links, …)
# Labels: zweizeilig (Regime + GW oben, Datum unten)
# =============================================================================
def plot_regime_grid_5x2(
    per_regime_gdfs: Dict[str, "gpd.GeoDataFrame"],
    per_regime_meta: Dict[str, Dict[str, object]],
    out_png: str,
    *,
    title: str,
    vmax_pv: float,
    vmax_on: float,
    vmax_off: float,
    cmap: str,
    dpi: int,
    regimes_order: Optional[List[str]] = None,
) -> None:
    if not per_regime_gdfs:
        print("[WARN] plot_regime_grid_5x2: no regimes."); return

    regs = list(per_regime_gdfs.keys())
    regs = sorted(regs, key=_sort_key_regime) if regimes_order is None \
           else [r for r in regimes_order if r in per_regime_gdfs]

    pages    = [regs[i:i+10] for i in range(0, len(regs), 10)]
    proj     = ccrs.LambertConformal(central_longitude=10, central_latitude=50)
    cmap_obj = plt.get_cmap(cmap)
    norm_pv  = TwoSlopeNorm(vmin=-vmax_pv,  vcenter=0.0, vmax=vmax_pv)
    norm_on  = TwoSlopeNorm(vmin=-vmax_on,  vcenter=0.0, vmax=vmax_on)
    norm_off = TwoSlopeNorm(vmin=-vmax_off, vcenter=0.0, vmax=vmax_off)
    col_trip    = [("pv_anom_gw", norm_pv), ("on_anom_gw", norm_on), ("off_anom_gw", norm_off)]
    head_titles = ["PV anom [GW]", "Onshore anom [GW]", "Offshore anom [GW]"]

    for page_i, page_regs in enumerate(pages, start=1):
        N_ROWS = 5
        fig, gs = _make_grid_figure(N_ROWS)

        for slot, reg in enumerate(page_regs):
            # Row-first: slot 0→(row=0,left), slot 1→(row=0,right),
            #            slot 2→(row=1,left), slot 3→(row=1,right), …
            row  = slot // 2
            side = slot % 2          # 0 = left (cols 0-2), 1 = right (cols 3-5)

            gg   = per_regime_gdfs[reg]
            meta = per_regime_meta.get(reg, {})
            st   = pd.to_datetime(meta.get("start", pd.NaT))
            en   = pd.to_datetime(meta.get("end",   pd.NaT))
            val  = meta.get("value_gw", np.nan)

            st_s  = st.strftime("%Y-%m-%d") if pd.notna(st) else "?"
            en_s  = en.strftime("%Y-%m-%d") if pd.notna(en) else "?"
            try:    val_s = f"{float(val):.1f} GW"
            except: val_s = "?"

            for j, (col, norm) in enumerate(col_trip):
                ax = fig.add_subplot(gs[row, 3 * side + j], projection=proj)
                _setup_grid_ax(ax)
                _draw_country_panel(ax, gg, col, norm, cmap_obj)

                # Spaltenüberschriften nur in erster Zeile
                if row == 0:
                    ax.set_title(head_titles[j], fontsize=18)

                # Zweizeiliges Label oben-links im linken Panel
                # clip_on=False: Text wird nicht am Achsenrand abgeschnitten
                if j == 0:
                    ax.text(
                        0.01, 0.98,
                        f"Reg {reg}  |  {val_s}\n{st_s} → {en_s}",
                        transform=ax.transAxes,
                        ha="left", va="top", fontsize=18,
                        clip_on=False,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey",
                                  lw=0.5, alpha=0.90),
                    )

        _add_colorbars(fig, norm_pv, norm_on, norm_off, cmap_obj)
        t = title + (f" (Seite {page_i}/{len(pages)})" if len(pages) > 1 else "")
        fig.suptitle(t, fontsize=18, y=0.975)

        out = out_png
        if len(pages) > 1:
            root, ext = os.path.splitext(out_png)
            out = f"{root}__page{page_i:02d}{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Regime grid gespeichert: {out}")


# =============================================================================
# plot_event_grid_5x2_triplets
# Gleiche Fixes wie plot_regime_grid_5x2
# =============================================================================
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
    proj     = ccrs.LambertConformal(central_longitude=10, central_latitude=50)
    cmap_obj = plt.get_cmap(cmap)
    norm_pv  = TwoSlopeNorm(vmin=-vmax_pv,  vcenter=0.0, vmax=vmax_pv)
    norm_on  = TwoSlopeNorm(vmin=-vmax_on,  vcenter=0.0, vmax=vmax_on)
    norm_off = TwoSlopeNorm(vmin=-vmax_off, vcenter=0.0, vmax=vmax_off)
    head_titles = ["PV anom [GW]", "Onshore anom [GW]", "Offshore anom [GW]"]

    N_ROWS = 5
    fig, gs = _make_grid_figure(N_ROWS)

    for i in range(10):
        # Row-first layout: slot 0→(row=0,left), slot 1→(row=0,right), …
        row  = i // 2
        side = i % 2

        if i >= len(panels):
            for j in range(3):
                ax = fig.add_subplot(gs[row, 3 * side + j])
                ax.axis("off")
            continue

        gg    = panels[i]["gdf"]
        label = str(panels[i].get("label", ""))

        # Label in zwei Zeilen aufteilen: "#01  |  123.4 GW\n2031-12-06 → 2031-12-13"
        # Das Label kommt bereits als String – wir teilen am " | " nach dem Rank
        label_lines = label
        parts = label.split(" | ", 2)   # ["#01", "2031-12-06->2031-12-13", "123.4 GW"]
        if len(parts) == 3:
            label_lines = f"{parts[0]}  |  {parts[2]}\n{parts[1].replace('->', ' → ')}"
        elif len(parts) == 2:
            label_lines = f"{parts[0]}\n{parts[1].replace('->', ' → ')}"

        for j, (col, norm) in enumerate(
            [("pv_anom_gw", norm_pv), ("on_anom_gw", norm_on), ("off_anom_gw", norm_off)]
        ):
            ax = fig.add_subplot(gs[row, 3 * side + j], projection=proj)
            _setup_grid_ax(ax)
            _draw_country_panel(ax, gg, col, norm, cmap_obj)

            if row == 0:
                ax.set_title(head_titles[j], fontsize=18)

            if j == 0:
                ax.text(
                    0.01, 0.98,
                    label_lines,
                    transform=ax.transAxes,
                    ha="left", va="top", fontsize=18,
                    clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey",
                              lw=0.5, alpha=0.90),
                )

    _add_colorbars(fig, norm_pv, norm_on, norm_off, cmap_obj)
    t = title + (f" ({page_label})" if page_label else "")
    fig.suptitle(t, fontsize=18, y=0.975)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Event grid gespeichert: {out_png}")


# =============================================================================
# MAIN
# =============================================================================
def _require_file(path: str, desc: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} fehlt: {path}")

def main() -> None:
    if gpd is None:
        raise ImportError("geopandas ist nicht verfuegbar.")
    _require_file(IN_HEAVIEST_CSV,         "Heaviest-events CSV")
    _require_file(IN_TOP10_PER_REGIME_CSV, "Top10-per-regime CSV")
    heaviest          = pd.read_csv(IN_HEAVIEST_CSV,         parse_dates=["start", "end"])
    top10_per_regime  = pd.read_csv(IN_TOP10_PER_REGIME_CSV, parse_dates=["start", "end"])
    if heaviest.empty:         raise RuntimeError("Heaviest-events CSV ist leer.")
    if top10_per_regime.empty: raise RuntimeError("Top10-per-regime CSV ist leer.")
    world    = load_country_shapes()
    base_gdf = build_base_gdf(world, COUNTRIES_TO_PLOT)
    print("--- Indexiere generation CSVs ---")
    gen_files = index_generation_csvs(PORTFOLIO_BASE)
    if not gen_files:
        raise RuntimeError(f"Keine generation CSV candidates unter {PORTFOLIO_BASE}.")
    print(f"[OK] Generation CSV candidates: {len(gen_files)}")
    print("--- Lade generation Zeitreihen pro Land ---")
    gen_ts: Dict[str, pd.DataFrame] = {}
    for cc in COUNTRIES_TO_PLOT:
        try:
            df = load_country_generation_series(cc, gen_files=gen_files)
            gen_ts[cc] = df
            print(f" - {cc}: {df.shape} cols={list(df.columns)}")
        except Exception as e:
            print(f" [WARN] {cc}: konnte nicht geladen werden: {e}")
    if not gen_ts:
        raise RuntimeError("Keine Generation-Zeitreihen geladen.")
    print("\n--- Diagnose: Coverage + NaNs pro Land ---")
    for cc, df in gen_ts.items():
        y0i, y1i   = int(df.index.min().year), int(df.index.max().year)
        nan_share  = df.isna().mean().to_dict()
        print(f"{cc}: years={y0i}-{y1i} | NaN-share={nan_share}")

    def compute_event_map(start, end):
        anom_pv, anom_on, anom_off = {}, {}, {}
        for cc in COUNTRIES_TO_PLOT:
            if cc not in gen_ts:
                anom_pv[cc] = anom_on[cc] = anom_off[cc] = np.nan
                continue
            a = compute_country_event_anomalies(
                gen_ts[cc], start, end, clim_mode=CLIM_MODE, clim_years=CLIM_YEARS
            )
            anom_pv[cc] = a["pv"]; anom_on[cc] = a["on"]; anom_off[cc] = a["off"]
        return anom_pv, anom_on, anom_off

    print("\n--- Berechne Anomalien: heaviest event je Regime ---")
    pv_vals, on_vals, off_vals = [], [], []
    per_regime_gdf:  Dict[str, "gpd.GeoDataFrame"]  = {}
    per_regime_meta: Dict[str, Dict[str, object]]    = {}
    for rec in heaviest.to_dict("records"):
        reg   = str(rec["regime"])
        start = pd.to_datetime(rec["start"])
        end   = pd.to_datetime(rec["end"])
        val_gw = float(rec.get("value_gw", np.nan))
        pv_map, on_map, off_map = compute_event_map(start, end)
        pv_vals.extend([v for v in pv_map.values()  if np.isfinite(v)])
        on_vals.extend([v for v in on_map.values()  if np.isfinite(v)])
        off_vals.extend([v for v in off_map.values() if np.isfinite(v)])
        gg = base_gdf.copy()
        gg["pv_anom_gw"]  = gg["cc2"].map(pv_map)
        gg["on_anom_gw"]  = gg["cc2"].map(on_map)
        gg["off_anom_gw"] = gg["cc2"].map(off_map)
        per_regime_gdf[reg]  = gg
        per_regime_meta[reg] = {"start": start, "end": end, "value_gw": val_gw}

    vmax_pv  = float(ANOM_VMAX_PV)  if ANOM_VMAX_PV  is not None else _robust_vmax(pv_vals)
    vmax_on  = float(ANOM_VMAX_ON)  if ANOM_VMAX_ON  is not None else _robust_vmax(on_vals)
    vmax_off = float(ANOM_VMAX_OFF) if ANOM_VMAX_OFF is not None else _robust_vmax(off_vals)
    print(f"[OK] vmax PV/ON/OFF = {vmax_pv:.3f} / {vmax_on:.3f} / {vmax_off:.3f} GW")

    # A) Einzelkarten
    print("\n--- Plot: pro Regime (heaviest) ---")
    for reg, gg in per_regime_gdf.items():
        meta  = per_regime_meta[reg]
        start = pd.to_datetime(meta["start"])
        end   = pd.to_datetime(meta["end"])
        val_gw = float(meta.get("value_gw", np.nan))
        safe_reg = _norm_str(reg)
        stamp    = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}__{EVENT_WINDOW_DAYS}d"
        title_str = (
            f"Regime {reg} | Generation anomalies vs Monatsklima ({CLIM_MODE})\n"
            f"{start.strftime('%Y-%m-%d %H:%M')} -> {end.strftime('%Y-%m-%d %H:%M')} | Residual mean: {val_gw:.2f} GW"
        )
        outpng = os.path.join(GEN_MAPS_DIR, f"generation_anom__heaviest__regime_{safe_reg}__{stamp}.png")
        plot_three_panel_generation_anom(
            gdf=gg, col_pv="pv_anom_gw", col_on="on_anom_gw", col_off="off_anom_gw",
            title=title_str, outpath=outpng,
            vmax_pv=vmax_pv, vmax_on=vmax_on, vmax_off=vmax_off,
            cmap=ANOM_CMAP, dpi=260,
        )

    # B) Overview grid
    print("\n--- Plot: Overview grid (heaviest per regime) ---")
    out_overview = os.path.join(
        GRID_HEAVIEST_DIR,
        f"overview__heaviest_per_regime__{EVENT_WINDOW_DAYS}d__{CLIM_MODE}"
        f"__vmax_pv{vmax_pv:.2f}_on{vmax_on:.2f}_off{vmax_off:.2f}.png"
    )
    plot_regime_grid_5x2(
        per_regime_gdfs=per_regime_gdf,
        per_regime_meta=per_regime_meta,
        out_png=out_overview,
        vmax_pv=vmax_pv, vmax_on=vmax_on, vmax_off=vmax_off,
        title=f"",
        cmap=ANOM_CMAP, dpi=GRID_DPI,
    )

    # D) Top-10 pro Regime
    print("\n--- Plot: Top-10 pro Regime ---")
    for reg, g in top10_per_regime.groupby("regime"):
        gg_sorted = g.sort_values("rank") if "rank" in g.columns else g.sort_values(["end"])
        rows_reg: List[Dict[str, object]] = []
        for _, rec in gg_sorted.head(10).iterrows():
            start  = pd.to_datetime(rec["start"])
            end    = pd.to_datetime(rec["end"])
            val_gw = float(rec.get("value_gw", np.nan))
            rk     = int(rec["rank"]) if "rank" in gg_sorted.columns and pd.notna(rec["rank"]) else None
            pv_map, on_map, off_map = compute_event_map(start, end)
            gg_evt = base_gdf.copy()
            gg_evt["pv_anom_gw"]  = gg_evt["cc2"].map(pv_map)
            gg_evt["on_anom_gw"]  = gg_evt["cc2"].map(on_map)
            gg_evt["off_anom_gw"] = gg_evt["cc2"].map(off_map)
            prefix = f"#{rk}" if rk is not None else "#?"
            # Label wird in plot_event_grid_5x2_triplets zweizeilig aufgeteilt
            label  = f"{prefix} | {start.strftime('%Y-%m-%d')}->{end.strftime('%Y-%m-%d')} | {val_gw:.1f} GW"
            rows_reg.append({"gdf": gg_evt, "label": label})

        if not rows_reg:
            print(f"[WARN] Regime {reg}: keine rows."); continue

        pages    = [rows_reg[i:i+10] for i in range(0, len(rows_reg), 10)]
        safe_reg = _norm_str(reg)
        for page_i, page_panels in enumerate(pages, start=1):
            page_suffix = "" if len(pages) == 1 else f"__page{page_i:02d}"
            out_reg = os.path.join(
                GRID_TOP10_PER_REGIME_DIR,
                f"top10__regime_{safe_reg}__{EVENT_WINDOW_DAYS}d__{CLIM_MODE}{page_suffix}.png"
            )
            plot_event_grid_5x2_triplets(
                panels=page_panels, out_png=out_reg,
                title=f"",
                vmax_pv=vmax_pv, vmax_on=vmax_on, vmax_off=vmax_off,
                cmap=ANOM_CMAP, dpi=GRID_DPI,
                page_label=f"Seite {page_i}/{len(pages)}" if len(pages) > 1 else None,
            )

    print("\nFertig.")

if __name__ == "__main__":
    main()