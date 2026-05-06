#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# GLOBAL XARRAY OPTIONS
# =============================================================================
xr.set_options(use_new_combine_kwarg_defaults=True)


# =============================================================================
# KONFIGURATION (ANPASSEN)
# =============================================================================

# --- Input: Extremwert-Tabelle aus deinem Script ---
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
EVENT_WINDOW_DAYS = 7         # M
N_EVENTS = 10                 # N (wird ggf. gekappt, wenn weniger Einträge)

# --- Interpretation der Startdaten aus der CSV ---
START_HOUR = 0

# --- CORDEX Rohdaten ---
CORDEX_BASE = "/mnt/endata/Cordex/Copernicus_CORDEX_data"
CORDEX_YEAR_DIR_PATTERN = "cordex_{year}"

TAS_VAR_CANDIDATES = ["tas", "t2m", "temperature"]
WIND_VAR_CANDIDATES = ["sfcWind", "wind", "windspeed"]
U_VAR_CANDIDATES = ["uas", "u10", "u10m", "u"]
V_VAR_CANDIDATES = ["vas", "v10", "v10m", "v"]
RSDS_VAR_CANDIDATES = ["rsds", "Rsds", "SWDOWN", "surface_downwelling_shortwave_flux_in_air"]

# --- Europa-Extent ---
EUROPE_EXTENT = (-12, 35, 35, 72)

# --- Anomalien (Monatsklimatologie) ---
CLIM_MODE = "multi_year_month"
CLIM_YEARS = list(range(2025, 2051))
TAS_ANOM_VMAX = 10.0

WIND_ANOM_VMAX: Optional[float] = None
RSDS_ANOM_VMAX: Optional[float] = None

# --- Output ---
OUT_ROOT = os.path.join(MA_BASE, "final_results_portfolio_uncorrected", "top_dunkelflaute_maps__from_extrema_table")
OUT_RUN = os.path.join(OUT_ROOT, f"window{EVENT_WINDOW_DAYS}d__top{N_EVENTS}__{CLIM_MODE}")
MAPS_DIR = os.path.join(OUT_RUN, "maps_single")
OVERVIEW_DIR = os.path.join(OUT_RUN, "overview")
CLIM_CACHE_DIR = os.path.join(OUT_RUN, "climatology_cache")
TABLES_DIR = os.path.join(OUT_RUN, "tables")

for d in [MAPS_DIR, OVERVIEW_DIR, CLIM_CACHE_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[OK] EXTREMA_CSV: {EXTREMA_CSV}")
print(f"[OK] OUT_RUN:    {OUT_RUN}")


# =============================================================================
# Helper: Extremwert-Tabelle lesen und Events extrahieren
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
# Helper: CORDEX loading
# =============================================================================

def _glob_year_files(year_dir: Path, keyword: str) -> List[Path]:
    return sorted([p for p in year_dir.rglob("*.nc") if keyword.lower() in p.name.lower()])


def _open_mfd(files: List[Path]) -> xr.Dataset:
    if not files:
        raise FileNotFoundError("Keine passenden NetCDF-Dateien gefunden.")
    return xr.open_mfdataset(
        [str(f) for f in files],
        combine="by_coords",
        parallel=True,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )


def _combine_by_coords(dsets: List[xr.Dataset]) -> xr.Dataset:
    if not dsets:
        raise ValueError("combine: empty dsets")
    if len(dsets) == 1:
        return dsets[0]
    return xr.combine_by_coords(
        dsets,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )


def _find_data_var(ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
    vars_lower = {v.lower(): v for v in ds.data_vars}
    for c in candidates:
        if c.lower() in vars_lower:
            return vars_lower[c.lower()]
    return None


def _infer_rotated_pole_crs(ds: xr.Dataset) -> Tuple[ccrs.CRS, ccrs.CRS]:
    data_crs: ccrs.CRS = ccrs.PlateCarree()

    rp = None
    if "rotated_pole" in ds.variables:
        rp = ds["rotated_pole"]
    else:
        for v in ds.data_vars:
            gm = ds[v].attrs.get("grid_mapping", None)
            if gm and gm in ds.variables:
                cand = ds[gm]
                if "grid_north_pole_longitude" in cand.attrs or "pole_longitude" in cand.attrs:
                    rp = cand
                    break

    if rp is not None:
        pole_lon = rp.attrs.get("grid_north_pole_longitude", rp.attrs.get("pole_longitude", None))
        pole_lat = rp.attrs.get("grid_north_pole_latitude", rp.attrs.get("pole_latitude", None))
        if pole_lon is not None and pole_lat is not None:
            data_crs = ccrs.RotatedPole(pole_longitude=float(pole_lon), pole_latitude=float(pole_lat))

    plot_crs = ccrs.LambertConformal(central_longitude=10, central_latitude=50)
    return data_crs, plot_crs


def _ensure_tas_celsius(tas: xr.DataArray) -> xr.DataArray:
    units = str(tas.attrs.get("units", "")).lower()
    if units in {"k", "kelvin"} or "kelvin" in units:
        out = tas - 273.15
        out.attrs["units"] = "°C"
        return out
    try:
        if float(tas.quantile(0.5)) > 150:
            out = tas - 273.15
            out.attrs["units"] = "°C"
            return out
    except Exception:
        pass
    return tas


def load_event_fields_tas_wind_rsds(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, ccrs.CRS, ccrs.CRS]:
    years = sorted(set([start.year, end.year] + pd.date_range(start.normalize(), end.normalize(), freq="D").year.tolist()))

    tas_dsets, wind_dsets, u_dsets, v_dsets, rsds_dsets = [], [], [], [], []

    for y in years:
        year_dir = Path(CORDEX_BASE) / CORDEX_YEAR_DIR_PATTERN.format(year=y)
        if not year_dir.exists():
            raise FileNotFoundError(f"CORDEX Jahresordner fehlt: {year_dir}")

        tas_files: List[Path] = []
        for key in TAS_VAR_CANDIDATES:
            tas_files = _glob_year_files(year_dir, key)
            if tas_files:
                break
        if not tas_files:
            raise FileNotFoundError(f"Keine tas-Dateien gefunden in {year_dir}")
        tas_dsets.append(_open_mfd(tas_files))

        rsds_files: List[Path] = []
        for key in RSDS_VAR_CANDIDATES:
            rsds_files = _glob_year_files(year_dir, key)
            if rsds_files:
                break
        if not rsds_files:
            raise FileNotFoundError(f"Keine rsds-Dateien gefunden in {year_dir}")
        rsds_dsets.append(_open_mfd(rsds_files))

        wind_files: List[Path] = []
        for key in WIND_VAR_CANDIDATES:
            wind_files = _glob_year_files(year_dir, key)
            if wind_files:
                break
        if wind_files:
            wind_dsets.append(_open_mfd(wind_files))
        else:
            u_files: List[Path] = []
            v_files: List[Path] = []
            for key in U_VAR_CANDIDATES:
                u_files = _glob_year_files(year_dir, key)
                if u_files:
                    break
            for key in V_VAR_CANDIDATES:
                v_files = _glob_year_files(year_dir, key)
                if v_files:
                    break
            if not u_files or not v_files:
                raise FileNotFoundError(f"Weder sfcWind noch uas/vas gefunden in {year_dir}.")
            u_dsets.append(_open_mfd(u_files))
            v_dsets.append(_open_mfd(v_files))

    ds_tas = _combine_by_coords(tas_dsets)
    ds_rsds = _combine_by_coords(rsds_dsets)
    ds_wind = _combine_by_coords(wind_dsets) if wind_dsets else None
    ds_u = _combine_by_coords(u_dsets) if u_dsets else None
    ds_v = _combine_by_coords(v_dsets) if v_dsets else None

    tas_var = _find_data_var(ds_tas, TAS_VAR_CANDIDATES) or list(ds_tas.data_vars)[0]
    tas = ds_tas[tas_var].sel(time=slice(start, end))

    rsds_var = _find_data_var(ds_rsds, RSDS_VAR_CANDIDATES) or list(ds_rsds.data_vars)[0]
    rsds = ds_rsds[rsds_var].sel(time=slice(start, end))

    if ds_wind is not None:
        wind_var = _find_data_var(ds_wind, WIND_VAR_CANDIDATES) or list(ds_wind.data_vars)[0]
        wind = ds_wind[wind_var].sel(time=slice(start, end))
    else:
        u_var = _find_data_var(ds_u, U_VAR_CANDIDATES) or list(ds_u.data_vars)[0]
        v_var = _find_data_var(ds_v, V_VAR_CANDIDATES) or list(ds_v.data_vars)[0]
        u = ds_u[u_var].sel(time=slice(start, end))
        v = ds_v[v_var].sel(time=slice(start, end))
        wind = np.sqrt(u ** 2 + v ** 2)
        wind.attrs["units"] = ds_u[u_var].attrs.get("units", "m s-1")
        wind.attrs["long_name"] = "Wind speed (computed from u/v)"

    tas_mean = _ensure_tas_celsius(tas.mean(dim="time", skipna=True))
    wind_mean = wind.mean(dim="time", skipna=True)
    rsds_mean = rsds.mean(dim="time", skipna=True)

    ds_for_crs = tas_mean.to_dataset(name="tas_mean")
    data_crs, plot_crs = _infer_rotated_pole_crs(ds_for_crs)

    return tas_mean, wind_mean, rsds_mean, data_crs, plot_crs


# =============================================================================
# Climatology + anomalies (cached)
# =============================================================================

_CLIM_MEM_CACHE: Dict[str, xr.DataArray] = {}


def _clim_cache_key(varname: str, month: int, mode: str) -> str:
    return f"{mode}__{varname}__m{month:02d}"


def _cache_path(varname: str, month: int, mode: str) -> str:
    safe = _clim_cache_key(varname, month, mode)
    return os.path.join(CLIM_CACHE_DIR, f"clim_{safe}.nc")


def _open_var_over_years(var_candidates: List[str], years: List[int]) -> Tuple[xr.Dataset, str]:
    dsets = []
    chosen_var: Optional[str] = None

    for y in years:
        year_dir = Path(CORDEX_BASE) / CORDEX_YEAR_DIR_PATTERN.format(year=y)
        if not year_dir.exists():
            continue

        files: List[Path] = []
        for key in var_candidates:
            files = _glob_year_files(year_dir, key)
            if files:
                break
        if not files:
            continue

        ds = _open_mfd(files)
        if chosen_var is None:
            chosen_var = _find_data_var(ds, var_candidates) or list(ds.data_vars)[0]

        if chosen_var in ds.data_vars:
            ds = ds[[chosen_var]]
        else:
            ds = ds[[list(ds.data_vars)[0]]]
            chosen_var = list(ds.data_vars)[0]

        dsets.append(ds)

    if not dsets or chosen_var is None:
        raise FileNotFoundError(f"Climatology: keine Dateien gefunden für {var_candidates} in years={years[:3]}...")

    ds_all = _combine_by_coords(dsets)
    return ds_all, chosen_var


def _open_wind_over_years(years: List[int]) -> xr.DataArray:
    wind_dsets: List[xr.Dataset] = []
    u_dsets: List[xr.Dataset] = []
    v_dsets: List[xr.Dataset] = []

    for y in years:
        year_dir = Path(CORDEX_BASE) / CORDEX_YEAR_DIR_PATTERN.format(year=y)
        if not year_dir.exists():
            continue

        wind_files: List[Path] = []
        for key in WIND_VAR_CANDIDATES:
            wind_files = _glob_year_files(year_dir, key)
            if wind_files:
                break
        if wind_files:
            wind_dsets.append(_open_mfd(wind_files))
            continue

        u_files: List[Path] = []
        v_files: List[Path] = []
        for key in U_VAR_CANDIDATES:
            u_files = _glob_year_files(year_dir, key)
            if u_files:
                break
        for key in V_VAR_CANDIDATES:
            v_files = _glob_year_files(year_dir, key)
            if v_files:
                break
        if u_files and v_files:
            u_dsets.append(_open_mfd(u_files))
            v_dsets.append(_open_mfd(v_files))

    if wind_dsets:
        ds_wind = _combine_by_coords(wind_dsets)
        wind_var = _find_data_var(ds_wind, WIND_VAR_CANDIDATES) or list(ds_wind.data_vars)[0]
        return ds_wind[wind_var]

    if u_dsets and v_dsets:
        ds_u = _combine_by_coords(u_dsets)
        ds_v = _combine_by_coords(v_dsets)
        u_var = _find_data_var(ds_u, U_VAR_CANDIDATES) or list(ds_u.data_vars)[0]
        v_var = _find_data_var(ds_v, V_VAR_CANDIDATES) or list(ds_v.data_vars)[0]
        u = ds_u[u_var]
        v = ds_v[v_var]
        wind = np.sqrt(u ** 2 + v ** 2)
        wind.attrs["units"] = ds_u[u_var].attrs.get("units", "m s-1")
        wind.attrs["long_name"] = "Wind speed (computed from u/v)"
        return wind

    raise FileNotFoundError(
        f"Climatology: Weder sfcWind noch uas/vas gefunden in years={years[:3]}..."
    )


def load_monthly_climatology(month: int, var_kind: str, event_year: int) -> xr.DataArray:
    if var_kind not in {"tas", "wind", "rsds"}:
        raise ValueError("var_kind muss 'tas' oder 'wind' oder 'rsds' sein")

    key = _clim_cache_key(var_kind, month, CLIM_MODE)
    if key in _CLIM_MEM_CACHE:
        return _CLIM_MEM_CACHE[key]

    fp = _cache_path(var_kind, month, CLIM_MODE)
    if os.path.exists(fp):
        da = xr.open_dataset(fp)["clim"]
        _CLIM_MEM_CACHE[key] = da
        return da

    years = [event_year] if CLIM_MODE == "same_year_month" else CLIM_YEARS

    if var_kind == "tas":
        ds_all, v = _open_var_over_years(TAS_VAR_CANDIDATES, years)
        da = _ensure_tas_celsius(ds_all[v])
    elif var_kind == "rsds":
        ds_all, v = _open_var_over_years(RSDS_VAR_CANDIDATES, years)
        da = ds_all[v]
    else:
        da = _open_wind_over_years(years)

    if "time" not in da.dims:
        raise ValueError(f"Climatology variable has no time dim: {da.dims}")

    da_m = da.sel(time=da["time"].dt.month == month).mean(dim="time", skipna=True).rename("clim")
    da_m.to_dataset(name="clim").to_netcdf(fp)
    _CLIM_MEM_CACHE[key] = da_m
    return da_m


def compute_tas_anomaly(tas_mean: xr.DataArray, start: pd.Timestamp) -> xr.DataArray:
    tas_clim = load_monthly_climatology(month=int(start.month), var_kind="tas", event_year=int(start.year))
    tas_anom = (tas_mean - tas_clim).rename("tas_anom")
    tas_anom.attrs["units"] = "°C"
    return tas_anom


def compute_wind_anomaly(wind_mean: xr.DataArray, start: pd.Timestamp) -> xr.DataArray:
    wind_clim = load_monthly_climatology(month=int(start.month), var_kind="wind", event_year=int(start.year))
    wind_anom = (wind_mean - wind_clim).rename("wind_anom")
    wind_anom.attrs["units"] = wind_mean.attrs.get("units", "m s-1")
    return wind_anom


def compute_rsds_anomaly(rsds_mean: xr.DataArray, start: pd.Timestamp) -> xr.DataArray:
    rsds_clim = load_monthly_climatology(month=int(start.month), var_kind="rsds", event_year=int(start.year))
    rsds_anom = (rsds_mean - rsds_clim).rename("rsds_anom")
    rsds_anom.attrs["units"] = rsds_mean.attrs.get("units", "W m-2")
    return rsds_anom


# =============================================================================
# Plotting helpers
# =============================================================================

def _get_xy_coords(da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray, str, str]:
    for xname in ["lon", "longitude", "rlon", "x"]:
        if xname in da.coords:
            x = da.coords[xname].values
            xlab = xname
            break
    else:
        xlab = da.dims[-1]
        x = da[xlab].values

    for yname in ["lat", "latitude", "rlat", "y"]:
        if yname in da.coords:
            y = da.coords[yname].values
            ylab = yname
            break
    else:
        ylab = da.dims[-2]
        y = da[ylab].values

    return x, y, xlab, ylab


def make_anom_norm(vmax: float) -> TwoSlopeNorm:
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _setup_overview_ax(ax):
    ax.set_extent(EUROPE_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.45)
    ax.add_feature(cfeature.BORDERS, linewidth=0.30)
    ax.add_feature(cfeature.LAKES, alpha=0.20)
    ax.add_feature(cfeature.RIVERS, alpha=0.12)
    try:
        if "geo" in ax.spines:
            ax.spines["geo"].set_linewidth(0.55)
            ax.spines["geo"].set_alpha(0.7)
    except Exception:
        pass


def _robust_global_percentile_limits(
    arrays: List[xr.DataArray], p_lo: float = 2.0, p_hi: float = 98.0
) -> Tuple[float, float]:
    vals = []
    for da in arrays:
        a = np.asarray(da.values).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return (0.0, 1.0)
    v = np.concatenate(vals)
    lo = float(np.percentile(v, p_lo))
    hi = float(np.percentile(v, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    return lo, hi


def _add_colorbars(
    fig: plt.Figure,
    mappable_t, mappable_w, mappable_s,
    tas_norm: TwoSlopeNorm,
    wind_norm: TwoSlopeNorm,
    rsds_norm: TwoSlopeNorm,
) -> None:
    """
    Drei horizontale Colorbars in festen Figure-Koordinaten.
    Kein Überlappen, unabhängig von GridSpec-Höhe.

    Layout von oben nach unten:
      Temp  bottom = 0.14
      Wind  bottom = 0.08
      Rsds  bottom = 0.02
    """
    CB_LEFT   = 0.10
    CB_WIDTH  = 0.80
    CB_HEIGHT = 0.022

    cax_t = fig.add_axes([CB_LEFT, 0.14, CB_WIDTH, CB_HEIGHT])
    cax_w = fig.add_axes([CB_LEFT, 0.08, CB_WIDTH, CB_HEIGHT])
    cax_s = fig.add_axes([CB_LEFT, 0.02, CB_WIDTH, CB_HEIGHT])

    cb_t = fig.colorbar(mappable_t, cax=cax_t, orientation="horizontal", extend="both")
    cb_t.set_label("Temperatur-Anomalie (Event − Monatsklima) [°C]", fontsize=20)
    cb_t.set_ticks(np.linspace(tas_norm.vmin, tas_norm.vmax, 9))
    cb_t.ax.tick_params(labelsize=20)

    cb_w = fig.colorbar(mappable_w, cax=cax_w, orientation="horizontal", extend="both")
    cb_w.set_label("Wind-Anomalie (Event − Monatsklima) [m s⁻¹]", fontsize=20)
    cb_w.set_ticks(np.linspace(wind_norm.vmin, wind_norm.vmax, 9))
    cb_w.ax.tick_params(labelsize=20)

    cb_s = fig.colorbar(mappable_s, cax=cax_s, orientation="horizontal", extend="both")
    cb_s.set_label("Rsds-Anomalie (Event − Monatsklima) [W m⁻²]", fontsize=20)
    cb_s.set_ticks(np.linspace(rsds_norm.vmin, rsds_norm.vmax, 9))
    cb_s.ax.tick_params(labelsize=20)


def plot_overview_top_events(
    events: List[Dict[str, object]],
    out_png: str,
    *,
    title: str,
    tas_norm: TwoSlopeNorm,
    wind_norm: TwoSlopeNorm,
    rsds_norm: TwoSlopeNorm,
    tas_cmap: str = "RdBu_r",
    wind_cmap: str = "PuOr",
    rsds_cmap: str = "RdYlBu_r",
    dpi: int = 300,
):
    """
    2 Events pro Zeile, pro Event 3 Karten:
      [Temp-Anomalie] [Wind-Anomalie] [Rsds-Anomalie]
    => 6 Karten pro Zeile

    Colorbars werden via fig.add_axes() in festen Figure-Koordinaten
    platziert – kein Überlappen, egal wie viele Zeilen.
    """
    if not events:
        raise ValueError("events ist leer – nichts zu plotten.")

    plot_crs = events[0]["plot_crs"]
    n = len(events)
    nrows = int(np.ceil(n / 2))

    # Karten-GridSpec: bottom=0.22 lässt Platz für 3 Colorbars
    fig = plt.figure(figsize=(30.0, 5.5 * nrows + 2.0), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=6,
        hspace=0.04,
        wspace=0.01,
        top=0.93,
        bottom=0.22,
        left=0.02,
        right=0.99,
    )

    mappable_t = None
    mappable_w = None
    mappable_s = None

    def _draw_event(ev: Dict, col_offset: int, row: int, is_first_row: bool) -> None:
        nonlocal mappable_t, mappable_w, mappable_s

        tas_da:  xr.DataArray = ev["tas_anom"]
        wind_da: xr.DataArray = ev["wind_anom"]
        rsds_da: xr.DataArray = ev["rsds_anom"]
        dcrs:    ccrs.CRS     = ev["data_crs"]

        tx, ty, _, _ = _get_xy_coords(tas_da)
        wx, wy, _, _ = _get_xy_coords(wind_da)
        sx, sy, _, _ = _get_xy_coords(rsds_da)

        ax_t = fig.add_subplot(gs[row, col_offset + 0], projection=plot_crs)
        ax_w = fig.add_subplot(gs[row, col_offset + 1], projection=plot_crs)
        ax_s = fig.add_subplot(gs[row, col_offset + 2], projection=plot_crs)
        _setup_overview_ax(ax_t)
        _setup_overview_ax(ax_w)
        _setup_overview_ax(ax_s)

        mt = ax_t.pcolormesh(tx, ty, tas_da.values,  transform=dcrs, shading="auto", cmap=tas_cmap,  norm=tas_norm)
        mw = ax_w.pcolormesh(wx, wy, wind_da.values, transform=dcrs, shading="auto", cmap=wind_cmap, norm=wind_norm)
        ms = ax_s.pcolormesh(sx, sy, rsds_da.values, transform=dcrs, shading="auto", cmap=rsds_cmap, norm=rsds_norm)

        if mappable_t is None: mappable_t = mt
        if mappable_w is None: mappable_w = mw
        if mappable_s is None: mappable_s = ms

        # Zweizeiliges Label: Rang + GW oben, Datum unten
        rank  = int(ev["rank"])
        start = pd.to_datetime(ev["start"])
        end   = pd.to_datetime(ev["end"])
        val   = float(ev["value_gw"])
        ax_t.text(
            0.01, 0.98,
            f"#{rank:02d}  |  {val:.2f} GW\n{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}",
            transform=ax_t.transAxes,
            ha="left", va="top", fontsize=20,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
        )

        if is_first_row:
            ax_t.set_title("Temp-Anomalie [°C]",    fontsize=20)
            ax_w.set_title("Wind-Anomalie [m s⁻¹]", fontsize=20)
            ax_s.set_title("Rsds-Anomalie [W m⁻²]", fontsize=20)

    idx = 0
    for r in range(nrows):
        if idx < n:
            _draw_event(events[idx],     col_offset=0, row=r, is_first_row=(r == 0))
        if idx + 1 < n:
            _draw_event(events[idx + 1], col_offset=3, row=r, is_first_row=(r == 0))
        idx += 2

    fig.suptitle(title, fontsize=20, y=0.975)
    _add_colorbars(fig, mappable_t, mappable_w, mappable_s, tas_norm, wind_norm, rsds_norm)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Overview gespeichert: {out_png}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not os.path.exists(EXTREMA_CSV):
        raise FileNotFoundError(f"Extrema-CSV nicht gefunden: {EXTREMA_CSV}")

    print("\n--- Lade Top-Events aus Extremwert-Tabelle ---")
    top_df = load_top_events_from_extrema_csv(
        csv_path=EXTREMA_CSV,
        window_days=EVENT_WINDOW_DAYS,
        n_events=N_EVENTS,
    )
    if top_df.empty:
        raise RuntimeError("Keine Events aus CSV extrahiert (prüfe Spalten / Parsing).")

    print(top_df.to_string(index=False))

    print("\n--- Lade CORDEX Felder & berechne Anomalien (tas, wind, rsds) ---")
    events_payload: List[Dict[str, object]] = []
    all_wind_anom: List[xr.DataArray] = []
    all_rsds_anom: List[xr.DataArray] = []

    for row in top_df.to_dict("records"):
        rank   = int(row["rank"])
        start  = pd.to_datetime(row["start"])
        end    = pd.to_datetime(row["end"])
        val_gw = float(row["value_gw"])

        print(f"  -> Event #{rank:02d}: {start} → {end} | {val_gw:.2f} GW")

        tas_mean, wind_mean, rsds_mean, data_crs, plot_crs = load_event_fields_tas_wind_rsds(start, end)

        tas_anom  = compute_tas_anomaly(tas_mean,   start=start)
        wind_anom = compute_wind_anomaly(wind_mean, start=start)
        rsds_anom = compute_rsds_anomaly(rsds_mean, start=start)

        events_payload.append({
            "rank": rank, "start": start, "end": end, "value_gw": val_gw,
            "tas_anom": tas_anom, "wind_anom": wind_anom, "rsds_anom": rsds_anom,
            "data_crs": data_crs, "plot_crs": plot_crs,
        })

        all_wind_anom.append(wind_anom)
        all_rsds_anom.append(rsds_anom)

    # Wind-Anom norm
    if WIND_ANOM_VMAX is None:
        wlo, whi = _robust_global_percentile_limits(all_wind_anom, p_lo=2.0, p_hi=98.0)
        wind_vmax = max(abs(wlo), abs(whi)) if np.isfinite(wlo) and np.isfinite(whi) else 1.0
        wind_vmax = max(wind_vmax, 1e-6)
    else:
        wind_vmax = float(WIND_ANOM_VMAX)
        if not np.isfinite(wind_vmax) or wind_vmax <= 0:
            wind_vmax = 1.0

    # Rsds-Anom norm
    if RSDS_ANOM_VMAX is None:
        slo, shi = _robust_global_percentile_limits(all_rsds_anom, p_lo=2.0, p_hi=98.0)
        rsds_vmax = max(abs(slo), abs(shi)) if np.isfinite(slo) and np.isfinite(shi) else 1.0
        rsds_vmax = max(rsds_vmax, 1.0)
    else:
        rsds_vmax = float(RSDS_ANOM_VMAX)
        if not np.isfinite(rsds_vmax) or rsds_vmax <= 0:
            rsds_vmax = 1.0

    out_overview = os.path.join(
        OVERVIEW_DIR,
        f"overview__top{len(events_payload)}__window{EVENT_WINDOW_DAYS}d__{CLIM_MODE}"
        f"__tas{TAS_ANOM_VMAX:g}__windanom{wind_vmax:.3f}__rsdsanom{rsds_vmax:.1f}.png"
    )

    plot_overview_top_events(
        events=events_payload,
        out_png=out_overview,
        title=(
            f""
        ),
        tas_norm=make_anom_norm(vmax=TAS_ANOM_VMAX),
        wind_norm=make_anom_norm(vmax=wind_vmax),
        rsds_norm=make_anom_norm(vmax=rsds_vmax),
        tas_cmap="RdBu_r",
        wind_cmap="PuOr",
        rsds_cmap="RdYlBu_r",
        dpi=300,
    )

    print("\nFertig.")


if __name__ == "__main__":
    main()