import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MaxNLocator

# =============================================================================
# GLOBAL XARRAY OPTIONS
# =============================================================================
xr.set_options(use_new_combine_kwarg_defaults=True)

# =============================================================================
# KONFIGURATION
# =============================================================================
PERSISTENCE_DAYS = 5
K_CLUSTERS = 8
Y0, Y1 = 2025, 2050
REGIME_WINDOW_COVERAGE = 0.95
RUN_TAG = f"p{PERSISTENCE_DAYS}d_k{K_CLUSTERS}_{Y0}_{Y1}"

ANALYSIS_TYPE = "SYNTHETIC_DEMAND"
COUNTRIES_TO_SUM = [
    "DE", "FR", "PL", "CZ", "AT", "CH", "IT", "ES", "BE", "NL",
    "DK", "SE", "PT", "NO", "FI", "LT", "LV", "EE", "GB"
]
VALUE_COLUMN = "residual_load_mw"

MA_BASE = "/mnt/endata/MA_Arthur"
BASE_RESIDUAL_DIR = os.path.join(
    MA_BASE, "final_results_portfolio_uncorrected", "residual_load_analysis",
)

if ANALYSIS_TYPE == "SYNTHETIC_DEMAND":
    INPUT_DIR = BASE_RESIDUAL_DIR
    FILENAME_PATTERN = "residual_load_{country_name}_weather2025-2050_cap2024_demandyearly_dir.csv"
else:
    raise ValueError("Bitte ANALYSIS_TYPE passend setzen (hier nur SYNTHETIC_DEMAND vorgesehen).")

COUNTRY_CODE_TO_FILENAME_MAP = {
    "DE": "Germany", "FR": "France", "GB": "United_Kingdom", "ES": "Spain",
    "IT": "Italy", "PL": "Poland", "NL": "Netherlands", "BE": "Belgium",
    "DK": "Denmark", "CZ": "Czechia", "AT": "Austria", "CH": "Switzerland",
    "EE": "Estonia", "SE": "Sweden", "PT": "Portugal", "NO": "Norway",
    "FI": "Finland", "LT": "Lithuania", "LV": "Latvia",
}

REGIME_DIR = os.path.join(
    "/mnt/endata/Cordex/Copernicus_CORDEX_data",
    f"results_grams_strict_p{PERSISTENCE_DAYS}d_k{K_CLUSTERS}_{Y0}_{Y1}",
)
REGIME_GLOB = "regime_periods_*_3hourly.csv"

CORDEX_BASE = "/mnt/endata/Cordex/Copernicus_CORDEX_data"
CORDEX_YEAR_DIR_PATTERN = "cordex_{year}"

TAS_VAR_CANDIDATES   = ["tas", "t2m", "temperature"]
WIND_VAR_CANDIDATES  = ["sfcWind", "wind", "windspeed"]
U_VAR_CANDIDATES     = ["uas", "u10", "u10m", "u"]
V_VAR_CANDIDATES     = ["vas", "v10", "v10m", "v"]
RSDS_VAR_CANDIDATES  = ["rsds", "Rsds", "SWDOWN", "surface_downwelling_shortwave_flux_in_air"]

EVENT_WINDOW_DAYS    = 7
ROLLING_MIN_COVERAGE = 0.95
EUROPE_EXTENT        = (-12, 35, 35, 72)

CLIM_MODE  = "multi_year_month"
CLIM_YEARS = list(range(2025, 2051))

TAS_ANOM_VMAX = 10.0          # °C
RSDS_ANOM_VMIN: Optional[float] = None
RSDS_ANOM_VMAX: Optional[float] = None
WIND_ABS_VMIN:  Optional[float] = None
WIND_ABS_VMAX:  Optional[float] = None
ABS_TAS_NORM = TwoSlopeNorm(vmin=-30.0, vcenter=0.0, vmax=30.0)

OPEN_PARALLEL = False
OPEN_CHUNKS   = {"time": 168}

OUT_BASE         = os.path.join(MA_BASE, "final_results_portfolio_uncorrected",
                                "regime_dunkelflaute_maps__experiments_v2")
OUT_ROOT         = os.path.join(OUT_BASE, RUN_TAG)
MAPS_ABS_DIR     = os.path.join(OUT_ROOT, "maps_abs")
MAPS_ANOM_DIR    = os.path.join(OUT_ROOT, "maps_anom")
TABLES_DIR       = os.path.join(OUT_ROOT, "tables")
CLIM_CACHE_DIR   = os.path.join(OUT_ROOT, "climatology_cache")
OVERVIEW_DIR     = os.path.join(OUT_ROOT, "overview_grids")
TOP10_REGIME_DIR = os.path.join(OUT_ROOT, "top10_per_regime_grids")

for d in [MAPS_ABS_DIR, MAPS_ANOM_DIR, TABLES_DIR, CLIM_CACHE_DIR, OVERVIEW_DIR, TOP10_REGIME_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[OK] RUN_TAG: {RUN_TAG}")
print(f"[OK] REGIME_DIR: {REGIME_DIR}")
print(f"[OK] OUT_ROOT: {OUT_ROOT}")

# =============================================================================
# Helper: Regime parsing + alignment
# =============================================================================
NO_REGIME_LABEL       = "no_regime"
NO_REGIME_ALIASES_STR = {"no_regime", "none", "nan", "no-regime", "no regime", "noregime"}
NO_REGIME_ALIASES_INT = {-1, 99}


def _parse_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def _detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def read_regimes_timeseries(regime_dir: str) -> pd.DataFrame:
    p = Path(regime_dir)
    files = sorted(list(p.rglob(REGIME_GLOB)))
    if not files:
        raise FileNotFoundError(f"Keine Regime-Dateien gefunden via rglob('{REGIME_GLOB}') in: {regime_dir}")
    all_ts = []
    for f in files:
        df = pd.read_csv(f)
        time_col  = _detect_column(df, ["time", "datetime", "date", "timestamp"])
        reg_col   = _detect_column(df, ["regime", "cluster", "label", "wr", "weather_regime", "kmeans_label"])
        start_col = _detect_column(df, ["start", "start_time", "start_datetime", "begin"])
        end_col   = _detect_column(df, ["end", "end_time", "end_datetime", "stop", "finish"])
        if time_col is not None and reg_col is not None and (start_col is None or end_col is None):
            ts = pd.DataFrame({"time": _parse_datetime_series(df[time_col]), "regime": df[reg_col]})
            ts = ts.dropna(subset=["time"])
            all_ts.append(ts)
            continue
        if start_col is not None and end_col is not None and reg_col is not None:
            starts = _parse_datetime_series(df[start_col])
            ends   = _parse_datetime_series(df[end_col])
            regs   = df[reg_col]
            parts  = []
            for s, e, r in zip(starts, ends, regs):
                if pd.isna(s) or pd.isna(e) or e < s:
                    continue
                idx = pd.date_range(s, e, freq="3h")
                if len(idx) == 0:
                    continue
                parts.append(pd.DataFrame({"time": idx, "regime": r}))
            if parts:
                all_ts.append(pd.concat(parts, ignore_index=True))
            continue
        raise ValueError(f"Unbekanntes Regime-Dateiformat: {f}")
    ts = pd.concat(all_ts, ignore_index=True).dropna(subset=["time"]).sort_values("time")
    ts = ts.drop_duplicates(subset=["time"], keep="last")
    return ts


def normalize_regime_labels(regime_s: pd.Series) -> pd.Series:
    def _parse(v):
        if pd.isna(v):
            return NO_REGIME_LABEL
        if isinstance(v, str):
            t = v.strip().lower()
            if t == "" or t in NO_REGIME_ALIASES_STR:
                return NO_REGIME_LABEL
            try:
                v = int(t)
            except Exception:
                raise ValueError(f"Unbekanntes Regime-Label (string): {v}")
        if isinstance(v, (int, np.integer)):
            iv = int(v)
            if iv == int(K_CLUSTERS):         return NO_REGIME_LABEL
            if 0 <= iv < int(K_CLUSTERS):     return str(iv)
            if iv in NO_REGIME_ALIASES_INT:   return NO_REGIME_LABEL
            raise ValueError(f"Regime-ID {v} außerhalb erwartetem Bereich [0..{K_CLUSTERS}]")
        if isinstance(v, float) and float(v).is_integer():
            iv = int(v)
            if iv == int(K_CLUSTERS):         return NO_REGIME_LABEL
            if 0 <= iv < int(K_CLUSTERS):     return str(iv)
            if iv in NO_REGIME_ALIASES_INT:   return NO_REGIME_LABEL
            raise ValueError(f"Regime-ID {v} außerhalb erwartetem Bereich [0..{K_CLUSTERS}]")
        raise ValueError(f"Unbekannter Regime-Typ: {type(v)} / {v}")
    return regime_s.map(_parse).astype("object")


def align_regimes_to_hourly(regime_ts: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.Series:
    s = regime_ts.set_index("time")["regime"].sort_index()
    s = s.loc[(s.index >= target_index.min()) & (s.index <= target_index.max())]
    aligned = s.reindex(target_index, method="ffill").fillna(NO_REGIME_LABEL)
    return aligned


# =============================================================================
# Helper: Residual aggregation
# =============================================================================
def load_aggregated_residual_series() -> pd.DataFrame:
    all_country_dfs = []
    for cc in COUNTRIES_TO_SUM:
        if cc not in COUNTRY_CODE_TO_FILENAME_MAP:
            print(f"  WARN: Kein Mapping für {cc} – skip")
            continue
        cname = COUNTRY_CODE_TO_FILENAME_MAP[cc]
        fp = os.path.join(INPUT_DIR, FILENAME_PATTERN.format(country_name=cname))
        if not os.path.exists(fp):
            print(f"  FEHLER: Datei fehlt: {fp}")
            continue
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if VALUE_COLUMN not in df.columns:
            print(f"  FEHLER: {cname} hat keine Spalte '{VALUE_COLUMN}' – vorhanden: {list(df.columns)}")
            continue
        df = df[[VALUE_COLUMN]].copy()
        df.columns = [cc]
        all_country_dfs.append(df)
        print(f"  - geladen: {cname} {df.shape}")
    if not all_country_dfs:
        raise RuntimeError("Keine Residual-Load Dateien geladen.")
    combined = pd.concat(all_country_dfs, axis=1)
    agg = pd.DataFrame({"series_mw": combined.sum(axis=1, skipna=True)})
    agg.index = pd.to_datetime(agg.index).tz_localize(None)
    if agg["series_mw"].notna().sum() == 0:
        raise RuntimeError("Aggregierte Serie ist nur NaN – VALUE_COLUMN prüfen.")
    return agg


# =============================================================================
# Helper: Event finden
# =============================================================================
def find_heaviest_event_per_regime(
    series_mw: pd.Series,
    regime_hourly: pd.Series,
    window_days: int = 7,
    min_coverage: float = 0.95,
    *,
    regime_window_coverage: float = 0.95,
    strict_within_regime: bool = False,
    regimes_ordered: Optional[List[str]] = None,
    no_regime_label: str = "no_regime",
) -> pd.DataFrame:
    idx = series_mw.index
    if len(idx) < 2:
        raise ValueError("series_mw hat zu wenige Zeitpunkte.")
    dt = pd.Series(idx).diff().dropna()
    step_seconds = float(dt.dt.total_seconds().median())
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("Konnte Zeitauflösung nicht bestimmen.")
    window_steps     = max(int(round(window_days * 24 * 3600 / step_seconds)), 1)
    min_periods_data = int(np.ceil(window_steps * min_coverage))
    rm = series_mw.rolling(window=window_steps, min_periods=min_periods_data).mean()

    def _regime_frac(reg: str) -> pd.Series:
        mask = (regime_hourly == reg).astype(float)
        return mask.rolling(window=window_steps, min_periods=window_steps).mean()

    if regimes_ordered is None:
        uniq      = [str(u) for u in pd.unique(regime_hourly.dropna())]
        uniq_main = [u for u in uniq if u != no_regime_label]
        def _key(x):
            try:    return (0, int(x))
            except: return (1, x)
        regimes = sorted(uniq_main, key=_key) + ([no_regime_label] if no_regime_label in uniq else [])
    else:
        regimes = [str(r) for r in regimes_ordered]

    rows = []
    for reg in regimes:
        is_end_in_regime = regime_hourly == reg
        frac = _regime_frac(reg)
        ok   = is_end_in_regime & (frac >= (1.0 if strict_within_regime else float(regime_window_coverage)))
        masked = rm.where(ok)
        if masked.notna().sum() == 0:
            print(f"  Regime {reg}: keine gültigen rolling Fenster (Coverage-Filter).")
            continue
        end      = masked.idxmax()
        val      = float(masked.loc[end])
        end_pos  = idx.get_loc(end)
        start_pos = end_pos - (window_steps - 1)
        if start_pos < 0:
            continue
        start = idx[start_pos]
        rows.append({
            "regime": reg, "start": start, "end": end,
            "value_mw": val, "value_gw": val / 1000.0,
            "window_days": window_days, "window_steps": window_steps,
            "regime_frac": float(frac.loc[end]) if end in frac.index and pd.notna(frac.loc[end]) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("value_mw", ascending=False).reset_index(drop=True)


def find_top_events_per_regime(
    series_mw: pd.Series,
    regime_hourly: pd.Series,
    *,
    top_n: int = 10,
    window_days: int = 7,
    min_coverage: float = 0.95,
    regime_window_coverage: float = 0.95,
    strict_within_regime: bool = False,
    regimes_ordered: Optional[List[str]] = None,
    no_regime_label: str = "no_regime",
) -> pd.DataFrame:
    idx = series_mw.index
    if len(idx) < 2:
        raise ValueError("series_mw hat zu wenige Zeitpunkte.")
    dt = pd.Series(idx).diff().dropna()
    step_seconds = float(dt.dt.total_seconds().median())
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("Konnte Zeitauflösung nicht bestimmen.")
    window_steps     = max(int(round(window_days * 24 * 3600 / step_seconds)), 1)
    min_periods_data = int(np.ceil(window_steps * min_coverage))
    rm = series_mw.rolling(window=window_steps, min_periods=min_periods_data).mean()

    def _regime_frac(reg: str) -> pd.Series:
        mask = (regime_hourly == reg).astype(float)
        return mask.rolling(window=window_steps, min_periods=window_steps).mean()

    if regimes_ordered is None:
        uniq      = [str(u) for u in pd.unique(regime_hourly.dropna())]
        uniq_main = [u for u in uniq if u != no_regime_label]
        def _key(x):
            try:    return (0, int(x))
            except: return (1, x)
        regimes = sorted(uniq_main, key=_key) + ([no_regime_label] if no_regime_label in uniq else [])
    else:
        regimes = [str(r) for r in regimes_ordered]

    all_rows = []
    for reg in regimes:
        is_end_in_regime = regime_hourly == reg
        frac   = _regime_frac(reg)
        ok     = is_end_in_regime & (frac >= (1.0 if strict_within_regime else float(regime_window_coverage)))
        masked = rm.where(ok).dropna()
        if masked.empty:
            continue
        cand             = masked.sort_values(ascending=False)
        chosen_intervals = []
        chosen           = []
        for end_time, val in cand.items():
            end_pos   = idx.get_loc(end_time)
            start_pos = end_pos - (window_steps - 1)
            if start_pos < 0:
                continue
            overlaps = any(not (end_pos < s0 or start_pos > e0) for (s0, e0) in chosen_intervals)
            if overlaps:
                continue
            chosen_intervals.append((start_pos, end_pos))
            chosen.append((end_time, float(val),
                           float(frac.loc[end_time]) if end_time in frac.index else np.nan))
            if len(chosen) >= int(top_n):
                break
        for rank, (end_time, val, frac_end) in enumerate(chosen, start=1):
            end_pos   = idx.get_loc(end_time)
            start_pos = end_pos - (window_steps - 1)
            start_time = idx[start_pos]
            all_rows.append({
                "regime": reg, "rank": rank,
                "start": start_time, "end": end_time,
                "value_mw": val, "value_gw": val / 1000.0,
                "window_days": window_days, "window_steps": window_steps,
                "regime_frac": frac_end,
            })
    out = pd.DataFrame(all_rows)
    if out.empty:
        return out
    return out.sort_values(["regime", "rank"]).reset_index(drop=True)


# =============================================================================
# Helper: CORDEX loading
# =============================================================================
def _glob_year_files(year_dir: Path, keyword: str) -> List[Path]:
    return sorted([p for p in year_dir.rglob("*.nc") if keyword.lower() in p.name.lower()])


def _open_mfd(files: List[Path]) -> xr.Dataset:
    if not files:
        raise FileNotFoundError("Keine passenden NetCDF-Dateien gefunden.")
    return xr.open_mfdataset(
        [str(f) for f in files], combine="by_coords",
        parallel=OPEN_PARALLEL, chunks=OPEN_CHUNKS,
        data_vars="minimal", coords="minimal", compat="override",
        combine_attrs="drop_conflicts", decode_times=True, mask_and_scale=True,
    )


def _combine_by_coords(dsets: List[xr.Dataset]) -> xr.Dataset:
    if not dsets:
        raise ValueError("combine: empty dsets")
    if len(dsets) == 1:
        return dsets[0]
    return xr.combine_by_coords(
        dsets, data_vars="minimal", coords="minimal", compat="override",
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
        pole_lat = rp.attrs.get("grid_north_pole_latitude",  rp.attrs.get("pole_latitude",  None))
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


def _safe_close(ds: Optional[xr.Dataset]) -> None:
    try:
        if ds is not None:
            ds.close()
    except Exception:
        pass


def load_event_fields_tas_wind_rsds(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, ccrs.CRS, ccrs.CRS]:
    years = sorted(set(
        [start.year, end.year]
        + pd.date_range(start.normalize(), end.normalize(), freq="D").year.tolist()
    ))
    tas_dsets, wind_dsets, u_dsets, v_dsets, rsds_dsets = [], [], [], [], []
    for y in years:
        year_dir = Path(CORDEX_BASE) / CORDEX_YEAR_DIR_PATTERN.format(year=y)
        if not year_dir.exists():
            raise FileNotFoundError(f"CORDEX Jahresordner fehlt: {year_dir}")
        # tas
        tas_files: List[Path] = []
        for key in TAS_VAR_CANDIDATES:
            tas_files = _glob_year_files(year_dir, key)
            if tas_files: break
        if not tas_files:
            raise FileNotFoundError(f"Keine tas-Dateien gefunden in {year_dir}")
        tas_dsets.append(_open_mfd(tas_files))
        # rsds
        rsds_files: List[Path] = []
        for key in RSDS_VAR_CANDIDATES:
            rsds_files = _glob_year_files(year_dir, key)
            if rsds_files: break
        if not rsds_files:
            raise FileNotFoundError(f"Keine rsds-Dateien gefunden in {year_dir}")
        rsds_dsets.append(_open_mfd(rsds_files))
        # wind
        wind_files: List[Path] = []
        for key in WIND_VAR_CANDIDATES:
            wind_files = _glob_year_files(year_dir, key)
            if wind_files: break
        if wind_files:
            wind_dsets.append(_open_mfd(wind_files))
        else:
            u_files: List[Path] = []
            v_files: List[Path] = []
            for key in U_VAR_CANDIDATES:
                u_files = _glob_year_files(year_dir, key)
                if u_files: break
            for key in V_VAR_CANDIDATES:
                v_files = _glob_year_files(year_dir, key)
                if v_files: break
            if not u_files or not v_files:
                raise FileNotFoundError(f"Weder sfcWind noch uas/vas gefunden in {year_dir}.")
            u_dsets.append(_open_mfd(u_files))
            v_dsets.append(_open_mfd(v_files))

    ds_tas = ds_rsds = ds_wind = ds_u = ds_v = None
    try:
        ds_tas  = _combine_by_coords(tas_dsets)
        ds_rsds = _combine_by_coords(rsds_dsets)
        ds_wind = _combine_by_coords(wind_dsets) if wind_dsets else None
        ds_u    = _combine_by_coords(u_dsets)    if u_dsets    else None
        ds_v    = _combine_by_coords(v_dsets)    if v_dsets    else None

        tas_var  = _find_data_var(ds_tas,  TAS_VAR_CANDIDATES)  or list(ds_tas.data_vars)[0]
        rsds_var = _find_data_var(ds_rsds, RSDS_VAR_CANDIDATES) or list(ds_rsds.data_vars)[0]
        tas  = ds_tas[tas_var].sel(time=slice(start, end))
        rsds = ds_rsds[rsds_var].sel(time=slice(start, end))

        if ds_wind is not None:
            wind_var = _find_data_var(ds_wind, WIND_VAR_CANDIDATES) or list(ds_wind.data_vars)[0]
            wind = ds_wind[wind_var].sel(time=slice(start, end))
        else:
            u_var = _find_data_var(ds_u, U_VAR_CANDIDATES) or list(ds_u.data_vars)[0]
            v_var = _find_data_var(ds_v, V_VAR_CANDIDATES) or list(ds_v.data_vars)[0]
            u = ds_u[u_var].sel(time=slice(start, end))
            v = ds_v[v_var].sel(time=slice(start, end))
            wind = np.sqrt(u**2 + v**2)
            wind.attrs["units"]     = ds_u[u_var].attrs.get("units", "m s-1")
            wind.attrs["long_name"] = "Wind speed (computed from u/v)"

        tas_mean  = _ensure_tas_celsius(tas.mean(dim="time", skipna=True)).compute()
        wind_mean = wind.mean(dim="time", skipna=True).compute()
        rsds_mean = rsds.mean(dim="time", skipna=True).compute()
        rsds_mean.attrs["long_name"] = rsds.attrs.get("long_name", rsds_var)

        ds_for_crs = tas_mean.to_dataset(name="tas_mean")
        data_crs, plot_crs = _infer_rotated_pole_crs(ds_for_crs)
        return tas_mean, wind_mean, rsds_mean, data_crs, plot_crs
    finally:
        _safe_close(ds_tas); _safe_close(ds_rsds); _safe_close(ds_wind)
        _safe_close(ds_u);   _safe_close(ds_v)


# =============================================================================
# Climatology + Anomalies (cached)
# =============================================================================
_CLIM_MEM_CACHE: Dict[str, xr.DataArray] = {}


def _clim_cache_key(varname: str, month: int, mode: str) -> str:
    return f"{mode}__{varname}__m{month:02d}"


def _cache_path(varname: str, month: int, mode: str) -> str:
    return os.path.join(CLIM_CACHE_DIR, f"clim_{_clim_cache_key(varname, month, mode)}.nc")


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
            if files: break
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
        raise FileNotFoundError(f"Climatology: keine Dateien für {var_candidates} in years={years[:3]}...")
    return _combine_by_coords(dsets), chosen_var


def _open_wind_over_years(years: List[int]) -> xr.DataArray:
    wind_dsets, u_dsets, v_dsets = [], [], []
    for y in years:
        year_dir = Path(CORDEX_BASE) / CORDEX_YEAR_DIR_PATTERN.format(year=y)
        if not year_dir.exists():
            continue
        wind_files: List[Path] = []
        for key in WIND_VAR_CANDIDATES:
            wind_files = _glob_year_files(year_dir, key)
            if wind_files: break
        if wind_files:
            wind_dsets.append(_open_mfd(wind_files))
            continue
        u_files: List[Path] = []
        v_files: List[Path] = []
        for key in U_VAR_CANDIDATES:
            u_files = _glob_year_files(year_dir, key)
            if u_files: break
        for key in V_VAR_CANDIDATES:
            v_files = _glob_year_files(year_dir, key)
            if v_files: break
        if u_files and v_files:
            u_dsets.append(_open_mfd(u_files))
            v_dsets.append(_open_mfd(v_files))
    ds_wind = ds_u = ds_v = None
    try:
        if wind_dsets:
            ds_wind  = _combine_by_coords(wind_dsets)
            wind_var = _find_data_var(ds_wind, WIND_VAR_CANDIDATES) or list(ds_wind.data_vars)[0]
            return ds_wind[wind_var]
        if u_dsets and v_dsets:
            ds_u  = _combine_by_coords(u_dsets)
            ds_v  = _combine_by_coords(v_dsets)
            u_var = _find_data_var(ds_u, U_VAR_CANDIDATES) or list(ds_u.data_vars)[0]
            v_var = _find_data_var(ds_v, V_VAR_CANDIDATES) or list(ds_v.data_vars)[0]
            u = ds_u[u_var]; v = ds_v[v_var]
            wind = np.sqrt(u**2 + v**2)
            wind.attrs["units"]     = ds_u[u_var].attrs.get("units", "m s-1")
            wind.attrs["long_name"] = "Wind speed (computed from u/v)"
            return wind
        raise FileNotFoundError(f"Climatology: Weder sfcWind noch uas/vas gefunden in years={years[:3]}...")
    finally:
        _safe_close(ds_wind); _safe_close(ds_u); _safe_close(ds_v)


def load_monthly_climatology(month: int, var_kind: str, event_year: int) -> xr.DataArray:
    if var_kind not in {"tas", "rsds", "wind"}:
        raise ValueError("var_kind muss 'tas' oder 'rsds' oder 'wind' sein.")
    mode = CLIM_MODE
    key  = _clim_cache_key(var_kind, month, mode)
    if key in _CLIM_MEM_CACHE:
        return _CLIM_MEM_CACHE[key]
    fp = _cache_path(var_kind, month, mode)
    if os.path.exists(fp):
        da = xr.open_dataset(fp)["clim"]
        _CLIM_MEM_CACHE[key] = da
        return da
    years = [event_year] if mode == "same_year_month" else CLIM_YEARS if mode == "multi_year_month" else None
    if years is None:
        raise ValueError(f"Unbekannter CLIM_MODE={mode}")
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
    da_m = da.sel(time=da["time"].dt.month == month).mean(dim="time", skipna=True).compute()
    da_m = da_m.rename("clim")
    da_m.attrs["clim_mode"]  = mode
    da_m.attrs["clim_month"] = int(month)
    da_m.to_dataset(name="clim").to_netcdf(fp)
    _CLIM_MEM_CACHE[key] = da_m
    return da_m


def compute_tas_anomaly(tas_mean: xr.DataArray, start: pd.Timestamp) -> xr.DataArray:
    tas_clim = load_monthly_climatology(month=int(start.month), var_kind="tas", event_year=int(start.year))
    tas_anom = (tas_mean - tas_clim).rename("tas_anom")
    tas_anom.attrs["units"]     = "°C"
    tas_anom.attrs["long_name"] = "T2m anomaly (event mean - monthly climatology)"
    return tas_anom


def compute_rsds_anomaly(rsds_mean: xr.DataArray, start: pd.Timestamp) -> xr.DataArray:
    rsds_clim = load_monthly_climatology(month=int(start.month), var_kind="rsds", event_year=int(start.year))
    rsds_anom = (rsds_mean - rsds_clim).rename("rsds_anom")
    rsds_anom.attrs["units"]     = rsds_mean.attrs.get("units", "W m-2")
    rsds_anom.attrs["long_name"] = "Rsds anomaly (event mean - monthly climatology)"
    return rsds_anom


def compute_wind_anomaly(wind_mean: xr.DataArray, start: pd.Timestamp) -> xr.DataArray:
    wind_clim = load_monthly_climatology(month=int(start.month), var_kind="wind", event_year=int(start.year))
    wind_anom = (wind_mean - wind_clim).rename("wind_anom")
    wind_anom.attrs["units"]     = wind_mean.attrs.get("units", "m s-1")
    wind_anom.attrs["long_name"] = "Wind speed anomaly (event mean - monthly climatology)"
    return wind_anom


# =============================================================================
# Plotting helpers
# =============================================================================
def make_anom_norm(vmax: float) -> TwoSlopeNorm:
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _get_xy_coords(da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray, str, str]:
    for xname in ["lon", "longitude", "rlon", "x"]:
        if xname in da.coords:
            x, xlab = da.coords[xname].values, xname
            break
    else:
        xlab = da.dims[-1]; x = da[xlab].values
    for yname in ["lat", "latitude", "rlat", "y"]:
        if yname in da.coords:
            y, ylab = da.coords[yname].values, yname
            break
    else:
        ylab = da.dims[-2]; y = da[ylab].values
    return x, y, xlab, ylab


def plot_three_panel_map(
    da1: xr.DataArray, da2: xr.DataArray, da3: xr.DataArray,
    data_crs: ccrs.CRS, plot_crs: ccrs.CRS,
    title: str, t1: str, t2: str, t3: str, outpath: str,
    *, cmap1=None, norm1=None, cmap2=None, norm2=None, cmap3=None, norm3=None,
    extend1: str = "both", extend2: str = "max", extend3: str = "both",
) -> None:
    x1, y1, _, _ = _get_xy_coords(da1)
    x2, y2, _, _ = _get_xy_coords(da2)
    x3, y3, _, _ = _get_xy_coords(da3)
    fig = plt.figure(figsize=(21, 6), dpi=220)
    gs  = fig.add_gridspec(1, 3, wspace=0.06)
    fig.suptitle(title, fontsize=20, y=0.98)

    def setup_ax(ax):
        ax.set_extent(EUROPE_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4)
        ax.add_feature(cfeature.LAKES,     alpha=0.3)
        ax.add_feature(cfeature.RIVERS,    alpha=0.2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.35)
        gl.top_labels = False; gl.right_labels = False
        return ax

    ax1 = fig.add_subplot(gs[0, 0], projection=plot_crs); setup_ax(ax1)
    m1  = ax1.pcolormesh(x1, y1, da1.values, transform=data_crs, shading="auto", cmap=cmap1, norm=norm1)
    ax1.set_title(t1)
    cb1 = fig.colorbar(m1, ax=ax1, orientation="horizontal", pad=0.05, fraction=0.045, extend=extend1)
    cb1.ax.tick_params(labelsize=20)

    ax2 = fig.add_subplot(gs[0, 1], projection=plot_crs); setup_ax(ax2)
    m2  = ax2.pcolormesh(x2, y2, da2.values, transform=data_crs, shading="auto", cmap=cmap2, norm=norm2)
    ax2.set_title(t2)
    cb2 = fig.colorbar(m2, ax=ax2, orientation="horizontal", pad=0.05, fraction=0.045, extend=extend2)
    cb2.ax.tick_params(labelsize=20)

    ax3 = fig.add_subplot(gs[0, 2], projection=plot_crs); setup_ax(ax3)
    m3  = ax3.pcolormesh(x3, y3, da3.values, transform=data_crs, shading="auto", cmap=cmap3, norm=norm3)
    ax3.set_title(t3)
    cb3 = fig.colorbar(m3, ax=ax3, orientation="horizontal", pad=0.05, fraction=0.045, extend=extend3)
    cb3.ax.tick_params(labelsize=20)

    fig.text(0.5, 0.01,
             "Hinweis: Regime-Zuordnung am Fensterende; Karten zeigen Zeitmittel über das Event-Fenster.",
             ha="center", fontsize=20, alpha=0.8)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _setup_overview_ax(ax) -> None:
    ax.set_extent(EUROPE_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.45)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.30)
    ax.add_feature(cfeature.LAKES,     alpha=0.20)
    ax.add_feature(cfeature.RIVERS,    alpha=0.12)
    try:
        if "geo" in ax.spines:
            ax.spines["geo"].set_linewidth(0.55)
            ax.spines["geo"].set_alpha(0.7)
    except Exception:
        pass


def _sort_key_regime(r: str) -> Tuple[int, int]:
    if r == NO_REGIME_LABEL:
        return (1, 10**9)
    try:    return (0, int(r))
    except: return (0, 10**8)


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
    v  = np.concatenate(vals)
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
    rsds_anom_norm: TwoSlopeNorm,
) -> None:
    """
    Fügt drei horizontale Colorbars am unteren Bildrand ein.
    Positionen werden manuell via fig.add_axes() gesetzt –
    dadurch kein Überlappen unabhängig von der GridSpec-Höhe.

    Layout (Figure-Koordinaten, Ursprung unten-links):
      [left, bottom, width, height]
      Colorbar 1 (Temp):  bottom = 0.14
      Colorbar 2 (Wind):  bottom = 0.08
      Colorbar 3 (Rsds):  bottom = 0.02
    """
    CB_LEFT   = 0.10
    CB_WIDTH  = 0.80
    CB_HEIGHT = 0.022
    CB_BOTTOM = [0.14, 0.08, 0.02]   # Temp, Wind, Rsds – je 0.06 Abstand

    cax_t = fig.add_axes([CB_LEFT, CB_BOTTOM[0], CB_WIDTH, CB_HEIGHT])
    cax_w = fig.add_axes([CB_LEFT, CB_BOTTOM[1], CB_WIDTH, CB_HEIGHT])
    cax_s = fig.add_axes([CB_LEFT, CB_BOTTOM[2], CB_WIDTH, CB_HEIGHT])

    cb_t = fig.colorbar(mappable_t, cax=cax_t, orientation="horizontal", extend="both")
    cb_t.set_label("Temperatur-Anomalie (Event − Monatsklima) [°C]", fontsize=20)
    cb_t.set_ticks(np.linspace(tas_norm.vmin, tas_norm.vmax, 9))
    cb_t.ax.tick_params(labelsize=20)

    cb_w = fig.colorbar(mappable_w, cax=cax_w, orientation="horizontal", extend="both")
    cb_w.set_label("Windgeschwindigkeit-Anomalie (Event − Monatsklima) [m s⁻¹]", fontsize=20)
    cb_w.set_ticks(np.linspace(wind_norm.vmin, wind_norm.vmax, 9))
    cb_w.ax.tick_params(labelsize=20)

    cb_s = fig.colorbar(mappable_s, cax=cax_s, orientation="horizontal", extend="both")
    cb_s.set_label("Solare Einstrahlung Rsds-Anomalie (Event − Monatsklima) [W m⁻²]", fontsize=20)
    cb_s.set_ticks(np.linspace(rsds_anom_norm.vmin, rsds_anom_norm.vmax, 9))
    cb_s.ax.tick_params(labelsize=20)


# =============================================================================
# Overview grid: 2 Regime pro Zeile, je (Temp-Anom | Wind-Anom | Rsds-Anom)
# =============================================================================
def plot_overview_grid_two_regimes_per_row(
    by_regime: Dict[str, Dict[str, object]],
    out_png: str,
    *,
    title: str,
    tas_norm: TwoSlopeNorm,
    wind_norm: TwoSlopeNorm,
    rsds_anom_norm: TwoSlopeNorm,
    tas_cmap:  str = "RdBu_r",
    wind_cmap: str = "PuOr",
    rsds_cmap: str = "RdYlBu_r",
    dpi: int = 300,
) -> None:
    if not by_regime:
        raise ValueError("by_regime ist leer – nichts zu plotten.")
    regimes  = sorted(by_regime.keys(), key=_sort_key_regime)
    n_reg    = len(regimes)
    n_pairs  = (n_reg + 1) // 2
    plot_crs = by_regime[regimes[0]]["plot_crs"]

    # Karten-GridSpec lässt unten 0.22 (22 %) Platz frei für die 3 Colorbars
    fig = plt.figure(figsize=(24.0, 4.2 * n_pairs + 2.0), dpi=dpi)
    gs  = fig.add_gridspec(
        nrows=n_pairs, ncols=6,
        hspace=0.10, wspace=0.02,
        top=0.93, bottom=0.22, left=0.03, right=0.99,
    )
    mappable_t = mappable_w = mappable_s = None

    for i in range(n_pairs):
        reg_a = regimes[2 * i]
        reg_b = regimes[2 * i + 1] if (2 * i + 1) < n_reg else None

        def _draw_triplet(dX, col_offset):
            nonlocal mappable_t, mappable_w, mappable_s
            tasX  = dX["tas_anom"]
            windX = dX["wind_anom"]
            rsdsX = dX["rsds_anom"]
            dcrs  = dX["data_crs"]
            tx, ty, _, _ = _get_xy_coords(tasX)
            wx, wy, _, _ = _get_xy_coords(windX)
            sx, sy, _, _ = _get_xy_coords(rsdsX)
            axT = fig.add_subplot(gs[i, col_offset + 0], projection=plot_crs); _setup_overview_ax(axT)
            axW = fig.add_subplot(gs[i, col_offset + 1], projection=plot_crs); _setup_overview_ax(axW)
            axS = fig.add_subplot(gs[i, col_offset + 2], projection=plot_crs); _setup_overview_ax(axS)
            mt = axT.pcolormesh(tx, ty, tasX.values,  transform=dcrs, shading="auto", cmap=tas_cmap,  norm=tas_norm)
            mw = axW.pcolormesh(wx, wy, windX.values, transform=dcrs, shading="auto", cmap=wind_cmap, norm=wind_norm)
            ms = axS.pcolormesh(sx, sy, rsdsX.values, transform=dcrs, shading="auto", cmap=rsds_cmap, norm=rsds_anom_norm)
            if mappable_t is None: mappable_t = mt
            if mappable_w is None: mappable_w = mw
            if mappable_s is None: mappable_s = ms
            reg_lbl = reg_a if col_offset == 0 else reg_b
            st  = pd.to_datetime(dX["start"])
            en  = pd.to_datetime(dX["end"])
            val = float(dX["val_gw"])
            axT.text(0.01, 0.98,
                     f"Reg {reg_lbl}  |  {val:.1f} GW\n{st.strftime('%Y-%m-%d')} → {en.strftime('%Y-%m-%d')}",
                     transform=axT.transAxes, ha="left", va="top", fontsize=20,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))
            if i == 0:
                axT.set_title("Temp-Anomalie [°C]",     fontsize=20)
                axW.set_title("Wind-Anomalie [m s⁻¹]",  fontsize=20)
                axS.set_title("Rsds-Anomalie [W m⁻²]",  fontsize=20)

        _draw_triplet(by_regime[reg_a], col_offset=0)
        if reg_b is not None:
            _draw_triplet(by_regime[reg_b], col_offset=3)
        else:
            for c in range(3, 6):
                ax_e = fig.add_subplot(gs[i, c]); ax_e.axis("off")

    fig.suptitle(title, fontsize=20, y=0.975)
    _add_colorbars(fig, mappable_t, mappable_w, mappable_s, tas_norm, wind_norm, rsds_anom_norm)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Overview grid gespeichert: {out_png}")


# =============================================================================
# Top-10 pro Regime: Grid
# =============================================================================
def plot_top10_grid_for_one_regime(
    rows: List[Dict[str, object]],
    out_png: str,
    *,
    title: str,
    tas_norm: TwoSlopeNorm,
    wind_norm: TwoSlopeNorm,
    rsds_anom_norm: TwoSlopeNorm,
    tas_cmap:  str = "RdBu_r",
    wind_cmap: str = "PuOr",
    rsds_cmap: str = "RdYlBu_r",
    dpi: int = 300,
) -> None:
    if not rows:
        print("[WARN] Keine Rows für Top10-Grid – skip.")
        return

    rows = sorted(rows, key=lambda r: int(r.get("rank", 10**9)))
    slots: List[Optional[Dict[str, object]]] = [None] * 10
    for r in rows[:10]:
        rk = int(r.get("rank", 0))
        if 1 <= rk <= 10:
            slots[rk - 1] = r
        else:
            for i in range(10):
                if slots[i] is None:
                    slots[i] = r
                    break

    plot_crs = next(s["plot_crs"] for s in slots if s is not None)

    N_ROWS = 5
    # Karten-GridSpec lässt unten 0.22 Platz für 3 Colorbars
    fig = plt.figure(figsize=(24.0, 4.2 * N_ROWS + 2.0), dpi=dpi)
    gs  = fig.add_gridspec(
        nrows=N_ROWS, ncols=6,
        hspace=0.10, wspace=0.02,
        top=0.93, bottom=0.22, left=0.03, right=0.99,
    )
    mappable_t = mappable_w = mappable_s = None

    def _plot_slot(slot_idx: int, col_offset: int, row_i: int) -> None:
        nonlocal mappable_t, mappable_w, mappable_s
        slot_data = slots[slot_idx]
        ax_t = fig.add_subplot(gs[row_i, col_offset + 0], projection=plot_crs)
        ax_w = fig.add_subplot(gs[row_i, col_offset + 1], projection=plot_crs)
        ax_s = fig.add_subplot(gs[row_i, col_offset + 2], projection=plot_crs)
        _setup_overview_ax(ax_t); _setup_overview_ax(ax_w); _setup_overview_ax(ax_s)

        if slot_data is None:
            for ax in (ax_t, ax_w, ax_s):
                ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                        ha="center", va="center", fontsize=20, alpha=0.35)
        else:
            tas  = slot_data["tas_anom"]
            wind = slot_data["wind_anom"]
            rsds = slot_data["rsds_anom"]
            dcrs = slot_data["data_crs"]
            tx, ty, _, _ = _get_xy_coords(tas)
            wx, wy, _, _ = _get_xy_coords(wind)
            sx, sy, _, _ = _get_xy_coords(rsds)
            mt = ax_t.pcolormesh(tx, ty, tas.values,  transform=dcrs, shading="auto", cmap=tas_cmap,  norm=tas_norm)
            mw = ax_w.pcolormesh(wx, wy, wind.values, transform=dcrs, shading="auto", cmap=wind_cmap, norm=wind_norm)
            ms = ax_s.pcolormesh(sx, sy, rsds.values, transform=dcrs, shading="auto", cmap=rsds_cmap, norm=rsds_anom_norm)
            if mappable_t is None: mappable_t = mt
            if mappable_w is None: mappable_w = mw
            if mappable_s is None: mappable_s = ms
            st  = pd.to_datetime(slot_data["start"])
            en  = pd.to_datetime(slot_data["end"])
            val = float(slot_data["val_gw"])
            rk  = int(slot_data.get("rank", 0))
            ax_t.text(0.01, 0.98,
                      f"#{rk}  |  {val:.1f} GW\n{st.strftime('%Y-%m-%d')} → {en.strftime('%Y-%m-%d')}",
                      transform=ax_t.transAxes, ha="left", va="top", fontsize=20,
                      bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))

        if row_i == 0:
            ax_t.set_title("Temp-Anomalie [°C]",    fontsize=20)
            ax_w.set_title("Wind-Anomalie [m s⁻¹]", fontsize=20)
            ax_s.set_title("Rsds-Anomalie [W m⁻²]", fontsize=20)

    for row_i in range(N_ROWS):
        _plot_slot(row_i,     col_offset=0, row_i=row_i)   # Events #1–#5  (links)
        _plot_slot(row_i + 5, col_offset=3, row_i=row_i)   # Events #6–#10 (rechts)

    fig.suptitle(title, fontsize=20, y=0.975)
    _add_colorbars(fig, mappable_t, mappable_w, mappable_s, tas_norm, wind_norm, rsds_anom_norm)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Top10-Grid gespeichert: {out_png}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("--- Lade Residual Load & Regime ---")
    agg = load_aggregated_residual_series()
    print(f"Aggregiert: {agg.index.min()} bis {agg.index.max()}")

    print("Regime laden…")
    reg_ts = read_regimes_timeseries(REGIME_DIR)
    reg_ts["regime"] = normalize_regime_labels(reg_ts["regime"])
    reg_hourly = align_regimes_to_hourly(reg_ts, agg.index)
    print("Regime counts (hourly):")
    print(reg_hourly.value_counts().to_string())

    regimes_order = [str(i) for i in range(K_CLUSTERS)] + [NO_REGIME_LABEL]

    # 1) Heftigstes Event pro Regime
    print(f"\n--- Finde heftigste Events pro Regime (window={EVENT_WINDOW_DAYS}d) ---")
    events = find_heaviest_event_per_regime(
        series_mw=agg["series_mw"], regime_hourly=reg_hourly,
        window_days=EVENT_WINDOW_DAYS, min_coverage=ROLLING_MIN_COVERAGE,
        regime_window_coverage=REGIME_WINDOW_COVERAGE, strict_within_regime=True,
        regimes_ordered=regimes_order, no_regime_label=NO_REGIME_LABEL,
    )
    if events.empty:
        raise RuntimeError("Keine Events gefunden (prüfe Regime-Alignment oder Rolling-Fenster).")
    events_csv = os.path.join(TABLES_DIR, f"heaviest_event_per_regime__{EVENT_WINDOW_DAYS}d.csv")
    events.to_csv(events_csv, index=False)
    print(f"[OK] Event-Liste gespeichert: {events_csv}")
    print(events.to_string(index=False))

    # 2) Top-10 pro Regime
    top_events = find_top_events_per_regime(
        series_mw=agg["series_mw"], regime_hourly=reg_hourly,
        top_n=10, window_days=EVENT_WINDOW_DAYS, min_coverage=ROLLING_MIN_COVERAGE,
        regime_window_coverage=REGIME_WINDOW_COVERAGE, strict_within_regime=True,
        regimes_ordered=regimes_order, no_regime_label=NO_REGIME_LABEL,
    )
    top_events_csv = os.path.join(TABLES_DIR, f"top10_events_per_regime__{EVENT_WINDOW_DAYS}d.csv")
    top_events.to_csv(top_events_csv, index=False)
    print(f"[OK] Top-10-Eventliste gespeichert: {top_events_csv}")

    # =========================================================================
    # A) Einzelkarten + Overview (heftigstes Event je Regime)
    # =========================================================================
    overview_by_regime: Dict[str, Dict[str, object]] = {}
    overview_wind_anom_arrays: List[xr.DataArray] = []
    overview_rsds_anom_arrays: List[xr.DataArray] = []

    print("\n--- Erzeuge Karten (ABS + ANOM) für heftigstes Event je Regime ---")
    for row in events.to_dict("records"):
        reg    = str(row["regime"])
        start  = pd.to_datetime(row["start"])
        end    = pd.to_datetime(row["end"])
        val_gw = float(row["value_gw"])
        safe_reg = re.sub(r"[^A-Za-z0-9_\-]+", "_", reg)
        stamp    = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}__{EVENT_WINDOW_DAYS}d"

        print(f"\nRegime {reg}: lade CORDEX Felder für {start} → {end}")
        tas_mean, wind_mean, rsds_mean, data_crs, plot_crs = load_event_fields_tas_wind_rsds(start, end)

        try:
            tas_anom  = compute_tas_anomaly(tas_mean,  start=start)
            wind_anom = compute_wind_anomaly(wind_mean, start=start)
            rsds_anom = compute_rsds_anomaly(rsds_mean, start=start)
            anom_ok   = True
        except Exception as e:
            print(f"  WARN: Anomalie-Berechnung fehlgeschlagen: {e}")
            tas_anom = wind_anom = rsds_anom = None
            anom_ok  = False

        # Einzelkarte ABS
        outpng_abs = os.path.join(MAPS_ABS_DIR, f"maps_abs_regime_{safe_reg}__{stamp}.png")
        plot_three_panel_map(
            da1=tas_mean, da2=wind_mean, da3=rsds_mean,
            data_crs=data_crs, plot_crs=plot_crs,
            title=(f"Regime: {reg} | Heftigste Dunkelflaute ({EVENT_WINDOW_DAYS}d rolling mean)\n"
                   f"{start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} "
                   f"| Residual mean: {val_gw:.2f} GW"),
            t1=f"Temperatur ABS (Zeitmittel) [{tas_mean.attrs.get('units', '')}]",
            t2=f"Wind ABS (Zeitmittel) [{wind_mean.attrs.get('units', '')}]",
            t3=f"Rsds ABS (Zeitmittel) [{rsds_mean.attrs.get('units', '')}]",
            outpath=outpng_abs,
            cmap1="RdBu_r", norm1=ABS_TAS_NORM,
            cmap2="viridis", norm2=None, extend2="max",
            cmap3="viridis", norm3=None, extend3="max",
        )
        print(f"  -> ABS map gespeichert: {outpng_abs}")

        # Einzelkarte ANOM
        if anom_ok and tas_anom is not None and wind_anom is not None and rsds_anom is not None:
            rs_lo, rs_hi = _robust_global_percentile_limits([rsds_anom])
            rs_v = max(max(abs(rs_lo), abs(rs_hi)) if np.isfinite(rs_lo) and np.isfinite(rs_hi) else 1.0, 1.0)
            w_lo,  w_hi  = _robust_global_percentile_limits([wind_anom])
            w_v  = max(max(abs(w_lo),  abs(w_hi))  if np.isfinite(w_lo)  and np.isfinite(w_hi)  else 1.0, 1e-6)

            outpng_anom = os.path.join(MAPS_ANOM_DIR, f"maps_anom_regime_{safe_reg}__{stamp}.png")
            plot_three_panel_map(
                da1=tas_anom, da2=wind_anom, da3=rsds_anom,
                data_crs=data_crs, plot_crs=plot_crs,
                title=(f"Regime: {reg} | Anomalien vs Monatsklima ({CLIM_MODE})\n"
                       f"{start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} "
                       f"| Residual mean: {val_gw:.2f} GW"),
                t1="Temp-Anomalie (Event − Monatsklima) [°C]",
                t2=f"Wind-Anomalie (Event − Monatsklima) [{wind_anom.attrs.get('units', '')}]",
                t3=f"Rsds-Anomalie (Event − Monatsklima) [{rsds_anom.attrs.get('units', '')}]",
                outpath=outpng_anom,
                cmap1="RdBu_r", norm1=make_anom_norm(vmax=TAS_ANOM_VMAX),
                cmap2="RdBu_r", norm2=make_anom_norm(vmax=w_v),
                cmap3="RdBu_r", norm3=make_anom_norm(vmax=rs_v),
            )
            print(f"  -> ANOM map gespeichert: {outpng_anom}")

            overview_by_regime[reg] = {
                "tas_anom": tas_anom, "wind_anom": wind_anom, "rsds_anom": rsds_anom,
                "data_crs": data_crs, "plot_crs": plot_crs,
                "start": start, "end": end, "val_gw": val_gw,
            }
            overview_wind_anom_arrays.append(wind_anom)
            overview_rsds_anom_arrays.append(rsds_anom)
        else:
            print(f"  [WARN] Regime {reg}: keine Anomalien verfügbar -> fehlt in Overview.")

    # Overview-Norms aus Overview-Arrays ableiten
    wind_vmax = rsds_vmax = None
    if overview_by_regime:
        wlo, whi = _robust_global_percentile_limits(overview_wind_anom_arrays)
        wind_vmax = max(max(abs(wlo), abs(whi)) if np.isfinite(wlo) and np.isfinite(whi) else 1.0, 1e-6)

        if RSDS_ANOM_VMIN is None or RSDS_ANOM_VMAX is None:
            slo, shi = _robust_global_percentile_limits(overview_rsds_anom_arrays)
            vmax = max(max(abs(slo), abs(shi)) if np.isfinite(slo) and np.isfinite(shi) else 1.0, 1.0)
            rsds_vmax = vmax if RSDS_ANOM_VMAX is None else float(RSDS_ANOM_VMAX)
        else:
            rsds_vmax = float(RSDS_ANOM_VMAX)
        if not np.isfinite(rsds_vmax) or rsds_vmax <= 0:
            rsds_vmax = 1.0

        out_overview = os.path.join(
            OVERVIEW_DIR,
            f"overview__TEMP_ANOM_WIND_ANOM_RSDS_ANOM__{EVENT_WINDOW_DAYS}d__{CLIM_MODE}"
            f"__tas{TAS_ANOM_VMAX:g}__windanom{wind_vmax:.3f}__rsdsanom{rsds_vmax:.1f}.png",
        )
        plot_overview_grid_two_regimes_per_row(
            by_regime=overview_by_regime, out_png=out_overview,
            title=(""),
            tas_norm=make_anom_norm(vmax=TAS_ANOM_VMAX),
            wind_norm=make_anom_norm(vmax=wind_vmax),
            rsds_anom_norm=make_anom_norm(vmax=rsds_vmax),
            tas_cmap="RdBu_r", wind_cmap="PuOr", rsds_cmap="RdYlBu_r", dpi=300,
        )
    else:
        print("[WARN] Overview-Daten leer – keine Übersicht erzeugt.")

    # =========================================================================
    # B) Top-10 pro Regime Gridplots
    # =========================================================================
    if top_events.empty:
        print("[WARN] top_events leer – keine Top10-Grids erzeugt.")
    else:
        print("\n--- Erzeuge Top-10-Übersichten pro Regime (inkl. no_regime) ---")
        if wind_vmax is None: wind_vmax = 5.0
        if rsds_vmax is None: rsds_vmax = 80.0

        cache: Dict[Tuple[pd.Timestamp, pd.Timestamp], Dict[str, object]] = {}

        def _get_fields_for_event(st: pd.Timestamp, en: pd.Timestamp) -> Optional[Dict[str, object]]:
            key = (st, en)
            if key in cache:
                return cache[key]
            try:
                tas_mean, wind_mean, rsds_mean, data_crs, plot_crs = load_event_fields_tas_wind_rsds(st, en)
                cache[key] = {
                    "tas_anom":  compute_tas_anomaly(tas_mean,  start=st),
                    "wind_anom": compute_wind_anomaly(wind_mean, start=st),
                    "rsds_anom": compute_rsds_anomaly(rsds_mean, start=st),
                    "data_crs":  data_crs, "plot_crs": plot_crs,
                }
                return cache[key]
            except Exception as e:
                print(f"  [WARN] Event load failed {st} -> {en}: {e}")
                return None

        for reg, g in top_events.groupby("regime"):
            rows = []
            for rec in g.sort_values("rank").to_dict("records"):
                st = pd.to_datetime(rec["start"]); en = pd.to_datetime(rec["end"])
                f  = _get_fields_for_event(st, en)
                if f is None:
                    continue
                rows.append({
                    "rank": int(rec["rank"]), "start": st, "end": en,
                    "val_gw": float(rec["value_gw"]),
                    **{k: f[k] for k in ("tas_anom", "wind_anom", "rsds_anom", "data_crs", "plot_crs")},
                })
            if not rows:
                print(f"[WARN] Regime {reg}: keine plottbaren Top10-Events")
                continue
            safe_reg = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(reg))
            out_png  = os.path.join(
                TOP10_REGIME_DIR,
                f"top10_grid__regime_{safe_reg}__{EVENT_WINDOW_DAYS}d__{CLIM_MODE}__RSDS_ANOM.png",
            )
            plot_top10_grid_for_one_regime(
                rows=rows, out_png=out_png,
                title=f"",
                tas_norm=make_anom_norm(vmax=TAS_ANOM_VMAX),
                wind_norm=make_anom_norm(vmax=wind_vmax),
                rsds_anom_norm=make_anom_norm(vmax=rsds_vmax),
                tas_cmap="RdBu_r", wind_cmap="PuOr", rsds_cmap="RdYlBu_r", dpi=300,
            )

    print("\nFertig.")


if __name__ == "__main__":
    main()