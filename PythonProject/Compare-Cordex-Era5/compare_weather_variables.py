from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Hashable

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# =============================================================================
# ERA5 normalization to CORDEX-like vars
# =============================================================================

_PARAMID_TO_CANON = {
    167: "t2m",   # 2m temperature
    165: "u10",   # 10m u wind
    166: "v10",   # 10m v wind
    169: "ssrd",  # surface solar radiation downwards (J m-2 over timestep in many exports)
    134: "sp",    # surface pressure
    168: "d2m",   # 2m dewpoint temperature (often)
    247: "ssrdc", # clear-sky solar rad downwards (optional)
}

_FALLBACK_VARXXX = {
    "var167": "t2m",
    "var165": "u10",
    "var166": "v10",
    "var169": "ssrd",
    "var134": "sp",
    "var168": "d2m",
    "var247": "ssrdc",
}

_ERA5NAME_TO_CORDEX_DIRECT = {
    "t2m": "tas",
    "2t": "tas",
    "2m_temperature": "tas",
    "temperature": "tas",  # cutout
    "surface_solar_radiation_downwards": "rsds",  # if already W m-2
    "specific_humidity": "huss",
    "huss": "huss",
}


# =============================================================================
# Defaults
# =============================================================================
VARS = ["rsds", "sfcWind", "tas"]
COUNTRIES_ISO2 = ["DE", "DK", "IT", "CH", "FR", "ES", "GB"]

ISO2_TO_NAME = {
    "DE": "Germany",
    "DK": "Denmark",
    "IT": "Italy",
    "CH": "Switzerland",
    "FR": "France",
    "ES": "Spain",
    "GB": "United Kingdom",
}

COLORS = {"era5": "black", "rcp26": "tab:blue", "rcp45": "tab:orange"}
ROLLING_WINDOWS_DAYS = [2, 5, 7, 14, 30]


# =============================================================================
# Utils
# =============================================================================
_WARNED_ONCE: set[tuple[str, str]] = set()  # (dataset_name, var)


def warn(msg: str) -> None:
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def warn_once(key: tuple[str, str], msg: str) -> None:
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    warn(msg)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _find_time_coord(obj) -> Optional[str]:
    coords = getattr(obj, "coords", {})
    dims = getattr(obj, "dims", ())
    for c in ["time", "valid_time", "date", "datetime", "forecast_time"]:
        if (c in coords) or (c in dims):
            return c
    for c in list(coords):
        try:
            if np.issubdtype(coords[c].dtype, np.datetime64):
                return c
        except Exception:
            pass
    return None

def _get_units(ds: xr.Dataset, var: str) -> str:
    """
    Best-effort units lookup for a variable in an xarray Dataset.
    """
    try:
        da = _get_var_da(ds, var)
    except Exception:
        return ""

    a = da.attrs or {}
    for k in ["units", "unit", "Units", "UNITS"]:
        u = a.get(k, "")
        if u is not None and str(u).strip():
            return str(u).strip()
    return ""



def _wind_100m_to_10m_log_profile(w100: xr.DataArray, z0: xr.DataArray | float = 0.03) -> xr.DataArray:
    """
    Convert wind speed from 100 m to 10 m using log wind profile (neutral stability):
        u10 = u100 * ln(10/z0) / ln(100/z0)
    """
    if isinstance(z0, xr.DataArray):
        z0_da = z0
    else:
        # keep shape without exploding memory
        tdim = _find_time_coord(w100)
        template = (
            w100.isel({tdim: 0})
            if (tdim and tdim in w100.dims)
            else (w100.isel({w100.dims[0]: 0}) if w100.ndim > 0 else w100)
        )
        z0_da = xr.full_like(template, float(z0))

    z0_da = xr.where(z0_da > 1e-6, z0_da, 1e-6)
    u10 = w100 * (np.log(10.0 / z0_da) / np.log(100.0 / z0_da))
    u10.attrs.update(
        {
            "long_name": "Near-surface wind speed (10m) derived from 100m via log profile (neutral)",
            "units": w100.attrs.get("units", "m s-1"),
            "source": "u10 = u100 * ln(10/z0)/ln(100/z0)",
        }
    )
    return u10


def _ensure_datetime_index(idx: pd.Index) -> pd.DatetimeIndex:
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    out = pd.to_datetime(idx, errors="coerce")
    return pd.DatetimeIndex(out)


def _maybe_parse_time_axis(da: xr.DataArray) -> pd.DatetimeIndex:
    tdim = _find_time_coord(da)
    if tdim is None or tdim not in da.dims:
        raise RuntimeError("No time dimension found.")
    t = pd.to_datetime(da[tdim].values, errors="coerce")
    if len(t) == 0 or pd.isna(t).all():
        raise RuntimeError("Time axis not parseable.")
    return pd.DatetimeIndex(t)


# =============================================================================
# Time harmonization (ERA5 1h vs CORDEX 3h)
# =============================================================================
def _sanitize_series(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    ss = s.copy()
    ss.index = pd.DatetimeIndex(pd.to_datetime(ss.index, errors="coerce"))
    ss = ss[~ss.index.isna()].sort_index()
    if ss.index.has_duplicates:
        ss = ss.groupby(level=0).mean()
    return ss


def harmonize_to_freq(series_by_ds: Dict[str, pd.Series], target_freq: str, how: str = "mean") -> Dict[str, pd.Series]:
    """
    Resample each series to a common frequency (default: 3h).
    """
    target_freq = str(target_freq).strip()
    # pandas deprecates 'H' -> use lowercase
    target_freq = target_freq.replace("H", "h")

    out: Dict[str, pd.Series] = {}
    for k, s in series_by_ds.items():
        ss = _sanitize_series(s)
        if ss.empty:
            out[k] = ss
            continue
        if how == "mean":
            out[k] = ss.resample(target_freq).mean()
        elif how == "median":
            out[k] = ss.resample(target_freq).median()
        else:
            raise ValueError(f"Unknown resample how='{how}'")
    return out


# =============================================================================
# ERA5 normalization (methodically correct solar conversion)
# =============================================================================
def _is_accumulated_radiation_jm2(da: xr.DataArray) -> bool:
    """
    Decide if a radiation variable is accumulated energy (J m-2) rather than flux (W m-2).
    Works even if attrs are missing.
    """
    a = da.attrs or {}
    units = str(a.get("units", "")).strip().lower()
    step = str(a.get("GRIB_stepType", a.get("stepType", ""))).strip().lower()

    # explicit signals
    if "j" in units and ("m-2" in units or "m**-2" in units or "m^-2" in units):
        return True
    if step in {"accum", "accumulation"}:
        return True

    # heuristic if attrs missing:
    # hourly J/m2 can reach millions; flux rarely exceeds ~1500 W/m2.
    try:
        vmax = float(da.max(skipna=True))
        if np.isfinite(vmax) and vmax > 5e4:
            return True
    except Exception:
        pass

    return False


def _infer_dt_seconds_from_timecoord(acc: xr.DataArray) -> float:
    tdim = _find_time_coord(acc)
    if tdim is None or tdim not in acc.dims:
        raise RuntimeError("No time dimension found for dt inference.")
    t = pd.to_datetime(acc[tdim].values, errors="coerce")
    if len(t) < 2 or pd.isna(t).any():
        raise RuntimeError("Time axis not usable for dt inference.")
    dts = (t[1:] - t[:-1]) / np.timedelta64(1, "s")
    dts = np.asarray(dts, dtype="float64")
    dts = dts[np.isfinite(dts) & (dts > 0)]
    if dts.size == 0:
        raise RuntimeError("Could not infer positive dt from time axis.")
    return float(np.median(dts))


def _sample_point_series(acc: xr.DataArray) -> pd.Series:
    """
    Pick a single gridpoint (and first non-time indices) to diagnose accumulation behavior
    without loading the full field into memory.
    """
    tdim = _find_time_coord(acc)
    if tdim is None:
        return pd.Series(dtype=float)

    sel = {d: 0 for d in acc.dims if d != tdim}
    try:
        s = acc.isel(**sel).to_series()
    except Exception:
        # fallback: force to numpy on small slice
        x = acc.isel(**sel).values
        t = pd.to_datetime(acc[tdim].values, errors="coerce")
        s = pd.Series(x, index=t)
    s = s.dropna()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index, errors="coerce"))
    s = s[~s.index.isna()].sort_index()
    return s


def _classify_accumulation_type(acc: xr.DataArray) -> str:
    """
    Classify J m-2 radiation time series as either:
      - "step"   : value represents energy accumulated over the step interval (e.g. 1h total), so flux = acc/dt
      - "running": running accumulation since a reference time (often 00 UTC), so flux = diff(acc)/dt with reset handling

    Heuristic:
      - step accumulation tends to have frequent negative diffs across the day
      - running accumulation tends to be mostly non-decreasing except at reset(s), so negative diffs are rare
    """
    s = _sample_point_series(acc)
    if s.empty or len(s) < 48:
        # be conservative: step is safer (no diff-induced artifacts)
        return "step"

    dif = s.diff()
    neg_share = float((dif < 0).mean())

    # If negative diffs are common, it cannot be a running accumulation.
    # (your all2010.nc shows ~0.28)
    if neg_share >= 0.05:
        return "step"

    # Rare negatives => likely running accumulation (daily resets etc.)
    return "running"


def _jm2_to_wm2(acc: xr.DataArray) -> xr.DataArray:
    """
    Convert radiation from energy (J m-2) to mean flux (W m-2), methodically correct for BOTH ERA5 encodings:
      - step accumulation:   flux(t) = acc(t) / dt
      - running accumulation:flux(t_i) = max(acc(t_i)-acc(t_{i-1}), 0) / dt_i
    """
    tdim = _find_time_coord(acc)
    if tdim is None or tdim not in acc.dims:
        raise RuntimeError("Radiation variable has no time dimension.")

    acc = acc.sortby(acc[tdim])
    dt_sec = _infer_dt_seconds_from_timecoord(acc)

    mode = _classify_accumulation_type(acc)

    if mode == "step":
        flux = acc / dt_sec
        src = f"J m-2 over step -> W m-2 via division by dt={dt_sec:.0f}s (step-accumulation inferred)"
    else:
        # running accumulation
        t = pd.to_datetime(acc[tdim].values, errors="coerce")
        dt = (t[1:] - t[:-1]) / np.timedelta64(1, "s")
        dt = np.asarray(dt, dtype="float64")
        if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
            raise RuntimeError("non-positive or invalid dt in running-accumulation conversion.")

        dacc = acc.diff(tdim)
        dt_da = xr.DataArray(dt, dims=(tdim,), coords={tdim: dacc[tdim]})
        flux = (dacc / dt_da).clip(min=0)
        src = "running J m-2 -> W m-2 via diff(time)/dt per-step, clipped >=0 (running-accumulation inferred)"

    flux.attrs = dict(acc.attrs or {})
    flux.attrs.update({"units": "W m-2", "source": src})
    return flux


def normalize_era5_to_cordex(ds: xr.Dataset) -> xr.Dataset:
    """
    Make ERA5 dataset usable with CORDEX-like names:
      tas, rsds, sfcWind, (huss if present)

    Key correction vs old version:
      rsds conversion is now correct for BOTH ERA5 solar encodings
      (step-accumulation: acc/dt, running accumulation: diff/dt).
    """
    ds = ds.copy()

    # Step 0: resolve varXXX -> canonical names (attrs often missing; fallback mapping covers that)
    rename_varxxx: dict[str, str] = {}
    for v in list(ds.data_vars):
        if not str(v).lower().startswith("var"):
            continue
        a = ds[v].attrs or {}
        pid = a.get("GRIB_paramId", None)
        if pid is None:
            try:
                pid = int(a.get("paramId"))
            except Exception:
                pid = None
        if isinstance(pid, (int, np.integer)) and int(pid) in _PARAMID_TO_CANON:
            rename_varxxx[v] = _PARAMID_TO_CANON[int(pid)]

    for src, tgt in _FALLBACK_VARXXX.items():
        if src in ds.data_vars and tgt not in ds.data_vars and src not in rename_varxxx:
            rename_varxxx[src] = tgt

    if rename_varxxx:
        ds = ds.rename(rename_varxxx)

    # Step 1: direct renames to CORDEX-like vars
    rename_direct: dict[str, str] = {}
    for v in list(ds.data_vars):
        if v in _ERA5NAME_TO_CORDEX_DIRECT:
            tgt = _ERA5NAME_TO_CORDEX_DIRECT[v]
            if tgt not in ds.data_vars and v != tgt:
                rename_direct[v] = tgt
    if "temperature" in ds.data_vars and "tas" not in ds.data_vars:
        rename_direct["temperature"] = "tas"
    if rename_direct:
        ds = ds.rename(rename_direct)

    # Step 2: rsds derivation
    # If already present (e.g. cutout W/m2), keep it.
    if "rsds" not in ds.data_vars:
        if ("influx_direct" in ds.data_vars) and ("influx_diffuse" in ds.data_vars):
            ds["rsds"] = ds["influx_direct"] + ds["influx_diffuse"]
            ds["rsds"].attrs.update(
                {
                    "long_name": "Surface Downwelling Shortwave Radiation",
                    "units": ds["influx_diffuse"].attrs.get("units", "W m-2"),
                    "source": "influx_direct + influx_diffuse",
                }
            )
        elif "ssrd" in ds.data_vars:
            ds = ds.rename({"ssrd": "rsds"})

    # Step 2b: accumulated J/m2 -> flux W/m2 (methodically correct)
    if "rsds" in ds.data_vars and _is_accumulated_radiation_jm2(ds["rsds"]):
        ds["rsds"] = _jm2_to_wm2(ds["rsds"])

    # Step 3: sfcWind
    if "sfcWind" not in ds.data_vars and ("u10" in ds.data_vars) and ("v10" in ds.data_vars):
        ds["sfcWind"] = np.hypot(ds["u10"], ds["v10"])
        ds["sfcWind"].attrs.update(
            {"long_name": "Near-surface wind speed (10m)", "units": "m s-1", "source": "hypot(u10,v10)"}
        )

    if "sfcWind" not in ds.data_vars and "wnd100m" in ds.data_vars:
        z0 = None
        for cand in ["roughness", "z0", "z0m", "roughness_length"]:
            if cand in ds.data_vars:
                z0 = ds[cand]
                break
            if cand in ds.coords:
                z0 = ds.coords[cand]
                break
        if z0 is None:
            z0 = 0.03
        ds["sfcWind"] = _wind_100m_to_10m_log_profile(ds["wnd100m"], z0=z0)

    return ds


def _get_var_da(ds: xr.Dataset, var: str) -> xr.DataArray:
    if var in ds.data_vars:
        return ds[var]
    cand = [v for v in ds.data_vars if v.lower() == var.lower()]
    if cand:
        return ds[cand[0]]
    raise KeyError(f"Variable '{var}' not found. Available: {list(ds.data_vars)[:40]} ...")


def _subset_year_month(da: xr.DataArray, year: int, month: int) -> xr.DataArray:
    tdim = _find_time_coord(da)
    if tdim is None or tdim not in da.dims:
        return da
    try:
        t = da[tdim]
        return da.sel({tdim: (t.dt.year == year) & (t.dt.month == month)})
    except Exception:
        idx = pd.to_datetime(da[tdim].values, errors="coerce")
        mask = (pd.DatetimeIndex(idx).year == year) & (pd.DatetimeIndex(idx).month == month)
        return da.isel({tdim: np.where(mask)[0]})


# =============================================================================
# Country aggregation
# =============================================================================
def _country_ts_via_existing_dim(da: xr.DataArray, country_iso2: str) -> Optional[pd.Series]:
    tdim = _find_time_coord(da)
    if tdim is None:
        return None

    for dim in ["country", "countries", "region", "regions", "iso2", "nuts0", "area"]:
        if dim in da.dims or dim in da.coords:
            coord = da.coords.get(dim, None)
            if coord is None:
                continue
            vals = [str(v) for v in coord.values]

            if country_iso2 in vals:
                x = da.sel({dim: country_iso2})
                other_dims = [d for d in x.dims if d != tdim]
                if other_dims:
                    x = x.mean(dim=other_dims, skipna=True)
                return x.to_series()

            nm = ISO2_TO_NAME.get(country_iso2)
            if nm and nm in vals:
                x = da.sel({dim: nm})
                other_dims = [d for d in x.dims if d != tdim]
                if other_dims:
                    x = x.mean(dim=other_dims, skipna=True)
                return x.to_series()

    return None


def _get_regionmask_countries():
    import regionmask  # type: ignore

    for attr in ["natural_earth_v5_0_0", "natural_earth_v4_1_0", "natural_earth"]:
        if hasattr(regionmask.defined_regions, attr):
            return getattr(regionmask.defined_regions, attr).countries_110
    return regionmask.defined_regions.natural_earth.countries_110


def _get_spatial_dims(da: xr.DataArray) -> Tuple[Hashable, Hashable]:
    tdim = _find_time_coord(da)
    spatial_dims = [d for d in da.dims if d != tdim]
    if len(spatial_dims) != 2:
        raise RuntimeError(f"Expected exactly 2 spatial dims (besides time), got {spatial_dims} for da.dims={da.dims}")
    return spatial_dims[0], spatial_dims[1]


def _wrap_lon180(lon_vals: np.ndarray) -> np.ndarray:
    return ((lon_vals + 180.0) % 360.0) - 180.0


def _get_lonlat_for_da(ds: xr.Dataset, da: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    lat_names = ["lat", "latitude", "nav_lat"]
    lon_names = ["lon", "longitude", "nav_lon"]

    lat = next((da.coords[n] for n in lat_names if n in da.coords), None)
    lon = next((da.coords[n] for n in lon_names if n in da.coords), None)

    if lat is None:
        lat = next((ds.coords[n] for n in lat_names if n in ds.coords), None)
    if lon is None:
        lon = next((ds.coords[n] for n in lon_names if n in ds.coords), None)

    if lat is None:
        lat = next((ds[n] for n in lat_names if n in ds.data_vars), None)
    if lon is None:
        lon = next((ds[n] for n in lon_names if n in ds.data_vars), None)

    if lat is None or lon is None:
        raise RuntimeError("No lat/lon found for polygon aggregation (need lat/lon coords or vars).")

    if not isinstance(lat, xr.DataArray):
        lat = xr.DataArray(lat)
    if not isinstance(lon, xr.DataArray):
        lon = xr.DataArray(lon)

    # regionmask expects lon in [-180, 180] in practice; wrap if needed
    try:
        lmax = float(np.nanmax(lon.values))
        lmin = float(np.nanmin(lon.values))
        if np.isfinite(lmax) and lmax > 180.0 and np.isfinite(lmin) and lmin >= 0.0:
            lon = xr.apply_ufunc(_wrap_lon180, lon)
    except Exception:
        pass

    return lat, lon


_MASK_CACHE: dict[tuple[int, str, tuple[Hashable, Hashable], tuple[int, int]], xr.DataArray] = {}
_RID_CACHE: dict[str, int] = {}  # country_iso2 -> rid


def _country_rid(ne, country_iso2: str) -> int:
    if country_iso2 in _RID_CACHE:
        return _RID_CACHE[country_iso2]

    abbr = [str(a) for a in getattr(ne, "abbr", [])]
    names = [str(n) for n in getattr(ne, "names", [])]

    if country_iso2 in abbr:
        rid = int(abbr.index(country_iso2))
        _RID_CACHE[country_iso2] = rid
        return rid

    nm = ISO2_TO_NAME.get(country_iso2)
    if nm and nm in names:
        rid = int(names.index(nm))
        _RID_CACHE[country_iso2] = rid
        return rid

    raise KeyError(f"{country_iso2} not found in Natural Earth regions.")


def _build_mask_da_for_grid(ds: xr.Dataset, da: xr.DataArray, ne, country_iso2: str) -> xr.DataArray:
    y_dim, x_dim = _get_spatial_dims(da)
    grid_shape = (int(da.sizes[y_dim]), int(da.sizes[x_dim]))

    cache_key = (id(ds), country_iso2, (y_dim, x_dim), grid_shape)
    if cache_key in _MASK_CACHE:
        return _MASK_CACHE[cache_key]

    lat, lon = _get_lonlat_for_da(ds, da)
    rid = _country_rid(ne, country_iso2)

    if (lon.ndim == 2) and (lat.ndim == 2):
        if lon.shape != grid_shape or lat.shape != grid_shape:
            raise RuntimeError(
                f"lon/lat 2D shapes do not match da grid. da grid={grid_shape}, lon={lon.shape}, lat={lat.shape}"
            )
        lon2d = lon.values
        lat2d = lat.values
    elif (lon.ndim == 1) and (lat.ndim == 1):
        if lat.size != grid_shape[0] or lon.size != grid_shape[1]:
            raise RuntimeError(
                f"lon/lat 1D lengths do not match da grid. "
                f"da grid={grid_shape} (y,x), lon={lon.size}, lat={lat.size}."
            )
        lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    else:
        raise RuntimeError(
            f"Unsupported lon/lat shapes for masking: lon.ndim={lon.ndim}, lat.ndim={lat.ndim}. Need both 1D or both 2D."
        )

    reg_mask_np = ne.mask(lon2d, lat2d)
    reg_mask_da = xr.DataArray(
        reg_mask_np,
        dims=(y_dim, x_dim),
        coords={y_dim: da.coords[y_dim], x_dim: da.coords[x_dim]},
        name="region_id",
    )

    inside = (reg_mask_da == rid)
    inside.name = f"mask_{country_iso2}"

    _MASK_CACHE[cache_key] = inside
    return inside


def _area_weights(lat: xr.DataArray, da: xr.DataArray) -> xr.DataArray:
    """
    Build cos(lat) weights that broadcast over the data grid without creating huge arrays.
    Supports lat as 1D or 2D.
    """
    lat_rad = np.deg2rad(lat)
    w = np.cos(lat_rad)
    w = xr.where(w > 0, w, 0.0)
    return w


def country_ts_polygon_mask(ds: xr.Dataset, da: xr.DataArray, country_iso2: str) -> pd.Series:
    tdim = _find_time_coord(da)
    if tdim is None:
        return pd.Series(dtype=float)

    ne = _get_regionmask_countries()
    inside = _build_mask_da_for_grid(ds, da, ne, country_iso2)
    da_c = da.where(inside)

    y_dim, x_dim = _get_spatial_dims(da_c)

    # area-weighted mean using cos(lat)
    lat, _lon = _get_lonlat_for_da(ds, da_c)
    if lat.ndim == 1:
        if lat.dims and lat.dims[0] != y_dim:
            lat = lat.rename({lat.dims[0]: y_dim})
    w = _area_weights(lat, da_c)

    try:
        ts = da_c.weighted(w).mean(dim=[y_dim, x_dim], skipna=True)
    except Exception:
        ts = da_c.mean(dim=[y_dim, x_dim], skipna=True)

    s = ts.to_series()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    return s


def country_month_timeseries(ds: xr.Dataset, var: str, year: int, month: int, country_iso2: str) -> pd.Series:
    da = _get_var_da(ds, var)
    da_m = _subset_year_month(da, year, month)
    if da_m.size == 0:
        return pd.Series(dtype=float)

    s = _country_ts_via_existing_dim(da_m, country_iso2)
    if s is not None and len(s) > 0:
        return s

    return country_ts_polygon_mask(ds, da_m, country_iso2)


def country_year_timeseries(ds: xr.Dataset, var: str, year: int, country_iso2: str) -> pd.Series:
    da = _get_var_da(ds, var)
    tdim = _find_time_coord(da)
    if tdim is None or tdim not in da.dims:
        return pd.Series(dtype=float)

    try:
        t = da[tdim]
        da_y = da.sel({tdim: (t.dt.year == year)})
    except Exception:
        idx = pd.to_datetime(da[tdim].values, errors="coerce")
        mask = (pd.DatetimeIndex(idx).year == year)
        da_y = da.isel({tdim: np.where(mask)[0]})

    if da_y.size == 0:
        return pd.Series(dtype=float)

    s = _country_ts_via_existing_dim(da_y, country_iso2)
    if s is not None and len(s) > 0:
        return s

    return country_ts_polygon_mask(ds, da_y, country_iso2)


# =============================================================================
# Plotting
# =============================================================================
def plot_country_month(
    series_by_ds: Dict[str, pd.Series],
    var: str,
    year: int,
    month: int,
    country: str,
    out_fp: Path,
    units: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))

    any_data = False
    for dsname in ["era5", "rcp26", "rcp45"]:
        s = series_by_ds.get(dsname, pd.Series(dtype=float))
        if s is None or s.empty:
            continue
        ax.plot(s.index, s.values, label=dsname, color=COLORS.get(dsname), linewidth=1.4, alpha=0.95)
        any_data = True

    unit_str = f" [{units}]" if units else ""
    ax.set_title(f"{var}{unit_str} — {country} — {year:04d}-{month:02d}")
    ax.set_xlabel("Zeit")
    ax.set_ylabel(f"{var}{unit_str}")
    ax.grid(True, alpha=0.25)
    if any_data:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")

    save_fig(fig, out_fp)



# =============================================================================
# Openers
# =============================================================================
def _open_cordex_var_year(base_dir: Path, var: str, year: int) -> xr.Dataset:
    if not base_dir.exists():
        raise FileNotFoundError(base_dir)

    pats = [
        f"**/*{var}*{year}*.nc",
        f"**/{var}*{year}*.nc",
        f"**/*{var}*.nc",
    ]

    files: List[Path] = []
    for pat in pats:
        files = sorted(base_dir.glob(pat))
        if files:
            break

    if not files:
        raise FileNotFoundError(f"No CORDEX files found for var={var} year={year} under {base_dir}")

    if len(files) == 1:
        return xr.open_dataset(files[0], decode_times=True)

    return xr.open_mfdataset([str(f) for f in files], combine="by_coords", decode_times=True)


def _open_era5_allvars(path_like: Path) -> xr.Dataset:
    p = Path(path_like)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.is_file():
        print(f"[ERA5] using file: {p} (MB={p.stat().st_size/1e6:.1f})")
        return xr.open_dataset(p, decode_times=True)

    cands = sorted([*p.rglob("*.nc"), *p.rglob("*.nc4")], key=lambda q: q.stat().st_size, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No ERA5 .nc/.nc4 found under {p}")

    fp = cands[0]
    print(f"[ERA5] using file: {fp} (MB={fp.stat().st_size/1e6:.1f})")
    return xr.open_dataset(fp, decode_times=True)


# =============================================================================
# Metrics: correlations + rolling means (FULL YEAR, per country + ALL)
# =============================================================================
def _pairwise_corr(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 2:
        return (np.nan, int(len(df)))
    return (float(df.iloc[:, 0].corr(df.iloc[:, 1])), int(len(df)))


def _rolling_mean_timebased(s: pd.Series, window_days: int) -> pd.Series:
    return s.rolling(f"{int(window_days)}D", min_periods=1).mean()


def _align_union_index(series_by_ds: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    idx = None
    for s in series_by_ds.values():
        ss = _sanitize_series(s)
        if ss.empty:
            continue
        idx = ss.index if idx is None else idx.union(ss.index)

    if idx is None:
        return {k: pd.Series(dtype=float) for k in series_by_ds.keys()}

    idx = idx.sort_values()
    out: Dict[str, pd.Series] = {}
    for k, s in series_by_ds.items():
        ss = _sanitize_series(s)
        out[k] = ss.reindex(idx) if not ss.empty else pd.Series(index=idx, dtype=float)
    return out


def compute_metrics_rows_full_period(
    var: str,
    country: str,
    year: int,
    series_by_ds: Dict[str, pd.Series],
) -> List[Dict[str, object]]:
    aligned = _align_union_index(series_by_ds)

    era5 = aligned.get("era5", pd.Series(dtype=float))
    rcp26 = aligned.get("rcp26", pd.Series(dtype=float))
    rcp45 = aligned.get("rcp45", pd.Series(dtype=float))

    rows: List[Dict[str, object]] = []

    c_e26, n_e26 = _pairwise_corr(era5, rcp26)
    c_e45, n_e45 = _pairwise_corr(era5, rcp45)
    c_2645, n_2645 = _pairwise_corr(rcp26, rcp45)

    rows.append(
        {
            "var": var,
            "country": country,
            "year": year,
            "metric": "corr_raw_full_year",
            "window_days": np.nan,
            "corr_era5_rcp26": c_e26,
            "n_era5_rcp26": n_e26,
            "corr_era5_rcp45": c_e45,
            "n_era5_rcp45": n_e45,
            "corr_rcp26_rcp45": c_2645,
            "n_rcp26_rcp45": n_2645,
            "era5_roll_mean": np.nan,
            "era5_roll_std": np.nan,
            "rcp26_roll_mean": np.nan,
            "rcp26_roll_std": np.nan,
            "rcp45_roll_mean": np.nan,
            "rcp45_roll_std": np.nan,
        }
    )

    for wd in ROLLING_WINDOWS_DAYS:
        era5_r = _rolling_mean_timebased(era5, wd)
        rcp26_r = _rolling_mean_timebased(rcp26, wd)
        rcp45_r = _rolling_mean_timebased(rcp45, wd)

        c_e26_w, n_e26_w = _pairwise_corr(era5_r, rcp26_r)
        c_e45_w, n_e45_w = _pairwise_corr(era5_r, rcp45_r)
        c_2645_w, n_2645_w = _pairwise_corr(rcp26_r, rcp45_r)

        rows.append(
            {
                "var": var,
                "country": country,
                "year": year,
                "metric": "corr_rolling_mean_full_year",
                "window_days": wd,
                "corr_era5_rcp26": c_e26_w,
                "n_era5_rcp26": n_e26_w,
                "corr_era5_rcp45": c_e45_w,
                "n_era5_rcp45": n_e45_w,
                "corr_rcp26_rcp45": c_2645_w,
                "n_rcp26_rcp45": n_2645_w,
                "era5_roll_mean": float(np.nanmean(era5_r.values)) if len(era5_r) else np.nan,
                "era5_roll_std": float(np.nanstd(era5_r.values)) if len(era5_r) else np.nan,
                "rcp26_roll_mean": float(np.nanmean(rcp26_r.values)) if len(rcp26_r) else np.nan,
                "rcp26_roll_std": float(np.nanstd(rcp26_r.values)) if len(rcp26_r) else np.nan,
                "rcp45_roll_mean": float(np.nanmean(rcp45_r.values)) if len(rcp45_r) else np.nan,
                "rcp45_roll_std": float(np.nanstd(rcp45_r.values)) if len(rcp45_r) else np.nan,
            }
        )

    return rows


def _concat_country_series_for_all(countries: List[str], by_country: Dict[str, Dict[str, pd.Series]]) -> Dict[str, pd.Series]:
    out_parts: Dict[str, List[pd.Series]] = {"era5": [], "rcp26": [], "rcp45": []}
    for c in countries:
        d = by_country.get(c, {})
        for dsname in ["era5", "rcp26", "rcp45"]:
            s = _sanitize_series(d.get(dsname, pd.Series(dtype=float)))
            if not s.empty:
                out_parts[dsname].append(s)

    merged: Dict[str, pd.Series] = {}
    for dsname, parts in out_parts.items():
        if not parts:
            merged[dsname] = pd.Series(dtype=float)
            continue
        cat = pd.concat(parts, axis=0).sort_index()
        if cat.index.has_duplicates:
            cat = cat.groupby(level=0).mean()
        merged[dsname] = cat
    return merged


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2010)
    ap.add_argument("--out", default="metvar_timeseries_out")
    ap.add_argument("--vars", nargs="*", default=VARS)

    ap.add_argument("--rcp45-dir", default="/mnt/endata/Cordex/Copernicus_CORDEX_data/cordex_2010")
    ap.add_argument("--rcp26-dir", default="/mnt/endata/Cordex/Copernicus_CORDEX_data_RCP_2.6/cordex_2010")
    ap.add_argument("--era5-path", default="/mnt/endata/era5/all2010.nc")

    ap.add_argument("--countries", nargs="*", default=COUNTRIES_ISO2)

    ap.add_argument("--target-freq", default="3h", help="Common frequency for ALL series (default: 3h).")
    ap.add_argument("--resample-how", default="mean", choices=["mean", "median"])

    args = ap.parse_args()

    year = int(args.year)
    out_dir = ensure_dir(Path(args.out))
    countries = [c.strip().upper() for c in args.countries if c.strip()]

    target_freq = str(args.target_freq).replace("H", "h")
    resample_how = str(args.resample_how)

    metrics_rows: List[Dict[str, object]] = []

    print("[LOAD] ERA5 (all vars in one file OR cutout)")
    ds_era5 = _open_era5_allvars(Path(args.era5_path))
    ds_era5 = normalize_era5_to_cordex(ds_era5)
    print("[ERA5] data_vars:", list(ds_era5.data_vars))
    print(f"[TIME] Harmonization target frequency: {target_freq} (how={resample_how})")

    for var in args.vars:
        var = str(var)
        print(f"\n[VAR] {var}")

        print(f"[LOAD] CORDEX rcp26 {var}")
        ds_rcp26 = _open_cordex_var_year(Path(args.rcp26_dir), var=var, year=year)

        print(f"[LOAD] CORDEX rcp45 {var}")
        ds_rcp45 = _open_cordex_var_year(Path(args.rcp45_dir), var=var, year=year)

        units = _get_units(ds_rcp45, var) or _get_units(ds_rcp26, var) or _get_units(ds_era5, var) or ""

        if var not in ds_era5.data_vars and not any(v.lower() == var.lower() for v in ds_era5.data_vars):
            warn_once(("era5", var), f"ERA5: variable '{var}' not found after normalization. Will skip ERA5 for {var}.")

        # Full-year series per country (for metrics)
        by_country_full: Dict[str, Dict[str, pd.Series]] = {}

        for country in countries:
            series_full: Dict[str, pd.Series] = {}
            for dsname, ds in [("era5", ds_era5), ("rcp26", ds_rcp26), ("rcp45", ds_rcp45)]:
                try:
                    s_full = country_year_timeseries(ds, var, year, country)
                    series_full[dsname] = s_full
                except Exception as e:
                    warn(f"{dsname} {var} {country} {year}: {e}")
                    series_full[dsname] = pd.Series(dtype=float)

            series_full = harmonize_to_freq(series_full, target_freq=target_freq, how=resample_how)
            by_country_full[country] = series_full

        # Metrics per country (full year)
        for country in countries:
            metrics_rows.extend(compute_metrics_rows_full_period(var, country, year, by_country_full.get(country, {})))

        # Metrics for ALL countries combined (full year)
        all_series = _concat_country_series_for_all(countries, by_country_full)
        metrics_rows.extend(compute_metrics_rows_full_period(var, "ALL", year, all_series))

        # Monthly plots (harmonized before plotting)
        for country in countries:
            cdir = ensure_dir(out_dir / var / country)

            for month in range(1, 13):
                series_by_ds: Dict[str, pd.Series] = {}
                for dsname, ds in [("era5", ds_era5), ("rcp26", ds_rcp26), ("rcp45", ds_rcp45)]:
                    try:
                        s = country_month_timeseries(ds, var, year, month, country)
                        series_by_ds[dsname] = s
                    except Exception as e:
                        warn(f"{dsname} {var} {country} {year}-{month:02d}: {e}")
                        series_by_ds[dsname] = pd.Series(dtype=float)

                series_by_ds = harmonize_to_freq(series_by_ds, target_freq=target_freq, how=resample_how)

                out_fp = cdir / f"{year:04d}-{month:02d}.png"
                plot_country_month(series_by_ds, var, year, month, country, out_fp, units=units)

                if any((not series_by_ds[k].empty) for k in series_by_ds):
                    print(f"[OK] {out_fp}")

    # write CSV
    metrics_fp = out_dir / f"metrics_{year}.csv"
    if metrics_rows:
        dfm = pd.DataFrame(metrics_rows)
        cols = [
            "var", "country", "year", "metric", "window_days",
            "corr_era5_rcp26", "n_era5_rcp26",
            "corr_era5_rcp45", "n_era5_rcp45",
            "corr_rcp26_rcp45", "n_rcp26_rcp45",
            "era5_roll_mean", "era5_roll_std",
            "rcp26_roll_mean", "rcp26_roll_std",
            "rcp45_roll_mean", "rcp45_roll_std",
        ]
        for c in cols:
            if c not in dfm.columns:
                dfm[c] = np.nan
        dfm = dfm[cols]
        ensure_dir(metrics_fp.parent)
        dfm.to_csv(metrics_fp, index=False)
        print(f"[CSV] wrote {metrics_fp} (rows={len(dfm)})")
    else:
        warn("No metrics rows produced; CSV not written.")

    print(f"\n[DONE] Output in {out_dir}")


if __name__ == "__main__":
    main()
