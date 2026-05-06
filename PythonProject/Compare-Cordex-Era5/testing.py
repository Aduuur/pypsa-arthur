#!/usr/bin/env python3
# debug_era5_rsds_conversion.py
#
# Purpose:
#   Debug ERA5 ssrd(J/m^2 accumulated) -> rsds(W/m^2) conversion and identify scaling issues.
#   - Finds/opens an ERA5 file (file or directory -> largest .nc/.nc4)
#   - Locates ssrd (or var169 / paramId 169) and prints:
#       * time axis sanity (sorted?, duplicates?)
#       * timestep distribution (unique dt seconds, median, mode)
#       * units + GRIB stepType hints (accumulation?)
#       * value ranges
#   - Computes rsds via:
#       A) median-dt method (your current approach)
#       B) per-step dt method (robust)
#   - Optionally aggregates to a country mean (polygon mask via regionmask) to compare amplitudes
#   - Optionally resamples both to 3h to mimic your comparison
#
# Usage examples:
#   python debug_era5_rsds_conversion.py --era5-path /mnt/endata/era5/all2010.nc
#   python debug_era5_rsds_conversion.py --era5-path /mnt/endata/era5 --country ES --month 7 --resample 3h
#
# Notes:
#   - This script does NOT need CORDEX. It is purely ERA5 debugging.
#   - If you are using an atlite/PyPSA cutout, ssrd might not exist; then it will look for rsds-like vars.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Hashable, Dict

import numpy as np
import pandas as pd
import xarray as xr


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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
    print(f"[ERA5] using file (largest): {fp} (MB={fp.stat().st_size/1e6:.1f})")
    return xr.open_dataset(fp, decode_times=True)


def _maybe_parse_time_axis(da: xr.DataArray) -> pd.DatetimeIndex:
    tdim = _find_time_coord(da)
    if tdim is None or tdim not in da.dims:
        raise RuntimeError("No time dimension found.")
    t = pd.to_datetime(da[tdim].values, errors="coerce")
    if len(t) == 0 or pd.isna(t).all():
        raise RuntimeError("Time axis not parseable.")
    return pd.DatetimeIndex(t)


def _describe_time_axis(t: pd.DatetimeIndex) -> None:
    print("\n[TIME] axis diagnostics")
    print("  n:", len(t))
    print("  min:", t.min(), " max:", t.max())
    is_sorted = bool(t.is_monotonic_increasing)
    print("  sorted:", is_sorted)
    dup = int(t.duplicated().sum())
    print("  duplicates:", dup)
    nat = int(pd.isna(t).sum())
    print("  NaT:", nat)

    if len(t) >= 2:
        dts = pd.Series(t[1:] - t[:-1]).dt.total_seconds()
        dts = dts.dropna()
        if len(dts):
            uniq = np.unique(dts.values)
            uniq = uniq[np.isfinite(uniq)]
            uniq_sorted = np.sort(uniq)
            print("  dt unique (s) first 20:", uniq_sorted[:20].tolist())
            print("  dt median (s):", float(dts.median()))
            # mode (most frequent)
            vc = dts.value_counts()
            mode = float(vc.index[0]) if len(vc) else np.nan
            print("  dt mode (s):", mode, " (count:", int(vc.iloc[0]) if len(vc) else 0, ")")
            neg = int((dts <= 0).sum())
            print("  dt<=0 count:", neg)
        else:
            print("  dt stats: no valid deltas")


def _find_candidate_ssrd(ds: xr.Dataset) -> Tuple[str, xr.DataArray]:
    """
    Prefer:
      - 'ssrd'
      - 'var169'
      - any var whose attrs paramId/GRIB_paramId == 169
    """
    if "ssrd" in ds.data_vars:
        return "ssrd", ds["ssrd"]

    if "var169" in ds.data_vars:
        return "var169", ds["var169"]

    # search by attributes
    for v in ds.data_vars:
        a = ds[v].attrs or {}
        pid = a.get("GRIB_paramId", None)
        if pid is None:
            try:
                pid = int(a.get("paramId"))
            except Exception:
                pid = None
        if isinstance(pid, (int, np.integer)) and int(pid) == 169:
            return v, ds[v]

    raise KeyError("Could not find ssrd/var169/paramId169 in dataset.")


def _is_accumulated_hint(da: xr.DataArray) -> bool:
    a = da.attrs or {}
    units = str(a.get("units", "")).strip().lower()
    step = str(a.get("GRIB_stepType", a.get("stepType", ""))).strip().lower()
    if "j" in units and ("m-2" in units or "m**-2" in units or "m^-2" in units):
        return True
    if step in {"accum", "accumulation"}:
        return True
    return False


def _subset_year_month(da: xr.DataArray, year: int, month: Optional[int]) -> xr.DataArray:
    tdim = _find_time_coord(da)
    if tdim is None or tdim not in da.dims:
        return da

    t = pd.to_datetime(da[tdim].values, errors="coerce")
    mask = (pd.DatetimeIndex(t).year == year)
    if month is not None:
        mask = mask & (pd.DatetimeIndex(t).month == int(month))
    idx = np.where(mask)[0]
    return da.isel({tdim: idx})


def _median_dt_conversion(ssrd: xr.DataArray) -> xr.DataArray:
    tdim = _find_time_coord(ssrd)
    if not tdim or tdim not in ssrd.dims:
        raise RuntimeError("No time dim for conversion")

    t = _maybe_parse_time_axis(ssrd)
    if len(t) < 2:
        raise RuntimeError("Time axis too short")

    dt_seconds = float(pd.Series(t[1:] - t[:-1]).dt.total_seconds().median())
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        raise RuntimeError(f"Invalid dt_seconds={dt_seconds}")

    rsds = ssrd.diff(tdim) / dt_seconds
    rsds = rsds.clip(min=0)
    rsds = rsds.assign_coords({tdim: ssrd[tdim].isel({tdim: slice(1, None)})})
    rsds.attrs = dict(ssrd.attrs or {})
    rsds.attrs.update({"units": "W m-2", "source": f"diff/dt using median dt={dt_seconds:.0f}s"})
    return rsds


def _per_step_dt_conversion(ssrd: xr.DataArray) -> xr.DataArray:
    tdim = _find_time_coord(ssrd)
    if not tdim or tdim not in ssrd.dims:
        raise RuntimeError("No time dim for conversion")

    # enforce sorted time
    ssrd_s = ssrd.sortby(ssrd[tdim])

    t_vals = pd.to_datetime(ssrd_s[tdim].values, errors="coerce")
    if len(t_vals) < 2:
        raise RuntimeError("Time axis too short")

    dt = (t_vals[1:] - t_vals[:-1]).astype("timedelta64[s]").astype(np.float64)
    dt = np.where(dt > 0, dt, np.nan)

    dacc = ssrd_s.diff(tdim)
    dt_da = xr.DataArray(dt, dims=(tdim,), coords={tdim: dacc[tdim]})

    rsds = dacc / dt_da
    rsds = rsds.clip(min=0)

    rsds.attrs = dict(ssrd_s.attrs or {})
    rsds.attrs.update({"units": "W m-2", "source": "diff/dt using per-step dt (robust)"})
    return rsds


def _to_series(da: xr.DataArray) -> pd.Series:
    tdim = _find_time_coord(da)
    if not tdim:
        raise RuntimeError("No time coord for series conversion")

    # if there are spatial dims, just average everything (debug-only)
    other_dims = [d for d in da.dims if d != tdim]
    x = da
    if other_dims:
        x = x.mean(dim=other_dims, skipna=True)

    s = x.to_series()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    if s.index.has_duplicates:
        s = s.groupby(level=0).mean()
    return s


def _resample_series(s: pd.Series, freq: Optional[str]) -> pd.Series:
    if s.empty or not freq:
        return s
    f = str(freq).strip().replace("H", "h")  # avoid deprecation warning
    return s.resample(f).mean()


# -----------------------------------------------------------------------------
# Optional: polygon mask (country mean) - light reuse of your approach
# -----------------------------------------------------------------------------
ISO2_TO_NAME = {
    "DE": "Germany",
    "DK": "Denmark",
    "IT": "Italy",
    "CH": "Switzerland",
    "FR": "France",
    "ES": "Spain",
    "GB": "United Kingdom",
}


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
        raise RuntimeError("No lat/lon found (need lat/lon coords or vars).")

    if not isinstance(lat, xr.DataArray):
        lat = xr.DataArray(lat)
    if not isinstance(lon, xr.DataArray):
        lon = xr.DataArray(lon)

    return lat, lon


def country_mean_series(ds: xr.Dataset, da: xr.DataArray, country_iso2: str) -> pd.Series:
    try:
        ne = _get_regionmask_countries()
    except Exception as e:
        raise RuntimeError(f"regionmask required. pip install regionmask. ({e})")

    y_dim, x_dim = _get_spatial_dims(da)
    lat, lon = _get_lonlat_for_da(ds, da)

    # regionmask expects lon/lat arrays
    if lon.ndim == 1 and lat.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    elif lon.ndim == 2 and lat.ndim == 2:
        lon2d, lat2d = lon.values, lat.values
    else:
        raise RuntimeError("Unsupported lon/lat dims.")

    # find region id
    abbr = [str(a) for a in getattr(ne, "abbr", [])]
    names = [str(n) for n in getattr(ne, "names", [])]
    if country_iso2 in abbr:
        rid = int(abbr.index(country_iso2))
    else:
        nm = ISO2_TO_NAME.get(country_iso2)
        if nm and nm in names:
            rid = int(names.index(nm))
        else:
            raise KeyError(f"{country_iso2} not found in Natural Earth regions")

    reg_mask = ne.mask(lon2d, lat2d)
    inside = xr.DataArray(reg_mask == rid, dims=(y_dim, x_dim), coords={y_dim: da.coords[y_dim], x_dim: da.coords[x_dim]})

    da_c = da.where(inside)
    ts = da_c.mean(dim=[y_dim, x_dim], skipna=True)
    s = ts.to_series()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    if s.index.has_duplicates:
        s = s.groupby(level=0).mean()
    return s


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5-path", required=True, help="ERA5 file or directory (largest .nc chosen).")
    ap.add_argument("--year", type=int, default=2010)
    ap.add_argument("--month", type=int, default=None, help="Optional month (1-12) to subset.")
    ap.add_argument("--country", default=None, help="Optional ISO2 country to compute polygon-mean series (e.g. ES).")

    ap.add_argument("--resample", default=None, help="Optional resample frequency, e.g. 3h (recommended to mimic CORDEX).")
    ap.add_argument("--max-print-rows", type=int, default=10)

    args = ap.parse_args()

    ds = _open_era5_allvars(Path(args.era5_path))
    print("\n[DATASET] vars:", list(ds.data_vars)[:50], "..." if len(ds.data_vars) > 50 else "")
    print("[DATASET] coords:", list(ds.coords))

    # locate radiation variable
    try:
        vname, ssrd = _find_candidate_ssrd(ds)
        print(f"\n[FOUND] radiation candidate: {vname}")
    except Exception as e:
        print("\n[WARN] Could not find ssrd-like var:", e)
        print("       If this is a cutout, you may already have rsds (W/m^2).")
        if "rsds" in ds.data_vars:
            vname, ssrd = "rsds", ds["rsds"]
            print("[FOUND] using rsds directly.")
        else:
            raise

    print("\n[ATTRS] units:", ssrd.attrs.get("units"))
    print("[ATTRS] GRIB_stepType/stepType:", ssrd.attrs.get("GRIB_stepType", ssrd.attrs.get("stepType")))
    print("[ATTRS] accumulated hint:", _is_accumulated_hint(ssrd))

    # subset
    da = _subset_year_month(ssrd, year=int(args.year), month=args.month)
    print(f"\n[SUBSET] year={args.year} month={args.month} -> shape={tuple(da.shape)} dims={da.dims}")

    # time diagnostics
    t = _maybe_parse_time_axis(da)
    _describe_time_axis(t)

    # show some raw values
    try:
        s_raw = _to_series(da)
        print("\n[RAW] series head:")
        print(s_raw.head(args.max_print_rows))
        print("[RAW] min/mean/max:", float(np.nanmin(s_raw.values)), float(np.nanmean(s_raw.values)), float(np.nanmax(s_raw.values)))
    except Exception as e:
        print("\n[RAW] could not convert to series:", e)
        s_raw = pd.Series(dtype=float)

    # conversion if needed (if already rsds W/m^2, conversions are meaningless)
    if vname.lower() in {"rsds"} and (str(da.attrs.get("units", "")).lower().strip().startswith("w")):
        print("\n[INFO] Variable appears already to be flux (W/m^2). Skipping conversions.")
        rsds_med = None
        rsds_step = None
    else:
        # median dt
        try:
            rsds_med = _median_dt_conversion(da)
            s_med = _to_series(rsds_med)
            print("\n[CONV A] median-dt rsds head:")
            print(s_med.head(args.max_print_rows))
            print("[CONV A] min/mean/max:", float(np.nanmin(s_med.values)), float(np.nanmean(s_med.values)), float(np.nanmax(s_med.values)))
        except Exception as e:
            print("\n[CONV A] median-dt conversion failed:", e)
            rsds_med = None

        # per-step dt
        try:
            rsds_step = _per_step_dt_conversion(da)
            s_step = _to_series(rsds_step)
            print("\n[CONV B] per-step-dt rsds head:")
            print(s_step.head(args.max_print_rows))
            print("[CONV B] min/mean/max:", float(np.nanmin(s_step.values)), float(np.nanmean(s_step.values)), float(np.nanmax(s_step.values)))
        except Exception as e:
            print("\n[CONV B] per-step-dt conversion failed:", e)
            rsds_step = None

    # optional country mean comparison
    if args.country:
        cc = str(args.country).strip().upper()
        print(f"\n[COUNTRY] polygon mean for {cc}")
        if rsds_med is not None:
            s_c_med = country_mean_series(ds, rsds_med, cc)
            s_c_med = _resample_series(s_c_med, args.resample)
            print("[COUNTRY][A] median-dt rsds (resampled) min/mean/max:",
                  float(np.nanmin(s_c_med.values)) if len(s_c_med) else np.nan,
                  float(np.nanmean(s_c_med.values)) if len(s_c_med) else np.nan,
                  float(np.nanmax(s_c_med.values)) if len(s_c_med) else np.nan)
            print(s_c_med.head(args.max_print_rows))
        if rsds_step is not None:
            s_c_step = country_mean_series(ds, rsds_step, cc)
            s_c_step = _resample_series(s_c_step, args.resample)
            print("[COUNTRY][B] per-step-dt rsds (resampled) min/mean/max:",
                  float(np.nanmin(s_c_step.values)) if len(s_c_step) else np.nan,
                  float(np.nanmean(s_c_step.values)) if len(s_c_step) else np.nan,
                  float(np.nanmax(s_c_step.values)) if len(s_c_step) else np.nan)
            print(s_c_step.head(args.max_print_rows))

    # optional resample overview (global mean)
    if args.resample:
        f = str(args.resample).strip().replace("H", "h")
        print(f"\n[RESAMPLE] to {f} (global mean)")
        if not s_raw.empty:
            s_r = _resample_series(s_raw, f)
            print("[RAW resampled] min/mean/max:", float(np.nanmin(s_r.values)), float(np.nanmean(s_r.values)), float(np.nanmax(s_r.values)))
            print(s_r.head(args.max_print_rows))
        if rsds_med is not None:
            s_med = _to_series(rsds_med)
            s_med_r = _resample_series(s_med, f)
            print("[CONV A resampled] min/mean/max:", float(np.nanmin(s_med_r.values)), float(np.nanmean(s_med_r.values)), float(np.nanmax(s_med_r.values)))
            print(s_med_r.head(args.max_print_rows))
        if rsds_step is not None:
            s_step = _to_series(rsds_step)
            s_step_r = _resample_series(s_step, f)
            print("[CONV B resampled] min/mean/max:", float(np.nanmin(s_step_r.values)), float(np.nanmean(s_step_r.values)), float(np.nanmax(s_step_r.values)))
            print(s_step_r.head(args.max_print_rows))

    print("\n[DONE]")


if __name__ == "__main__":
    main()
