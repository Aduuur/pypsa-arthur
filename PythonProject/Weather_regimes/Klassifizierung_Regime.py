#%%
"""
STRICT GRAMS (2017) Weather Regime Classification — CORDEX Z500, 3-hourly, multi-year, RAM-stable
================================================================================================

Strict implementation of Methods described in Grams et al. (2017), adapted to 3-hourly CORDEX Z500:

Core methodological elements:
1) Domain: 80W..40E, 30N..90N (mask by lat/lon).
2) Reference climatology: "90-day running mean at the respective calendar time as reference climatology"
   -> climatological mean by calendar-time (doy_noleap x intraday slot), then 90-day running mean on the calendar axis (cyclic).
3) Anomaly: z_anom(t) = z(t) - baseline(calendar_time(t)).
4) 10-day low-pass: centered running mean on time axis (computed with overlap per chunk).
5) Remove seasonal amplitude cycle via calendar-time 30-day running std of low-pass anomalies, per grid point,
   then normalize by spatial mean of that std in EOF domain:
   field(t) = anom_lp(t) / mean_space(std30_calendar(calendar_time(t))).
6) EOF/PCA and k-means:
   - 7 EOFs/PCs, as in paper
   - k-means with n_init=10 (repeat 10 times)
7) Regime index Iwr:
   Iwr(t,r) = dot(PC(t), C_r) / ||C_r||  in PC space
   sigma(Iwr) computed over configurable sigma period (default = all training years).
8) Life cycles:
   time steps attributed to regime life cycles if:
     Iwr > sigma(Iwr), duration >= 5 days, contains local maximum with monotonic flanks +/- 5 days
   subsequent life cycles of same regime merged if mean Iwr over joint interval > sigma(Iwr)
   if multiple regimes fulfill: assign to max Iwr.

Performance / RAM stability:
- Year-by-year, time-chunked streaming.
- Disk caching via Zarr (baseline climatology, sigma_spatial_mean, preprocessed per-year fields).


Requirements:
- xarray + zarr + numcodecs installed (for caching).
"""

from __future__ import annotations

import os
import gc
import glob
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.array as da

from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans


# =============================================================================
# Runtime stability knobs (no methodological impact)
# =============================================================================
DASK_SCHEDULER = os.environ.get("DASK_SCHEDULER", "threads")  # "threads" | "single-threaded"
dask.config.set(scheduler=DASK_SCHEDULER)

# Prevent BLAS thread explosions
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# =============================================================================
# Dependency sanity checks (Zarr backend)
# =============================================================================
def assert_zarr_available() -> None:
    engines = xr.backends.list_engines()
    if "zarr" not in engines:
        raise RuntimeError(
            "Xarray Zarr backend not available (engine='zarr' missing). "
            "Install dependencies in your current environment, e.g.\n"
            "  conda install -c conda-forge zarr numcodecs\n"
            "or\n"
            "  pip install zarr numcodecs\n"
            f"Available engines: {list(engines.keys())}"
        )


# =============================================================================
# Helpers
# =============================================================================
def discover_yearly_files(
    base_dir: str,
    years: Iterable[int],
    filename_glob: str = "zg500_*.nc",
) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for y in years:
        folder = os.path.join(base_dir, f"cordex_{y}")
        files = sorted(glob.glob(os.path.join(folder, filename_glob)))
        if not files:
            raise FileNotFoundError(f"No files found for year {y} in {folder} with pattern {filename_glob}")
        out[y] = files
    return out


def to_lon180(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0

def write_excel_timeseries_long_to_short(
    excel_path: str,
    dfs: List[Tuple[str, pd.DataFrame]],
) -> None:
    """
    Write multiple DataFrames into one Excel file, ordering sheets from long to short (by number of rows).
    Ensures Excel sheet name length <= 31.
    """
    # sort long -> short
    dfs_sorted = sorted(dfs, key=lambda x: len(x[1]), reverse=True)

    def safe_sheet_name(name: str) -> str:
        name = name.replace(":", "_").replace("/", "_").replace("\\", "_").strip()
        return name[:31] if len(name) > 31 else name

    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, df in dfs_sorted:
            sn = safe_sheet_name(sheet_name)

            # If it looks like a time series (DatetimeIndex), write index as a column named "time"
            if isinstance(df.index, pd.DatetimeIndex):
                out = df.copy()
                out.insert(0, "time", out.index)
                out.reset_index(drop=True, inplace=True)
                out.to_excel(writer, sheet_name=sn, index=False)
            else:
                df.to_excel(writer, sheet_name=sn, index=False)


def infer_time_step_hours(time_index: pd.DatetimeIndex) -> float:
    diffs = pd.Series(time_index).diff().dropna()
    dt = diffs.mode().iloc[0]
    return float(dt / pd.Timedelta(hours=1))


def compute_domain_mask_from_latlon(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    return (
        (lat2d >= lat_min) & (lat2d <= lat_max) &
        (lon2d >= lon_min) & (lon2d <= lon_max)
    )


def window_steps(days: int, steps_per_day: int, require_odd: bool = True) -> int:
    w = int(days * steps_per_day)
    if require_odd and (w % 2 == 0):
        w += 1
    return w


def drop_feb29(t: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask: keep all timestamps except Feb 29."""
    return ~((t.month == 2) & (t.day == 29))


def doy_noleap(month: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Map (month, day) to day-of-year 1..365 in a no-leap calendar."""
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int32)
    return cum[month - 1] + day


def centered_running_mean_block(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Centered running mean along axis=0 for a block that already includes the needed overlap.
    Returns same shape as input, with NaNs at edges where centered window does not fit.
    """
    n = arr.shape[0]
    h = window // 2
    out = np.full_like(arr, np.nan, dtype=np.float32)

    if n < window:
        return out

    c = np.cumsum(np.vstack([np.zeros((1, arr.shape[1]), dtype=np.float64), arr.astype(np.float64)]), axis=0)
    sums = c[window:] - c[:-window]  # (n-window+1, S)

    centers = sums / float(window)
    out[h:n - h, :] = centers.astype(np.float32)
    return out


def find_true_runs(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    n = len(mask_1d)
    i = 0
    while i < n:
        if not mask_1d[i]:
            i += 1
            continue
        j = i
        while j < n and mask_1d[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def is_lifecycle_segment_truncated(
    s: np.ndarray,
    start: int,
    end: int,
    flank_steps: int,
    min_inner_steps: int = 1,   # Peak muss mind. 1 Schritt links+rechts haben
) -> bool:

    if end - start < 3:
        return False

    seg = s[start:end]
    if not np.isfinite(seg).all():
        return False

    # Peak: lokales Maximum (>= Nachbarn)
    # Wir verlangen einen Peak, der nicht direkt am Rand liegt (min_inner_steps).
    L = end - start
    p_candidates = []
    for p in range(min_inner_steps, L - min_inner_steps):
        if seg[p] >= seg[p - 1] and seg[p] >= seg[p + 1]:
            p_candidates.append(p)
    if not p_candidates:
        return False

    # Nimm stärksten Peak
    p_star = max(p_candidates, key=lambda p: seg[p])

    # Flankenlänge trunkiert auf Segmentränder
    left_len = min(flank_steps, p_star)               # max so weit wie verfügbar
    right_len = min(flank_steps, (L - 1) - p_star)

    # Optional:
    # if left_len < flank_steps or right_len < flank_steps: return False
    # Aber das würde wieder 10-Tage-Effekte erzeugen. Daher standardmäßig NICHT.

    left = seg[p_star - left_len : p_star + 1]
    right = seg[p_star : p_star + right_len + 1]

    # Monoton: links nicht fallend, rechts nicht steigend
    if np.any(np.diff(left) < 0):
        return False
    if np.any(np.diff(right) > 0):
        return False

    return True



def merge_cycles_if_mean_above_threshold(
    cycles: List[Tuple[int, int]],
    series: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int]]:
    if not cycles:
        return cycles

    cycles = sorted(cycles, key=lambda x: x[0])
    csum = np.concatenate([[0.0], np.cumsum(series.astype(np.float64))])

    def mean_over(a: int, b: int) -> float:
        if b <= a:
            return -np.inf
        return float((csum[b] - csum[a]) / (b - a))

    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = cycles[0]

    for s, e in cycles[1:]:
        prop_s, prop_e = cur_s, e
        if mean_over(prop_s, prop_e) > threshold:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))
    return merged


@dataclass
class CalendarGeometry:
    steps_per_day: int
    dt_hours: float
    cal_size: int  # 365 * steps_per_day


# =============================================================================
# Main pipeline (strict Grams)
# =============================================================================
class GramsStrictCordexRegimePipeline:
    def __init__(
        self,
        base_dir: str,
        years_train: List[int],
        years_classify: Optional[List[int]] = None,
        years_sigma: Optional[List[int]] = None,  # sigma(Iwr) period; if None -> years_train
        filename_glob: str = "zg500_*.nc",
        z_var: str = "zg500",
        time_dim: str = "time",
        y_dim: str = "y",
        x_dim: str = "x",
        lat_coord: str = "lat",
        lon_coord: str = "lon",
        # Grams domain
        domain_lat_min: float = 30.0,
        domain_lat_max: float = 90.0,
        domain_lon_min: float = -80.0,
        domain_lon_max: float = 40.0,
        # performance knob (set to 1 for max fidelity)
        coarsen_factor: int = 2,
        # streaming chunk size (time steps)
        time_chunk_steps: int = 1024,
        # method parameters
        climatology_days: int = 90,
        std_calendar_days: int = 30,
        lowpass_days: int = 10,
        n_pcs: int = 7,
        n_clusters: int = 7,
        kmeans_n_init: int = 10,
        random_state: int = 42,
        # lifecycle parameters
        min_persistence_days: int = 5,
        flank_days: int = 5,
        eps: float = 1e-12,
        # caching
        cache_dir: Optional[str] = None,
        cache_overwrite: bool = False,
    ):
        self.base_dir = base_dir
        self.years_train = years_train
        self.years_classify = years_classify if years_classify is not None else years_train
        self.years_sigma = years_sigma if years_sigma is not None else years_train

        self.filename_glob = filename_glob
        self.z_var = z_var
        self.time_dim = time_dim
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.lat_coord = lat_coord
        self.lon_coord = lon_coord

        self.domain_lat_min = domain_lat_min
        self.domain_lat_max = domain_lat_max
        self.domain_lon_min = domain_lon_min
        self.domain_lon_max = domain_lon_max

        self.coarsen_factor = int(coarsen_factor)
        self.time_chunk_steps = int(time_chunk_steps)

        self.climatology_days = int(climatology_days)
        self.std_calendar_days = int(std_calendar_days)
        self.lowpass_days = int(lowpass_days)

        self.n_pcs = int(n_pcs)
        self.n_clusters = int(n_clusters)
        self.kmeans_n_init = int(kmeans_n_init)
        self.random_state = int(random_state)

        self.min_persistence_days = int(min_persistence_days)
        self.flank_days = int(flank_days)
        self.eps = float(eps)

        self.cache_dir = cache_dir
        self.cache_overwrite = bool(cache_overwrite)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # derived / learned
        self.calgeom_: Optional[CalendarGeometry] = None
        self.domain_mask_flat_: Optional[np.ndarray] = None
        self.space_indices_: Optional[np.ndarray] = None
        self.space_size_: Optional[int] = None

        self.baseline_zarr_: Optional[str] = None
        self.sigma_spatial_mean_path_: Optional[str] = None

        self.ipca_: Optional[IncrementalPCA] = None
        self.kmeans_: Optional[MiniBatchKMeans] = None
        self.iwr_sigma_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _year_files(self, year: int) -> List[str]:
        return discover_yearly_files(self.base_dir, [year], filename_glob=self.filename_glob)[year]

    def _open_year(self, year: int) -> xr.Dataset:
        files = self._year_files(year)
        return xr.open_mfdataset(files, engine="netcdf4", combine="by_coords", chunks="auto")

    def _cache_path(self, name: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, name)

    def _cache_preprocessed_year_path(self, year: int) -> Optional[str]:
        return self._cache_path(f"preproc_field_{year}.zarr")

    def _cache_baseline_path(self) -> Optional[str]:
        return self._cache_path("baseline_climatology_90d.zarr")

    def _cache_sigma_spatial_mean_path(self) -> Optional[str]:
        return self._cache_path("sigma30_spatial_mean.npy")

    # ------------------------------------------------------------------
    # Calendar index
    # ------------------------------------------------------------------
    def _calendar_index(self, t: pd.DatetimeIndex) -> np.ndarray:
        if self.calgeom_ is None:
            raise RuntimeError("Calendar geometry not initialized.")
        steps_per_day = self.calgeom_.steps_per_day

        doy = doy_noleap(t.month.values.astype(np.int32), t.day.values.astype(np.int32))  # 1..365
        minutes = (t.hour.values.astype(np.int32) * 60 + t.minute.values.astype(np.int32))
        slot = np.round(minutes / (self.calgeom_.dt_hours * 60.0)).astype(np.int32) % steps_per_day
        cal_idx = (doy - 1) * steps_per_day + slot
        return cal_idx.astype(np.int32)

    # ------------------------------------------------------------------
    # Initialize (domain + timestep)
    # ------------------------------------------------------------------
    def initialize(self) -> "GramsStrictCordexRegimePipeline":
        assert_zarr_available()

        ds0 = self._open_year(self.years_train[0])
        try:
            t0 = pd.to_datetime(ds0[self.time_dim].values)
            dt_hours = infer_time_step_hours(t0)
            steps_per_day = int(round(24.0 / dt_hours))
            cal_size = 365 * steps_per_day
            self.calgeom_ = CalendarGeometry(steps_per_day=steps_per_day, dt_hours=dt_hours, cal_size=cal_size)

            lat2d = ds0[self.lat_coord].values
            lon2d = to_lon180(ds0[self.lon_coord].values)

            if self.coarsen_factor > 1:
                lat2d = xr.DataArray(lat2d, dims=(self.y_dim, self.x_dim)).coarsen(
                    {self.y_dim: self.coarsen_factor, self.x_dim: self.coarsen_factor}, boundary="trim"
                ).mean().values
                lon2d = xr.DataArray(lon2d, dims=(self.y_dim, self.x_dim)).coarsen(
                    {self.y_dim: self.coarsen_factor, self.x_dim: self.coarsen_factor}, boundary="trim"
                ).mean().values

            dom = compute_domain_mask_from_latlon(
                lat2d, lon2d,
                lat_min=self.domain_lat_min, lat_max=self.domain_lat_max,
                lon_min=self.domain_lon_min, lon_max=self.domain_lon_max,
            )
            mask_flat = dom.ravel().astype(bool)
            space_idx = np.where(mask_flat)[0].astype(np.int64)

            if space_idx.size < 1000:
                raise RuntimeError(f"Domain mask keeps too few points ({space_idx.size}). Check lat/lon bounds.")

            self.domain_mask_flat_ = mask_flat
            self.space_indices_ = space_idx
            self.space_size_ = int(space_idx.size)

        finally:
            ds0.close()
            del ds0
            gc.collect()

        return self

    # ------------------------------------------------------------------
    # Read raw Z500 slice as numpy (B, space_domain)
    # ------------------------------------------------------------------
    def _read_z_slice(self, ds: xr.Dataset, t0: int, t1: int) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        if self.space_indices_ is None:
            raise RuntimeError("initialize() not called.")

        z = ds[self.z_var].isel({self.time_dim: slice(t0, t1)}).astype(np.float32)

        if self.coarsen_factor > 1:
            z = z.coarsen({self.y_dim: self.coarsen_factor, self.x_dim: self.coarsen_factor}, boundary="trim").mean()
            z = z.astype(np.float32)

        times = pd.to_datetime(z[self.time_dim].values)
        keep = drop_feb29(times)
        z = z.isel({self.time_dim: keep})
        times = times[keep]

        arr = da.asarray(z.data).compute().astype(np.float32)  # (B, y, x)
        B = arr.shape[0]
        X = arr.reshape(B, -1)  # (B, y*x)
        X = X[:, self.space_indices_]
        return X, times

    # ------------------------------------------------------------------
    # 1) Baseline climatology: mean by calendar time, then 90-day running mean (cyclic)
    # ------------------------------------------------------------------
    def compute_baseline_climatology(self) -> "GramsStrictCordexRegimePipeline":
        if self.calgeom_ is None:
            self.initialize()

        assert self.calgeom_ is not None
        assert self.space_size_ is not None

        baseline_path = self._cache_baseline_path()
        if baseline_path and (not self.cache_overwrite) and os.path.exists(baseline_path):
            self.baseline_zarr_ = baseline_path
            print(f"[CACHE] Using existing baseline climatology: {baseline_path}")
            return self

        if not baseline_path:
            raise RuntimeError("cache_dir must be set to store baseline climatology (required for this pipeline).")

        cal_size = self.calgeom_.cal_size
        S = self.space_size_

        sum_z = np.zeros((cal_size, S), dtype=np.float32)
        cnt = np.zeros((cal_size,), dtype=np.int64)

        print("[STEP] Computing baseline climatology (calendar mean + 90-day running mean)...")
        for year in self.years_train:
            print(f"  - accumulating year {year}")
            ds = self._open_year(year)
            try:
                nT = ds.sizes[self.time_dim]
                t = 0
                while t < nT:
                    t1 = min(nT, t + self.time_chunk_steps)
                    X, times = self._read_z_slice(ds, t, t1)
                    if X.size == 0:
                        t = t1
                        continue

                    cal_idx = self._calendar_index(times)

                    order = np.argsort(cal_idx)
                    cal_sorted = cal_idx[order]
                    Xs = X[order, :]

                    g_start = 0
                    while g_start < len(cal_sorted):
                        g_cal = int(cal_sorted[g_start])
                        g_end = g_start + 1
                        while g_end < len(cal_sorted) and int(cal_sorted[g_end]) == g_cal:
                            g_end += 1

                        sum_z[g_cal, :] += Xs[g_start:g_end, :].sum(axis=0, dtype=np.float64).astype(np.float32)
                        cnt[g_cal] += (g_end - g_start)
                        g_start = g_end

                    t = t1
            finally:
                ds.close()
                del ds
                gc.collect()

        mean_clim = np.full((cal_size, S), np.nan, dtype=np.float32)
        valid = cnt > 0
        mean_clim[valid, :] = (sum_z[valid, :] / cnt[valid, None]).astype(np.float32)

        W = window_steps(self.climatology_days, self.calgeom_.steps_per_day, require_odd=True)
        h = W // 2

        baseline = np.full_like(mean_clim, np.nan, dtype=np.float32)

        # space-chunked smoothing to reduce peak RAM
        space_block = 4096
        print("[STEP] Smoothing baseline climatology with 90-day running mean on calendar axis (cyclic)...")
        for s0 in range(0, S, space_block):
            s1 = min(S, s0 + space_block)
            M = mean_clim[:, s0:s1]

            pad = np.concatenate([M[-h:, :], M, M[:h, :]], axis=0)
            w_ok = np.isfinite(pad).astype(np.float32)
            pad0 = np.nan_to_num(pad, nan=0.0).astype(np.float64)

            cs = np.cumsum(np.vstack([np.zeros((1, pad0.shape[1]), dtype=np.float64), pad0]), axis=0)
            cw = np.cumsum(np.vstack([np.zeros((1, w_ok.shape[1]), dtype=np.float64), w_ok.astype(np.float64)]), axis=0)

            sums = cs[W:] - cs[:-W]
            wsum = cw[W:] - cw[:-W]

            sums_main = sums[: cal_size, :]
            wsum_main = wsum[: cal_size, :]

            out_block = (sums_main / np.maximum(wsum_main, 1.0)).astype(np.float32)
            out_block[wsum_main == 0] = np.nan

            baseline[:, s0:s1] = out_block

        ds_out = xr.Dataset(
            {"baseline": (("cal", "space"), baseline.astype(np.float32))},
            coords={"cal": np.arange(cal_size, dtype=np.int32), "space": np.arange(S, dtype=np.int32)},
        )
        ds_out.to_zarr(baseline_path, mode="w")

        self.baseline_zarr_ = baseline_path
        print(f"[OK] Baseline climatology written to: {baseline_path}")
        return self

    # ------------------------------------------------------------------
    # 2) sigma30(calendar time) spatial mean
    # ------------------------------------------------------------------
    def compute_sigma30_spatial_mean(self) -> "GramsStrictCordexRegimePipeline":
        if self.calgeom_ is None:
            self.initialize()
        if self.baseline_zarr_ is None:
            self.compute_baseline_climatology()

        assert self.calgeom_ is not None
        assert self.space_size_ is not None
        assert self.baseline_zarr_ is not None

        sigma_path = self._cache_sigma_spatial_mean_path()
        if sigma_path and (not self.cache_overwrite) and os.path.exists(sigma_path):
            self.sigma_spatial_mean_path_ = sigma_path
            print(f"[CACHE] Using existing sigma_spatial_mean: {sigma_path}")
            return self

        if not sigma_path:
            raise RuntimeError("cache_dir must be set to store sigma_spatial_mean (required for this pipeline).")

        cal_size = self.calgeom_.cal_size
        S = self.space_size_

        base_ds = xr.open_zarr(self.baseline_zarr_)
        baseline = base_ds["baseline"]  # (cal, space)

        lp_w = window_steps(self.lowpass_days, self.calgeom_.steps_per_day, require_odd=True)
        lp_h = lp_w // 2

        sum_a = np.zeros((cal_size, S), dtype=np.float32)
        sumsq_a = np.zeros((cal_size, S), dtype=np.float32)
        cnt = np.zeros((cal_size,), dtype=np.int64)

        print("[STEP] Computing sigma30 spatial mean (calendar-binned moments + 30-day running window)...")
        for year in self.years_train:
            print(f"  - accumulating low-pass anomaly moments for year {year}")
            ds = self._open_year(year)
            try:
                nT = ds.sizes[self.time_dim]
                t = 0
                while t < nT:
                    t0_ext = max(0, t - lp_h)
                    t1_ext = min(nT, t + self.time_chunk_steps + lp_h)

                    X_ext, times_ext = self._read_z_slice(ds, t0_ext, t1_ext)
                    if X_ext.size == 0:
                        t += self.time_chunk_steps
                        continue

                    cal_ext = self._calendar_index(times_ext)

                    ucal, inv = np.unique(cal_ext, return_inverse=True)
                    base_u = baseline.isel(cal=xr.DataArray(ucal, dims=("u",))).values.astype(np.float32)
                    base_ext = base_u[inv, :]

                    anom_ext = X_ext - base_ext
                    anom_lp_ext = centered_running_mean_block(anom_ext, lp_w)

                    i0 = t - t0_ext
                    i1 = min(t + self.time_chunk_steps, nT) - t0_ext

                    anom_lp = anom_lp_ext[i0:i1, :]
                    cal_blk = cal_ext[i0:i1]

                    valid_rows = np.isfinite(anom_lp).all(axis=1)
                    if not np.any(valid_rows):
                        t += self.time_chunk_steps
                        continue

                    A = anom_lp[valid_rows, :].astype(np.float32)
                    cal_v = cal_blk[valid_rows]

                    order = np.argsort(cal_v)
                    cal_sorted = cal_v[order]
                    As = A[order, :]

                    g_start = 0
                    while g_start < len(cal_sorted):
                        g_cal = int(cal_sorted[g_start])
                        g_end = g_start + 1
                        while g_end < len(cal_sorted) and int(cal_sorted[g_end]) == g_cal:
                            g_end += 1

                        block = As[g_start:g_end, :]
                        sum_a[g_cal, :] += block.sum(axis=0, dtype=np.float64).astype(np.float32)
                        sumsq_a[g_cal, :] += (block.astype(np.float64) ** 2).sum(axis=0).astype(np.float32)
                        cnt[g_cal] += (g_end - g_start)

                        g_start = g_end

                    t += self.time_chunk_steps
            finally:
                ds.close()
                del ds
                gc.collect()

        base_ds.close()
        del base_ds
        gc.collect()

        W = window_steps(self.std_calendar_days, self.calgeom_.steps_per_day, require_odd=True)
        h = W // 2

        def roll_sum_cyclic(mat: np.ndarray, win: int) -> np.ndarray:
            pad = np.concatenate([mat[-h:, :], mat, mat[:h, :]], axis=0)
            cs = np.cumsum(np.vstack([np.zeros((1, pad.shape[1]), dtype=np.float64), pad.astype(np.float64)]), axis=0)
            sums = cs[win:] - cs[:-win]
            return sums[: cal_size, :].astype(np.float64)

        cnt_pad = np.concatenate([cnt[-h:], cnt, cnt[:h]]).astype(np.float64)
        cc = np.cumsum(np.concatenate([[0.0], cnt_pad]))
        cnt_w = (cc[W:] - cc[:-W])[:cal_size]  # (cal,)

        sum_w = roll_sum_cyclic(sum_a, W)
        sumsq_w = roll_sum_cyclic(sumsq_a, W)

        cnt_w2 = np.maximum(cnt_w, 1.0)[:, None]
        mean_w = sum_w / cnt_w2
        ex2 = sumsq_w / cnt_w2
        var = np.maximum(ex2 - mean_w ** 2, 0.0)
        std_grid = np.sqrt(var).astype(np.float32)  # (cal, S)

        sigma_spatial_mean = np.nanmean(std_grid, axis=1).astype(np.float32)
        sigma_spatial_mean = np.where(sigma_spatial_mean <= 0, 1.0, sigma_spatial_mean).astype(np.float32)

        np.save(sigma_path, sigma_spatial_mean)
        self.sigma_spatial_mean_path_ = sigma_path
        print(f"[OK] sigma_spatial_mean saved to: {sigma_path}")
        return self

    # ------------------------------------------------------------------
    # 3) Preprocess year -> cache (time, space): field = anom_lp / sigma_spatial_mean(calendar_time)
    # ------------------------------------------------------------------
    def build_preprocessed_year_cache(self, year: int) -> str:
        if self.calgeom_ is None:
            self.initialize()
        if self.baseline_zarr_ is None:
            self.compute_baseline_climatology()
        if self.sigma_spatial_mean_path_ is None:
            self.compute_sigma30_spatial_mean()

        assert self.calgeom_ is not None
        assert self.space_size_ is not None
        assert self.baseline_zarr_ is not None
        assert self.sigma_spatial_mean_path_ is not None

        out_path = self._cache_preprocessed_year_path(year)
        if out_path and (not self.cache_overwrite) and os.path.exists(out_path):
            return out_path
        if not out_path:
            raise RuntimeError("cache_dir must be set to store preprocessed year fields (required).")

        sigma_spatial_mean = np.load(self.sigma_spatial_mean_path_).astype(np.float32)
        base_ds = xr.open_zarr(self.baseline_zarr_)
        baseline = base_ds["baseline"]

        lp_w = window_steps(self.lowpass_days, self.calgeom_.steps_per_day, require_odd=True)
        lp_h = lp_w // 2

        ds = self._open_year(year)
        try:
            nT_full = ds.sizes[self.time_dim]
            out_times: List[np.ndarray] = []
            out_blocks: List[np.ndarray] = []

            t = 0
            while t < nT_full:
                t0_ext = max(0, t - lp_h)
                t1_ext = min(nT_full, t + self.time_chunk_steps + lp_h)

                X_ext, times_ext = self._read_z_slice(ds, t0_ext, t1_ext)
                if X_ext.size == 0:
                    t += self.time_chunk_steps
                    continue

                cal_ext = self._calendar_index(times_ext)

                ucal, inv = np.unique(cal_ext, return_inverse=True)
                base_u = baseline.isel(cal=xr.DataArray(ucal, dims=("u",))).values.astype(np.float32)
                base_ext = base_u[inv, :]
                anom_ext = X_ext - base_ext

                anom_lp_ext = centered_running_mean_block(anom_ext, lp_w)

                i0 = t - t0_ext
                i1 = min(t + self.time_chunk_steps, nT_full) - t0_ext

                anom_lp = anom_lp_ext[i0:i1, :]
                times_blk = times_ext[i0:i1]
                cal_blk = cal_ext[i0:i1]

                valid_rows = np.isfinite(anom_lp).all(axis=1)
                if np.any(valid_rows):
                    A = anom_lp[valid_rows, :].astype(np.float32)
                    cal_v = cal_blk[valid_rows]
                    sig = sigma_spatial_mean[cal_v].astype(np.float32)

                    field_norm = (A / (sig[:, None] + self.eps)).astype(np.float32)

                    out_times.append(times_blk[valid_rows].values)
                    out_blocks.append(field_norm)

                t += self.time_chunk_steps

            if out_blocks:
                times_all = np.concatenate(out_times)
                data_all = np.concatenate(out_blocks, axis=0)  # (T_valid, S)

                ds_out = xr.Dataset(
                    {"field": (("time", "space"), data_all)},
                    coords={"time": pd.to_datetime(times_all), "space": np.arange(self.space_size_, dtype=np.int32)},
                )
                ds_out.to_zarr(out_path, mode="w")
            else:
                ds_out = xr.Dataset(
                    {"field": (("time", "space"), np.zeros((0, self.space_size_), dtype=np.float32))},
                    coords={"time": [], "space": np.arange(self.space_size_, dtype=np.int32)},
                )
                ds_out.to_zarr(out_path, mode="w")

            return out_path

        finally:
            ds.close()
            base_ds.close()
            del ds, base_ds
            gc.collect()

    # ------------------------------------------------------------------
    # 4) Fit EOF/PCA and KMeans + compute sigma(Iwr)
    # ------------------------------------------------------------------
    def fit(self) -> "GramsStrictCordexRegimePipeline":
        if self.calgeom_ is None:
            self.initialize()
        if self.baseline_zarr_ is None:
            self.compute_baseline_climatology()
        if self.sigma_spatial_mean_path_ is None:
            self.compute_sigma30_spatial_mean()

        # Build caches for training years
        print("[STEP] Building / loading preprocessed year caches for training years...")
        train_paths = []
        for y in self.years_train:
            print(f"  - preproc cache for {y}")
            train_paths.append(self.build_preprocessed_year_cache(y))

        print("[STEP] Fitting IncrementalPCA (EOFs)...")
        ipca = IncrementalPCA(n_components=self.n_pcs, batch_size=self.time_chunk_steps)
        for path in train_paths:
            ds = xr.open_zarr(path)
            try:
                X = ds["field"]
                nT = X.sizes["time"]
                t = 0
                while t < nT:
                    t1 = min(nT, t + self.time_chunk_steps)
                    block = X.isel(time=slice(t, t1)).values.astype(np.float32)
                    if block.shape[0] >= 2:
                        ipca.partial_fit(block)
                    t = t1
            finally:
                ds.close()
                del ds
                gc.collect()

        self.ipca_ = ipca

        print("[STEP] Fitting MiniBatchKMeans in PC space...")
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=self.time_chunk_steps,
            init_size=max(4096, self.time_chunk_steps),
            n_init=self.kmeans_n_init,
        )

        for path in train_paths:
            ds = xr.open_zarr(path)
            try:
                X = ds["field"]
                nT = X.sizes["time"]
                t = 0
                while t < nT:
                    t1 = min(nT, t + self.time_chunk_steps)
                    block = X.isel(time=slice(t, t1)).values.astype(np.float32)
                    if block.shape[0] >= 2:
                        PCs = self.ipca_.transform(block).astype(np.float32)
                        kmeans.partial_fit(PCs)
                    t = t1
            finally:
                ds.close()
                del ds
                gc.collect()

        self.kmeans_ = kmeans

        print("[STEP] Computing sigma(Iwr) over configured sigma years...")
        self.compute_iwr_sigma()
        print("[OK] Fit complete.")
        return self

    # ------------------------------------------------------------------
    # 5) Iwr and sigma(Iwr)
    # ------------------------------------------------------------------
    def _compute_iwr_for_path(self, path: str) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        if self.ipca_ is None or self.kmeans_ is None:
            raise RuntimeError("fit() not done.")

        ds = xr.open_zarr(path)
        try:
            X = ds["field"]
            times = pd.to_datetime(ds["time"].values)
            nT = X.sizes["time"]
            k = self.n_clusters
            C = self.kmeans_.cluster_centers_.astype(np.float64)
            denom = np.linalg.norm(C, axis=1) + self.eps

            I = np.full((nT, k), np.nan, dtype=np.float32)

            t = 0
            while t < nT:
                t1 = min(nT, t + self.time_chunk_steps)
                block = X.isel(time=slice(t, t1)).values.astype(np.float32)
                if block.shape[0] >= 1:
                    PCs = self.ipca_.transform(block).astype(np.float64)
                    Iwr = (PCs @ C.T) / denom[None, :]
                    I[t:t1, :] = Iwr.astype(np.float32)
                t = t1

            return I, times
        finally:
            ds.close()
            del ds
            gc.collect()

    def compute_iwr_sigma(self) -> "GramsStrictCordexRegimePipeline":
        paths = [self.build_preprocessed_year_cache(y) for y in self.years_sigma]

        k = self.n_clusters
        n = 0
        mean = np.zeros((k,), dtype=np.float64)
        m2 = np.zeros((k,), dtype=np.float64)

        for y, path in zip(self.years_sigma, paths):
            I, _ = self._compute_iwr_for_path(path)
            I = I.astype(np.float64)
            good = np.isfinite(I).all(axis=1)
            I = I[good, :]
            for row in I:
                n += 1
                delta = row - mean
                mean += delta / n
                delta2 = row - mean
                m2 += delta * delta2

            print(f"  - sigma(Iwr) accumulation done for {y}")

        if n < 2:
            raise RuntimeError("Insufficient data to compute sigma(Iwr).")

        var = m2 / (n - 1)
        std = np.sqrt(np.maximum(var, 0.0))
        std = np.where(std <= 0, 1.0, std)

        self.iwr_sigma_ = std.astype(np.float64)
        return self

    # ------------------------------------------------------------------
    # 6) Classification with strict life-cycle logic
    # ------------------------------------------------------------------
    def classify_year(self, year: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.iwr_sigma_ is None:
            raise RuntimeError("fit() must be run first (needs sigma(Iwr)).")
        if self.calgeom_ is None:
            raise RuntimeError("initialize() must be run first.")

        path = self.build_preprocessed_year_cache(year)
        I, times = self._compute_iwr_for_path(path)

        steps_per_day = self.calgeom_.steps_per_day
        min_len_steps = self.min_persistence_days * steps_per_day
        flank_steps = self.flank_days * steps_per_day

        T, k = I.shape
        no_id = k

        active = np.zeros((T, k), dtype=bool)

        for r in range(k):
            s = I[:, r].astype(np.float64)
            thr = float(self.iwr_sigma_[r])

            ok = np.isfinite(s) & (s > thr)
            runs = find_true_runs(ok)

            good_cycles: List[Tuple[int, int]] = []
            for a, b in runs:
                if (b - a) < min_len_steps:
                    continue
                if is_lifecycle_segment_truncated(s, a, b, flank_steps=flank_steps, min_inner_steps=1):
                    good_cycles.append((a, b))

            good_cycles = merge_cycles_if_mean_above_threshold(good_cycles, s, threshold=thr)

            for a, b in good_cycles:
                active[a:b, r] = True

        labels = np.full((T,), no_id, dtype=np.int16)
        any_active = active.any(axis=1)
        if np.any(any_active):
            I_active = np.where(active, I, -np.inf)
            winner = np.argmax(I_active, axis=1).astype(np.int16)
            labels[any_active] = winner[any_active]

        ts = pd.Series(labels.astype(int), index=pd.to_datetime(times), name="regime")
        ts_df = ts.to_frame()
        ts_df["regime_label"] = ts_df["regime"].astype(int)

        # periods
        label_break = ts.ne(ts.shift())
        expected_dt = pd.Timedelta(hours=self.calgeom_.dt_hours)
        gap_break = ts.index.to_series().diff().gt(1.5 * expected_dt)
        block_id = (label_break | gap_break).cumsum()

        first_val = ts.groupby(block_id).first()
        start = ts.groupby(block_id).apply(lambda x: x.index[0])
        end = ts.groupby(block_id).apply(lambda x: x.index[-1])
        length = ts.groupby(block_id).size()

        periods_df = pd.DataFrame(
            {
                "regime": first_val.values.astype(int),
                "start": pd.to_datetime(start.values),
                "end": pd.to_datetime(end.values),
                "length_steps": length.values.astype(int),
            }
        )

        dist = ts.value_counts().sort_index()
        total = int(dist.sum())
        dist_df = pd.DataFrame({
            "regime": dist.index.astype(int),
            "steps": dist.values.astype(int),
            "share_percent": 100.0 * dist.values / max(total, 1),
            "regime_type": ["No Regime" if int(rr) == no_id else f"Regime {int(rr)}" for rr in dist.index],
        })

          # Iwr as DataFrame (time series)
        iwr_cols = [f"Iwr_{r}" for r in range(k)]
        iwr_df = pd.DataFrame(I, index=pd.to_datetime(times), columns=iwr_cols)

        return ts_df, periods_df, dist_df, iwr_df



# =============================================================================
# RUN: 2025..2050 (inclusive) training + classification
# =============================================================================
if __name__ == "__main__":
    BASE = "/mnt/endata/Cordex/Copernicus_CORDEX_data_RCP_8.5"

    # 2025..2050 inclusive
    YEARS_ALL = list(range(2025, 2051))

    MIN_PERSIST_DAYS = 5
    N_CLUSTERS = 8

    RUN_TAG = f"p{MIN_PERSIST_DAYS}d_k{N_CLUSTERS}"


    CACHE_DIR = os.path.join(BASE, f"_cache_grams_strict_{RUN_TAG}_{YEARS_ALL[0]}_{YEARS_ALL[-1]}")
    OUT_DIR   = os.path.join(BASE, f"results_grams_strict_{RUN_TAG}_{YEARS_ALL[0]}_{YEARS_ALL[-1]}")
    os.makedirs(OUT_DIR, exist_ok=True)

    pipe = GramsStrictCordexRegimePipeline(
        base_dir=BASE,
        years_train=YEARS_ALL,
        years_classify=YEARS_ALL,
        years_sigma=YEARS_ALL,
        filename_glob="zg500_*.nc",
        z_var="zg500",
        domain_lat_min=30.0,
        domain_lat_max=90.0,
        domain_lon_min=-80.0,
        domain_lon_max=40.0,
        # performance knobs (do not change the paper logic)
        coarsen_factor=2,           # set 1 for max fidelity
        time_chunk_steps=1024,
        # strict method params
        climatology_days=90,
        std_calendar_days=30,
        lowpass_days=10,
        n_pcs=7,
        n_clusters=N_CLUSTERS,
        kmeans_n_init=10,
        random_state=42,
        min_persistence_days=MIN_PERSIST_DAYS,
        flank_days=5,
        cache_dir=CACHE_DIR,
        cache_overwrite=False,
    ).initialize()

    # Build global caches and fit (done once)
    pipe.compute_baseline_climatology()
    pipe.compute_sigma30_spatial_mean()
    pipe.fit()

    # Classify all years; write outputs immediately (RAM-stable)
    dist_all = []

    for y in YEARS_ALL:
        print(f"\n[CLASSIFY] Year {y}")
        ts_df, periods_df, dist_df, iwr_df = pipe.classify_year(y)

         # Year-specific output folder
        YEAR_DIR = os.path.join(OUT_DIR, str(y))
        os.makedirs(YEAR_DIR, exist_ok=True)

        ts_path = os.path.join(OUT_DIR, f"regimes_{y}_3hourly.csv")
        per_path = os.path.join(OUT_DIR, f"regime_periods_{y}_3hourly.csv")
        dist_path = os.path.join(OUT_DIR, f"regime_distribution_{y}_3hourly.csv")

        ts_df.to_csv(ts_path)
        periods_df.to_csv(per_path, index=False)
        dist_df.to_csv(dist_path, index=False)

         # Excel with time series in sheets, ordered long -> short
        excel_path = os.path.join(YEAR_DIR, f"timeseries_{y}.xlsx")
        write_excel_timeseries_long_to_short(
            excel_path,
            dfs=[
                ("Iwr_timeseries", iwr_df),          # time series (T x k)
                ("Regime_labels", ts_df),            # time series (T x 2)
                ("Regime_periods", periods_df),      # shorter (segments)
                ("Distribution", dist_df),           # shortest
            ],
        )



        dist_all.append(dist_df.assign(year=y))

        print(f"[OK] Saved: {ts_path}")
        print(f"[OK] Saved: {per_path}")
        print(f"[OK] Saved: {dist_path}")

    # Aggregate distribution across years
    dist_all_years = pd.concat(dist_all, ignore_index=True)
    dist_all_path = os.path.join(OUT_DIR, "regime_distribution_all_years.csv")
    dist_all_years.to_csv(dist_all_path, index=False)
    print(f"\n[OK] Saved aggregate distribution: {dist_all_path}")
