#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# so starten python Weather_regimes/Visualisierung_regime.py --k 8 --p 5 --include-no-regime
from __future__ import annotations

import os
import glob
import gc
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Optional: hübsche Karten
_HAS_CARTOPY = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False


# =============================================================================
# STYLE / LAYOUT DEFAULTS
# =============================================================================
FS_TITLE_MAIN = 20
FS_TITLE_SINGLE = 20
FS_PANEL_LABEL = 21
FS_AXIS_LABEL = 20
FS_TICK = 20
FS_CBAR_LABEL = 21
FS_CBAR_TICK = 20

GRIDLINE_COLOR = "gray"
GRIDLINE_ALPHA = 0.28
GRIDLINE_WIDTH = 0.35


# =============================================================================
# Helpers (pfad-/grid-/domain)
# =============================================================================
def to_lon180(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def discover_yearly_files(base_dir: str, year: int, filename_glob: str = "zg500_*.nc") -> List[str]:
    folder = os.path.join(base_dir, f"cordex_{year}")
    files = sorted(glob.glob(os.path.join(folder, filename_glob)))
    if not files:
        raise FileNotFoundError(f"No files found for year {year} in {folder} with pattern {filename_glob}")
    return files


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


def run_tag(persistence_days: int, n_clusters: int) -> str:
    return f"p{int(persistence_days)}d_k{int(n_clusters)}"


def resolve_dirs(base: str, y0: int, y1: int, persistence_days: int, n_clusters: int) -> Tuple[str, str, str]:
    tag = run_tag(persistence_days, n_clusters)
    cache_dir = os.path.join(base, f"_cache_grams_strict_{tag}_{y0}_{y1}")
    out_dir = os.path.join(base, f"results_grams_strict_{tag}_{y0}_{y1}")
    plot_dir = os.path.join(base, f"plots_grams_strict_{tag}_{y0}_{y1}")
    return cache_dir, out_dir, plot_dir


def load_latlon_and_domain_indices_from_netcdf(
    base_dir: str,
    sample_year: int,
    filename_glob: str,
    lat_coord: str,
    lon_coord: str,
    y_dim: str,
    x_dim: str,
    coarsen_factor: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    files = discover_yearly_files(base_dir, sample_year, filename_glob)
    ds = xr.open_mfdataset(files, engine="netcdf4", combine="by_coords", chunks="auto")
    try:
        lat2d = ds[lat_coord].values
        lon2d = to_lon180(ds[lon_coord].values)

        if coarsen_factor > 1:
            lat2d = xr.DataArray(lat2d, dims=(y_dim, x_dim)).coarsen(
                {y_dim: coarsen_factor, x_dim: coarsen_factor}, boundary="trim"
            ).mean().values
            lon2d = xr.DataArray(lon2d, dims=(y_dim, x_dim)).coarsen(
                {y_dim: coarsen_factor, x_dim: coarsen_factor}, boundary="trim"
            ).mean().values

        dom = compute_domain_mask_from_latlon(
            lat2d, lon2d,
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
        )
        mask_flat = dom.ravel().astype(bool)
        space_indices = np.where(mask_flat)[0].astype(np.int64)
        if space_indices.size < 1000:
            raise RuntimeError(f"Domain mask keeps too few points ({space_indices.size}). Check bounds.")

        grid_shape = lat2d.shape
        return lat2d.astype(np.float32), lon2d.astype(np.float32), space_indices, grid_shape
    finally:
        ds.close()
        del ds
        gc.collect()


def expand_space_to_grid(field_space: np.ndarray, space_indices: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
    ny, nx = grid_shape
    grid = np.full((ny * nx,), np.nan, dtype=np.float32)
    grid[space_indices] = field_space.astype(np.float32)
    return grid.reshape(ny, nx)


# =============================================================================
# Geo-axis helpers
# =============================================================================
def _format_geo_axis_single(ax, extent: Tuple[float, float, float, float]) -> None:
    if _HAS_CARTOPY:
        ax.set_extent(list(extent), crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=False,   # <- auch hier alles aus
            linewidth=GRIDLINE_WIDTH,
            color=GRIDLINE_COLOR,
            alpha=GRIDLINE_ALPHA,
            linestyle="-",
        )
        _ = gl
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True, alpha=GRIDLINE_ALPHA, linewidth=GRIDLINE_WIDTH)
        ax.set_xticks([])
        ax.set_yticks([])

def _format_geo_axis_overview(
    ax,
    extent: Tuple[float, float, float, float],
    *,
    show_left_labels: bool,
    show_bottom_labels: bool,
) -> None:
    if _HAS_CARTOPY:
        ax.set_extent(list(extent), crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.55)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=False,   # <- alles aus
            linewidth=GRIDLINE_WIDTH,
            color=GRIDLINE_COLOR,
            alpha=GRIDLINE_ALPHA,
            linestyle="-",
        )
        _ = gl
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True, alpha=GRIDLINE_ALPHA, linewidth=GRIDLINE_WIDTH)
        ax.set_xticks([])
        ax.set_yticks([])


# =============================================================================
# Streaming accumulation of composites
# =============================================================================
def accumulate_regime_composites_streaming(
    years: List[int],
    out_dir: str,
    cache_dir: str,
    n_clusters: int,
    include_no_regime: bool = False,
    regimes_filename_tpl: str = "regimes_{y}_3hourly.csv",
    preproc_filename_tpl: str = "preproc_field_{y}.zarr",
    time_chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Streaming accumulation of regime composites in 'space' coordinates.

    Returns:
      mean_fields: (R, S) float32 with NaNs where no samples
      counts:     (R,) int64
    """
    R = n_clusters + (1 if include_no_regime else 0)
    no_id = n_clusters  # per your pipeline

    sum_fields: Optional[np.ndarray] = None
    cnt = np.zeros((R,), dtype=np.int64)

    for y in years:
        regimes_csv = os.path.join(out_dir, regimes_filename_tpl.format(y=y))
        preproc_zarr = os.path.join(cache_dir, preproc_filename_tpl.format(y=y))

        if not os.path.exists(regimes_csv):
            raise FileNotFoundError(f"Missing regimes CSV: {regimes_csv}")
        if not os.path.exists(preproc_zarr):
            raise FileNotFoundError(f"Missing preprocessed zarr: {preproc_zarr}")

        df = pd.read_csv(regimes_csv, index_col=0, parse_dates=True)
        if "regime" not in df.columns:
            raise KeyError(f"'regime' column missing in {regimes_csv}")

        labels = df["regime"].astype(int).values
        times_labels = pd.to_datetime(df.index.values).tz_localize(None)

        ds = xr.open_zarr(preproc_zarr)
        try:
            X = ds["field"]  # (time, space)
            times_field = pd.to_datetime(ds["time"].values).tz_localize(None)

            # time intersection (exact timestamps)
            common = np.intersect1d(times_field.values, times_labels.values)
            if common.size == 0:
                print(f"[WARN] No common timestamps for year {y}. Skipping.")
                continue

            idx_f = np.searchsorted(times_field.values, common)
            idx_l = np.searchsorted(times_labels.values, common)
            labels_c = labels[idx_l]

            S = int(X.sizes["space"])
            if sum_fields is None:
                sum_fields = np.zeros((R, S), dtype=np.float64)

            nC = int(len(common))
            for i0 in range(0, nC, time_chunk):
                i1 = min(nC, i0 + time_chunk)
                lf = labels_c[i0:i1]
                block = X.isel(time=xr.DataArray(idx_f[i0:i1], dims="t")).values.astype(np.float32)

                for r in range(n_clusters):
                    m = (lf == r)
                    if np.any(m):
                        sum_fields[r] += block[m].sum(axis=0, dtype=np.float64)
                        cnt[r] += int(m.sum())

                if include_no_regime:
                    rN = n_clusters
                    mN = (lf == no_id)
                    if np.any(mN):
                        sum_fields[rN] += block[mN].sum(axis=0, dtype=np.float64)
                        cnt[rN] += int(mN.sum())

        finally:
            ds.close()
            del ds
            gc.collect()

        print(f"[OK] Accumulated year {y}")

    if sum_fields is None:
        raise RuntimeError("No data accumulated. Check paths and time alignment.")

    mean_fields = np.full_like(sum_fields, np.nan, dtype=np.float32)
    for r in range(R):
        if cnt[r] > 0:
            mean_fields[r] = (sum_fields[r] / cnt[r]).astype(np.float32)

    return mean_fields, cnt


# =============================================================================
# Plot helpers
# =============================================================================
def _derive_common_norm_from_grids(
    grids: np.ndarray,
    robust_percentile: float,
    fixed_vmax: Optional[float],
    eps: float = 1e-6,
) -> TwoSlopeNorm:
    vals = grids[np.isfinite(grids)]
    if vals.size == 0:
        raise RuntimeError("No finite values to derive norm.")
    if fixed_vmax is None:
        vmax = float(max(np.nanpercentile(np.abs(vals), robust_percentile), eps))
    else:
        vmax = float(max(abs(fixed_vmax), eps))
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s)).strip("_")


# =============================================================================
# Plot: single regime map
# =============================================================================
def plot_single_regime_map(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    grid: np.ndarray,
    count: int,
    out_png: str,
    *,
    title: str,
    norm: TwoSlopeNorm,
    cmap: str,
    extent: Tuple[float, float, float, float],
    dpi: int = 300,
) -> None:
    Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.5, 8.8), dpi=dpi)
    proj = ccrs.PlateCarree() if _HAS_CARTOPY else None
    ax = fig.add_subplot(1, 1, 1, projection=proj) if _HAS_CARTOPY else fig.add_subplot(1, 1, 1)

    ax.set_title(f"{title}\n(n={count})", fontsize=FS_TITLE_SINGLE, pad=14)

    if _HAS_CARTOPY:
        m = ax.pcolormesh(
            lon2d, lat2d, grid,
            transform=ccrs.PlateCarree(),
            shading="auto",
            norm=norm,
            cmap=cmap,
        )
    else:
        m = ax.pcolormesh(
            lon2d, lat2d, grid,
            shading="auto",
            norm=norm,
            cmap=cmap,
        )

    _format_geo_axis_single(ax, extent)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.06, fraction=0.055, extend="both")
    cb.set_label("Kompositionsmittelwert des normalisierten Anomaliefeldes", fontsize=FS_CBAR_LABEL)
    cb.ax.tick_params(labelsize=FS_CBAR_TICK)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")


# =============================================================================
# Plot: overview 3x3 (row-major)
# =============================================================================
def plot_overview_grid_3x3(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    grids: np.ndarray,
    counts: np.ndarray,
    labels: List[str],
    out_png: str,
    *,
    title: str,
    norm: TwoSlopeNorm,
    cmap: str = "RdBu_r",
    extent: Tuple[float, float, float, float] = (-12, 35, 35, 72),
    dpi: int = 300,
    max_panels: int = 9,
) -> None:
    """
    3x3 overview, row-major:
    first fill left->right, then top->bottom
    """
    Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)

    R = int(grids.shape[0])
    n_panels = min(max_panels, R)

    nrows_maps = 3
    ncols_maps = 3

    fig = plt.figure(figsize=(19.5, 15.8), dpi=dpi)

    # [title] + 3*(label,map) + [cbar]
    height_ratios = [0.13] + sum(([0.085, 1.0] for _ in range(nrows_maps)), []) + [0.08]

    gs = fig.add_gridspec(
        nrows=len(height_ratios),
        ncols=ncols_maps,
        height_ratios=height_ratios,
        width_ratios=[1.0, 1.0, 1.0],
        left=0.04, right=0.98, top=0.98, bottom=0.06,
        wspace=0.03,
        hspace=0.06,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5, title,
        ha="center", va="center",
        fontsize=FS_TITLE_MAIN, fontweight="semibold"
    )

    proj = ccrs.PlateCarree() if _HAS_CARTOPY else None

    for i in range(n_panels):
        rix = i // ncols_maps
        col = i % ncols_maps

        base_row = 1 + rix * 2
        label_row = base_row
        map_row = base_row + 1

        ax_lab = fig.add_subplot(gs[label_row, col])
        ax_lab.axis("off")
        ax_lab.text(
            0.5, 0.02,
            f"{labels[i]} (n={int(counts[i])})",
            ha="center", va="bottom",
            fontsize=FS_PANEL_LABEL, fontweight="semibold",
        )

        ax = (
            fig.add_subplot(gs[map_row, col], projection=proj)
            if _HAS_CARTOPY else
            fig.add_subplot(gs[map_row, col])
        )

        if _HAS_CARTOPY:
            ax.pcolormesh(
                lon2d, lat2d, grids[i],
                transform=ccrs.PlateCarree(),
                shading="auto",
                norm=norm,
                cmap=cmap,
            )
        else:
            ax.pcolormesh(
                lon2d, lat2d, grids[i],
                shading="auto",
                norm=norm,
                cmap=cmap,
            )

        show_left = (col == 0)
        show_bottom = (rix == nrows_maps - 1)
        _format_geo_axis_overview(ax, extent, show_left_labels=show_left, show_bottom_labels=show_bottom)

    cax = fig.add_subplot(gs[-1, :])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("Kompositionsmittelwert des normalisierten Anomaliefeldes", fontsize=FS_CBAR_LABEL, labelpad = 12)
    cb.ax.tick_params(labelsize=FS_CBAR_TICK, pad = 4)

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved overview: {out_png}")


# =============================================================================
# Plot: overview 5x2 (NOW ALSO ROW-MAJOR)
# =============================================================================
def plot_overview_grid_5x2(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    grids: np.ndarray,
    counts: np.ndarray,
    labels: List[str],
    out_png: str,
    *,
    title: str,
    norm: TwoSlopeNorm,
    cmap: str = "RdBu_r",
    extent: Tuple[float, float, float, float] = (-12, 35, 35, 72),
    dpi: int = 300,
    max_panels: int = 10,
) -> None:
    """
    5x2 overview, row-major:
    panel order is:
      0 1
      2 3
      4 5
      6 7
      8 9
    """
    Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)

    R = int(grids.shape[0])
    n_panels = min(max_panels, R)

    nrows_maps = 5
    ncols_maps = 2

    fig = plt.figure(figsize=(18.8, 20.0), dpi=dpi)

    height_ratios = [0.11] + sum(([0.085, 1.0] for _ in range(nrows_maps)), []) + [0.07]

    gs = fig.add_gridspec(
        nrows=len(height_ratios),
        ncols=ncols_maps,
        height_ratios=height_ratios,
        width_ratios=[1.0, 1.0],
        left=0.04, right=0.98, top=0.98, bottom=0.05,
        wspace=0.03,
        hspace=0.05,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5, title,
        ha="center", va="center",
        fontsize=FS_TITLE_MAIN, fontweight="semibold"
    )

    proj = ccrs.PlateCarree() if _HAS_CARTOPY else None

    for i in range(n_panels):
        # ROW-MAJOR instead of column-major
        rix = i // ncols_maps
        col = i % ncols_maps

        base_row = 1 + rix * 2
        label_row = base_row
        map_row = base_row + 1

        ax_lab = fig.add_subplot(gs[label_row, col])
        ax_lab.axis("off")
        ax_lab.text(
            0.5, 0.02,
            f"{labels[i]} (n={int(counts[i])})",
            ha="center", va="bottom",
            fontsize=FS_PANEL_LABEL, fontweight="normal"
        )

        ax = (
            fig.add_subplot(gs[map_row, col], projection=proj)
            if _HAS_CARTOPY else
            fig.add_subplot(gs[map_row, col])
        )

        if _HAS_CARTOPY:
            ax.pcolormesh(
                lon2d, lat2d, grids[i],
                transform=ccrs.PlateCarree(),
                shading="auto",
                norm=norm,
                cmap=cmap,
            )
        else:
            ax.pcolormesh(
                lon2d, lat2d, grids[i],
                shading="auto",
                norm=norm,
                cmap=cmap,
            )

        show_left = (col == 0)
        show_bottom = (rix == nrows_maps - 1)
        _format_geo_axis_overview(ax, extent, show_left_labels=show_left, show_bottom_labels=show_bottom)

    cax = fig.add_subplot(gs[-1, :])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("Kompositmittelwert des normalisierten Tiefpass-Anomaliefeldes", fontsize=FS_CBAR_LABEL, labelpad = 12)
    cb.ax.tick_params(labelsize=FS_CBAR_TICK, pad = 4)

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved overview: {out_png}")


# =============================================================================
# Legacy sqrt-grid
# =============================================================================
def plot_legacy_sqrt_grid(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    grids: np.ndarray,
    counts: np.ndarray,
    labels: List[str],
    out_png: str,
    out_pdf: Optional[str],
    *,
    title: str,
    norm: TwoSlopeNorm,
    cmap: str,
    extent: Tuple[float, float, float, float],
    dpi: int = 200,
) -> None:
    Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)

    R = int(grids.shape[0])
    ncols = int(np.ceil(np.sqrt(R)))
    nrows = int(np.ceil(R / ncols))

    fig = plt.figure(figsize=(12.5, 11.5), constrained_layout=True, dpi=dpi)
    proj = ccrs.PlateCarree() if _HAS_CARTOPY else None

    axes = []
    for i in range(R):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=proj) if _HAS_CARTOPY else fig.add_subplot(nrows, ncols, i + 1)
        axes.append(ax)

    mappable = None
    for i, ax in enumerate(axes):
        if _HAS_CARTOPY:
            mappable = ax.pcolormesh(
                lon2d, lat2d, grids[i],
                transform=ccrs.PlateCarree(),
                shading="auto",
                norm=norm,
                cmap=cmap,
            )
        else:
            mappable = ax.pcolormesh(lon2d, lat2d, grids[i], shading="auto", norm=norm, cmap=cmap)

        row = i // ncols
        col = i % ncols
        show_left = (col == 0)
        show_bottom = (row == nrows - 1)
        _format_geo_axis_overview(ax, extent, show_left_labels=show_left, show_bottom_labels=show_bottom)

        ax.text(
            0.5, 1.04,
            f"{labels[i]} (n={int(counts[i])})",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=FS_PANEL_LABEL - 2,
            fontweight="normal",
            clip_on=False,
        )

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.88, pad=0.02)
        cbar.set_label("Kompositmittelwert des normalisierten Tiefpass-Anomaliefeldes", fontsize=FS_CBAR_LABEL)
        cbar.ax.tick_params(labelsize=FS_CBAR_TICK)

    fig.suptitle(title, fontsize=FS_TITLE_SINGLE)
    fig.savefig(out_png)
    if out_pdf:
        fig.savefig(out_pdf)
    plt.close(fig)

    print(f"[OK] Saved legacy grid: {out_png}")
    if out_pdf:
        print(f"[OK] Saved legacy grid: {out_pdf}")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paper-style plots for GramsStrict regime composites (p/k selection).")

    ap.add_argument("--base", type=str, default="/mnt/endata/Cordex/Copernicus_CORDEX_data_RCP_8.5", help="Base directory.")
    ap.add_argument("--y0", type=int, default=2025, help="Start year (inclusive).")
    ap.add_argument("--y1", type=int, default=2050, help="End year (inclusive).")
    ap.add_argument("--k", type=int, default=8, help="Number of clusters (k).")
    ap.add_argument("--p", type=int, default=5, help="Min persistence days (p).")

    ap.add_argument("--include-no-regime", action="store_true", help="Include No-Regime composite (id==k).")
    ap.add_argument("--coarsen", type=int, default=2, help="Coarsen factor used in the regime pipeline.")
    ap.add_argument("--filename-glob", type=str, default="zg500_*.nc", help="NetCDF filename glob.")

    ap.add_argument("--lat-coord", type=str, default="lat")
    ap.add_argument("--lon-coord", type=str, default="lon")
    ap.add_argument("--y-dim", type=str, default="y")
    ap.add_argument("--x-dim", type=str, default="x")

    ap.add_argument("--lat-min", type=float, default=30.0)
    ap.add_argument("--lat-max", type=float, default=90.0)
    ap.add_argument("--lon-min", type=float, default=-80.0)
    ap.add_argument("--lon-max", type=float, default=40.0)

    ap.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=(-12.0, 35.0, 35.0, 72.0),
        metavar=("LON0", "LON1", "LAT0", "LAT1"),
        help="Plot extent (lon_min lon_max lat_min lat_max)."
    )

    ap.add_argument("--robust-percentile", type=float, default=99.0, help="Robust percentile for common color scale.")
    ap.add_argument("--vmax", type=float, default=None, help="Fixed symmetric vmax for all panels (overrides percentile).")
    ap.add_argument("--cmap", type=str, default="RdBu_r", help="Colormap for anomalies.")

    ap.add_argument("--dpi", type=int, default=300, help="DPI for outputs.")
    ap.add_argument("--legacy-grid", action="store_true", help="Also write legacy sqrt-grid to old output filenames.")

    args, _unknown = ap.parse_known_args(argv)
    return args


# =============================================================================
# MAIN
# =============================================================================
def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    years = list(range(int(args.y0), int(args.y1) + 1))
    tag = run_tag(int(args.p), int(args.k))
    cache_dir, out_dir, plot_dir = resolve_dirs(args.base, int(args.y0), int(args.y1), int(args.p), int(args.k))

    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"OUT_DIR not found: {out_dir}")
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"CACHE_DIR not found: {cache_dir}")

    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    maps_dir = os.path.join(plot_dir, "maps_per_regime")
    overview_dir = os.path.join(plot_dir, "overview_grids")
    Path(maps_dir).mkdir(parents=True, exist_ok=True)
    Path(overview_dir).mkdir(parents=True, exist_ok=True)

    print(f"[OK] tag={tag}")
    print(f"[OK] cache_dir={cache_dir}")
    print(f"[OK] out_dir={out_dir}")
    print(f"[OK] plot_dir={plot_dir}")

    sample_year = years[0]
    lat2d, lon2d, space_indices, grid_shape = load_latlon_and_domain_indices_from_netcdf(
        base_dir=args.base,
        sample_year=sample_year,
        filename_glob=args.filename_glob,
        lat_coord=args.lat_coord,
        lon_coord=args.lon_coord,
        y_dim=args.y_dim,
        x_dim=args.x_dim,
        coarsen_factor=int(args.coarsen),
        lat_min=float(args.lat_min),
        lat_max=float(args.lat_max),
        lon_min=float(args.lon_min),
        lon_max=float(args.lon_max),
    )

    mean_fields, counts = accumulate_regime_composites_streaming(
        years=years,
        out_dir=out_dir,
        cache_dir=cache_dir,
        n_clusters=int(args.k),
        include_no_regime=bool(args.include_no_regime),
    )

    R = int(mean_fields.shape[0])
    grids = np.stack(
        [expand_space_to_grid(mean_fields[r], space_indices, grid_shape) for r in range(R)],
        axis=0
    )

    labels = []
    for r in range(R):
        if bool(args.include_no_regime) and r == (R - 1):
            labels.append("No Regime")
        else:
            labels.append(f"Regime {r}")

    norm = _derive_common_norm_from_grids(
        grids=grids,
        robust_percentile=float(args.robust_percentile),
        fixed_vmax=(None if args.vmax is None else float(args.vmax)),
    )

    extent = tuple(float(x) for x in args.extent)

    # -------------------------------------------------------------------------
    # A) Einzelkarten je Regime
    # -------------------------------------------------------------------------
    print("\n--- A) Einzelkarten je Regime (paper-style) ---")
    for r in range(R):
        reg_dir = os.path.join(maps_dir, f"regime_{_safe(labels[r])}")
        Path(reg_dir).mkdir(parents=True, exist_ok=True)

        out_png = os.path.join(reg_dir, f"composite__{_safe(labels[r])}__{tag}_{args.y0}_{args.y1}.png")
        title = f"GramsStrict CORDEX Composite | {labels[r]} | {tag} | {args.y0}–{args.y1}"
        plot_single_regime_map(
            lat2d=lat2d,
            lon2d=lon2d,
            grid=grids[r],
            count=int(counts[r]),
            out_png=out_png,
            title=title,
            norm=norm,
            cmap=str(args.cmap),
            extent=extent,
            dpi=int(args.dpi),
        )

    # -------------------------------------------------------------------------
    # B) Overview-Grid (3x3)
    # -------------------------------------------------------------------------
    print("\n--- B) Overview-Grid (3x3) ---")
    out_overview_png = os.path.join(overview_dir, f"overview__{tag}_{args.y0}_{args.y1}__3x3.png")
    plot_overview_grid_3x3(
        lat2d=lat2d,
        lon2d=lon2d,
        grids=grids,
        counts=counts,
        labels=labels,
        out_png=out_overview_png,
        title=f"",
        norm=norm,
        cmap=str(args.cmap),
        extent=extent,
        dpi=int(args.dpi),
        max_panels=9,
    )

    # -------------------------------------------------------------------------
    # C) Optional: legacy grid
    # -------------------------------------------------------------------------
    if bool(args.legacy_grid):
        print("\n--- C) Legacy sqrt-grid (optional) ---")
        legacy_png = os.path.join(plot_dir, f"regime_composites_{tag}_{args.y0}_{args.y1}.png")
        legacy_pdf = os.path.join(plot_dir, f"regime_composites_{tag}_{args.y0}_{args.y1}.pdf")
        plot_legacy_sqrt_grid(
            lat2d=lat2d,
            lon2d=lon2d,
            grids=grids,
            counts=counts,
            labels=labels,
            out_png=legacy_png,
            out_pdf=legacy_pdf,
            title=f"GramsStrict CORDEX Regime Composites ({args.y0}–{args.y1}) | {tag}",
            norm=norm,
            cmap=str(args.cmap),
            extent=extent,
            dpi=200,
        )

    vmin, vmax = float(norm.vmin), float(norm.vmax)
    print("\nFertig.")
    print(f"[OK] Common scale: [{vmin:.3f}, {vmax:.3f}]")
    print(f"[OK] Outputs under: {plot_dir}")


if __name__ == "__main__":
    main()