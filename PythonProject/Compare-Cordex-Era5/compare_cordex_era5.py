# compare_pypsa_runs_monthly.py
# -------------------------------------------------------------------
# Compare three PyPSA runs (same year, e.g. 2010) across:
#   1) energy_totals.csv                        -> grouped bar plot by category (single plot)
#   2) electricity_demand*.csv                  -> MONTHLY plots (time series; sum over numeric columns)
#   3) avail_profile_s_15.csv                   -> MONTHLY plots (time series; mean over numeric columns)
#   4) profile_*.nc (listed below, NO hydro)    -> MONTHLY plots (time series; spatial mean over non-time dims)
#
# Per-month outputs are written to: ./compare_plots_monthly/YYYY-MM/
#
# Requested change:
# - Exclude Switzerland (CH) EVERYWHERE:
#   * in "ALL" aggregates (drop CH columns before aggregating)
#   * in per-country outputs (do not generate CH plots)
#
# Line styles:
# - Only avail_profile plots use distinct line styles/markers (Option A) to show overlaps.
# - All other time series plots use solid lines.
#
# Units on y-axis:
# - NetCDF: from var attrs["units"] if present
# - CSV: inferred from column names like "... [MW]" or "... (MW)" (if present)
# -------------------------------------------------------------------

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
RUN_DIRS: Dict[str, Path] = {
    "cordex_rcp26": Path("/home/endata/PycharmProjects/pypsa-ee/resources/compare-cordex-rcp26"),
    "cordex_rcp45": Path("/home/endata/PycharmProjects/pypsa-ee/resources/compare-cordex-rcp45"),
    "era5":         Path("/home/endata/PycharmProjects/pypsa-ee/resources/compare-era5"),
}

PROFILE_NC_FILES: List[str] = [
    "profile_15_offwind-ac.nc",
    "profile_15_offwind-dc.nc",
    "profile_15_offwind-float.nc",
    "profile_15_solar.nc",
    "profile_15_onwind.nc",
    "profile_15_solar-hsat.nc",
]

OUT_DIR = Path("Compare-Cordex-Era5/compare_plots_monthly")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed colors per run across ALL plots
RUN_COLORS: Dict[str, str] = {
    "cordex_rcp26": "#1f77b4",
    "cordex_rcp45": "#ff7f0e",
    "era5":         "#2ca02c",
}

# Only for avail_profile: make overlaps visible
AVAIL_STYLE: Dict[str, Dict[str, object]] = {
    "cordex_rcp26": {"linestyle": "--", "marker": "o", "markevery": 24, "alpha": 0.80, "linewidth": 1.2},
    "cordex_rcp45": {"linestyle": ":",  "marker": "s", "markevery": 24, "alpha": 0.80, "linewidth": 1.4},
    "era5":         {"linestyle": "-",  "marker": None, "markevery": None, "alpha": 0.95, "linewidth": 1.8},
}

# Exclude CH everywhere (plots + aggregates)
EXCLUDE_COUNTRIES = {"CH"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _assert_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _month_folder(base: Path, year: int, month: int) -> Path:
    p = base / f"{year:04d}-{month:02d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, which="major", alpha=0.25)


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["snapshot", "snapshots", "time", "datetime", "date", "timestamp"]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    return None


def _is_numeric_like(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    try:
        pd.to_numeric(s.dropna().astype(str).head(200), errors="raise")
        return True
    except Exception:
        return False


def _build_hourly_index_for_year(year: int, n: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01 00:00:00", periods=n, freq="H")


def _coerce_time_index(raw_index: pd.Index, year: int, n_expected: Optional[int] = None) -> pd.DatetimeIndex:
    if isinstance(raw_index, pd.DatetimeIndex):
        return raw_index

    parsed = pd.to_datetime(raw_index, errors="coerce", utc=False)
    if isinstance(parsed, pd.DatetimeIndex) and parsed.notna().sum() > 0:
        return parsed

    s = pd.Series(raw_index)
    if _is_numeric_like(s):
        n = len(raw_index) if n_expected is None else n_expected
        return _build_hourly_index_for_year(year, n)

    return parsed


_UNIT_RE = re.compile(r"""
    (?:\[(?P<u1>[^\]]+)\])     # [MW]
  | (?:\((?P<u2>[^)]+)\))      # (MW)
""", re.VERBOSE)


def _infer_units_from_columns(cols: List[str]) -> Optional[str]:
    for c in cols:
        m = _UNIT_RE.search(c)
        if m:
            return (m.group("u1") or m.group("u2") or "").strip() or None
    return None


def _ylabel_with_units(base: str, units: Optional[str]) -> str:
    return f"{base} [{units}]" if units else base


def _slice_month(series: pd.Series, year: int, month: int) -> pd.Series:
    s = series.copy()
    s = s[~s.index.isna()].sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        return s.iloc[0:0]
    mask = (s.index.year == year) & (s.index.month == month)
    return s.loc[mask]


def _plot_series_solid(ax: plt.Axes, run: str, x, y, linewidth: float = 1.6, alpha: float = 0.95) -> None:
    ax.plot(
        x,
        y,
        label=run,
        color=RUN_COLORS[run],
        linestyle="-",
        linewidth=linewidth,
        alpha=alpha,
    )


def _plot_series_avail_styled(ax: plt.Axes, run: str, x, y) -> None:
    st = AVAIL_STYLE[run]
    ax.plot(
        x,
        y,
        label=run,
        color=RUN_COLORS[run],
        linestyle=st["linestyle"],
        linewidth=st["linewidth"],
        alpha=st["alpha"],
        marker=st["marker"],
        markevery=st["markevery"],
        markersize=3 if st["marker"] is not None else None,
    )


# -----------------------------------------------------------------------------
# Country parsing + CH filtering
# -----------------------------------------------------------------------------
_COUNTRY_RE = re.compile(r"^\s*([A-Z]{2})\b")


def _country_from_col(col: str) -> Optional[str]:
    m = _COUNTRY_RE.match(col)
    return m.group(1) if m else None


def _group_columns_by_country(cols: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for c in cols:
        cc = _country_from_col(c)
        if cc is None:
            continue
        if cc in EXCLUDE_COUNTRIES:
            continue
        out.setdefault(cc, []).append(c)
    return out


def _drop_excluded_country_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] == 0:
        return df
    keep_cols = []
    for c in df.columns:
        cc = _country_from_col(c)
        if cc is None:
            keep_cols.append(c)
        elif cc not in EXCLUDE_COUNTRIES:
            keep_cols.append(c)
    return df[keep_cols]


# -----------------------------------------------------------------------------
# Readers
# -----------------------------------------------------------------------------
def _read_energy_totals_series(fp: Path) -> pd.Series:
    df = pd.read_csv(fp, index_col=0)
    if "carrier" in df.columns:
        df = df.set_index("carrier")

    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {fp}")

    col_candidates = [c for c in num.columns if any(k in c.lower() for k in ["total", "energy", "sum", "value"])]
    col = col_candidates[0] if col_candidates else num.columns[0]

    s = num[col].copy()
    s.index = s.index.astype(str)
    s = s.groupby(level=0).sum()
    return s.sort_index()


def _resolve_demand_file(run_dir: Path) -> Path:
    cand1 = run_dir / "electricity_demand_non_hist.csv"
    cand2 = run_dir / "electricity_demand.csv"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"Neither demand file found in {run_dir}: {cand1.name} or {cand2.name}")


def _read_csv_table(fp: Path, year: int) -> Tuple[pd.DataFrame, Optional[str]]:
    df = pd.read_csv(fp)
    time_col = _find_time_column(df)

    if time_col is not None:
        raw_t = df[time_col]
        num = df.drop(columns=[time_col]).select_dtypes(include=[np.number])
        if _is_numeric_like(raw_t):
            t = _build_hourly_index_for_year(year, len(df))
        else:
            t = pd.to_datetime(raw_t, errors="coerce", utc=False)
            if pd.isna(t).all():
                t = _build_hourly_index_for_year(year, len(df))
        num.index = pd.DatetimeIndex(t)
    else:
        df2 = pd.read_csv(fp, index_col=0)
        df2 = df2.drop(columns=[c for c in df2.columns if c.lower().startswith("unnamed")], errors="ignore")
        num = df2.select_dtypes(include=[np.number])
        t = _coerce_time_index(df2.index, year=year, n_expected=len(df2))
        num.index = pd.DatetimeIndex(t)

    if num.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {fp}")

    num = num[~num.index.isna()].sort_index()

    # drop CH columns everywhere
    num = _drop_excluded_country_columns(num)

    units = _infer_units_from_columns(list(num.columns))
    return num, units


def _pick_first_data_var(ds: xr.Dataset) -> str:
    if len(ds.data_vars) == 0:
        raise ValueError("NetCDF has no data variables.")
    return list(ds.data_vars.keys())[0]


def _pick_time_dim(da: xr.DataArray) -> str:
    for cand in ["time", "snapshot", "snapshots", "date"]:
        if cand in da.dims:
            return cand
    for d in da.dims:
        if d in da.coords and np.issubdtype(da.coords[d].dtype, np.datetime64):
            return d
    raise ValueError(f"Could not identify time dimension for DataArray dims={da.dims} coords={list(da.coords)}")


def _read_profile_nc_timeseries(fp: Path) -> Tuple[pd.Series, Optional[str]]:
    _assert_exists(fp)
    ds = xr.open_dataset(fp, decode_times=True)
    var = _pick_first_data_var(ds)
    da = ds[var]

    time_dim = _pick_time_dim(da)
    other_dims = [d for d in da.dims if d != time_dim]
    da_mean = da.mean(dim=other_dims, skipna=True) if other_dims else da

    idx = da_mean[time_dim].to_index()
    if hasattr(idx, "to_datetimeindex"):
        try:
            idx = idx.to_datetimeindex()
        except Exception:
            idx = pd.to_datetime(np.array(idx), errors="coerce")

    idx = pd.DatetimeIndex(idx)
    y = np.asarray(da_mean.values, dtype=float)

    units = None
    if isinstance(da.attrs, dict):
        units = da.attrs.get("units") or da_mean.attrs.get("units")

    ds.close()

    s = pd.Series(y, index=idx)
    s = s[~s.index.isna()].sort_index()
    return s, units


# -----------------------------------------------------------------------------
# Plotting core
# -----------------------------------------------------------------------------
def _plot_monthly_timeseries(
    title: str,
    ylabel: str,
    series_by_run: Dict[str, pd.Series],
    year: int,
    month: int,
    out_path: Path,
    style_mode: str,  # "solid" or "avail"
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    any_plotted = False
    for run, s in series_by_run.items():
        sm = _slice_month(s, year, month)
        if len(sm) == 0:
            continue

        if style_mode == "avail":
            _plot_series_avail_styled(ax, run, sm.index, sm.values)
        elif style_mode == "solid":
            _plot_series_solid(ax, run, sm.index, sm.values)
        else:
            raise ValueError("style_mode must be 'solid' or 'avail'")

        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No data in this month for any run", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel("Zeit")
    ax.set_ylabel(ylabel)
    _format_time_axis(ax)
    ax.legend()

    _save_fig(fig, out_path)


def _plot_monthly_country_timeseries(
    title_prefix: str,
    ylabel: str,
    df_by_run: Dict[str, pd.DataFrame],
    cols_by_country: Dict[str, List[str]],
    year: int,
    month: int,
    out_dir: Path,
    agg_mode: str,    # "sum" or "mean"
    style_mode: str,  # "solid" or "avail"
) -> None:
    for country, cols in sorted(cols_by_country.items()):
        if country in EXCLUDE_COUNTRIES:
            continue

        fig, ax = plt.subplots(figsize=(13, 5))
        any_plotted = False

        for run, df in df_by_run.items():
            use_cols = [c for c in cols if c in df.columns]
            if not use_cols:
                continue

            if agg_mode == "sum":
                s = df[use_cols].sum(axis=1)
            elif agg_mode == "mean":
                s = df[use_cols].mean(axis=1)
            else:
                raise ValueError("agg_mode must be 'sum' or 'mean'")

            sm = _slice_month(s, year, month)
            if len(sm) == 0:
                continue

            if style_mode == "avail":
                _plot_series_avail_styled(ax, run, sm.index, sm.values)
            else:
                _plot_series_solid(ax, run, sm.index, sm.values)

            any_plotted = True

        if not any_plotted:
            ax.text(0.5, 0.5, f"No data for {country} in this month", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(f"{title_prefix} — {country} ({year}-{month:02d})")
        ax.set_xlabel("Zeit")
        ax.set_ylabel(ylabel)
        _format_time_axis(ax)
        ax.legend()

        _save_fig(fig, out_dir / f"{title_prefix.replace(' ', '_')}__{country}__compare.jpeg")


# -----------------------------------------------------------------------------
# Plot tasks
# -----------------------------------------------------------------------------
def plot_energy_totals(run_dirs: Dict[str, Path], out_dir: Path) -> None:
    series_by_run: Dict[str, pd.Series] = {}
    for run, rdir in run_dirs.items():
        fp = rdir / "energy_totals.csv"
        _assert_exists(fp)
        series_by_run[run] = _read_energy_totals_series(fp)

    all_cats = sorted(set().union(*[s.index.tolist() for s in series_by_run.values()]))

    cat_score = {}
    for cat in all_cats:
        vals = [float(series_by_run[r].get(cat, np.nan)) for r in run_dirs.keys()]
        vals = [v for v in vals if not np.isnan(v)]
        cat_score[cat] = np.mean(np.abs(vals)) if vals else 0.0
    cat_order = sorted(all_cats, key=lambda c: cat_score[c], reverse=True)

    runs = list(run_dirs.keys())
    x = np.arange(len(cat_order))
    width = 0.25 if len(runs) >= 3 else 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, run in enumerate(runs):
        s = series_by_run[run].reindex(cat_order).fillna(0.0)
        ax.bar(
            x + (i - (len(runs) - 1) / 2) * width,
            s.values,
            width=width,
            label=run,
            color=RUN_COLORS[run],
        )

    ax.set_title("energy_totals.csv — Vergleich der Runs")
    ax.set_xlabel("Kategorie")
    ax.set_ylabel("Wert")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_order, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    _save_fig(fig, out_dir / "energy_totals__compare.jpeg")


def plot_monthly_demand(year: int, out_base: Path) -> None:
    df_by_run: Dict[str, pd.DataFrame] = {}
    units_seen: List[str] = []

    for run, rdir in RUN_DIRS.items():
        fp = _resolve_demand_file(rdir)
        df, units = _read_csv_table(fp, year=year)
        df_by_run[run] = df
        if units:
            units_seen.append(units)

    units = units_seen[0] if units_seen else None
    ylabel = _ylabel_with_units("Demand", units)

    # aggregate over all cols (CH already removed)
    series_by_run = {run: df.sum(axis=1) for run, df in df_by_run.items()}

    # country grouping from UNION across runs (after CH removal)
    all_cols = sorted(set().union(*[set(df.columns) for df in df_by_run.values()]))
    cols_by_country = _group_columns_by_country(all_cols)

    for month in range(1, 13):
        mdir = _month_folder(out_base, year, month)

        _plot_monthly_timeseries(
            title=f"electricity_demand*.csv — Sum über Spalten (ohne CH) ({year}-{month:02d})",
            ylabel=ylabel,
            series_by_run=series_by_run,
            year=year,
            month=month,
            out_path=mdir / "electricity_demand__ALL__compare.jpeg",
            style_mode="solid",
        )

        country_dir = mdir / "countries"
        country_dir.mkdir(parents=True, exist_ok=True)
        _plot_monthly_country_timeseries(
            title_prefix="electricity_demand",
            ylabel=ylabel,
            df_by_run=df_by_run,
            cols_by_country=cols_by_country,
            year=year,
            month=month,
            out_dir=country_dir,
            agg_mode="sum",
            style_mode="solid",
        )


def plot_monthly_avail_profile(year: int, out_base: Path) -> None:
    df_by_run: Dict[str, pd.DataFrame] = {}
    units_seen: List[str] = []

    for run, rdir in RUN_DIRS.items():
        fp = rdir / "avail_profile_s_15.csv"
        _assert_exists(fp)
        df, units = _read_csv_table(fp, year=year)
        df_by_run[run] = df
        if units:
            units_seen.append(units)

    units = units_seen[0] if units_seen else None
    ylabel = _ylabel_with_units("Avail profile", units)

    # aggregate over all cols (CH already removed)
    series_by_run = {run: df.mean(axis=1) for run, df in df_by_run.items()}

    # country grouping from UNION across runs (after CH removal)
    all_cols = sorted(set().union(*[set(df.columns) for df in df_by_run.values()]))
    cols_by_country = _group_columns_by_country(all_cols)

    for month in range(1, 13):
        mdir = _month_folder(out_base, year, month)

        _plot_monthly_timeseries(
            title=f"avail_profile_s_15.csv — Mittel über Spalten (ohne CH) ({year}-{month:02d})",
            ylabel=ylabel,
            series_by_run=series_by_run,
            year=year,
            month=month,
            out_path=mdir / "avail_profile_s_15__ALL__compare.jpeg",
            style_mode="avail",  # styled lines only here
        )

        country_dir = mdir / "countries"
        country_dir.mkdir(parents=True, exist_ok=True)
        _plot_monthly_country_timeseries(
            title_prefix="avail_profile_s_15",
            ylabel=ylabel,
            df_by_run=df_by_run,
            cols_by_country=cols_by_country,
            year=year,
            month=month,
            out_dir=country_dir,
            agg_mode="mean",
            style_mode="avail",  # styled lines only here
        )


def plot_monthly_profiles_nc(year: int, out_base: Path) -> None:
    for nc_name in PROFILE_NC_FILES:
        series_by_run: Dict[str, pd.Series] = {}
        units_seen: List[str] = []

        for run, rdir in RUN_DIRS.items():
            fp = rdir / nc_name
            _assert_exists(fp)
            s, units = _read_profile_nc_timeseries(fp)
            series_by_run[run] = s
            if units:
                units_seen.append(units)

        units = units_seen[0] if units_seen else None
        ylabel = _ylabel_with_units("Profile (räuml. Mittelwert)", units)

        for month in range(1, 13):
            mdir = _month_folder(out_base, year, month)
            stem = nc_name.replace(".nc", "")
            _plot_monthly_timeseries(
                title=f"{nc_name} — räuml. Mittelwert ({year}-{month:02d})",
                ylabel=ylabel,
                series_by_run=series_by_run,
                year=year,
                month=month,
                out_path=mdir / f"{stem}__compare.jpeg",
                style_mode="solid",
            )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(year: int = 2010) -> None:
    for run, rdir in RUN_DIRS.items():
        if run not in RUN_COLORS:
            raise ValueError(f"Missing color for run '{run}'.")
        if run not in AVAIL_STYLE:
            raise ValueError(f"Missing avail style for run '{run}'.")
        if not rdir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {rdir}")

    plot_energy_totals(RUN_DIRS, OUT_DIR)
    plot_monthly_demand(year=year, out_base=OUT_DIR)
    plot_monthly_avail_profile(year=year, out_base=OUT_DIR)
    plot_monthly_profiles_nc(year=year, out_base=OUT_DIR)

    print(f"Done. Monthly JPEG plots in: {OUT_DIR.resolve()} (folders {year}-01 ... {year}-12)")


if __name__ == "__main__":
    main(year=2010)

