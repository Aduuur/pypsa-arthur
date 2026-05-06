# plot_compare_prices_monthly.py
# Robust version: handles unsolved networks, empty/NaN marginal_price, snapshot dtype issues,
# missing/zero loads, missing months, and avoids saving "empty-looking" figures without warnings.
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from compare_config import CompareConfig
from compare_utils import ensure_dir, save_fig, format_time_axis, load_first_network, snapshot_weight, warn


# -----------------------------
# Helpers
# -----------------------------
def _to_datetime_index(snapshots) -> pd.DatetimeIndex:
    """Convert snapshots to a robust DatetimeIndex (handles strings/object, tz-aware, etc.)."""
    if isinstance(snapshots, pd.DatetimeIndex):
        idx = snapshots
    else:
        idx = pd.to_datetime(pd.Index(snapshots), errors="coerce")

    # drop timezone info if present (matplotlib + comparisons easier)
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
    except Exception:
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass

    # If everything became NaT -> return empty
    idx = pd.DatetimeIndex(idx)
    idx = idx[~idx.isna()]
    return idx


def _safe_df(df: Optional[pd.DataFrame], idx: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    """Reindex a time-dependent dataframe to idx safely."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    # If df index is not datetime but snapshots were, try to align by position if sizes match.
    try:
        return df.reindex(idx)
    except Exception:
        if len(df.index) == len(idx):
            out = df.copy()
            out.index = idx
            return out
        return None


def _safe_series(s: Optional[pd.Series], idx: pd.DatetimeIndex, fill=1.0) -> pd.Series:
    if s is None or not isinstance(s, pd.Series) or s.empty:
        return pd.Series(fill, index=idx, dtype=float)
    try:
        return pd.to_numeric(s.reindex(idx), errors="coerce").fillna(fill).astype(float)
    except Exception:
        if len(s.index) == len(idx):
            out = pd.to_numeric(s.values, errors="coerce")
            out = pd.Series(out, index=idx).fillna(fill).astype(float)
            return out
        return pd.Series(fill, index=idx, dtype=float)


def _detect_price_table(n) -> Optional[pd.DataFrame]:
    """
    Try common places for nodal prices in PyPSA exports.
    Prefers buses_t.marginal_price, but also tries buses_t.mu_* variants if present.
    """
    if hasattr(n, "buses_t"):
        bt = n.buses_t
        if hasattr(bt, "marginal_price") and isinstance(bt.marginal_price, pd.DataFrame) and not bt.marginal_price.empty:
            return bt.marginal_price
        # Some pipelines store shadow prices with different names (rare, but worth checking)
        # Search for any dataframe-like attribute containing 'marginal' or 'price'
        for name in dir(bt):
            if "price" in name.lower() or "marginal" in name.lower():
                obj = getattr(bt, name, None)
                if isinstance(obj, pd.DataFrame) and not obj.empty:
                    return obj
    return None


def _load_per_bus(n, idx: pd.DatetimeIndex, buses: pd.Index) -> Optional[pd.DataFrame]:
    """
    Compute load per bus (time x bus) from loads_t.p_set and loads['bus'].
    Returns aligned dataframe with columns==buses, index==idx.
    """
    if not (hasattr(n, "loads_t") and hasattr(n.loads_t, "p_set") and hasattr(n, "loads")):
        return None
    if not isinstance(n.loads, pd.DataFrame) or n.loads.empty:
        return None

    pset = _safe_df(getattr(n.loads_t, "p_set", None), idx)
    if pset is None or pset.empty:
        return None

    if "bus" not in n.loads.columns:
        return None

    # Align p_set columns to loads index if needed
    # Typically, p_set columns are load names (same as n.loads.index)
    if not pset.columns.equals(n.loads.index):
        # try to reindex columns to loads index (keeps matching ones)
        pset = pset.reindex(columns=n.loads.index)

    bus_of_load = n.loads["bus"].astype(str)

    # Group loads by bus
    try:
        load_per_bus = pset.groupby(bus_of_load, axis=1).sum(min_count=1)
    except Exception:
        # fallback: manual aggregation
        load_per_bus = pd.DataFrame(index=pset.index)
        for bus, cols in bus_of_load.groupby(bus_of_load).groups.items():
            load_per_bus[str(bus)] = pset.loc[:, list(cols)].sum(axis=1, min_count=1)

    # Align to price buses
    load_per_bus = load_per_bus.reindex(columns=buses.astype(str), fill_value=0.0)
    load_per_bus = load_per_bus.fillna(0.0)

    # If total load is (almost) zero everywhere, treat as unavailable
    if not (load_per_bus.sum(axis=1) > 1e-9).any():
        return None

    return load_per_bus


# -----------------------------
# Core
# -----------------------------
def system_price_timeseries(n) -> pd.Series:
    """
    Returns a system price series indexed by snapshots.
    Priority:
      1) load-weighted average nodal marginal prices (if loads exist & nonzero)
      2) simple average across buses
    Robust to:
      - missing/empty marginal prices (returns empty)
      - non-datetime snapshots
      - NaNs/infs
    """
    mp = _detect_price_table(n)
    if mp is None or not isinstance(mp, pd.DataFrame) or mp.empty:
        return pd.Series(dtype=float)

    idx = _to_datetime_index(getattr(n, "snapshots", mp.index))
    if len(idx) == 0:
        return pd.Series(dtype=float)

    mp = _safe_df(mp, idx)
    if mp is None or mp.empty:
        return pd.Series(dtype=float)

    # sanitize values
    mp = mp.replace([np.inf, -np.inf], np.nan)

    # weights (not used for the raw timeseries, but we keep it for potential future extension)
    _ = _safe_series(snapshot_weight(n) if callable(snapshot_weight) else None, idx, fill=1.0)

    # load weights per bus (aligned to mp columns)
    load_w = _load_per_bus(n, idx, mp.columns)

    if load_w is not None:
        num = (mp * load_w).sum(axis=1, min_count=1)
        den = load_w.sum(axis=1).replace(0.0, np.nan)
        p_sys = num / den
    else:
        p_sys = mp.mean(axis=1)

    p_sys = pd.to_numeric(p_sys, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # If everything is NaN -> empty
    if p_sys.notna().sum() == 0:
        return pd.Series(dtype=float)

    return pd.Series(p_sys.values, index=idx, dtype=float).sort_index()


def main():
    cfg = CompareConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(cfg.out_dir))
    ap.add_argument("--year", type=int, default=cfg.year)
    ap.add_argument("--debug", action="store_true", help="print per-run diagnostics")
    ap.add_argument("--skip-empty-months", action="store_true", help="do not save plots for months with no data")
    ap.add_argument("--min-non-nan", type=int, default=24, help="minimum non-NaN points per month to plot (default 24)")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out))
    year = int(args.year)

    series_by_run: Dict[str, pd.Series] = {}
    loaded: Dict[str, str] = {}

    for run, rdir in cfg.run_dirs.items():
        n, fp = load_first_network(rdir)
        if n is None:
            warn(f"run={run}: no network -> skipping prices")
            continue

        loaded[run] = str(fp)
        s = system_price_timeseries(n)

        if args.debug:
            # inspect whether marginal prices exist and their shape
            mp = _detect_price_table(n)
            mp_shape = None if mp is None else mp.shape
            print(f"[DEBUG] {run}: file={fp}")
            print(f"[DEBUG] {run}: snapshots={len(getattr(n,'snapshots',[]))}")
            print(f"[DEBUG] {run}: marginal_price table shape={mp_shape}")
            if isinstance(s, pd.Series) and not s.empty:
                print(f"[DEBUG] {run}: price series non-NaN={int(s.notna().sum())}, min={float(np.nanmin(s.values)):.3f}, max={float(np.nanmax(s.values)):.3f}")

        if s.empty:
            warn(f"run={run}: price series empty (missing/empty marginal prices or all-NaN)")
            continue

        series_by_run[run] = s

    if not series_by_run:
        raise SystemExit("No price series available (missing networks or marginal prices are empty).")

    # Plot month by month
    for month in range(1, 13):
        # Gather month slices
        month_slices: Dict[str, pd.Series] = {}
        for run, s in series_by_run.items():
            sm = s[(s.index.year == year) & (s.index.month == month)]
            sm = sm.dropna()
            if len(sm) >= args.min_non_nan:
                month_slices[run] = sm

        if not month_slices:
            msg = f"{year:04d}-{month:02d}: no runs with >= {args.min_non_nan} non-NaN points"
            if args.skip_empty_months:
                warn(msg + " -> skipping save")
                continue
            else:
                warn(msg + " -> saving empty plot for traceability")

        mdir = ensure_dir(out_dir / f"{year:04d}-{month:02d}")
        fig, ax = plt.subplots(figsize=(13, 5))

        any_ = False
        for run, sm in month_slices.items():
            color = cfg.colors.get(run, None)  # allow matplotlib default
            ax.plot(sm.index, sm.values, color=color, linestyle="-", linewidth=1.7, alpha=0.95, label=run)
            any_ = True

        ax.set_title(f"System price (load-weighted; fallback mean) — {year}-{month:02d}")
        ax.set_xlabel("Zeit")
        ax.set_ylabel("Price [EUR/MWh]")
        format_time_axis(ax)

        if any_:
            ax.legend()
            # set y-limits robustly to avoid "flat line looks empty" due to extreme outliers
            y = np.concatenate([v.values for v in month_slices.values()]) if month_slices else np.array([])
            if y.size > 0:
                lo, hi = np.nanpercentile(y, [1, 99])
                if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                    ax.set_ylim(lo, hi)

        ax.grid(True, axis="y", alpha=0.25)
        save_fig(fig, mdir / "system_price__compare.jpeg")

    print("[OK] Networks used:")
    for run, fp in loaded.items():
        print(f"  - {run}: {fp}")
    print(f"[OK] Price plots in {out_dir}")


if __name__ == "__main__":
    main()
