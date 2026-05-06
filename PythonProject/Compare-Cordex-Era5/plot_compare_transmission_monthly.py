# plot_compare_transmission_monthly.py
# Robust version:
# - handles non-datetime snapshots
# - handles missing/unsolved networks (no lines_t.p0 / links_t.p0)
# - aligns weights + time series safely (index mismatch)
# - supports alternative flow tables (p0/p1) if present
# - skips saving empty-looking plots unless explicitly requested
# - adds diagnostics mode

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from compare_config import CompareConfig
from compare_utils import ensure_dir, save_fig, load_first_network, snapshot_weight, warn


# -----------------------------
# Helpers
# -----------------------------
def _to_datetime_index(snapshots) -> pd.DatetimeIndex:
    """Robust conversion of snapshots to DatetimeIndex."""
    if isinstance(snapshots, pd.DatetimeIndex):
        idx = snapshots
    else:
        idx = pd.to_datetime(pd.Index(snapshots), errors="coerce")
    idx = pd.DatetimeIndex(idx)
    idx = idx[~idx.isna()]
    # drop tz if present
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
    except Exception:
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass
    return idx


def _month_mask(idx: pd.DatetimeIndex, year: int, month: int) -> np.ndarray:
    """Boolean mask for idx within year-month."""
    return (idx.year == year) & (idx.month == month)


def _safe_df(df: Optional[pd.DataFrame], idx: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    """Reindex df to idx; fallback to position-based alignment if needed."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    try:
        return df.reindex(idx)
    except Exception:
        if len(df.index) == len(idx):
            out = df.copy()
            out.index = idx
            return out
        return None


def _safe_weights(n, idx: pd.DatetimeIndex) -> pd.Series:
    """Snapshot weights aligned to idx, default 1.0."""
    try:
        w = snapshot_weight(n)
    except Exception:
        w = None
    if w is None or not isinstance(w, (pd.Series, pd.DataFrame)):
        return pd.Series(1.0, index=idx, dtype=float)

    if isinstance(w, pd.DataFrame):
        # Some pipelines store weights as DataFrame; reduce to series
        if w.shape[1] >= 1:
            w = w.iloc[:, 0]
        else:
            return pd.Series(1.0, index=idx, dtype=float)

    try:
        w = pd.to_numeric(w.reindex(idx), errors="coerce").fillna(1.0).astype(float)
    except Exception:
        if len(w.index) == len(idx):
            w = pd.Series(pd.to_numeric(w.values, errors="coerce"), index=idx).fillna(1.0).astype(float)
        else:
            w = pd.Series(1.0, index=idx, dtype=float)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return w


def _detect_flow_table(n, comp: str) -> Optional[pd.DataFrame]:
    """
    Detect a flow time series table for a component group.
    comp: 'lines' or 'links'
    Prefer p0; if missing, try p1; else None.
    """
    t = getattr(n, f"{comp}_t", None)
    if t is None:
        return None
    for cname in ("p0", "p1"):
        obj = getattr(t, cname, None)
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            return obj
    return None


# -----------------------------
# Core
# -----------------------------
def transmission_energy_month(n, year: int, month: int) -> pd.Series:
    """
    Energy proxy: sum_t sum_assets |flow(t, asset)| * weight(t)
    Returns a Series with entries for AC lines and links if available.
    Units: MWh if weight is hours; otherwise "weighted-MWh" proxy (consistent across runs).
    """
    idx = _to_datetime_index(getattr(n, "snapshots", []))
    if len(idx) == 0:
        return pd.Series(dtype=float)

    m = _month_mask(idx, year, month)
    if not m.any():
        return pd.Series(dtype=float)

    w = _safe_weights(n, idx).loc[idx[m]]
    out: Dict[str, float] = {}

    # Lines (AC)
    if hasattr(n, "lines") and isinstance(getattr(n, "lines"), pd.DataFrame) and len(n.lines) > 0:
        p = _detect_flow_table(n, "lines")
        p = _safe_df(p, idx)
        if p is not None and not p.empty:
            pm = p.loc[idx[m]].replace([np.inf, -np.inf], np.nan)
            # If all NaN -> skip
            if pm.notna().sum().sum() > 0:
                e = (pm.abs().mul(w, axis=0)).sum().sum(min_count=1)
                if pd.notna(e) and float(e) != 0.0:
                    out["AC lines |p|"] = float(e)

    # Links (often DC / converters)
    if hasattr(n, "links") and isinstance(getattr(n, "links"), pd.DataFrame) and len(n.links) > 0:
        p = _detect_flow_table(n, "links")
        p = _safe_df(p, idx)
        if p is not None and not p.empty:
            pm = p.loc[idx[m]].replace([np.inf, -np.inf], np.nan)
            if pm.notna().sum().sum() > 0:
                e = (pm.abs().mul(w, axis=0)).sum().sum(min_count=1)
                if pd.notna(e) and float(e) != 0.0:
                    out["Links |p|"] = float(e)

    return pd.Series(out, dtype=float).sort_values(ascending=False)


def main():
    cfg = CompareConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(cfg.out_dir))
    ap.add_argument("--year", type=int, default=cfg.year)
    ap.add_argument("--debug", action="store_true", help="print per-run diagnostics")
    ap.add_argument("--skip-empty-months", action="store_true", help="do not save plots for months with no data")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out))
    year = int(args.year)

    nets: Dict[str, object] = {}
    loaded: Dict[str, str] = {}

    for run, rdir in cfg.run_dirs.items():
        n, fp = load_first_network(rdir)
        if n is None:
            warn(f"run={run}: no network -> skipping transmission")
            continue
        nets[run] = n
        loaded[run] = str(fp)

        if args.debug:
            idx = _to_datetime_index(getattr(n, "snapshots", []))
            lt = getattr(n, "lines_t", None)
            kt = getattr(n, "links_t", None)
            has_lines_p0 = isinstance(getattr(lt, "p0", None), pd.DataFrame) if lt is not None else False
            has_lines_p1 = isinstance(getattr(lt, "p1", None), pd.DataFrame) if lt is not None else False
            has_links_p0 = isinstance(getattr(kt, "p0", None), pd.DataFrame) if kt is not None else False
            has_links_p1 = isinstance(getattr(kt, "p1", None), pd.DataFrame) if kt is not None else False
            print(f"[DEBUG] {run}: file={fp}")
            print(f"[DEBUG] {run}: snapshots={len(idx)}")
            print(f"[DEBUG] {run}: lines={len(getattr(n,'lines',[]))}, links={len(getattr(n,'links',[]))}")
            print(f"[DEBUG] {run}: lines_t has p0={has_lines_p0}, p1={has_lines_p1}; links_t has p0={has_links_p0}, p1={has_links_p1}")

    if not nets:
        raise SystemExit("No networks loaded; cannot compute transmission metrics.")

    for month in range(1, 13):
        # Compute metrics for all runs
        ser_by_run: Dict[str, pd.Series] = {}
        keys = set()

        for run, n in nets.items():
            s = transmission_energy_month(n, year, month)
            ser_by_run[run] = s
            keys |= set(s.index.tolist())

        keys = sorted(keys)

        # If nothing available this month
        if not keys:
            msg = f"{year:04d}-{month:02d}: no transmission flow tables (unsolved network?)"
            if args.skip_empty_months:
                warn(msg + " -> skipping save")
                continue
            else:
                warn(msg + " -> saving empty plot for traceability")

        # Build dataframe and drop all-zero keys
        df = pd.DataFrame({run: ser_by_run[run].reindex(keys).fillna(0.0) for run in ser_by_run.keys()})
        df = df.loc[df.abs().sum(axis=1) != 0.0]

        if df.empty:
            msg = f"{year:04d}-{month:02d}: metrics exist but all are zero"
            if args.skip_empty_months:
                warn(msg + " -> skipping save")
                continue
            else:
                warn(msg + " -> saving empty plot for traceability")

        # Convert to TWh (assuming weights are hours); if weights differ, it's still comparable proxy
        df_twh = df / 1e6

        mdir = ensure_dir(out_dir / f"{year:04d}-{month:02d}")

        runs = df_twh.columns.tolist()
        metrics = df_twh.index.tolist()
        x = np.arange(len(metrics))
        wbar = 0.8 / max(1, len(runs))

        fig, ax = plt.subplots(figsize=(13, 5))
        for i, run in enumerate(runs):
            y = df_twh[run].values
            color = cfg.colors.get(run, None)
            ax.bar(x + (i - (len(runs) - 1) / 2) * wbar, y, width=wbar, label=run, color=color)

        ax.set_title(f"Transmission energy (sum |flow|) — {year}-{month:02d}")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Energy [TWh]")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

        save_fig(fig, mdir / "transmission_energy__compare.jpeg")

    print("[OK] Networks used:")
    for run, fp in loaded.items():
        print(f"  - {run}: {fp}")
    print(f"[OK] Transmission plots in {out_dir}")


if __name__ == "__main__":
    main()
