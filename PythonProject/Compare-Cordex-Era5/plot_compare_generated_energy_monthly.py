# plot_compare_generated_energy_monthly.py
# Robust version:
# - handles non-datetime snapshots + index mismatches
# - handles unsolved networks (missing generators_t.p / storage_units_t.p) gracefully
# - aligns weights to month slice safely
# - drops all-zero / all-NaN categories so plots don't look empty
# - optional debug diagnostics
# - avoids relying on compare_utils.month_mask / safe_group_sum (keeps behavior but more defensive)

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
    return (idx.year == year) & (idx.month == month)


def _safe_df(df: Optional[pd.DataFrame], idx: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
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

    if w is None:
        return pd.Series(1.0, index=idx, dtype=float)

    if isinstance(w, pd.DataFrame):
        w = w.iloc[:, 0] if w.shape[1] else pd.Series(1.0, index=idx, dtype=float)

    if not isinstance(w, pd.Series):
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


def _safe_group_sum(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Group-sum robustly, aligning indices and dropping NaN groups."""
    if values is None or values.empty:
        return pd.Series(dtype=float)
    if groups is None or groups.empty:
        g = pd.Series("unknown", index=values.index)
    else:
        g = groups.reindex(values.index)
        g = g.astype(str).fillna("unknown")
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s = v.groupby(g).sum()
    s = s[s != 0.0]
    return s.astype(float)


def _energy_from_power(p: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Return energy per asset: sum_t p(t,asset)*w(t)."""
    if p is None or not isinstance(p, pd.DataFrame) or p.empty:
        return pd.Series(dtype=float)
    p = p.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # align weights
    w = w.reindex(p.index).fillna(1.0)
    e = (p.mul(w, axis=0)).sum(axis=0)
    e = pd.to_numeric(e, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return e


# -----------------------------
# Core
# -----------------------------
def gen_energy_by_carrier_month(n, year: int, month: int) -> pd.Series:
    """
    Generated energy by carrier for a month.
    Uses:
      - generators_t.p  (MW)  -> MWh via snapshot weights
      - storage_units_t.p split into discharge/charge (MW) -> MWh
    Returns Series in MWh (weighted), not scaled.
    """
    idx = _to_datetime_index(getattr(n, "snapshots", []))
    if len(idx) == 0:
        return pd.Series(dtype=float)

    m = _month_mask(idx, year, month)
    if not m.any():
        return pd.Series(dtype=float)

    idxm = idx[m]
    w = _safe_weights(n, idx).loc[idxm]

    out: Dict[str, float] = {}

    # ---- Generators ----
    gt = getattr(n, "generators_t", None)
    gdf = getattr(n, "generators", None)
    if gt is not None and isinstance(gdf, pd.DataFrame) and not gdf.empty:
        p = getattr(gt, "p", None)
        p = _safe_df(p, idx)
        if p is not None and not p.empty:
            pm = p.loc[idxm]
            e = _energy_from_power(pm, w)  # MWh per generator
            if not e.empty and e.abs().sum() > 0:
                carriers = gdf["carrier"] if "carrier" in gdf.columns else pd.Series("generator", index=gdf.index)
                s = _safe_group_sum(e, carriers)
                for k, v in s.items():
                    out[str(k)] = out.get(str(k), 0.0) + float(v)

    # ---- Storage Units (charge/discharge) ----
    sut = getattr(n, "storage_units_t", None)
    sudf = getattr(n, "storage_units", None)
    if sut is not None and isinstance(sudf, pd.DataFrame) and not sudf.empty:
        p = getattr(sut, "p", None)
        p = _safe_df(p, idx)
        if p is not None and not p.empty:
            pm = p.loc[idxm].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            carriers = sudf["carrier"] if "carrier" in sudf.columns else pd.Series("storage_unit", index=sudf.index)

            discharge = pm.clip(lower=0.0)
            charge = (-pm.clip(upper=0.0))

            e_dis = _energy_from_power(discharge, w)
            e_chg = _energy_from_power(charge, w)

            if not e_dis.empty and e_dis.sum() != 0.0:
                s_dis = _safe_group_sum(e_dis, carriers)
                for k, v in s_dis.items():
                    key = f"{k} discharge"
                    out[key] = out.get(key, 0.0) + float(v)

            if not e_chg.empty and e_chg.sum() != 0.0:
                s_chg = _safe_group_sum(e_chg, carriers)
                for k, v in s_chg.items():
                    key = f"{k} charge"
                    out[key] = out.get(key, 0.0) + float(v)

    s = pd.Series(out, dtype=float)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s = s[s != 0.0].sort_values(ascending=False)
    return s


def main():
    cfg = CompareConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(cfg.out_dir))
    ap.add_argument("--year", type=int, default=cfg.year)
    ap.add_argument("--top", type=int, default=15, help="top carriers to show per month")
    ap.add_argument("--debug", action="store_true", help="print per-run diagnostics")
    ap.add_argument("--skip-empty-months", action="store_true", help="do not save plots for months with no data")
    ap.add_argument("--min-nonzero-twh", type=float, default=0.0, help="drop categories with abs(TWh) <= threshold")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out))
    year = int(args.year)

    nets: Dict[str, object] = {}
    loaded: Dict[str, str] = {}

    for run, rdir in cfg.run_dirs.items():
        n, fp = load_first_network(rdir)
        if n is None:
            warn(f"run={run}: no network -> skipping generated energy")
            continue
        nets[run] = n
        loaded[run] = str(fp)

        if args.debug:
            idx = _to_datetime_index(getattr(n, "snapshots", []))
            gt = getattr(n, "generators_t", None)
            sut = getattr(n, "storage_units_t", None)
            g_has_p = isinstance(getattr(gt, "p", None), pd.DataFrame) if gt is not None else False
            su_has_p = isinstance(getattr(sut, "p", None), pd.DataFrame) if sut is not None else False
            print(f"[DEBUG] {run}: file={fp}")
            print(f"[DEBUG] {run}: snapshots={len(idx)}")
            print(f"[DEBUG] {run}: generators={len(getattr(n,'generators',[]))}, storage_units={len(getattr(n,'storage_units',[]))}")
            print(f"[DEBUG] {run}: generators_t.p exists={g_has_p}, storage_units_t.p exists={su_has_p}")

    if not nets:
        raise SystemExit("No networks loaded; cannot compute generated energy.")

    for month in range(1, 13):
        ser_by_run: Dict[str, pd.Series] = {}
        for run, n in nets.items():
            s = gen_energy_by_carrier_month(n, year, month)
            ser_by_run[run] = s

        cats = sorted(set().union(*[s.index.tolist() for s in ser_by_run.values()]))
        if not cats:
            msg = f"{year:04d}-{month:02d}: no generated energy (unsolved network? no generators_t.p?)"
            if args.skip_empty_months:
                warn(msg + " -> skipping save")
                continue
            else:
                warn(msg + " -> saving empty plot for traceability")
                # still save an empty plot
                mdir = ensure_dir(out_dir / f"{year:04d}-{month:02d}")
                fig, ax = plt.subplots(figsize=(13, 5))
                ax.set_title(f"Generated energy by carrier (Top {args.top}) — {year}-{month:02d}")
                ax.set_xlabel("Carrier")
                ax.set_ylabel("Energy [TWh]")
                ax.grid(True, axis="y", alpha=0.25)
                save_fig(fig, mdir / "generated_energy__compare.jpeg")
                continue

        # pick top by max across runs (in MWh)
        maxv = pd.Series({c: max(float(ser_by_run[r].get(c, 0.0)) for r in ser_by_run) for c in cats})
        top = maxv.sort_values(ascending=False).head(args.top).index.tolist()

        # Build df for plot
        df = pd.DataFrame({run: ser_by_run[run].reindex(top).fillna(0.0) for run in ser_by_run.keys()})
        df_twh = df / 1e6  # MWh -> TWh
        if args.min_nonzero_twh > 0:
            keep = df_twh.abs().max(axis=1) > args.min_nonzero_twh
            df_twh = df_twh.loc[keep]
            top = df_twh.index.tolist()

        if df_twh.empty:
            msg = f"{year:04d}-{month:02d}: all top categories are ~0 after filtering"
            if args.skip_empty_months:
                warn(msg + " -> skipping save")
                continue
            else:
                warn(msg + " -> saving empty plot for traceability")

        mdir = ensure_dir(out_dir / f"{year:04d}-{month:02d}")

        runs = df_twh.columns.tolist()
        x = np.arange(len(top))
        wbar = 0.8 / max(1, len(runs))

        fig, ax = plt.subplots(figsize=(13, 5))
        for i, run in enumerate(runs):
            y = df_twh[run].reindex(top).fillna(0.0).values
            color = cfg.colors.get(run, None)
            ax.bar(x + (i - (len(runs) - 1) / 2) * wbar, y, width=wbar, label=run, color=color)

        ax.set_title(f"Generated energy by carrier (Top {args.top}) — {year}-{month:02d}")
        ax.set_xlabel("Carrier")
        ax.set_ylabel("Energy [TWh]")
        ax.set_xticks(x)
        ax.set_xticklabels(top, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

        save_fig(fig, mdir / "generated_energy__compare.jpeg")

    print("[OK] Networks used:")
    for run, fp in loaded.items():
        print(f"  - {run}: {fp}")
    print(f"[OK] Generated energy plots in {out_dir}")


if __name__ == "__main__":
    main()
