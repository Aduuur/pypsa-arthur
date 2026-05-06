# plot_compare_installed_capacity_from_network.py
# Robust version: defensive loading + type coercion + NaN handling + explicit diagnostics
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from compare_config import CompareConfig
from compare_utils import ensure_dir, save_fig, load_first_network, warn


def _numeric_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Return df[col] as numeric series (coerce errors), or None if col missing."""
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    # treat inf as NaN
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def _cap_series(
    df: pd.DataFrame,
    pref_cols=("p_nom_opt", "p_nom"),
    allow_fallbacks=True,
) -> Optional[pd.Series]:
    """
    Try to extract a capacity series robustly.

    Priority:
      1) p_nom_opt
      2) p_nom
    Optional fallbacks (if allow_fallbacks):
      - p_nom_max (if p_nom is all-NaN/0 in some exported networks)
      - p_nom_min (rare, but better than empty)
    """
    if df is None or len(df) == 0:
        return None

    for c in pref_cols:
        s = _numeric_series(df, c)
        if s is None:
            continue
        # If series is entirely NaN, try next
        if s.notna().sum() == 0:
            continue
        return s

    if allow_fallbacks:
        for c in ("p_nom_max", "p_nom_min"):
            s = _numeric_series(df, c)
            if s is None:
                continue
            if s.notna().sum() == 0:
                continue
            return s

    return None


def _sum_by_carrier(cap: pd.Series, carrier: Optional[pd.Series], default_key: str) -> pd.Series:
    """Group cap by carrier robustly; missing carriers go to default_key."""
    if cap is None or cap.size == 0:
        return pd.Series(dtype=float)

    cap = cap.fillna(0.0)
    if carrier is None or carrier.size == 0:
        key = pd.Series(default_key, index=cap.index)
    else:
        key = carrier.astype(str).fillna(default_key)
    s = cap.groupby(key).sum()
    # drop keys that are exactly 0 (prevents "empty-looking" plots)
    s = s[s != 0.0]
    return s.astype(float)


def installed_capacity_by_carrier(n) -> pd.Series:
    """
    Installed capacities (MW) aggregated by carrier across:
      - Generators (p_nom_opt/p_nom)
      - Links (p_nom_opt/p_nom)
      - StorageUnits (p_nom_opt/p_nom)
    Plus Stores energy capacity (MWh) as '<carrier> (E)' from e_nom_opt/e_nom.
    """
    out: Dict[str, float] = {}
    store_energy: Dict[str, float] = {}

    # ---- Generators ----
    if hasattr(n, "generators") and isinstance(n.generators, pd.DataFrame) and len(n.generators) > 0:
        g = n.generators
        cap = _cap_series(g)
        if cap is None:
            warn(f"generators: no usable capacity column found. cols={list(g.columns)[:40]}")
        else:
            s = _sum_by_carrier(cap, g["carrier"] if "carrier" in g.columns else None, "generator")
            for k, v in s.items():
                out[str(k)] = out.get(str(k), 0.0) + float(v)

    # ---- Links ----
    if hasattr(n, "links") and isinstance(n.links, pd.DataFrame) and len(n.links) > 0:
        l = n.links
        cap = _cap_series(l)
        if cap is None:
            warn(f"links: no usable capacity column found. cols={list(l.columns)[:40]}")
        else:
            s = _sum_by_carrier(cap, l["carrier"] if "carrier" in l.columns else None, "link")
            for k, v in s.items():
                out[str(k)] = out.get(str(k), 0.0) + float(v)

    # ---- Storage Units ----
    if hasattr(n, "storage_units") and isinstance(n.storage_units, pd.DataFrame) and len(n.storage_units) > 0:
        su = n.storage_units
        cap = _cap_series(su)
        if cap is None:
            warn(f"storage_units: no usable capacity column found. cols={list(su.columns)[:40]}")
        else:
            s = _sum_by_carrier(cap, su["carrier"] if "carrier" in su.columns else None, "storage_unit")
            for k, v in s.items():
                out[str(k)] = out.get(str(k), 0.0) + float(v)

    # ---- Stores (energy) ----
    if hasattr(n, "stores") and isinstance(n.stores, pd.DataFrame) and len(n.stores) > 0:
        st = n.stores
        ecap = None
        for c in ("e_nom_opt", "e_nom"):
            ecap = _numeric_series(st, c)
            if ecap is not None and ecap.notna().sum() > 0:
                break
            ecap = None

        if ecap is None:
            # no warning necessary; stores are optional
            pass
        else:
            ecap = ecap.fillna(0.0)
            key = st["carrier"].astype(str).fillna("store") if "carrier" in st.columns else pd.Series("store", index=st.index)
            s = ecap.groupby(key).sum()
            s = s[s != 0.0].astype(float)
            for k, v in s.items():
                store_energy[str(k) + " (E)"] = float(v)

    ser = pd.Series(out, dtype=float).sort_values(ascending=False)
    ser_e = pd.Series(store_energy, dtype=float).sort_values(ascending=False)

    if ser.empty and ser_e.empty:
        return pd.Series(dtype=float)

    # Combine, keep MW first then MWh(E) at end
    return pd.concat([ser, ser_e])


def main():
    cfg = CompareConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(cfg.out_dir))
    ap.add_argument("--min", type=float, default=0.0, help="drop carriers with abs(value) <= min (default 0)")
    ap.add_argument("--top", type=int, default=0, help="keep only top N carriers by max across runs (0 = keep all)")
    ap.add_argument("--debug", action="store_true", help="print debug summaries per run")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out))

    caps: Dict[str, pd.Series] = {}
    loaded: Dict[str, str] = {}

    for run, rdir in cfg.run_dirs.items():
        n, fp = load_first_network(rdir)
        if n is None:
            warn(f"run={run}: no network -> skipping installed capacity")
            continue

        loaded[run] = str(fp)

        if args.debug:
            # lightweight sanity checks to catch "empty-looking" plots quickly
            try:
                gcols = list(getattr(n, "generators").columns) if hasattr(n, "generators") else []
                print(f"[DEBUG] {run}: file={fp}")
                print(f"[DEBUG] {run}: generators={len(getattr(n,'generators',[]))}, links={len(getattr(n,'links',[]))}, storage_units={len(getattr(n,'storage_units',[]))}, stores={len(getattr(n,'stores',[]))}")
                print(f"[DEBUG] {run}: generator cols has p_nom={('p_nom' in gcols)}, p_nom_opt={('p_nom_opt' in gcols)}")
            except Exception as e:
                warn(f"run={run}: debug failed: {e}")

        s = installed_capacity_by_carrier(n)

        if s.empty:
            warn(f"run={run}: installed_capacity series empty (all zeros / missing columns / no components?)")
        else:
            # drop tiny/near-zero entries
            if args.min > 0:
                s = s[s.abs() > args.min]

        caps[run] = s

    if not caps:
        raise SystemExit("No networks loaded; cannot compute installed capacities.")

    # Union carriers, but drop carriers that are zero everywhere
    carriers = sorted(set().union(*[s.index.tolist() for s in caps.values()]))
    if not carriers:
        raise SystemExit("Installed capacity empty for all runs (check network file / columns).")

    # Build wide table for robust filtering + plotting
    df = pd.DataFrame({run: caps[run].reindex(carriers).fillna(0.0) for run in caps.keys()})
    # Drop carriers that are zero across all runs (common reason for "empty" chart)
    df = df.loc[(df.abs().sum(axis=1) != 0.0)]
    if df.empty:
        raise SystemExit("All carriers are zero across runs. Plot would be empty. Check capacity columns and values.")

    # Optionally keep top N carriers by maximum across runs
    if args.top and args.top > 0 and len(df) > args.top:
        order = df.max(axis=1).sort_values(ascending=False).head(args.top).index
        df = df.loc[order]
    else:
        # default ordering: by maximum across runs
        df = df.loc[df.max(axis=1).sort_values(ascending=False).index]

    carriers = df.index.tolist()
    runs = df.columns.tolist()

    x = np.arange(len(carriers))
    w = 0.8 / max(1, len(runs))

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, run in enumerate(runs):
        y = df[run].values
        # if cfg.colors doesn't define run, let matplotlib choose
        color = cfg.colors.get(run, None)
        ax.bar(x + (i - (len(runs) - 1) / 2) * w, y, width=w, label=run, color=color)

    ax.set_title("Installed capacity by carrier (Generators+Links+StorageUnits; Stores as '(E)')")
    ax.set_xlabel("Carrier")
    ax.set_ylabel("Capacity (MW) / Energy capacity (MWh) for '(E)'")
    ax.set_xticks(x)
    ax.set_xticklabels(carriers, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    save_fig(fig, out_dir / "installed_capacity__compare.jpeg")

    print("[OK] Networks used:")
    for run, fp in loaded.items():
        print(f"  - {run}: {fp}")
    print(f"[OK] Saved: {out_dir / 'installed_capacity__compare.jpeg'}")
    print(f"[OK] Nonzero carriers plotted: {len(carriers)}")


if __name__ == "__main__":
    main()
