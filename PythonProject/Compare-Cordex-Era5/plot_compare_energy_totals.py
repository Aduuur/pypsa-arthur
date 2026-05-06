# plot_compare_energy_totals.py
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from compare_config import CompareConfig
from compare_utils import ensure_dir, save_fig, warn


def read_energy_totals(fp: Path) -> pd.Series:
    df = pd.read_csv(fp, index_col=0)
    if "carrier" in df.columns:
        df = df.set_index("carrier")
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError(f"No numeric columns in {fp}")
    # pick first numeric column (robust)
    s = num.iloc[:, 0].copy()
    s.index = s.index.astype(str)
    return s.groupby(level=0).sum().sort_index()


def main():
    cfg = CompareConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(cfg.out_dir), help="output directory")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out))
    series = {}
    for run, rdir in cfg.run_dirs.items():
        fp = rdir / "energy_totals.csv"
        if not fp.exists():
            warn(f"Missing {fp}, skipping run={run}")
            continue
        series[run] = read_energy_totals(fp)

    if not series:
        raise SystemExit("No energy_totals.csv found for any run.")

    cats = sorted(set().union(*[s.index.tolist() for s in series.values()]))

    # order by magnitude
    score = {c: np.nanmean([abs(float(series[r].get(c, np.nan))) for r in series]) for c in cats}
    order = sorted(cats, key=lambda c: score[c], reverse=True)

    runs = list(series.keys())
    x = np.arange(len(order))
    w = 0.8 / max(1, len(runs))

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, run in enumerate(runs):
        y = series[run].reindex(order).fillna(0.0).values
        ax.bar(x + (i - (len(runs)-1)/2)*w, y, width=w, label=run, color=cfg.colors.get(run))

    ax.set_title("energy_totals.csv — Vergleich der Runs")
    ax.set_xlabel("Kategorie")
    ax.set_ylabel("Wert (Jahresaggregation)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    save_fig(fig, out_dir / "energy_totals__compare.jpeg")
    print(f"[OK] {out_dir / 'energy_totals__compare.jpeg'}")


if __name__ == "__main__":
    main()
