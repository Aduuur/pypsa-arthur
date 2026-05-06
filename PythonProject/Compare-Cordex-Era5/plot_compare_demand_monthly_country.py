# plot_compare_demand_monthly_country.py
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from compare_config import CompareConfig
from compare_utils import (
    ensure_dir, save_fig, format_time_axis,
    robust_read_csv_timeindexed, drop_countries, group_cols_by_country,
    pick_units_from_cols, ylabel, warn
)

EXCLUDE_COUNTRIES = {"CH"}  # wie gewünscht überall raus


def resolve_demand_file(run_dir: Path) -> Path:
    a = run_dir / "electricity_demand_non_hist.csv"
    b = run_dir / "electricity_demand.csv"
    if a.exists():
        return a
    if b.exists():
        return b
    raise FileNotFoundError(f"No demand CSV in {run_dir}")


def main():
    cfg = CompareConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(cfg.out_dir), help="output directory")
    ap.add_argument("--year", type=int, default=cfg.year)
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out))
    year = args.year

    df_by_run = {}
    for run, rdir in cfg.run_dirs.items():
        try:
            fp = resolve_demand_file(rdir)
            df = robust_read_csv_timeindexed(fp, year=year)
            df = drop_countries(df, EXCLUDE_COUNTRIES)
            df_by_run[run] = df
        except Exception as e:
            warn(f"run={run}: demand read failed ({e})")

    if not df_by_run:
        raise SystemExit("No demand data for any run.")

    units = pick_units_from_cols(list(next(iter(df_by_run.values())).columns))
    ylab = ylabel("Electricity demand", units)

    # union columns -> per-country
    all_cols = sorted(set().union(*[set(df.columns) for df in df_by_run.values()]))
    cols_by_country = group_cols_by_country(all_cols, EXCLUDE_COUNTRIES)

    for month in range(1, 13):
        mdir = ensure_dir(out_dir / f"{year:04d}-{month:02d}")
        cdir = ensure_dir(mdir / "countries")

        # ALL plot (sum over cols)
        fig, ax = plt.subplots(figsize=(13, 5))
        any_ = False
        for run, df in df_by_run.items():
            s = df.sum(axis=1)
            sm = s[(s.index.year == year) & (s.index.month == month)]
            if sm.empty:
                continue
            ax.plot(sm.index, sm.values, color=cfg.colors.get(run), linestyle="-", linewidth=1.7, alpha=0.95, label=run)
            any_ = True
        ax.set_title(f"Demand (sum, ohne CH) — {year}-{month:02d}")
        ax.set_xlabel("Zeit")
        ax.set_ylabel(ylab)
        format_time_axis(ax)
        if any_:
            ax.legend()
        save_fig(fig, mdir / "electricity_demand__ALL__compare.jpeg")

        # per-country plots (sum within country)
        for cc, cols in sorted(cols_by_country.items()):
            fig, ax = plt.subplots(figsize=(13, 5))
            any_ = False
            for run, df in df_by_run.items():
                use = [c for c in cols if c in df.columns]
                if not use:
                    continue
                s = df[use].sum(axis=1)
                sm = s[(s.index.year == year) & (s.index.month == month)]
                if sm.empty:
                    continue
                ax.plot(sm.index, sm.values, color=cfg.colors.get(run), linestyle="-", linewidth=1.7, alpha=0.95, label=run)
                any_ = True
            ax.set_title(f"Demand — {cc} — {year}-{month:02d}")
            ax.set_xlabel("Zeit")
            ax.set_ylabel(ylab)
            format_time_axis(ax)
            if any_:
                ax.legend()
            save_fig(fig, cdir / f"electricity_demand__{cc}__compare.jpeg")

    print(f"[OK] Demand plots in {out_dir}")


if __name__ == "__main__":
    main()
