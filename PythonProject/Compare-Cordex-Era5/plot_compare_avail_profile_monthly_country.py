# plot_compare_avail_profile_monthly_country.py
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

EXCLUDE_COUNTRIES = {"CH"}

AVAIL_STYLE = {
    "cordex_rcp26": {"ls": "--", "mk": "o", "me": 24, "a": 0.80, "lw": 1.2},
    "cordex_rcp45": {"ls": ":",  "mk": "s", "me": 24, "a": 0.80, "lw": 1.4},
    "era5":         {"ls": "-",  "mk": None,"me": None,"a": 0.95, "lw": 1.8},
}


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
        fp = rdir / "avail_profile_s_15.csv"
        if not fp.exists():
            warn(f"Missing {fp}")
            continue
        try:
            df = robust_read_csv_timeindexed(fp, year=year)
            df = drop_countries(df, EXCLUDE_COUNTRIES)
            df_by_run[run] = df
        except Exception as e:
            warn(f"run={run}: avail_profile read failed ({e})")

    if not df_by_run:
        raise SystemExit("No avail_profile data for any run.")

    units = pick_units_from_cols(list(next(iter(df_by_run.values())).columns))
    ylab = ylabel("Availability profile", units)

    all_cols = sorted(set().union(*[set(df.columns) for df in df_by_run.values()]))
    cols_by_country = group_cols_by_country(all_cols, EXCLUDE_COUNTRIES)

    def plot_line(ax, run, x, y):
        st = AVAIL_STYLE.get(run, {"ls": "-", "mk": None, "me": None, "a": 0.9, "lw": 1.6})
        ax.plot(
            x, y,
            color=cfg.colors.get(run),
            linestyle=st["ls"], linewidth=st["lw"], alpha=st["a"],
            marker=st["mk"], markevery=st["me"], markersize=3 if st["mk"] else None,
            label=run
        )

    for month in range(1, 13):
        mdir = ensure_dir(out_dir / f"{year:04d}-{month:02d}")
        cdir = ensure_dir(mdir / "countries")

        # ALL (mean over cols)
        fig, ax = plt.subplots(figsize=(13, 5))
        any_ = False
        for run, df in df_by_run.items():
            s = df.mean(axis=1)
            sm = s[(s.index.year == year) & (s.index.month == month)]
            if sm.empty:
                continue
            plot_line(ax, run, sm.index, sm.values)
            any_ = True
        ax.set_title(f"Avail profile (mean, ohne CH) — {year}-{month:02d}")
        ax.set_xlabel("Zeit")
        ax.set_ylabel(ylab)
        format_time_axis(ax)
        if any_:
            ax.legend()
        save_fig(fig, mdir / "avail_profile_s_15__ALL__compare.jpeg")

        # per-country (mean within country)
        for cc, cols in sorted(cols_by_country.items()):
            fig, ax = plt.subplots(figsize=(13, 5))
            any_ = False
            for run, df in df_by_run.items():
                use = [c for c in cols if c in df.columns]
                if not use:
                    continue
                s = df[use].mean(axis=1)
                sm = s[(s.index.year == year) & (s.index.month == month)]
                if sm.empty:
                    continue
                plot_line(ax, run, sm.index, sm.values)
                any_ = True
            ax.set_title(f"Avail profile — {cc} — {year}-{month:02d}")
            ax.set_xlabel("Zeit")
            ax.set_ylabel(ylab)
            format_time_axis(ax)
            if any_:
                ax.legend()
            save_fig(fig, cdir / f"avail_profile_s_15__{cc}__compare.jpeg")

    print(f"[OK] Avail profile plots in {out_dir}")


if __name__ == "__main__":
    main()
