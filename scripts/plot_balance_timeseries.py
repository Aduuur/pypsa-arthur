# scripts/plot_balance_timeseries.py
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Plot balance time series.

PATCH (ARO compatibility):
- In ARO, snapshots are often NOT a plain DatetimeIndex (can be MultiIndex or other Index).
  Pandas resample() then crashes: "Only valid with DatetimeIndex ... but got Index".
- Fix: coerce the plotted dataframe index to a DatetimeIndex (best-effort) before resampling.
- Also make monthly slicing robust (derive months from the coerced index, not from snakemake params).
- Force non-interactive matplotlib backend (avoids Qt/wayland issues on headless nodes).
"""

import logging
import os
from functools import partial
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")  # headless-safe backend (prevents Qt/wayland plugin issues)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from tqdm import tqdm

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers: ARO snapshot index -> DatetimeIndex coercion
# =============================================================================
def _coerce_to_datetime_index(idx: pd.Index) -> pd.DatetimeIndex:
    """
    Best-effort conversion of an arbitrary Index into a DatetimeIndex.

    Handles:
    - DatetimeIndex: pass-through
    - PeriodIndex: to_timestamp()
    - MultiIndex: pick a level that is (or can be) datetime (prefer last levels)
    - plain Index of tuples/strings: try to_datetime() directly

    Raises ValueError if it cannot obtain a usable DatetimeIndex.
    """
    if isinstance(idx, pd.DatetimeIndex):
        return idx

    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp()

    if isinstance(idx, pd.MultiIndex):
        # Prefer last level(s), since ARO often uses (scenario, datetime) or (period, timestep)
        for i in reversed(range(idx.nlevels)):
            v = idx.get_level_values(i)
            # already datetime-like?
            if pd.api.types.is_datetime64_any_dtype(v):
                return pd.DatetimeIndex(v)
            # try conversion
            dt = pd.to_datetime(v, errors="coerce")
            if dt.notna().mean() > 0.95:
                return pd.DatetimeIndex(dt)
        # last resort: try to_datetime on the tuple representation
        dt = pd.to_datetime(idx.astype(str), errors="coerce")
        if dt.notna().mean() > 0.95:
            return pd.DatetimeIndex(dt)
        raise ValueError("Could not coerce MultiIndex to DatetimeIndex.")

    # plain Index
    dt = pd.to_datetime(idx, errors="coerce")
    if dt.notna().mean() > 0.95:
        return pd.DatetimeIndex(dt)

    # maybe objects like "(scenario, 2050-01-01 00:00:00)"
    dt = pd.to_datetime(idx.astype(str), errors="coerce")
    if dt.notna().mean() > 0.95:
        return pd.DatetimeIndex(dt)

    raise ValueError("Could not coerce Index to DatetimeIndex.")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df.index is a DatetimeIndex; sort it (resample expects monotonic increasing index).
    If coercion fails, return df unchanged (and caller decides to skip plotting).
    """
    try:
        dt = _coerce_to_datetime_index(df.index)
    except Exception as e:
        logger.warning(f"Could not convert balance dataframe index to DatetimeIndex: {e}")
        return df

    out = df.copy()
    out.index = dt
    out = out.sort_index()
    return out


def _month_slices(idx: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    """
    Return a list of DatetimeIndex objects, one per calendar month contained in idx.
    """
    if len(idx) == 0:
        return []
    periods = idx.to_period("M").unique()
    return [idx[idx.to_period("M") == p] for p in periods]


# =============================================================================
# Plotting
# =============================================================================
def plot_stacked_area_steplike(
    ax: plt.Axes, df: pd.DataFrame, colors: dict | pd.Series = {}
):
    """Plot stacked area chart with step-like transitions."""
    if isinstance(colors, pd.Series):
        colors = colors.to_dict()

    df_cum = df.cumsum(axis=1)
    previous_series = np.zeros_like(df_cum.iloc[:, 0].values)

    for col in df_cum.columns:
        ax.fill_between(
            df_cum.index,
            previous_series,
            df_cum[col],
            step="pre",
            linewidth=0,
            color=colors.get(col, "grey"),
            label=col,
        )
        previous_series = df_cum[col].values


def setup_time_axis(ax: plt.Axes, timespan: pd.Timedelta):
    """Configure time axis formatting based on timespan."""
    long_time_frame = timespan > pd.Timedelta(weeks=5)

    if not long_time_frame:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%e\n%b"))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%e"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%e\n%b"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%e"))

    ax.tick_params(axis="x", which="minor", labelcolor="grey")


def plot_energy_balance_timeseries(
    df: pd.DataFrame,
    time: pd.DatetimeIndex | None = None,
    ylim: float | None = None,
    resample: str | None = None,
    rename: dict = {},
    preferred_order: pd.Index | list = [],
    ylabel: str = "",
    colors: dict | pd.Series = {},
    max_threshold: float = 0.0,
    mean_threshold: float = 0.0,
    directory="",
):
    """Create energy balance time series plot with positive/negative stacked areas."""

    # Ensure datetime index for slicing/resampling
    df = _ensure_datetime_index(df)

    if time is not None:
        # time must be a DatetimeIndex (we enforce this upstream)
        df = df.loc[time]

    # Handle small values and renaming
    techs_below_threshold = df.columns[
        (df.abs().max() < max_threshold) & (df.abs().mean() < mean_threshold)
    ].tolist()
    if techs_below_threshold:
        rename.update({tech: "other" for tech in techs_below_threshold})
        if isinstance(colors, dict):
            colors["other"] = "grey"

    if rename:
        df = df.T.groupby(df.columns.map(lambda a: rename.get(a, a))).sum().T

    # Upsample to hourly resolution to handle overlapping snapshots
    if resample is not None:
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(
                f"Index is not DatetimeIndex after coercion; skipping resample for '{ylabel}'."
            )
        else:
            df = df.resample("1h").ffill().resample(resample).mean()

    if df.empty:
        return

    # Sort columns by variance
    denom = df.max().replace(0.0, np.nan)
    order = (df / denom).var().sort_values().index
    if preferred_order is not None and len(preferred_order) > 0:
        if isinstance(preferred_order, list):
            preferred_order = pd.Index(preferred_order)
        order = preferred_order.intersection(order).append(order.difference(preferred_order))
    df = df.loc[:, order]

    # Split into positive and negative values
    pos = df.where(df > 0).fillna(0.0)
    neg = df.where(df < 0).fillna(0.0)

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    plot_stacked_area_steplike(ax, pos, colors)
    plot_stacked_area_steplike(ax, neg, colors)

    # Set x and y limits
    plt.xlim((df.index[0], df.index[-1]))
    if isinstance(df.index, pd.DatetimeIndex):
        setup_time_axis(ax, df.index[-1] - df.index[0])

    # Configure y-axis and grid
    ax.grid(axis="y")
    ax.axhline(0, color="grey", linewidth=0.5)

    if ylim is None:
        # ensure y-axis extent is symmetric around origin in steps of 50 units
        ylim = np.ceil(max(-neg.sum(axis=1).min(), pos.sum(axis=1).max()) / 50) * 50
    plt.ylim([-ylim, ylim])

    # Set labels and legend
    unit = "kt/h" if "co2" in ylabel.lower() else "GW"
    plt.ylabel(f"{ylabel} balance [{unit}]")

    # half the labels because pos and neg create duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    half = int(len(handles) / 2)
    fig.legend(
        handles=handles[:half],
        labels=labels[:half],
        loc="outside right upper",
        fontsize=6,
    )

    # Save figures
    if resample is None:
        resample = "native"
    fn = f"ts-balance-{ylabel.replace(' ', '_')}-{resample}.pdf"
    plt.savefig(f"{directory}/{fn}")
    plt.close()


def process_carrier(group_item, balance, colors, config, output_dir):
    """Process carrier data and create plots for specific carrier group."""

    group, carriers = group_item
    if not isinstance(carriers, list):
        carriers = [carriers]

    # balance: typically indexed by (bus_carrier, carrier, ...) and columns are snapshots
    mask = balance.index.get_level_values("bus_carrier").isin(carriers)
    df = balance[mask].groupby("carrier").sum().div(1e3).T

    if df.empty:
        logger.warning(
            f"No carriers of group '{group}' in energy balance. Skipping carrier group: '{group}'"
        )
        return

    # Ensure datetime index early (needed for resample + monthly slicing)
    df = _ensure_datetime_index(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning(
            f"Balance timeseries index is not datetime-like for group '{group}'. Skipping plots."
        )
        return

    kwargs = dict(
        ylabel=group,
        colors=colors,
        max_threshold=config["max_threshold"],
        mean_threshold=config["mean_threshold"],
        directory=output_dir,
    )

    # annual plot (resampled)
    if config.get("annual", False):
        plot_energy_balance_timeseries(df, resample=config["annual_resolution"], **kwargs)

    # monthly plots (native slice + resample)
    if config.get("monthly", False):
        for month_idx in _month_slices(df.index):
            plot_energy_balance_timeseries(
                df, resample=config["monthly_resolution"], time=month_idx, **kwargs
            )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_balance_timeseries",
            simpl="",
            clusters="10",
            opts="",
            sector_opts="",
            planning_horizons=2050,
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    plt.style.use(["bmh", snakemake.input.rc])

    # Load network and prepare data
    n = pypsa.Network(snakemake.input.network)
    config = snakemake.params.plotting["balance_timeseries"]
    output_dir = snakemake.output[0]
    os.makedirs(output_dir, exist_ok=True)

    # Calculate energy balance
    balance = n.statistics.energy_balance(aggregate_time=False, nice_names=False)

    # Get colors for carriers
    n.carriers.update({"color": snakemake.params.plotting["tech_colors"]})
    colors = n.carriers.color.copy().replace("", "grey")

    # Setup carrier groups for plotting
    groups = config["carrier_groups"]
    groups.update({c: [c] for c in config["carriers"]})

    missing_bus_carriers = set(config["carriers"]).difference(n.buses.carrier.unique())
    if missing_bus_carriers:
        logger.warning(f"Skipping missing bus carriers: {missing_bus_carriers}")
    groups = {k: v for k, v in groups.items() if k not in missing_bus_carriers}

    # Process each carrier group in parallel
    threads = snakemake.threads
    tqdm_kwargs = dict(
        ascii=False,
        unit=" carrier",
        total=len(groups),
        desc="Plotting carrier balance time series",
    )
    func = partial(
        process_carrier,
        balance=balance,
        colors=colors,
        config=config,
        output_dir=output_dir,
    )
    with Pool(processes=min(threads, len(groups))) as pool:
        list(tqdm(pool.imap(func, groups.items()), **tqdm_kwargs))