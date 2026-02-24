# scripts/plot_heatmap_timeseries.py
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Plot heatmap time series of marginal prices, utilisation rates, state of charge profiles.

PATCH (ARO compatibility):
- PyPSA statistics helpers sometimes return Series/DataFrames whose index is NOT a MultiIndex
  with a 'carrier' level (e.g. only asset names). The upstream script assumes `.groupby("carrier")`
  works and crashes with KeyError('carrier').
- We fix this robustly by injecting a carrier level from the network component tables when needed.
"""

import logging
import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import seaborn as sns

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config

logger = logging.getLogger(__name__)


def unstack_day_hour(
    s: pd.Series, sns: pd.DatetimeIndex, drop_leap_day: bool = True
) -> pd.DataFrame:
    s_h = s.reindex(sns).ffill()
    grouped = s_h.groupby(s_h.index.hour).agg(list)
    index = [f"{i:02d}:00" for i in grouped.index]
    columns = pd.date_range(s_h.index[0], s_h.index[-1], freq="D")
    if drop_leap_day:
        columns = columns[(columns.month != 2) | (columns.day != 29)]
    return pd.DataFrame(grouped.to_list(), index=index, columns=columns)


def plot_heatmap(
    df: pd.DataFrame,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "Greens",
    label: str = "",
    title: str = "",
    cbar_kws: dict = {},
    fn: str | None = None,
):
    _cbar_kws = dict(label=label, aspect=17, pad=0.015)
    _cbar_kws.update(cbar_kws)
    fig, ax = plt.subplots(figsize=(8.5, 4), constrained_layout=True)
    sns.heatmap(
        df,
        cmap=cmap,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar_kws=_cbar_kws,
    )
    plt.ylabel("hour of the day")
    plt.xlabel("day of the year")
    plt.title(title, fontsize="large")

    ax.grid(axis="y")

    hours = list(range(0, 24))
    ax.set_yticks(hours[0::2])
    ax.set_yticklabels(df.index[0::2], rotation=0)
    ax.set_yticks(hours, minor=True)

    major_ticks = [i for i, date in enumerate(df.columns) if date.day == 1]
    minor_ticks = [i for i, date in enumerate(df.columns) if date.day == 15]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(
        [df.columns[i].strftime("%e\n%b") for i in major_ticks], rotation=0
    )
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels(
        [df.columns[i].strftime("%e") for i in minor_ticks],
        rotation=0,
        minor=True,
        color="grey",
    )

    cb = ax.collections[0].colorbar
    cb.outline.set_linewidth(0)

    if fn is not None:
        plt.savefig(fn)
        plt.close()


# =============================================================================
# ARO compatibility helpers
# =============================================================================
def _ensure_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(
                "Expected a single-column DataFrame when converting to Series for carrier injection."
            )
        return x.iloc[:, 0]
    raise TypeError(f"Expected Series/DataFrame, got {type(x)}")


def _inject_carrier_level_from_mapping(
    s: pd.Series, carrier_by_name: pd.Series, level_name: str = "name"
) -> pd.Series:
    """
    Transform Series indexed by asset name -> MultiIndex('carrier', level_name)
    so that `.groupby('carrier')` works.
    """
    idx = pd.Index(s.index, dtype="object")
    carriers = carrier_by_name.reindex(idx).fillna("unknown").astype(str)

    s2 = s.copy()
    s2.index = pd.MultiIndex.from_arrays(
        [carriers.to_numpy(), idx.to_numpy()],
        names=["carrier", level_name],
    )
    return s2


def _groupby_carrier_sum(obj: pd.Series | pd.DataFrame, n: pypsa.Network) -> pd.Series:
    """
    Robust equivalent of: obj.groupby("carrier").sum()

    Supports:
    - Series/DataFrame with MultiIndex containing 'carrier'
    - Series indexed by component names (generators/links/storage_units/stores/lines)
    """
    if isinstance(obj, pd.DataFrame):
        # If it's a DataFrame (e.g. statistics returns df), sum over columns if needed later.
        # For our use we expect a 1D vector of capacities; force Series.
        obj = _ensure_series(obj)

    s = _ensure_series(obj)

    # If already has carrier level, just group
    if isinstance(s.index, pd.MultiIndex) and "carrier" in s.index.names:
        return s.groupby("carrier").sum()

    # Otherwise, attempt to infer which component the index refers to.
    idx = pd.Index(s.index, dtype="object")

    try:
        if idx.isin(n.generators.index).all() and "carrier" in n.generators.columns:
            return _inject_carrier_level_from_mapping(s, n.generators["carrier"]).groupby("carrier").sum()
    except Exception:
        pass

    try:
        if idx.isin(n.links.index).all() and "carrier" in n.links.columns:
            return _inject_carrier_level_from_mapping(s, n.links["carrier"]).groupby("carrier").sum()
    except Exception:
        pass

    try:
        if idx.isin(n.storage_units.index).all() and "carrier" in n.storage_units.columns:
            return _inject_carrier_level_from_mapping(s, n.storage_units["carrier"]).groupby("carrier").sum()
    except Exception:
        pass

    try:
        if idx.isin(n.stores.index).all() and "carrier" in n.stores.columns:
            return _inject_carrier_level_from_mapping(s, n.stores["carrier"]).groupby("carrier").sum()
    except Exception:
        pass

    try:
        if idx.isin(n.lines.index).all():
            if "carrier" in n.lines.columns:
                return _inject_carrier_level_from_mapping(s, n.lines["carrier"]).groupby("carrier").sum()
            # fallback: one bucket
            return pd.Series({"AC line": float(s.sum())})
    except Exception:
        pass

    # Fallback: one bucket
    return pd.Series({"unknown": float(s.sum())})


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_heatmap_timeseries",
            simpl="",
            clusters="10",
            opts="",
            sector_opts="",
            planning_horizons=2050,
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    plt.style.use(["bmh", snakemake.input.rc])

    config = snakemake.params.plotting["heatmap_timeseries"]
    drop_leap_day = snakemake.params.drop_leap_day

    output_dir = snakemake.output[0]
    os.makedirs(output_dir, exist_ok=True)

    n = pypsa.Network(snakemake.input.network)

    snapshots = get_snapshots(snakemake.params.snapshots, drop_leap_day)
    carriers = n.carriers

    diffs = snapshots.to_series().diff().dropna()
    if any(diffs > pd.Timedelta("30D")):
        logger.warning("Snapshots contain a gap longer than 1 month. Skipping heatmaps.")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # filter for built capacities
    # -------------------------------------------------------------------------
    # BEFORE (breaks under ARO sometimes):
    # optimal_capacity = n.statistics.optimal_capacity(nice_names=False).groupby("carrier").sum()
    # AFTER:
    optimal_capacity_raw = n.statistics.optimal_capacity(nice_names=False)
    optimal_capacity = _groupby_carrier_sum(optimal_capacity_raw, n)
    built_idx = optimal_capacity.where(optimal_capacity > 100).dropna().index

    # -------------------------------------------------------------------------
    # utilisation rates
    # -------------------------------------------------------------------------
    cf = (
        n.statistics.capacity_factor(aggregate_time=False, nice_names=False)
        .dropna()
    )

    # In upstream, this expects a MultiIndex with 'carrier'. Make it robust:
    # We want cf as DataFrame indexed by carrier, with columns being snapshots.
    if isinstance(cf, pd.DataFrame):
        # If cf index has carrier level -> group as expected
        if isinstance(cf.index, pd.MultiIndex) and "carrier" in cf.index.names:
            cf = cf.groupby("carrier").sum()
        else:
            # If indexed by asset name, we cannot faithfully reconstruct per-carrier time series
            # for all possible components without deeper knowledge. We do a best-effort:
            # - if index matches generators, map to generator carriers
            idx = pd.Index(cf.index, dtype="object")
            if idx.isin(n.generators.index).all() and "carrier" in n.generators.columns:
                cf.index = pd.MultiIndex.from_arrays(
                    [n.generators["carrier"].reindex(idx).fillna("unknown").astype(str).to_numpy(), idx.to_numpy()],
                    names=["carrier", "name"],
                )
                cf = cf.groupby("carrier").sum()
            elif idx.isin(n.links.index).all() and "carrier" in n.links.columns:
                cf.index = pd.MultiIndex.from_arrays(
                    [n.links["carrier"].reindex(idx).fillna("unknown").astype(str).to_numpy(), idx.to_numpy()],
                    names=["carrier", "name"],
                )
                cf = cf.groupby("carrier").sum()
            else:
                logger.warning(
                    "capacity_factor returned an unexpected index without 'carrier'; "
                    "skipping utilisation rate heatmaps."
                )
                cf = pd.DataFrame()
    else:
        logger.warning("capacity_factor returned unexpected type; skipping utilisation rate heatmaps.")
        cf = pd.DataFrame()

    if not cf.empty:
        cf = cf.mul(100)
        idx = pd.Index(cf.index).intersection(config["utilisation_rate"]).intersection(built_idx)
        cf = cf.loc[idx]

        for carrier, s in cf.iterrows():
            logger.info(f"Plotting utilisation rate heatmap time series for {carrier}")
            df = unstack_day_hour(s, snapshots, drop_leap_day)
            label = "utilisation rate [%]"
            fn = output_dir + "/ts-heatmap-utilisation_rate-" + carrier.replace(" ", "_") + ".pdf"
            plot_heatmap(
                df,
                cmap="Greens",
                label=label,
                title=carrier,
                vmin=0,
                vmax=100,
                fn=fn,
            )

    # -------------------------------------------------------------------------
    # marginal prices (unchanged; already groups by n.buses.carrier)
    # -------------------------------------------------------------------------
    prices = n.buses_t.marginal_price.T.groupby(n.buses.carrier).mean()
    prices = prices.loc[prices.index.intersection(config["marginal_price"])]

    for carrier, s in prices.iterrows():
        logger.info(f"Plotting marginal prices heatmap time series for {carrier}")
        df = unstack_day_hour(s, snapshots, drop_leap_day)
        label = (
            "marginal price [€/t]"
            if "co2" in carrier.lower()
            else "marginal price [€/MWh]"
        )
        fn = output_dir + "/ts-heatmap-marginal_price-" + carrier.replace(" ", "_") + ".pdf"
        plot_heatmap(
            df,
            cmap="Spectral_r",
            label=label,
            title=carrier,
            fn=fn,
        )

    # -------------------------------------------------------------------------
    # SOCs (stores) (unchanged logic; but keep robust if carrier column missing)
    # -------------------------------------------------------------------------
    if len(n.stores.index) == 0:
        sys.exit(0)

    if "carrier" not in n.stores.columns:
        logger.warning("n.stores has no 'carrier' column; skipping SOC heatmaps.")
        sys.exit(0)

    e_nom_opt = n.stores.groupby("carrier").e_nom_opt.sum()
    socs = n.stores_t.e.T.groupby(n.stores.carrier).sum().div(e_nom_opt, axis=0) * 100
    socs = socs.loc[socs.index.intersection(config["soc"]).intersection(built_idx)]

    for carrier, s in socs.iterrows():
        logger.info(f"Plotting SOC heatmap time series for {carrier}")
        df = unstack_day_hour(s, snapshots, drop_leap_day)
        label = "SOC [%]"
        fn = output_dir + "/ts-heatmap-soc-" + carrier.replace(" ", "_") + ".pdf"
        plot_heatmap(
            df,
            cmap="Blues",
            vmin=0,
            vmax=100,
            label=label,
            title=carrier,
            fn=fn,
        )