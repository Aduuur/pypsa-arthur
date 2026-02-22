# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Create summary CSV files for all scenario runs including costs, capacities,
capacity factors, curtailment, energy balances, prices and other metrics.
"""

import logging

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config

idx = pd.IndexSlice
logger = logging.getLogger(__name__)

OUTPUTS = [
    "costs",
    "capacities",
    "energy",
    "energy_balance",
    "capacity_factors",
    "metrics",
    "curtailment",
    "prices",
    "weighted_prices",
    "market_values",
    "nodal_costs",
    "nodal_capacities",
    "nodal_energy_balance",
    "nodal_capacity_factors",
]


def assign_carriers(n: pypsa.Network) -> None:
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"


def assign_locations(n: pypsa.Network) -> None:
    for c in n.iterate_components(n.one_port_components):
        c.df["location"] = c.df.bus.map(n.buses.location)

    for c in n.iterate_components(n.branch_components):
        c_bus_cols = c.df.filter(regex="^bus")
        locs = c_bus_cols.apply(lambda c: c.map(n.buses.location)).sort_index(axis=1)
        # Use first location that is not "EU"; take "EU" if nothing else available
        c.df["location"] = locs.apply(
            lambda row: next(
                (loc for loc in row.dropna() if loc != "EU"),
                "EU",
            ),
            axis=1,
        )


def calculate_nodal_capacity_factors(n: pypsa.Network) -> pd.Series:
    """
    Calculate the regional dispatched capacity factors / utilisation rates for each technology carrier based on location bus attribute.
    """
    comps = n.one_port_components ^ {"Store"} | n.passive_branch_components
    return n.statistics.capacity_factor(comps=comps, groupby=["location", "carrier"])


def calculate_capacity_factors(n: pypsa.Network) -> pd.Series:
    """
    Calculate the average dispatched capacity factors / utilisation rates for each technology carrier.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["component", "carrier"]
    """

    comps = n.one_port_components ^ {"Store"} | n.passive_branch_components
    return n.statistics.capacity_factor(comps=comps).sort_index()


def calculate_nodal_costs(n: pypsa.Network) -> pd.Series:
    """
    Calculate optimized regional costs for each technology split by marginal and capital costs and based on location bus attribute.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["cost", "component", "location", "carrier"]
    """
    grouper = ["location", "carrier"]
    costs = pd.concat(
        {
            "capital": n.statistics.capex(groupby=grouper),
            "marginal": n.statistics.opex(groupby=grouper),
        }
    )
    costs.index.names = ["cost", "component", "location", "carrier"]

    return costs


def calculate_costs(n: pypsa.Network) -> pd.Series:
    """
    Calculate optimized total costs for each technology split by marginal and capital costs.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["cost", "component", "carrier"]
    """
    costs = pd.concat(
        {
            "capital": n.statistics.capex(),
            "marginal": n.statistics.opex(),
        }
    )
    costs.index.names = ["cost", "component", "carrier"]

    return costs


def calculate_nodal_capacities(n: pypsa.Network) -> pd.Series:
    """
    Calculate optimized regional capacities for each technology relative to bus/bus0 based on location bus attribute.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["component", "location", "carrier"]
    """
    return n.statistics.optimal_capacity(groupby=["location", "carrier"])


def calculate_capacities(n: pypsa.Network) -> pd.Series:
    """
    Calculate optimized total capacities for each technology relative to bus/bus0.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["component", "carrier"]
    """
    return n.statistics.optimal_capacity()


def calculate_curtailment(n: pypsa.Network) -> pd.Series:
    """
    Calculate the curtailment of electricity generation technologies in percent.
    """

    carriers = ["AC", "low voltage"]

    duration = n.snapshot_weightings.generators.sum()

    curtailed_abs = n.statistics.curtailment(
        bus_carrier=carriers, aggregate_across_components=True
    )
    available = (
        n.statistics.optimal_capacity("Generator", bus_carrier=carriers) * duration
    )

    curtailed_rel = curtailed_abs / available * 100

    return curtailed_rel.sort_index()


def calculate_energy(n: pypsa.Network) -> pd.Series:
    """
    Calculate the net energy supply (positive) and consumption (negative) by technology carrier across all ports.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["component", "carrier"]
    """
    return n.statistics.energy_balance(groupby="carrier").sort_values(ascending=False)


def calculate_energy_balance(n: pypsa.Network) -> pd.Series:
    """
    Calculate the energy supply (positive) and consumption (negative) by technology carrier for each bus carrier.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["component", "carrier", "bus_carrier"]

    Examples
    --------
    >>> eb = calculate_energy_balance(n)
    >>> eb.xs("methanol", level='bus_carrier')
    """
    return n.statistics.energy_balance().sort_values(ascending=False)


def calculate_nodal_energy_balance(n: pypsa.Network) -> pd.Series:
    """
    Calculate the regional energy balances (positive values for supply, negative values for consumption) for each technology carrier and bus carrier based on the location bus attribute.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["component", "carrier", "location", "bus_carrier"]

    Examples
    --------
    >>> eb = calculate_nodal_energy_balance(n)
    >>> eb.xs(("AC", "BE0 0"), level=["bus_carrier", "location"])
    """
    return n.statistics.energy_balance(groupby=["carrier", "location", "bus_carrier"])


def calculate_metrics(n: pypsa.Network) -> pd.Series:
    """
    Calculate system-level metrics, e.g. shadow prices, grid expansion, total costs.
    Also calculate average, standard deviation and share of zero hours for electricity prices.

    This function is written to work robustly with both:
      - the standard PyPSA-Eur workflow networks, and
      - ARO/robust workflow networks (including cases where marginal_price columns
        do not match the expected AC bus index, or where some optimisation fields
        are missing).
    """
    import numpy as np
    import pandas as pd

    metrics: dict[str, object] = {}

    # -------------------------
    # Grid expansion (robust)
    # -------------------------
    # DC links: be defensive about missing columns
    if hasattr(n, "links") and not n.links.empty:
        dc_links = n.links[n.links.get("carrier", pd.Series(index=n.links.index)).eq("DC")]
        # prefer p_nom_opt, fallback to p_nom, else 0
        if "length" in dc_links.columns:
            if "p_nom_opt" in dc_links.columns:
                metrics["line_volume_DC"] = (dc_links["length"] * dc_links["p_nom_opt"]).sum()
            elif "p_nom" in dc_links.columns:
                metrics["line_volume_DC"] = (dc_links["length"] * dc_links["p_nom"]).sum()
            else:
                metrics["line_volume_DC"] = 0.0
        else:
            metrics["line_volume_DC"] = 0.0
    else:
        metrics["line_volume_DC"] = 0.0

    # AC lines: prefer s_nom_opt, fallback to s_nom, else 0
    if hasattr(n, "lines") and not n.lines.empty and "length" in n.lines.columns:
        if "s_nom_opt" in n.lines.columns:
            metrics["line_volume_AC"] = (n.lines["length"] * n.lines["s_nom_opt"]).sum()
        elif "s_nom" in n.lines.columns:
            metrics["line_volume_AC"] = (n.lines["length"] * n.lines["s_nom"]).sum()
        else:
            metrics["line_volume_AC"] = 0.0
    else:
        metrics["line_volume_AC"] = 0.0

    metrics["line_volume"] = float(metrics["line_volume_AC"]) + float(metrics["line_volume_DC"])

    # -------------------------
    # Total costs (robust)
    # -------------------------
    # statistics might be missing or fail if not solved in a standard way
    total_costs = np.nan
    try:
        total_costs = float(n.statistics.capex().sum() + n.statistics.opex().sum())
    except Exception:
        # Fallback: try objective if present; otherwise NaN
        total_costs = float(getattr(n, "objective", np.nan)) if getattr(n, "objective", None) is not None else np.nan
    metrics["total costs"] = total_costs

    # -------------------------
    # Electricity price metrics (robust)
    # -------------------------
    # Handle:
    #  - missing buses_t / marginal_price
    #  - AC bus names not matching marginal_price columns
    #  - networks without carrier == 'AC' (sector-coupled, etc.)
    price_mean = np.nan
    price_std = np.nan
    price_zero_share = np.nan

    try:
        mp = getattr(n, "buses_t", None)
        mp = None if mp is None else getattr(n.buses_t, "marginal_price", None)

        if mp is not None and isinstance(mp, pd.DataFrame) and not mp.empty:
            # Determine candidate electricity buses (prefer carrier == 'AC')
            if hasattr(n, "buses") and not n.buses.empty and "carrier" in n.buses.columns:
                ac_buses = n.buses.index[n.buses["carrier"].eq("AC")]
            else:
                ac_buses = pd.Index([], dtype=object)

            # If no explicit AC buses, fall back to all marginal_price columns
            if len(ac_buses) == 0:
                available = mp.columns
            else:
                # Only keep those actually present in marginal_price columns
                available = ac_buses.intersection(mp.columns)

                # If intersection is empty (common in ARO/canonical conversions),
                # fall back to using all columns rather than erroring.
                if len(available) == 0:
                    available = mp.columns

            prices = mp.loc[:, available]

            # threshold higher than marginal_cost of VRE
            zero_hours = prices.where(prices < 0.1).count().sum()
            price_zero_share = float(zero_hours / prices.size) if prices.size else np.nan

            # match original behaviour: prices.unstack().mean()/std()
            # (works for DatetimeIndex and MultiIndex snapshots)
            price_mean = prices.unstack().mean()
            price_std = prices.unstack().std()

    except Exception:
        # Keep NaNs if anything unexpected happens
        pass

    metrics["electricity_price_zero_hours"] = price_zero_share
    metrics["electricity_price_mean"] = price_mean
    metrics["electricity_price_std"] = price_std

    # -------------------------
    # Shadow prices / constraints (robust)
    # -------------------------
    if hasattr(n, "global_constraints") and n.global_constraints is not None and not n.global_constraints.empty:
        gc = n.global_constraints

        if "lv_limit" in gc.index:
            if "constant" in gc.columns:
                metrics["line_volume_limit"] = gc.at["lv_limit", "constant"]
            if "mu" in gc.columns:
                metrics["line_volume_shadow"] = gc.at["lv_limit", "mu"]

        if "CO2Limit" in gc.index and "mu" in gc.columns:
            metrics["co2_shadow"] = gc.at["CO2Limit", "mu"]

        if "co2_sequestration_limit" in gc.index and "mu" in gc.columns:
            metrics["co2_storage_shadow"] = gc.at["co2_sequestration_limit", "mu"]

    return pd.Series(metrics).sort_index()


def calculate_prices(n: pypsa.Network) -> pd.Series:
    """
    Calculate time-averaged prices per carrier.
    """
    return n.buses_t.marginal_price.mean().groupby(n.buses.carrier).mean().sort_index()


def calculate_weighted_prices(n: pypsa.Network) -> pd.Series:
    """
    Calculate load-weighted prices per bus carrier.
    """
    carriers = n.buses.carrier.unique()

    weighted_prices = {}

    for carrier in carriers:
        load = n.statistics.withdrawal(
            groupby="bus",
            aggregate_time=False,
            bus_carrier=carrier,
            aggregate_across_components=True,
        ).T

        if not load.empty and load.sum().sum() > 0:
            price = n.buses_t.marginal_price.loc[:, n.buses.carrier == carrier]
            price = price.reindex(columns=load.columns, fill_value=1)

            weights = n.snapshot_weightings.generators
            a = weights @ (load * price).sum(axis=1)
            b = weights @ load.sum(axis=1)
            weighted_prices[carrier] = a / b

    return pd.Series(weighted_prices).sort_index()


def calculate_market_values(n: pypsa.Network) -> pd.Series:
    """
    Calculate market values for electricity.
    """
    return (
        n.statistics.market_value(bus_carrier="AC", aggregate_across_components=True)
        .sort_values()
        .dropna()
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "make_summary",
            clusters="5",
            opts="",
            sector_opts="",
            planning_horizons="2030",
            configfiles="config/test/config.overnight.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)
    assign_carriers(n)
    assign_locations(n)

    pypsa.set_option("params.statistics.nice_names", False)
    pypsa.set_option("params.statistics.drop_zero", False)

    for output in OUTPUTS:
        globals()["calculate_" + output](n).to_csv(snakemake.output[output])
