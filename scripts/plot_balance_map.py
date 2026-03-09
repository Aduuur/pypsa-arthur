# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Create energy balance maps for the defined carriers.

This version is written defensively so that missing carriers, empty statistics,
missing marginal prices, or incomplete component sets do not crash the
post-processing workflow. Instead, the script logs warnings and produces a
reduced/fallback plot whenever possible.
"""

import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers

from scripts._helpers import (
    PYPSA_V1,
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.add_electricity import sanitize_carriers
from scripts.plot_power_network import load_projection

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1
logger = logging.getLogger(__name__)


def empty_series(dtype=float) -> pd.Series:
    return pd.Series(dtype=dtype)


def ensure_carrier_colors(n: pypsa.Network, tech_colors: dict | None = None) -> None:
    """
    Make sure all carriers have a usable color.

    Existing explicit colors are preserved unless overwritten by tech_colors.
    Missing/empty colors fall back to light grey.
    """
    if "color" not in n.carriers.columns:
        n.carriers["color"] = pd.Series(index=n.carriers.index, dtype="object")

    if tech_colors:
        for tech, color in tech_colors.items():
            if tech in n.carriers.index:
                n.carriers.loc[tech, "color"] = color

    mask = n.carriers["color"].isna() | n.carriers["color"].astype(str).eq("")
    n.carriers.loc[mask, "color"] = "lightgrey"


def safe_unique_level_values(obj: pd.Series | pd.DataFrame, level: str) -> pd.Index:
    """
    Return unique values from a MultiIndex level, or an empty index if absent.
    """
    idx = obj.index
    if isinstance(idx, pd.MultiIndex) and level in idx.names:
        return pd.Index(idx.get_level_values(level).unique())
    return pd.Index([])


def safe_columns(df: pd.DataFrame, cols) -> pd.Index:
    """
    Return the intersection of requested columns and actual columns.
    """
    return pd.Index(cols).intersection(df.columns)


def safe_series_get(flow: pd.Series | pd.DataFrame, key: str) -> pd.Series:
    """
    Return flow[key] if available, else an empty series.

    Works for:
    - Series with a MultiIndex containing a 'component' level
    - DataFrames with a column named key
    """
    if isinstance(flow, pd.DataFrame):
        if key in flow.columns:
            out = flow[key]
            return out if isinstance(out, pd.Series) else empty_series()
        return empty_series()

    if isinstance(flow, pd.Series):
        idx = flow.index
        if isinstance(idx, pd.MultiIndex) and "component" in idx.names:
            components = pd.Index(idx.get_level_values("component").unique())
            if key in components:
                mask = idx.get_level_values("component") == key
                out = flow[mask]
                # drop the component level if possible for downstream plotting
                if isinstance(out.index, pd.MultiIndex) and "component" in out.index.names:
                    out = out.droplevel("component")
                return out
        return empty_series()

    return empty_series()


def get_bus_sizes(
    n: pypsa.Network,
    carrier: str,
    conversion: float,
) -> pd.Series:
    """
    Compute bus sizes from the energy balance in a defensive way.
    """
    try:
        eb = n.statistics.energy_balance(bus_carrier=carrier, groupby=["bus", "carrier"])
    except Exception:
        logger.exception("Failed to compute energy balance for carrier '%s'.", carrier)
        return empty_series()

    if eb.empty:
        logger.warning("No energy balance found for carrier '%s'.", carrier)
        return empty_series()

    try:
        transmission_carriers = get_transmission_carriers(n, bus_carrier=carrier).rename(
            {"name": "carrier"}
        )
    except Exception:
        logger.exception(
            "Failed to obtain transmission carriers for carrier '%s'. Proceeding without transmission filtering.",
            carrier,
        )
        transmission_carriers = pd.Series(dtype="object")

    if (
        not transmission_carriers.empty
        and isinstance(eb.index, pd.MultiIndex)
        and "component" in eb.index.names
        and "carrier" in eb.index.names
    ):
        try:
            existing_components = pd.Index(
                transmission_carriers.unique("component")
            ).intersection(safe_unique_level_values(eb, "component"))

            existing_carriers = pd.Index(
                transmission_carriers.unique("carrier")
            ).intersection(safe_unique_level_values(eb, "carrier"))

            if len(existing_components) > 0 and len(existing_carriers) > 0:
                mask = (
                    eb.index.get_level_values("component").isin(existing_components)
                    & eb.index.get_level_values("carrier").isin(existing_carriers)
                )
                eb = eb.loc[~mask]
        except Exception:
            logger.exception(
                "Failed while filtering transmission-related balances for carrier '%s'. Continuing with unfiltered energy balance.",
                carrier,
            )

    eb = eb.dropna()
    if eb.empty:
        logger.warning(
            "Energy balance for carrier '%s' is empty after filtering.", carrier
        )
        return empty_series()

    if not (isinstance(eb.index, pd.MultiIndex) and "bus" in eb.index.names and "carrier" in eb.index.names):
        logger.warning(
            "Energy balance for carrier '%s' does not have expected MultiIndex levels ('bus', 'carrier').",
            carrier,
        )
        return empty_series()

    try:
        bus_sizes = eb.groupby(level=["bus", "carrier"]).sum().div(conversion)
        bus_sizes = bus_sizes.sort_values(ascending=False)
        return bus_sizes
    except Exception:
        logger.exception("Failed to aggregate bus sizes for carrier '%s'.", carrier)
        return empty_series()


def get_flow(
    n: pypsa.Network,
    carrier: str,
    conversion: float,
) -> pd.Series:
    """
    Compute transmission flow in a defensive way.
    """
    try:
        flow = n.statistics.transmission(groupby=False, bus_carrier=carrier)
    except Exception:
        logger.exception("Failed to compute transmission statistics for carrier '%s'.", carrier)
        return empty_series()

    if flow.empty:
        logger.info("No transmission flow found for carrier '%s'.", carrier)
        return empty_series()

    try:
        flow = flow.div(conversion)
    except Exception:
        logger.exception("Failed to convert transmission flow units for carrier '%s'.", carrier)
        return empty_series()

    try:
        if isinstance(flow.index, pd.MultiIndex) and len(flow.index.names) > 1:
            level_1 = flow.index.get_level_values(1).astype(str)
            flow_reversed_mask = level_1.str.contains("reversed", na=False)

            if flow_reversed_mask.any():
                flow_reversed = flow[flow_reversed_mask].rename(
                    lambda x: x.replace("-reversed", "")
                )
                flow = flow[~flow_reversed_mask].subtract(flow_reversed, fill_value=0)
    except Exception:
        logger.exception(
            "Failed to collapse reversed flows for carrier '%s'. Continuing with raw flow.",
            carrier,
        )

    return flow


def get_price_by_region(
    n: pypsa.Network,
    carrier: str,
) -> pd.Series:
    """
    Compute average marginal price per region for a bus carrier.

    Returns an empty series if prices cannot be computed.
    """
    buses = n.buses.index[n.buses.carrier == carrier]
    if buses.empty:
        logger.warning("No buses found for carrier '%s'.", carrier)
        return empty_series()

    if n.buses_t.marginal_price.empty:
        logger.warning("No marginal prices available in network for carrier '%s'.", carrier)
        return empty_series()

    existing_buses = safe_columns(n.buses_t.marginal_price, buses)
    if existing_buses.empty:
        logger.warning(
            "No marginal_price columns found for buses of carrier '%s'.", carrier
        )
        return empty_series()

    weights = n.snapshot_weightings.generators
    aligned_weights = weights.reindex(n.buses_t.marginal_price.index).fillna(0.0)

    total_weight = aligned_weights.sum()
    if total_weight == 0:
        logger.warning("Snapshot weights sum to zero for carrier '%s'.", carrier)
        return empty_series()

    try:
        prices = (
            aligned_weights @ n.buses_t.marginal_price[existing_buses]
        ) / total_weight
    except Exception:
        logger.exception(
            "Failed to compute weighted average marginal prices for carrier '%s'.", carrier
        )
        return empty_series()

    try:
        bus_locations = n.buses.loc[existing_buses, "location"].replace("", "EU").fillna("EU")
        price = prices.rename(bus_locations)
        level = "name" if PYPSA_V1 else "Bus"
        price = price.groupby(level=level).mean()
        return price
    except Exception:
        logger.exception(
            "Failed to aggregate prices by region for carrier '%s'.", carrier
        )
        return empty_series()


def prepare_regions_price(
    regions: gpd.GeoDataFrame,
    price: pd.Series,
    carrier: str,
    carrier_config: dict,
) -> tuple[gpd.GeoDataFrame, float, float]:
    """
    Attach regional prices to the regions GeoDataFrame and derive vmin/vmax.
    """
    regions = regions.copy()

    if price.empty:
        logger.warning(
            "No regional prices available for carrier '%s'. Using zero-filled regions.",
            carrier,
        )
        regions["price"] = 0.0
        vmin, vmax = -1.0, 1.0
    elif price.size == 1:
        val = float(price.iloc[0])
        regions["price"] = val
        shift = round(abs(val) / 20, 0)
        vmin, vmax = val - shift, val + shift
    else:
        regions["price"] = price.reindex(regions.index).fillna(0.0)
        vmin = float(regions["price"].min())
        vmax = float(regions["price"].max())
        if vmin == vmax:
            shift = round(abs(vmin) / 20, 0) if vmin != 0 else 1.0
            vmin, vmax = vmin - shift, vmax + shift

    if carrier_config.get("vmin") is not None:
        vmin = carrier_config["vmin"]
    if carrier_config.get("vmax") is not None:
        vmax = carrier_config["vmax"]

    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    return regions, vmin, vmax


def safe_add_patch_legend(ax, colors, labels, title, bbox_to_anchor, legend_kwargs):
    """
    Add patch legend only if there are labels to show.
    """
    labels = list(labels)
    if not labels:
        return

    legend_colors = pd.Series(colors).reindex(labels).fillna("lightgrey")
    add_legend_patches(
        ax,
        legend_colors,
        labels,
        legend_kw={
            "bbox_to_anchor": bbox_to_anchor,
            "ncol": 1,
            "title": title,
            **legend_kwargs,
        },
    )


def get_supply_consumption_carriers(
    bus_sizes: pd.Series,
) -> tuple[list[str], list[str]]:
    """
    Split carriers into supply and consumption groups using total absolute contribution.
    """
    if bus_sizes.empty:
        return [], []

    try:
        pos_carriers = bus_sizes[bus_sizes > 0].index.unique("carrier")
        neg_carriers = bus_sizes[bus_sizes < 0].index.unique("carrier")
    except Exception:
        logger.exception("Failed to derive positive/negative carriers for legend.")
        return [], []

    common_carriers = pos_carriers.intersection(neg_carriers)

    def get_total_abs(carrier_name: str, sign: int) -> float:
        try:
            values = bus_sizes.loc[:, carrier_name]
            values = values[values * sign > 0]
            return float(values.abs().sum())
        except Exception:
            return 0.0

    supp_carriers = sorted(
        set(pos_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) >= get_total_abs(c, -1)}
    )
    cons_carriers = sorted(
        set(neg_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) < get_total_abs(c, -1)}
    )

    return supp_carriers, cons_carriers


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_balance_map",
            clusters="10",
            opts="",
            sector_opts="",
            planning_horizons="2050",
            carrier="H2",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    n = pypsa.Network(snakemake.input.network)
    sanitize_carriers(n, snakemake.config)

    pypsa.set_option("params.statistics.round", 3)
    pypsa.set_option("params.statistics.drop_zero", True)
    pypsa.set_option("params.statistics.nice_names", False)

    regions = gpd.read_file(snakemake.input.regions).set_index("name")
    plotting_cfg = snakemake.params.plotting
    carrier = snakemake.wildcards.carrier

    # Ensure carrier colors are usable.
    ensure_carrier_colors(n, plotting_cfg.get("tech_colors", {}))

    # Set EU location if possible.
    eu_location = plotting_cfg["eu_node_location"]
    if "EU" in n.buses.index:
        n.buses.loc["EU", ["x", "y"]] = eu_location["x"], eu_location["y"]
    else:
        logger.warning("Bus 'EU' not present in network; skipping explicit EU coordinate assignment.")

    boundaries = plotting_cfg["map"]["boundaries"]

    if carrier not in plotting_cfg["balance_map"]:
        raise KeyError(
            f"Carrier '{carrier}' not found in plotting.balance_map configuration."
        )

    carrier_cfg = plotting_cfg["balance_map"][carrier]
    conversion = carrier_cfg["unit_conversion"]

    # Prepare bus metadata for plotting.
    if "location" not in n.buses.columns:
        n.buses["location"] = "EU"
    else:
        n.buses["location"] = n.buses["location"].replace("", "EU").fillna("EU")

    # Map coordinates via location if possible; keep originals otherwise.
    try:
        x_by_location = n.buses.groupby("location")["x"].first()
        y_by_location = n.buses.groupby("location")["y"].first()
        mapped_x = n.buses["location"].map(x_by_location)
        mapped_y = n.buses["location"].map(y_by_location)
        n.buses["x"] = mapped_x.fillna(n.buses["x"])
        n.buses["y"] = mapped_y.fillna(n.buses["y"])
    except Exception:
        logger.exception("Failed to remap bus coordinates by location. Keeping original coordinates.")

    # Bus sizes from energy balance.
    bus_sizes = get_bus_sizes(n, carrier, conversion)

    # Colors for balance-map carriers.
    if bus_sizes.empty:
        colors = pd.Series(dtype="object")
    else:
        carrier_colors = n.carriers["color"].replace("", "lightgrey").fillna("lightgrey")
        try:
            colors = (
                bus_sizes.index.get_level_values("carrier")
                .unique()
                .to_series()
                .map(carrier_colors)
                .fillna("lightgrey")
            )
        except Exception:
            logger.exception("Failed to derive bus colors for carrier '%s'. Falling back to grey.", carrier)
            colors = pd.Series("lightgrey", index=bus_sizes.index.get_level_values("carrier").unique())

    # Branch flows.
    flow = get_flow(n, carrier, conversion)
    line_widths = safe_series_get(flow, "Line").abs()
    link_widths = safe_series_get(flow, "Link").abs()
    line_flow = safe_series_get(flow, "Line")
    link_flow = safe_series_get(flow, "Link")
    transformer_flow = safe_series_get(flow, "Transformer")

    # Plot scaling.
    bus_size_factor = carrier_cfg["bus_factor"]
    branch_width_factor = carrier_cfg["branch_factor"]
    flow_size_factor = carrier_cfg["flow_factor"]

    # Prices per region.
    price = get_price_by_region(n, carrier)

    if carrier == "co2 stored" and "CO2Limit" in n.global_constraints.index and not price.empty:
        try:
            co2_price = n.global_constraints.loc["CO2Limit", "mu"]
            price = price - co2_price
        except Exception:
            logger.exception("Failed to adjust 'co2 stored' prices by CO2Limit shadow price.")

    regions, vmin, vmax = prepare_regions_price(regions, price, carrier, carrier_cfg)

    crs = load_projection(plotting_cfg)

    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )

    # Try to plot the network layer. If it fails, continue with the regional price map.
    try:
        n.plot(
            bus_sizes=bus_sizes * bus_size_factor if not bus_sizes.empty else None,
            bus_colors=colors if not colors.empty else None,
            bus_split_circles=not bus_sizes.empty,
            line_widths=line_widths * branch_width_factor if not line_widths.empty else None,
            link_widths=link_widths * branch_width_factor if not link_widths.empty else None,
            line_flow=line_flow * flow_size_factor if not line_flow.empty else None,
            link_flow=link_flow * flow_size_factor if not link_flow.empty else None,
            transformer_flow=transformer_flow * flow_size_factor if not transformer_flow.empty else None,
            ax=ax,
            margin=0.2,
            geomap_colors={"border": "darkgrey", "coastline": "darkgrey"},
            geomap=True,
            boundaries=boundaries,
        )
    except Exception:
        logger.exception(
            "Network plotting failed for carrier '%s'. Continuing with regional price layer only.",
            carrier,
        )

    # Plot regions price layer.
    try:
        # proj4_init is not always guaranteed across CRS implementations
        target_crs = getattr(crs, "proj4_init", None)
        regions_plot = regions.to_crs(target_crs) if target_crs is not None else regions
        regions_plot.plot(
            ax=ax,
            column="price",
            cmap=carrier_cfg["cmap"],
            vmin=vmin,
            vmax=vmax,
            edgecolor="None",
            linewidth=0,
        )
    except Exception:
        logger.exception(
            "Failed to plot regional prices for carrier '%s'.", carrier
        )

    ax.set_title(carrier)

    # Colorbar
    try:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=carrier_cfg["cmap"], norm=norm)
        price_unit = carrier_cfg["region_unit"]
        cbr = fig.colorbar(
            sm,
            ax=ax,
            label=f"Average Marginal Price [{price_unit}]",
            shrink=0.95,
            pad=0.03,
            aspect=50,
            orientation="horizontal",
        )
        cbr.outline.set_edgecolor("None")
    except Exception:
        logger.exception("Failed to add colorbar for carrier '%s'.", carrier)

    # Legend config
    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "title_fontproperties": {"weight": "bold"},
    }
    pad = 0.18

    # Keep compatibility with prior behavior.
    if "" not in n.carriers.index:
        n.carriers.loc["", "color"] = "None"
    else:
        n.carriers.loc["", "color"] = "None"

    # Supply/consumption legends
    supp_carriers, cons_carriers = get_supply_consumption_carriers(bus_sizes)

    safe_add_patch_legend(
        ax=ax,
        colors=n.carriers["color"],
        labels=supp_carriers,
        title="Supply",
        bbox_to_anchor=(0, -pad),
        legend_kwargs=legend_kwargs,
    )
    safe_add_patch_legend(
        ax=ax,
        colors=n.carriers["color"],
        labels=cons_carriers,
        title="Consumption",
        bbox_to_anchor=(0.5, -pad),
        legend_kwargs=legend_kwargs,
    )

    # Bus-size legend
    legend_bus_sizes = carrier_cfg.get("bus_sizes")
    carrier_unit = carrier_cfg.get("unit", "")
    if legend_bus_sizes is not None:
        try:
            add_legend_semicircles(
                ax,
                [
                    s * bus_size_factor * SEMICIRCLE_CORRECTION_FACTOR
                    for s in legend_bus_sizes
                ],
                [f"{s} {carrier_unit}" for s in legend_bus_sizes],
                patch_kw={"color": "#666"},
                legend_kw={
                    "bbox_to_anchor": (0, 1),
                    **legend_kwargs,
                },
            )
        except Exception:
            logger.exception("Failed to add bus-size legend for carrier '%s'.", carrier)

    # Branch-size legend
    legend_branch_sizes = carrier_cfg.get("branch_sizes")
    if legend_branch_sizes is not None:
        try:
            add_legend_lines(
                ax,
                [s * branch_width_factor for s in legend_branch_sizes],
                [f"{s} {carrier_unit}" for s in legend_branch_sizes],
                patch_kw={"color": "#666"},
                legend_kw={"bbox_to_anchor": (0.25, 1), **legend_kwargs},
            )
        except Exception:
            logger.exception("Failed to add branch-size legend for carrier '%s'.", carrier)

    # Save figure even if parts of the plot failed.
    try:
        fig.savefig(
            snakemake.output[0],
            dpi=400,
            bbox_inches="tight",
        )
        logger.info("Saved balance map for carrier '%s' to %s", carrier, snakemake.output[0])
    finally:
        plt.close(fig)