# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Creates plots for optimised power network topologies and regional generation,
storage and conversion capacities built.

This version is defensive:
- missing components or attributes do not crash plotting
- missing colors fall back to light grey
- missing EU buses do not crash coordinate assignment
- empty cost tables, missing AC buses, or absent DC/B2B links are handled gracefully
- a reduced/fallback map is still written whenever possible
"""

import logging

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches

from scripts._helpers import configure_logging, rename_techs, retry, set_scenario_config
from scripts.make_summary import assign_locations
from scripts.plot_summary import preferred_order

logger = logging.getLogger(__name__)


def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    elif tech in ["H2 Electrolysis", "methanation", "H2 liquefaction"]:
        return "power-to-gas"
    elif tech == "H2":
        return "H2 storage"
    elif tech in ["NH3", "Haber-Bosch", "ammonia cracker", "ammonia store"]:
        return "ammonia"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    elif tech in ["Fischer-Tropsch", "methanolisation"]:
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "CC" in tech or "sequestration" in tech:
        return "CCS"
    else:
        return tech


def load_projection(plotting_params):
    proj_kwargs = plotting_params.get("projection", dict(name="EqualEarth")).copy()
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)


def empty_series(dtype=float) -> pd.Series:
    return pd.Series(dtype=dtype)


def empty_df(index=None, columns=None) -> pd.DataFrame:
    return pd.DataFrame(
        index=pd.Index([]) if index is None else index,
        columns=pd.Index([]) if columns is None else columns,
    )


def get_tech_colors(plotting_params: dict) -> dict:
    tech_colors = plotting_params.get("tech_colors", {})
    if not isinstance(tech_colors, dict):
        logger.warning("plotting.tech_colors is not a dict. Falling back to empty mapping.")
        return {}
    return tech_colors


def safe_attr(obj, name, default=None):
    return getattr(obj, name, default)


def ensure_boundaries(map_opts: dict, regions: gpd.GeoDataFrame) -> dict:
    map_opts = map_opts.copy()
    if map_opts.get("boundaries") is None:
        map_opts["boundaries"] = regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1]
    return map_opts


def safe_plot_regions_boundary(ax, regions: gpd.GeoDataFrame, proj, boundaries=None):
    try:
        target_crs = getattr(proj, "proj4_init", None)
        regions_plot = regions.to_crs(target_crs) if target_crs is not None else regions
        regions_plot.boundary.plot(ax=ax, color="darkgrey", linewidth=0.4)
    except Exception:
        logger.exception("Failed to plot fallback regional boundaries.")

    if boundaries is not None:
        try:
            xmin, xmax, ymin, ymax = boundaries
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        except Exception:
            logger.exception("Failed to set map boundaries.")


def drop_non_ac_buses(n: pypsa.Network) -> None:
    if n.buses.empty:
        logger.warning("Network has no buses.")
        return

    if "carrier" not in n.buses.columns:
        logger.warning("Bus table has no 'carrier' column; keeping all buses.")
        return

    ac_buses = n.buses.index[n.buses.carrier == "AC"]
    if len(ac_buses) == 0:
        logger.warning("No AC buses found; keeping all buses.")
        return

    n.buses = n.buses.loc[ac_buses].copy()


def ensure_eu_bus_coordinates(n: pypsa.Network, plotting_params: dict) -> None:
    eu_location = plotting_params.get("eu_node_location", dict(x=-5.5, y=46))

    for eu_bus in ["EU gas", "EU"]:
        if eu_bus in n.buses.index:
            try:
                n.buses.loc[eu_bus, "x"] = eu_location["x"]
                n.buses.loc[eu_bus, "y"] = eu_location["y"]
            except Exception:
                logger.exception("Failed to assign coordinates to '%s'.", eu_bus)


def build_component_costs(n: pypsa.Network, components) -> pd.DataFrame:
    """
    Build location x technology annualized capital cost table.
    """
    costs = pd.DataFrame(index=n.buses.index)

    for comp in components:
        if not hasattr(n, comp):
            logger.info("Network has no component table '%s'.", comp)
            continue

        df_c = getattr(n, comp)
        if df_c is None or df_c.empty:
            continue

        required_cols = {"carrier", "location", "capital_cost"}
        missing = required_cols.difference(df_c.columns)
        if missing:
            logger.warning(
                "Skipping component '%s' because required columns are missing: %s",
                comp,
                sorted(missing),
            )
            continue

        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"
        if attr not in df_c.columns:
            logger.warning(
                "Skipping component '%s' because attribute '%s' is missing.",
                comp,
                attr,
            )
            continue

        try:
            df_c = df_c.copy()
            df_c["nice_group"] = df_c["carrier"].map(rename_techs_tyndp)

            values = df_c["capital_cost"].fillna(0.0) * df_c[attr].fillna(0.0)

            costs_c = (
                values.groupby([df_c["location"], df_c["nice_group"]])
                .sum()
                .unstack()
                .fillna(0.0)
            )

            costs = pd.concat([costs, costs_c], axis=1)
            logger.debug("%s costs head:\n%s", comp, costs_c.head())
        except Exception:
            logger.exception("Failed while processing component '%s'.", comp)

    if costs.empty:
        return costs

    try:
        costs = costs.T.groupby(costs.columns).sum().T
    except Exception:
        logger.exception("Failed to consolidate cost columns.")
        return pd.DataFrame(index=n.buses.index)

    try:
        zero_cols = list(costs.columns[(costs == 0.0).all()])
        if zero_cols:
            costs = costs.drop(zero_cols, axis=1)
    except Exception:
        logger.exception("Failed to drop all-zero cost columns.")

    return costs


def order_cost_columns(costs: pd.DataFrame) -> pd.DataFrame:
    if costs.empty:
        return costs

    try:
        new_columns = preferred_order.intersection(costs.columns).append(
            costs.columns.difference(preferred_order)
        )
        return costs[new_columns]
    except Exception:
        logger.exception("Failed to reorder cost columns; keeping original order.")
        return costs


def stack_costs(costs: pd.DataFrame) -> pd.Series:
    if costs.empty:
        return empty_series()

    try:
        return costs.stack()
    except Exception:
        logger.exception("Failed to stack costs.")
        return empty_series()


def filter_costs_to_existing_buses(costs: pd.Series, buses: pd.Index) -> pd.Series:
    if costs.empty:
        return costs

    if not isinstance(costs.index, pd.MultiIndex) or len(costs.index.names) < 2:
        logger.warning("Cost index is not the expected MultiIndex; skipping bus filtering.")
        return costs

    try:
        level0 = costs.index.get_level_values(0)
        keep = level0.isin(buses)
        dropped = pd.Index(level0[~keep].unique())
        if len(dropped) > 0:
            logger.info("Dropping non-buses %s", dropped.tolist())
        costs = costs.loc[keep]
        costs.index = pd.MultiIndex.from_tuples(costs.index.values)
        return costs
    except Exception:
        logger.exception("Failed to filter costs to existing buses.")
        return costs


def get_significant_carriers(costs: pd.Series, threshold: float = 100e6) -> list[str]:
    if costs.empty:
        return []

    try:
        carriers = costs.groupby(level=1).sum()
        carriers = carriers.where(carriers > threshold).dropna()
        return list(carriers.index)
    except Exception:
        logger.exception("Failed to determine significant carriers.")
        return []


def prepare_transmission_links(n: pypsa.Network) -> None:
    """
    Keep only DC/B2B links for the power network plot.
    """
    if n.links.empty:
        return

    if "carrier" not in n.links.columns:
        logger.warning("Link table has no 'carrier' column; dropping all links for power plot.")
        n.links = n.links.iloc[0:0].copy()
        return

    keep_mask = n.links["carrier"].isin(["DC", "B2B"])
    n.links = n.links.loc[keep_mask].copy()


def compute_line_and_link_widths(
    n: pypsa.Network,
    transmission_limit,
    transmission: bool,
) -> tuple[pd.Series, pd.Series, float, str]:
    """
    Compute robust line/link widths and metadata for legend title/scaling.
    """
    line_lower_threshold = 500.0
    line_upper_threshold = 1e4
    linewidth_factor = 4e3
    title = "added grid"

    lines = n.lines.copy() if hasattr(n, "lines") and n.lines is not None else pd.DataFrame()
    links = n.links.copy() if hasattr(n, "links") and n.links is not None else pd.DataFrame()

    line_widths = empty_series()
    link_widths = empty_series()

    def get_series(df: pd.DataFrame, preferred: str, fallback: str | None = None) -> pd.Series:
        if df.empty:
            return empty_series()
        if preferred in df.columns:
            return df[preferred].fillna(0.0)
        if fallback and fallback in df.columns:
            return df[fallback].fillna(0.0)
        return pd.Series(0.0, index=df.index)

    try:
        if transmission_limit == "lv1.0":
            if transmission:
                line_widths = get_series(lines, "s_nom_opt", "s_nom")
                link_widths = get_series(links, "p_nom_opt", "p_nom")
                linewidth_factor = 2e3
                line_lower_threshold = 0.0
                title = "current grid"
            else:
                line_widths = get_series(lines, "s_nom_opt", "s_nom") - get_series(lines, "s_nom", None)
                link_widths = get_series(links, "p_nom_opt", "p_nom") - get_series(links, "p_nom", None)
        else:
            if transmission:
                line_widths = get_series(lines, "s_nom_opt", "s_nom")
                link_widths = get_series(links, "p_nom_opt", "p_nom")
                title = "total grid"
            else:
                line_widths = get_series(lines, "s_nom_opt", "s_nom") - get_series(lines, "s_nom_min", None)
                link_widths = get_series(links, "p_nom_opt", "p_nom") - get_series(links, "p_nom_min", None)

        if not line_widths.empty:
            line_widths = line_widths.clip(line_lower_threshold, line_upper_threshold)
            line_widths = line_widths.replace(line_lower_threshold, 0.0)

        if not link_widths.empty:
            link_widths = link_widths.clip(line_lower_threshold, line_upper_threshold)
            link_widths = link_widths.replace(line_lower_threshold, 0.0)

    except Exception:
        logger.exception("Failed to compute line/link widths.")
        line_widths = empty_series()
        link_widths = empty_series()

    return line_widths, link_widths, linewidth_factor, title


def safe_add_patch_legend(ax, colors, labels, legend_kw):
    if not labels:
        return
    try:
        add_legend_patches(ax, colors, labels, legend_kw=legend_kw)
    except Exception:
        logger.exception("Failed to add patch legend.")


@retry
def plot_map(
    n,
    components=("links", "stores", "storage_units", "generators"),
    bus_size_factor=2e10,
    transmission=False,
    with_legend=True,
):
    tech_colors = get_tech_colors(snakemake.params.plotting)

    try:
        assign_locations(n)
    except Exception:
        logger.exception("assign_locations failed. Continuing without reassigned locations.")

    # Work on a copy so in-place changes do not affect callers.
    n = n.copy()

    # Keep network usable even if AC filtering is impossible or empty.
    drop_non_ac_buses(n)

    costs = build_component_costs(n, components)
    costs = order_cost_columns(costs)

    for item in costs.columns if not costs.empty else []:
        if item not in tech_colors:
            logger.warning("%s not in config/plotting/tech_colors", item)

    costs = stack_costs(costs)

    ensure_eu_bus_coordinates(n, snakemake.params.plotting)
    prepare_transmission_links(n)

    costs = filter_costs_to_existing_buses(costs, n.buses.index)
    carriers = get_significant_carriers(costs, threshold=100e6)

    line_widths, link_widths, linewidth_factor, title = compute_line_and_link_widths(
        n,
        snakemake.params.get("transmission_limit"),
        transmission,
    )

    ac_color = "rosybrown"
    dc_color = "darkseagreen"

    fig, ax = plt.subplots(subplot_kw={"projection": proj})
    fig.set_size_inches(7, 6)

    # Always try to show at least regional boundaries.
    safe_plot_regions_boundary(ax, regions, proj, boundaries=map_opts.get("boundaries"))

    # Prepare bus colors with fallback.
    if costs.empty:
        bus_sizes_plot = 0.0
        bus_colors_plot = None
    else:
        bus_sizes_plot = costs / bus_size_factor
        bus_colors_plot = {k: tech_colors.get(k, "lightgrey") for k in carriers}
        # n.plot can still need colors for techs below threshold if present in bus_sizes
        try:
            all_techs = pd.Index(costs.index.get_level_values(1).unique())
            for tech in all_techs:
                bus_colors_plot.setdefault(tech, tech_colors.get(tech, "lightgrey"))
        except Exception:
            logger.exception("Failed to prepare bus color mapping.")

    try:
        n.plot(
            bus_sizes=bus_sizes_plot,
            bus_colors=bus_colors_plot,
            line_colors=ac_color,
            link_colors=dc_color,
            line_widths=line_widths / linewidth_factor if not line_widths.empty else 0.0,
            link_widths=link_widths / linewidth_factor if not link_widths.empty else 0.0,
            ax=ax,
            **map_opts,
        )
    except Exception:
        logger.exception("Failed to plot power network layer. Keeping fallback base map.")

    ax.set_title("Power network")

    # Circle legend
    try:
        sizes = [20, 10, 5]
        labels = [f"{s} bEUR/a" for s in sizes]
        sizes = [s / bus_size_factor * 1e9 for s in sizes]

        legend_kw = dict(
            loc="upper left",
            bbox_to_anchor=(0.01, 1.06),
            labelspacing=0.8,
            frameon=False,
            handletextpad=0,
            title="system cost",
        )

        add_legend_circles(
            ax,
            sizes,
            labels,
            srid=getattr(n, "srid", 4326),
            patch_kw=dict(facecolor="lightgrey"),
            legend_kw=legend_kw,
        )
    except Exception:
        logger.exception("Failed to add system cost circle legend.")

    # Line legend
    try:
        sizes = [10, 5]
        labels = [f"{s} GW" for s in sizes]
        scale = 1e3 / linewidth_factor if linewidth_factor != 0 else 0.0
        sizes = [s * scale for s in sizes]

        legend_kw = dict(
            loc="upper left",
            bbox_to_anchor=(0.27, 1.06),
            frameon=False,
            labelspacing=0.8,
            handletextpad=1,
            title=title,
        )

        add_legend_lines(
            ax,
            sizes,
            labels,
            patch_kw=dict(color="lightgrey"),
            legend_kw=legend_kw,
        )
    except Exception:
        logger.exception("Failed to add line legend.")

    # Patch legend
    if with_legend:
        labels = carriers + ["HVAC line", "HVDC link"]
        colors = [tech_colors.get(c, "lightgrey") for c in carriers] + [ac_color, dc_color]

        legend_kw = dict(
            bbox_to_anchor=(1.52, 1.04),
            frameon=False,
        )

        safe_add_patch_legend(ax, colors, labels, legend_kw)

    try:
        fig.savefig(snakemake.output.map, bbox_inches="tight")
        logger.info("Saved power network map to %s", snakemake.output.map)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_power_network",
            opts="",
            clusters="37",
            sector_opts="4380H-T-H-B-I-A-dist1",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)

    regions = gpd.read_file(snakemake.input.regions).set_index("name")

    map_opts = ensure_boundaries(snakemake.params.plotting["map"], regions)

    proj = load_projection(snakemake.params.plotting)

    plot_map(n)