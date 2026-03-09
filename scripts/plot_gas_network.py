# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Creates map of optimised gas network, storage and selected other
infrastructure.

This version is defensive:
- missing components do not crash the plot
- empty time series are handled gracefully
- missing carriers/colors fall back to defaults
- a reduced map is still written whenever possible
"""

import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches

from scripts._helpers import configure_logging, retry, set_scenario_config
from scripts.make_summary import assign_locations
from scripts.plot_power_network import load_projection

logger = logging.getLogger(__name__)


def empty_series(dtype=float) -> pd.Series:
    return pd.Series(dtype=dtype)


def get_series_by_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    cols = pd.Index(cols).intersection(df.columns)
    if len(cols) == 0:
        return df.iloc[:, 0:0]
    return df.loc[:, cols]


def safe_bus_aggregate(series: pd.Series, buses: pd.Series, rename_suffix: str = "") -> pd.Series:
    """
    Aggregate a time-integrated Series by bus and optionally strip a suffix
    from the resulting bus names.
    """
    if series.empty:
        return empty_series()

    out = series.groupby(buses).sum()

    if rename_suffix:
        out = out.rename(index=lambda x: x.replace(rename_suffix, "") if isinstance(x, str) else x)

    return out


def make_fake_multiindex(series: pd.Series, label: str) -> pd.Series:
    """
    Convert a bus-indexed series into a MultiIndex(bus, label) series for PyPSA plotting.
    """
    if series.empty:
        return empty_series()

    series = series.copy()
    series.index = pd.MultiIndex.from_product([series.index, [label]])
    return series


def get_weighted_sum(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Weighted sum over snapshots, robust to index misalignment and empty data.
    """
    if df.empty:
        return empty_series()

    aligned_weights = weights.reindex(df.index).fillna(0.0)
    if aligned_weights.sum() == 0:
        logger.warning("Snapshot weights sum to zero.")
        return empty_series()

    return df.mul(aligned_weights, axis=0).sum()


def get_fossil_gas(n: pypsa.Network, bus_size_factor: float) -> pd.Series:
    if n.generators.empty or "carrier" not in n.generators.columns:
        logger.info("No generators or no generator carrier column found.")
        return empty_series()

    fossil_gas_i = n.generators.index[n.generators.carrier == "gas"]
    if len(fossil_gas_i) == 0:
        logger.info("No fossil gas generators found.")
        return empty_series()

    if n.generators_t.p.empty:
        logger.info("No generator dispatch time series found for fossil gas.")
        return empty_series()

    dispatch = get_series_by_columns(n.generators_t.p, fossil_gas_i)
    if dispatch.empty:
        logger.info("No fossil gas generator dispatch columns found.")
        return empty_series()

    summed = get_weighted_sum(dispatch, n.snapshot_weightings.generators)
    if summed.empty:
        return empty_series()

    buses = n.generators.loc[summed.index, "bus"]
    fossil_gas = safe_bus_aggregate(summed, buses, rename_suffix=" gas") / bus_size_factor
    fossil_gas = fossil_gas.reindex(n.buses.index).fillna(0.0)

    return make_fake_multiindex(fossil_gas, "fossil gas")


def get_methanation(n: pypsa.Network, bus_size_factor: float) -> pd.Series:
    if n.links.empty or "carrier" not in n.links.columns:
        logger.info("No links or no link carrier column found.")
        return empty_series()

    methanation_i = n.links.index[n.links.carrier == "Sabatier"]
    if len(methanation_i) == 0:
        logger.info("No Sabatier links found.")
        return empty_series()

    if not hasattr(n, "links_t") or not hasattr(n.links_t, "p1") or n.links_t.p1.empty:
        logger.info("No link p1 time series found for methanation.")
        return empty_series()

    p1 = get_series_by_columns(n.links_t.p1, methanation_i)
    if p1.empty:
        logger.info("No Sabatier p1 columns found.")
        return empty_series()

    summed = get_weighted_sum(p1.abs(), n.snapshot_weightings.generators)
    if summed.empty:
        return empty_series()

    buses = n.links.loc[summed.index, "bus1"]
    methanation = safe_bus_aggregate(summed, buses, rename_suffix=" gas") / bus_size_factor
    methanation = methanation.groupby(methanation.index).sum()

    return make_fake_multiindex(methanation, "methanation")


def get_biogas(n: pypsa.Network, bus_size_factor: float) -> pd.Series:
    if n.stores.empty or "carrier" not in n.stores.columns:
        logger.info("No stores or no store carrier column found.")
        return empty_series()

    biogas_i = n.stores.index[n.stores.carrier == "biogas"]
    if len(biogas_i) == 0:
        logger.info("No biogas stores found.")
        return empty_series()

    if not hasattr(n, "stores_t") or not hasattr(n.stores_t, "p") or n.stores_t.p.empty:
        logger.info("No store dispatch time series found for biogas.")
        return empty_series()

    p = get_series_by_columns(n.stores_t.p, biogas_i)
    if p.empty:
        logger.info("No biogas store columns found in stores_t.p.")
        return empty_series()

    summed = get_weighted_sum(p, n.snapshot_weightings.generators)
    if summed.empty:
        return empty_series()

    buses = n.stores.loc[summed.index, "bus"]
    biogas = safe_bus_aggregate(summed, buses, rename_suffix=" biogas") / bus_size_factor
    biogas = biogas.groupby(biogas.index).sum()

    return make_fake_multiindex(biogas, "biogas")


def prepare_pipe_network(n: pypsa.Network, linewidth_factor: float, line_lower_threshold: float):
    """
    Return gas-pipeline-only network widths/colors, robustly.
    """
    if n.links.empty or "carrier" not in n.links.columns:
        logger.info("No links or no link carrier column present.")
        return empty_series(), empty_series(), empty_series(), pd.Series(dtype="object")

    link_carrier = n.links.carrier.fillna("").astype(str)
    gas_mask = link_carrier.str.contains("gas pipeline", na=False)

    gas_links = n.links.index[gas_mask]
    if len(gas_links) == 0:
        logger.info("No gas pipeline links found.")
        return empty_series(), empty_series(), empty_series(), pd.Series(dtype="object")

    # Restrict the network to gas links only
    n.links = n.links.loc[gas_links].copy()

    if hasattr(n, "links_t") and hasattr(n.links_t, "p0") and not n.links_t.p0.empty:
        p0 = get_series_by_columns(n.links_t.p0, n.links.index)
    else:
        p0 = pd.DataFrame()

    link_widths_rem = n.links.p_nom_opt.reindex(n.links.index).fillna(0.0) / linewidth_factor
    link_widths_rem[n.links.p_nom_opt.reindex(n.links.index).fillna(0.0) < line_lower_threshold] = 0.0

    link_widths_orig = n.links.p_nom.reindex(n.links.index).fillna(0.0) / linewidth_factor
    link_widths_orig[n.links.p_nom.reindex(n.links.index).fillna(0.0) < line_lower_threshold] = 0.0

    if p0.empty:
        max_usage = pd.Series(0.0, index=n.links.index)
    else:
        max_usage = p0.abs().max(axis=0).reindex(n.links.index).fillna(0.0)

    link_widths_used = max_usage / linewidth_factor
    link_widths_used[max_usage < line_lower_threshold] = 0.0

    pipe_colors = {
        "gas pipeline": "#f08080",
        "gas pipeline new": "#c46868",
        "gas pipeline (in 2020)": "lightgrey",
        "gas pipeline (available)": "#e8d1d1",
    }

    link_color_used = n.links.carrier.map(pipe_colors).fillna(pipe_colors["gas pipeline"])

    # Strip gas suffix from bus names if present
    if "bus0" in n.links.columns:
        n.links["bus0"] = n.links["bus0"].astype(str).str.replace(" gas", "", regex=False)
    if "bus1" in n.links.columns:
        n.links["bus1"] = n.links["bus1"].astype(str).str.replace(" gas", "", regex=False)

    return link_widths_orig, link_widths_rem, link_widths_used, link_color_used


def get_bus_colors(snakemake) -> dict:
    tech_colors = snakemake.params.plotting.get("tech_colors", {})

    return {
        "fossil gas": tech_colors.get("fossil gas", "lightgrey"),
        "methanation": tech_colors.get("methanation", "tab:orange"),
        "biogas": tech_colors.get("biogas", "seagreen"),
    }


def plot_base_map(ax, regions, proj, map_opts):
    """
    Plot a fallback base map if network plotting is partially unavailable.
    """
    try:
        target_crs = getattr(proj, "proj4_init", None)
        regions_plot = regions.to_crs(target_crs) if target_crs is not None else regions
        regions_plot.boundary.plot(ax=ax, color="darkgrey", linewidth=0.4)
    except Exception:
        logger.exception("Failed to plot fallback regional boundaries.")

    if map_opts.get("boundaries") is not None:
        try:
            xmin, xmax, ymin, ymax = map_opts["boundaries"]
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        except Exception:
            logger.exception("Failed to set map boundaries.")


@retry
def plot_ch4_map(n):
    try:
        assign_locations(n)
    except Exception:
        logger.exception("assign_locations failed. Continuing without reassigned locations.")

    # Work on a copy so in-place modifications do not leak unexpectedly
    n = n.copy()

    bus_size_factor = 8e7
    linewidth_factor = 1e4
    line_lower_threshold = 1e3  # MW below which not drawn

    # Drop non-electric buses so they don't clutter the plot
    if not n.buses.empty and "carrier" in n.buses.columns:
        ac_buses = n.buses.index[n.buses.carrier == "AC"]
        if len(ac_buses) == 0:
            logger.warning("No AC buses found. Plot will fall back to base map and any compatible overlays.")
        else:
            n.buses = n.buses.loc[ac_buses].copy()
    else:
        logger.warning("No buses or no bus carrier column found.")

    fossil_gas = get_fossil_gas(n, bus_size_factor)
    methanation = get_methanation(n, bus_size_factor)
    biogas = get_biogas(n, bus_size_factor)

    bus_size_parts = [s for s in [fossil_gas, methanation, biogas] if not s.empty]
    if bus_size_parts:
        bus_sizes = pd.concat(bus_size_parts)
    else:
        bus_sizes = empty_series()

    if not bus_sizes.empty:
        non_buses = bus_sizes.index.unique(level=0).difference(n.buses.index)
        if len(non_buses) > 0:
            logger.info("Dropping non-buses %s for CH4 network plot.", non_buses.tolist())
            keep_mask = ~bus_sizes.index.get_level_values(0).isin(non_buses)
            bus_sizes = bus_sizes.loc[keep_mask]
        bus_sizes = bus_sizes.sort_index()
    else:
        logger.info("No gas-source bus sizes available for plotting.")

    link_widths_orig, link_widths_rem, link_widths_used, link_color_used = prepare_pipe_network(
        n, linewidth_factor, line_lower_threshold
    )

    pipe_colors = {
        "gas pipeline": "#f08080",
        "gas pipeline new": "#c46868",
        "gas pipeline (in 2020)": "lightgrey",
        "gas pipeline (available)": "#e8d1d1",
    }
    bus_colors = get_bus_colors(snakemake)

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": proj})

    # Always try to show at least a base map
    plot_base_map(ax, regions, proj, map_opts)

    # Original pipeline network
    try:
        if not n.links.empty:
            n.plot(
                bus_sizes=bus_sizes if not bus_sizes.empty else 0.0,
                bus_colors=bus_colors if not bus_sizes.empty else None,
                link_colors=pipe_colors["gas pipeline (in 2020)"],
                link_widths=link_widths_orig if not link_widths_orig.empty else 0.0,
                branch_components=["Link"],
                ax=ax,
                **map_opts,
            )
        else:
            logger.info("Skipping original gas network layer because no gas links are available.")
    except Exception:
        logger.exception("Failed to plot original gas pipeline layer.")

    # Available/removable capacity layer
    try:
        if not n.links.empty and not link_widths_rem.empty:
            n.plot(
                ax=ax,
                bus_sizes=0.0,
                link_colors=pipe_colors["gas pipeline (available)"],
                link_widths=link_widths_rem,
                branch_components=["Link"],
                geomap_colors=False,
                boundaries=map_opts["boundaries"],
            )
    except Exception:
        logger.exception("Failed to plot available gas pipeline layer.")

    # Used capacity layer
    try:
        if not n.links.empty and not link_widths_used.empty:
            n.plot(
                ax=ax,
                bus_sizes=0.0,
                link_colors=link_color_used if not link_color_used.empty else pipe_colors["gas pipeline"],
                link_widths=link_widths_used,
                branch_components=["Link"],
                geomap_colors=False,
                boundaries=map_opts["boundaries"],
            )
    except Exception:
        logger.exception("Failed to plot used gas pipeline layer.")

    ax.set_title("CH4 network")

    # Circle legend
    try:
        sizes = [100, 10]
        labels = [f"{s} TWh" for s in sizes]
        sizes = [s / bus_size_factor * 1e6 for s in sizes]

        legend_kw = dict(
            loc="upper left",
            bbox_to_anchor=(0, 1.03),
            labelspacing=0.8,
            frameon=False,
            handletextpad=1,
            title="gas sources",
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
        logger.exception("Failed to add gas source circle legend.")

    # Line legend
    try:
        sizes = [50, 10]
        labels = [f"{s} GW" for s in sizes]
        scale = 1e3 / linewidth_factor
        sizes = [s * scale for s in sizes]

        legend_kw = dict(
            loc="upper left",
            bbox_to_anchor=(0.25, 1.03),
            frameon=False,
            labelspacing=0.8,
            handletextpad=1,
            title="gas pipeline",
        )

        add_legend_lines(
            ax,
            sizes,
            labels,
            patch_kw=dict(color="lightgrey"),
            legend_kw=legend_kw,
        )
    except Exception:
        logger.exception("Failed to add gas pipeline line legend.")

    # Patch legend
    try:
        colors = list(pipe_colors.values()) + list(bus_colors.values())
        labels = list(pipe_colors.keys()) + list(bus_colors.keys())

        legend_kw = dict(
            loc="upper left",
            bbox_to_anchor=(0, 1.24),
            ncol=2,
            frameon=False,
        )

        add_legend_patches(
            ax,
            colors,
            labels,
            legend_kw=legend_kw,
        )
    except Exception:
        logger.exception("Failed to add patch legend.")

    try:
        fig.savefig(snakemake.output.map, bbox_inches="tight")
        logger.info("Saved gas network map to %s", snakemake.output.map)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_gas_network",
            opts="",
            clusters="37",
            sector_opts="4380H-T-H-B-I-A-dist1",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)

    regions = gpd.read_file(snakemake.input.regions).set_index("name")

    map_opts = snakemake.params.plotting["map"].copy()

    if map_opts["boundaries"] is None:
        map_opts["boundaries"] = regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1]

    proj = load_projection(snakemake.params.plotting)

    plot_ch4_map(n)