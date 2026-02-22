# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Creates plots from summary CSV files.

This version is robust against:
- missing tech_colors entries (e.g. 'load_shedding')
- 'carrier' missing as column (may be index level)
- summary CSVs that contain meta rows like ('cluster', 'opt', 'planning_horizon', 'cost')
- empty/degenerate inputs: always writes balances-energy.svg (placeholder if needed)
"""

import matplotlib
matplotlib.use("Agg", force=True)

import logging
from pathlib import Path
import html

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

from scripts._helpers import configure_logging, rename_techs, set_scenario_config
from scripts.prepare_sector_network import co2_emissions_year

logger = logging.getLogger(__name__)
plt.style.use("bmh")


# -----------------------------------------------------------------------------
# Preferred plotting order
# -----------------------------------------------------------------------------
preferred_order = pd.Index(
    [
        "transmission lines",
        "hydroelectricity",
        "hydro reservoir",
        "run of river",
        "pumped hydro storage",
        "solid biomass",
        "biogas",
        "onshore wind",
        "offshore wind",
        "offshore wind (AC)",
        "offshore wind (DC)",
        "solar PV",
        "solar thermal",
        "solar rooftop",
        "solar",
        "building retrofitting",
        "ground heat pump",
        "air heat pump",
        "heat pump",
        "resistive heater",
        "power-to-heat",
        "gas-to-power/heat",
        "CHP",
        "OCGT",
        "gas boiler",
        "gas",
        "natural gas",
        "methanation",
        "ammonia",
        "hydrogen storage",
        "power-to-gas",
        "power-to-liquid",
        "battery storage",
        "hot water storage",
        "CO2 sequestration",
    ]
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
META_INDEX_VALUES = {"cluster", "opt", "planning_horizon", "cost"}

def _write_placeholder_svg(path: str, title: str) -> None:
    """Write a minimal SVG so Snakemake outputs always exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    safe_title = html.escape(str(title))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="220">
  <rect width="100%" height="100%" fill="white"/>
  <text x="20" y="60" font-size="24" font-family="sans-serif" fill="black">{safe_title}</text>
  <text x="20" y="115" font-size="14" font-family="sans-serif" fill="black">
    Plot skipped because the input summary was empty or not plottable.
  </text>
</svg>"""
    p.write_text(svg, encoding="utf-8")


def _safe_total_firstcol(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    s = df.sum(numeric_only=True)
    if len(s) == 0:
        return 0.0
    return float(s.iloc[0])


def _drop_meta_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your current global summaries sometimes contain meta-rows in the index,
    e.g. first rows with index value 'cluster', 'opt', 'planning_horizon', 'cost'.
    We drop them defensively.
    """
    if df is None or df.empty:
        return df

    out = df

    # If index is simple
    if not isinstance(out.index, pd.MultiIndex):
        return out.drop(index=list(META_INDEX_VALUES), errors="ignore")

    # If MultiIndex: drop rows where the FIRST level is one of the meta labels
    lvl0 = out.index.get_level_values(0)
    mask = ~lvl0.isin(META_INDEX_VALUES)
    out = out.loc[mask]

    return out


def _read_csv_any(path: str, *, index_levels: int, n_header: int) -> pd.DataFrame:
    """
    Read a summary CSV that is typically produced by PyPSA-Eur global summary.
    We try two strategies:
      (A) MultiIndex columns with multi-row header (standard upstream)
      (B) Simple header with single column (common when only one scenario exists)
    """
    # Try "standard" PyPSA-Eur: multi-row header
    try:
        df = pd.read_csv(path, index_col=list(range(index_levels)), header=list(range(n_header)))
        df = _drop_meta_rows(df)
        return df
    except Exception:
        pass

    # Fallback: simpler read
    df = pd.read_csv(path, index_col=list(range(index_levels)))
    df = _drop_meta_rows(df)
    return df


def _carrier_groupby(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group and sum by carrier. Handles:
    - carrier as a column
    - carrier as an index level (named or unnamed)
    """
    if df is None or df.empty:
        return df

    # carrier as column
    if "carrier" in df.columns:
        return df.groupby("carrier").sum(numeric_only=True)

    # named index level
    if isinstance(df.index, pd.MultiIndex) and "carrier" in df.index.names:
        return df.groupby(level="carrier").sum(numeric_only=True)

    # single index named carrier
    if df.index.name == "carrier":
        return df.groupby(level=0).sum(numeric_only=True)

    # heuristic: many summary CSVs have (component, carrier) as the last two index levels
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels >= 2:
        # if we have names, try to find the level that looks like carrier
        for i, nm in enumerate(df.index.names):
            if nm and "carrier" in str(nm).lower():
                return df.groupby(level=i).sum(numeric_only=True)

        # else assume last level is carrier
        return df.groupby(level=df.index.nlevels - 1).sum(numeric_only=True)

    raise KeyError(
        "Could not group by carrier: no carrier column, and index has no suitable level."
    )


def _tech_colors():
    """Get tech color mapping (may be missing keys)."""
    return snakemake.params.plotting.get("tech_colors", {})


def _color_for(tech: str) -> str:
    """Robust color lookup."""
    return _tech_colors().get(tech, "#777777")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_costs():
    cost_df = _read_csv_any(snakemake.input.costs, index_levels=3, n_header=n_header)
    df = _carrier_groupby(cost_df)

    # convert to billions
    df = df / 1e9
    df = df.groupby(df.index.map(rename_techs)).sum()

    thr = float(snakemake.params.plotting.get("costs_threshold", 0))
    if not df.empty and thr > 0:
        to_drop = df.index[df.max(axis=1) < thr]
        logger.info("Dropping technology with costs below %s EUR billion per year", thr)
        df = df.drop(to_drop, errors="ignore")

    total_cost = _safe_total_firstcol(df)
    logger.info("Total system cost of %s EUR billion per year", round(total_cost))

    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        fig.savefig(snakemake.output.costs, bbox_inches="tight")
        plt.close(fig)
        return

    new_index = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))

    fig, ax = plt.subplots(figsize=(12, 8))
    df.loc[new_index].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=[_color_for(i) for i in new_index],
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()

    ax.set_ylim([0, snakemake.params.plotting.get("costs_max", 1000)])
    ax.set_ylabel("System Cost [EUR billion per year]")
    ax.set_xlabel("")
    ax.grid(axis="x")
    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)

    fig.savefig(snakemake.output.costs, bbox_inches="tight")
    plt.close(fig)


def plot_energy():
    energy_df = _read_csv_any(snakemake.input.energy, index_levels=2, n_header=n_header)
    df = _carrier_groupby(energy_df)

    # convert MWh to TWh
    df = df / 1e6
    df = df.groupby(df.index.map(rename_techs)).sum()

    thr = float(snakemake.params.plotting.get("energy_threshold", 50.0))
    if not df.empty and thr > 0:
        to_drop = df.index[df.abs().max(axis=1) < thr]
        logger.info("Dropping all technology with energy consumption or production below %s TWh/a", thr)
        df = df.drop(to_drop, errors="ignore")

    total_energy = _safe_total_firstcol(df)
    logger.info("Total energy of %s TWh/a", round(total_energy))

    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        fig.savefig(snakemake.output.energy, bbox_inches="tight")
        plt.close(fig)
        return

    new_index = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))

    fig, ax = plt.subplots(figsize=(12, 8))
    df.loc[new_index].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=[_color_for(i) for i in new_index],
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()

    ax.set_ylim([snakemake.params.plotting.get("energy_min", -20000), snakemake.params.plotting.get("energy_max", 20000)])
    ax.set_ylabel("Energy [TWh/a]")
    ax.set_xlabel("")
    ax.grid(axis="x")
    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)

    fig.savefig(snakemake.output.energy, bbox_inches="tight")
    plt.close(fig)


def plot_balances_energy_only():
    """
    Robust balance plotting that guarantees writing snakemake.output.balances
    (which is expected to be 'balances-energy.svg').
    """
    out_path = snakemake.output.balances

    try:
        balances_df = _read_csv_any(snakemake.input.balances, index_levels=3, n_header=n_header)

        # Goal: build df indexed by carrier and plot stacks per scenario column(s).
        # We try to aggregate by carrier.
        df = balances_df.copy()
        df = _drop_meta_rows(df)

        # If the CSV has a bus_carrier level, select energy bus_carrier if present
        if isinstance(df.index, pd.MultiIndex) and "bus_carrier" in df.index.names:
            try:
                df_energy = df.xs("energy", level="bus_carrier", drop_level=True)
                df = df_energy
            except Exception:
                # fallback: sum over bus_carrier into an energy-like aggregate
                df = df.groupby([lvl for lvl in df.index.names if lvl != "bus_carrier"]).sum(numeric_only=True)

        # Now group by carrier
        df = _carrier_groupby(df)

        # convert MWh to TWh
        df = df / 1e6
        df = df.groupby(df.index.map(rename_techs)).sum()

        thr = float(snakemake.params.plotting.get("energy_threshold", 50.0)) / 10.0
        if not df.empty and thr > 0:
            to_drop = df.index[df.abs().max(axis=1) < thr]
            df = df.drop(to_drop, errors="ignore")

        if df is None or df.empty:
            logger.warning("Energy balances are empty after filtering; writing placeholder: %s", out_path)
            _write_placeholder_svg(out_path, "No balances-energy plot (empty input)")
            return

        new_index = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))
        new_columns = df.columns

        fig, ax = plt.subplots(figsize=(12, 8))
        df.loc[new_index, new_columns].T.plot(
            kind="bar",
            ax=ax,
            stacked=True,
            color=[_color_for(i) for i in new_index],
        )

        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels.reverse()

        ax.set_ylabel("Energy [TWh/a]")
        ax.set_xlabel("")
        ax.grid(axis="x")
        ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        logger.warning("plot_balances_energy_only failed (%r); writing placeholder: %s", e, out_path)
        _write_placeholder_svg(out_path, "No balances-energy plot (error / invalid input)")


# -----------------------------------------------------------------------------
# Historic emissions / carbon budget plot (unchanged; kept for compatibility)
# -----------------------------------------------------------------------------
def historical_emissions(countries):
    df = pd.read_csv(snakemake.input.co2, encoding="latin-1", low_memory=False)
    df.loc[df["Year"] == "1985-1987", "Year"] = 1986
    df["Year"] = df["Year"].astype(int)
    df = df.set_index(["Year", "Sector_name", "Country_code", "Pollutant_name"]).sort_index()

    e = pd.Series()
    e["electricity"] = "1.A.1.a - Public Electricity and Heat Production"
    e["residential non-elec"] = "1.A.4.b - Residential"
    e["services non-elec"] = "1.A.4.a - Commercial/Institutional"
    e["rail non-elec"] = "1.A.3.c - Railways"
    e["road non-elec"] = "1.A.3.b - Road Transportation"
    e["domestic navigation"] = "1.A.3.d - Domestic Navigation"
    e["international navigation"] = "1.D.1.b - International Navigation"
    e["domestic aviation"] = "1.A.3.a - Domestic Aviation"
    e["international aviation"] = "1.D.1.a - International Aviation"
    e["total energy"] = "1 - Energy"
    e["industrial processes"] = "2 - Industrial Processes and Product Use"
    e["agriculture"] = "3 - Agriculture"
    e["LULUCF"] = "4 - Land Use, Land-Use Change and Forestry"
    e["waste management"] = "5 - Waste management"
    e["other"] = "6 - Other Sector"
    e["indirect"] = "ind_CO2 - Indirect CO2"
    e["other LULUCF"] = "4.H - Other LULUCF"

    pol = ["CO2"]
    if "GB" in countries:
        countries.remove("GB")
        countries.append("UK")

    year = df.index.levels[0][df.index.levels[0] >= 1990]

    missing = pd.Index(countries).difference(df.index.levels[2])
    if not missing.empty:
        logger.warning(
            "Missing countries in historic emissions: %s", list(missing)
        )
        countries = pd.Index(df.index.levels[2]).intersection(countries)

    idx = pd.IndexSlice
    co2_totals = (
        df.loc[idx[year, e.values, countries, pol], "emissions"]
        .unstack("Year")
        .rename(index=pd.Series(e.index, e.values))
    )

    co2_totals = (1 / 1e6) * co2_totals.groupby(level=0, axis=0).sum()  # Gton CO2

    co2_totals.loc["industrial non-elec"] = (
        co2_totals.loc["total energy"]
        - co2_totals.loc[
            [
                "electricity",
                "services non-elec",
                "residential non-elec",
                "road non-elec",
                "rail non-elec",
                "domestic aviation",
                "international aviation",
                "domestic navigation",
                "international navigation",
            ]
        ].sum()
    )

    emissions = co2_totals.loc["electricity"]
    options = snakemake.params.sector
    if options.get("transport", False):
        emissions += co2_totals.loc[[i + " non-elec" for i in ["rail", "road"]]].sum()
    if options.get("heating", False):
        emissions += co2_totals.loc[[i + " non-elec" for i in ["residential", "services"]]].sum()
    if options.get("industry", False):
        emissions += co2_totals.loc[
            [
                "industrial non-elec",
                "industrial processes",
                "domestic aviation",
                "international aviation",
                "domestic navigation",
                "international navigation",
            ]
        ].sum()
    return emissions


def plot_carbon_budget_distribution(input_eurostat, options):
    import seaborn as sns

    sns.set()
    sns.set_style("ticks")
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20

    emissions_scope = snakemake.params.emissions_scope

    countries = snakemake.params.countries
    e_1990 = co2_emissions_year(
        countries,
        input_eurostat,
        options,
        emissions_scope,
        snakemake.input.co2,
        year=1990,
    )
    emissions = historical_emissions(countries)

    emissions.loc[2019] = 3.414362
    emissions.loc[2020] = 3.092434
    emissions.loc[2021] = 3.290418
    emissions.loc[2022] = 3.213025

    # NOTE: This branch expects standard balances CSV in upstream perfect foresight.
    # We keep it as-is for compatibility.
    if snakemake.config["foresight"] == "myopic":
        path_cb = "results/" + snakemake.params.RDIR + "/csvs/"
        co2_cap = pd.read_csv(path_cb + "carbon_budget_distribution.csv", index_col=0)[["cb"]]
        co2_cap *= e_1990
    else:
        supply_energy = pd.read_csv(
            snakemake.input.balances, index_col=[0, 1, 2], header=[0, 1, 2, 3]
        )
        co2_cap = supply_energy.loc["co2"].droplevel(0).drop("co2").sum().unstack().T / 1e9
        co2_cap.rename(index=lambda x: int(x), inplace=True)

    plt.figure(figsize=(10, 7))
    gs1 = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs1[0, 0])
    ax1.set_ylabel("CO$_2$ emissions \n [Gt per year]", fontsize=22)
    ax1.set_xlim([1990, snakemake.params.planning_horizons[-1] + 1])

    ax1.plot(emissions, color="black", linewidth=3, label=None)

    ax1.plot([2020], [0.8 * emissions[1990]], marker="*", markersize=12,
             markerfacecolor="black", markeredgecolor="black")
    ax1.plot([2030], [0.45 * emissions[1990]], marker="*", markersize=12,
             markerfacecolor="black", markeredgecolor="black")
    ax1.plot([2030], [0.6 * emissions[1990]], marker="*", markersize=12,
             markerfacecolor="black", markeredgecolor="black")

    ax1.plot([2050, 2050], [x * emissions[1990] for x in [0.2, 0.05]],
             color="gray", linewidth=2, marker="_", alpha=0.5)

    ax1.plot([2050], [0.0 * emissions[1990]], marker="*", markersize=12,
             markerfacecolor="black", markeredgecolor="black", label="EU committed target")

    for col in co2_cap.columns:
        ax1.plot(co2_cap[col], linewidth=3, label=col)

    ax1.legend(fancybox=True, fontsize=18, loc=(0.01, 0.01),
               facecolor="white", frameon=True)

    plt.grid(axis="y")
    path = snakemake.output.balances.split("balances")[0] + "carbon_budget.svg"
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake("plot_summary")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n_header = 3

    plot_costs()
    plot_energy()

    # IMPORTANT: always produce balances-energy.svg
    plot_balances_energy_only()
    if not Path(snakemake.output.balances).exists():
        _write_placeholder_svg(snakemake.output.balances, "No balances-energy plot (missing output safeguard)")

    co2_budget = snakemake.params.get("co2_budget")
    if (isinstance(co2_budget, str) and co2_budget.startswith("cb")) or snakemake.params.get("foresight") == "perfect":
        options = snakemake.params.sector
        plot_carbon_budget_distribution(snakemake.input.eurostat, options)