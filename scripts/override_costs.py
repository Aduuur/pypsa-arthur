# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Override fuel costs based on a per-horizon CSV.

Expected CSV format (example: data/custom_cost_adaptions_per_horizon.csv):
Fuel,2020,2025,2030,2035,2040,2045,2050,source,currency_year
gas,24.9,23.8,22.6,21.5,20.3,19.2,18.1,TYNDP 2024,2022.0
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def override_fuel_costs(
    costs: pd.DataFrame,
    custom_costs_file: str | Path,
    year: int | str,
) -> pd.DataFrame:
    year_col = str(year)
    df = pd.read_csv(custom_costs_file)
    if "Fuel" not in df.columns or year_col not in df.columns:
        logger.warning(
            "override_costs: missing columns Fuel or %s in %s",
            year_col, custom_costs_file,
        )
        return costs

    fuel_map = (
        df[["Fuel", year_col]]
        .dropna(subset=["Fuel"])
        .set_index("Fuel")
    )

    tech_map = {
        "coal": "coal",
        "lignite": "lignite",
        "gas": "gas",
        "oil": "oil",
        "nuclear": "nuclear",
        "solid biomass": "solid biomass",
        "biogas": "biogas",
    }

    for fuel_name, tech_name in tech_map.items():
        if fuel_name not in fuel_map.index:
            continue
        try:
            value = float(fuel_map.loc[fuel_name, year_col])
        except Exception:
            continue

        # pivotiertes Format: Index=technology, Spalte="fuel"
        if tech_name not in costs.index:
            logger.info("override_costs: %r nicht in costs, skip", tech_name)
            continue
        if "fuel" not in costs.columns:
            logger.warning("override_costs: keine 'fuel' Spalte in costs")
            continue

        old = costs.at[tech_name, "fuel"]
        costs.at[tech_name, "fuel"] = value

        # marginal_cost neu berechnen falls vorhanden
        if "marginal_cost" in costs.columns and "efficiency" in costs.columns:
            eff = costs.at[tech_name, "efficiency"]
            vom = costs.at[tech_name, "VOM"] if "VOM" in costs.columns else 0.0
            if pd.notna(eff) and eff > 0:
                new_mc = value / eff + (vom if pd.notna(vom) else 0.0)
                costs.at[tech_name, "marginal_cost"] = new_mc

        logger.info(
            "override_costs: %r fuel %s → %.2f EUR/MWh (war: %.2f)",
            tech_name, year_col, value, old if pd.notna(old) else float("nan"),
        )

    return costs

# Alias for backwards compatibility
override_costs = override_fuel_costs