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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def override_fuel_costs(
    costs: pd.DataFrame,
    custom_costs_file: str | Path,
    year: int | str,
) -> pd.DataFrame:
    """
    Override only fuel costs for technologies listed in `tech_map`.

    IMPORTANT:
    - Only finite, non-missing values from the custom CSV are used.
    - Missing values in the CSV do NOT overwrite existing values in `costs`.
    - This function does NOT touch capital_cost / investment-like fields.
    """
    year_col = str(year)
    custom_costs_file = Path(custom_costs_file)

    if not custom_costs_file.exists():
        logger.warning("override_costs: file not found: %s", custom_costs_file)
        return costs

    # Work on a copy to avoid accidental side effects
    costs = costs.copy()

    try:
        df = pd.read_csv(custom_costs_file)
    except Exception as exc:
        logger.warning("override_costs: could not read %s (%s)", custom_costs_file, exc)
        return costs

    required = {"Fuel", year_col}
    missing = required.difference(df.columns)
    if missing:
        logger.warning(
            "override_costs: missing required columns %s in %s",
            sorted(missing),
            custom_costs_file,
        )
        return costs

    # Keep only relevant columns
    fuel_df = df[["Fuel", year_col]].copy()

    # Clean fuel names
    fuel_df["Fuel"] = fuel_df["Fuel"].astype(str).str.strip()

    # Convert target year column safely to numeric
    fuel_df[year_col] = pd.to_numeric(fuel_df[year_col], errors="coerce")

    # Drop rows with missing Fuel names
    fuel_df = fuel_df.dropna(subset=["Fuel"])

    # Handle duplicates explicitly
    dup_mask = fuel_df["Fuel"].duplicated(keep=False)
    if dup_mask.any():
        dups = sorted(fuel_df.loc[dup_mask, "Fuel"].unique().tolist())
        logger.warning(
            "override_costs: duplicate Fuel entries found in %s for %s. "
            "Keeping first occurrence. Duplicates: %s",
            custom_costs_file,
            year_col,
            dups,
        )
        fuel_df = fuel_df.drop_duplicates(subset=["Fuel"], keep="first")

    fuel_map = fuel_df.set_index("Fuel")[year_col]

    tech_map = {
        "coal": "coal",
        "lignite": "lignite",
        "gas": "gas",
        "oil": "oil",
        "nuclear": "nuclear",
        "solid biomass": "solid biomass",
        "biogas": "biogas",
    }

    if "fuel" not in costs.columns:
        logger.warning("override_costs: costs has no 'fuel' column")
        return costs

    n_overridden = 0

    for fuel_name, tech_name in tech_map.items():
        if fuel_name not in fuel_map.index:
            logger.info(
                "override_costs: fuel %r not present in custom file for year %s -> keep existing value",
                fuel_name,
                year_col,
            )
            continue

        value = fuel_map.loc[fuel_name]

        # Only override when the custom value is finite
        if pd.isna(value) or not np.isfinite(value):
            logger.info(
                "override_costs: fuel %r has no finite value for year %s -> keep existing value",
                fuel_name,
                year_col,
            )
            continue

        value = float(value)

        if tech_name not in costs.index:
            logger.info("override_costs: technology %r not in costs -> skip", tech_name)
            continue

        old_fuel = costs.at[tech_name, "fuel"]
        costs.at[tech_name, "fuel"] = value
        n_overridden += 1

        # Recompute marginal_cost only if all required inputs are available
        if "marginal_cost" in costs.columns and "efficiency" in costs.columns:
            eff = pd.to_numeric(pd.Series([costs.at[tech_name, "efficiency"]]), errors="coerce").iloc[0]

            vom = 0.0
            if "VOM" in costs.columns:
                vom_val = pd.to_numeric(pd.Series([costs.at[tech_name, "VOM"]]), errors="coerce").iloc[0]
                if pd.notna(vom_val) and np.isfinite(vom_val):
                    vom = float(vom_val)

            if pd.notna(eff) and np.isfinite(eff) and eff > 0:
                old_mc = costs.at[tech_name, "marginal_cost"]
                new_mc = value / float(eff) + vom
                costs.at[tech_name, "marginal_cost"] = new_mc

                logger.info(
                    "override_costs: %r fuel[%s] %.4f -> %.4f ; marginal_cost %.4f -> %.4f",
                    tech_name,
                    year_col,
                    float(old_fuel) if pd.notna(old_fuel) else float("nan"),
                    value,
                    float(old_mc) if pd.notna(old_mc) else float("nan"),
                    new_mc,
                )
            else:
                logger.warning(
                    "override_costs: %r fuel overridden, but marginal_cost not recomputed "
                    "(invalid efficiency=%r)",
                    tech_name,
                    costs.at[tech_name, "efficiency"],
                )
        else:
            logger.info(
                "override_costs: %r fuel[%s] %.4f -> %.4f",
                tech_name,
                year_col,
                float(old_fuel) if pd.notna(old_fuel) else float("nan"),
                value,
            )

    logger.info(
        "override_costs: finished for year %s, overridden fuel values for %d technologies",
        year_col,
        n_overridden,
    )

    return costs


# Alias for backwards compatibility
override_costs = override_fuel_costs