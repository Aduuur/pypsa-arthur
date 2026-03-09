# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Prepare and extend default cost data with custom cost modifications. Custom costs
can target all planning horizons and / or technologies using the 'all' identifier.

Preparing the cost data includes:
- aligning all units to conventional units (i.e. MW / MWh),
- filling in missing data,
- computing 'capital_cost' parameter (annualised investment costs and FOM),
- computing 'marginal_cost' parameter (fuel costs and VOM),
- computing storage costs for batteries and hydrogen,
- (deprecated) overwriting attributes using config-based modifications.

IMPORTANT BEHAVIOUR
-------------------
Custom costs only overwrite values that are explicitly present in the custom file.
Missing values in the custom file do NOT overwrite base values.

Also, economically critical fields like `investment` and `fuel` are NOT blindly
filled with zero via config fill_values, because that creates artificial zero
capital_cost / marginal_cost entries and can distort optimisation heavily.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from scripts.add_electricity import calculate_annuity

logger = logging.getLogger(__name__)


def overwrite_costs(costs: pd.DataFrame, custom_costs: pd.DataFrame) -> pd.DataFrame:
    """
    Apply custom cost modifications to costs data.

    Only non-missing values from `custom_costs` overwrite existing values in `costs`.
    Missing values in `custom_costs` leave the base values untouched.

    Special case:
    If technology == "all", only non-missing parameter values are propagated to
    all technologies.
    """
    if custom_costs is None or custom_costs.empty:
        return costs

    costs = costs.copy()

    all_techs = custom_costs.query("technology == 'all'").dropna(axis=1, how="all")
    custom_costs = custom_costs.query("technology != 'all'").dropna(axis=1, how="all")

    # Add technologies that don't already exist
    missing_idx = custom_costs.index.difference(costs.index)
    if len(missing_idx) > 0:
        logger.info("Adding %d missing technologies from custom costs: %s",
                    len(missing_idx), list(missing_idx)[:10])
        costs = pd.concat([costs, custom_costs.loc[missing_idx]], axis=0)

    # Overwrite only where custom values are actually present
    for param in custom_costs.columns:
        custom_col = custom_costs[param].dropna()
        if custom_col.empty:
            continue
        costs.loc[custom_col.index, param] = custom_col

    # Propagate "all" values only where they are non-missing
    if not all_techs.empty and "all" in all_techs.index:
        for param in all_techs.columns:
            value = all_techs.loc["all", param]
            if pd.notna(value):
                logger.info("Applying custom 'all' overwrite for parameter %r = %r", param, value)
                costs.loc[:, param] = value

    return costs


def _log_problematic_costs(costs: pd.DataFrame) -> None:
    """
    Log technologies with suspicious or missing cost inputs.
    """
    for col in ["investment", "fuel", "efficiency", "lifetime", "discount rate"]:
        if col not in costs.columns:
            logger.warning("Processed costs missing expected column %r", col)

    if "investment" in costs.columns:
        bad_inv = costs[costs["investment"].isna() | (costs["investment"] < 0)]
        if not bad_inv.empty:
            logger.warning(
                "Technologies with missing/negative investment costs: %s",
                bad_inv.index.tolist()[:50],
            )

        zero_inv = costs[costs["investment"] == 0]
        if not zero_inv.empty:
            logger.warning(
                "Technologies with zero investment cost: %s",
                zero_inv.index.tolist()[:50],
            )

    if "fuel" in costs.columns:
        missing_fuel = costs[costs["fuel"].isna()]
        if not missing_fuel.empty:
            logger.warning(
                "Technologies with missing fuel cost: %s",
                missing_fuel.index.tolist()[:50],
            )

    if "capital_cost" in costs.columns:
        bad_cap = costs[costs["capital_cost"].isna() | (costs["capital_cost"] < 0)]
        if not bad_cap.empty:
            logger.warning(
                "Technologies with missing/negative capital_cost: %s",
                bad_cap.index.tolist()[:50],
            )

        zero_cap = costs[costs["capital_cost"] == 0]
        if not zero_cap.empty:
            logger.warning(
                "Technologies with zero capital_cost: %s",
                zero_cap.index.tolist()[:50],
            )

    if "marginal_cost" in costs.columns:
        bad_mc = costs[costs["marginal_cost"].isna()]
        if not bad_mc.empty:
            logger.warning(
                "Technologies with missing marginal_cost: %s",
                bad_mc.index.tolist()[:50],
            )


def prepare_costs(
    costs: pd.DataFrame,
    config: dict,
    max_hours: dict = None,
    nyears: float = 1.0,
    custom_costs_fn: str = None,
) -> pd.DataFrame:
    """
    Standardize and prepare extended costs data.
    """
    custom_raw = pd.DataFrame()
    custom_prepared = pd.DataFrame()

    # Load custom costs and categorize into two sets:
    # - Raw attributes: overwritten before cost preparation
    # - Prepared attributes: overwritten after cost preparation
    if custom_costs_fn is not None and Path(custom_costs_fn).exists():
        custom_costs = pd.read_csv(
            custom_costs_fn,
            dtype={"planning_horizon": "str"},
            index_col=["technology", "parameter"],
        ).query("planning_horizon in [@planning_horizon, 'all']")

        custom_costs = custom_costs.drop("planning_horizon", axis=1).value.unstack(
            level=1
        )

        prepared_attrs = ["marginal_cost", "capital_cost"]
        raw_attrs = list(set(custom_costs.columns) - set(prepared_attrs))

        custom_raw = custom_costs[raw_attrs].dropna(axis=0, how="all")
        custom_prepared = custom_costs.filter(prepared_attrs).dropna(axis=0, how="all")

        logger.info(
            "Loaded custom costs from %s: raw=%d prepared=%d",
            custom_costs_fn,
            len(custom_raw),
            len(custom_prepared),
        )
    elif custom_costs_fn is not None:
        logger.warning("Custom costs file not found: %s", custom_costs_fn)

    # Copy marginal_cost and capital_cost for backward compatibility
    for key in ("marginal_cost", "capital_cost"):
        if key in config:
            config["overwrites"][key] = config[key]

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("/GW"), "value"] /= 1e3

    costs.unit = costs.unit.str.replace("/kW", "/MW")
    costs.unit = costs.unit.str.replace("/GW", "/MW")

    # min_count=1 is important to generate NaNs
    costs = costs.value.unstack(level=1).groupby("technology").sum(min_count=1)

    # Apply raw custom overwrites BEFORE filling defaults
    costs = overwrite_costs(costs, custom_raw)

    # ------------------------------------------------------------------
    # Fill only "safe" defaults.
    # Do NOT blindly fill investment/fuel/discount rate with zero.
    # ------------------------------------------------------------------
    fill_values = dict(config.get("fill_values", {}))
    unsafe_keys = {"investment", "fuel", "capital_cost", "marginal_cost"}
    safe_fill_values = {k: v for k, v in fill_values.items() if k not in unsafe_keys}

    if safe_fill_values:
        costs = costs.fillna(safe_fill_values)

    # Process deprecated config overwrites
    for attr in (
        "investment",
        "lifetime",
        "FOM",
        "VOM",
        "efficiency",
        "fuel",
        "standing losses",
        "discount rate",
    ):
        overwrites = config["overwrites"].get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs.loc[overwrites.index, attr] = overwrites
            warnings.warn(
                "Config-based cost overwrites is deprecated. Use external file instead (by default 'data/custom_costs.csv').",
                DeprecationWarning,
            )
            logger.info("Overwriting %s with:\n%s", attr, overwrites)

    # ------------------------------------------------------------------
    # Validate critical columns BEFORE deriving costs
    # ------------------------------------------------------------------
    required_cols = ["investment", "lifetime", "discount rate", "FOM", "VOM", "efficiency", "fuel"]
    missing_required = [c for c in required_cols if c not in costs.columns]
    if missing_required:
        raise ValueError(f"Missing required cost columns after preparation: {missing_required}")

    # Do not silently accept missing values in critical columns
    critical_missing = {}
    for c in required_cols:
        miss = costs.index[costs[c].isna()].tolist()
        if miss:
            critical_missing[c] = miss[:20]

    if critical_missing:
        logger.warning("Critical missing cost inputs detected: %s", critical_missing)

    # Avoid division by zero in marginal cost
    eff_bad = costs["efficiency"].isna() | (costs["efficiency"] <= 0)
    if eff_bad.any():
        logger.warning(
            "Technologies with invalid efficiency (<=0 or NaN): %s",
            costs.index[eff_bad].tolist()[:50],
        )

    annuity_factor = calculate_annuity(costs["lifetime"], costs["discount rate"])
    annuity_factor_fom = annuity_factor + costs["FOM"] / 100.0
    costs["capital_cost"] = annuity_factor_fom * costs["investment"] * nyears

    # Gas-based thermal plants inherit gas fuel assumptions
    if "gas" in costs.index:
        if "OCGT" in costs.index:
            costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
        if "CCGT" in costs.index:
            costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]

    costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

    if "gas" in costs.index:
        if "OCGT" in costs.index and "CO2 intensity" in costs.columns:
            costs.at["OCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]
        if "CCGT" in costs.index and "CO2 intensity" in costs.columns:
            costs.at["CCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]

    if "solar-utility" in costs.index and "solar" in costs.index:
        costs.at["solar", "capital_cost"] = costs.at["solar-utility", "capital_cost"]

    costs = costs.rename({"solar-utility single-axis tracking": "solar-hsat"})
    costs = costs.rename(columns={"standing losses": "standing_losses"})

    # Calculate storage costs if max_hours is provided
    if max_hours is not None:

        def costs_for_storage(store, link1, link2=None, max_hours=1.0):
            capital_cost = link1["capital_cost"] + max_hours * store["capital_cost"]
            if link2 is not None:
                capital_cost += link2["capital_cost"]
            return pd.Series(
                {
                    "capital_cost": capital_cost,
                    "marginal_cost": 0.0,
                    "CO2 intensity": 0.0,
                    "standing_losses": 0.0,
                }
            )

        if all(x in costs.index for x in ["battery storage", "battery inverter"]):
            costs.loc["battery"] = costs_for_storage(
                costs.loc["battery storage"],
                costs.loc["battery inverter"],
                max_hours=max_hours["battery"],
            )

        if all(x in costs.index for x in ["hydrogen storage underground", "fuel cell", "electrolysis"]):
            costs.loc["H2"] = costs_for_storage(
                costs.loc["hydrogen storage underground"],
                costs.loc["fuel cell"],
                costs.loc["electrolysis"],
                max_hours=max_hours["H2"],
            )

    # Overwrite prepared attributes LAST
    costs = overwrite_costs(costs, custom_prepared)

    for attr in ("marginal_cost", "capital_cost"):
        overwrites = config["overwrites"].get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            idx = overwrites.index.intersection(costs.index)
            costs.loc[idx, attr] = overwrites.loc[idx]
            warnings.warn(
                "Config-based cost overwrites is deprecated. Use external file instead (by default 'data/custom_costs.csv').",
                DeprecationWarning,
            )
            logger.info("Overwriting %s with:\n%s", attr, overwrites)

    _log_problematic_costs(costs)

    return costs


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("process_cost_data", planning_horizons=2030)

    cost_params = snakemake.params["costs"]

    n = pypsa.Network(snakemake.input.network)
    nyears = n.snapshot_weightings.generators.sum() / 8760.0
    planning_horizon = str(snakemake.wildcards.planning_horizons)

    # Retrieve costs assumptions
    costs = pd.read_csv(snakemake.input.costs, index_col=["technology", "parameter"])

    # Prepare costs
    costs_processed = prepare_costs(
        costs,
        cost_params,
        snakemake.params.max_hours,
        nyears,
        snakemake.input.custom_costs,
    )

    costs_processed.to_csv(snakemake.output[0])