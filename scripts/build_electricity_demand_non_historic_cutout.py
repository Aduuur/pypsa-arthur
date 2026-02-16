"""
Generate hourly profiles and calculate thermal electricity demand per country and sector.

Key fixes vs. your current version:
- Normalize snapshots end if it arrives as YYYY-12-31 00:00 (date-only bug) -> set to YYYY+1-01-01 00:00.
- Build a full hourly index using inclusive='left' (robust end-exclusive).
- Reindex ALL components (no-thermal, heating, cooling, and final total) to full_index before writing NetCDF/CSV.
  This prevents xarray dimension conflicts (8736 vs 8760) and makes missing hours explicit as NaN.
- Make calc_hourly_space_demand truly hourly by expanding daily -> hourly first and then applying intraday profile.
"""

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur
# SPDX-License-Identifier: MIT


import logging
from itertools import product
from typing import Iterable, List, Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    checkBool,
    generate_periodic_profiles,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def expected_hours_in_year(year: int, drop_leap_day: bool) -> int:
    is_leap = pd.Timestamp(year, 12, 31).is_leap_year
    if is_leap and not drop_leap_day:
        return 8784
    return 8760


def change_index_year(df: pd.DataFrame, new_year: int) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.map(lambda x: x.replace(year=new_year))
    return out


def _detect_time_col(df: pd.DataFrame) -> str:
    for c in ["Date", "date", "time", "Time", "timestamp", "Timestamp"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _normalize_snapshot_window(time_start: pd.Timestamp, time_end: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Defensive normalization: sometimes configs/scenarios pass end='YYYY-12-31' (interpreted as 00:00),
    which yields 364*24=8736 hours for non-leap years. For a yearly run we want end-exclusive at Jan 1 next year.
    """
    year = time_start.year

    # If end is in same year and equals Dec 31 00:00, interpret as date-only and normalize.
    if (
        time_end.year == year
        and time_end.month == 12
        and time_end.day == 31
        and time_end.hour == 0
        and time_end.minute == 0
        and time_end.second == 0
    ):
        fixed = pd.Timestamp(year + 1, 1, 1, 0, 0, 0)
        logger.warning(
            f"Snapshots end is {time_end} (looks date-only). Normalizing to {fixed} (end-exclusive)."
        )
        time_end = fixed

    return time_start, time_end


def _build_full_hourly_index(time_start: pd.Timestamp, time_end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Build end-exclusive hourly index [time_start, time_end).
    """
    return pd.date_range(start=time_start, end=time_end, freq="h", inclusive="left")


# -----------------------------------------------------------------------------
# Core functions
# -----------------------------------------------------------------------------
def calc_hourly_space_demand(
    daily_space_demand: pd.DataFrame,
    intraday_profiles_path: str,
    drop_leap_day: bool,
    uses: List[str],
    sectors: List[str],
) -> xr.Dataset:
    """
    Convert daily space heating/cooling demand shape to hourly using intraday profiles.

    Important: generate_periodic_profiles preserves the frequency of the passed index.
    Therefore we must pass an hourly index here (not daily).
    """
    intraday_profiles = pd.read_csv(intraday_profiles_path, index_col=0)
    heat_demand: Dict[str, pd.DataFrame] = {}

    # Expand daily -> hourly: each daily value applies to all 24 hours of that day (then shaped by intraday profile)
    day_start = daily_space_demand.index.min()
    day_end_exclusive = daily_space_demand.index.max() + pd.Timedelta(days=1)
    hourly_index = pd.date_range(start=day_start, end=day_end_exclusive, freq="h", inclusive="left")

    daily_hourly = daily_space_demand.reindex(hourly_index).ffill()
    daily_hourly.index.name = "snapshots"

    # Build hourly intraday factors per (sector,use)
    for sector, use in product(sectors, uses):
        weekday = list(intraday_profiles[f"{sector} {use} weekday"])
        weekend = list(intraday_profiles[f"{sector} {use} weekend"])
        weekly_profile = weekday * 5 + weekend * 2  # 7*24 = 168

        intraday_year_profile = generate_periodic_profiles(
            daily_hourly.index.tz_localize("UTC"),
            nodes=daily_hourly.columns,
            weekly_profile=weekly_profile,
        )

        # intraday_year_profile is a DataFrame indexed like daily_hourly (hourly now)
        if use == "space":
            heat_demand[f"{sector} {use}"] = daily_hourly * intraday_year_profile
        else:
            # for 'water' (in your prior logic), profile itself acts as shape
            heat_demand[f"{sector} {use}"] = intraday_year_profile

    heat_demand_df = pd.concat(heat_demand, axis=1, names=["sector use", "node"])
    heat_demand_df.index.name = "snapshots"

    ds = heat_demand_df.stack(future_stack=True).to_xarray()

    if drop_leap_day:
        ds = ds.sel(snapshots=~((ds.snapshots.dt.month == 2) & (ds.snapshots.dt.day == 29)))

    return ds


def load_daily_demand(thermal_demand_path: str) -> pd.DataFrame:
    """
    Read daily heating/cooling demand shape CSV.
    Expect index column is date-like.
    """
    df = pd.read_csv(thermal_demand_path, index_col=0, parse_dates=True)
    df.index.name = "time"

    leap_days = df.index[(df.index.month == 2) & (df.index.day == 29)]
    if len(leap_days) > 0:
        logger.warning(f"Leap days present in daily demand: {list(leap_days[:5])}{' ...' if len(leap_days) > 5 else ''}")
    else:
        logger.info("No leap days present in daily demand.")

    return df


def thermal_elec_demand(
    hourly_thermal_demand: xr.Dataset,
    energy_totals: pd.DataFrame,
    year: int,
    uses: List[str],
    sectors: List[str],
) -> pd.DataFrame:
    """
    Compute hourly electricity demand per country for heating/cooling.

    hourly_thermal_demand is an xr.Dataset created by calc_hourly_space_demand with dims (snapshots, sector use, node).
    """
    # Filter to target year (defensive; should already be the right year)
    thermal_demand_year = hourly_thermal_demand.sel(
        snapshots=hourly_thermal_demand["snapshots"].dt.year == year
    )

    thermal_demand_shape = thermal_demand_year.to_dataframe().unstack(level=1)

    year_totals = energy_totals.loc[energy_totals["year"] == year].drop(columns=["year"])
    year_totals.index.name = "name"

    electric_thermal_supply: Dict[str, pd.DataFrame] = {}

    for sector, use in product(sectors, uses):
        name = f"{sector} {use}"
        if name not in thermal_demand_shape.columns.get_level_values(0):
            raise KeyError(f"Missing thermal demand shape for '{name}' in dataset columns.")

        denom = thermal_demand_shape[name].sum().replace(0.0, np.nan)

        col_name = f"electricity {name}"
        if col_name not in year_totals.columns:
            raise KeyError(f"Missing column '{col_name}' in energy totals.")

        electric_thermal_supply[name] = (
            (thermal_demand_shape[name] / denom)
            .multiply(year_totals[col_name])
            * 1e6
        )

    electric_thermal_supply_df = pd.concat(electric_thermal_supply, axis=1)
    # aggregate to countries (level=1 in multiindex: node/country)
    electric_thermal_supply_agg = electric_thermal_supply_df.T.groupby(level=1).sum().T
    electric_thermal_supply_agg.index.name = "time"
    return electric_thermal_supply_agg


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake("build_electricity_demand_non_historic_cutout")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Inputs / params
    intraday_profiles = snakemake.input.heat_profile
    cooling_demand_path = snakemake.input.energy_totals_cool
    heating_demand_path = snakemake.input.energy_totals_heat
    electric_demand_no_thermal_path = snakemake.input.demand_no_thermal

    drop_leap_day = checkBool(snakemake.params.drop_leap_day)
    energy_totals_year = int(snakemake.params.energy_totals_year)
    et_regression = checkBool(snakemake.params.et_regression)

    time_start = pd.to_datetime(snakemake.params.snapshots["start"])
    time_end = pd.to_datetime(snakemake.params.snapshots["end"])
    snapshot_year = int(time_start.year)

    logger.info(f"SNAPSHOTS(raw)={snakemake.params.snapshots}")
    logger.info(f"Parsed time_start={time_start} time_end={time_end}")

    time_start, time_end = _normalize_snapshot_window(time_start, time_end)
    full_index = _build_full_hourly_index(time_start, time_end)

    # Determine which year totals to use
    year_demand = snapshot_year if et_regression else energy_totals_year
    logger.info(f"Using year_demand={year_demand} (et_regression={et_regression})")

    # Load daily thermal shapes (already daily 365 rows)
    daily_heat = load_daily_demand(snakemake.input.hdd)
    daily_cool = load_daily_demand(snakemake.input.cdd)

    # Convert daily -> hourly shaped profiles (xr.Dataset)
    ds_heat = calc_hourly_space_demand(
        daily_heat, intraday_profiles, drop_leap_day,
        uses=["water", "space"], sectors=["residential", "services"]
    )
    ds_cool = calc_hourly_space_demand(
        daily_cool, intraday_profiles, drop_leap_day,
        uses=["space"], sectors=["residential", "services"]
    )

    # Load non-thermal demand (hourly, but 2019 in your inputs)
    electric_demand_no_thermal = pd.read_csv(
        electric_demand_no_thermal_path, index_col="Date", parse_dates=["Date"]
    )
    electric_demand_no_thermal.index.name = "time"

    # Load energy totals
    energy_totals_cool = pd.read_csv(cooling_demand_path, index_col="country")
    energy_totals_heat = pd.read_csv(heating_demand_path, index_col="country")

    # Compute thermal hourly electricity demand (per country)
    elec_cooling = thermal_elec_demand(
        ds_cool, energy_totals_cool, year_demand,
        uses=["space"], sectors=["residential", "services"]
    )
    elec_heating = thermal_elec_demand(
        ds_heat, energy_totals_heat, year_demand,
        uses=["space", "water"], sectors=["residential", "services"]
    )

    # Harmonize columns (countries)
    common = sorted(set(elec_cooling.columns) & set(elec_heating.columns) & set(electric_demand_no_thermal.columns))
    if not common:
        raise ValueError("No common country columns among cooling/heating/no-thermal demand inputs.")

    # Reindex all components to the full snapshot index (this prevents 8736 vs 8760 conflicts)
    no_th = change_index_year(electric_demand_no_thermal[common], snapshot_year).reindex(full_index)
    cool = change_index_year(elec_cooling[common], snapshot_year).reindex(full_index)
    heat = change_index_year(elec_heating[common], snapshot_year).reindex(full_index)

    # Optionally drop leap day (should already be absent if snapshots/drop_leap_day dictates)
    if drop_leap_day:
        mask = ~((full_index.month == 2) & (full_index.day == 29))
        full_index = full_index[mask]
        no_th = no_th.reindex(full_index)
        cool = cool.reindex(full_index)
        heat = heat.reindex(full_index)

    # Combine
    elec_with_thermal = (no_th + cool + heat)

    # Sanity check
    exp = expected_hours_in_year(snapshot_year, drop_leap_day)
    if len(full_index) != exp:
        logger.warning(f"Full index has {len(full_index)} hours, expected {exp}. (Check snapshots config.)")

    # Write CSV
    elec_with_thermal.sort_index(axis=1).to_csv(snakemake.output.elec_demand)

    # Write NetCDF components (all aligned to full_index)
    aggregated_demand = xr.Dataset(
        {
            "elec_cooling_demand": (["time", "country"], cool.values),
            "elec_heating_demand": (["time", "country"], heat.values),
            "elec_demand_without_thermal": (["time", "country"], no_th.values),
        },
        coords={"time": full_index, "country": common},
    )
    aggregated_demand.to_netcdf(snakemake.output.elec_thermal_demand)

    # Final report
    n_nans = int(elec_with_thermal.isna().any(axis=1).sum())
    if n_nans > 0:
        logger.warning(f"Output contains {n_nans} hourly timesteps with NaNs (missing in some input component).")
    logger.info(
        f"Output demand index: start={full_index[0]} end={full_index[-1]} n={len(full_index)} expected={exp}"
    )
