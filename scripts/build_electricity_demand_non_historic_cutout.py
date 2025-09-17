"""
Compute daily heating and cooling demand using atlite, generate hourly profiles,
and calculate thermal electricity demand per country and sector.
"""

import os
import logging
from tempfile import NamedTemporaryFile
from itertools import product
from typing import Callable, Union

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import pytz
import atlite
from dask.distributed import Client, LocalCluster

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config, load_cutout, generate_periodic_profiles

# --- Helper functions --- #

# def get_snapshots(
#     snapshots: dict, drop_leap_day: bool = False, freq: str = "h", **kwargs
# ) -> pd.DatetimeIndex:
#     """
#     Return a DateTimeIndex of snapshots for multiple time ranges.
#     """
#     start = snapshots["start"] if isinstance(snapshots["start"], list) else [snapshots["start"]]
#     end = snapshots["end"] if isinstance(snapshots["end"], list) else [snapshots["end"]]

#     assert len(start) == len(end), "Start and end lists must have the same length"

#     time_periods = [pd.date_range(start=s, end=e, freq=freq, inclusive=snapshots["inclusive"], **kwargs)
#                     for s, e in zip(start, end)]

#     time = pd.DatetimeIndex([])
#     for period in time_periods:
#         time = time.append(period)

#     if drop_leap_day and time.is_leap_year.any():
#         time = time[~((time.month == 2) & (time.day == 29))]

#     return time


# def load_cutout(
#     cutout_files: Union[str, list[str]], time: Union[None, pd.DatetimeIndex] = None
# ) -> atlite.Cutout:
#     """
#     Load one or multiple atlite cutouts and optionally select specific times.
#     """
#     if isinstance(cutout_files, str):
#         cutout = atlite.Cutout(cutout_files)
#     elif isinstance(cutout_files, list):
#         cutout_da = [atlite.Cutout(c).data for c in cutout_files]
#         combined_data = xr.concat(cutout_da, dim="time", data_vars="minimal")
#         cutout = atlite.Cutout(NamedTemporaryFile().name, data=combined_data)

#     if time is not None:
#         cutout.data = cutout.data.sel(time=time)

#     return cutout


# def generate_periodic_profiles(dt_index, nodes, weekly_profile, localize=None):
#     """
#     Expand daily weekly profiles to hourly profiles for each node, considering timezones.
#     """
#     weekly_profile = pd.Series(weekly_profile, range(24 * 7))
#     week_df = pd.DataFrame(index=dt_index, columns=nodes)

#     for node in nodes:
#         ct = node[:2] if node[:2] != "XK" else "RS"
#         timezone = pytz.timezone(pytz.country_timezones[ct][0])
#         tz_dt_index = dt_index.tz_convert(timezone)
#         week_df[node] = [24 * dt.weekday() + dt.hour for dt in tz_dt_index]
#         week_df[node] = week_df[node].map(weekly_profile)

#     week_df = week_df.tz_localize(localize)
#     return week_df


def calc_hourly_space_demand(daily_space_demand, intraday_profiles, uses, sectors):
    """
    Calculate hourly space heating/cooling demand from daily values using intraday profiles.
    """
    intraday_profiles = pd.read_csv(intraday_profiles, index_col=0)
    heat_demand = {}

    for sector, use in product(sectors, uses):
        weekday = list(intraday_profiles[f"{sector} {use} weekday"])
        weekend = list(intraday_profiles[f"{sector} {use} weekend"])
        weekly_profile = weekday * 5 + weekend * 2

        intraday_year_profile = generate_periodic_profiles(
            daily_space_demand.index.tz_localize("UTC"),
            nodes=daily_space_demand.columns,
            weekly_profile=weekly_profile,
        )

        if use == "space":
            heat_demand[f"{sector} {use}"] = daily_space_demand * intraday_year_profile
        else:
            heat_demand[f"{sector} {use}"] = intraday_year_profile

    heat_demand = pd.concat(heat_demand, axis=1, names=["sector use", "node"])
    heat_demand.index.name = "snapshots"
    ds = heat_demand.stack(future_stack=True).to_xarray()
    return ds


def load_and_reindex_demand(hdd):
    """
    Reindex daily demand to hourly and forward fill missing values.
    """
    hdd.index.name = "time"
    leap_days = hdd.index[(hdd.index.month == 2) & (hdd.index.day == 29)]

    if len(leap_days) > 0:
        logging.warning(f"Leap days present: {leap_days}")
    else:
        logging.info("No leap days present.")

    hourly_index = pd.date_range(
        start=hdd.index.min(),
        end=hdd.index.max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
        freq="h"
    )

    hdd_hourly = hdd.reindex(hourly_index).ffill()
    hdd_hourly.index.name = "time"
    return hdd_hourly


def thermal_elec_demand(hourly_thermal_demand: xr.Dataset, energy_totals: pd.DataFrame, demand_day_ratio, year: int,
                        uses=["water", "space"], sectors=["residential", "services"]):
    """
    Compute hourly electricity demand per country per sector for heating/cooling.
    """
    thermal_demand_shape = hourly_thermal_demand.sel(snapshots=hourly_thermal_demand['snapshots'].dt.year == year)
    thermal_demand_shape = hourly_thermal_demand.to_dataframe().unstack(level=1)

    year_totals = energy_totals.loc[energy_totals['year'] == year].drop(columns=['year'])
    year_totals.index.name = 'name'

    electric_thermal_supply = {}
    for sector, use in product(sectors, uses):
        name = f"{sector} {use}"
        if 'space' in name:
            elec_sector_use = year_totals[f"electricity {sector} {use}"] * demand_day_ratio
        else:
            elec_sector_use = year_totals[f"electricity {sector} {use}"]

        electric_thermal_supply[name] = (thermal_demand_shape[name] / thermal_demand_shape[name].sum()).multiply(elec_sector_use) * 1e6

    electric_thermal_supply = pd.concat(electric_thermal_supply, axis=1)
    electric_thermal_supply_agg = electric_thermal_supply.T.groupby(level=1).sum().T

    return electric_thermal_supply_agg


# --- Main execution --- #
if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake("build_electricity_demand")

    configure_logging(snakemake)
    set_scenario_config(snakemake)


    # Input files
    cutout_name = snakemake.input.cutout
    regions_onshore = snakemake.input.country_shapes
    pop_layout = snakemake.input.pop_layout_total
    intraday_profiles = snakemake.input.heat_profile
    cooling_demand_path = snakemake.input.energy_totals_cool
    heating_demand_path = snakemake.input.energy_totals
    electric_demand_no_thermal_path = snakemake.input.demand_no_thermal
    hist_demand_day_calc_path = snakemake.input.hist_demand_day_calc
    drop_leap_day = snakemake.params.drop_leap_day

    time_start = pd.to_datetime(snakemake.params.snapshots["start"])
    time_end = pd.to_datetime(snakemake.params.snapshots["end"])

    overlapping_year = (
        (time_end - time_start) > (time_start + pd.DateOffset(years=1) - time_start)
    )

    if overlapping_year:
        raise ValueError("Provide snapshots for one year only; overlapping time series are not supported.")
    else:
        snapshot_year=time_start.year #equals time_end.year
        #use whole year for calculation
        snapshots={'start': f'{snapshot_year}-01-01', 'end': f'{snapshot_year+1}-01-01', 'inclusive': 'left'}

    # Generate time indices
    time = get_snapshots(snapshots, drop_leap_day)
    daily = get_snapshots(
            snapshots,
            snakemake.params.drop_leap_day,
            freq="D",
        )
    

    # Load cutout and regions
    cutout = load_cutout(cutout_name, time=time)
    clustered_regions = gpd.read_file(regions_onshore).set_index("name").buffer(0)
    I = cutout.indicatormatrix(clustered_regions)
    pop_layout = xr.open_dataarray(pop_layout)
    stacked_pop = pop_layout.stack(spatial=("y", "x"))
    M = I.T.dot(np.diag(I.dot(stacked_pop)))

    # Compute daily heat/cooling demand
    heat_demand = cutout.heat_demand(
        threshold=15.0,
        a=1.0,
        constant=0.0,
        hour_shift=0.0,
        matrix=M.T,
        index=clustered_regions.index,
        show_progress=False,
    ).sel(time=daily)

    df_heat = pd.DataFrame(heat_demand.values, index=heat_demand["time"].values, columns=heat_demand["name"].values)

    cool_demand = cutout.cooling_demand(
        threshold=18.0,
        a=1.0,
        constant=0.0,
        hour_shift=0.0,
        matrix=M.T,
        index=clustered_regions.index,
        show_progress=False,
    ).sel(time=daily)

    df_cool = pd.DataFrame(cool_demand.values, index=cool_demand["time"].values, columns=cool_demand["name"].values)

    # Hourly demand
    daily_space_heating_demand = load_and_reindex_demand(df_heat)
    ds_heat = calc_hourly_space_demand(daily_space_heating_demand, intraday_profiles,
                                       uses=["water", "space"], sectors=["residential", "services"])

    daily_space_cooling_demand = load_and_reindex_demand(df_cool)
    ds_cool = calc_hourly_space_demand(daily_space_cooling_demand, intraday_profiles,
                                       uses=["space"], sectors=["residential", "services"])

    # Load total energy data
    electric_demand_no_thermal = pd.read_csv(electric_demand_no_thermal_path, index_col="Date", parse_dates=["Date"])
    hist_demand_day_calc = pd.read_csv(hist_demand_day_calc_path, index_col=["country", 'year'])
    energy_totals_heat = pd.read_csv(heating_demand_path, index_col="country")
    energy_totals_cool = pd.read_csv(cooling_demand_path, index_col="country")

    #equals config_provider('energy','energy_totals_year')
    year_demand = electric_demand_no_thermal.index.year.unique()[0]

    if len(time) >= 8760:
        ratio_heat = df_heat.sum(axis=0) / hist_demand_day_calc['sum_hdd'].xs(year_demand, level='year')
        ratio_cool = df_cool.sum(axis=0) / hist_demand_day_calc['sum_cdd'].xs(year_demand, level='year')
        logger.info("Historical energy totals scaled by yearly sum of demand days.")
        logger.info(f"Heat ratio: {ratio_heat.to_dict()}")
        logger.info(f"Cooling ratio: {ratio_cool.to_dict()}")
    else:
        logger.info("Energy ratios not scaled. Provide a full-year snapshot for scaling.")
        ratio_heat = 1
        ratio_cool = 1

    # Compute thermal electricity demand
    elec_cooling = thermal_elec_demand(ds_cool, energy_totals_cool, ratio_cool, year_demand,
                                       uses=["space"], sectors=["residential", "services"])
    elec_heating = thermal_elec_demand(ds_heat, energy_totals_heat, ratio_heat, year_demand,
                                       uses=["space", "water"], sectors=["residential", "services"])

    # Combine with non-thermal demand
    common = list(set(elec_cooling.columns) & set(elec_heating.columns) & set(electric_demand_no_thermal.columns))

    def change_index_year(df, new_year):
        df = df.copy()
        df.index = df.index.map(lambda x: x.replace(year=new_year))
        return df

    elec_with_thermal = (change_index_year(electric_demand_no_thermal[common], snapshot_year)
                         + change_index_year(elec_cooling[common], snapshot_year)
                         + change_index_year(elec_heating[common], snapshot_year))

    if drop_leap_day:
        elec_with_thermal = elec_with_thermal.loc[~((elec_with_thermal.index.month == 2) & (elec_with_thermal.index.day == 29))]

    elec_with_thermal = elec_with_thermal.sort_index(axis=1)
    
    #Filter by given snapshot
    elec_with_thermal=elec_with_thermal.loc[time_start:time_end]
    
    elec_with_thermal.to_csv(snakemake.output.elec_demand)
