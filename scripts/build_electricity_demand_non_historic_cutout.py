"""
Generate hourly profiles and calculate thermal electricity demand per country and sector.
"""


import logging
from tempfile import NamedTemporaryFile
from itertools import product

import pandas as pd

import xarray as xr



from scripts._helpers import configure_logging, set_scenario_config, generate_periodic_profiles, checkBool


def calc_hourly_space_demand(daily_space_demand, intraday_profiles,drop_leap_day, uses, sectors):
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
    if drop_leap_day:
        ds = ds.sel(snapshots=~((ds.snapshots.dt.month == 2) & (ds.snapshots.dt.day == 29)))
    return ds


def load_and_reindex_demand(thermal_demand_path):
    """
    Read daily heating and ccoling demand shape. Reindex daily demand to hourly and forward fill missing values.
    """
    hdd = pd.read_csv(
        thermal_demand_path,
        index_col=0,
        parse_dates=True
    )

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
    return hdd, hdd_hourly


def thermal_elec_demand(hourly_thermal_demand: xr.Dataset, energy_totals: pd.DataFrame, year: int,
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

        electric_thermal_supply[name] = (thermal_demand_shape[name] / thermal_demand_shape[name].sum()).multiply(year_totals[f'electricity {name}']) * 1e6

    electric_thermal_supply = pd.concat(electric_thermal_supply, axis=1)
    electric_thermal_supply_agg = electric_thermal_supply.T.groupby(level=1).sum().T

    return electric_thermal_supply_agg


# --- Main execution --- #
if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake("build_electricity_demand_non_historic_cutout")

    configure_logging(snakemake)
    set_scenario_config(snakemake)


    # Input files
    intraday_profiles = snakemake.input.heat_profile
    cooling_demand_path = snakemake.input.energy_totals_cool
    heating_demand_path = snakemake.input.energy_totals_heat
    electric_demand_no_thermal_path = snakemake.input.demand_no_thermal
    drop_leap_day = checkBool(snakemake.params.drop_leap_day)
    energy_totals_year=int(snakemake.params.energy_totals_year)
    et_regression = checkBool(snakemake.params.et_regression)

    time_start = pd.to_datetime(snakemake.params.snapshots["start"])
    time_end = pd.to_datetime(snakemake.params.snapshots["end"])
    snapshot_year = int(time_start.year)
    
    # Note: whole script only applies if config['ee']['historic_cutout']['enable'] == true (if statement in snakefile)
    if et_regression == True:
        year_demand = snapshot_year  # equals time_end.year
        logger.info(f'Using demand year {year_demand} derived by regression')
    elif et_regression == False:
        year_demand = energy_totals_year
        logger.info(f'Using reported demand year {year_demand}')
    else:
        raise ValueError('config[ee][non_historic_cutout][et_regression] must be false or true')




    # Hourly demand
    hdd, daily_space_heating_demand = load_and_reindex_demand(snakemake.input.hdd)
    ds_heat = calc_hourly_space_demand(daily_space_heating_demand, intraday_profiles,drop_leap_day,
                                       uses=["water", "space"], sectors=["residential", "services"])

    cdd, daily_space_cooling_demand = load_and_reindex_demand(snakemake.input.cdd)
    ds_cool = calc_hourly_space_demand(daily_space_cooling_demand, intraday_profiles,drop_leap_day,
                                       uses=["space"], sectors=["residential", "services"])

    # Load total energy data
    electric_demand_no_thermal = pd.read_csv(electric_demand_no_thermal_path, index_col="Date", parse_dates=["Date"])
    energy_totals_cool = pd.read_csv(cooling_demand_path, index_col="country")
    energy_totals_heat = pd.read_csv(heating_demand_path, index_col=["country"])
    
    # Compute thermal electricity demand
    elec_cooling = thermal_elec_demand(ds_cool, energy_totals_cool, year_demand,
                                       uses=["space"], sectors=["residential", "services"])
    elec_heating = thermal_elec_demand(ds_heat, energy_totals_heat, year_demand,
                                       uses=["space", "water"], sectors=["residential", "services"])

    # Combine with non-thermal demand
    common = list(set(elec_cooling.columns) & set(elec_heating.columns) & set(electric_demand_no_thermal.columns))

    def change_index_year(df, new_year):
        df = df.copy()
        df.index = df.index.map(lambda x: x.replace(year=new_year))
        return df


    electric_demand_no_thermal_time=electric_demand_no_thermal.copy()
    electric_demand_no_thermal_time.index.name = "time"
    aggregated_demand = xr.Dataset(
        {
            "elec_cooling_demand": (["time", "country"], elec_cooling[common].values),
            "elec_heating_demand": (["time", "country"], elec_heating[common].values),
            "elec_demand_without_thermal": (["time", "country"],  electric_demand_no_thermal_time[common].values),
        },
        coords={
            "time": change_index_year(electric_demand_no_thermal[common],snapshot_year).index,
            "country": common,
        },
    )
    aggregated_demand.to_netcdf(snakemake.output.elec_thermal_demand)


    elec_with_thermal = (change_index_year(electric_demand_no_thermal[common], snapshot_year)
                         + change_index_year(elec_cooling[common], snapshot_year)
                         + change_index_year(elec_heating[common], snapshot_year))

    if drop_leap_day:
        elec_with_thermal = elec_with_thermal.loc[~((elec_with_thermal.index.month == 2) & (elec_with_thermal.index.day == 29))]

    elec_with_thermal = elec_with_thermal.sort_index(axis=1)
    
    #Filter by given snapshot
    elec_with_thermal=elec_with_thermal.loc[time_start:time_end]
    
    elec_with_thermal.to_csv(snakemake.output.elec_demand)