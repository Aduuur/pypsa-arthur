"""
Compute daily heating and cooling demand using atlite
"""
import logging
from tempfile import NamedTemporaryFile

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config, load_cutout


# --- Main execution --- #
if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake("build_daily_thermal_demand")

    configure_logging(snakemake)
    set_scenario_config(snakemake)


    drop_leap_day = snakemake.params.drop_leap_day
    cutout_name = snakemake.input.cutout
    regions_onshore = snakemake.input.country_shapes
    pop_layout = snakemake.input.pop_layout_total
    thermal_type=snakemake.params.thermal_type
    
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
    if thermal_type == 'heating':

        heat_demand = cutout.heat_demand(
            threshold=15.0,
            a=1.0,
            constant=0.0,
            hour_shift=0.0,
            matrix=M.T,
            index=clustered_regions.index,
            show_progress=False,
        ).sel(time=daily)

        df = pd.DataFrame(heat_demand.values, index=heat_demand["time"].values, columns=heat_demand["name"].values)

    elif thermal_type == 'cooling':

        cool_demand = cutout.cooling_demand(
            threshold=18.0,
            a=1.0,
            constant=0.0,
            hour_shift=0.0,
            matrix=M.T,
            index=clustered_regions.index,
            show_progress=False,
        ).sel(time=daily)

        df = pd.DataFrame(cool_demand.values, index=cool_demand["time"].values, columns=cool_demand["name"].values)


    df.to_csv(snakemake.output[0])