'''
Loads capacity factors from cd2es and aggregate by map. Returns csv with timeseries/bus for powerplant and cooling type.
'''

import os
import xarray as xr
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import logging

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake("build_powerplants_tpp_p_max_pu")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    #input
    pp_type = snakemake.params.pp_type
    cool_type = snakemake.params.cool_type
    clim = snakemake.params.clim
    climatedata_generators_t_p_max_pu_path = snakemake.params.climatedata_generators_t_p_max_pu_path

    cooling_mapping={'CL':'closed-loop','OT':'once-through'}

    regions = gpd.read_file(snakemake.input.regions_onshore)

    climdat_path=os.path.join(climatedata_generators_t_p_max_pu_path,f"tpp{cool_type}_p{pp_type}_m{clim}_notAgg_pypsa.nc")
    climate_data=xr.open_dataset(climdat_path)

    # Convert DataFrame to GeoDataFrame
    lon_name='x'
    lat_name='y'

    # Extract coordinates from climate_data file as 1D arrays
    lons = climate_data[lon_name].values
    lats = climate_data[lat_name].values

    # Create a DataFrame with all combinations of lon and lat
    df = pd.DataFrame([(lon, lat) for lat in lats for lon in lons],columns=[lon_name, lat_name])

    # Convert DataFrame to GeoDataFrame
    points = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df[lon_name], df[lat_name])])

    # Check that both geopandas files have the same reference system
    points.crs = regions.crs

    # find intersections between points and regions and drop possible duplicates in lon and lat
    names = gpd.sjoin(points, regions, how='inner',predicate='intersects').drop_duplicates([lon_name, lat_name])

    # convert to xarray
    names = names.set_index([lon_name, lat_name])['name'].to_xarray()

    merge_ds = xr.merge([names.fillna('misc'), climate_data]).groupby('name')

    res= merge_ds.mean(skipna=True).drop_sel(name='misc').to_dataframe().groupby(['time', "name"]).mean().reset_index().pivot(index='name', columns='time', values='specific generation')
    res=res.T.add_suffix(f' {pp_type} {cooling_mapping[cool_type]}')
    res.to_csv(snakemake.output.CF_profile_tpp_agg)