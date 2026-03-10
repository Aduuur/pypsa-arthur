# cd2es - covert cordex climate data to energy system input data
# Copyright (C) 2024 Leonie Sara Plaga

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""1. Open climate data file with appropriate chunking along the time dimension.
2. Adjust longitude values if needed.
3. Determine regions file based on configuration.
4. Read regions file as a GeoDataFrame.
5. Extract coordinates from climate data and create a DataFrame with all combinations of lon and lat.
6. Convert the DataFrame to a GeoDataFrame.
7. Find intersections between points and regions, drop duplicates, and convert to xarray.
8. For offshore wind, filter locations based on sea depth.
9. If not all data shall be aggregated, calculate full load hours and select the best x percent.
10. Convert the merged dataset to a DataFrame (sum for hydro, mean for all other).
11. Drop unnecessary columns and set the column index.
12. Write the final DataFrame to a CSV file specified in the output.

Variables:
    ``config`` (yaml): Snakemake configuration file
    ``domain`` (string): Snakemake wildcard for domain
    ``climate_variable`` (string): name of climate variable
    ``lat_name`` (string): 'lat' for cordex or 'latitude' for era5 data
    ``lon_name`` (string): 'lon' for cordex or 'longitude' for era5 data
    ``time_name`` (string): 'time' or determined from climate data file
    ``choose_best_x_percent`` (float): Percentage of locations to be used in aggregation
    ``climate_data_file`` (xarray dataset): Input climate data file

Returns:
    pandas dataframe: aggregated file
"""

import pandas as pd
from shapely.geometry import Point
import xarray as xr
import numpy as np
import geopandas as gpd
from timeout_function_decorator import timeout
import os

config = snakemake.config
domain = snakemake.wildcards.domain
wildcard_names = [wildcard_name for wildcard_name,
                  wildcard_value in snakemake.wildcards.items()]

@timeout(float(config["maxTime"]))
def load(ds):
    """Loads a chunked dataset with timeout to catch crashing of dask.

    Args:
        ds (xarray dataset): chunked dataset to be loaded

    Returns:
       xarray dataset: not chunked dataset
    """    
    return ds.load()

if 'technology' in wildcard_names:
    climate_variable = snakemake.wildcards.technology
else:
    climate_variable = snakemake.wildcards.climate_variable
    if 'observed' in snakemake.output[0]:
        climate_variable = config['bias_adaption']['climateVariablesDict'][climate_variable]

lat_name = 'lat'
lon_name = 'lon'
if 'observed' in snakemake.output[0]:
    lat_name = 'latitude'
    lon_name = 'longitude'

if climate_variable == 'sfcWind' or climate_variable == 'wind':
    if 'offshore' in snakemake.output[0]:
        choose_best_x_percent = config['choose_only_x_percent_of_country']['offwind']
    else:
        choose_best_x_percent = config['choose_only_x_percent_of_country']['onwind']
elif climate_variable == 'pv':
    choose_best_x_percent = config['choose_only_x_percent_of_country']['pv']
else:
    choose_best_x_percent = 1

climate_data_file = snakemake.input.climate_data_file

time_name = 'time'
if 'observed' in snakemake.output[0]:
    climate_data_test = xr.open_dataset(str(climate_data_file))
    lat_length = climate_data_test[lat_name].size
    lon_length = climate_data_test[lon_name].size
    climate_data_test = climate_data_test.chunk({lon_name: int(
        lon_length/config['numberOfChunks']), lat_name: int(lat_length/config['numberOfChunks'])})
    for coord in climate_data_test.coords:
        if 'time' in coord:
            time_name = coord


# in this case, the chunking must go along the time dimension to allow for geographical aggregation
climate_data = xr.open_dataset(str(climate_data_file))
climate_data = climate_data.chunk({time_name: int(8760/(config['numberOfChunks']**2*4))})

# era5 might report data in degree east/west, there fore values smaller or bigger than 180 degrees must be converted
climate_data[lon_name] = xr.where(
    climate_data[lon_name] > 180, -360+climate_data[lon_name], climate_data[lon_name])
climate_data[lon_name] = xr.where(
    climate_data[lon_name] < -180, 360+climate_data[lon_name], climate_data[lon_name])

if config['use_custom_bus_map']:
    regions_file = config['bus_map']
else:
    regions_file = f'resources/maps/{domain}.geojson'

if 'offshore' in snakemake.output[0]:
    if config['use_custom_bus_map']:
        regions_file = config['offshore_map']
    else:
        regions_file = f'resources/maps/{domain}_offshore.geojson'

# read regions file
regions = gpd.read_file(regions_file)

# Extract coordinates from climate_data file as 1D arrays
lons = climate_data[lon_name].values
lats = climate_data[lat_name].values

# Create a DataFrame with all combinations of lon and lat
df = pd.DataFrame([(lon, lat) for lat in lats for lon in lons],
                  columns=[lon_name, lat_name])

# Convert DataFrame to GeoDataFrame
points = gpd.GeoDataFrame(
    df, geometry=[Point(xy) for xy in zip(df[lon_name], df[lat_name])])

# Check that both geopandas files have the same reference system
points.crs = regions.crs

# find intersections between points and regions and drop possible duplicates in lon and lat
names = gpd.sjoin(points, regions, how='inner',
                  predicate='intersects').drop_duplicates([lon_name, lat_name])

# convert to xarray
names = names.set_index([lon_name, lat_name])['name'].to_xarray()

# for offshore wind: read in gebco height file to omit places with sea depth < max depth
if 'offshore' in snakemake.output[0]:
    gebco = xr.open_dataset(snakemake.input.gebco_file, chunks={'lon': 25, 'lat': 25}).reindex_like(
        names, method='nearest')
    if 'observed' in climate_data_file:
        gebco = gebco.rename({'lon': 'longitude', 'lat': 'latitude'})
    names_height = xr.merge(
        [names, gebco])
    names = xr.where(names_height['elevation'] < -config['offwind']
                     ['max_depth'], np.nan, names_height['name'])
    names.name = 'name'
    
    try:
        names = load(names.dropna(dim="lat", how="all").dropna(dim="lon", how="all"))
    except TimeoutError:
        print('Error: time-out, to allocate more time, change maxTime parameter in config')
        os._exit(50)

# if locations shall be filtered (e.g. only use 30% best locations per country)
if choose_best_x_percent != 1:
    # calculate full load hours per location
    fullLoadHours = climate_data.chunk({lon_name: int(config['geography']['x_size']/config['numberOfChunks']), lat_name: int(
        config['geography']['y_size']/config['numberOfChunks']), time_name: -1}).sum(dim=time_name)
    # add countries to fullLoadHours, 'misc' is proxy for points which do not belong to any country, will be dropped later
    fullLoadHours_names = xr.merge([fullLoadHours, names.to_dataset().fillna("misc")],join="outer",fill_value="misc").set_coords("name")

    # calculate best x percent value of fullLoadHours for all buses
    try:
        quantile = load(fullLoadHours_names).groupby('name').quantile(1-choose_best_x_percent)
    except TimeoutError:
        print('Error: time-out, to allocate more time, change maxTime parameter in config')
        os._exit(50)
    
    # set all values for locations with fullLoadHours under the quantile value to nan
    merge_ds = climate_data.sortby(lat_name).where((fullLoadHours_names.groupby(
        'name') > quantile)).groupby('name')
else:
    # merge names with climate data file to assign names to coordinates and groupby names
    merge_ds = xr.merge([names.fillna('misc'), climate_data],join="outer").groupby("name")

maxTime = int(config["maxTime"])
if 'observed' in snakemake.output[0]:
    maxTime *= 2

@timeout(maxTime)
def toDataFrame(merge_ds, climate_variable, sum=False):
    """
    Performs aggregation with timeout, because dask sometimes crashes.
    Args:
        merge_ds (xarray dataset): climate data file to be aggregated
        climate_variable (string): name of climate variable
        sum (bool, optional): If False, calculate mean for every geometry, else calculate sum (only for hydro inflow). Defaults to False.

    Returns:
       pandas dataframe: aggregated file
    """    
    if sum:
        if 'observed' in snakemake.output[0]:
            return merge_ds.sum(skipna=True).drop_sel(name='misc').to_dataframe().reset_index().pivot(
                index='name', columns=time_name, values=climate_variable)
        else:
            return merge_ds.sum(skipna=True).drop_sel(name='misc').to_dataframe().groupby([time_name, "name"]).mean().reset_index().pivot(
                index='name', columns=time_name, values=climate_variable)
    else:
        if 'observed' in snakemake.output[0]:
            return merge_ds.mean(skipna=True).drop_sel(name='misc').to_dataframe().reset_index().pivot(
                index='name', columns=time_name, values=climate_variable)
        else:
            return merge_ds.mean(skipna=True).drop_sel(name='misc').to_dataframe().groupby([time_name, "name"]).mean().reset_index().pivot(
                index='name', columns=time_name, values=climate_variable)


if climate_variable == 'hydro':  # hydro needs sum of all power instead of average
    # calculate sum per name, convert to pandas and drop unnecessary columns
    try:
        final_df = toDataFrame(merge_ds, climate_variable, True)
    except TimeoutError:
        print('Error: time-out, to allocate more time, change maxTime parameter in config')
        os._exit(50)
else:
    # calculate mean per name, convert to pandas and drop unnecessary columns
    try:
        final_df = toDataFrame(merge_ds, climate_variable, False)
    except TimeoutError:
        print('Error: time-out, to allocate more time, change maxTime parameter in config')
        os._exit(50)

if 'height' in final_df.columns:
    final_df.drop(columns='height', inplace=True)

# set column index
final_df.columns = [f't{i:06d}' for i in range(
    1, len(climate_data[time_name].values)+1)]

# write to csv
final_df.to_csv(snakemake.output[0], sep=';')