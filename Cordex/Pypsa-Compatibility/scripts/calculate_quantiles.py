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

"""1. Load historical climate data for each year, handling specific cases for observed wind speed data.
2. Concatenate the historical climate data along the time dimension.
3. Adjust units for irradiance and longitude values.
4. Calculate quantiles for the historical data.
5. Write the quantiles to a NetCDF file, handling timeouts.

Variables:
    ``climate_variable`` (string): name of climate variable
    ``year_hist_start`` (int): Start year for historical data.
    ``year_hist_end`` (int): End year for historical data.
    ``years_hist`` (list): List of years in the historical period.
    ``lat_name`` (string): 'lat' for cordex or 'latitude' for era5 data
    ``lon_name`` (string): 'lon' for cordex or 'longitude' for era5 data
    ``time_name`` (string): Name of the time coordinate.
    ``quantiles`` (list): List of quantiles for empirical CDFs.
    ``maxTime`` (int): Maximum time for parallel computing.

Returns:
    xarray dataset: quantiles for given climate variable in given time periods
"""

import xarray as xr
import numpy as np
import warnings
from timeout_function_decorator import timeout
import os

# the quantile calculation always gives a warning for encountering all nan slices
# this filter is to supress this warning
warnings.filterwarnings('ignore', message='All-NaN slice encountered')

# this script does an empirical quantile delta bias adoption (https://doi.org/10.1175/JCLI-D-14-00754.1) based on the climate model data and era5 reanalysis data
config = snakemake.config

if 'OT_flow' in snakemake.output[0]:
    climate_variable = "mrro"
else:
    climate_variable = snakemake.wildcards.climate_variable

year_hist_start = config['bias_adaption']['yearHistStart']
year_hist_end = config['bias_adaption']['yearHistEnd']
years_hist = list(range(year_hist_start, year_hist_end+1))

number_of_quantiles = 100

lat_name = 'lat'
lon_name = 'lon'
if 'observed' in snakemake.output[0]:
    lat_name = 'latitude'
    lon_name = 'longitude'

climate_data_hist = xr.Dataset()

for i in range(len(years_hist)):
    # era 5 reports wind speed in two components
    if climate_variable == 'sfcWind' and 'observed' in snakemake.output[0]:
        climate_data_hist_file = snakemake.input[i*2]
        climate_data_hist_file2 = snakemake.input[i*2 + 1]
        climate_data_hist_one = xr.open_dataset(climate_data_hist_file)
        lat_length = climate_data_hist_one[lat_name].size
        lon_length = climate_data_hist_one[lon_name].size
        climate_data_hist_one = climate_data_hist_one.chunk({lon_name: int(
            lon_length/config['numberOfChunks']), lat_name: int(lat_length/config['numberOfChunks'])})
        climate_data_hist_one2 = xr.open_dataset(climate_data_hist_file2)
        climate_data_hist_one2 = climate_data_hist_one2.chunk({lon_name: int(
            lon_length/config['numberOfChunks']), lat_name: int(lat_length/config['numberOfChunks'])})
        wind_names_obs = config['bias_adaption']['climateVariablesDict']['sfcWind']
        climate_data_hist_one['sfcWind'] = np.sqrt(climate_data_hist_one[wind_names_obs[0]]**2
                                                   + climate_data_hist_one2[wind_names_obs[1]]**2)
        climate_data_hist_one = climate_data_hist_one.drop_vars(
            wind_names_obs[0]).rename({'latitude': 'lat', 'longitude': 'lon'})
        climate_data_hist_one2.close()
    elif 'observed' in snakemake.output[0]:
        climate_data_hist_file = snakemake.input[i]
        # observed data is already chunked
        climate_data_hist_one = xr.open_dataset(climate_data_hist_file)
        lat_length = climate_data_hist_one[lat_name].size
        lon_length = climate_data_hist_one[lon_name].size
        climate_data_hist_one = climate_data_hist_one.chunk({lon_name: int(
            lon_length/config['numberOfChunks']), lat_name: int(lat_length/config['numberOfChunks'])})
        # rename from era5 names to cordex names if calculated for observed data (do not rename for OT flow calc)
        if not 'OT_flow' in snakemake.output[0]:
            climate_data_hist_one = climate_data_hist_one.rename(
                {config['bias_adaption']['climateVariablesDict'][climate_variable]: climate_variable, 'latitude': 'lat', 'longitude': 'lon'})
    else:
        climate_data_hist_file = snakemake.input[i]
        climate_data_hist_one = xr.open_dataset(climate_data_hist_file)
        climate_data_hist_one = climate_data_hist_one.chunk({lon_name: int(config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})

    time_name = 'time'
    if 'observed' in snakemake.output[0]:
        for coord in climate_data_hist_one.coords:
            if 'time' in coord:
                time_name = coord

    if i == 0:
        climate_data_hist = climate_data_hist_one
    else:
        climate_data_hist = xr.concat(
            [climate_data_hist, climate_data_hist_one], dim=time_name)

# rename lon and lat name to lon and lat, as historic data should now be renamed to cordex names
if not 'OT_flow' in snakemake.output[0]:
    lat_name = 'lat'
    lon_name = 'lon'


climate_data_hist = climate_data_hist.chunk({lon_name: int(
    config['geography']['x_size']*len(years_hist)/config['numberOfChunks']), lat_name: int(config['geography']['y_size']*len(years_hist)/config['numberOfChunks']), time_name: -1})

# era5 provides irradiance in J/m^2. Must be converted to W/m^2 by dividing by 3600
if climate_variable == 'rsds' and 'observed' in snakemake.output[0]:
    climate_data_hist[climate_variable] = climate_data_hist[climate_variable]/3600

# era5 might report data in degree east/west, there fore values smaller or bigger than 180 degrees must be converted
climate_data_hist[lon_name] = xr.where(
    climate_data_hist[lon_name] > 180, -360+climate_data_hist[lon_name], climate_data_hist[lon_name])
climate_data_hist[lon_name] = xr.where(
    climate_data_hist[lon_name] < -180, 360+climate_data_hist[lon_name], climate_data_hist[lon_name])

# ensure physical validity before quantile calculation
climate_data_hist[climate_variable] = climate_data_hist[climate_variable].clip(min=0)

# build empirical cdfs
if "OT_flow" in snakemake.output[0]:
    quantiles = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
else:
    quantiles = np.arange(0, 1, 1/number_of_quantiles)
# calculate quantiles
climate_data_hist_q = climate_data_hist.quantile(quantiles, dim=time_name)

# parallel computing sometimes fails - after maximum time, process is stopped and restarted
if climate_variable == 'sfcWind':
    maxTime = int(config["maxTime"])*2
else:
    maxTime = int(config["maxTime"])

@timeout(maxTime)
def writeToNetCdf(file, path):    
    """Writes profile to path with timeout to catch crashing of dask

    Args:
        file (xarray dataset): file of calculated profile
        path (string): path to write file to
    """    
    file.to_netcdf(path)

try:
    writeToNetCdf(climate_data_hist_q, snakemake.output[0])
except TimeoutError:
    print('Error: time-out, to allocate more time, change maxTime parameter in config')
    os._exit(50)