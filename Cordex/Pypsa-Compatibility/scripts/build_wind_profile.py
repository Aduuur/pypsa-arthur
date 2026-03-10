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

"""1. Open climate data file(s)
2. Load fit parameters from a pickle file
3. Apply the fit function to wind speed data to obtain wind speed capacity factors
4. Adjust wind speed values to be within a specific range
5. Rename the wind variable
6. Write the processed wind speed data to a NetCDF file, with a timeout mechanism for parallel computing. If a timeout occurs, print an error message.

Variables:
    ``lat_name`` (string): 'lat' for cordex or 'latitude' for era5 data
    ``lon_name`` (string): 'lon' for cordex or 'longitude' for era5 data
    ``wind_name`` (string): 'sfcWind'
    ``wind_speed`` calculated wind speed data
    ``wind_names_obs`` (dict): dictionary of wind variable names in era5 data
    ``filename`` (string): input climate data file
    ``filename_2`` (string): additional input climate data file (only for era5)
    ``popt``(list): fit parameters for turbine curve

Returns:
    xarray dataset: calculated wind capacity factors
"""

import xarray as xr
import numpy as np
import pickle
from timeout_function_decorator import timeout
import os

config = snakemake.config

lat_name = 'lat'
lon_name = 'lon'
if 'observed' in snakemake.output[0]:
    lat_name = 'latitude'
    lon_name = 'latitude'

# adapt naming of variables depending on the use of era5 or cordex data
wind_name = 'sfcWind'

# in the era5 data, windspeeds are reported as u and v component of wind -> has to be summed to find
if 'observed' in snakemake.output[0]:
    filename = snakemake.input.climate_data[0]
    filename_2 = snakemake.input.climate_data[1]
    wind_speed = xr.open_dataset(filename, chunks = {lon_name: int(config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})
    wind_speed_2 = xr.open_dataset(filename_2, chunks = {lon_name: int(config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})
    wind_names_obs = config['bias_adaption']['climateVariablesDict']['sfcWind']
    wind_speed[wind_name] = np.sqrt(
        wind_speed[wind_names_obs[0]]**2+wind_speed_2[wind_names_obs[1]]**2)
    wind_speed = wind_speed.drop_vars(wind_names_obs[0])
    wind_speed_2.close()
else:
    filename = snakemake.input.climate_data
    wind_speed = xr.open_dataset(filename, chunks = {lon_name: int(config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})

# clip wind to limits from calculate_turbine_curve.py
wind_speed[wind_name] = wind_speed[wind_name].clip(min=0, max=40)

# load fit parameter from pickle
file = open(snakemake.input.turbine_curve, 'rb')
popt = pickle.load(file)
file.close()


def f(x, a, b, c, d):
    """fit function for turbine curve

    Args:
        x (float): wind speed
        a, b, c, d (float): constant from fit in calculate turbine curve

    Returns:
       float: capacity factor of turbine at wind speed x
    """    
    return np.exp(-x**2*a)*(b*x+c*x**2+d*x**3)


wind_speed = f(wind_speed, popt[0], popt[1], popt[2], popt[3])

wind_speed = wind_speed.where(wind_speed > 0, 0)
wind_speed = wind_speed.where(wind_speed < 1, 1)

wind_speed = wind_speed.rename({wind_name: 'wind'})

# parallel computing sometimes fails - after maximum time, process is stopped and restarted
@timeout(float(config["maxTime"]))
def writeToNetCdf(file, path):    
    """Writes profile to path with timeout to catch crashing of dask

    Args:
        file (xarray dataset): file of calculated profile
        path (string): path to write file to
    """    
    file.to_netcdf(path)

try:
    writeToNetCdf(wind_speed, snakemake.output[0])
except TimeoutError:
    print('Error: time-out, to allocate more time, change maxTime parameter in config')
    os._exit(50)