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

"""1. Open the irradiance dataset from the input file.
2. Adapt variable naming based on the data source (ERA5 or cordex).
3. If the input data is ERA5, convert irradiance from J/m^2 to W/m^2.
4. Rename the irradiance variable to 'csp'.
5. Define a function ``writeToNetCdf`` to write the dataset to a NetCDF file with a timeout.
6. Attempt to write the irradiance dataset to the output file. If a timeout occurs, print an error message and exit.

Variables:
    ``lat_name`` (string): 'lat' for cordex or 'latitude' for era5 data
    ``lon_name`` (string): 'lon' for cordex or 'longitude' for era5 data
    ``irradiance`` (xarray dataset): Opened irradiance dataset from input file
    ``rsds_name`` (string): 'rsds' for cordex, from config for era 4
    ``config`` (yaml): Snakemake configuration

Returns:
    xarray dataset: available irradiance for csp
"""
import xarray as xr
import numpy as np
from timeout_function_decorator import timeout
import os

config = snakemake.config

lat_name = 'lat'
lon_name = 'lon'
if 'observed' in snakemake.output[0]:
    lat_name = 'latitude'
    lon_name = 'latitude'

irradiance = xr.open_dataset(str(snakemake.input.irradiance_file), chunks={lon_name: int(
    config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})

# adapt naming of variables depending on the use of era5 or cordex data
rsds_name = 'rsds'
if 'observed' in snakemake.output[0]:
    rsds_name = config['bias_adaption']['climateVariablesDict'][rsds_name]

# era5 provides irradiance in J/m^2. Must be converted to W/m^2 by dividing by 3600
if 'observed' in snakemake.output[0]:
    irradiance[rsds_name] = irradiance[rsds_name]/3600

irradiance = irradiance.rename({rsds_name: 'csp'})

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
    writeToNetCdf(irradiance, snakemake.output[0])
except TimeoutError:
    print('Error: time-out, to allocate more time, change maxTime parameter in config')
    os._exit(50)