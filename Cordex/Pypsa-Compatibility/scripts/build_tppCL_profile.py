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

"""1. Open temperature data from input
2. Calculate availability for closed-loop cooled thermal powerplants based on temperature and constants
3. Rename availability variable based on ``plantType``
4. Write availability to NetCDF output
5. Handle timeout errors and print message if exceeded maximum time

Variables:
    ``lat_name`` (string): 'lat' for cordex or 'latitude' for era5 data
    ``lon_name`` (string): 'lon' for cordex or 'longitude' for era5 data
    ``plantType`` (string): type of plant to calculate availability for
    ``plantTypeData`` (string): replace plantType for some plantTypes without data: 'Coal' for biomass, 'CCGT' for H2, otherwise same as ``plantType``
    ``tas_name`` (string): 'tas' or configured based on ``bias_adaption`` for ERA5 or cordex data
    ``temp_const`` (float): temperature constant for specific ``plantTypeData``
    ``T_health`` (float): health temperature for specific ``plantTypeData``

Returns:
    xarray dataset: availability for chosen power plant type
"""

import xarray as xr
from timeout_function_decorator import timeout
import os

config = snakemake.config

lat_name = 'lat'
lon_name = 'lon'
if 'observed' in snakemake.output[0]:
    lat_name = 'latitude'
    lon_name = 'latitude'

plantType = snakemake.wildcards.plantType
if plantType.lower() == "biomass":
    plantTypeData = "Coal"
elif plantType.lower() == "h2":
    plantTypeData = "CCGT"
else:
    plantTypeData = plantType

# adapt naming of variables depending on the use of era5 or cordex data
tas_name = 'tas'
if 'observed' in snakemake.output[0]:
    tas_name = config['bias_adaption']['climateVariablesDict'][tas_name]

# based on 'A modeling and optimization framework for power systems design with operational flexibility and resilience against extreme heat waves and drought events', Abdin et al., 2019

temperature = xr.open_dataset(str(snakemake.input[0]), chunks={lon_name: int(
    config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})

temp_const = config['tpp']['CL'][plantTypeData.lower()]['temp_const']
T_health = config['tpp']['CL'][plantTypeData.lower()]['T_health']

availability = temperature.where(
    temperature < T_health, (1-temp_const*(temperature-T_health))).where(temperature > T_health, 1)

availability = availability.rename({tas_name: f'tppCL_p{plantType}'})

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
    writeToNetCdf(availability, snakemake.output[0])
except TimeoutError:
    print('Error: time-out, to allocate more time, change maxTime parameter in config')
    os._exit(50)