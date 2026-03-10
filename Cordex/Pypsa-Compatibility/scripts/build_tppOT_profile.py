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

"""1. Calculate ``temperatureStream`` based on the provided formula
2. Adjust ``time`` based on leap years and remove duplicates
3. Modify ``mrro`` for non-observed output by adding missing timesteps
4. Calculate ``waterAvailability`` based on quantile flows and plant availability
5. Rename and adjust ``temperatureStream`` based on the plant type
6. Extract configuration parameters for temperature calculations
7. Calculate ``availability`` based on temperature conditions and water availability
8. Handle NaN values in the availability calculation
9. Write the processed ``availability`` data to a NetCDF file at the specified output path
10. If a timeout occurs, print an error message and exit with code 50.

Variables:
    ``lat_name`` (string): 'lat' for cordex or 'latitude' for era5 data
    ``lon_name`` (string): 'lon' for cordex or 'longitude' for era5 data
    ``plantType`` (string): type of plant (biomass, h2, etc.)
    ``year`` (int): year extracted from the input
    ``temperature`` (xarray dataset): Dataset containing temperature data from climate model
    ``mrro`` (xarray dataset): Dataset containing river runoff data
    ``quantileFlows`` (xarray dataset): Dataset containing quantiles for historic river runoff

Returns:
    xarray dataset: availability for chosen power plant type
"""

import xarray as xr
import numpy as np
import pandas as pd
import warnings
from timeout_function_decorator import timeout
import os

# the quantile calculation always gives a warning for encountering all nan slices
# this filter is to supress this warning
warnings.filterwarnings('ignore', message='All-NaN slice encountered')

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
    
year = int(snakemake.wildcards.year)

# adapt naming of variables depending on the use of era5 or cordex data
tas_name = 'tas'
if 'observed' in snakemake.output[0]:
    tas_name = config['bias_adaption']['climateVariablesDict'][tas_name]

mrro_name = 'mrro'
if 'observed' in snakemake.output[0]:
    mrro_name = config['bias_adaption']['climateVariablesDict'][mrro_name]

# based on 'Drought and climate change impacts on cooling water shortages and electricity prices in Great Britain', Byers et al., 2020, 10.1038/s41467-020-16012-2

plantAvailabilityDict = {0.1: 1, 0.09: 0.9, 0.08: 0.8,
                         0.07: 0.7, 0.06: 0.6, 0.05: 0.5, 0.04: 0.4, 0.03: 0.3, 0.02: 0}

# based on 'A modeling and optimization framework for power systems design with operational flexibility and resilience against extreme heat waves and drought events', Abdin et al., 2019, 10.1016/j.rser.2019.06.006

temperature = xr.open_dataset(str(snakemake.input.tas), chunks = {lon_name: int(config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})
mrro = xr.open_dataset(str(snakemake.input.mrro), chunks = {lon_name: int(config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})
quantileFlows = xr.open_dataset(str(snakemake.input.historicFlowFile))

# calculate stream temperature

tStreamMin = 273  # minimum stream temperature in K
tStreamMax = 303.4  # maximum stream temperature in K
lambdaStream = 0.14  # constant for exponential function
tStreamIn = 289.5  # temperature at inflection point in K

temperatureStream = tStreamMin + \
    (tStreamMax-tStreamMin)/(1+np.exp(lambdaStream*(tStreamIn-temperature)))

lastDayNumber = 363
# remove 29.02. from leap years
if year % 4 == 0 and year % 400 != 0:
    lastDayNumber = 364
    time = pd.to_datetime(range(0, 8760+24), unit='h', origin=f'{year}0101')
    if not 'observed' in snakemake.output[0]:
        time = time.delete(range(59*24, 60*24))
else:
    time = pd.to_datetime(range(0, 8760), unit='h', origin=f'{year}0101')

if 'observed' in snakemake.output[0]:
    mrro = mrro.interp(valid_time=time)
else:
    mrro = mrro.interp(time=time)

if not 'observed' in snakemake.output[0]:
    # add missing timesteps (first day = second day, last day = second last day) (not for era5 data)
    # remove 29.02. from leap years
    if year % 4 == 0 and year % 400 != 0:
        mrro = mrro.where(mrro['time.dayofyear'] != 60, drop=True)

    firstDay = mrro.where(mrro['time.dayofyear'] == 2, drop=True)
    for ix in range(24):
        firstDay['time'].values[ix] = firstDay['time'].values[ix] - \
            24*60*60*1000000000
    # delete first 12 timesteps
    mrro = mrro.drop_isel(time=range(0, 12))
    # remove duplicates
    for time in firstDay['time'].values:
        if time in mrro['time'].values:
            firstDay = firstDay.drop_sel(time=time)
    mrro = xr.concat([firstDay, mrro], dim='time')

    lastDay = mrro.where(mrro['time.dayofyear'] == lastDayNumber, drop=True)
    for ix in range(24):
        lastDay['time'].values[ix] = lastDay['time'].values[ix] + \
            2*24*60*60*1000000000
    # remove duplicates
    mrro = mrro.drop_isel(time=range(8749, 8760))
    # remove duplicates
    for time in lastDay['time'].values:
        if time in mrro['time'].values:
            lastDay = lastDay.drop_sel(time=time)

    mrro = xr.concat([mrro, lastDay], dim='time')

waterAvailability = mrro.copy()

waterAvailability = waterAvailability.where(
    mrro <= quantileFlows.sel(quantile=0.1), 1)
for q in [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]:
    waterAvailability = waterAvailability.where(
        mrro > quantileFlows.sel(quantile=q), plantAvailabilityDict[q])

waterAvailability = waterAvailability.drop_vars(
    'quantile').rename({mrro_name: f'tppOT_p{plantType}'})

temperatureStream = temperatureStream.rename({tas_name: f'tppOT_p{plantType}'})

if 'height' in temperatureStream.coords:
    temperatureStream = temperatureStream.drop_vars('height')

temp_const = config['tpp']['OT'][plantTypeData.lower()]['temp_const']
T_health = config['tpp']['OT'][plantTypeData.lower()]['T_health']
T_shutdown = config['tpp']['OT'][plantTypeData.lower()]['T_shutdown']
T_outmax = config['tpp']['OT'][plantTypeData.lower()]['T_outmax']
deltaT = config['tpp']['OT'][plantTypeData.lower()]['deltaT']

availability = waterAvailability

T_risk = T_outmax-deltaT*1/waterAvailability
delta = waterAvailability+temp_const*deltaT - \
    temp_const*waterAvailability*(T_outmax-T_health)

availability = availability.where(temperatureStream < T_health, (waterAvailability *
                                                                 (1-temp_const*(temperatureStream-T_health))))
availability = availability.where(temperatureStream < T_risk, (waterAvailability *
                                                               delta*(T_outmax-temperatureStream)/deltaT))
availability = availability.where(temperatureStream < T_shutdown, 0)

availability = xr.where(np.isnan(mrro.rename(
    {mrro_name: f'tppOT_p{plantType}'})), np.nan, availability)

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