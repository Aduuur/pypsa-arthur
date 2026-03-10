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
"""1. Open the irradiance dataset.
2. Adapt variable names based on the dataset being observed or not.
3. Convert irradiance values from J/m^2 to W/m^2 if using observed data.
4. Calculate solar capacity factor based on the selected option (1, 2, or 3).
5. Ensure solar capacity factor values are between 0 and 1.
6. Write the solar capacity factor data to a NetCDF file, handling timeouts.

Variables:
    ``config`` (yaml):  Snakemake configuration
    ``lat_name`` (string): 'lat' for cordex or 'latitude' for era5 data
    ``lon_name`` (string): 'lon' for cordex or 'longitude' for era5 data
    ``irradiance`` (xarray dataset): irrandiance in climate model
    ``temperature`` (xarray dataset): temperature in climate model
    ``tas_name`` (string): tas for cordex, different for era5
    ``rsds_name`` (string): rsds for cordex, different for era5
    ``wind_name` (string):` sfcWind for cordex, different for era5
    ``GStc``, ``TStc``, ``c1``, ``c2``, ``c3``,  ``c4``, ``beta``, ``gamma`` (float): specifications of PV cell
    ``Tcell`` (xarray dataset): temperature of cell
    ``cf_solar`` calculated solar capacity factor

Returns:
    xarray dataset: calculated solar capacity factor
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

irradiance = xr.open_dataset(str(snakemake.input[0]), chunks={lon_name: int(
    config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})

# adapt naming of variables depending on the use of era5 or cordex data
tas_name = 'tas'
rsds_name = 'rsds'
wind_name = 'sfcWind'
if 'observed' in snakemake.output[0]:
    tas_name = config['bias_adaption']['climateVariablesDict'][tas_name]
    rsds_name = config['bias_adaption']['climateVariablesDict'][rsds_name]

# era5 provides irradiance in J/m^2. Must be converted to W/m^2 by dividing by 3600
if 'observed' in snakemake.output[0]:
    irradiance[rsds_name] = irradiance[rsds_name]/3600

if config['solar']['option'] == 1:
    GStc = config['solar']['GStc']
    cf_solar = irradiance.rename({rsds_name: 'pv'})
    # following climix model https://doi.org/10.1016/j.rser.2014.09.041 option 1
    cf_solar['pv'] = 0.75*irradiance[rsds_name]/GStc
elif config['solar']['option'] == 2:
    c1 = config['solar']['c1']
    c2 = config['solar']['c2']
    c3 = config['solar']['c3']
    TStc = config['solar']['TStc']
    beta = config['solar']['beta']
    gamma = config['solar']['gamma']
    GStc = config['solar']['GStc']

    temperature = xr.open_dataset(str(snakemake.input[1]), chunks={lon_name: int(
        config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})
    cf_solar = temperature.rename({tas_name: 'pv'})

    # following climix model https://doi.org/10.1016/j.rser.2014.09.041 option 2
    # change input temperature from K to °C
    Tcell = c1+c2*(temperature[tas_name]-273.15)+c3*irradiance[rsds_name]
    cf_solar['pv'] = (1-beta*(Tcell-TStc)+gamma *
                      np.log10(irradiance[rsds_name]))*irradiance[rsds_name]/GStc
else:
    c1 = config['solar']['c1']
    c2 = config['solar']['c2']
    c3 = config['solar']['c3']
    c4 = config['solar']['c4']
    TStc = config['solar']['TStc']
    gamma = config['solar']['gamma']
    GStc = config['solar']['GStc']

    temperature = xr.open_dataset(str(snakemake.input[1]), chunks={lon_name: int(
        config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})
    if 'observed' in snakemake.output[0]:
        filename = snakemake.input[2]
        filename_2 = snakemake.input[3]
        wind_speed = xr.open_dataset(filename, chunks={lon_name: int(
            config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})
        wind_speed_2 = xr.open_dataset(filename_2, chunks={lon_name: int(
            config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})
        wind_names_obs = config['bias_adaption']['climateVariablesDict']['sfcWind']
        wind_speed[wind_name] = np.sqrt(
            wind_speed[wind_names_obs[0]]**2+wind_speed_2[wind_names_obs[1]]**2)
        wind_speed = wind_speed.drop_vars(wind_names_obs[0])
        wind_speed_2.close()
    else:
        wind_speed = xr.open_dataset(str(snakemake.input[2]), chunks={lon_name: int(
            config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})

    cf_solar = temperature.rename({tas_name: 'pv'})

    # following climix model https://doi.org/10.1016/j.rser.2014.09.041 option 3
    # change input temperature from K to °C
    Tcell = c1+c2*(temperature[tas_name]-273.15)+c3 * \
        irradiance[rsds_name]+c4*wind_speed[wind_name]
    cf_solar['pv'] = (1+gamma*(Tcell-TStc))*irradiance[rsds_name]/GStc

cf_solar = cf_solar.where(cf_solar > 0, 0)
cf_solar = cf_solar.where(cf_solar < 1, 1)

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
    writeToNetCdf(cf_solar, snakemake.output[0])
except TimeoutError:
    print('Error: time-out, to allocate more time, change maxTime parameter in config')
    os._exit(50)