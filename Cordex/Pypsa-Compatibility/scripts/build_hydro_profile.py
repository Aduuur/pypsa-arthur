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

"""1. Read ``powerplant_database`` CSV file.
2. Open the runoff as ``mrro``.
3. Identify the nearest point to each power plant in the ``mrro`` file.
4. Convert ``powerplant_database`` to xarray, grouping by longitude and latitude.
5. Merge ``mrro`` and ``powerplant_database`` into one dataset.
6. Set negative values in the dataset to 0.
7. Calculate hourly hydro capacity based on actual runoff, historic runoff, and installed capacity.
8. Cap the capacity if it exceeds the installed capacity.
9. Set capacity values less than 0 to 0.
10. Convert capacity to a dataset named 'hydro'.
11. Write the dataset to a NetCDF file, with a timeout mechanism for parallel computing. If a timeout occurs, print an error message and exit.

Variables:
    ``powerplant_database`` (pandas dataframe): information on power plants in Europe (capacity, location etc.)
    ``runoff_file`` (xarray dataset): river runoff file

Returns:
    xarray data set: hydro inflow
"""

import pandas as pd
import xarray as xr
from timeout_function_decorator import timeout
import os

config = snakemake.config

year = int(snakemake.wildcards.year)

powerplant_database = pd.read_csv(
    snakemake.input.powerplant_database, sep=';', index_col='id')

lat_name = 'lat'
lon_name = 'lon'
if 'observed' in snakemake.output[0]:
    lat_name = 'latitude'
    lon_name = 'longitude'

mrro_name = 'mrro'
if 'observed' in snakemake.output[0]:
    mrro_name = config['bias_adaption']['climateVariablesDict'][mrro_name]
    mrro = xr.open_dataset(snakemake.input.runoff_file[0], chunks={lon_name: int(
        config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})
    # interpolate mrro
    lastDayNumber = 363
    # remove 29.02. from leap years
    if year % 4 == 0 and year % 400 != 0:
        lastDayNumber = 364
        time = pd.to_datetime(range(0, 8760+24), unit='h', origin=f'{year}0101')
        time = time.delete(range(59*24, 60*24))
    else:
        time = pd.to_datetime(range(0, 8760), unit='h', origin=f'{year}0101')
    mrro = mrro.interp(valid_time=time)

    powerplant_database = powerplant_database.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
else:
    mrro = xr.open_dataset(snakemake.input.runoff_file, chunks={lon_name: int(
        config['geography']['x_size']/config['numberOfChunks']), lat_name: int(config['geography']['y_size']/config['numberOfChunks'])})

# Make latitude and longitude real indexes
if 'observed' in snakemake.output[0]:
    mrro = mrro.set_index(latitude="latitude", longitude="longitude")
else:
    mrro = mrro.set_index(lat="lat", lon="lon")

# identify point nearest to power plant in mrro file
for id in powerplant_database.index:
    if 'observed' in snakemake.output[0]:        
        nearestMrro = mrro.sel(
            latitude=powerplant_database.loc[id, lat_name], longitude=powerplant_database.loc[id, lon_name], method='nearest')
    else:
        nearestMrro = mrro.sel(
            lat=powerplant_database.loc[id, lat_name], lon=powerplant_database.loc[id, lon_name], method='nearest')
    powerplant_database.loc[id, lat_name] = nearestMrro[lat_name].values
    powerplant_database.loc[id, lon_name] = nearestMrro[lon_name].values

# convert powerplant database to xarray
pp_db_xr = powerplant_database.groupby([lon_name, lat_name]).agg(
    {'average runoff': 'mean', 'installed_capacity_MW': 'sum'}).to_xarray()

# merge to one dataset
# mrro = abs(xr.merge([mrro, pp_db_xr]))
mrro = xr.merge([mrro, pp_db_xr])
mrro = xr.where(mrro < 0, 0, mrro)

# calculate hourly hydro capacity by multiplying with the ratio of the actual runoff and the average historic
# runoff with the installed capacity
capacity = mrro[mrro_name] / mrro['average runoff'] * \
    mrro['installed_capacity_MW']
# cap time series if capacity exceeds installed capacity
capacity = xr.where(
    capacity > mrro['installed_capacity_MW'], mrro['installed_capacity_MW'], capacity)

# if smaller than 0, set to 0
capacity = xr.where(capacity < 0, 0, capacity)

capacity = capacity.to_dataset(name='hydro')

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
    writeToNetCdf(capacity, snakemake.output[0])
except TimeoutError:
    print('Error: time-out, to allocate more time, change maxTime parameter in config')
    os._exit(50)
