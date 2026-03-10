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

"""1. Define the ``config`` variable using ``snakemake.config``.
2. Create the ``data_dir`` path based on the doSummary of function wildcard.
3. Extract ``model``, ``year``, ``toTemporal``, and ``climate_variable`` from ``snakemake.wildcards``.
4. Check if the output directory for the remapped data does not exist, then create it.
5. Set the ``input_file`` and ``output_file`` paths, adjusting for Windows naming if necessary.
6. If the temporal resolution is hourly (``toTemporal == '1h'``):

   * Interpolate, remap, and invert latitudes of the input data to hourly resolution.
   * Open the output file, convert timesteps to datetime, adjust for 360-day models, and handle missing timesteps.
   * Write the processed data to the output file.
7. If the temporal resolution is daily (``toTemporal == 'd'``):

   * Interpolate, remap, and invert latitudes of the input data to daily resolution.
   * Open the output file, convert timesteps to datetime, adjust for 360-day models, and handle leap years.
   * Write the processed data to the output file.

Returns:
    xarray dataset: processed climate datafile
"""

import os
import xarray as xr
import cftime
import datetime as dt
import numpy as np
import sys
import timeout_decorator
import os
import pandas as pd

config = snakemake.config

# parallel computing sometimes fails - after maximum time, process is stopped and restarted
@timeout_decorator.timeout(2*int(config["maxTime"]))
def writeToNetCdf(file, path):
    """Writes file to path with timeout to catch crashing of dask

    Args:
        file (xarray dataset): file of calculated file
        path (string): path to write file to
    """   
    file.to_netcdf(path)

# convert different time formats


def convert_to_npdatetime(date):
    """Converts different time formats to numpy datetime. Raises an exception if the conversion fails.

    Args:
        date (misc): date to be converted

    Raises:
        Exception: is raised if datetype cannot be converted

    Returns:
        numpy datetime64: converted date
    """    
    if isinstance(date, dt.datetime):
        return date.astype(np.datetime64)
    elif isinstance(date, cftime.DatetimeNoLeap):
        return np.datetime64(date)
    elif isinstance(date, cftime.DatetimeGregorian):
        return np.datetime64(date)
    elif isinstance(date, np.datetime64):
        return date
    else:
        raise Exception(f'Could not convert {type(date)} to numpy datetime!')


data_dir = os.path.join(config['data_dir'], snakemake.wildcards.domain)
model = snakemake.wildcards.model
year = int(snakemake.wildcards.year)
toTemporal = snakemake.wildcards.toTemporal
climate_variable = snakemake.wildcards.climate_variable

if not os.path.exists(os.path.join(data_dir, 'remap', model)):
    os.mkdir(os.path.join(data_dir, 'remap', model))

input_file = snakemake.input.climate_data_file

output_file = snakemake.output[0].replace('.nc', '_2.nc')
if sys.platform == 'win32':
    # Linux naming of files is different than windows naming, because cdo is executed on linux also for windows platforms, renaming is necesarry
    harddrive = input_file[0]
    input_file = input_file.replace(
        harddrive+':', '/mnt/'+harddrive.lower()).replace('\\', '/')
    output_file = output_file.replace(
        harddrive+':', '/mnt/'+harddrive.lower()).replace('\\', '/')

# if aimed temporal resolution is hourly
if toTemporal == '1h':
    # does three things: -inttime: interpolates data to one hour, -remapcbic: remaps data to rectangular grid -invertlat: inverts latitudes, because they are the wrong way round
    if sys.platform == 'win32':
        os.system(
            f'wsl cdo -f nc -setcalendar,365_day -selyear,{year} -inttime,{year}-01-02,00:00,1hour -remapbic,{snakemake.input[0]} -invertlat {input_file} {output_file}')
        output_file = output_file.replace(
            '/mnt/'+harddrive.lower(), harddrive+':')
    else:
        os.system(
            f'cdo -f nc -setcalendar,365_day -selyear,{year} -inttime,{year}-01-02,00:00,1hour -remapbic,{snakemake.input[0]} -invertlat {input_file} {output_file}')

    ds = xr.open_dataset(output_file, chunks={'lon': 20, 'lat': 20})

    # set timesteps to datetime
    timesteps = list()
    for timestep in ds['time'].values:
        timesteps.append(convert_to_npdatetime(timestep))
    ds['time'] = timesteps

    # double 5 days for 360 day models to get to a 365 day calendar
    if snakemake.wildcards.model in config['360day_models']:
        for day in [3, 76, 149, 222, 295]:
            newDay = ds.where(ds['time.dayofyear'] == day, drop=True).groupby('time', squeeze = False).mean()
            ds = xr.concat([newDay, ds], dim='time')
            ds = ds.sortby(["time.month", "time.day"])
        if year % 4 == 0 and year % 400 != 0:
            time = pd.to_datetime(range(0, len(ds['time'])+24), unit='h', origin=f'{year}0102')
            time = time.delete(range(58*24, 59*24))
        else:
            time = pd.to_datetime(range(0, len(ds['time'])), unit='h', origin=f'{year}0102')
        ds['time'] = time

    # add missing timesteps (first day = second day, last day = second last day)
    firstDay = ds.where(ds['time.dayofyear'] == 2, drop=True)
    for ix in range(24):
        firstDay['time'].values[ix] = firstDay['time'].values[ix] - \
            24*60*60*1000000000
    # remove duplicates
    for time in firstDay['time'].values:
        if time in ds["time"].values:
            firstDay = firstDay.drop_sel(time=time)
    ds = xr.concat([firstDay, ds], dim='time')

    lastDayNumber = 363
    # leap year, remove 29.02. for consistency
    if year % 4 == 0 and year % 400 != 0 :
        lastDayNumber = 364
        if not snakemake.wildcards.model in config['360day_models']:
            ds = ds.where(ds['time.dayofyear'] != 60, drop=True)
            ds = ds.groupby("time", squeeze = False).mean()
    
    # sometimes, the last time steps of the year are missing
    # fill them with values from second last day
    lastDay = ds.where(ds['time.dayofyear'] == lastDayNumber, drop=True)
    for ix in range(24):
        lastDay['time'].values[ix] = lastDay['time'].values[ix] + \
            2*24*60*60*1000000000
    # remove duplicates
    for time in lastDay['time'].values:
        if time in ds["time"].values:
            lastDay = lastDay.drop_sel(time=time)
    ds = xr.concat([ds, lastDay], dim='time')

    output_file_new = output_file.replace('_2.nc', '.nc')
    try:
        writeToNetCdf(ds, snakemake.output[0])
    except timeout_decorator.TimeoutError:
        print('Error: time-out, to allocate more time, change maxTime parameter in config')
        os._exit(50)
    ds.close()
    os.remove(output_file)
# if aimed temporal resolution is daily
elif toTemporal == 'd':
    # does three things: -inttime: interpolates data to one day, -remapcbic: remaps data to rectangular grid -invertlat: inverts latitudes, because they are the wrong way round
    if sys.platform == 'win32':
        os.system(
            f'wsl cdo -f nc -setcalendar,365_day -selyear,{year} -inttime,{year}-01-01,12:00,24hour -remapbic,{snakemake.input[0]} -invertlat {input_file} {output_file}')
        output_file = output_file.replace(
            '/mnt/'+harddrive.lower(), harddrive+':')
    else:
        os.system(
            f'cdo -f nc -setcalendar,365_day -selyear,{year} -inttime,{year}-01-01,12:00,24hour -remapbic,{snakemake.input[0]} -invertlat {input_file} {output_file}')

    ds = xr.open_dataset(output_file, chunks={'lon': 20, 'lat': 20})
    # set timesteps to datetime
    timesteps = list()
    for timestep in ds['time'].values:
        timesteps.append(convert_to_npdatetime(timestep))
    ds['time'] = timesteps

    # double 5 days for 360 day models to get to a 365 day calendar
    if snakemake.wildcards.model in config['360day_models']:
        for day in [3, 76, 149, 222, 295]:
            newDay = ds.where(ds['time.dayofyear'] == day, drop=True).groupby('time', squeeze = False).mean()
            ds = xr.concat([newDay, ds], dim='time')
            ds = ds.sortby(["time.month", "time.day"])
        if year % 4 == 0 and year % 400 != 0:
            time = pd.to_datetime(range(0, len(ds['time'])+1), unit='D', origin=f'{year}0101')
            time = time.delete(range(59, 60))
        else:
            time = pd.to_datetime(range(0, len(ds['time'])), unit='D', origin=f'{year}0101')
        ds['time'] = time

    # leap year, remove 29.02. for consistency
    if year % 4 == 0 and year % 400 != 0 and not snakemake.wildcards.model in config['360day_models']:
        print(f"removed 29.02. for {year}")
        ds = ds.where(ds['time.dayofyear'] != 60, drop=True)
        ds = ds.groupby("time", squeeze = False).mean()

    output_file_new = output_file.replace('_2.nc', '.nc')
    try:
        writeToNetCdf(ds, snakemake.output[0])
    except timeout_decorator.TimeoutError:
        print('Error: time-out, to allocate more time, change maxTime parameter in config')
        os._exit(50)
    ds.close()
    os.remove(output_file)
