# cd2es - covert cordex climate data to energy system input data

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Script for remapping climate data to a target grid using bilinear interpolation.

1. Checks if the target grid is within the climate data grid to avoid NaN values.
2. Remaps the climate data to the target grid using bilinear interpolation.
3. Validates the remapped data for any NaN or infinite values.
4. Renames the latitude and longitude variables to lat and lon.
5. Saves the remapped data to the output file.

"""
import sys
import os
import xarray as xr
import numpy as np

from scripts._helpers import is_nc_grid_inside

input_file = snakemake.input.climate_data_file
target_grid = snakemake.input.topoFile
output_file = snakemake.output[0]
output_file_shortname=output_file.replace(".nc", "_lat_lon_short.nc")

#check if grid is within era5 data
inside, gval = is_nc_grid_inside(target_grid, input_file, latnameB='latitude', lonnameB='longitude')
if not inside:
    raise ValueError('Grid of climate data file is not within grid of topo! Would lead to NaN values if interpolating.\n'
    f'Grid A: lat from {gval['lat_A_min']} to {gval['lat_A_max']} and lon from {gval['lon_A_min']} to {gval['lon_A_max']} for {target_grid}\n'
    f'Grid B: lat from {gval['lat_B_min']} to {gval['lat_B_max']} and lon from {gval['lon_B_min']} to {gval['lon_B_max']} for {input_file}'
    )
else:
    print('Check completed. Grid of topo is inide of era5 climate data file')


print(f'Remaping {input_file}')
if sys.platform == 'win32':
    # Linux naming of files is different than windows naming, because cdo is executed on linux also for windows platforms, renaming is necesarry
    harddrive = input_file[0]
    input_file = input_file.replace(
        harddrive+':', '/mnt/'+harddrive.lower()).replace('\\', '/')
    output_file = output_file.replace(
        harddrive+':', '/mnt/'+harddrive.lower()).replace('\\', '/')

if sys.platform == 'win32':
    os.system(
        f'wsl cdo -f nc  -remapbil,{traget_grid} {input_file} {output_file}')
    output_file = output_file.replace(
        '/mnt/'+harddrive.lower(), harddrive+':')
else:
    os.system(
        f'cdo -f nc  -remapbil,{target_grid} {input_file} {output_file_shortname}')

ds_shortname=xr.open_dataset(output_file_shortname)

#chek if any Nan or infinit data
if bool(np.any(np.isnan(ds_shortname)).to_dataarray()) or bool(np.any(np.isinf(ds_shortname)).to_dataarray()):
    raise ValueError(f'Error: Infinit or NaN values in data file {output_file_shortname}')

ds_longname=ds_shortname.rename({"lat":"latitude", "lon":"longitude"}) #change from short to long names in remaped data

if os.path.exists(output_file):
    os.remove(output_file)
ds_longname.to_netcdf(output_file, mode="w")

try:
    os.remove(output_file_shortname)
    print('Successfully removed temporary data!')
except:
    print(f'Not able to remove temporary data {output_file_shortname}')