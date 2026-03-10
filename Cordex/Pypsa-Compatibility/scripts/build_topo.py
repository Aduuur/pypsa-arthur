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

"""
1. Calculates grid specifications for tool cdo.
    Uses either predefined borders or a grid from an external cutout for spatial referencing.
2. Writes the grid specifications to a file.

Variables:
    ``config`` (yaml) Contains configuration settings.
    ``x_min``, ``y_min``, ``x_max``, ``y_max`` (float): Define the minimum and maximum longitude and latitude values for the specified doSummary of function.
    ``x_size``, ``y_size``, ``x_step``, ``y_step``(float): Define the size and step values for the grid.

Returns:
    xarray dataset: specification of grid
"""

import os
import sys
import xarray as xr

config = snakemake.config
domain = snakemake.wildcards.domain

def create_grid(domain, config):
    # taken from cordex domains https://cordex.org/domains/
    if domain == 'southamerica':
        x_min = -86.74
        y_min = -54.6
        x_max = -17.98
        y_max = 18.5
    elif domain == 'centralamerica':
        x_min = -124.26
        y_min = -17.23
        x_max = -30.54
        y_max = 28.79
    elif domain == 'northamerica':
        x_min = -170.74
        y_min = 12.55
        x_max = -23.3
        y_max = 75.88
    elif domain == 'europe':
        x_min = -44.14
        y_min = 25.63
        x_max = 36.3
        y_max = 71.84
    elif domain == 'africa':
        x_min = -24.64
        y_min = -45.76
        x_max = 60.28
        y_max = 42.24
    elif domain == 'southasia':
        x_min = 19.88
        y_min = -15.23
        x_max = 106.43
        y_max = 43.5
    elif domain == 'eastasia':
        x_min = 51.59
        y_min = -0.24
        x_max = 156.08
        y_max = 50.5
    elif domain == 'centralasia':
        x_min = 11.05
        y_min = 18.34
        x_max = 139.13
        y_max = 69.37
    elif domain == 'australasia':
        x_min = 89.25
        y_min = -52.36
        x_max = 206.57
        y_max = 12.21
    elif domain == 'southeastasia':
        x_min = 89.26
        y_min = -14.81
        x_max = 146.96
        y_max = 27.26

    x_size = config['geography']['x_size']
    y_size = config['geography']['y_size']

    return(x_min, y_min, x_max, y_max, x_size, y_size)


if config['desired_ESM'] == 'PyPSA-Eur':
    try:
        ds= xr.open_dataset(config['pypsa']['cutout_template_path'])

        # External grid extents
        x_min = ds['x'].values.min()
        y_min = ds['y'].values.min()
        x_max = ds['x'].values.max()
        y_max = ds['y'].values.max()
        x_size = len(ds['x'])
        y_size = len(ds['y'])

        print(f'External Grid Input: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, x_size={x_size}, y_size={y_size}')

        # CORDEX grid extents
        x_min_cor, y_min_cor, x_max_cor, y_max_cor, x_size_cor, y_size_cor = create_grid(domain, config)

        # Check boundaries and warn
        if x_min < x_min_cor:
           raise ValueError(f"x_min ({x_min}) is smaller than CORDEX x_min ({x_min_cor}) → this causes NaN values in the climate data.")
        if y_min < y_min_cor:
            raise ValueError(f"y_min ({y_min}) is smaller than CORDEX y_min ({y_min_cor}) → this causes NaN values in the climate data.")
        if x_max > x_max_cor:
            raise ValueError(f"x_max ({x_max}) is larger than CORDEX x_max ({x_max_cor}) → this causes NaN values in the climate data.")
        if y_max > y_max_cor:
            raise ValueError(f"y_max ({y_max}) is larger than CORDEX y_max ({y_max_cor}) → this causes NaN values in the climate data.")
    
    except:
        raise ValueError('Download CORDEX: No Valid NetCDF file as input for grid!')      
else:
    x_min, y_min, x_max, y_max, x_size, y_size = create_grid(domain, config)
    print(f'Create grid with default settings: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max} and x_size={x_size}, y_size={y_size}')       




x_step = (x_max-x_min)/(x_size-1)
y_step = (y_max-y_min)/(y_size-1)



with open(f'results/{domain}/grid_spec.txt', 'w') as file:
    file.write(f'gridtype: lonlat\n'
               f'xsize: {x_size}\n'
               f'ysize: {y_size}\n'
               f'xfirst: {x_min}\n'
               f'xinc: {x_step}\n'
               f'yfirst: {y_min}\n'
               f'yinc: {y_step}')


if sys.platform == 'win32':
    os.system(
        f'wsl cdo -f nc -sellonlatbox,{x_min},{x_max},{y_min},{y_max}  -invertlat -topo,results/{domain}/grid_spec.txt {snakemake.output[0]}')
else:
    os.system(
        f'cdo -f nc -sellonlatbox,{x_min},{x_max},{y_min},{y_max}  -invertlat -topo,results/{domain}/grid_spec.txt {snakemake.output[0]}')
