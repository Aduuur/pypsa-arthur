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

"""1. Retrieves configuration data from ``snakemake.config``.
2. Determines the geographic boundaries based on the ``doSummary of function`` wildcard.
3. Determines the climate variable based on the ``climate_variable_ERA5`` wildcard.
4. Calls the appropriate retrieval function (``retrieveHourly`` or ``retrieveDaily``) based on the climate variable.

Variables:
    ``c``: ``cdsapi.Client`` object
    ``climate_variable`` (string): Name of climate variable
    ``year`` (int): Year of the data to download
    ``x_min``, ``x_max``, ``y_min``, ``y_max`` (float): Geographic boundaries
    ``climate_data_file`` (string): path to output file for the climate data.

Returns:
    xarray dataset: downloaded era5 datafile
"""

import cdsapi
import xarray as xr

def retrieveHourly(c, variable, year, x_min, x_max, y_min, y_max, climate_data_file):
    """Downloads hourly era5 files

    Args:
        c (cdsapi.Client): cdsapi Client
        variable (string): name of variable
        year (int): year of the data to download
        x_min, x_max, y_min, y_max (float): Geographic boundaries
        climate_data_file (string): path to output
    """    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': variable,
            'year': year,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                y_max, x_min, y_min, x_max,
            ],
            'format': 'netcdf',
        },
        climate_data_file)


def retrieveDaily(c, variable, year, x_min, x_max, y_min, y_max, climate_data_file):
    """Downloads daily era5 files

    Args:
        c (cdsapi.Client): cdsapi Client
        variable (string): name of variable
        year (int): year of the data to download
        x_min, x_max, y_min, y_max (float): Geographic boundaries
        climate_data_file (string): path to output
    """    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': variable,
            'year': year,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00'
            ],
            'area': [
                y_max, x_min, y_min, x_max,
            ],
            'format': 'netcdf',
        },
        climate_data_file)


config = snakemake.config

if config["loginDataAsParams"]:
    c = cdsapi.Client(url=snakemake.params.url, 
                      key=snakemake.params.key)
else:
    c = cdsapi.Client()

config = snakemake.config
domain = snakemake.wildcards.domain
climate_variable = snakemake.wildcards.climate_variable_era5
year = snakemake.wildcards.year
climate_data_file = snakemake.output.climate_data_file

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
    
    print(f'Used grid with default settings: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}')
    
    return(x_min, y_min, x_max, y_max)

if config['desired_ESM'] == 'PyPSA-Eur':
    try:
        ds= xr.open_dataset(config['pypsa']['cutout_template_path']) 
        x_min = max(min(ds['x'].values) - 0.26, -180) #add one point (>0.25°) in each direction so no NaN values when interpolation later on
        x_max = min(max(ds['x'].values) + 0.26, 180)
        y_min = max(min(ds['y'].values) - 0.26, -90)
        y_max = min(max(ds['y'].values) + 0.26, 90)

        print(f'External Grid as Input with x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}')
    except:
       raise ValueError('Download era5: No Valid NetCDF file as input for grid!')
else:
    x_min, y_min, x_max, y_max = create_grid(domain, config)     


if climate_variable == "ro":
    variable = "runoff"
    retrieveDaily(c, variable, year, x_min, x_max,
                  y_min, y_max, climate_data_file)
else:
    variableDict = {'v10': '10m_v_component_of_wind', 'u10': '10m_u_component_of_wind',
                    't2m': '2m_temperature', 'ssr': 'surface_net_solar_radiation', 'stl4':'soil_temperature_level_4'}
    variable = variableDict[climate_variable]
    retrieveHourly(c, variable, year, x_min, x_max,
                   y_min, y_max, climate_data_file)
