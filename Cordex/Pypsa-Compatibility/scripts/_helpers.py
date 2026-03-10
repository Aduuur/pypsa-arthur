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
A collection of useful functions.
"""

import xarray as xr

def is_nc_grid_inside(file_A, file_B, latnameA='lat', lonnameA='lon',latnameB='lat', lonnameB='lon'):
    """
    Check if the latitude and longitude range of NetCDF file A
    is fully inside the range of NetCDF file B.

    Parameters
    ----------
    file_A : str
        Path to the first NetCDF file (topo file)
    file_B : str
        Path to the second NetCDF file (grid to check)

    Returns
    -------
    bool
        True if all lat/lon of A are within the min/max of B, False otherwise
    """
    # Load datasets
    ds_A = xr.open_dataset(file_A)
    ds_B = xr.open_dataset(file_B)
    
    # Get latitude and longitude arrays
    lat_A = ds_A[latnameA].values
    lon_A = ds_A[lonnameA].values
    lat_B = ds_B[latnameB].values
    lon_B = ds_B[lonnameB].values

    # Check if all points of A are inside B
    lat_inside = lat_A.min() >= lat_B.min() and lat_A.max() <= lat_B.max()
    lon_inside = lon_A.min() >= lon_B.min() and lon_A.max() <= lon_B.max()

    return (lat_inside and lon_inside), {'lat_A_min':lat_A.min(), 'lat_A_max':lat_A.max(),'lat_B_min':lat_B.min(), 'lat_B_max':lat_B.max(),'lon_A_min':lon_A.min(), 'lon_A_max':lon_A.max(),'lon_B_min':lon_B.min(), 'lon_B_max':lon_B.max()}