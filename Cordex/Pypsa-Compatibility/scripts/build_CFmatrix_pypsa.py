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
Creates the capacity factor (CF) dataset for PyPSA-Eur.
1. Loads the input CF dataset and the reference cutout.
2. Checks and aligns the dimensions and sort order of the input dataset with the reference cutout.
3. Verifies that the spatial coordinates of the datasets match within a specified tolerance.
4. Transforms the input dataset to ensure it has exactly the same grid as the reference cutout. Renames coords to x and y.
5. Saves the transformed dataset to a NetCDF file.
"""

import xarray as xr
import numpy as np
from timeout_function_decorator import timeout

def match_sort_order(ds_tech, ds_cutout, dims=['y', 'x']):
    '''
    Sorts the dimensions of the CF matrix from cd2es so that they match pypsa cutout.
    Should only affect lattitude (y).
    '''
    for dim in dims:
        if dim in ds_tech.dims and dim in ds_cutout.dims:
            # Check if ds_tech is ascending in this dimension
            tech_asc = ds_tech[dim].values[1] >= ds_tech[dim].values[0]
            # Check if ds_cutout is ascending in this dimension
            cutout_asc = ds_cutout[dim].values[1] >= ds_cutout[dim].values[0]
            
            # If the sort order is different, reverse ds_tech along this dimension
            if tech_asc != cutout_asc:
                ds_tech = ds_tech.sortby(dim, ascending=not tech_asc)
            
    return ds_tech

def check_coords_match(ds1, ds2, dims=['x', 'y'], tol=0.2):
    '''
    Checks that the spatial grid is the same within tollerance
    '''
    for dim in dims:
        if dim not in ds1.coords or dim not in ds2.coords:
            raise ValueError(f"Coordinate '{dim}' missing in one of the datasets.")
        if ds1[dim].size != ds2[dim].size:
            raise ValueError(f"Different lengths for '{dim}': ds1={ds1[dim].size}, ds2={ds2[dim].size}")
        if not np.allclose(ds1[dim].values, ds2[dim].values, atol=tol):
            raise ValueError(f"Coordinate '{dim}' differs by more than {tol}.")

config = snakemake.config
maxTime = int(config["maxTime"])
@timeout(maxTime)
def load_and_transform(cd2es_CF_tech:str):
    """
    Loads and transforms a CF dataset and ensures it matches the grid of a reference cutout.
    """
    # Load datasets with chunking
    ds_tech = xr.open_dataset(cd2es_CF_tech, chunks="auto") #Matrix of CF


    try:
        ds_cutout = xr.open_dataset(snakemake.input.blanc_cutout, chunks="auto") #blanc cutout to get grid. Must be the same as for cutout
    except Exception as e:
        raise ValueError(f"Cannot open blanc cutout: {e}")

    dat_var=list(ds_tech.data_vars)
    if len(dat_var) != 1 :
        raise ValueError(f'More/less than one data variable in File! ({dat_var})')
    elif dat_var[0] in ['pv', 'wind', 'wind_offshore', 'hydro', 'tppCL_pNuclear','tppOT_pNuclear', 'tppCL_pCoal','tppOT_pCoal', 'tppCL_pCCGT','tppOT_pCCGT', 'tppCL_pBiomass','tppOT_pBiomass', 'tppCL_pH2','tppOT_pH2']:
        ds_tech = ds_tech.rename({dat_var[0]:'specific generation'})
    else:
        raise ValueError(f'Variable {dat_var} is unknown!')


    # Prepare temperature data
    ds_tech = ds_tech['specific generation'].rename({'lat': 'y', 'lon': 'x'}) #in pypsa all tech named specific generation

    #Checks and preperation
    ds_tech = match_sort_order(ds_tech,ds_cutout) # sort dimensions first
    check_coords_match(ds_cutout, ds_tech) # then check if grid matches within tollerance

    #make sure it is exectly the same grid. Overwrite grid with pypsa grid
    ds_tech = ds_tech.assign_coords(x=ds_cutout['x']) #assign values of coords
    ds_tech = ds_tech.assign_coords(y=ds_cutout['y']) #assign values of coords
    return ds_tech

@timeout(maxTime)
def safe_ds(ds, path):
    ds.to_netcdf(path)

if __name__=='__main__':
    cd2es_CF_tech = snakemake.input.cd2es_CF_tech

    ds_tech=load_and_transform(cd2es_CF_tech)

    # Save the combined dataset
    print('Write cutout')
    safe_ds(ds_tech,snakemake.output[0])