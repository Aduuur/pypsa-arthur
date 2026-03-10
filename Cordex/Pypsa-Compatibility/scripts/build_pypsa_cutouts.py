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
Builds PyPSA cutout from CORDEX data using the grid of a reference cutout which already contains the height.
1. Sorts the dimensions of input datasets to match the sort order of the reference dataset.
2. Checks and aligns the coordinates of input datasets with the reference dataset within a specified tolerance.
3. Processes individual climate variable datasets, renaming coordinates, sorting dimensions, and aligning coordinates with the reference dataset.
4. Saves to NetCDF file.

"""
import xarray as xr
import numpy as np
from timeout_function_decorator import timeout


def match_sort_order(ds_tech, ds_cutout, dims=['y', 'x']):
    """
    Sorts the dimensions of ds_tech so that they match ds_cutout.
    """
    for dim in dims:
        if dim in ds_tech.dims and dim in ds_cutout.dims:
            # Global check, nicht nur erstes Element
            tech_asc = np.all(np.diff(ds_tech[dim].values) > 0)
            cutout_asc = np.all(np.diff(ds_cutout[dim].values) > 0)

            if tech_asc != cutout_asc:
                ds_tech = ds_tech.sortby(dim, ascending=cutout_asc)
    return ds_tech


def align_coords(ds1, ds2, dims=['x', 'y'], tol=0.01):
    """
    Checks coordinate equality within tolerance.
    If okay, force ds1 to adopt ds2's coordinates (assign_coords).
    Otherwise raises ValueError.
    """
    for dim in dims:
        if dim not in ds1.coords or dim not in ds2.coords:
            raise ValueError(f"Coordinate '{dim}' missing in one of the datasets.")
        if ds1[dim].size != ds2[dim].size:
            raise ValueError(f"Different lengths for '{dim}': ds1={ds1[dim].size}, ds2={ds2[dim].size}")

        diff = np.abs(ds1[dim].values - ds2[dim].values)
        if not np.all(diff <= tol):
            raise ValueError(
                f"Coordinate '{dim}' differs by more than {tol}. "
                f"Max diff: {diff.max()}"
            )

    # Wenn alles passt: harte Angleichung der Koordinaten
    return ds1.assign_coords({dim: ds2[dim] for dim in dims})


config = snakemake.config
maxTime = int(config["maxTime"])
@timeout(maxTime)
def load_and_prepare_data(ds_input, variable_name):
    """
    Load and prepare a single dataset.

    Parameters:
    ds_input: Input dataset path.
    variable_name: Name of the variable to process.

    Returns:
    A processed xarray Dataset.
    """
    ds = xr.open_dataset(ds_input, chunks="auto")
    var = ds[variable_name].rename({'lat': 'y', 'lon': 'x'})
    var = match_sort_order(var, ds_global)  # sort first
    var = align_coords(var, ds_global)  # then align
    if "height" in var.coords:  # Remove the height coordinate if it exists
        var = var.reset_coords("height", drop=True)
        print(f'Removed height from {variable_name}')
    return var

@timeout(maxTime)
def safe_ds(ds, path):
    ds.to_netcdf(path)

if __name__=='__main__':
    try:
        ds_global = xr.open_dataset(snakemake.input.blanc_cutout)
    except Exception as e:
        raise ValueError(f"Cannot open blanc cutout: {e}")

    mapping= {'tas': 'temperature', 'tso' : 'soil temperature', 'mrro' : 'runoff'}
    for key, climvar_path in snakemake.input.items():
        
        if key.startswith('climvar'):
            climvar_name=key.split('_')[-1]
            print(f'Add {mapping[climvar_name]} to cutout')
            ds = load_and_prepare_data(climvar_path, climvar_name)
            ds_global = ds_global.assign({mapping[climvar_name]:ds})

    # Save the combined dataset
    print('Write cutout')
    safe_ds(ds_global,snakemake.output[0])