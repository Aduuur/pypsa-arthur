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

"""1. Suppress the warning for encountering all NaN slices.
2. Load historic model data (``climate_model_hist_q``) and observed data (``climate_observed_hist_q``), reindexing the latter to match the former.
3. Load future model data (``climate_model_future``) and calculate quantiles.
4. ``climate_model_future_q_upper`` contains lower bounds for quantiles, calculate upper bounds for quantiles in ``climate_model_future_q_upper``.
5. Assign quantiles to ``climate_model_future`` data by comparing values with lower and upper bounds.
6. Ensure valid values and avoid invalid values like NaN or 0 in ``climate_model_hist_q``, ``climate_observed_hist_q``, and ``climate_model_future_q``.
7. Calculate the bias adaption factor by summing over all quantiles and multiply it with the model data.
8. Attempt to write the bias adopted dataset to a NetCDF file, handling timeouts. If a timeout occurs, print an error message and exit with code 50.

Returns:
    xarray dataset: bias adopted file
"""

import pandas as pd
import numpy as np
import xarray as xr
import warnings
from dask.diagnostics import ProgressBar
from timeout_function_decorator import timeout
import os

# the quantile calculation always gives a warning for encountering all nan slices
# this filter is to supress this warning
warnings.filterwarnings('ignore', message='All-NaN slice encountered')

# this script does an empirical quantile delta bias adoption (https://doi.org/10.1175/JCLI-D-14-00754.1) based on the climate model data and era5 reanalysis data
config = snakemake.config

climate_variable = snakemake.wildcards.climate_variable

number_of_quantiles = 100

# load historic model data
climate_model_hist_q = xr.open_dataset(snakemake.input.climate_model_hist_q, chunks="auto").reindex(
    quantile=np.arange(0, 1, 1/number_of_quantiles).round(2), method="nearest")

# load historic observed data and reindex like model data
climate_observed_hist_q = xr.open_dataset(snakemake.input.climate_observed_hist_q, chunks="auto")
climate_observed_hist_q = climate_observed_hist_q.reindex_like(climate_model_hist_q, method='nearest', copy=True)

# load future model data
climate_model_future = xr.open_dataset(snakemake.input.climate_model_future, chunks="auto")

# build empirical cdfs
quantiles = np.arange(0, 1, 1/number_of_quantiles).round(2)
# calculate quantiles
climate_model_future_q = climate_model_future.quantile(quantiles, dim='time')

# climate_model_future_q reports lower bound for quantiles, calculate upper bounds
# reload from chunked array to be able to assign values
climate_model_future_q_upper = climate_model_future_q.copy(deep=True)

# rotate coordinates for quantiles so that the lower bound of the next quantile is the upper bound of the active quantile
climate_model_future_q_upper.loc[dict(quantile=0.0)] = np.inf
climate_model_future_q_upper['quantile'] = np.concatenate(
    ([1-1/number_of_quantiles], np.arange(0, 1-1/number_of_quantiles, 1/number_of_quantiles).round(2)))
climate_model_future_q_upper = climate_model_future_q_upper.reindex(
    quantile=np.arange(0, 1, 1/number_of_quantiles).round(2))
climate_model_future_q.loc[dict(quantile=0.0)] = 0

climate_model_future_q = climate_model_future_q.chunk({'lon': int(
    config['geography']['x_size']/config['numberOfChunks']), 'lat': int(config['geography']['y_size']/config['numberOfChunks']), 'quantile': -1})

# assign quantiles to climate_model_future data by comparing the values with lower and upper bounds
quantile_mask = (
    (climate_model_future >= climate_model_future_q) &
    (climate_model_future < climate_model_future_q_upper)
).astype("bool")

# Quantiles below 0 should not occur in the first place, but better safe than sorry
quantile_mask = quantile_mask.where(quantile_mask, 0)


# avoid invalid values like nan or 0
validValues = (climate_model_hist_q != 0 & climate_model_hist_q.notnull(
) & climate_observed_hist_q.notnull() & climate_model_future_q.notnull())
climate_model_hist_q[climate_variable] = climate_model_hist_q[climate_variable].where(validValues[climate_variable], 1)
climate_observed_hist_q[climate_variable] = climate_observed_hist_q[climate_variable].where(validValues[climate_variable], 1)

# improve numeric stability
climate_model_hist_q[climate_variable] = climate_model_hist_q[climate_variable].clip(min=1e-4)


# because quantileAssignment is False/zero for all wrong quantiles, we can multiply the bias adaption with quantileAssignment
# and find the value for the point by summing over all quantiles
factor = (quantile_mask * climate_observed_hist_q / climate_model_hist_q).chunk("auto").sum(dim="quantile", min_count=1).astype("float32")

# parallel computing sometimes fails - after maximum time, process is stopped and restarted
@timeout(float(config["maxTime"])*2)
def writeToNetCdf(file, path):
    """Performs bias adaption and writes file to disk with timeout, because dask sometimes crashes.

    Args:
        ``file`` (xarray dataset): climate file to be bias adopted
        ``path`` (string): path to write file to
    """    
    # actual bias adaption - multiplying factor with the model data
    print("Calculating of bias adopted dataset:")
    with ProgressBar():
        file.to_netcdf(path)

try:
    result = climate_model_future * factor
    result[climate_variable] = result[climate_variable].clip(min=0)
    writeToNetCdf(result, snakemake.output[0])

except TimeoutError:
    print('Error: bias adaption time-out, to allocate more time, change maxTime parameter in config')
    os._exit(50)