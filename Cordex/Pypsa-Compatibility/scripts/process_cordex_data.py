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
5. Set the ``input_file`` and ``output_file_temp`` paths, adjusting for Windows naming if necessary.
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
from timeout_function_decorator import timeout
import os
import pandas as pd
import re

from scripts._helpers import is_nc_grid_inside

# =============================================================================
# process_cordex_data.py  (Snakemake script)
# - Modified: write ALL intermediate + final NetCDFs locally (tmpdir) first,
#   then copy atomically to /mnt/endata (CIFS) to avoid hanging writes.
# =============================================================================

import os
import sys
import re
import shutil
import tempfile
import subprocess
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
import cftime

# NOTE: assumes these exist in your environment / original script
# from your_module import is_nc_grid_inside
# from your_timeout_module import timeout

config = snakemake.config

data_dir = os.path.join(config["data_dir"], snakemake.wildcards.domain)
model = snakemake.wildcards.model
year = int(snakemake.wildcards.year)
toTemporal = snakemake.wildcards.toTemporal
climate_variable = snakemake.wildcards.climate_variable

# Ensure remap dir exists
remap_dir = os.path.join(data_dir, "remap", model)
os.makedirs(remap_dir, exist_ok=True)

topo_file = snakemake.input[0]
input_file = snakemake.input.climate_data_file
output_file = snakemake.output[0]

# =============================================================================
# Local tmp workspace (WRITE LOCALLY FIRST)
# =============================================================================
tmp_root = getattr(snakemake, "resources", {}).get("tmpdir", None)
if not tmp_root:
    tmp_root = os.environ.get("TMPDIR", "/tmp")

job_tag = f"{snakemake.rule}_{getattr(snakemake, 'jobid', 'job')}_{model}_{climate_variable}_{year}_{toTemporal}"
local_workdir = tempfile.mkdtemp(prefix=f"smk_{job_tag}_", dir=tmp_root)

# Local intermediate files (always local, never on /mnt/endata)
local_temp2 = os.path.join(local_workdir, "temp_2.nc")
local_temp3 = os.path.join(local_workdir, "temp_3.nc")

# Local final output (write here first, then copy to output_file)
local_final = os.path.join(local_workdir, os.path.basename(output_file))

# =============================================================================
# Helpers
# =============================================================================
def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def copy_atomic_to_dest(src_path: str, dest_path: str) -> None:
    """
    Copy src_path to dest_path on destination filesystem atomically:
    - copy to dest_path + ".tmp"
    - fsync
    - os.replace(tmp, dest) (atomic on destination FS)
    """
    _ensure_parent_dir(dest_path)
    tmp_dest = dest_path + ".tmp"

    # Copy bytes
    with open(src_path, "rb") as fsrc, open(tmp_dest, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, length=1024 * 1024 * 16)  # 16MB chunks
        fdst.flush()
        os.fsync(fdst.fileno())

    # Atomic replace on destination filesystem
    os.replace(tmp_dest, dest_path)


def run_cmd(cmd: list[str], timeout_s: int | None = None) -> None:
    """
    Run command, raise on failure.
    """
    subprocess.run(cmd, check=True, timeout=timeout_s)


# parallel computing sometimes fails - after maximum time, process is stopped and restarted
@timeout(2 * int(config["maxTime"]))
def writeToNetCdf(file, path):
    """Writes file to path with timeout to catch crashing of dask"""
    file.to_netcdf(path)


# convert different time formats
def convert_to_npdatetime(date):
    """Converts different time formats to numpy datetime. Raises an exception if the conversion fails."""
    if isinstance(date, dt.datetime):
        return date.astype(np.datetime64)
    elif isinstance(date, cftime.DatetimeNoLeap):
        return np.datetime64(date)
    elif isinstance(date, cftime.DatetimeGregorian):
        return np.datetime64(date)
    elif isinstance(date, np.datetime64):
        return date
    else:
        raise Exception(f"Could not convert {type(date)} to numpy datetime!")


def _file_covers_year(path: str, target_year: int) -> bool:
    """
    Try to infer covered year range from CORDEX filename tail:
      ..._YYYYMMDDHHMM-YYYYMMDDHHMM.nc
    Return True if target_year is within [start_year, end_year].
    If pattern not found, fall back to checking if 'target_year' string is contained.
    """
    base = os.path.basename(path)
    m = re.search(r"_(\d{4})\d{8,10}-(\d{4})\d{8,10}\.nc$", base)
    if m:
        y0 = int(m.group(1))
        y1 = int(m.group(2))
        return (y0 <= target_year <= y1)
    return str(target_year) in base


def _find_mrro_file_for_year(target_year: int) -> str:
    """
    Minimal fallback: for mrro (always 6hr locally), search in cordex_local_dir/cordex_<year>
    for a single matching file.
    """
    cordex_local_dir = config.get("cordex_local_dir")
    if not cordex_local_dir:
        raise FileNotFoundError("cordex_local_dir not set in config, cannot fallback for mrro.")

    domain = snakemake.wildcards.domain
    rcp = str(snakemake.wildcards.rcp)

    source_dir = os.path.join(cordex_local_dir, f"cordex_{target_year}")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"mrro fallback source directory does not exist: {source_dir}")

    fileList = os.listdir(source_dir)

    domainDict = {
        "africa": ["AFR-22"],
        "australasia": ["AUS-22"],
        "centralamerica": ["CAM-22"],
        "centralasia": ["CAS-22"],
        "eastasia": ["EAS-22"],
        "europe": ["EUR-11", "EUR-22"],
        "northamerica": ["NAM-22"],
        "southamerica": ["SAM-22"],
        "southasia": ["WAS-22"],
        "southeastasia": ["SEA-22"],
    }
    domain_tokens = domainDict.get(domain, [])

    def model_matches(filename: str) -> bool:
        return all(tok in filename for tok in model.split("-"))

    matches = []
    for file in fileList:
        if not file.endswith(".nc"):
            continue
        if not file.startswith("mrro_"):
            continue
        if f"rcp{rcp}" not in file:
            continue
        if domain_tokens and not any(token in file for token in domain_tokens):
            continue
        if not model_matches(file):
            continue
        if "_6hr_" not in file:
            continue
        if str(target_year) not in file:
            continue
        matches.append(os.path.join(source_dir, file))

    if not matches:
        raise FileNotFoundError(
            f"mrro fallback could not find a matching 6hr file in {source_dir} for year {target_year}."
        )
    if len(matches) > 1:
        raise FileExistsError(
            f"mrro fallback found multiple matches in {source_dir} for year {target_year}: {matches}"
        )

    return matches[0]


# =============================================================================
# Pre-check: grid coverage
# =============================================================================
inside, gval = is_nc_grid_inside(topo_file, input_file)
if not inside:
    target_grid = topo_file  # keep original variable reference used in your error message
    raise ValueError(
        "Grid of climate data file is not within grid of topo! Would lead to NaN values if interpolating.\n"
        f'Grid A: lat from {gval["lat_A_min"]} to {gval["lat_A_max"]} and lon from {gval["lon_A_min"]} to {gval["lon_A_max"]} for {target_grid}\n'
        f'Grid B: lat from {gval["lat_B_min"]} to {gval["lat_B_max"]} and lon from {gval["lon_B_min"]} to {gval["lon_B_max"]} for {input_file}'
    )
else:
    print("Check completed. Grid of topo is inide of cordex climate data file")


# =============================================================================
# Windows path adjustment (kept from original)
# =============================================================================
if sys.platform == "win32":
    harddrive = input_file[0]
    input_file = input_file.replace(harddrive + ":", "/mnt/" + harddrive.lower()).replace("\\", "/")
    # local temps remain linux paths under WSL; final copy not used in win32 branch here.


# =============================================================================
# Main processing
# =============================================================================
try:
    if toTemporal == "1h":
        # NOTE: use local_temp3 -> local_temp2, never write intermediates to /mnt/endata
        if sys.platform == "win32":
            # WSL call; still writes to output_file_temp (kept behavior)
            # If you need local write on Windows too, we can adapt separately.
            output_file_temp = output_file.replace(".nc", "_2.nc")
            os.system(
                f"wsl cdo -f nc -selyear,{year} -inttime,{year}-01-02,00:00,1hour "
                f"-remapbic,{topo_file} -invertlat {input_file} {output_file_temp}"
            )
        else:
            # CDO timeout: align roughly with your maxTime (seconds). If maxTime in minutes, adjust accordingly.
            cdo_timeout = 2 * int(config["maxTime"])

            run_cmd(
                [
                    "cdo", "-f", "nc",
                    f"-selyear,{year}",
                    f"-inttime,{year}-01-02,00:00,1hour",
                    f"-remapbic,{topo_file}",
                    "-invertlat",
                    input_file,
                    local_temp3,
                ],
                timeout_s=cdo_timeout,
            )
            run_cmd(["cdo", "fillmiss", local_temp3, local_temp2], timeout_s=cdo_timeout)

            # remove local_temp3 now that local_temp2 exists
            try:
                os.remove(local_temp3)
            except FileNotFoundError:
                pass

            # Now open LOCAL temp and write LOCAL final, then copy atomically to output_file on /mnt/endata
            with xr.open_dataset(local_temp2, chunks={"lon": 20, "lat": 20}) as ds:
                # set timesteps to datetime
                timesteps = [convert_to_npdatetime(t) for t in ds["time"].values]
                ds["time"] = timesteps

                # double 5 days for 360 day models to get to a 365 day calendar
                if snakemake.wildcards.model in config["360day_models"]:
                    for day in [3, 76, 149, 222, 295]:
                        newDay = (
                            ds.where(ds["time.dayofyear"] == day, drop=True)
                            .groupby("time", squeeze=False)
                            .mean()
                        )
                        ds = xr.concat([newDay, ds], dim="time")
                        ds = ds.sortby(["time.month", "time.day"])

                    if year % 4 == 0 and year % 400 != 0:
                        time = pd.to_datetime(range(0, len(ds["time"]) + 24), unit="h", origin=f"{year}0102")
                        time = time.delete(range(58 * 24, 59 * 24))
                    else:
                        time = pd.to_datetime(range(0, len(ds["time"])), unit="h", origin=f"{year}0102")
                    ds["time"] = time

                # add missing timesteps (first day = second day, last day = second last day)
                firstDay = ds.where(ds["time.dayofyear"] == 2, drop=True)
                for ix in range(24):
                    firstDay["time"].values[ix] = firstDay["time"].values[ix] - 24 * 60 * 60 * 1_000_000_000

                # remove duplicates
                for t in firstDay["time"].values:
                    if t in ds["time"].values:
                        firstDay = firstDay.drop_sel(time=t)

                ds = xr.concat([firstDay, ds], dim="time")

                lastDayNumber = 363
                # leap year, remove 29.02. for consistency
                if (
                    year % 4 == 0
                    and year % 400 != 0
                    and len(ds["time"]) > 8760
                    and snakemake.wildcards.model not in config["360day_models"]
                ):
                    lastDayNumber = 364
                    ds = ds.where(ds["time.dayofyear"] != 60, drop=True)

                lastDay = ds.where(ds["time.dayofyear"] == lastDayNumber, drop=True)
                for ix in range(24):
                    lastDay["time"].values[ix] = lastDay["time"].values[ix] + 2 * 24 * 60 * 60 * 1_000_000_000

                # remove duplicates
                for t in lastDay["time"].values:
                    if t in ds["time"].values:
                        lastDay = lastDay.drop_sel(time=t)

                ds = xr.concat([ds, lastDay], dim="time")

                try:
                    # write locally first
                    writeToNetCdf(ds, local_final)
                except TimeoutError:
                    print("Error: time-out, to allocate more time, change maxTime parameter in config")
                    os._exit(50)

            # Copy local_final -> output_file atomically on destination FS
            copy_atomic_to_dest(local_final, output_file)

            # cleanup local intermediate
            try:
                os.remove(local_temp2)
            except FileNotFoundError:
                pass


    elif toTemporal == "d":
        # --- minimal fallback for mrro + yearly local files ---
        if climate_variable == "mrro":
            try:
                if not _file_covers_year(input_file, year):
                    input_file = _find_mrro_file_for_year(year)
                    print(f"mrro fallback: using per-year input file {input_file}")
            except Exception as e:
                print(f"mrro fallback warning: {e}")

        if sys.platform == "win32":
            output_file_temp = output_file.replace(".nc", "_2.nc")
            os.system(
                f"wsl cdo -f nc -selyear,{year} -inttime,{year}-01-01,12:00,24hour "
                f"-remapbic,{topo_file} -invertlat {input_file} {output_file_temp}"
            )
        else:
            cdo_timeout = 2 * int(config["maxTime"])

            run_cmd(
                [
                    "cdo", "-f", "nc",
                    f"-selyear,{year}",
                    f"-inttime,{year}-01-01,12:00,24hour",
                    f"-remapbic,{topo_file}",
                    "-invertlat",
                    input_file,
                    local_temp3,
                ],
                timeout_s=cdo_timeout,
            )
            run_cmd(["cdo", "fillmiss", local_temp3, local_temp2], timeout_s=cdo_timeout)

            try:
                os.remove(local_temp3)
            except FileNotFoundError:
                pass

            with xr.open_dataset(local_temp2, chunks={"lon": 20, "lat": 20}) as ds:
                timesteps = [convert_to_npdatetime(t) for t in ds["time"].values]
                ds["time"] = timesteps

                if snakemake.wildcards.model in config["360day_models"]:
                    for day in [3, 76, 149, 222, 295]:
                        newDay = (
                            ds.where(ds["time.dayofyear"] == day, drop=True)
                            .groupby("time", squeeze=False)
                            .mean()
                        )
                        ds = xr.concat([newDay, ds], dim="time")
                        ds = ds.sortby(["time.month", "time.day"])

                    if year % 4 == 0 and year % 400 != 0:
                        time = pd.to_datetime(range(0, len(ds["time"]) + 1), unit="D", origin=f"{year}0101")
                        time = time.delete(range(59, 60))
                    else:
                        time = pd.to_datetime(range(0, len(ds["time"])), unit="D", origin=f"{year}0101")
                    ds["time"] = time

                # leap year, remove 29.02. for consistency
                if year % 4 == 0 and year % 400 != 0 and snakemake.wildcards.model not in config["360day_models"]:
                    print(f"removed 29.02. for {year}")
                    ds = ds.where(ds["time.dayofyear"] != 60, drop=True)

                if climate_variable == "mrro":
                    ds[climate_variable] = ds[climate_variable].fillna(0.0)

                # check NaN/Inf only in the actual climate variable
                da = ds[climate_variable]
                has_nan = bool(da.isnull().any().compute().item())
                has_inf = bool(np.isinf(da).any().compute().item())
                if has_nan or has_inf:
                    raise ValueError(f"Error: Infinit or NaN values in {climate_variable} data!")

                try:
                    # write locally first
                    writeToNetCdf(ds, local_final)
                except TimeoutError:
                    print("Error: time-out, to allocate more time, change maxTime parameter in config")
                    os._exit(50)

            # Copy local_final -> output_file atomically on destination FS
            copy_atomic_to_dest(local_final, output_file)

            try:
                os.remove(local_temp2)
            except FileNotFoundError:
                pass

    else:
        raise ValueError(f"Unsupported toTemporal={toTemporal}")

finally:
    # Always cleanup local workspace directory (best effort)
    try:
        shutil.rmtree(local_workdir, ignore_errors=True)
    except Exception:
        pass
