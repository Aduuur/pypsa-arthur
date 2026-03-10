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

"""1. Iterate through ``fileList``
2. If the file matches specific criteria based on ``model``, ``climate_variable``, ``rcp``, ``year``, and ``toTemporalCordex1`` or ``toTemporalCordex2``, rename the file to ``newPath`` with the specified format.

Variables:
    ``config`` (yaml): snakemake.config
    ``model`` (string): name of climate model
    ``climate_variable`` (string): name of climate variable
    ``rcp`` (string): name of RCP scenario
    ``year`` (int): year
    ``toTemporal`` (string): desired temporal resoultion of result (``h`` hourly or ``d`` daily)
    ``path`` (string): path where cordex data is stored
    ``toTemporalCordex1`` (string): "day" if ``toTemporal`` is "d", otherwise "1hr"
    ``toTemporalCordex2`` (string):  "day" if ``toTemporal`` is "d", otherwise "3hr"
    ``fileList`` (list): list of files in ``path``
    ``newPath`` (string): path for the renamed file

Returns:
    renamed files
"""

import os
import re
import shutil

config = snakemake.config
domain = snakemake.wildcards.domain
model = snakemake.wildcards.model
climate_variable = snakemake.wildcards.climate_variable
rcp = snakemake.wildcards.rcp
year = str(snakemake.wildcards.year)
toTemporal = snakemake.wildcards.toTemporal  # e.g. "1h" or "d"

target_dir = os.path.join(config["data_dir"], domain, "original", model)
os.makedirs(target_dir, exist_ok=True)

cordex_local_dir = config.get("cordex_local_dir")
if cordex_local_dir:
    source_dir = os.path.join(cordex_local_dir, f"cordex_{year}")
else:
    source_dir = target_dir

if not os.path.isdir(source_dir):
    raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

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
    # robust against slightly different naming (e.g. CNRM-CERFACS-CNRM-CM5)
    return all(tok in filename for tok in model.split("-"))

def temporal_matches(filename: str) -> bool:
    """
    CORDEX filenames may contain 1hr/3hr/6hr/... or day.
    If target is daily ("d") -> require "day".
    Otherwise -> accept any "*hr" cadence.

    Special case:
      - mrro is always 6-hourly in this dataset -> accept 6hr even if toTemporal == "d"
        so that the file can be found and renamed for the workflow.
    """
    if toTemporal == "d":
        if climate_variable == "mrro":
            return "_6hr_" in filename
        return "_day_" in filename
    return re.search(r"_(\d+hr)_", filename) is not None  # matches 1hr, 3hr, 6hr, ...

matches = []

for file in fileList:
    if not file.endswith(".nc"):
        continue

    # variable (prefix) + scenario
    if not file.startswith(climate_variable + "_"):
        continue
    if f"rcp{rcp}" not in file:
        continue

    # domain token
    if domain_tokens and not any(token in file for token in domain_tokens):
        continue

    # model
    if not model_matches(file):
        continue

    # year appears inside the timestamp range (e.g. 20500101...-20501231...)
    if year not in file:
        continue

    # accept any hourly cadence for non-daily, accept day for daily
    if not temporal_matches(file):
        continue

    matches.append(file)

if not matches:
    raise FileNotFoundError(
        "No matching CORDEX file found."
        f" Looked in {source_dir} for variable {climate_variable}, model {model},"
        f" rcp{rcp}, domain {domain}, year {year}."
    )

if len(matches) > 1:
    raise FileExistsError(
        "Multiple matching CORDEX files found:"
        f" {matches}. Please narrow the selection."
    )

source_file = os.path.join(source_dir, matches[0])
target_file = os.path.join(
    target_dir, f"{climate_variable}_m{model}_rcp{rcp}_{year}_{toTemporal}.nc"
)

if cordex_local_dir:
    shutil.copy2(source_file, target_file)
else:
    os.rename(source_file, target_file)

