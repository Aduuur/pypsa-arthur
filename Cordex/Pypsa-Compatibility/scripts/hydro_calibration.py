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

"""1. Read hydro power plant database from "resources/jrc-hydro-power-plant-database.csv".
2. Filter for hydro dam (HDAM) and run-of-river (HROR) plants.
3. Initialize "average runoff" column in the power plant database.
4. For each climate data file, calculate the average runoff for each power plant location.
5. Adjust the average runoff based on the hydro factor and the number of climate data files.
6. Save the relevant data (lon, lat, average runoff, installed capacity) to a CSV file.

Variables:
    ``config`` (yaml): Snakemake configuration
    ``mrro_df`` (xarray dataset): river runoff dataset

Returns:
    pandas dataframe: list of hydro power plant with installed capacity, average runoff and location"""

import pandas as pd
import xarray as xr

config = snakemake.config

lat_name = 'lat'
lon_name = 'lon'
if 'observed' in snakemake.output[0]:
    lat_name = 'latitude'
    lon_name = 'latitude'

mrro_name = 'mrro'
if 'observed' in snakemake.output[0]:
    mrro_name = config['bias_adaption']['climateVariablesDict'][mrro_name]

# based on the assumption, that the average yearly runoff at a plants location produced
# the installed capacity as electricty output

# read in hydro power plant database
powerplant_database = pd.read_csv(
    "resources/jrc-hydro-power-plant-database.csv", index_col="id")
# only consider hydro dam and ror plants
powerplant_database = powerplant_database.loc[(powerplant_database["type"] == "HDAM")
                                              | (powerplant_database["type"] == "HROR")]
powerplant_database["average runoff"] = 0.0

# read in historic runoff files to calibrate future runoff
for inputFile in snakemake.input.climate_data:
    mrro_df = xr.open_dataset(inputFile, chunks={lon_name: int(
        config["geography"]["x_size"]/config["numberOfChunks"]), lat_name: int(config["geography"]["y_size"]/config["numberOfChunks"])})
    time_name = 'time'
    if 'observed' in snakemake.output[0]:
        for coord in mrro_df.coords:
            if 'time' in coord:
                time_name = coord
    for id, lat, lon in zip(powerplant_database.index, powerplant_database["lat"], powerplant_database["lon"]):
        # calculate average runoff
        if 'observed' in snakemake.output[0]:
            avRunoff = mrro_df.sel(latitude=lat, longitude=lon, method="nearest").mean(
                dim=time_name)[mrro_name].values
        else:
            avRunoff = mrro_df.sel(lat=lat, lon=lon, method="nearest").mean(
                dim=time_name)[mrro_name].values
        if avRunoff > 0:
            powerplant_database.loc[id, "average runoff"] += abs(avRunoff)

# calculate average runoff per year
powerplant_database["average runoff"] = powerplant_database["average runoff"] * \
    config['hydro']['factor'] / len(snakemake.input.climate_data)

powerplant_database[["lon", "lat", "average runoff",
                     "installed_capacity_MW"]].to_csv(snakemake.output[0], sep=";")
