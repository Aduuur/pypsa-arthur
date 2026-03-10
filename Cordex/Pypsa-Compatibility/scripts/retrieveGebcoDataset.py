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

"""1. Downloads gebco file for height data.
2. Unzips and saves it.

Returns:
    xarray dataset: gebco dataset
"""

from urllib.request import urlretrieve
import zipfile
import os

# download GEBCO file for height data (necessary for offshore wind depth restricition)
url = ('https://www.bodc.ac.uk/data/open_download/gebco/GEBCO_30SEC/zip/')
urlretrieve(url, snakemake.output[0].replace('.nc', '.zip'))

# unzip gebco
with zipfile.ZipFile(snakemake.output[0].replace('.nc', '.zip'), 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(snakemake.output[0]))

# remove unnecessary files
os.remove(snakemake.output[0].replace('.nc', '.zip'))
os.remove(os.path.join(os.path.dirname(snakemake.output[0]), 'GEBCO_2014.html'))
os.remove(os.path.join(os.path.dirname(snakemake.output[0]), 'Terms_of_Use_of_The_GEBCO_Grid_and_derived_information_products.html'))