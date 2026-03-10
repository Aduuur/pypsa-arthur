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

"""1. Query ESGF data node based on specified parameters.
2. Create a wget script for downloading climate data for a specific year.
3. Modify the wget script for quietness and write it to a file.
4. Run the wget script to download the file.
5. Rename the downloaded file and handle download failures.
6. Remove the wget script and wget status file.

Variables:
    ``config`` (yaml): Snakemake configuration
    ``model`` (string): name of climate model
    ``rcp`` (string): name of RCP scenario 
    ``climate_variable`` (string): name of climate variable
    ``toTemporal`` (string): desired temporal resoultion of result (``h`` hourly or ``d`` daily)
    ``year`` (int): year to download
    ``timeFrequency`` (string): time frequency to download from esgf 
    ``ensemble``, ``institute``, ``RCMModel``, ``downscalingRealisation`` (string): additional paramerers for download of cordex data
    ``outputPath`` (string): where data is downloaded to
    
Return:
    xarray dataset: downloaded cordex datafile
    """

from pyesgf.search import SearchConnection
import os

config = snakemake.config

model = snakemake.wildcards.model
rcp = snakemake.wildcards.rcp
climate_variable = snakemake.wildcards.climate_variable
toTemporal = snakemake.wildcards.toTemporal
year = int(snakemake.wildcards.year)
domain = snakemake.wildcards.domain

domainDict = {'africa': ['AFR-22'], 'australasia': ['AUS-22'], 'centralamerica': ['CAM-22'], 'centralasia': ['CAS-22'], 'eastasia': ['EAS-22'],
              'europe': ['EUR-11', 'EUR-22'], 'northamerica': ['NAM-22'], 'southamerica': ['SAM-22'], 'southasia': ['WAS-22'], 'southeastasia': ['SEA-22']}

ensemble = config['cordexParameter']['ensemble']
institute = config['cordexParameter']['institute']
RCMModel = config['cordexParameter']['RCMModel']
downscalingRealisation = config['cordexParameter']['downscalingRealisation']

conn = SearchConnection('https://esgf-data.dkrz.de/esg-search', distrib=True)
ctx = conn.new_context(project='CORDEX', facets='project')
print(f"Test search results: {ctx.hit_count}")
# query esgf data node
if toTemporal == 'd':
    timeFrequency = ['day']
else:
    timeFrequency = ['1hr', '3hr']

for timeFrequency in timeFrequency:
    for domainShort in domainDict[domain]:
        ctx = conn.new_context(project=['CORDEX', 'CORDEX-Reklies'], ensemble=ensemble, institute=institute, rcm_name=RCMModel, rcm_version=downscalingRealisation,
                               driving_model=model, experiment=f'rcp{rcp}', time_frequency=timeFrequency, variable=climate_variable, domain=domainShort, facets='*')
        if len(ctx.search()) > 0:
            break
try:
    ds = ctx.search()[0]
except TimeoutError:
    raise Exception(
        'No climate data available with the given parameters. Check model name and the cordex parameter in the config.')

dsString = ds.file_context().get_download_script()

# create wget script for only one year (remove all other years)
delete = False
dsList = dsString.split('\n')
for element in dsList.copy():
    if 'dataset.file.url.chksum_type.chksum' in element:
        delete = not delete
        continue
    if delete:
        # print(element)
        if not f'{timeFrequency}_{year}' in element:
            dsList.remove(element)
        else:
            # save name of file for renaming
            fileName = element.split(" ")[0].replace("'", "")

dsStringNew = '\n'.join(dsList).replace('Script created for 95 file(s)', 'Script created for 1 file(s)')\
    .replace('Script created for 19 file(s)', 'Script created for 1 file(s)').replace('download failed', '')

# change quietness of wget (non quiet mode has to much output but normal quiet mode does not work (no clue why it does not work))
dsStringNew = dsStringNew.replace("${quiet:+-q} ${quiet:--v}", "-nv")

# write wget script to file
script_path = os.path.join(config['data_dir'], domain, 'original', model,
                           f'wget_{climate_variable}_m{model}_rcp{rcp}_{year}_{toTemporal}.sh').replace('\\', '/')

snakePath = os.getcwd()

# check whether script path is absolute, else make it absolute for bash
if not os.path.isabs(script_path):
    script_path = os.path.join(snakePath, script_path).replace('\\', '/')

with open(script_path, 'w') as writer:
    writer.write(dsStringNew)

# run wget script to download the file
downloadPath = os.path.join(config['data_dir'], domain,
                            'original', model).replace('\\', '/')
if not os.path.isabs(downloadPath):
    downloadPath = os.path.join(snakePath, downloadPath).replace('\\', '/')

os.chdir(downloadPath)

os.system(f'bash {script_path} -s')

outputFilePath = os.path.join(
    config['data_dir'], domain, 'original', model, fileName).replace('\\', '/')

if not os.path.isabs(outputFilePath):
    outputFilePath = os.path.join(snakePath, outputFilePath).replace('\\', '/')

outputPath = snakemake.output[0]

if not os.path.isabs(outputPath):
    outputPath = os.path.join(snakePath, outputPath).replace('\\', '/')

try:   
    if os.path.getsize(outputFilePath) > 0:
        print("renaming")
        os.rename(outputFilePath, outputPath)
    else:
        os.remove(outputFilePath)
        print(f'Download failed for {outputPath}')
except TimeoutError:
    print(f'Download failed for {outputPath}')

# remove wget script and wget status file
os.remove(script_path)
os.remove(script_path.replace('wget', '.wget').replace('.sh', '.sh.status'))