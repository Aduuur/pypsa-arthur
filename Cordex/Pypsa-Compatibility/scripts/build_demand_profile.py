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

"""1. **Data Preparation**

   * Filter ``demand_basic`` for the reference year
   * Read ``regression_file``, ``temperature``, ``regions_file``, and ``nuts3``
   * Aggregate demand for nodes with multiple countries
   * Calculate overlay areas and merge data for nodes and countries

2. **Calculate Scaling Factors**

   * Calculate scaling factors for nodes based on GDP and population

3. **Adjust Demand**

   * Adjust historical demand based on temperature and regression coefficients
   * Scale demand using the calculated factors and merge with ``demand_final``

4. **Output**

   * Save the final demand data to a CSV file specified in ``snakemake.output[0]``

Variables:
    ``config`` (yaml): Snakemake configuration
    ``reference_year`` (int): Year from the configuration for demand
    ``climate_data_path`` (string): Path to aggregated climate data
    ``regression_file_path`` (string): Path to regression file
    ``demand_basic_file`` (string): Path to historical demand data
    ``regions_file_path`` (string): Path to regions file based on custom bus map or doSummary of function
    ``demand_basic`` (pandas dataframe): DataFrame containing historical demand data for the reference year
    ``regression_file`` (pandas dataframe): DataFrame from the regression file
    ``temperature`` (pandas dataframe): DataFrame containing daily temperatures from climate data
    ``scaling_factors`` (pandas dataframe): Empty DataFrame for country and scaling factor
    ``regions_file`` (geopandas dataframe): GeoDataFrame of regions
    ``nuts3`` (geopandas dataframe): GeoDataFrame of NUTS3 regions
    ``moreCountriesThanOne`` (list): List of country codes with more than one country
    ``overlay`` (geopandas dataframe): GeoDataFrame of overlay areas between bus and NUTS3 regions
    ``perNode`` (geopandas dataframe): Aggregated data per bus node (population, GDP, country)
    ``perCountry`` (geopandas dataframe): Aggregated data per country (population, GDP, country)
    ``mergeDf`` (geopandas dataframe): Merged DataFrame of perNode and perCountry
    ``demand_final`` (pandas dataframe): DataFrame for final demand data

Returns:
    pandas dataframe: demand data per node in regions file
"""

import pandas as pd
import geopandas as gpd
import numpy as np

config = snakemake.config

domain = snakemake.wildcards.domain

reference_year = config['demand']['reference_year']
climate_data_path = snakemake.input.aggregated_climate_file
regression_file_path = snakemake.input.regression_file
demand_basic_file = 'resources/Entsoe_demand_2015-2017.csv'
if config['use_custom_bus_map']:
    regions_file_path = config['bus_map']
else:
    regions_file_path = f'resources/maps/{domain}.geojson'

europe2Letter = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU',
                 'IE', 'IT', 'LT', 'LU', 'LV', 'ME', 'MK', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SE', 'SI', 'SK']
europe3Letter = ['ALB', 'AUT', 'BIH', 'BEL', 'BGR', 'CHE', 'CZE', 'DEU', 'DNK', 'EST', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC',
                 'HRV', 'HUN', 'IRL', 'ITA', 'LTU', 'LUX', 'LVA', 'MNE', 'MKD', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'SRB',
                 'SWE', 'SVN', 'SVK']

threeToTwo = dict(zip(europe3Letter, europe2Letter))

# historic demand file to be scaled by climate data
demand_basic = pd.read_csv(demand_basic_file, delimiter=';')
demand_basic = demand_basic.loc[:, demand_basic.columns.intersection(
    ['CountryCode', 'DateShort', 'TimeFrom', 'Value_ScaleTo100'])]

# choose only reference year
demand_basic['year'] = pd.to_datetime(
    demand_basic['DateShort'], format='%d.%m.%Y').dt.year
demand_basic = demand_basic.where(
    demand_basic['year'] == reference_year).dropna().drop('year', axis=1)
    
# regression file created with 'demand_regression.py'
regression_file = pd.read_csv(
    regression_file_path, delimiter=';', index_col=0, header=0)

# daily temperatures from climate data
temperature = pd.read_csv(
    climate_data_path, delimiter=';', index_col=0, header=0)

# historic climate data is always hourly, make it daily
if 'observed' in snakemake.output[0]:
    days = [np.ceil((i+1)/24) for i in range(len(temperature.columns))]
    temperature.loc['day'] = days
    temperature = temperature.T.groupby('day').mean().T
    temperature.columns = [f't{int(i):06d}' for i in list(set(days))]

scaling_factors = pd.DataFrame(data=[], index=['country', 'scaling factor'])

# read regions file
regions_file = gpd.read_file(
    regions_file_path).set_index('name').to_crs(epsg=3047)
nuts3 = gpd.read_file(
    'resources/nuts3_shapes.geojson').set_index('index').fillna(0).to_crs(
    epsg=3047)

# if nodes contain more than one country, demand must be aggregated
moreCountriesThanOne = [
    countryCode for countryCode in regions_file['countryCode'] if ';' in countryCode]
for countryCodes in moreCountriesThanOne:
    concatList = []
    for countryCode in countryCodes.split(';'):
        concatList.append(
            demand_basic.loc[demand_basic['CountryCode'] == threeToTwo[countryCode]])
    newDemand = pd.concat(concatList).groupby(
        ['DateShort', 'TimeFrom']).sum().reset_index()
    newDemand['CountryCode'] = countryCodes
    demand_basic = pd.concat([demand_basic, newDemand])

# calculate overlay area of each bus with nuts3 regions
overlay = gpd.overlay(regions_file.reset_index(),
                      nuts3.reset_index(), keep_geom_type=False)
# remove all overlaps where country codes are different for bus and nut3 region
overlay['countryCode'] = overlay['countryCode'].replace(threeToTwo)
overlay = overlay.loc[overlay['countryCode'] == overlay['country']]

# sort nuts3 region to bus with maximum overlap
overlay['area'] = overlay.area
val = overlay.groupby('index')['area']
overlay = overlay.loc[val.idxmax()]

# calculate gdp and population per bus
perNode = overlay.drop(columns='geometry').groupby('name').agg(
    {'pop': 'sum', 'gdp': 'sum', 'country': lambda x: x.value_counts().index[0], 'countryCode': lambda x: x.value_counts().index[0]})

# calculate gdp and population per nut3 region
perCountry = overlay.drop(columns='geometry').groupby('country').agg(
    {'pop': 'sum', 'gdp': 'sum', 'country': lambda x: x.value_counts().index[0], 'countryCode': lambda x: x.value_counts().index[0]})

# calculate combinded score from gdp and pop
perNode['gdppop'] = perNode['pop']*0.4+perNode['gdp']*0.6
perCountry['gdppop_country'] = perCountry['pop']*0.4+perCountry['gdp']*0.6

# drop unnecessary columns
perNode.drop(columns=['pop', 'gdp', 'countryCode'], inplace=True)
perCountry.drop(columns=['pop', 'gdp', 'countryCode'], inplace=True)

# merge per bus and per nut3 region
mergeDf = pd.merge(perNode.reset_index(), perCountry.reset_index(
    drop=True), on='country').set_index('name').drop(columns='country')

# calculate scaling factors
mergeDf['scaling_factor'] = mergeDf['gdppop']/mergeDf['gdppop_country']

# add 1 for all nodes where the former did not work
mergeDfFill = pd.DataFrame(1, index=regions_file.index.difference(
    mergeDf.index), columns=['scaling_factor'])
mergeDf = pd.concat([mergeDf, mergeDfFill])

demand_final = pd.DataFrame()

for node in temperature.index:
    # get corresponding s from regions files
    country = regions_file.loc[node, 'countryCode']
    if country in regression_file.index:
        q = regression_file.loc[country]['quadratic coefficient in MWh/K^2']
        l = regression_file.loc[country]['linear coefficient in MWh/K']
        i = regression_file.loc[country]['intercept in MWh']

        # read historic data, choose right country and year and set right index
        if ';' in country:
            demand = demand_basic.where(demand_basic['CountryCode'] == country).dropna(
            ).drop(['CountryCode'], axis=1)
        else:
            demand = demand_basic.where(demand_basic['CountryCode'] == threeToTwo[country]).dropna(
            ).drop(['CountryCode'], axis=1)
        demand = demand.sort_values(['DateShort', 'TimeFrom']).drop(
            ['DateShort', 'TimeFrom'], axis=1).transpose()
        demand.columns = [f't{i:06d}' for i in range(1, 8761)]
        demand.index = [node]
        for day in range(365):
            daily_tas = temperature.loc[node].iloc[day]
            daily_load_new = q*daily_tas**2+l*daily_tas+i
            daily_load_old = demand.iloc[:, day *
                                         24:(day+1)*24].sum(axis=1).values[0]
            factor = daily_load_new/daily_load_old
            demand.iloc[:, day*24:(day+1)*24] = demand.iloc[:, day*24:(day+1)*24] * \
                factor*mergeDf.loc[node, 'scaling_factor']
        demand_final = pd.concat([demand_final, demand])

demand_final.to_csv(snakemake.output[0], sep=';')
