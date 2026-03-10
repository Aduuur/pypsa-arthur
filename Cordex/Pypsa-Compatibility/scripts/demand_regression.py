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

"""1. Generate date series.
2. Read the bus map to map node names to country codes.
3. Process historic temperature data and group by country.
4. Load historic demand data.
5. Aggregate demand for nodes with multiple countries.
6. Perform regression analysis between historic demand and temperatures for each country and save the results.
7. Save regression parameters to an output file.

Variables:
    ``yearStart`` (int): Start year for demand data.
    ``yearEnd`` (int): End year for demand data.
    ``years`` (list): List of years within the specified range.
    ``climate_model`` (string): name of climate model.
    ``rcp`` (string): name of RCP scenario.
    ``date_series`` (list): List of formatted date strings.
    ``regions_file_path`` (string): Path to the regions file.
    ``regions_file`` (geopandas dataframe): GeoDataFrame containing region information.
    ``temperature_historic`` (pandas dataframe): DataFrame storing historic temperature data.
    ``demand`` (pandas dataframe): DataFrame containing demand data.
    ``moreCountriesThanOne`` (list): List of country codes with multiple countries in a node.
    ``regression_parameters`` (pandas dataframe): DataFrame to store regression parameters.
    
Returns:
    pandas dataframe: regression parameter
"""

import pandas as pd
import numpy as np
from datetime import timedelta, date
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import geopandas as gpd

config = snakemake.config


def fitfunc(x, a, b, c):
    """quadratic fitting function

    Args:
        x (float): input data
        a, b, c (float): fitting parameters
    Returns:
        float: function values
    """    
    return a * x**2 + b * x + c

# taken from the internet


def polyfit(x, y, coeffs):
    """fitting for quadratic function

    Args:
        x (array): x values
        y (array): y values
        coeffs (array): coefficients

    Returns:
        array: coefficients of fit function
    """    
    results = {}
    p = np.poly1d(coeffs)
    # calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results = ssreg / sstot

    return results

# taken from the internet


def daterange(date1, date2):
    """calculate date series between two dates

    Args:
        date1 (date): start date
        date2 (date): end date

    Yields:
        list: all dates between start and end date
    """    
    for n in range(int((date2 - date1).days)+1):
        yield date1 + timedelta(n)


europe2Letter = ['AL', 'AT', 'BA', 'BE', 'BY', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU',
                 'IE', 'IT', 'LT', 'LU', 'LV', 'ME', 'MK', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SE', 'SI', 'SK']
europe3Letter = ['ALB', 'AUT', 'BIH', 'BEL', 'BLR', 'BGR', 'CHE', 'CZE', 'DEU', 'DNK', 'EST', 'ESP', 'FIN', 'FRA', 'GBR', 'GRC',
                 'HRV', 'HUN', 'IRL', 'ITA', 'LTU', 'LUX', 'LVA', 'MNE', 'MKD', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'SRB',
                 'SWE', 'SVN', 'SVK']
threeToTwo = dict(zip(europe3Letter, europe2Letter))

yearStart = config['demand']['yearStart']
yearEnd = config['demand']['yearEnd']
years = list(range(yearStart, yearEnd+1))

domain = snakemake.wildcards.domain

if 'observed' in snakemake.output[0]:
    name = 'observed'
else:
    climate_model = snakemake.wildcards.model
    rcp = snakemake.wildcards.rcp
    name = f'm{climate_model}_rcp{rcp}'

# generate date series
date_series = list()
# taken from the internet
start_dt = date(yearStart, 1, 1)
end_dt = date(yearEnd, 12, 31)
for dt in daterange(start_dt, end_dt):
    if not (dt.month == 2 and dt.day == 29):
        date_series.append(dt.strftime('%Y-%m-%d'))

# read in bus map to get mapping from node names to countryCodes
if config['use_custom_bus_map']:
    regions_file_path = config['bus_map']
else:
    domain = snakemake.wildcards.domain
    regions_file_path = f'resources/maps/{domain}.geojson'

regions_file = gpd.read_file(regions_file_path).set_index('name')

temperature_historic = pd.DataFrame()
for i in range(len(years)):
    file = pd.read_csv(snakemake.input[i],
                       delimiter=';', index_col=0, header=0)
    file.index = [regions_file.loc[name, 'countryCode'] for name in file.index]
    file.index.name = 'country'
    # historic climate data is always hourly, make it daily
    if 'observed' in snakemake.output[0]:
        days = [np.ceil((i+1)/24) for i in range(len(file.columns))]
        file.loc['day'] = days
        file = file.T.groupby('day').mean().T
        file.columns = [f't{int(i):06d}' for i in list(set(days))]
        print(file)
        if 't000366' in file.columns:
            file = file.drop(columns = ['t000366'])
    temperature_historic = pd.concat([temperature_historic, file], axis=1)

temperature_historic.columns = date_series

# group by country, if there is more than one node per country
temperature_historic = temperature_historic.groupby('country').mean()

demand = pd.read_csv('resources/Entsoe_demand_2015-2017.csv', delimiter=';')
demand = demand.set_index(['DateShort']).loc[:, demand.columns.intersection(
    ['CountryCode', 'Value_ScaleTo100'])]
demand2 = pd.read_csv('resources/Entsoe_demand_2018-2019.csv', delimiter=';')
demand2 = demand2.set_index(['DateShort']).loc[:, demand2.columns.intersection(
    ['CountryCode', 'Value_ScaleTo100'])]
demand = pd.concat([demand, demand2])

# scale demand before applying cd2es to match projections for future
if config['demand']['scaleDemand']:
    # calculate total Balkan load, because Balkan demand is grouped in Pietzcker et all
    concatList = []
    for country in ['AL', 'BA', 'ME', 'MK', 'RS']:
        concatList.append(
            demand.loc[demand['CountryCode'] == country])
    balkanDemand = pd.concat(concatList).groupby(
        ['DateShort']).sum().reset_index()
    totalLoadBalkan = balkanDemand['Value_ScaleTo100'].sum()

    scaleDemandCountries = [threeToTwo[country] if not country ==
                            "Balkan" else "Balkan" for country in config['demand']['scaleDemandCountries']]
    newDemand = pd.DataFrame(
        config['demand']['scaleDemandNewLoad'], index=scaleDemandCountries, columns=["newDemand"])

    balkanFactor = (newDemand.loc["Balkan"]*10**6*(yearEnd-yearStart+1))/totalLoadBalkan

    factor = ((newDemand["newDemand"]*10**6*(yearEnd-yearStart+1)) /
              demand.groupby("CountryCode")["Value_ScaleTo100"].sum()).fillna(1)
    for country in ["AL", "BA", "ME", "MK", "RS"]:
        factor.loc[country] = balkanFactor["newDemand"]
    factor = factor.to_frame(name="factor")
    factor.index.name = "CountryCode"

    demand.set_index(
        ['CountryCode'], append=True, inplace=True)
    demand["Value_ScaleTo100"] = demand["Value_ScaleTo100"] * \
        factor["factor"]
    demand.reset_index(inplace=True)
    demand.set_index(['DateShort'], inplace=True)
    
demand.index = pd.to_datetime(
    demand.index, format='%d.%m.%Y').strftime('%Y-%m-%d')
demand = demand.set_index('CountryCode', append=True).reorder_levels(
    ['CountryCode', 'DateShort'])

# if nodes contain more than one country, demand must be aggregated
moreCountriesThanOne = [
    countryCode for countryCode in regions_file['countryCode'] if ";" in countryCode]
for countryCodes in moreCountriesThanOne:
    concatList = []
    for countryCode in countryCodes.split(";"):
        concatList.append(demand.loc[threeToTwo[countryCode]])
    newDemand = pd.concat(concatList).groupby(['DateShort']).sum()
    newDemand["CountryCode"] = countryCodes
    newDemand = newDemand.set_index('CountryCode', append=True).reorder_levels(
        ['CountryCode', 'DateShort'])
    demand = pd.concat([demand, newDemand])

regression_parameters = pd.DataFrame(data=[], index=[
                                     'quadratic coefficient in MWh/K^2', 'linear coefficient in MWh/K', 'intercept in MWh', 'Rsquare'])

if not os.path.isdir(f'results/{domain}/demand_plots') and config['demand']['makePlots']:
    os.mkdir(f'results/{domain}/demand_plots')

for country in regions_file["countryCode"].drop_duplicates():
    if (not country in temperature_historic.index) or (not country in demand.index):
        continue
    if ";" in country:
        daily_demand = demand.loc[country].groupby('DateShort').sum()
    else:
        daily_demand = demand.loc[threeToTwo[country]
                                  ].groupby('DateShort').sum()
        
    daily_demand.columns = ['demand']
    daily_temperature = temperature_historic.loc[country].to_frame()
    daily_temperature.index.name = 'DateShort'
    daily_temperature.columns = ['temperature']
    intersect = pd.concat([daily_temperature, daily_demand], axis=1).dropna()
    x = intersect.loc[:, 'temperature'].values
    y = intersect.loc[:, 'demand'].values
    initial_guess = [17, -11747, 1959552]
    param_bounds = [(0, -np.inf, 0), (np.inf, np.inf, np.inf)]
    fit = curve_fit(fitfunc, x, y, p0=initial_guess, bounds=param_bounds)[0]
    fit[0] = abs(fit[0])
    model = np.poly1d(fit)
    if config['demand']['makePlots']:
        x_line = np.linspace(min(x), max(x)+10, 50)
        plt.scatter(x, y)
        plt.plot(x_line, model(x_line), color='red')
        plt.xlabel('daily average temperature in K')
        plt.ylabel('daily load in MWh')
        plt.title(country)
        plt.savefig(
            f'results/{domain}/demand_plots/demand_regression_{country}_{name}.png', dpi=300)
        plt.close()
    new_row = pd.Series(data={'quadratic coefficient in MWh/K^2': fit[0], 'linear coefficient in MWh/K': fit[1],
                        'intercept in MWh': fit[2], 'Rsquare': polyfit(x, y, fit)}, name=country)
    regression_parameters = pd.concat(
        [regression_parameters, new_row], axis=1, ignore_index=False)

regression_parameters = regression_parameters.T

regression_parameters.to_csv(snakemake.output[0], sep=';')
