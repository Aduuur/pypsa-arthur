# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Approximate heat demand for all weather years.

:func:`approximate_heat_demand` approximates annual heat demand based on energy totals and heating degree days (HDD) using a regression of heat demand on HDDs.

Outputs
-------
- `resources/<run_name>/heat_totals.csv`: Approximated annual heat demand for each country.
"""

import logging
from itertools import product

import pandas as pd
import numpy as np
import geopandas as gpd 
from numpy.polynomial import Polynomial

from scripts._helpers import configure_logging, load_cutout, checkBool  # added load_cutout

logger = logging.getLogger(__name__)

idx = pd.IndexSlice


def merge_hdd(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two DataFrames with a DatetimeIndex on shared columns.
    """
    # Ensure we only use the columns which are in both DataFrames
    common_cols = df1.columns.intersection(df2.columns)

    # Prepare new DataFrame
    df_new = df1[common_cols].copy()

    # Iterate over all timestamps in df2
    for ts in df2.index:
        # Overwrite values for all timestamps
        df_new[common_cols] = df_new[common_cols].astype(float)
        df_new.loc[ts, common_cols] = df2.loc[ts, common_cols].astype(float)

    # Sort by index (time)
    df_new = df_new.sort_index()

    return df_new


def approximate_heat_demand(
    energy_totals: pd.DataFrame, hdd: pd.DataFrame,cols
) -> pd.DataFrame:
    """
    Approximate heat demand for a set of countries based on energy totals and
    heating degree days (HDD). A polynomial regression of heat demand on HDDs
    is performed using data from 2007 to 2021. Then, for 2022 and 2023 (and other missing years),
    the heat demand is estimated from known HDDs based on the regression.

    Parameters
    ----------
    energy_totals : pd.DataFrame
        DataFrame with energy consumption by sector (columns), country, and year. Output of :func:`scripts.build_energy_totals.py`.
    hdd : pd.DataFrame
        DataFrame with number of heating degree days by year (columns) and country (index).

    Returns
    -------
    pd.DataFrame
        DataFrame with approximated heat demand for each country.

    Notes
    -----
    - Missing data is forward-filled for GB in 2020 and backward-filled for CH from 2007 to 2009.
    - If only one year of heating data is available for a country, a point (0, 0) is added to allow the polynomial fit to work.
    """

    countries = hdd.columns.intersection(energy_totals.index.levels[0])

    demands = {}

    for kind, sector, com in cols:
        # Use reduced number of years (2007-2021) for regression because it implicitly
        # assumes a constant building stock
        row = idx[:, 2007:2021]
        col = f"{kind} {sector} {com}"
        demand = energy_totals.loc[row, col].unstack(0)
    
        #Forward fill works only with Nan, not with 0.0 Values
        for c in list(demand.columns):
            for i in list(demand.index):
                if demand[c].loc[i] == 0.0:
                     demand.loc[i, c] = np.nan
                     logger.info(f'For country {c} in year {i} changed {col} for heat from 0.0 to NaN.')

        # Forward-fill for GB in 2020 and backward-fill for CH 2007-2009
        # compromise to have more years available for the fit
        demand = demand.ffill(axis=0).bfill(axis=0)

        demand_approx = {}

        for c in countries:
            Y = demand[c].dropna()
            X = hdd.loc[Y.index, c]

            # Sometimes (looking at Switzerland) we only have
            # _one_ year of heating data to base the prediction on. In
            # this case we add a point at 0, 0 to make the polynomial
            # fit work.
            if len(X) == len(Y) == 1:
                X.loc[-1] = 0
                Y.loc[-1] = 0

            to_predict = hdd.index.difference(Y.index)
            X_pred = hdd.loc[to_predict, c]

            p = Polynomial.fit(X, Y, 1)
            Y_pred = p(X_pred)

            demand_approx[c] = pd.Series(Y_pred, index=to_predict)

        demand_approx = pd.DataFrame(demand_approx)
        demand_approx = pd.concat([demand, demand_approx]).sort_index()
        demands[f"{kind} {sector} {com}"] = demand_approx.groupby(
            demand_approx.index
        ).sum()

    demands = pd.concat(demands).unstack().T.clip(lower=0)
    demands.index.names = ["country", "year"]

    return demands


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_heat_totals")

    configure_logging(snakemake)
    
    non_historic_cutout = checkBool(snakemake.params.non_historic_cutout)
    heat_regression = checkBool(snakemake.params.et_regression)
    drop_leap_day = checkBool(snakemake.params.drop_leap_day)
    
    hdd = pd.read_csv(snakemake.input.hdd, index_col=0, parse_dates=True)  # all countries in PyPSA
    energy_totals = pd.read_csv(snakemake.input.energy_totals, index_col=[0, 1])
    
    if non_historic_cutout == True:
        # snippets from https://gist.github.com/fneum/d99e24e19da423038fd55fe3a4ddf875
        country_shapes_file = snakemake.input.country_shapes
        cutout_input = snakemake.input.cutout
        country_shapes = gpd.read_file(country_shapes_file).set_index('name')['geometry']  # only countries considered in this run
        cutout = load_cutout(cutout_input)
        da = cutout.heat_demand(shapes=country_shapes)  # atlite function
        s = da.to_pandas()
        if drop_leap_day == True:
            s = s.drop(s.index[(s.index.month == 2) & (s.index.day == 29)])

        s = s.apply(lambda x: x.astype(int))  # convert all entries to integer

        # Replace existing and fill all missing timestamps
        hdd = merge_hdd(hdd, s)
        
        if heat_regression == True:
            # Remove historical heat demand from data and so triggers regression 
            remove_years = list(s.index.year.unique())
            logger.info(
                f'If year(s) {remove_years} exist in historic heat data (between 1990 and 2022), they are removed -> regression for missing values triggered'
            )
            indices_to_drop = energy_totals.index.get_level_values('year').isin(remove_years)
            energy_totals = energy_totals.drop(index=energy_totals.index[indices_to_drop])
        elif heat_regression == False:
            pass
        else:
            raise ValueError('config[ee][non_historic_cutout][et_regression] must be false or true')

        #Allow regression for water too
        cols=product(["total", "electricity"], ["services", "residential"],['space', 'water'])
    elif non_historic_cutout == False:
        #Preserve default functionality. Updated for water by energy totals in prepare_sector_network
        cols=product(["total", "electricity"], ["services", "residential"],['space'])
    else:
        raise ValueError('config[ee][non_historic_cutout][enable] must be false or true')

    hdd = hdd.groupby(hdd.index.year).sum().div(1e3)

    # Automatically replace all missing values/years present in HDD but missing in energy_totals using regression
    heat_demand = approximate_heat_demand(energy_totals, hdd, cols)

    heat_demand.to_csv(snakemake.output.heat_totals)
