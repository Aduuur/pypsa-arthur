# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Approximate cooling demand for all weather years. Derived from script build_heat_totals.py
"""

import logging
from itertools import product

import pandas as pd
import numpy as np
import geopandas as gpd
from numpy.polynomial import Polynomial

from scripts._helpers import configure_logging, load_cutout, checkBool

logger = logging.getLogger(__name__)

idx = pd.IndexSlice


def merge_cdd(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
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


def approximate_cooling_demand(
    energy_totals_cool: pd.DataFrame, cdd: pd.DataFrame, cols
) -> pd.DataFrame:
    """
    Approximate cooling demand for a set of countries based on energy totals and
    cooling degree days (CDD). A polynomial regression of cooling demand on CDDs
    is performed using data from 2007 to 2021. Then, for 2022 and 2023 (and other missing years),
    the cooling demand is estimated from known CDDs based on the regression.

    Parameters
    ----------
    energy_totals_cool : pd.DataFrame
        DataFrame with energy consumption by sector (columns), country, and year.
    cdd : pd.DataFrame
        DataFrame with number of cooling degree days by year (columns) and country (index).

    Returns
    -------
    pd.DataFrame
        DataFrame with approximated cooling demand for each country.

    Notes
    -----
    - Missing data is forward-filled for GB in 2020 and 2021.
    - If only one year of cooling data is available for a country, a point (0, 0) is added to allow the polynomial fit to work.
    """

    countries = cdd.columns.intersection(energy_totals_cool.index.levels[0])

    demands = {}

    for kind, sector, com in cols:
        # Use reduced number of years (2007-2021) for regression because it implicitly
        # assumes a constant building stock
        row = idx[:, 2007:2021]
        col = f"{kind} {sector} {com}"
        demand = energy_totals_cool.loc[row, col].unstack(0)

        # Forward fill works only with NaN, not with 0.0 values
        for c in list(demand.columns):
            for i in list(demand.index):
                if demand[c].loc[i] == 0.0:
                    demand.loc[i, c] = np.nan
                    logger.info(f'For country {c} in year {i} changed {col} for cooling from 0.0 to NaN.')

        # Forward-fill for GB in 2020 and 2021
        demand = demand.ffill(axis=0).bfill(axis=0)

        demand_approx = {}

        for c in countries:
            Y = demand[c].dropna()
            X = cdd.loc[Y.index, c]

            if len(X) == len(Y) == 1:
                X.loc[-1] = 0
                Y.loc[-1] = 0

            to_predict = cdd.index.difference(Y.index)
            X_pred = cdd.loc[to_predict, c]

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

        snakemake = mock_snakemake("build_cooling_totals")

    configure_logging(snakemake)

    non_historic_cutout = checkBool(snakemake.params.non_historic_cutout)
    cool_regression = checkBool(snakemake.params.et_regression)
    drop_leap_day = checkBool(snakemake.params.drop_leap_day)

    cdd = pd.read_csv(snakemake.input.cdd, index_col=0, parse_dates=True)  # all countries in PyPSA
    energy_totals_cool = pd.read_csv(snakemake.input.energy_totals_cooling, index_col=[0, 1])

    if non_historic_cutout == True:
        country_shapes_file = snakemake.input.country_shapes
        cutout_input = snakemake.input.cutout
        country_shapes = gpd.read_file(country_shapes_file).set_index('name')['geometry']
        cutout = load_cutout(cutout_input)
        da = cutout.cooling_demand(shapes=country_shapes)
        s = da.to_pandas()
        if drop_leap_day == True:
            s = s.drop(s.index[(s.index.month == 2) & (s.index.day == 29)])
        s = s.apply(lambda x: x.astype(int))

        # Replace existing and fill all missing timestamps
        cdd = merge_cdd(cdd, s)

        if cool_regression == True:
            # Remove historical cooling demand from data and trigger regression
            remove_years = list(s.index.year.unique())
            logger.info(
                f'If year(s) {remove_years} exist in historic cooling data (between 1990 and 2022), they are removed -> regression for missing values triggered'
            )
            indices_to_drop = energy_totals_cool.index.get_level_values('year').isin(remove_years)
            energy_totals_cool = energy_totals_cool.drop(index=energy_totals_cool.index[indices_to_drop])
        elif cool_regression == False:
            pass
        else:
            raise ValueError('config[ee][non_historic_cutout][et_regression] must be false or true')
    
        # no water in cooling demand
        cols = product(["total", "electricity"], ["services", "residential"], ["space"])
    elif non_historic_cutout == False:
        cols = product(["total", "electricity"], ["services", "residential"], ["space"])
    else:
        raise ValueError('config[ee][non_historic_cutout][enable] must be false or true')

    cdd = cdd.groupby(cdd.index.year).sum().div(1e3)

    cooling_demand = approximate_cooling_demand(energy_totals_cool, cdd, cols)

    cooling_demand.to_csv(snakemake.output.cooling_totals)
