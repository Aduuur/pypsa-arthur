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
import geopandas as gpd ##add geopandas
from numpy.polynomial import Polynomial

from scripts._helpers import configure_logging, load_cutout ##add loadcutout

logger = logging.getLogger(__name__)

idx = pd.IndexSlice

##New function
def merge_hdd(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two DataFrames with a DatetimeIndex where df2 is a subset of df1's columns.
    
    The function performs the following:
    - Keeps only the columns that are present in df2.
    - Combines the indices (timestamps) of both DataFrames.
    - If a timestamp exists in both DataFrames:
        * Prints an info message whenever a value is overwritten 
          (including when the value is the same).
        * If values differ, an additional message highlights the difference.
        * Overwrites values in df1 with values from df2.
    - If a timestamp exists only in df2, the corresponding rows are added.
    - Returns a new DataFrame sorted by time index.
    
    Parameters
    ----------
    df1 : pd.DataFrame
        Base DataFrame containing all possible columns and timestamps.
    df2 : pd.DataFrame
        DataFrame with a subset of columns from df1, whose values should overwrite df1.
    
    Returns
    -------
    pd.DataFrame
        A new DataFrame containing only the columns of df2, with values from df2 
        overwriting df1 where applicable, and additional timestamps from df2 included.
    """
    # Ensure we only use the subset of columns from df2
    common_cols = df2.columns

    # Prepare new DataFrame
    df_new = df1[common_cols].copy()

    # Iterate over all timestamps in df2
    for ts in df2.index:
        if ts in df_new.index: #if timestamp already exists
            # Compare and overwrite values
            for col in common_cols:
                val1 = df_new.at[ts, col]
                val2 = df2.at[ts, col]

                if pd.notna(val1) and pd.notna(val2):
                    rel_diff=abs(val1-val2)/val1
                    tol=0.03
                    if rel_diff>tol:
                        print(f"Difference of {rel_diff}>{tol} at {ts} in column '{col}': df1={val1}, df2={val2}")
                        print(f"Overwriting value at {ts} in column '{col}' with {val2}")
                    else:
                        print(f'Values at {ts} exist in df1 and df2 and are identical')
                else:
                    print(f"Overwriting values at {ts} in column '{col}' with {val2}")
                
                df_new.at[ts, col] = val2
        else: #default
            # Add new rows for timestamps not present in df1
            df_new.loc[ts, common_cols] = df2.loc[ts, common_cols]

    # Sort by index (time)
    df_new = df_new.sort_index()

    return df_new


def approximate_heat_demand(
    energy_totals: pd.DataFrame, hdd: pd.DataFrame
) -> pd.DataFrame:
    """
    Approximate heat demand for a set of countries based on energy totals and
    heating degree days (HDD). A polynomial regression of heat demand on HDDs
    is performed on the data from 2007 to 2021. Then, for 2022 and 2023, the
    heat demand is estimated from known HDDs based on the regression.

    Parameters
    ----------
    energy_totals : pd.DataFrame
        DataFrame with energy consumption by sector (columns), country and year. Output of :func:`scripts.build_energy_totals.py`.
    hdd : pd.DataFrame
        DataFrame with number of heating degree days by year (columns) and country (index).

    Returns
    -------
    pd.DataFrame
        DataFrame with approximated heat demand for each country.

    Notes
    -----
    - Missing data is filled forward for GB in 2020 and backward for CH from 2007 to 2009.
    - If only one year of heating data is available for a country, a point (0, 0) is added to make the polynomial fit work.
    """

    countries = hdd.columns.intersection(energy_totals.index.levels[0])

    demands = {}

    for kind, sector in product(["total", "electricity"], ["services", "residential"]):
        # reduced number years (2007-2021) for regression because it implicitly
        # assumes a constant building stock
        row = idx[:, 2007:2021]
        col = f"{kind} {sector} space"
        demand = energy_totals.loc[row, col].unstack(0)

        # ffill for GB in 2020- and bfill for CH 2007-2009
        # compromise to have more years available for the fit
        demand = demand.ffill(axis=0).bfill(axis=0)

        demand_approx = {}

        for c in countries:
            Y = demand[c].dropna()
            X = hdd.loc[Y.index, c]

            # Sometimes (looking at you, Switzerland) we only have
            # _one_ year of heating data to base the prediction on. In
            # this case we add a point at 0, 0 to make a "polynomial"
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
        demands[f"{kind} {sector} space"] = demand_approx.groupby(
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

    hdd_full = pd.read_csv(snakemake.input.hdd, index_col=0, parse_dates=True) #all counties in pypsa
    
    ####Changes####
    #snippets from https://gist.github.com/fneum/d99e24e19da423038fd55fe3a4ddf875
    country_shapes_file=snakemake.input.country_shapes
    cutout_input=snakemake.input.cutout
    country_shapes = gpd.read_file(country_shapes_file).set_index('name')['geometry'] #file from resources with only the countries considered in this run
    cutout = load_cutout(cutout_input)
    da = cutout.heat_demand(shapes=country_shapes) #atlite function
    s = da.to_pandas()

    hdd=merge_hdd(hdd_full, s)
    ########
    
    hdd = hdd.groupby(hdd.index.year).sum().div(1e3)

    energy_totals = pd.read_csv(snakemake.input.energy_totals, index_col=[0, 1])

    heat_demand = approximate_heat_demand(energy_totals, hdd)

    heat_demand.to_csv(snakemake.output.heat_totals)
