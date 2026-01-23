# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT
"""
Approximate heat demand for all weather years.
"""

import logging
from itertools import product

import numpy as np
import pandas as pd
import geopandas as gpd
from numpy.polynomial import Polynomial

from scripts._helpers import configure_logging, load_cutout, checkBool

logger = logging.getLogger(__name__)
idx = pd.IndexSlice


def approximate_heat_demand(energy_totals: pd.DataFrame, hdd_yearly: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Regression heat demand ~ HDD (linear) based on years 2007..2021,
    then predict missing years.
    """
    # Make sure we have a proper MultiIndex: (country, year[int])
    if not isinstance(energy_totals.index, pd.MultiIndex) or energy_totals.index.nlevels != 2:
        raise ValueError("energy_totals must have a MultiIndex (country, year).")

    # Ensure year is int
    energy_totals = energy_totals.copy()
    energy_totals.index = pd.MultiIndex.from_arrays(
        [energy_totals.index.get_level_values(0),
         energy_totals.index.get_level_values(1).astype(int)],
        names=["country", "year"],
    )

    countries = hdd_yearly.columns.intersection(energy_totals.index.levels[0])
    demands = {}

    for kind, sector, com in cols:
        col = f"{kind} {sector} {com}"

        # Only regression years
        row = idx[:, 2007:2021]
        demand = energy_totals.loc[row, col].unstack(0)

        # Replace 0.0 by NaN (so ffill works)
        demand = demand.replace(0.0, np.nan)
        demand = demand.ffill(axis=0).bfill(axis=0)

        demand_approx = {}

        for c in countries:
            Y = demand[c].dropna()
            if Y.empty:
                continue

            X = hdd_yearly.loc[Y.index, c]

            if len(X) == len(Y) == 1:
                X.loc[-1] = 0
                Y.loc[-1] = 0

            to_predict = hdd_yearly.index.difference(Y.index)
            X_pred = hdd_yearly.loc[to_predict, c]

            p = Polynomial.fit(X, Y, 1)
            Y_pred = p(X_pred)

            demand_approx[c] = pd.Series(Y_pred, index=to_predict)

        demand_approx = pd.DataFrame(demand_approx)
        demand_all = pd.concat([demand, demand_approx]).sort_index()
        demands[col] = demand_all.groupby(demand_all.index).sum()

    demands = pd.concat(demands).unstack().T.clip(lower=0)
    demands.index.names = ["country", "year"]
    return demands


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake("build_heat_totals")

    configure_logging(snakemake)

    # ----------------------------
    # Robust config reading
    # ----------------------------
    ee_cfg = snakemake.config.get("ee", {})
    nh_cfg = ee_cfg.get("non_historic_cutout", {})

    # nh_cfg might be dict or already bool depending on the rule
    if isinstance(nh_cfg, dict):
        non_historic_cutout = checkBool(nh_cfg.get("enable", False))
        heat_regression = checkBool(nh_cfg.get("et_regression", False))
    else:
        non_historic_cutout = checkBool(nh_cfg)
        # fallback
        heat_regression = checkBool(snakemake.params.get("et_regression", False)) if hasattr(snakemake, "params") else False

    drop_leap_day = checkBool(snakemake.params.drop_leap_day)

    # ----------------------------
    # Read inputs
    # ----------------------------
    hdd = pd.read_csv(snakemake.input.hdd, index_col=0)
    energy_totals = pd.read_csv(snakemake.input.energy_totals, index_col=[0, 1])

    # Normalize index types
    # HDD index is typically year numbers
    if not np.issubdtype(hdd.index.dtype, np.number):
        # if it was parsed as datetime/string, try to convert to int year
        try:
            hdd.index = pd.to_datetime(hdd.index).year
        except Exception:
            hdd.index = hdd.index.astype(int)
    else:
        hdd.index = hdd.index.astype(int)

    # energy_totals year-level also int
    energy_totals.index = pd.MultiIndex.from_arrays(
        [energy_totals.index.get_level_values(0),
         energy_totals.index.get_level_values(1).astype(int)],
        names=["country", "year"],
    )

    # ----------------------------
    # Non-historic cutout path
    # ----------------------------
    if non_historic_cutout is True:
        country_shapes = (
            gpd.read_file(snakemake.input.country_shapes)
            .set_index("name")["geometry"]
        )
        cutout = load_cutout(snakemake.input.cutout)

        da = cutout.heat_demand(shapes=country_shapes)  # hourly timeseries
        s = da.to_pandas()

        if drop_leap_day:
            s = s.drop(s.index[(s.index.month == 2) & (s.index.day == 29)])

        # annualize cutout heat demand -> use as HDD-proxy (or demand proxy)
        s_yearly = s.groupby(s.index.year).sum() / 1e3  # keep your scaling

        # replace / add these years into HDD table
        # hdd is indexed by year (int)
        for y in s_yearly.index:
            # only overwrite intersecting countries
            common = hdd.columns.intersection(s_yearly.columns)
            hdd.loc[int(y), common] = s_yearly.loc[y, common].astype(float)

        if heat_regression is True:
            remove_years = list(map(int, s_yearly.index.unique()))
            logger.info(
                f"Drop years {remove_years} from energy_totals to trigger regression."
            )
            mask = energy_totals.index.get_level_values("year").isin(remove_years)
            energy_totals = energy_totals.loc[~mask]
        elif heat_regression is False:
            pass
        else:
            raise ValueError("config[ee][non_historic_cutout][et_regression] must be false or true")

        cols = product(["total", "electricity"], ["services", "residential"], ["space", "water"])

    elif non_historic_cutout is False:
        cols = product(["total", "electricity"], ["services", "residential"], ["space"])
    else:
        raise ValueError("config[ee][non_historic_cutout][enable] must be false or true")

    # annual HDD (already annual normally) – keep the same output format
    hdd_yearly = hdd.groupby(hdd.index).sum()

    heat_demand = approximate_heat_demand(energy_totals, hdd_yearly, cols)
    heat_demand.to_csv(snakemake.output.heat_totals)

