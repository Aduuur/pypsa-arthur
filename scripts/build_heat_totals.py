"""
build_heat_totals.py (robust)
- Optionally uses a non-historic cutout to overwrite HDD proxies (annual heat demand per country from cutout).
- Optionally drops those years from energy_totals to force regression-based reconstruction.
- Performs per-country linear regression: heat_demand ~ HDD (or HDD-proxy), trained on 2007..2021.
"""

import logging
from itertools import product
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from numpy.polynomial import Polynomial

# PyPSA-Eur helpers (available in repo)
from scripts._helpers import configure_logging, load_cutout, checkBool

logger = logging.getLogger(__name__)
idx = pd.IndexSlice


# =============================================================================
# Core logic
# =============================================================================
def _ensure_year_index_int(df: pd.DataFrame, name: str = "index") -> pd.DataFrame:
    """Ensure df.index represents integer years."""
    out = df.copy()
    if not np.issubdtype(out.index.dtype, np.number):
        # try datetime-like first
        try:
            out.index = pd.to_datetime(out.index).year
        except Exception:
            out.index = out.index.astype(int)
    else:
        out.index = out.index.astype(int)
    out.index.name = name
    return out


def _ensure_energy_totals_multiindex(energy_totals: pd.DataFrame) -> pd.DataFrame:
    """Ensure MultiIndex (country, year[int]) and normalized dtypes."""
    if not isinstance(energy_totals.index, pd.MultiIndex) or energy_totals.index.nlevels != 2:
        raise ValueError("energy_totals must have a MultiIndex (country, year).")

    et = energy_totals.copy()
    et.index = pd.MultiIndex.from_arrays(
        [
            et.index.get_level_values(0).astype(str),
            et.index.get_level_values(1).astype(int),
        ],
        names=["country", "year"],
    )
    return et


def _safe_overwrite_hdd_with_cutout_proxy(
    hdd: pd.DataFrame,
    s_yearly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Overwrite/add years in HDD table with annualized cutout heat demand (proxy).
    This is where your previous crash happened: ensure equal lengths + alignment.
    """
    hdd2 = hdd.copy()

    # normalize columns to strings (country names)
    hdd2.columns = hdd2.columns.astype(str)
    s_yearly2 = s_yearly.copy()
    s_yearly2.columns = s_yearly2.columns.astype(str)

    common = hdd2.columns.intersection(s_yearly2.columns)
    if len(common) == 0:
        raise ValueError(
            "No overlapping country columns between HDD table and cutout-derived series. "
            "Check column naming (e.g. DE/FR vs Germany/France)."
        )

    # ensure year index is int
    hdd2 = _ensure_year_index_int(hdd2, name="year")
    s_yearly2 = _ensure_year_index_int(s_yearly2, name="year")

    for y in s_yearly2.index.unique():
        y = int(y)

        # make sure the row exists (create if missing)
        if y not in hdd2.index:
            # create a new row with NaNs
            hdd2.loc[y, :] = np.nan

        # align strictly on 'common' and assign as ndarray to avoid pandas broadcasting quirks
        vals = s_yearly2.loc[y, common].astype(float).to_numpy()

        if len(vals) != len(common):
            raise ValueError(
                f"Internal alignment error while overwriting HDD at year={y}: "
                f"len(vals)={len(vals)} != len(common)={len(common)}"
            )

        hdd2.loc[y, common] = vals

    # keep sorted years
    hdd2 = hdd2.sort_index()
    return hdd2


def approximate_heat_demand(
    energy_totals: pd.DataFrame,
    hdd_yearly: pd.DataFrame,
    cols: Iterable[Tuple[str, str, str]],
) -> pd.DataFrame:
    """
    Regression heat demand ~ HDD (linear) based on years 2007..2021,
    then predict missing years.
    """
    energy_totals = _ensure_energy_totals_multiindex(energy_totals)
    hdd_yearly = _ensure_year_index_int(hdd_yearly, name="year")

    # intersection of countries available in both
    et_countries = pd.Index(energy_totals.index.get_level_values("country").unique())
    countries = hdd_yearly.columns.astype(str).intersection(et_countries)

    if len(countries) == 0:
        raise ValueError(
            "No overlapping countries between hdd_yearly columns and energy_totals index. "
            "Check country codes/names."
        )

    demands: Dict[str, pd.DataFrame] = {}

    for kind, sector, com in cols:
        col = f"{kind} {sector} {com}"
        if col not in energy_totals.columns:
            logger.warning("Column '%s' not found in energy_totals -> skip.", col)
            continue

        # Only regression years
        row = idx[:, 2007:2021]
        demand = energy_totals.loc[row, col].unstack(0)  # index: year, columns: country
        demand = demand.replace(0.0, np.nan).ffill(axis=0).bfill(axis=0)

        demand_approx: Dict[str, pd.Series] = {}

        for c in countries:
            if c not in demand.columns:
                continue

            Y = demand[c].dropna()
            if Y.empty:
                continue

            # align X to Y years (both indexed by int years)
            if not set(Y.index).issubset(set(hdd_yearly.index)):
                missing = sorted(set(Y.index) - set(hdd_yearly.index))
                logger.warning("Missing HDD years for %s: %s (skip those years in fit).", c, missing)
                Y = Y.loc[Y.index.intersection(hdd_yearly.index)]
                if Y.empty:
                    continue

            X = hdd_yearly.loc[Y.index, c].astype(float)

            # handle degenerate 1-point fit
            if len(X) == len(Y) == 1:
                X = pd.concat([X, pd.Series([0.0], index=[-1])])
                Y = pd.concat([Y, pd.Series([0.0], index=[-1])])

            to_predict = hdd_yearly.index.difference(Y.index)
            X_pred = hdd_yearly.loc[to_predict, c].astype(float)

            # linear regression using numpy.polynomial.Polynomial.fit
            p = Polynomial.fit(X.to_numpy(), Y.to_numpy(), 1)
            Y_pred = p(X_pred.to_numpy())

            demand_approx[c] = pd.Series(Y_pred, index=to_predict)

        # combine original + predicted
        demand_approx_df = pd.DataFrame(demand_approx)
        demand_all = pd.concat([demand, demand_approx_df]).sort_index()

        # if duplicate years exist (shouldn't), aggregate
        demand_all = demand_all.groupby(demand_all.index).sum()

        demands[col] = demand_all

    if not demands:
        raise ValueError("No demand columns could be processed. Check 'cols' and energy_totals columns.")

    # stack back to MultiIndex (country, year)
    out = pd.concat(demands).unstack().T.clip(lower=0)
    out.index.names = ["country", "year"]
    return out


# =============================================================================
# Main (Snakemake)
# =============================================================================
if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake("build_heat_totals")

    configure_logging(snakemake)

    # ----------------------------
    # Robust config reading
    # ----------------------------
    ee_cfg = snakemake.config.get("ee", {}) or {}
    nh_cfg = (ee_cfg.get("non_historic_cutout", {}) or {})

    if isinstance(nh_cfg, dict):
        non_historic_cutout = checkBool(nh_cfg.get("enable", False))
        heat_regression = checkBool(nh_cfg.get("et_regression", False))
    else:
        non_historic_cutout = checkBool(nh_cfg)
        heat_regression = checkBool(getattr(snakemake, "params", {}).get("et_regression", False))

    # drop_leap_day may live in config.enable.drop_leap_day OR params
    enable_cfg = snakemake.config.get("enable", {}) or {}
    params = getattr(snakemake, "params", {}) or {}
    drop_leap_day = checkBool(params.get("drop_leap_day", enable_cfg.get("drop_leap_day", False)))

    logger.info("non_historic_cutout=%s | heat_regression=%s | drop_leap_day=%s",
                non_historic_cutout, heat_regression, drop_leap_day)

    # ----------------------------
    # Read inputs
    # ----------------------------
    hdd = pd.read_csv(snakemake.input.hdd, index_col=0)
    energy_totals = pd.read_csv(snakemake.input.energy_totals, index_col=[0, 1])

    # normalize
    hdd = _ensure_year_index_int(hdd, name="year")
    energy_totals = _ensure_energy_totals_multiindex(energy_totals)

    # ----------------------------
    # Non-historic cutout path
    # ----------------------------
    if non_historic_cutout is True:
        country_shapes = (
            gpd.read_file(snakemake.input.country_shapes)
            .set_index("name")["geometry"]
        )

        cutout = load_cutout(snakemake.input.cutout)

        # hourly timeseries per country (shape aggregation)
        da = cutout.heat_demand(shapes=country_shapes)
        s = da.to_pandas()

        if drop_leap_day:
            s = s.drop(s.index[(s.index.month == 2) & (s.index.day == 29)])

        # annualize -> proxy in "HDD table"
        s_yearly = (s.groupby(s.index.year).sum() / 1e3)
        s_yearly.index = s_yearly.index.astype(int)

        # overwrite HDD with robust alignment
        hdd = _safe_overwrite_hdd_with_cutout_proxy(hdd, s_yearly)

        if heat_regression is True:
            remove_years = sorted({int(y) for y in s_yearly.index.unique()})
            logger.info("Dropping years %s from energy_totals to trigger regression.", remove_years)
            mask = energy_totals.index.get_level_values("year").isin(remove_years)
            energy_totals = energy_totals.loc[~mask]
        elif heat_regression is False:
            logger.info("heat_regression=false -> keep energy_totals as-is.")
        else:
            raise ValueError("config[ee][non_historic_cutout][et_regression] must be false or true")

        cols = product(["total", "electricity"], ["services", "residential"], ["space", "water"])

    elif non_historic_cutout is False:
        cols = product(["total", "electricity"], ["services", "residential"], ["space"])
    else:
        raise ValueError("config[ee][non_historic_cutout][enable] must be false or true")

    # annual HDD (already annual normally) – keep same output format
    hdd_yearly = hdd.groupby(hdd.index).sum()

    heat_demand = approximate_heat_demand(energy_totals, hdd_yearly, cols)
    heat_demand.to_csv(snakemake.output.heat_totals)
    logger.info("Wrote heat totals to %s", snakemake.output.heat_totals)


