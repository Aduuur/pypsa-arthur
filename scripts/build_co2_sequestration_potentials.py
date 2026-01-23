# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build regionalised geological sequestration potential for carbon dioxide using
data from `CO2Stop <https://setis.ec.europa.eu/european-co2-storage-
database_en>`_.
"""

from typing import Any, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sg
from shapely.ops import unary_union

CRS = "EPSG:4326"


def convert_to_2d(
    geom: Union[sg.base.BaseGeometry, Any],
) -> Union[sg.base.BaseGeometry, Any]:
    """
    Remove the third dimension (z-coordinate) from a shapely geometry object.

    Parameters
    ----------
    geom : shapely.geometry
        A shapely geometry object which may contain 3D coordinates

    Returns
    -------
    shapely.geometry
        The same type of geometry with only 2D coordinates (x,y)

    Raises
    ------
    RuntimeError
        If the geometry type is not supported
    """
    if geom is None or geom.is_empty:
        return geom

    # Handle coordinates directly for simple geometries
    if isinstance(geom, (sg.Point, sg.LineString, sg.LinearRing)):
        return type(geom)([xy[0:2] for xy in list(geom.coords)])

    # Handle Polygon
    elif isinstance(geom, sg.Polygon):
        new_exterior = convert_to_2d(geom.exterior)
        new_interiors = [convert_to_2d(interior) for interior in geom.interiors]
        return sg.Polygon(new_exterior, new_interiors)

    # Handle collections of geometries
    elif isinstance(
        geom,
        (sg.MultiPoint, sg.MultiLineString, sg.MultiPolygon, sg.GeometryCollection),
    ):
        return type(geom)([convert_to_2d(part) for part in geom.geoms])

    else:
        raise RuntimeError(f"Geometry type {type(geom)} is not supported.")


def create_capacity_map_storage(table_fn: str, map_fn: str) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame of CO2 storage capacities.

    Fix for CO2Stop KML schema:
    - KML provides only ['Name','Description','geometry'] (no ID/COUNTRYCOD).
    - CSV provides STORAGE_UNIT_NAME and COUNTRYCODE.
    We therefore:
      * map KML.Name -> ID
      * join CSV.COUNTRYCODE -> COUNTRYCOD via STORAGE_UNIT_NAME
      * merge capacities via STORAGE_UNIT_NAME (not STORAGE_UNIT_ID)
    """
    df = pd.read_csv(table_fn)

    # --- Read polygons from KML ---
    gdf = gpd.read_file(map_fn)

    # KML schema: Name, Description, geometry
    if "ID" not in gdf.columns:
        if "id" in gdf.columns:
            gdf = gdf.rename(columns={"id": "ID"})
        elif "Name" in gdf.columns:
            gdf = gdf.rename(columns={"Name": "ID"})
        else:
            # last resort: stable ID
            gdf["ID"] = gdf.index.astype(str)

    gdf["ID"] = gdf["ID"].astype(str).str.strip()

    # Fix potential invalid geometries
    gdf.geometry = gdf.geometry.buffer(0)

    # --- Add country code from CSV (via name match) ---
    # We have: CSV.STORAGE_UNIT_NAME and CSV.COUNTRYCODE
    if "COUNTRYCOD" not in gdf.columns:
        df["STORAGE_UNIT_NAME"] = df["STORAGE_UNIT_NAME"].astype(str).str.strip()

        cc_map = (
            df.dropna(subset=["STORAGE_UNIT_NAME", "COUNTRYCODE"])
              .drop_duplicates("STORAGE_UNIT_NAME")[["STORAGE_UNIT_NAME", "COUNTRYCODE"]]
              .rename(columns={"STORAGE_UNIT_NAME": "ID", "COUNTRYCODE": "COUNTRYCOD"})
        )

        gdf = gdf.merge(cc_map, on="ID", how="left")

    # Keep only what we need for grouping
    sel = ["COUNTRYCOD", "ID", "geometry"]
    for c in sel:
        if c not in gdf.columns:
            gdf[c] = None
    gdf = gdf[sel]

    # Combine shapes with the same (country, id) into one multi-polygon
    gdf = gdf.groupby(["COUNTRYCOD", "ID"]).agg(unary_union).reset_index()
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=CRS)

    # ---------------------------------------------------------------------
    # Capacity calculations (unchanged logic)
    # ---------------------------------------------------------------------

    # conservative estimate: use MIN
    df["conservative estimate Mt"] = (
        df["EST_STORECAP_MIN"]
        .replace(0, np.nan)
        .fillna(df["STORE_CAP_MIN"])
        .add(df["STORE_CAP_HCDAUGHTER"])
        .fillna(0)
    )

    # neutral estimate: use MEAN
    df["neutral estimate Mt"] = (
        df["EST_STORECAP_MEAN"]
        .replace(0, np.nan)
        .fillna(df["STORE_CAP_MEAN"])
        .add(df["STORE_CAP_HCDAUGHTER"])
        .replace(0, np.nan)
        .fillna(df["conservative estimate Mt"])
    )

    # optimistic estimate: use MAX
    df["optimistic estimate Mt"] = (
        df["EST_STORECAP_MAX"]
        .replace(0, np.nan)
        .fillna(df["STORE_CAP_MAX"])
        .add(df["STORE_CAP_HCDAUGHTER"])
        .replace(0, np.nan)
        .fillna(df["neutral estimate Mt"])
    )

    # Keep the relevant columns
    sel = [
        "STORAGE_UNIT_NAME",
        "ASSESS_UNIT_TYPE",
        "conservative estimate Mt",
        "neutral estimate Mt",
        "optimistic estimate Mt",
    ]
    df = df[sel].copy()
    df["STORAGE_UNIT_NAME"] = df["STORAGE_UNIT_NAME"].astype(str).str.strip()

    # ---------------------------------------------------------------------
    # IMPORTANT FIX: merge on NAME (KML.ID == CSV.STORAGE_UNIT_NAME)
    # ---------------------------------------------------------------------
    gdf = gdf.merge(df, left_on="ID", right_on="STORAGE_UNIT_NAME", how="left").drop(
        "STORAGE_UNIT_NAME", axis=1
    )

    return gdf

def create_capacity_map_traps(table_fn: list[str], map_fn: str) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame of CO2 trap capacities.

    Fix for CO2Stop KML schema:
    - KML provides only ['Name','Description','geometry'] (no ID/COUNTRYCOD).
    - Trap CSVs provide TRAP_NAME and COUNTRYCODE (in CO2Stop DataInterrogationSystem tables).
    We therefore:
      * map KML.Name -> ID
      * join COUNTRYCODE -> COUNTRYCOD via TRAP_NAME
      * merge capacities via TRAP_NAME (not TRAP_ID)
    """
    df = pd.concat([pd.read_csv(path) for path in table_fn], ignore_index=True)

    # --- Read polygons from KML ---
    gdf = gpd.read_file(map_fn)

    # Map KML Name -> ID
    if "ID" not in gdf.columns:
        if "id" in gdf.columns:
            gdf = gdf.rename(columns={"id": "ID"})
        elif "Name" in gdf.columns:
            gdf = gdf.rename(columns={"Name": "ID"})
        else:
            gdf["ID"] = gdf.index.astype(str)

    gdf["ID"] = gdf["ID"].astype(str).str.strip()
    gdf.geometry = gdf.geometry.buffer(0)

    # --- Add country code from CSV if possible ---
    # Typical CO2Stop trap tables contain TRAP_NAME and COUNTRYCODE
    if "COUNTRYCOD" not in gdf.columns:
        # Ensure required cols exist in df (some files may not have them)
        if "TRAP_NAME" in df.columns and "COUNTRYCODE" in df.columns:
            df["TRAP_NAME"] = df["TRAP_NAME"].astype(str).str.strip()

            cc_map = (
                df.dropna(subset=["TRAP_NAME", "COUNTRYCODE"])
                  .drop_duplicates("TRAP_NAME")[["TRAP_NAME", "COUNTRYCODE"]]
                  .rename(columns={"TRAP_NAME": "ID", "COUNTRYCODE": "COUNTRYCOD"})
            )
            gdf = gdf.merge(cc_map, on="ID", how="left")
        else:
            gdf["COUNTRYCOD"] = None

    # Keep only required columns for grouping
    sel = ["COUNTRYCOD", "ID", "geometry"]
    for c in sel:
        if c not in gdf.columns:
            gdf[c] = None
    gdf = gdf[sel]

    # Combine shapes with same (country, id)
    gdf = gdf.groupby(["COUNTRYCOD", "ID"]).agg(unary_union).reset_index()
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=CRS)

    # ---------------------------------------------------------------------
    # Capacity calculations (original logic unchanged)
    # ---------------------------------------------------------------------

    # conservative estimate: use MIN
    df["conservative estimate aquifer Mt"] = (
        df["EST_STORECAP_MIN"].replace(0, np.nan).fillna(df["STORE_CAP_MIN"])
    )
    df["conservative estimate OIL Mt"] = (
        df["MIN_EST_STORE_CAP_OIL"]
        .replace(0, np.nan)
        .fillna(df["MIN_CALC_STORE_CAP_OIL"])
    )
    df["conservative estimate GAS Mt"] = (
        df["MIN_EST_STORE_CAP_GAS"]
        .replace(0, np.nan)
        .fillna(df["MIN_CALC_STORE_CAP_GAS"])
    )

    sel = [
        "conservative estimate aquifer Mt",
        "conservative estimate OIL Mt",
        "conservative estimate GAS Mt",
    ]
    df["conservative estimate Mt"] = df[sel].sum(axis=1).fillna(0)

    # neutral estimate: use MEAN
    df["neutral estimate aquifer Mt"] = (
        df["EST_STORECAP_MEAN"].replace(0, np.nan).fillna(df["STORE_CAP_MEAN"])
    )
    df["neutral estimate OIL Mt"] = (
        df["MEAN_EST_STORE_CAP_OIL"]
        .replace(0, np.nan)
        .fillna(df["MEAN_CALC_STORE_CAP_OIL"])
    )
    df["neutral estimate GAS Mt"] = (
        df["MEAN_EST_STORE_CAP_GAS"]
        .replace(0, np.nan)
        .fillna(df["MEAN_CALC_STORE_CAP_GAS"])
    )

    sel = [
        "neutral estimate aquifer Mt",
        "neutral estimate OIL Mt",
        "neutral estimate GAS Mt",
    ]
    df["neutral estimate Mt"] = (
        df[sel].sum(axis=1).replace(0, np.nan).fillna(df["conservative estimate Mt"])
    )

    # optimistic estimate: use MAX
    df["optimistic estimate aquifer Mt"] = (
        df["EST_STORECAP_MAX"].replace(0, np.nan).fillna(df["STORE_CAP_MAX"])
    )
    df["optimistic estimate OIL Mt"] = (
        df["MAX_EST_STORE_CAP_OIL"]
        .replace(0, np.nan)
        .fillna(df["MAX_CALC_STORE_CAP_OIL"])
    )
    df["optimistic estimate GAS Mt"] = (
        df["MAX_EST_STORE_CAP_GAS"]
        .replace(0, np.nan)
        .fillna(df["MAX_CALC_STORE_CAP_GAS"])
    )

    sel = [
        "optimistic estimate aquifer Mt",
        "optimistic estimate OIL Mt",
        "optimistic estimate GAS Mt",
    ]
    df["optimistic estimate Mt"] = (
        df[sel].sum(axis=1).replace(0, np.nan).fillna(df["neutral estimate Mt"])
    )

    # Keep relevant cols (we merge by TRAP_NAME)
    sel = [
        "TRAP_NAME",
        "ASSESS_UNIT_TYPE",
        "optimistic estimate Mt",
        "neutral estimate Mt",
        "conservative estimate Mt",
        "optimistic estimate aquifer Mt",
        "optimistic estimate OIL Mt",
        "optimistic estimate GAS Mt",
        "neutral estimate aquifer Mt",
        "neutral estimate OIL Mt",
        "neutral estimate GAS Mt",
        "conservative estimate aquifer Mt",
        "conservative estimate OIL Mt",
        "conservative estimate GAS Mt",
    ]
    df = df[sel].copy()
    df["TRAP_NAME"] = df["TRAP_NAME"].astype(str).str.strip()

    # IMPORTANT: merge on name (KML.ID == CSV.TRAP_NAME)
    gdf = gdf.merge(df, left_on="ID", right_on="TRAP_NAME", how="left").drop(
        "TRAP_NAME", axis=1
    )

    return gdf



def merge_maps(
    traps_map: gpd.GeoDataFrame, storage_map: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Merge trap and storage map GeoDataFrames into a single map.

    Parameters
    ----------
    traps_map : gpd.GeoDataFrame
        GeoDataFrame containing trap geometries and capacity data
    storage_map : gpd.GeoDataFrame
        GeoDataFrame containing storage unit geometries and capacity data

    Returns
    -------
    gpd.GeoDataFrame
        Combined GeoDataFrame with all geometries and capacity data
    """
    # Ensure both DataFrames have the same columns
    missing_in_traps = set(storage_map.columns) - set(traps_map.columns)
    missing_in_storage = set(traps_map.columns) - set(storage_map.columns)
    for col in missing_in_traps:
        traps_map[col] = 0
    for col in missing_in_storage:
        storage_map[col] = 0

    storage_map["geometry"] = storage_map["geometry"].apply(convert_to_2d)
    traps_map["geometry"] = traps_map["geometry"].apply(convert_to_2d)

    gdf = gpd.GeoDataFrame(pd.concat([storage_map, traps_map]), crs=CRS)

    gdf.drop_duplicates(inplace=True)

    return gdf


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_co2_storage")

    table_fn = snakemake.input.storage_table
    map_fn = snakemake.input.storage_map
    storage_map = create_capacity_map_storage(table_fn, map_fn)

    table_fn = [
        snakemake.input.traps_table1,
        snakemake.input.traps_table2,
        snakemake.input.traps_table3,
    ]
    map_fn = snakemake.input.traps_map
    traps_map = create_capacity_map_traps(table_fn, map_fn)

    gdf = merge_maps(traps_map, storage_map)

    gdf.to_file(snakemake.output[0])
