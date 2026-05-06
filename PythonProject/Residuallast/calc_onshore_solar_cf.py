# calc onshore solar + wind cf
from __future__ import annotations

import os
import glob
import gc
import logging
import shutil
from typing import Optional, List

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import atlite
from joblib import Parallel, delayed

import rasterio
from rasterio.errors import RasterioIOError


# =============================================================================
# KONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MODELL / SZENARIO ---
MODEL_NAME = "CNRM-CERFACS-CM5"
RCP_LEVEL = "rcp85"   # z. B. "rcp26", "rcp45", "rcp85"

YEARS_TO_ANALYZE = list(range(2025, 2051))
COUNTRIES_TO_ANALYZE = [
    "Germany", "France", "Poland", "Austria", "Switzerland",
    "Italy", "Spain", "Belgium", "Netherlands", "Denmark", "Sweden",
    "Portugal", "Norway", "Finland", "Lithuania", "Latvia", "Estonia",
    "United Kingdom", "Czechia",
]

COUNTRY_TO_TURBINE_MAP = { #wird nicht verwendet, ist noch von Lisa
    "Germany": "eno_126_4_8",
    "France": "Siemens_SWT_107_3600kW",
    "Spain": "eno_126_4",
    "Italy": "eno_126_4_8",
    "United Kingdom": "eno_126_4_8",
    "Poland": "Vestas_V112_3MW",
}
DEFAULT_TURBINE = "Vestas_V112_3MW"

DEFAULT_EXPONENT = 2.0
COUNTRY_SPECIFIC_EXPONENTS = {"Spain": 2.0}

CUTOUT_DIR = f"/mnt/endata/Cordex/Pypsa-Compatibility/results/europe/{MODEL_NAME}/cutouts"
TECHFILES_DIR = f"/mnt/endata/Cordex/Pypsa-Compatibility/results/europe/{MODEL_NAME}/techfiles"

BASE_OUT = "/mnt/endata/MA_Arthur"
RESULTS_DIR = os.path.join(
    BASE_OUT,
    f"atlite_cf_results_{RCP_LEVEL}",
    f"results_per_year_uncorrected_{RCP_LEVEL}",
)
os.makedirs(RESULTS_DIR, exist_ok=True)

META_DIR = os.path.join(BASE_OUT, f"atlite_cf_results_{RCP_LEVEL}", "meta")
os.makedirs(META_DIR, exist_ok=True)

SHAPEFILE_PATH = "/home/endata/PycharmProjects/PythonProject/Residuallast/ne_110m_admin_0_countries.shp"
CORINE_PATH = "/mnt/endata/MA_Lisa/atlite_cutouts/corine/10776/Results/u2018_clc2018_v2020_20u1_raster100m/DATA/corine.tif"

NATURA2000_EU_PATH = "/mnt/endata/MA_Lisa/atlite_cutouts/excluder_offshore/natura2000/eea_v_3035_100_k_natura2000_p_2023_v01_r00/SHP files/Natura2000_end2023_epsg3035.shp"
NATURA2000_GB_PATH = "/mnt/endata/MA_Lisa/atlite_cutouts/shapefile_exclusion_gb/GB-SAC-OSGB36-20250401/GB_SAC_OSGB36_20250401.shp"

# ---- Robustness switches ----
USE_CORINE = True
CACHE_CORINE_LOCALLY = True
CORINE_LOCAL_CACHE_PATH = "/tmp/corine_cached.tif"  # muss auf VM genügend Platz haben
FALLBACK_WITHOUT_CORINE_ON_ERROR = True

# Joblib: Threads -> kann CIFS/GDAL stressen. Prozesse sind oft stabiler.
JOBLIB_BACKEND = "loky"   # "threading" oder "loky"
N_JOBS = 8                # -1 geht auch, aber stabiler ist oft < cores


# =============================================================================
# Helper: generische Match-Bewertung
# =============================================================================
def _score_candidate(path: str, year: int, rcp_level: str, prefer_pypsa: bool = True) -> tuple:
    """
    Höherer Score = besser.
    sort(reverse=True) ergibt beste Kandidaten zuerst.
    """
    name = os.path.basename(path).lower()

    score_rcp = 1 if rcp_level.lower() in name else 0
    score_year = 1 if str(year) in name else 0
    score_pypsa = 1 if ("pypsa" in name and prefer_pypsa) else 0
    score_notagg = 1 if "notagg" in name else 0

    # lieber spezifischere Dateien als komplett generische
    score_len = len(name)

    return (score_rcp, score_year, score_pypsa, score_notagg, score_len)


# =============================================================================
# Helper: Cutout finden
# =============================================================================
def find_cutout_for_year(cutout_dir: str, year: int, rcp_level: str) -> Optional[str]:
    """
    Sucht bevorzugt Cutouts, die sowohl Jahr als auch RCP-Level enthalten.
    Fallback auf generische Jahres-Cutouts, falls nichts anderes existiert.
    """
    patterns = [
        os.path.join(cutout_dir, f"*{rcp_level}*{year}*.nc"),
        os.path.join(cutout_dir, f"*{year}*{rcp_level}*.nc"),
        os.path.join(cutout_dir, f"europe*{rcp_level}*{year}*.nc"),
        os.path.join(cutout_dir, f"europe*{year}*{rcp_level}*.nc"),
        os.path.join(cutout_dir, f"europe-{year}.nc"),
        os.path.join(cutout_dir, f"*{year}*.nc"),
    ]

    candidates = []
    seen = set()
    for patt in patterns:
        for m in sorted(glob.glob(patt)):
            if m not in seen:
                candidates.append(m)
                seen.add(m)

    if not candidates:
        return None

    ranked = sorted(
        candidates,
        key=lambda p: _score_candidate(p, year, rcp_level, prefer_pypsa=False),
        reverse=True,
    )

    if len(ranked) > 1:
        logger.info("Cutout-Kandidaten für %s/%s:", year, rcp_level)
        for r in ranked[:5]:
            logger.info("  %s", r)

    chosen = ranked[0]
    if rcp_level.lower() not in os.path.basename(chosen).lower():
        logger.warning(
            "Gewähltes Cutout für Jahr %s enthält %s nicht explizit im Dateinamen: %s",
            year, rcp_level, chosen
        )
    return chosen


# =============================================================================
# Helper: Techfile finden
# =============================================================================
def find_techfile_for_year(
    kind: str,
    year: int,
    rcp_level: str,
    turbine: Optional[str] = None
) -> Optional[str]:
    """
    Sucht Techfiles robust nach kind, year und rcp_level.
    Für Wind optional zusätzlich turbine im Pattern.
    """
    patterns: List[str] = []

    if kind == "wind":
        if turbine:
            patterns += [
                os.path.join(TECHFILES_DIR, f"wind*{turbine}*{rcp_level}*{year}*_notAgg_pypsa.nc"),
                os.path.join(TECHFILES_DIR, f"wind*{turbine}*{year}*{rcp_level}*_notAgg_pypsa.nc"),
                os.path.join(TECHFILES_DIR, f"wind*{turbine}*{rcp_level}*{year}*_notAgg.nc"),
                os.path.join(TECHFILES_DIR, f"wind*{turbine}*{year}*{rcp_level}*_notAgg.nc"),
                os.path.join(TECHFILES_DIR, f"wind*{turbine}*{rcp_level}*{year}*.nc"),
                os.path.join(TECHFILES_DIR, f"wind*{turbine}*{year}*{rcp_level}*.nc"),
            ]

        patterns += [
            os.path.join(TECHFILES_DIR, f"wind*{rcp_level}*{year}*_notAgg_pypsa.nc"),
            os.path.join(TECHFILES_DIR, f"wind*{year}*{rcp_level}*_notAgg_pypsa.nc"),
            os.path.join(TECHFILES_DIR, f"wind*{rcp_level}*{year}*_notAgg.nc"),
            os.path.join(TECHFILES_DIR, f"wind*{year}*{rcp_level}*_notAgg.nc"),
            os.path.join(TECHFILES_DIR, f"wind*{rcp_level}*{year}*.nc"),
            os.path.join(TECHFILES_DIR, f"wind*{year}*{rcp_level}*.nc"),
            os.path.join(TECHFILES_DIR, f"*wind*{rcp_level}*{year}*.nc"),
            os.path.join(TECHFILES_DIR, f"*wind*{year}*{rcp_level}*.nc"),
        ]

    elif kind == "solar":
        patterns = [
            os.path.join(TECHFILES_DIR, f"pv*{rcp_level}*{year}*_notAgg_pypsa.nc"),
            os.path.join(TECHFILES_DIR, f"pv*{year}*{rcp_level}*_notAgg_pypsa.nc"),
            os.path.join(TECHFILES_DIR, f"solar*{rcp_level}*{year}*_notAgg_pypsa.nc"),
            os.path.join(TECHFILES_DIR, f"solar*{year}*{rcp_level}*_notAgg_pypsa.nc"),

            os.path.join(TECHFILES_DIR, f"pv*{rcp_level}*{year}*_notAgg.nc"),
            os.path.join(TECHFILES_DIR, f"pv*{year}*{rcp_level}*_notAgg.nc"),
            os.path.join(TECHFILES_DIR, f"solar*{rcp_level}*{year}*_notAgg.nc"),
            os.path.join(TECHFILES_DIR, f"solar*{year}*{rcp_level}*_notAgg.nc"),

            os.path.join(TECHFILES_DIR, f"pv_cf*{rcp_level}*{year}*_pypsa.nc"),
            os.path.join(TECHFILES_DIR, f"pv_cf*{year}*{rcp_level}*_pypsa.nc"),
            os.path.join(TECHFILES_DIR, f"solar_cf*{rcp_level}*{year}*_pypsa.nc"),
            os.path.join(TECHFILES_DIR, f"solar_cf*{year}*{rcp_level}*_pypsa.nc"),

            os.path.join(TECHFILES_DIR, f"pv*{rcp_level}*{year}*.nc"),
            os.path.join(TECHFILES_DIR, f"pv*{year}*{rcp_level}*.nc"),
            os.path.join(TECHFILES_DIR, f"solar*{rcp_level}*{year}*.nc"),
            os.path.join(TECHFILES_DIR, f"solar*{year}*{rcp_level}*.nc"),

            os.path.join(TECHFILES_DIR, f"*pv*{rcp_level}*{year}*.nc"),
            os.path.join(TECHFILES_DIR, f"*pv*{year}*{rcp_level}*.nc"),
            os.path.join(TECHFILES_DIR, f"*solar*{rcp_level}*{year}*.nc"),
            os.path.join(TECHFILES_DIR, f"*solar*{year}*{rcp_level}*.nc"),
        ]
    else:
        raise ValueError(f"Unknown kind: {kind}")

    candidates = []
    seen = set()
    for patt in patterns:
        for m in sorted(glob.glob(patt)):
            if m not in seen:
                candidates.append(m)
                seen.add(m)

    if not candidates:
        return None

    ranked = sorted(
        candidates,
        key=lambda p: _score_candidate(p, year, rcp_level, prefer_pypsa=True),
        reverse=True,
    )

    chosen = ranked[0]
    if rcp_level.lower() not in os.path.basename(chosen).lower():
        logger.warning(
            "Gewähltes %s-Techfile für Jahr %s enthält %s nicht explizit im Dateinamen: %s",
            kind, year, rcp_level, chosen
        )
    return chosen


# =============================================================================
# Helper: CF DataArray laden
# =============================================================================
def load_cf_da(path: str, prefer_varnames: List[str]) -> xr.DataArray:
    ds = xr.open_dataset(path)

    var = None
    for name in prefer_varnames:
        if name in ds.data_vars:
            var = name
            break
    if var is None:
        var = list(ds.data_vars)[0]
        logger.warning("Keine prefer_varnames gefunden in %s. Nutze fallback var: %s", path, var)

    da = ds[var]

    if ("lat" in da.dims) and ("lon" in da.dims):
        da = da.rename({"lat": "y", "lon": "x"})
    if ("latitude" in da.dims) and ("longitude" in da.dims):
        da = da.rename({"latitude": "y", "longitude": "x"})

    for d in ["time", "y", "x"]:
        if d not in da.dims:
            raise ValueError(f"Erwarte dims (time,y,x). Habe {da.dims} aus {path}")

    y = da["y"].values
    if not np.all(np.diff(y) > 0) and not np.all(np.diff(y) < 0):
        order = np.argsort(y)
        if np.any(np.diff(y[order]) == 0):
            raise ValueError(f"y hat Duplikate, kann nicht stabil sortieren: {path}")
        da = da.isel(y=order)

    x = da["x"].values
    if not np.all(np.diff(x) > 0) and not np.all(np.diff(x) < 0):
        order = np.argsort(x)
        if np.any(np.diff(x[order]) == 0):
            raise ValueError(f"x hat Duplikate, kann nicht stabil sortieren: {path}")
        da = da.isel(x=order)

    return da


def align_da_to_cutout_grid(da: xr.DataArray, cutout: atlite.Cutout, label: str) -> xr.DataArray:
    ds_cut = cutout.data
    if "x" not in ds_cut.coords or "y" not in ds_cut.coords:
        raise ValueError("Cutout hat keine x/y coords in cutout.data; kann nicht alignen.")

    x_target = ds_cut["x"].values
    y_target = ds_cut["y"].values

    da2 = da.reindex(x=x_target, y=y_target)

    if np.any(~np.isfinite(da2.isel(time=0).values)) and (da.sizes["x"] == len(x_target)) and (da.sizes["y"] == len(y_target)):
        raise ValueError(
            f"{label}: CF-Grid passt nicht zum Cutout-Grid (Koordinaten mismatch). "
            f"da[x]={da.sizes['x']} da[y]={da.sizes['y']} vs cutout[x]={len(x_target)} cutout[y]={len(y_target)}. "
            f"Du brauchst dann Regridding/Interpolation."
        )
    return da2


# =============================================================================
# Helper: Shapes in Cutout-CRS bringen
# =============================================================================
def shapes_for_cutout(world_raw: gpd.GeoDataFrame, name_col: str, cutout: atlite.Cutout) -> gpd.GeoDataFrame:
    if world_raw.crs is None:
        logger.warning("Shapefile hat kein CRS. Setze EPSG:4326 als Default.")
        world_raw = world_raw.set_crs(4326)

    target_crs = 3035
    try:
        world = world_raw.to_crs(target_crs)
    except Exception as e:
        logger.warning("to_crs(%s) fehlgeschlagen (%s). Nutze unverändert.", target_crs, e)
        world = world_raw

    country_shapes = world.set_index(name_col)
    shapes_to_process = country_shapes[country_shapes.index.isin(COUNTRIES_TO_ANALYZE)]
    missing = [c for c in COUNTRIES_TO_ANALYZE if c not in shapes_to_process.index]
    if missing:
        logger.warning("Folgende Länder wurden im Shapefile nicht gefunden: %s", missing)
    return shapes_to_process


# =============================================================================
# Helper: CORINE ggf. lokal cachen
# =============================================================================
def prepare_corine_path() -> Optional[str]:
    if not USE_CORINE:
        logger.info("CORINE deaktiviert (USE_CORINE=False).")
        return None

    if not os.path.exists(CORINE_PATH):
        logger.warning("CORINE Raster nicht gefunden: %s", CORINE_PATH)
        return None

    if not CACHE_CORINE_LOCALLY:
        return CORINE_PATH

    try:
        src_size = os.path.getsize(CORINE_PATH)
    except OSError:
        logger.warning("Kann CORINE nicht statten: %s", CORINE_PATH)
        return CORINE_PATH

    if os.path.exists(CORINE_LOCAL_CACHE_PATH):
        try:
            dst_size = os.path.getsize(CORINE_LOCAL_CACHE_PATH)
        except OSError:
            dst_size = -1
        if dst_size == src_size and dst_size > 0:
            logger.info("Nutze vorhandenen lokalen CORINE-Cache: %s", CORINE_LOCAL_CACHE_PATH)
            return CORINE_LOCAL_CACHE_PATH

    logger.info("Kopiere CORINE lokal nach %s (Quelle: %s)", CORINE_LOCAL_CACHE_PATH, CORINE_PATH)
    shutil.copyfile(CORINE_PATH, CORINE_LOCAL_CACHE_PATH)
    return CORINE_LOCAL_CACHE_PATH


# =============================================================================
# Excluder Builder
# =============================================================================
def build_excluders(corine_path: Optional[str]) -> tuple[atlite.ExclusionContainer, Optional[atlite.ExclusionContainer], bool]:
    """
    returns: (excluder_eu, excluder_gb, corine_enabled)
    """
    excluder_eu = atlite.ExclusionContainer(crs=3035)
    corine_enabled = False

    if corine_path is not None:
        codes_to_exclude = list(range(111, 143)) + list(range(311, 314)) + list(range(511, 524))
        try:
            excluder_eu.add_raster(corine_path, codes=codes_to_exclude, nodata=0)
            corine_enabled = True
            logger.info("CORINE in Excluder eingebunden: %s", corine_path)
        except Exception as e:
            logger.warning("Konnte CORINE nicht als Raster hinzufügen (%s). Fahre ohne CORINE fort.", e)

    if os.path.exists(NATURA2000_EU_PATH):
        excluder_eu.add_geometry(NATURA2000_EU_PATH)
    else:
        logger.warning("EU Natura2000 Shapefile nicht gefunden: %s", NATURA2000_EU_PATH)

    excluder_gb = None
    if os.path.exists(NATURA2000_GB_PATH):
        excluder_gb = atlite.ExclusionContainer(crs=27700)
        excluder_gb.add_geometry(NATURA2000_GB_PATH)
    else:
        logger.warning("GB Natura2000 Shapefile nicht gefunden. UK läuft ohne GB-Excluder.")

    return excluder_eu, excluder_gb, corine_enabled


# =============================================================================
# Funktion zur Verarbeitung eines einzelnen Landes
# =============================================================================
def process_country(
    country_name: str,
    year: int,
    cutout: atlite.Cutout,
    shapes_to_process: gpd.GeoDataFrame,
    excluder_eu: atlite.ExclusionContainer,
    excluder_gb: Optional[atlite.ExclusionContainer],
    corine_enabled: bool,
    cf_solar_cells: xr.DataArray,
    cf_wind_cells: xr.DataArray,
):
    print(f"    --- Starte Verarbeitung für {country_name} in {year} ---")

    output_filename = f"uncorrected_cf_{country_name.replace(' ', '_')}_{year}_{RCP_LEVEL}.csv"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    if os.path.exists(output_path):
        print(f"      Datei existiert bereits, wird übersprungen: {output_filename}")
        return

    if country_name not in shapes_to_process.index:
        print(f"      WARNUNG: Land nicht im Shapefile gefunden: {country_name}. Überspringe.")
        return
    country_shape = shapes_to_process.loc[[country_name]]

    if country_name == "United Kingdom" and excluder_gb is not None:
        current_excluder = excluder_gb
    else:
        current_excluder = excluder_eu

    try:
        availability_mask = cutout.availabilitymatrix(country_shape, current_excluder).squeeze().fillna(0)
    except RasterioIOError as e:
        msg = f"RasterioIOError bei availabilitymatrix für {country_name} ({year}): {e}"
        if FALLBACK_WITHOUT_CORINE_ON_ERROR and corine_enabled:
            logger.warning("%s -> Fallback: ohne Excluder/CORINE", msg)
            availability_mask = cutout.availabilitymatrix(country_shape).squeeze().fillna(0)
        else:
            raise

    cf_exponent = COUNTRY_SPECIFIC_EXPONENTS.get(country_name, DEFAULT_EXPONENT)

    # --- SOLAR ---
    mean_cf_solar = cf_solar_cells.mean(dim="time")
    weights_solar = (mean_cf_solar ** cf_exponent) * availability_mask
    if float(weights_solar.sum()) > 0.0:
        cf_solar_country = xr.dot(cf_solar_cells, weights_solar, dims=["y", "x"]) / weights_solar.sum()
    else:
        cf_solar_country = xr.zeros_like(cf_solar_cells.isel(x=0, y=0))

    # --- WIND ---
    turbine_name = COUNTRY_TO_TURBINE_MAP.get(country_name, DEFAULT_TURBINE)
    print(f"      Wind-Techfile (CF) genutzt. Turbine-Mapping informativ: {turbine_name}")

    mean_cf_wind = cf_wind_cells.mean(dim="time")
    weights_wind = (mean_cf_wind ** cf_exponent) * availability_mask
    if float(weights_wind.sum()) > 0.0:
        cf_wind_country = xr.dot(cf_wind_cells, weights_wind, dims=["y", "x"]) / weights_wind.sum()
    else:
        cf_wind_country = xr.zeros_like(cf_wind_cells.isel(x=0, y=0))

    yearly_df = pd.DataFrame(
        {
            "solar_cf": cf_solar_country.to_series(),
            "wind_cf": cf_wind_country.to_series(),
        }
    )
    yearly_df.to_csv(output_path)
    print(f"      Ergebnisse für {country_name} ({year}) gespeichert: {output_path}")


# =============================================================================
# HAUPTSKRIPT
# =============================================================================
if __name__ == "__main__":
    print(f"Starte Onshore-CF-Berechnung für Modell={MODEL_NAME}, RCP={RCP_LEVEL}")
    print("Schritt 1: Lade Geometrien...")

    if not os.path.exists(SHAPEFILE_PATH):
        raise FileNotFoundError(f"SHAPEFILE nicht gefunden: {SHAPEFILE_PATH}")
    if not os.path.isdir(CUTOUT_DIR):
        raise NotADirectoryError(f"CUTOUT_DIR existiert nicht oder ist kein Ordner: {CUTOUT_DIR}")
    if not os.path.isdir(TECHFILES_DIR):
        raise NotADirectoryError(f"TECHFILES_DIR existiert nicht oder ist kein Ordner: {TECHFILES_DIR}")

    world_raw = gpd.read_file(SHAPEFILE_PATH)

    candidate_cols = ["ADMIN", "name", "NAME", "NAME_EN", "CNTR_NAME", "SOVEREIGNT", "COUNTRY", "Country"]
    name_col = next((c for c in candidate_cols if c in world_raw.columns), None)
    if name_col is None:
        raise KeyError(f"Keine Länder-Spalte gefunden. Spalten: {list(world_raw.columns)}")

    corine_path_to_use = prepare_corine_path()

    excluder_eu, excluder_gb, corine_enabled = build_excluders(corine_path_to_use)
    logger.info("corine_enabled=%s", corine_enabled)

    for year in YEARS_TO_ANALYZE:
        print(f"\n{'=' * 25} START VERARBEITUNG FÜR JAHR {year} ({RCP_LEVEL}) {'=' * 25}")

        cutout_path = find_cutout_for_year(CUTOUT_DIR, year, RCP_LEVEL)
        if cutout_path is None:
            print(f"    WARNUNG: Cutout für {year} mit {RCP_LEVEL} nicht gefunden in {CUTOUT_DIR}. Überspringe.")
            continue

        print(f"  Verwende Cutout (nur Maske/Grid): {cutout_path}")
        cutout = atlite.Cutout(cutout_path)

        shapes_to_process = shapes_for_cutout(world_raw, name_col, cutout)

        wind_path = find_techfile_for_year("wind", year, RCP_LEVEL, turbine=None)
        solar_path = find_techfile_for_year("solar", year, RCP_LEVEL, turbine=None)

        if wind_path is None:
            print(f"    WARNUNG: Wind-Techfile für {year} mit {RCP_LEVEL} nicht gefunden in {TECHFILES_DIR}. Überspringe Jahr.")
            del cutout
            gc.collect()
            continue
        if solar_path is None:
            print(f"    WARNUNG: Solar/PV-Techfile für {year} mit {RCP_LEVEL} nicht gefunden in {TECHFILES_DIR}. Überspringe Jahr.")
            del cutout
            gc.collect()
            continue

        print(f"  Wind-Techfile:  {wind_path}")
        print(f"  Solar-Techfile: {solar_path}")

        cf_wind_cells = load_cf_da(
            wind_path,
            prefer_varnames=["specific generation", "wind", "cf", "capacity_factor"]
        )
        cf_solar_cells = load_cf_da(
            solar_path,
            prefer_varnames=["specific generation", "pv", "solar", "cf", "capacity_factor"]
        )

        cf_wind_cells = align_da_to_cutout_grid(cf_wind_cells, cutout, label="WIND")
        cf_solar_cells = align_da_to_cutout_grid(cf_solar_cells, cutout, label="SOLAR")

        print(f"  Starte Aggregation für {len(COUNTRIES_TO_ANALYZE)} Länder... (backend={JOBLIB_BACKEND}, n_jobs={N_JOBS})")

        Parallel(n_jobs=N_JOBS, backend=JOBLIB_BACKEND)(
            delayed(process_country)(
                country,
                year,
                cutout,
                shapes_to_process,
                excluder_eu,
                excluder_gb,
                corine_enabled,
                cf_solar_cells,
                cf_wind_cells,
            )
            for country in COUNTRIES_TO_ANALYZE
        )

        del cf_solar_cells, cf_wind_cells, cutout, shapes_to_process
        gc.collect()

    print(f"\n{'=' * 25} ALLE JAHRE VERARBEITET {'=' * 25}")
    print(f"Modell: {MODEL_NAME}")
    print(f"RCP-Level: {RCP_LEVEL}")
    print(f"Ergebnisse liegen unter: {RESULTS_DIR}")