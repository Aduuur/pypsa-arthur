# calc offshore
from __future__ import annotations

import os
import glob
import gc
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union

# Optional: Bathymetrie-Rasterprüfung
_HAS_RASTERIO = False
try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False


# =============================================================================
# KONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- RCP-LEVEL ---
RCP_LEVEL = "rcp85"   # z. B. "rcp26", "rcp45", "rcp85"

# --- JAHRE UND LÄNDER ---
YEARS_TO_ANALYZE = range(2025, 2051)
COUNTRIES_TO_ANALYZE = [
    "Germany", "France", "Poland", "Austria", "Switzerland",
    "Italy", "Spain", "Belgium", "Netherlands", "Denmark", "Sweden",
    "Portugal", "Norway", "Finland", "Lithuania", "Latvia", "Estonia",
    "United Kingdom", "Czechia",
]

COUNTRY_NAME_TO_ISO3_MAP = {
    "Germany": "DEU",
    "United Kingdom": "GBR",
    "France": "FRA",
    "Netherlands": "NLD",
    "Denmark": "DNK",
    "Belgium": "BEL",
    "Italy": "ITA",
    "Spain": "ESP",
    "Poland": "POL",
    "Portugal": "PRT",
    "Austria": "AUT",
    "Switzerland": "CHE",
    "Finland": "FIN",
    "Lithuania": "LTU",
    "Latvia": "LVA",
    "Estonia": "EST",
    "Czechia": "CZE",
    "Sweden": "SWE",
    "Norway": "NOR",
}

# --- Techfiles (Wind CF Grids) ---
TECHFILES_DIR = "/mnt/endata/Cordex/Pypsa-Compatibility/results/europe/CNRM-CERFACS-CM5/techfiles"

# --- Outputs ---
BASE_OUT = "/mnt/endata/MA_Arthur"
BASE_RESULTS_DIR = os.path.join(
    BASE_OUT,
    f"atlite_cf_results_offshore_{RCP_LEVEL}",
    f"results_per_year_uncorrected_{RCP_LEVEL}",
)
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# GIS-Pfade
ONSHORE_SHAPEFILE_PATH = "/home/endata/PycharmProjects/PythonProject/Residuallast/ne_110m_admin_0_countries.shp"
OFFSHORE_SHAPEFILE_PATH = "/mnt/endata/MA_Lisa/atlite_cutouts/shapefile_maritim_for_offshore/World_EEZ_v12_20231025/eez_v12.shp"
GEBCO_BATHYMETRY_PATH = "/mnt/endata/MA_Lisa/atlite_cutouts/excluder_offshore/water_depth/gebco_2024_n72.0_s34.0_w-10.0_e40.0.tif"
WDPA_BASE_DIR = "/mnt/endata/MA_Lisa/atlite_cutouts/excluder_offshore/WDPA_WDOECM_Aug2025_Public_EU_shp/"

# --- PARAMETER ---
CF_EXPONENT = 2.0
MIN_COAST_DISTANCE_KM = 0
MAX_COAST_DISTANCE_KM = 120
MAX_WATER_DEPTH_METERS = -50  # beachte: GEBCO ist meist negativ (Tiefe)


# =============================================================================
# Helper: Techfile finden
# =============================================================================
def find_wind_cf_file_for_year(tech_dir: str, year: int, rcp_level: str) -> Optional[str]:
    """
    Sucht ein Wind-Techfile für ein bestimmtes Jahr und RCP-Level.

    Bevorzugte Suchreihenfolge:
      1) wind_*_{rcp_level}_*_{year}_notAgg_pypsa.nc
      2) wind_*_{rcp_level}_*_{year}_notAgg.nc
      3) wind_*_{rcp_level}_*{year}*.nc
      4) *wind*{rcp_level}*{year}*_notAgg_pypsa.nc
      5) *wind*{rcp_level}*{year}*_notAgg.nc
      6) *wind*{rcp_level}*{year}*.nc

    Sehr robuster Fallback:
      - irgendeine wind-Datei mit year und rcp_level im Namen
    """
    patterns = [
        os.path.join(tech_dir, f"wind_*_{rcp_level}_*_{year}_notAgg_pypsa.nc"),
        #os.path.join(tech_dir, f"wind_*_{rcp_level}_*_{year}_notAgg.nc"),
        #os.path.join(tech_dir, f"wind_*_{rcp_level}_*{year}*.nc"),
        os.path.join(tech_dir, f"*wind*{rcp_level}*{year}*_notAgg_pypsa.nc"),
        #os.path.join(tech_dir, f"*wind*{rcp_level}*{year}*_notAgg.nc"),
        #os.path.join(tech_dir, f"*wind*{rcp_level}*{year}*.nc"),
    ]

    for patt in patterns:
        matches = sorted(glob.glob(patt))
        if matches:
            logger.info(f"Gefunden für Jahr {year}, {rcp_level}: {matches[0]}")
            return matches[0]

    return None


# =============================================================================
# Helper: CF DataArray + lon/lat 1D extrahieren
# =============================================================================
def load_cf_grid(path: str) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """
    Gibt zurück:
      cf: DataArray mit dims (time, y, x) oder (time, lat, lon)
      lon_1d: (nx,)
      lat_1d: (ny,)
    """
    ds = xr.open_dataset(path)

    # Variable wählen:
    # - pypsa file: "specific generation" (bei dir CF-like)
    # - raw file: meist "wind"
    if "specific generation" in ds.data_vars:
        var = "specific generation"
    else:
        # nimm erste data_var als fallback
        var = list(ds.data_vars)[0]
    cf = ds[var]

    # Koordinaten bestimmen:
    if "lon" in ds.coords:
        lon = ds["lon"].values
    elif "x" in ds.coords:
        lon = ds["x"].values
    else:
        raise ValueError(f"Keine lon/x Koordinate in {path}. coords={list(ds.coords)}")

    if "y" in ds.coords:
        y_vals = np.asarray(ds["y"].values, dtype=float)
    else:
        raise ValueError(f"Keine y-Koordinate in {path}. coords={list(ds.coords)}")

    lat = None
    if "lat" in ds.coords:
        lat_vals = np.asarray(ds["lat"].values, dtype=float)
        if np.any(~np.isfinite(lat_vals)):
            lat = y_vals
        else:
            lat = lat_vals
    else:
        lat = y_vals

    # Dims ggf. vereinheitlichen auf (time, y, x)
    # Falls raw file (time, lat, lon): rename -> (time, y, x)
    dim_map = {}
    if "lat" in cf.dims and "lon" in cf.dims:
        dim_map["lat"] = "y"
        dim_map["lon"] = "x"
        cf = cf.rename(dim_map)

    # Sicherstellen, dass time,y,x existieren
    for d in ["time", "y", "x"]:
        if d not in cf.dims:
            raise ValueError(f"Erwarte dims (time,y,x). Habe {cf.dims} aus {path}")

    return cf, np.asarray(lon), np.asarray(lat)


# =============================================================================
# Helper: Raster-Maske aus Polygonen erzeugen (lon/lat grid)
# =============================================================================
def polygon_mask_on_lonlat_grid(
    polygon_gdf: gpd.GeoDataFrame,
    lon_1d: np.ndarray,
    lat_1d: np.ndarray,
) -> np.ndarray:
    """
    Erzeugt eine 2D Maske (ny,nx) auf einem regular lon/lat 1D grid.
    Voraussetzung: polygon_gdf ist in EPSG:4326.
    """
    if not _HAS_RASTERIO:
        raise RuntimeError("rasterio ist nicht verfügbar, kann keine Rastermaske bauen.")

    nx = len(lon_1d)
    ny = len(lat_1d)

    lon_sorted = np.all(np.diff(lon_1d) > 0)
    if not lon_sorted:
        raise ValueError("lon_1d ist nicht streng aufsteigend; bitte prüfen/regridden.")

    lat_inc = np.all(np.diff(lat_1d) > 0)
    lat_dec = np.all(np.diff(lat_1d) < 0)
    if not (lat_inc or lat_dec):
        raise ValueError("lat_1d ist nicht monoton; bitte prüfen/regridden.")

    if lat_inc:
        lat_for_raster = lat_1d[::-1]
        flip_lat = True
    else:
        lat_for_raster = lat_1d
        flip_lat = False

    xres = float(np.median(np.diff(lon_1d)))
    yres = float(abs(np.median(np.diff(lat_for_raster))))

    west = float(lon_1d.min())
    north = float(lat_for_raster.max())

    transform = from_origin(west, north, xres, yres)

    geoms = [mapping(geom) for geom in polygon_gdf.geometry if geom is not None]
    if not geoms:
        return np.zeros((ny, nx), dtype=np.uint8)

    mask = rasterize(
        geoms,
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=False,
    )

    if flip_lat:
        mask = mask[::-1, :]

    return mask


# =============================================================================
# Helper: Bathymetrie-Maske auf lon/lat grid (optional)
# =============================================================================
def bathymetry_mask_on_lonlat_grid(
    gebco_tif: str,
    lon_1d: np.ndarray,
    lat_1d: np.ndarray,
    max_depth_m: float,
) -> Optional[np.ndarray]:
    """
    Gibt 2D Maske (ny,nx) zurück: 1 = erlaubt, 0 = ausgeschlossen.

    GEBCO: Tiefe meist negativ.
    Erlaubt ist nur, was NICHT tiefer als max_depth_m ist.
    Beispiel:
      max_depth_m = -50
      -30  => erlaubt
      -200 => ausgeschlossen
    """
    if (not _HAS_RASTERIO) or (not os.path.exists(gebco_tif)):
        return None

    nx = len(lon_1d)
    ny = len(lat_1d)

    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    coords = list(zip(lon2d.ravel(), lat2d.ravel()))

    with rasterio.open(gebco_tif) as src:
        vals = np.array([v[0] for v in src.sample(coords)], dtype=float).reshape(ny, nx)

    allowed = (vals >= max_depth_m).astype(np.uint8)
    return allowed


# =============================================================================
# HAUPT
# =============================================================================
if __name__ == "__main__":

    print(f"Starte Offshore-CF-Berechnung für {RCP_LEVEL}")
    print("Schritt 1: Lade und filtere Geometrien...")

    if not os.path.exists(ONSHORE_SHAPEFILE_PATH):
        raise FileNotFoundError(f"ONSHORE_SHAPEFILE_PATH nicht gefunden: {ONSHORE_SHAPEFILE_PATH}")
    if not os.path.exists(OFFSHORE_SHAPEFILE_PATH):
        raise FileNotFoundError(f"OFFSHORE_SHAPEFILE_PATH nicht gefunden: {OFFSHORE_SHAPEFILE_PATH}")

    onshore_world = gpd.read_file(ONSHORE_SHAPEFILE_PATH)

    # Länder-Spalte robust erkennen
    candidate_cols = ["ADMIN", "name", "NAME", "NAME_EN", "CNTR_NAME", "SOVEREIGNT", "COUNTRY", "Country"]
    name_col = next((c for c in candidate_cols if c in onshore_world.columns), None)
    if name_col is None:
        raise KeyError(f"Keine Länder-Spalte gefunden. Spalten: {list(onshore_world.columns)}")

    onshore_shapes = onshore_world[onshore_world[name_col].isin(COUNTRIES_TO_ANALYZE)].set_index(name_col)

    offshore_world = gpd.read_file(OFFSHORE_SHAPEFILE_PATH)
    iso3_codes_to_analyze = [
        COUNTRY_NAME_TO_ISO3_MAP.get(name)
        for name in COUNTRIES_TO_ANALYZE
        if COUNTRY_NAME_TO_ISO3_MAP.get(name)
    ]
    if "ISO_TER1" not in offshore_world.columns:
        raise KeyError(f"EEZ Shapefile hat keine Spalte ISO_TER1. Spalten: {list(offshore_world.columns)}")
    offshore_shapes = offshore_world[offshore_world["ISO_TER1"].isin(iso3_codes_to_analyze)].set_index("ISO_TER1")

    # WDPA laden (optional)
    print("  Lade und verbinde die WDPA-Schutzgebietsdateien...")
    wdpa_parts = []
    wdpa_found = False
    for i in range(3):
        wdpa_poly_path = os.path.join(
            WDPA_BASE_DIR,
            f"WDPA_WDOECM_Aug2025_Public_EU_shp_{i}",
            "WDPA_WDOECM_Aug2025_Public_EU_shp-polygons.shp",
        )
        if os.path.exists(wdpa_poly_path):
            wdpa_parts.append(gpd.read_file(wdpa_poly_path))
            wdpa_found = True

    if wdpa_found:
        wdpa_full = pd.concat(wdpa_parts, ignore_index=True)
        if "MARINE" in wdpa_full.columns:
            wdpa_marine_only = wdpa_full[wdpa_full["MARINE"] == "2"].copy()
        else:
            wdpa_marine_only = wdpa_full.copy()
        print(f"    ...insgesamt {len(wdpa_marine_only)} (marine) Schutzgebiete gefunden.")
    else:
        wdpa_marine_only = None
        print("    WARNUNG: Keine WDPA-Dateien gefunden. Fahre ohne WDPA-Ausschluss fort.")

    if not os.path.isdir(TECHFILES_DIR):
        raise NotADirectoryError(f"TECHFILES_DIR existiert nicht oder ist kein Ordner: {TECHFILES_DIR}")

    if not _HAS_RASTERIO:
        print(
            "WARNUNG: rasterio nicht verfügbar -> keine Rasterisierung möglich. "
            "Bitte rasterio installieren, sonst kann nicht länderweise maskiert werden."
        )
        raise SystemExit(2)

    # --- Jahre ---
    for year in YEARS_TO_ANALYZE:
        print(f"\n{'=' * 25} START VERARBEITUNG FÜR JAHR {year} ({RCP_LEVEL}) {'=' * 25}")

        cf_path = find_wind_cf_file_for_year(TECHFILES_DIR, year, RCP_LEVEL)
        if cf_path is None:
            print(
                f"  WARNUNG: CF-Techfile für {year} mit RCP-Level {RCP_LEVEL} "
                f"nicht gefunden in {TECHFILES_DIR}. Überspringe."
            )
            continue

        print(f"  Verwende CF-Techfile: {cf_path}")
        cf_cells, lon_1d, lat_1d = load_cf_grid(cf_path)

        # Mittelwert pro Zelle (für weights)
        mean_cf = cf_cells.mean(dim="time")

        # Bathymetrie (optional)
        bathy_allowed = bathymetry_mask_on_lonlat_grid(
            GEBCO_BATHYMETRY_PATH,
            lon_1d,
            lat_1d,
            MAX_WATER_DEPTH_METERS
        )
        if bathy_allowed is None:
            print("  INFO: Bathymetrie-Maske nicht verfügbar (Raster fehlt oder rasterio fehlt). Fahre ohne fort.")

        # Länder-Schleife
        for country_name in COUNTRIES_TO_ANALYZE:
            print(f"    --- Aggregiere für {country_name} ---")

            if country_name not in COUNTRY_NAME_TO_ISO3_MAP:
                print(f"      WARNUNG: Kein ISO3 Mapping für {country_name}. Überspringe.")
                continue
            iso3 = COUNTRY_NAME_TO_ISO3_MAP[country_name]

            onshore_candidates = onshore_shapes[onshore_shapes.index == country_name]
            offshore_candidates = offshore_shapes[offshore_shapes.index == iso3]

            if len(onshore_candidates) == 0:
                print(f"      ❌ Keine Onshore-Geometrie für {country_name} gefunden!")
                continue
            if len(offshore_candidates) == 0:
                print(f"      ❌ Keine Offshore (EEZ) Geometrie für {iso3} gefunden!")
                continue

            # Dissolve falls mehrere Geometrien
            onshore_geom = (
                onshore_candidates.dissolve().geometry.iloc[0]
                if len(onshore_candidates) > 1
                else onshore_candidates.geometry.iloc[0]
            )
            offshore_geom = (
                offshore_candidates.dissolve().geometry.iloc[0]
                if len(offshore_candidates) > 1
                else offshore_candidates.geometry.iloc[0]
            )

            # In metrische Projektion für Buffer/Distance
            onshore_proj = gpd.GeoSeries([onshore_geom], crs=onshore_candidates.crs).to_crs(3035).iloc[0]
            offshore_proj = gpd.GeoSeries([offshore_geom], crs=offshore_candidates.crs).to_crs(3035).iloc[0]

            # Küstenabstandsband: (buffer(max) \ buffer(min))
            inner = onshore_proj.buffer(MIN_COAST_DISTANCE_KM * 1000)
            outer = onshore_proj.buffer(MAX_COAST_DISTANCE_KM * 1000)
            band = outer.difference(inner)

            # Offshore-Zone = EEZ ∩ Band
            offshore_zone_proj = offshore_proj.intersection(band)
            if offshore_zone_proj.is_empty:
                print(f"      WARNUNG: Offshore-Zone (EEZ ∩ Küstenband) ist leer für {country_name}.")
                continue

            # WDPA ausschließen (optional)
            if wdpa_marine_only is not None:
                try:
                    wdpa_gdf = gpd.GeoDataFrame(wdpa_marine_only, geometry="geometry", crs=wdpa_marine_only.crs)
                    wdpa_proj = wdpa_gdf.to_crs(3035)
                    wdpa_union = unary_union([g for g in wdpa_proj.geometry if g is not None])
                    if wdpa_union and not wdpa_union.is_empty:
                        offshore_zone_proj = offshore_zone_proj.difference(wdpa_union)
                except Exception as e:
                    print(f"      WARNUNG: WDPA Ausschluss fehlgeschlagen ({country_name}): {e}")

            if offshore_zone_proj.is_empty:
                print(f"      WARNUNG: Offshore-Zone nach WDPA-Ausschluss leer für {country_name}.")
                continue

            # Für Rasterisierung zurück nach EPSG:4326
            offshore_zone_ll = gpd.GeoDataFrame(geometry=[offshore_zone_proj], crs=3035).to_crs(4326)

            # Maske auf Grid
            mask = polygon_mask_on_lonlat_grid(offshore_zone_ll, lon_1d, lat_1d).astype(float)

            # Bathy anwenden (optional)
            if bathy_allowed is not None:
                mask = mask * bathy_allowed

            # weights
            mean_cf_np = mean_cf.values
            weights = np.power(mean_cf_np, CF_EXPONENT) * mask

            wsum = float(np.nansum(weights))
            if wsum <= 0.0:
                print(f"      WARNUNG: Keine verfügbare Offshore-Fläche (weights sum=0) für {country_name}.")
                continue

            # Länderzeitreihe
            cf_np = cf_cells.values  # (T,ny,nx)
            num = np.nansum(cf_np * weights[None, :, :], axis=(1, 2))
            cf_country = num / wsum

            # Ausgabe
            country_folder_path = os.path.join(BASE_RESULTS_DIR, country_name.replace(" ", "_"))
            os.makedirs(country_folder_path, exist_ok=True)

            out = pd.DataFrame(
                {"wind_offshore_cf": cf_country},
                index=pd.to_datetime(cf_cells["time"].values),
            )
            out.index.name = "time"

            output_filename = f"offshore_wind_cf_{year}_{RCP_LEVEL}.csv"
            output_path = os.path.join(country_folder_path, output_filename)
            out.to_csv(output_path)
            print(f"      Ergebnis gespeichert: {output_path}")

        # Speicher freigeben nach jedem Jahr
        del cf_cells, mean_cf
        gc.collect()

    print(f"\n{'=' * 25} ALLE JAHRE VERARBEITET {'=' * 25}")
    print(f"RCP-Level: {RCP_LEVEL}")
    print(f"Ergebnisse liegen unter: {BASE_RESULTS_DIR}")