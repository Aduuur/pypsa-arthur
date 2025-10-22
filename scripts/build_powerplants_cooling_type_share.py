'''
Calculate the share of each cooling type per bus for each technology.
'''
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import logging

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_powerplants_cooling_type_share")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # List of powerplants with cooling type
    jrc_list = pd.read_csv(snakemake.input.jrc_list)

    # buses
    regions = gpd.read_file(snakemake.input.regions_onshore)

    # Create point geometries (lon = x, lat = y)
    geometry = [Point(xy) for xy in zip(jrc_list['lon'], jrc_list['lat'])]

    # Convert to GeoDataFrame
    jrc_gdf = gpd.GeoDataFrame(jrc_list, geometry=geometry, crs="EPSG:4326")

    # Spatial join: assign each plant to a bus
    jrc = gpd.sjoin(jrc_gdf, regions[['name', 'geometry']], predicate="within", how="left")
    jrc = jrc.rename(columns={'name': 'bus'})

    # Mapping: original tech -> standardized tech
    tech_mapping = {
        'Nuclear':'nuclear',
        'Fossil Gas':'CCGT',
        'Fossil Brown coal/Lignite':'lignite',
        'Fossil Hard coal':'coal',
        'Biomass':'biomass',
    }

    all_shares = pd.DataFrame()
    for original_tech, std_tech in tech_mapping.items():
        # Filter and sum capacity per bus + cooling_type
        
        shares = (
            jrc.loc[jrc['type_g'] == original_tech]
            .groupby(['bus', 'cooling_type'])
            .sum(numeric_only=True)['capacity_g']
        )

        # Normalize cooling types
        shares = shares.rename_axis(index=['bus', 'cooling_type']).rename({
            'Mechanical Draught Tower': 'closed-loop',
            'Natural Draught Tower': 'closed-loop',
            'Air Cooling': 'dry-cooling',
            'No Cooling': 'dry-cooling',
            'Once-through': 'once-through'
        })

        # Convert to DataFrame
        shares = shares.reset_index().rename(columns={'capacity_g': 'capacity'})

        # Compute total capacity per bus
        total_cap = (
            shares.groupby('bus')['capacity']
            .sum()
            .rename('total_capacity')
            .reset_index()
        )

        # Merge totals
        shares = shares.merge(total_cap, on='bus', how='left')

        # Compute share
        shares['share'] = shares['capacity'] / shares['total_capacity']

        # Add standardized tech column
        shares['carrier'] = std_tech

        # Keep only needed columns
        shares = shares[['bus', 'carrier', 'cooling_type', 'capacity', 'total_capacity', 'share']]
        # Combine with all_shares
        all_shares = pd.concat([all_shares, shares], ignore_index=True)

    # # Create combined index: bus + tech
    all_shares['bus_tech'] = all_shares['bus'] + ' ' + all_shares['carrier']
    all_shares = all_shares.set_index('bus_tech')

    logger.info(f'Created cooling type share per bus')
    all_shares.to_csv(snakemake.output.pp_ct_share)

