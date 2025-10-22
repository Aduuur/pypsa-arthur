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

    all_shares = pd.DataFrame()
    tech_list=['Nuclear', 'Coal', 'CCGT', 'Biomass']
    for tech in tech_list:
        # Filter by tech and sum capacity
        shares = (
            jrc.loc[jrc['type_g'] == tech]
            .groupby(['bus', 'cooling_type'])
            .sum(numeric_only=True)['capacity_g']
        )

        # Normalize cooling type names
        shares = shares.rename_axis(index=['bus', 'cooling_type']).rename({
            'Mechanical Draught Tower': 'closed-loop',
            'Natural Draught Tower': 'closed-loop',
            'Air Cooling': 'dry-cooling',
            'No Cooling': 'dry-cooling',
            'Once-through': 'once-through'
        })

        # Convert to DataFrame
        shares = shares.reset_index().rename(columns={'capacity_g': f'capacity_g_{tech}'})

        # Compute total capacity per bus
        total_cap = (
            shares.groupby('bus')[f'capacity_g_{tech}']
            .sum()
            .rename(f'total_cap_{tech}')
            .reset_index()
        )

        # Merge totals
        shares = shares.merge(total_cap, on='bus', how='left')

        # Compute shares
        shares[f'share_{tech}'] = shares[f'capacity_g_{tech}'] / shares[f'total_cap_{tech}']

        # Combine all techs
        if all_shares.empty:
            all_shares = shares
        else:
            all_shares = pd.merge(
                all_shares, shares,
                on=['bus', 'cooling_type'],
                how='outer'
            )

    # Fill missing values
    all_shares = all_shares.fillna(0)

    logger.info(f'Created cooling type share per bus for {tech_list}')
    all_shares.to_csv(snakemake.output.pp_ct_share, index=False)

