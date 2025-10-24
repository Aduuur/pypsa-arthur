"""
Calculate the share of each cooling type per bus for each technology.
Ensuring all expected cooling types exist for each bus and carrier.
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import logging
import itertools

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake("build_powerplants_cooling_type_share")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Inputs 
    pps_type = snakemake.params.pps_type
    logger.info(f"Split cooling type for {pps_type}")

    jrc_list = pd.read_csv(snakemake.input.jrc_list)
    regions = gpd.read_file(snakemake.input.regions_onshore)
    all_buses=regions.name.unique()

    # Location of each powerplant in jrc
    geometry = [Point(xy) for xy in zip(jrc_list["lon"], jrc_list["lat"])]
    jrc_gdf = gpd.GeoDataFrame(jrc_list, geometry=geometry, crs="EPSG:4326")

    # Spatial join: assign each plant to region (bus)
    jrc = gpd.sjoin(jrc_gdf, regions[["name", "geometry"]], predicate="within", how="left")
    jrc = jrc.rename(columns={"name": "bus"})

    # Mapping
    tech_mapping = {
        "nuclear": "Nuclear",
        "CCGT": "Fossil Gas",
        "lignite": "Fossil Brown coal/Lignite",
        "coal": "Fossil Hard coal",
        "biomass": "Biomass",
        "H2": "H2"
    }
    ct_mapping = {
        "Mechanical Draught Tower": "closed-loop",
        "Natural Draught Tower": "closed-loop",
        "Air Cooling": "dry-cooling",
        "No Cooling": "dry-cooling",
        "Once-through": "once-through"
    }

    all_shares = pd.DataFrame()
    for tech in pps_type:
        logger.info(f"Processing {tech}")

        # Aggregate per bus, carrier, and cooling_type
        capacities = (
            jrc.loc[jrc["type_g"] == tech_mapping[tech]]
            .assign(carrier=tech)  # add carrier now
            .replace({"cooling_type": ct_mapping})
            .groupby(["bus", "carrier", "cooling_type"], as_index=False)
            .sum(numeric_only=True)[["bus", "carrier", "cooling_type", "capacity_g"]]
        )
        capacities = capacities.rename(columns={"capacity_g": "capacity"})

        # Expected cooling types. Nuclear has no dry-cooling
        if tech == "nuclear":
            ct_exp_index = pd.Index(["closed-loop", "once-through"])
        else:
            ct_exp_index = pd.Index(["closed-loop", "once-through", "dry-cooling"])

        # Compute total capacity per cooling type (for dummy shares)
        cap_by_ct = capacities.groupby("cooling_type").sum(numeric_only=True)
        cap_by_ct = cap_by_ct.reindex(ct_exp_index, fill_value=0)
        share_ct = (cap_by_ct["capacity"] / cap_by_ct["capacity"].sum()).rename("share_ct")


        # Compute total capacity per bus+carrier
        total_cap = (
            capacities.groupby(["bus", "carrier"])["capacity"]
            .sum()
            .rename("total_capacity")
            .reset_index()
        )
        capacities = capacities.merge(total_cap, on=["bus", "carrier"], how="left")

        # Compute share per cooling type
        capacities["share"] = capacities["capacity"] / capacities["total_capacity"]
        capacities["carrier"] = tech

        shares = capacities[["bus", "carrier", "cooling_type", "capacity", "total_capacity", "share"]]
        shares_safe=shares.copy()
        ####
        # Ensuring all expected cooling types exist for each bus for carrier/tech
        
        for bus in all_buses:
            bus_tech_rows=shares.loc[(shares['bus'] == bus) & (shares['carrier'] == tech)].copy()
   
            existing_ct=bus_tech_rows.cooling_type.unique()
            missing_ct = [ct for ct in ct_exp_index if ct not in existing_ct]

            bus_tech_rows['val_origin']='jrc'
            all_shares=pd.concat([all_shares,bus_tech_rows], axis=0)
            
            
            new_row=bus_tech_rows.copy()
            
            for ct in missing_ct:
                if len(missing_ct) < len(ct_exp_index):
                    new_row.cooling_type = ct
                    new_row.capacity=0
                    new_row.share=0
                    new_row.val_origin='bus_zero'
                    all_shares=pd.concat([all_shares,new_row], axis=0)
                else:
                    new_row = pd.DataFrame({
                    "bus": [bus],
                    "carrier": [tech],
                    "cooling_type": [ct],
                    "capacity": [0],
                    "total_capacity": [0],
                    "share": [share_ct[ct]],
                    "val_origin": ["all_missing"]
                })
                    all_shares=pd.concat([all_shares,new_row], axis=0)
                    

    if not all_shares.empty:
        all_shares["bus_tech"] = all_shares["bus"] + " " + all_shares["carrier"]
        all_shares = all_shares.set_index("bus_tech")
        logger.info("Created cooling type share per bus (with completeness check)")
    else:
        logger.info("No powerplant split by cooling type. Saving empty DataFrame.")

    all_shares.to_csv(snakemake.output.pp_ct_share)