"""
Calculate the share of each cooling type per bus for each technology.
Ensuring all expected cooling types exist for each bus and carrier.
"""
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

    # Inputs 
    pps_type = snakemake.params.pps_type
    logger.info(f"Split cooling type for {pps_type}")

    jrc_list = pd.read_csv(snakemake.input.jrc_list)
    regions = gpd.read_file(snakemake.input.regions_onshore)

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

        ####
        # Ensuring all expected cooling types exist for each bus and carrier
        all_rows = []
        for (bus, carrier), group in shares.groupby(["bus", "carrier"], sort=False):
            existing_ct = set(group["cooling_type"])
            missing_ct = [ct for ct in ct_exp_index if ct not in existing_ct]

            if not missing_ct:
                # Case 0: all expected types present
                group = group.copy()
                group["status"] = "original"

            elif len(existing_ct) == 0:
                # Case 2: no cooling type at all
                new_rows = pd.DataFrame({
                    "bus": bus,
                    "carrier": carrier,
                    "cooling_type": ct_exp_index,
                    "capacity": 0,
                    "total_capacity": 0,
                    "share": share_ct.values,
                    "status": "all_missing"
                })
                group = new_rows

            else:
                # Case 1: some cooling types missing
                total_capacity = group["total_capacity"].iloc[0]
                new_rows = pd.DataFrame({
                    "bus": bus,
                    "carrier": carrier,
                    "cooling_type": missing_ct,
                    "capacity": 0,
                    "total_capacity": total_capacity,
                    "share": 0,
                    "status": "partial_replace"
                })
                group = pd.concat([group.assign(status="original"), new_rows], ignore_index=True)

            all_rows.append(group)

        shares_full = pd.concat(all_rows, ignore_index=True)
        shares_full["cooling_type"] = pd.Categorical(
            shares_full["cooling_type"],
            categories=ct_exp_index,
            ordered=True
        )
        shares_full = shares_full.sort_values(
            ["bus", "carrier", "cooling_type"]
        ).reset_index(drop=True)

        all_shares = pd.concat([all_shares, shares_full], ignore_index=True)

    if not all_shares.empty:
        all_shares["bus_tech"] = all_shares["bus"] + " " + all_shares["carrier"]
        all_shares = all_shares.set_index("bus_tech")
        logger.info("Created cooling type share per bus (with completeness check)")
    else:
        logger.info("No powerplant split by cooling type. Saving empty DataFrame.")

    all_shares.to_csv(snakemake.output.pp_ct_share)
