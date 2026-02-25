#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse der Speicher (Batterie & H₂) - FINALE VERSION
=====================================================

Kombiniert die Logik beider Skripte für eine präzise
und verständliche Visualisierung der Speicheroperationen.

Logik:
- Batterie: Lade-/Entladeleistung wird am elektrischen Bus gemessen.
- H₂-System:
  - H₂-Erzeugung (positiv): Elektrische Leistung, die in Elektrolyseure fließt.
  - H₂-Rückverstromung (negativ): Elektrische Leistung, die aus Brennstoffzellen/Turbinen kommt.
  - Füllstand: Energieinhalt des H₂-Speichers.
- Erstellt für jedes Jahr einen separaten Plot.
"""

import os
import re
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config_final import PlottingConfig


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def extract_country(bus_name):
    """Länderkürzel aus Busname."""
    return bus_name[:2]


def plot_country_storage(n, country, year, save_dir):
    """Erstellt den kombinierten Speicherplot für ein Land und Jahr."""

    # === 1. Batteriespeicher ===
    # Ladeleistung wird am elektrischen Bus gemessen (p0 > 0)
    charge_links = n.links.index[
        n.links.carrier.str.contains("charger", case=False) & n.links.bus0.str.startswith(country)]
    batt_charge = n.links_t.p0[charge_links].sum(axis=1) if not charge_links.empty else pd.Series(0, index=n.snapshots)

    # Entladeleistung wird am elektrischen Bus gemessen (p0 > 0, daher -p0 für negative Darstellung)
    discharge_links = n.links.index[
        n.links.carrier.str.contains("discharger", case=False) & n.links.bus0.str.startswith(country)]
    batt_discharge = n.links_t.p0[discharge_links].sum(axis=1) if not discharge_links.empty else pd.Series(0,
                                                                                                           index=n.snapshots)

    batt_stores = n.stores.index[
        n.stores.carrier.str.contains("battery", case=False) & n.stores.bus.str.startswith(country)]
    batt_soc = n.stores_t.e[batt_stores].sum(axis=1) if not batt_stores.empty else pd.Series(0, index=n.snapshots)
    batt_e_nom = n.stores.loc[batt_stores, "e_nom_opt"].sum()
    batt_soc_norm = batt_soc / batt_e_nom if batt_e_nom > 1e-3 else pd.Series(0, index=batt_soc.index)

    # === 2. H₂-System ===
    # H₂-Erzeugung: Strom, der in Elektrolyseure fließt (p0 > 0)
    electrolysis_links = n.links.index[n.links.carrier.isin(["H2 Electrolysis"]) & n.links.bus0.str.startswith(country)]
    h2_prod_power = n.links_t.p0[electrolysis_links].sum(axis=1) if not electrolysis_links.empty else pd.Series(0,
                                                                                                                index=n.snapshots)

    # H₂-Rückverstromung: Strom, der aus Brennstoffzellen/Turbinen kommt (p1 < 0)
    fuelcell_links = n.links.index[
        n.links.carrier.isin(["H2 Fuel Cell", "H2 turbine"]) & n.links.bus1.str.startswith(country)]
    h2_to_power_generation = -n.links_t.p1[fuelcell_links].sum(axis=1) if not fuelcell_links.empty else pd.Series(0,
                                                                                                                  index=n.snapshots)

    # Füllstand des H₂-Speichers
    h2_stores = n.stores.index[n.stores.carrier.isin(["H2 Store"]) & n.stores.bus.str.startswith(country)]
    h2_soc = n.stores_t.e[h2_stores].sum(axis=1) if not h2_stores.empty else pd.Series(0, index=n.snapshots)
    h2_e_nom = n.stores.loc[h2_stores, "e_nom_opt"].sum()
    h2_soc_norm = h2_soc / h2_e_nom if h2_e_nom > 1e-3 else pd.Series(0, index=h2_soc.index)

    # === 3. Tagesmittel für den Plot ===
    daily_batt_charge = batt_charge.resample("D").mean()
    daily_batt_discharge = batt_discharge.resample("D").mean()
    daily_batt_soc_norm = batt_soc_norm.resample("D").mean()

    daily_h2_prod_power = h2_prod_power.resample("D").mean()
    daily_h2_to_power_gen = h2_to_power_generation.resample("D").mean()
    daily_h2_soc_norm = h2_soc_norm.resample("D").mean()

    # === 4. Plotting ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Speicheranalyse für {country} im Jahr {year}", fontsize=16)

    # Plot Batterie
    ax1_pri = axes[0]
    p1, = ax1_pri.plot(daily_batt_charge.index, daily_batt_charge, color="orange", label="Laden (+)")
    p2, = ax1_pri.plot(daily_batt_discharge.index, -daily_batt_discharge, color="steelblue", label="Entladen (–)")
    ax1_pri.set_ylabel("Leistung [MW]")
    ax1_pri.set_title("Batteriespeicher – Lade-/Entladeflüsse & Füllstand")
    ax1_pri.grid(True, alpha=0.3)
    ax1_pri.axhline(0, color='black', linestyle='-', linewidth=0.5)

    ax1_sec = ax1_pri.twinx()
    p3 = ax1_sec.fill_between(daily_batt_soc_norm.index, 0, daily_batt_soc_norm, color="grey", alpha=0.2,
                              label="Füllstand")
    ax1_sec.set_ylabel("Füllstand (0–1)")
    ax1_sec.set_ylim(0, 1)
    ax1_pri.legend(handles=[p1, p2, p3], loc="upper right", frameon=False)

    # Plot H₂-System
    ax2_pri = axes[1]
    p4, = ax2_pri.plot(daily_h2_prod_power.index, daily_h2_prod_power, color="orange",
                       label="Strom für H₂-Erzeugung (+)")
    p5, = ax2_pri.plot(daily_h2_to_power_gen.index, -daily_h2_to_power_gen, color="steelblue",
                       label="Strom aus H₂-Rückverstromung (–)")
    ax2_pri.set_ylabel("Leistung [MW]")
    ax2_pri.set_title("H₂-System – Stromflüsse & Speicherfüllstand")
    ax2_pri.grid(True, alpha=0.3)
    ax2_pri.axhline(0, color='black', linestyle='-', linewidth=0.5)

    ax2_sec = ax2_pri.twinx()
    p6 = ax2_sec.fill_between(daily_h2_soc_norm.index, 0, daily_h2_soc_norm, color="grey", alpha=0.2,
                              label="H₂-Füllstand")
    ax2_sec.set_ylabel("Füllstand (0–1)")
    ax2_sec.set_ylim(0, 1)
    ax2_pri.legend(handles=[p4, p5, p6], loc="upper right", frameon=False)

    axes[-1].set_xlabel("Datum")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{country}_{year}_storage_flows.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Speicherplot gespeichert: {path}")


# ------------------------------------------------------------
# Hauptablauf
# ------------------------------------------------------------
def main():
    config = PlottingConfig()
    networks = config.get_networks()

    for path in networks:
        if not os.path.isfile(path):
            continue

        try:
            year = int(re.search(r"_(\d{4})\.nc$", path).group(1))
            n = pypsa.Network(path)
            print(f"\n📂 Analysiere {year}: {os.path.basename(path)}")
        except Exception as e:
            print(f"Fehler beim Laden von {path}: {e}")
            continue

        save_dir = os.path.join(config.BASE_SAVE_PATH,
                                os.path.basename(os.path.dirname(os.path.dirname(path))),
                                "storage_analysis")

        # --- Datenauswertung ---
        battery_stores = n.stores[n.stores.carrier.str.contains("battery", case=False)].copy()
        h2_stores = n.stores[n.stores.carrier.str.contains("H2", case=False)].copy()

        battery_stores.loc[:, "country"] = battery_stores["bus"].apply(extract_country)
        h2_stores.loc[:, "country"] = h2_stores["bus"].apply(extract_country)

        storage_country = pd.concat([
            battery_stores.groupby("country")["e_nom_opt"].sum().rename("battery_MWh"),
            h2_stores.groupby("country")["e_nom_opt"].sum().rename("H2_MWh")
        ], axis=1).fillna(0)

        electrolyser_links = n.links[n.links.carrier == "H2 Electrolysis"].copy()
        electrolyser_links.loc[:, "country"] = electrolyser_links["bus0"].str[:2]
        electrolyser_country = electrolyser_links.groupby("country")["p_nom_opt"].sum().rename("electrolyser_MW")

        combined = pd.concat([storage_country, electrolyser_country], axis=1).fillna(0)

        print("\n=== Übersicht: Speicher- und Elektrolyse-Kapazitäten pro Land ===")
        print(combined.round(1))

        # --- Plots (nur DE) ---
        if "DE" in combined.index:
            plot_country_storage(n, "DE", year, save_dir)


if __name__ == "__main__":
    main()