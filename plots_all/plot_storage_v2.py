#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse der Speicher (Batterie & Wasser) - FINALE VERSION
=========================================================

Visualisiert die Speicheroperationen für Batteriespeicher und Wasserspeicher.

Logik:
- Oben: Batteriespeicher (Laden/Entladen/Füllstand)
- Unten: Wasserspeicher (Pumpspeicher + Talsperren)
  - Laden: Pumpen (nur PHS)
  - Entladen: Turbinieren (PHS + Hydro)
  - Füllstand: Aggregierte Energie
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
    # Ladeleistung
    charge_links = n.links.index[
        n.links.carrier.str.contains("charger", case=False) & n.links.bus0.str.startswith(country)]
    batt_charge = n.links_t.p0[charge_links].sum(axis=1) if not charge_links.empty else pd.Series(0, index=n.snapshots)

    # Entladeleistung
    discharge_links = n.links.index[
        n.links.carrier.str.contains("discharger", case=False) & n.links.bus0.str.startswith(country)]
    batt_discharge = n.links_t.p0[discharge_links].sum(axis=1) if not discharge_links.empty else pd.Series(0,
                                                                                                           index=n.snapshots)

    # Füllstand
    batt_stores = n.stores.index[
        n.stores.carrier.str.contains("battery", case=False) & n.stores.bus.str.startswith(country)]
    batt_soc = n.stores_t.e[batt_stores].sum(axis=1) if not batt_stores.empty else pd.Series(0, index=n.snapshots)
    batt_e_nom = n.stores.loc[batt_stores, "e_nom_opt"].sum()
    batt_soc_norm = batt_soc / batt_e_nom if batt_e_nom > 1e-3 else pd.Series(0, index=batt_soc.index)

    # === 2. Wasserspeicher (PHS + Hydro) ===
    hydro_gen = pd.Series(0.0, index=n.snapshots)
    hydro_pump = pd.Series(0.0, index=n.snapshots)
    hydro_soc = pd.Series(0.0, index=n.snapshots)
    hydro_e_nom = 0.0

    # a) StorageUnits (PHS + Hydro)
    phs_su = n.storage_units.index[
        (n.storage_units.carrier == "PHS") & n.storage_units.bus.str.startswith(country)]

    hydro_su = n.storage_units.index[
        (n.storage_units.carrier == "hydro") & n.storage_units.bus.str.startswith(country)]

    # PHS: Turbinieren aus p_dispatch, Pumpen aus p_store
    if not phs_su.empty:
        # Turbinieren (Entladen): p_dispatch positiv
        phs_dispatch = n.storage_units_t.p_dispatch[phs_su].sum(axis=1)
        hydro_gen += phs_dispatch.clip(lower=0)

        # Pumpen (Laden): p_store (falls vorhanden)
        if 'p_store' in n.storage_units_t:
            hydro_pump += n.storage_units_t.p_store[phs_su].sum(axis=1).clip(lower=0)
        else:
            # Fallback: negative p_dispatch als Pumpen
            hydro_pump += -phs_dispatch.clip(upper=0)

        # Füllstand
        hydro_soc += n.storage_units_t.state_of_charge[phs_su].sum(axis=1)

        # Kapazität
        if "e_nom_opt" in n.storage_units.columns:
            hydro_e_nom += n.storage_units.loc[phs_su, "e_nom_opt"].sum()
        else:
            hydro_e_nom += (n.storage_units.loc[phs_su, "p_nom"] * n.storage_units.loc[phs_su, "max_hours"]).sum()

    # Hydro: nur Turbinieren
    if not hydro_su.empty:
        hydro_gen += n.storage_units_t.p_dispatch[hydro_su].clip(lower=0).sum(axis=1)
        hydro_soc += n.storage_units_t.state_of_charge[hydro_su].sum(axis=1)

        if "e_nom_opt" in n.storage_units.columns:
            hydro_e_nom += n.storage_units.loc[hydro_su, "e_nom_opt"].sum()
        else:
            hydro_e_nom += (n.storage_units.loc[hydro_su, "p_nom"] * n.storage_units.loc[hydro_su, "max_hours"]).sum()

    # b) PHS Pumpen über Links (wichtig!)
    # In PyPSA-Eur werden PHS-Pumpen oft als separate Links modelliert
    # Suche nach Links die zu PHS-StorageUnits gehören
    if not phs_su.empty:
        for su in phs_su:
            # Finde Links, die zu diesem StorageUnit führen (Pumpen)
            pump_links = n.links.index[
                (n.links.bus1 == n.storage_units.loc[su, "bus"]) &
                (n.links.carrier.str.contains("PHS", case=False, na=False))
                ]
            if not pump_links.empty:
                # p1 ist die Leistung am bus1 (Speicher) - positiv = hinein pumpen
                hydro_pump += n.links_t.p1[pump_links].sum(axis=1).clip(lower=0)

    # c) Stores (falls verwendet)
    hydro_stores = n.stores.index[
        n.stores.carrier.isin(["PHS", "hydro"]) & n.stores.bus.str.startswith(country)]
    if not hydro_stores.empty:
        hydro_soc += n.stores_t.e[hydro_stores].sum(axis=1)
        hydro_e_nom += n.stores.loc[hydro_stores, "e_nom_opt"].sum()

    hydro_soc_norm = hydro_soc / hydro_e_nom if hydro_e_nom > 1e-3 else pd.Series(0, index=hydro_soc.index)

    # Debug-Output (vereinfacht)
    print(f"  PHS: {len(phs_su)} Einheiten, Hydro: {len(hydro_su)} Einheiten")
    print(f"  Pumpen max: {hydro_pump.max():.1f} MW, Turbinieren max: {hydro_gen.max():.1f} MW")

    # === 3. Tagesmittel für den Plot ===
    daily_batt_charge = batt_charge.resample("D").mean()
    daily_batt_discharge = batt_discharge.resample("D").mean()
    daily_batt_soc_norm = batt_soc_norm.resample("D").mean()

    daily_hydro_pump = hydro_pump.resample("D").mean()
    daily_hydro_gen = hydro_gen.resample("D").mean()
    daily_hydro_soc_norm = hydro_soc_norm.resample("D").mean()

    # === 4. Plotting mit größeren Schriften ===
    # Setze global Schriftgrößen
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 22
    })

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    # Titel hier anpassen, wenn gewünscht (z.B. "Speicherbewirtschaftung in DE (2050)")
    fig.suptitle(f"Speicheranalyse für {country} im Jahr {year}", fontsize=22)

    # Plot Batterie
    ax1_pri = axes[0]
    p1, = ax1_pri.plot(daily_batt_charge.index, daily_batt_charge, color="orange", label="Laden (+)", linewidth=2)
    p2, = ax1_pri.plot(daily_batt_discharge.index, -daily_batt_discharge, color="steelblue", label="Entladen (–)",
                       linewidth=2)
    ax1_pri.set_ylabel("Leistung [MW]", fontsize=16)
    ax1_pri.set_title("Batteriespeicher – Lade-/Entladeflüsse & Füllstand", fontsize=18)
    ax1_pri.grid(True, alpha=0.3)
    ax1_pri.axhline(0, color='black', linestyle='-', linewidth=0.5)

    ax1_sec = ax1_pri.twinx()
    p3 = ax1_sec.fill_between(daily_batt_soc_norm.index, 0, daily_batt_soc_norm, color="grey", alpha=0.2,
                              label="Füllstand")
    ax1_sec.set_ylabel("Füllstand (0–1)", fontsize=16)
    ax1_sec.set_ylim(0, 1)
    ax1_pri.legend(handles=[p1, p2, p3], loc="upper right", frameon=False, fontsize=14)

    # Plot Wasser (PHS/Hydro)
    ax2_pri = axes[1]
    p4, = ax2_pri.plot(daily_hydro_pump.index, daily_hydro_pump, color="orange",
                       label="Pumpen (+)", linewidth=2)
    p5, = ax2_pri.plot(daily_hydro_gen.index, -daily_hydro_gen, color="steelblue",
                       label="Turbinieren (–)", linewidth=2)
    ax2_pri.set_ylabel("Leistung [MW]", fontsize=16)
    ax2_pri.set_title("Wasserspeicher (PHS & Hydro) – Stromflüsse & Füllstand", fontsize=18)
    ax2_pri.grid(True, alpha=0.3)
    ax2_pri.axhline(0, color='black', linestyle='-', linewidth=0.5)

    ax2_sec = ax2_pri.twinx()
    p6 = ax2_sec.fill_between(daily_hydro_soc_norm.index, 0, daily_hydro_soc_norm, color="grey", alpha=0.2,
                              label="Füllstand")
    ax2_sec.set_ylabel("Füllstand (0–1)", fontsize=16)
    ax2_sec.set_ylim(0, 1)
    ax2_pri.legend(handles=[p4, p5, p6], loc="upper right", frameon=False, fontsize=14)

    axes[-1].set_xlabel("Datum", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{country}_{year}_storage_flows.png")
    plt.savefig(path, dpi=300)
    plt.close()

    # Reset font settings (wichtig, falls das Skript als Modul genutzt wird)
    plt.rcParams.update(plt.rcParamsDefault)

    print(f"✅ Speicherplot gespeichert: {path}")


# ------------------------------------------------------------
# Hauptablauf
# ------------------------------------------------------------
def main():
    config = PlottingConfig()
    networks = config.get_networks()

    # Falls Dictionary, nimm das ausgewählte Szenario
    if isinstance(networks, dict):
        if config.SCENARIO_SELECTION == "both":
            # Nimm das erste Szenario (z.B. average oder dunkelflaute)
            networks = list(networks.values())[0]
        else:
            networks = networks[config.SCENARIO_SELECTION]

    for path in networks:
        if not os.path.isfile(path):
            continue

        try:
            year = int(re.search(r"_(\d{4})\.nc$", path).group(1))

            # Alle Jahre 2025, 2030-2050 plotten
            if year not in [2025, 2030, 2035, 2040, 2045, 2050]:
                continue

            n = pypsa.Network(path)
            print(f"\n📂 Analysiere {year}: {os.path.basename(path)}")
        except Exception as e:
            print(f"Fehler beim Laden von {path}: {e}")
            continue

        save_dir = os.path.join(config.BASE_SAVE_PATH,
                                os.path.basename(os.path.dirname(os.path.dirname(path))),
                                "plots_storage_analysis")

        # Länder aus Config
        countries = config.get_countries()
        if "ALL" in countries:
            countries.remove("ALL")  # Speicher macht für "ALL" weniger Sinn in Detailansicht

        for country in countries:
            plot_country_storage(n, country, year, save_dir)


if __name__ == "__main__":
    main()