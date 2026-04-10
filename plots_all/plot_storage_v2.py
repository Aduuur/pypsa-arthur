#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse der Speicher (Batterie & Wasser)
=========================================
Visualisiert die Speicheroperationen für Batteriespeicher und Wasserspeicher.

Funktioniert sowohl im normalen (myopischen) Modus als auch im ARO-Modus.
Ausgabe in config.PLOT_OUTPUT_PATH/storage_analysis/.

ARO-Fix: Jahr-Parsing defensiv – wenn kein YYYY im Pfad (z.B. Dispatch-Netz),
wird das Jahr aus n.snapshots[0].year gelesen.
"""

import os
import re
import pypsa
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# FIX: war config_final, PlottingConfig ist in master_config definiert
from master_config import PlottingConfig


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def _parse_year(n: pypsa.Network, path: str, fallback: int = 2050) -> int:
    """
    Extrahiert das Jahr aus dem Netzwerk.
    Priorität:
      1. n.snapshots[0].year  (immer korrekt, auch bei Dispatch-Netzen)
      2. Regex auf Dateinamen  (Fallback)
      3. fallback-Wert
    """
    try:
        return int(n.snapshots[0].year)
    except Exception:
        pass
    m = re.search(r"___(\d{4})\.nc$", path) or re.search(r"_(\d{4})\.nc", path)
    return int(m.group(1)) if m else fallback


def plot_country_storage(n, country, year, save_dir):
    """Erstellt den kombinierten Speicherplot für ein Land und Jahr."""

    # === 1. Batteriespeicher ===
    charge_links = n.links.index[
        n.links.carrier.str.contains("charger", case=False) &
        n.links.bus0.str.startswith(country)
    ]
    batt_charge = (
        n.links_t.p0[charge_links].sum(axis=1)
        if not charge_links.empty
        else pd.Series(0, index=n.snapshots)
    )

    discharge_links = n.links.index[
        n.links.carrier.str.contains("discharger", case=False) &
        n.links.bus0.str.startswith(country)
    ]
    batt_discharge = (
        n.links_t.p0[discharge_links].sum(axis=1)
        if not discharge_links.empty
        else pd.Series(0, index=n.snapshots)
    )

    batt_stores = n.stores.index[
        n.stores.carrier.str.contains("battery", case=False) &
        n.stores.bus.str.startswith(country)
    ]
    batt_soc = (
        n.stores_t.e[batt_stores].sum(axis=1)
        if not batt_stores.empty
        else pd.Series(0, index=n.snapshots)
    )
    batt_e_nom = n.stores.loc[batt_stores, "e_nom_opt"].sum() if not batt_stores.empty else 0.0
    batt_soc_norm = (
        batt_soc / batt_e_nom
        if batt_e_nom > 1e-3
        else pd.Series(0, index=batt_soc.index)
    )

    # === 2. Wasserspeicher (PHS + Hydro) ===
    hydro_gen  = pd.Series(0.0, index=n.snapshots)
    hydro_pump = pd.Series(0.0, index=n.snapshots)
    hydro_soc  = pd.Series(0.0, index=n.snapshots)
    hydro_e_nom = 0.0

    phs_su = n.storage_units.index[
        (n.storage_units.carrier == "PHS") &
        n.storage_units.bus.str.startswith(country)
    ]
    hydro_su = n.storage_units.index[
        (n.storage_units.carrier == "hydro") &
        n.storage_units.bus.str.startswith(country)
    ]

    if not phs_su.empty:
        phs_dispatch = n.storage_units_t.p_dispatch[phs_su].sum(axis=1)
        hydro_gen += phs_dispatch.clip(lower=0)

        if "p_store" in n.storage_units_t:
            hydro_pump += n.storage_units_t.p_store[phs_su].sum(axis=1).clip(lower=0)
        else:
            hydro_pump += -phs_dispatch.clip(upper=0)

        hydro_soc += n.storage_units_t.state_of_charge[phs_su].sum(axis=1)

        if "e_nom_opt" in n.storage_units.columns:
            hydro_e_nom += n.storage_units.loc[phs_su, "e_nom_opt"].sum()
        else:
            hydro_e_nom += (
                n.storage_units.loc[phs_su, "p_nom"] *
                n.storage_units.loc[phs_su, "max_hours"]
            ).sum()

    if not hydro_su.empty:
        hydro_gen += n.storage_units_t.p_dispatch[hydro_su].clip(lower=0).sum(axis=1)
        hydro_soc += n.storage_units_t.state_of_charge[hydro_su].sum(axis=1)

        if "e_nom_opt" in n.storage_units.columns:
            hydro_e_nom += n.storage_units.loc[hydro_su, "e_nom_opt"].sum()
        else:
            hydro_e_nom += (
                n.storage_units.loc[hydro_su, "p_nom"] *
                n.storage_units.loc[hydro_su, "max_hours"]
            ).sum()

    if not phs_su.empty:
        for su in phs_su:
            pump_links = n.links.index[
                (n.links.bus1 == n.storage_units.loc[su, "bus"]) &
                n.links.carrier.str.contains("PHS", case=False, na=False)
            ]
            if not pump_links.empty:
                hydro_pump += n.links_t.p1[pump_links].sum(axis=1).clip(lower=0)

    hydro_stores = n.stores.index[
        n.stores.carrier.isin(["PHS", "hydro"]) &
        n.stores.bus.str.startswith(country)
    ]
    if not hydro_stores.empty:
        hydro_soc    += n.stores_t.e[hydro_stores].sum(axis=1)
        hydro_e_nom  += n.stores.loc[hydro_stores, "e_nom_opt"].sum()

    hydro_soc_norm = (
        hydro_soc / hydro_e_nom
        if hydro_e_nom > 1e-3
        else pd.Series(0, index=hydro_soc.index)
    )

    print(f"  PHS: {len(phs_su)} Einheiten, Hydro: {len(hydro_su)} Einheiten")
    print(f"  Pumpen max: {hydro_pump.max():.1f} MW, Turbinieren max: {hydro_gen.max():.1f} MW")

    # === 3. Tagesmittel ===
    daily_batt_charge    = batt_charge.resample("D").mean()
    daily_batt_discharge = batt_discharge.resample("D").mean()
    daily_batt_soc_norm  = batt_soc_norm.resample("D").mean()

    daily_hydro_pump     = hydro_pump.resample("D").mean()
    daily_hydro_gen      = hydro_gen.resample("D").mean()
    daily_hydro_soc_norm = hydro_soc_norm.resample("D").mean()

    # === 4. Plot ===
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 14, "figure.titlesize": 22,
    })

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Speicheranalyse für {country} im Jahr {year}", fontsize=22)

    ax1 = axes[0]
    p1, = ax1.plot(daily_batt_charge.index,    daily_batt_charge,    color="orange",    label="Laden (+)",    linewidth=2)
    p2, = ax1.plot(daily_batt_discharge.index, -daily_batt_discharge, color="steelblue", label="Entladen (–)", linewidth=2)
    ax1.set_ylabel("Leistung [MW]", fontsize=16)
    ax1.set_title("Batteriespeicher – Lade-/Entladeflüsse & Füllstand", fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1_sec = ax1.twinx()
    p3 = ax1_sec.fill_between(daily_batt_soc_norm.index, 0, daily_batt_soc_norm, color="grey", alpha=0.2, label="Füllstand")
    ax1_sec.set_ylabel("Füllstand (0–1)", fontsize=16)
    ax1_sec.set_ylim(0, 1)
    ax1.legend(handles=[p1, p2, p3], loc="upper right", frameon=False, fontsize=14)

    ax2 = axes[1]
    p4, = ax2.plot(daily_hydro_pump.index, daily_hydro_pump,  color="orange",    label="Pumpen (+)",       linewidth=2)
    p5, = ax2.plot(daily_hydro_gen.index,  -daily_hydro_gen,  color="steelblue", label="Turbinieren (–)",  linewidth=2)
    ax2.set_ylabel("Leistung [MW]", fontsize=16)
    ax2.set_title("Wasserspeicher (PHS & Hydro) – Stromflüsse & Füllstand", fontsize=18)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2_sec = ax2.twinx()
    p6 = ax2_sec.fill_between(daily_hydro_soc_norm.index, 0, daily_hydro_soc_norm, color="grey", alpha=0.2, label="Füllstand")
    ax2_sec.set_ylabel("Füllstand (0–1)", fontsize=16)
    ax2_sec.set_ylim(0, 1)
    ax2.legend(handles=[p4, p5, p6], loc="upper right", frameon=False, fontsize=14)

    axes[-1].set_xlabel("Datum", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    path_out = os.path.join(save_dir, f"{country}_{year}_storage_flows.png")
    plt.savefig(path_out, dpi=300)
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"✅ Speicherplot gespeichert: {path_out}")


def main():
    config   = PlottingConfig()
    networks = config.get_networks()

    if isinstance(networks, dict):
        if config.SCENARIO_SELECTION == "both":
            networks = list(networks.values())[0]
        else:
            networks = networks.get(config.SCENARIO_SELECTION, list(networks.values())[0])

    # FIX: PLOT_OUTPUT_PATH statt BASE_SAVE_PATH + hardcodierter Pfad
    save_dir = os.path.join(config.PLOT_OUTPUT_PATH, "storage_analysis")

    countries = config.get_countries()
    if "ALL" in countries:
        countries = [c for c in countries if c != "ALL"]

    for path in networks:
        if not os.path.isfile(path):
            continue
        try:
            n    = pypsa.Network(path)
            # FIX: Jahr immer aus Snapshots lesen, nicht nur aus Dateinamen
            year = _parse_year(n, path, fallback=2050)
            print(f"\n📂 Analysiere {year}: {os.path.basename(path)}")
        except Exception as e:
            print(f"Fehler beim Laden von {path}: {e}")
            continue

        for country in countries:
            try:
                plot_country_storage(n, country, year, save_dir)
            except Exception as e:
                print(f"  Fehler {country}: {e}")


if __name__ == "__main__":
    main()