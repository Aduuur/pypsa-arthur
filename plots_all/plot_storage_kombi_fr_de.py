#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse der Speicher (Batterie DE & Wasser FR) - KOMBI-VERSION
==============================================================

Erstellt einen kombinierten Plot:
- Oben: Batteriespeicher in DE (Kurzzeitspeicher)
- Unten: Wasserspeicher in FR (Saisonaler Speicher)

Zweck: Visualisierung der Arbeitsteilung im europäischen Verbund.
"""

import os
import re
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config_final import PlottingConfig, fill_leap_day


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def _extract_year(n: pypsa.Network, path: str) -> int:
    """
    Extrahiert das Jahr robust:
    1. Aus n.snapshots (sicherste Methode, funktioniert auch bei ARO-Dispatches)
    2. Fallback: Regex auf den Dateinamen (fuer klassische Pfade wie _2050.nc)
    """
    try:
        year = int(n.snapshots[0].year)
        return year
    except Exception:
        pass
    m = re.search(r"_(\d{4})[\._ ]", os.path.basename(path))
    if m:
        return int(m.group(1))
    print(f"\u26a0\ufe0f  Kein Jahr im Dateinamen erkannt: {path}")
    return None


def get_battery_data(n, country):
    """Holt Batterie-Daten (Laden, Entladen, SOC) fuer ein Land."""
    charge_links = n.links.index[
        n.links.carrier.str.contains("charger", case=False) & n.links.bus0.str.startswith(country)]
    charge = n.links_t.p0[charge_links].sum(axis=1) if not charge_links.empty else pd.Series(0, index=n.snapshots)

    discharge_links = n.links.index[
        n.links.carrier.str.contains("discharger", case=False) & n.links.bus0.str.startswith(country)]
    discharge = n.links_t.p0[discharge_links].sum(axis=1) if not discharge_links.empty else pd.Series(0, index=n.snapshots)

    stores = n.stores.index[
        n.stores.carrier.str.contains("battery", case=False) & n.stores.bus.str.startswith(country)]
    soc = n.stores_t.e[stores].sum(axis=1) if not stores.empty else pd.Series(0, index=n.snapshots)
    e_nom = n.stores.loc[stores, "e_nom_opt"].sum()

    soc_norm = soc / e_nom if e_nom > 1e-3 else pd.Series(0, index=soc.index)

    return charge, discharge, soc_norm


def get_hydro_data(n, country):
    """Holt Wasserspeicher-Daten (Pumpen, Turbinieren, SOC) fuer ein Land."""
    gen = pd.Series(0.0, index=n.snapshots)
    pump = pd.Series(0.0, index=n.snapshots)
    soc = pd.Series(0.0, index=n.snapshots)
    e_nom = 0.0

    su_idx = n.storage_units.index[
        n.storage_units.carrier.isin(["PHS", "hydro"]) & n.storage_units.bus.str.startswith(country)]

    if not su_idx.empty:
        gen += n.storage_units_t.p_dispatch[su_idx].clip(lower=0).sum(axis=1)
        if 'p_store' in n.storage_units_t:
            pump += n.storage_units_t.p_store[su_idx].sum(axis=1).clip(lower=0)
        else:
            pump += -n.storage_units_t.p_dispatch[su_idx].clip(upper=0).sum(axis=1)

        soc += n.storage_units_t.state_of_charge[su_idx].sum(axis=1)

        if "e_nom_opt" in n.storage_units.columns:
            e_nom += n.storage_units.loc[su_idx, "e_nom_opt"].sum()
        else:
            e_nom += (n.storage_units.loc[su_idx, "p_nom"] * n.storage_units.loc[su_idx, "max_hours"]).sum()

    if not su_idx.empty:
        phs_buses = n.storage_units.loc[su_idx, "bus"].unique()
        pump_links = n.links.index[
            n.links.bus1.isin(phs_buses) &
            n.links.carrier.str.contains("PHS", case=False)
        ]
        if not pump_links.empty:
            pump += n.links_t.p1[pump_links].sum(axis=1).clip(lower=0)

    store_idx = n.stores.index[
        n.stores.carrier.isin(["PHS", "hydro"]) & n.stores.bus.str.startswith(country)]

    if not store_idx.empty:
        soc += n.stores_t.e[store_idx].sum(axis=1)
        e_nom += n.stores.loc[store_idx, "e_nom_opt"].sum()

    soc_norm = soc / e_nom if e_nom > 1e-3 else pd.Series(0, index=soc.index)

    return pump, gen, soc_norm


def plot_combined_storage(n, year, save_dir):
    """
    Erstellt den Kombi-Plot:
    Oben: DE Batterien
    Unten: FR Wasserkraft
    """
    de_charge, de_discharge, de_soc = get_battery_data(n, "DE")
    fr_pump, fr_gen, fr_soc = get_hydro_data(n, "FR")

    de_charge = fill_leap_day(de_charge.to_frame()).iloc[:,0]
    de_discharge = fill_leap_day(de_discharge.to_frame()).iloc[:,0]
    de_soc = fill_leap_day(de_soc.to_frame()).iloc[:,0]
    fr_pump = fill_leap_day(fr_pump.to_frame()).iloc[:,0]
    fr_gen = fill_leap_day(fr_gen.to_frame()).iloc[:,0]
    fr_soc = fill_leap_day(fr_soc.to_frame()).iloc[:,0]
    resample = "1D"
    de_charge = de_charge.resample(resample).mean()
    de_discharge = de_discharge.resample(resample).mean()
    de_soc = de_soc.resample(resample).mean()

    fr_pump = fr_pump.resample(resample).mean()
    fr_gen = fr_gen.resample(resample).mean()
    fr_soc = fr_soc.resample(resample).mean()

    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Speicherbewirtschaftung im Vergleich (Jahr {year})", fontsize=20, y=0.96)

    ax1 = axes[0]
    ax1_r = ax1.twinx()
    l1, = ax1.plot(de_charge.index, de_charge, color="#ace37f", label="Laden (+)", linewidth=1.5)
    l2, = ax1.plot(de_discharge.index, -de_discharge, color="#5f8c3f", label="Entladen (-)", linewidth=1.5)
    l3 = ax1_r.fill_between(de_soc.index, 0, de_soc, color="gray", alpha=0.2, label="Fuellstand (SoC)")
    ax1.set_ylabel("Leistung [MW]", fontsize=14)
    ax1_r.set_ylabel("Fuellstand [0-1]", fontsize=14)
    ax1.set_title("Kurzzeitspeicherung: Batterien in Deutschland", fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="black", linewidth=0.8)
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=12, framealpha=0.9)
    ax1_r.set_ylim(0, 1.1)

    ax2 = axes[1]
    ax2_r = ax2.twinx()
    l4, = ax2.plot(fr_pump.index, fr_pump, color="#00ced1", label="Pumpen (+)", linewidth=1.5)
    l5, = ax2.plot(fr_gen.index, -fr_gen, color="#007f82", label="Turbinieren (-)", linewidth=1.5)
    l6 = ax2_r.fill_between(fr_soc.index, 0, fr_soc, color="gray", alpha=0.2, label="Fuellstand (SoC)")
    ax2.set_ylabel("Leistung [MW]", fontsize=14)
    ax2_r.set_ylabel("Fuellstand [0-1]", fontsize=14)
    ax2.set_title("Saisonale Speicherung: Wasserkraft (Reservoir & PHS) in Frankreich", fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="black", linewidth=0.8)
    lines2 = [l4, l5, l6]
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc="upper right", fontsize=12, framealpha=0.9)
    ax2_r.set_ylim(0, 1.1)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.set_xlabel("Monat", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = f"storage_combined_DE_FR_{year}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"\u2705 Kombi-Plot gespeichert: {filepath}")


# ------------------------------------------------------------
# MAIN  (standalone + ARO-kompatibel via run_analysis.py)
# ------------------------------------------------------------
def main(aro_network=None):
    """
    Parameters
    ----------
    aro_network : str or None
        Im ARO-Modus: Pfad zum Worst-Case-Dispatch-Netzwerk.
        Wird von run_analysis.py gesetzt; None = klassischer Standalone-Modus.
    """
    config = PlottingConfig()

    # --- ARO-Modus: direktes Netzwerk uebergeben ---
    if aro_network is not None:
        path = aro_network
        if not os.path.isfile(path):
            print(f"\u26a0\ufe0f  ARO-Netzwerk nicht gefunden: {path}")
            return
        try:
            n = pypsa.Network(path)
            year = _extract_year(n, path)
            if year is None:
                return

            save_dir = os.path.join(
                config.BASE_SAVE_PATH,
                "plots_storage_analysis"
            )
            os.makedirs(save_dir, exist_ok=True)
            plot_combined_storage(n, year, save_dir)
        except Exception as e:
            print(f"Fehler: {e}")
        return

    # --- Klassischer Standalone-Modus ---
    networks = config.get_networks()

    if isinstance(networks, dict):
        if config.SCENARIO_SELECTION == "both":
            networks = list(networks.values())[0]
        else:
            networks = networks[config.SCENARIO_SELECTION]

    for path in networks:
        if not os.path.isfile(path):
            continue
        try:
            n = pypsa.Network(path)
            year = _extract_year(n, path)
            if year is None:
                continue
            if year != 2050:
                continue

            print(f"\n\U0001f4c2 Analysiere {year} fuer Kombi-Plot...")

            save_dir = os.path.join(
                config.BASE_SAVE_PATH,
                os.path.basename(os.path.dirname(os.path.dirname(path))),
                "plots_storage_analysis"
            )
            os.makedirs(save_dir, exist_ok=True)
            plot_combined_storage(n, year, save_dir)

        except Exception as e:
            print(f"Fehler: {e}")


if __name__ == "__main__":
    main()
