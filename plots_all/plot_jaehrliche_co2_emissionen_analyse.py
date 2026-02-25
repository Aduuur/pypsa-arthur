#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse: CO2-Infrastruktur Komplett (Quellen, Capture, Storage)
===============================================================
Kombiniert zwei Analysen in einem Skript:
1. Emissions-Quellen: Wer verbrennt Fossilien? (Brutto)
   -> Detail-Plot für Deutschland (Kuchendiagramm)
2. CO2-Infrastruktur: Wer fängt ein (Capture) und wer speichert (Storage)?
   -> Infrastruktur-Plot (Spiegel-Balken)
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pypsa
from config_final import PlottingConfig

# Zieljahr
TARGET_YEAR = 2050

# --- KONFIGURATION ---
EMITTING_CARRIERS = ["coal", "lignite", "oil", "gas", "CCGT", "OCGT", "waste"]


def get_country(bus_name):
    return str(bus_name)[:2]


# =============================================================================
# TEIL 1: BRUTTO-EMISSIONEN (QUELLEN)
# =============================================================================
def analyze_gross_emissions(n):
    print("... Analysiere Emissions-Quellen (Verbrennung) ...")
    weight = n.snapshot_weightings.generators
    data = []

    # 1. Generatoren (Strom)
    mask_gen = n.generators.carrier.str.contains("|".join(EMITTING_CARRIERS), case=False, na=False)
    mask_gen &= ~n.generators.carrier.str.contains("biomass", case=False)  # Biomasse ist hier neutral

    gens = n.generators[mask_gen]

    for carrier in gens.carrier.unique():
        co2_fac = n.carriers.at[carrier, "co2_emissions"] if carrier in n.carriers.index else 0
        if pd.isna(co2_fac) or co2_fac == 0:
            c_low = carrier.lower()
            if "lignite" in c_low:
                co2_fac = 0.406
            elif "coal" in c_low:
                co2_fac = 0.34
            elif "oil" in c_low:
                co2_fac = 0.28
            elif "gas" in c_low or "ccgt" in c_low:
                co2_fac = 0.202

        if co2_fac > 0:
            idx = gens[gens.carrier == carrier].index
            eff = n.generators.loc[idx, "efficiency"].replace(0, 1.0)
            p_el = n.generators_t.p[idx].multiply(weight, axis=0).sum(axis=0)
            emissions = (p_el / eff) * co2_fac

            for bus_name, val in emissions.items():
                if val > 0.01:
                    data.append({
                        "Country": get_country(n.generators.at[bus_name, "bus"]),
                        "Carrier": carrier,
                        "Type": "Stromerzeugung",
                        "Amount": val / 1e6  # Mt
                    })

    # 2. Links (Wärme/Boiler)
    mask_link = n.links.carrier.str.contains("boiler", case=False) & \
                n.links.carrier.str.contains("gas|oil", case=False)
    links = n.links[mask_link]

    for carrier in links.carrier.unique():
        co2_fac = 0.202 if "gas" in carrier.lower() else 0.28
        idx = links[links.carrier == carrier].index
        p_in = n.links_t.p0[idx].multiply(weight, axis=0).sum(axis=0)
        emissions = p_in * co2_fac

        for link_name, val in emissions.items():
            if val > 0.01:
                bus = n.links.at[link_name, "bus0"]
                data.append({
                    "Country": get_country(bus),
                    "Carrier": carrier,
                    "Type": "Wärmeerzeugung",
                    "Amount": val / 1e6  # Mt
                })

    return pd.DataFrame(data)


def plot_de_breakdown(df_gross, config, network_name):
    """Erstellt ein Kuchendiagramm für DE Emissionen."""
    df_de = df_gross[df_gross["Country"] == "DE"]
    if df_de.empty: return

    def map_cat(row):
        c = row["Carrier"].lower()
        if "rural" in c: return "Dezentral (Ländlich)"
        if "urban decentral" in c: return "Dezentral (Städtisch)"
        if "urban central" in c: return "Fernwärme"
        if "ccgt" in c or "ocgt" in c: return "Strom (Kraftwerke)"
        if "oil" in c: return "Öl-Kessel"
        return "Industrie/Sonstiges"

    df_de["Category"] = df_de.apply(map_cat, axis=1)
    stats = df_de.groupby("Category")["Amount"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(stats)))

    wedges, texts, autotexts = ax.pie(stats, labels=stats.index, autopct='%1.1f%%',
                                      startangle=90, colors=colors, pctdistance=0.85)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    total = stats.sum()
    ax.text(0, 0, f"Gesamt\n{total:.1f} Mt", ha='center', va='center', fontsize=12, fontweight='bold')

    ax.set_title(f"Quellen der CO₂-Emissionen in Deutschland ({TARGET_YEAR})", fontsize=14)

    plt.tight_layout()
    save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "emissions")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "de_emissions_breakdown.png"), dpi=300)
    print(f"✅ Plot gespeichert: de_emissions_breakdown.png")


# =====================================================================
# TEIL 2: CAPTURE & STORAGE (INFRASTRUKTUR)
# =====================================================================
def analyze_capture_storage(n):
    """Analysiert Capture und Storage."""
    print("... Analysiere Capture & Storage ...")
    weight = n.snapshot_weightings.generators
    cap_data = []
    store_data = []

    # --- CAPTURE ---
    mask_cc = n.links.carrier.str.contains("CC|DAC", case=False, regex=True)
    mask_cc &= ~n.links.carrier.str.contains("CCGT", case=False)  # Keine Kraftwerke

    candidates = n.links[mask_cc]

    for idx, row in candidates.iterrows():
        # Finde CO2 Output Port
        co2_port = None
        for col in ["bus0", "bus1", "bus2", "bus3", "bus4"]:
            if col in n.links.columns:
                if "co2" in str(row[col]).lower() and "atm" not in str(row[col]).lower():
                    co2_port = col
                    break

        if co2_port:
            port_num = co2_port.replace("bus", "p")
            if port_num in n.links_t:
                flow = n.links_t[port_num][idx].multiply(weight, axis=0).sum()
            else:
                flow = 0

            amount = abs(flow) / 1e6
            if amount > 0.01:
                tech_group = "Industrie/Process"
                if "DAC" in row.carrier:
                    tech_group = "Direct Air Capture (DAC)"
                elif "bio" in row.carrier.lower():
                    tech_group = "Biomasse-CCS (BECCS)"
                elif "gas" in row.carrier.lower():
                    tech_group = "Gas-CCS"

                cap_data.append({
                    "Country": get_country(row.bus0),
                    "Technology": tech_group,
                    "Amount": amount
                })

    # --- STORAGE ---
    seq_links = n.links[n.links.carrier.str.contains("sequestered", case=False) &
                        ~n.links.carrier.str.contains("pipeline", case=False)]

    for idx, row in seq_links.iterrows():
        flow = n.links_t.p0[idx].multiply(weight, axis=0).sum()
        amount = flow / 1e6
        if amount > 0.01:
            store_data.append({
                "Country": get_country(row.bus1),
                "Amount": amount
            })

    return pd.DataFrame(cap_data), pd.DataFrame(store_data)


def plot_co2_flows(df_cap, df_store, config, network_name):
    """
    Erstellt einen Spiegel-Balken-Plot:
    Oben: Capture (nach Tech)
    Unten: Storage (nach Land)
    """
    if df_cap.empty and df_store.empty: return

    # Aggregieren
    cap_agg = df_cap.groupby(["Country", "Technology"])["Amount"].sum().unstack().fillna(0)
    store_agg = df_store.groupby(["Country"])["Amount"].sum()

    # Länder vereinigen und sortieren
    countries = sorted(list(set(cap_agg.index) | set(store_agg.index)))
    cap_agg = cap_agg.reindex(countries).fillna(0)
    store_agg = store_agg.reindex(countries).fillna(0)

    # Sortieren nach Gesamtaktivität (Top 10)
    total = cap_agg.sum(axis=1) + store_agg
    countries = total.sort_values(ascending=False).index[:10]

    cap_agg = cap_agg.loc[countries]
    store_agg = store_agg.loc[countries]

    # PLOT
    fig, ax = plt.subplots(figsize=(12, 7))

    # 1. Capture (Positiv)
    colors = {"Direct Air Capture (DAC)": "#1f77b4",
              "Biomasse-CCS (BECCS)": "#2ca02c",
              "Industrie/Process": "#7f7f7f", "Gas-CCS": "#d62728"}

    bottom = np.zeros(len(countries))
    for tech in cap_agg.columns:
        vals = cap_agg[tech].values
        c = colors.get(tech, "grey")
        ax.bar(countries, vals, bottom=bottom, label=tech, color=c, edgecolor="white", width=0.6)
        bottom += vals

    # 2. Storage (Negativ)
    ax.bar(countries, -store_agg.values, color="#8c564b", label="Endlagerung (Sequestrierung)",
           edgecolor="white", width=0.6, hatch="//")

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title(f"CO₂-Infrastruktur Europa {TARGET_YEAR}: Einfang (oben) vs. Speicherung (unten)", fontsize=14)
    ax.set_ylabel("CO₂-Menge [Mio. Tonnen]", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "emissions")
    plt.savefig(os.path.join(save_dir, f"co2_infrastructure_{TARGET_YEAR}.png"), dpi=300)
    print(f"✅ Plot gespeichert: co2_infrastructure_{TARGET_YEAR}.png")


# =====================================================================
# MAIN
# =====================================================================
def main():
    config = PlottingConfig()
    networks = config.get_networks()
    if isinstance(networks, dict):
        paths = networks[config.SCENARIO_SELECTION]
    else:
        paths = networks

    ref_path = paths[0]
    network_name = os.path.basename(os.path.dirname(os.path.dirname(ref_path)))

    # Finde 2050 Datei
    target_file = None
    for path in paths:
        if f"_{TARGET_YEAR}.nc" in path:
            target_file = path
            break

    if not target_file: return

    print(f"Lade Netzwerk: {target_file}")
    n = pypsa.Network(target_file)

    # --- ANALYSE TEIL 1 (Brutto / DE Detail) ---
    df_gross = analyze_gross_emissions(n)

    print("\n🏆 TOP EMITTENTEN (Brutto, Mt):")
    top_emitters = df_gross.groupby("Country")["Amount"].sum().sort_values(ascending=False)
    print(top_emitters.head(8))

    plot_de_breakdown(df_gross, config, network_name)

    # --- ANALYSE TEIL 2 (Infrastruktur / Capture / Storage) ---
    df_cap, df_store = analyze_capture_storage(n)

    # --- DAC ZUSAMMENFASSUNG ---
    df_dac = df_cap[df_cap["Technology"] == "Direct Air Capture (DAC)"]

    total_dac = df_dac["Amount"].sum()
    total_capture = df_cap["Amount"].sum()

    print("\n🟦 DIRECT AIR CAPTURE (DAC)")
    print(f"Gesamt DAC Europa: {total_dac:.2f} Mt CO₂")

    if total_capture > 0:
        share = 100 * total_dac / total_capture
        print(f"Anteil an gesamtem Capture: {share:.1f} %")

    print("\n--- CAPTURE NACH LAND (Top 5) ---")
    print(df_cap.groupby("Country")["Amount"].sum().sort_values(ascending=False).head())

    print("\n--- STORAGE NACH LAND (Top 5) ---")
    print(df_store.groupby("Country")["Amount"].sum().sort_values(ascending=False).head())

    plot_co2_flows(df_cap, df_store, config, network_name)


if __name__ == "__main__":
    main()