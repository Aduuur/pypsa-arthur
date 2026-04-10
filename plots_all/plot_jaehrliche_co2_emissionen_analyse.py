#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse: CO2-Infrastruktur Komplett (Quellen, Capture, Storage)
===============================================================
Kombiniert zwei Analysen in einem Skript:
1. Emissions-Quellen: Wer verbrennt Fossilien? (Brutto)
   -> Detail-Plot für Deutschland (Kuchendiagramm)
2. CO2-Infrastruktur: Capture und Storage
   -> Infrastruktur-Plot (Spiegel-Balken)

FIXES:
  1 — Import war config_final → master_config
  2 — networks[SCENARIO_SELECTION] crashte bei "both": Keys sind
      "average"/"dunkelflaute", nicht "both". Jetzt defensiver Zugriff
      mit Fallback auf erstes verfügbares Netz.
  3 — Ausgabepfad nutzt jetzt PLOT_OUTPUT_PATH statt BASE_SAVE_PATH
  4 — Ziel-Netz wird aus n.snapshots.year gelesen, nicht aus Dateinamen
      (Dispatch-Netze haben kein _YYYY.nc im Namen)
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pypsa

# FIX 1: war config_final
from master_config import PlottingConfig

TARGET_YEAR = 2050
EMITTING_CARRIERS = ["coal", "lignite", "oil", "gas", "CCGT", "OCGT", "waste"]


def get_country(bus_name):
    return str(bus_name)[:2]


def _parse_year(n: pypsa.Network, fallback: int = TARGET_YEAR) -> int:
    """Jahr aus Snapshots lesen — sicher auch für Dispatch-Netze."""
    try:
        return int(n.snapshots[0].year)
    except Exception:
        return fallback


# =============================================================================
# TEIL 1: BRUTTO-EMISSIONEN
# =============================================================================

def analyze_gross_emissions(n):
    print("... Analysiere Emissions-Quellen (Verbrennung) ...")
    weight = n.snapshot_weightings.generators
    data = []

    # Generatoren
    mask_gen = n.generators.carrier.str.contains(
        "|".join(EMITTING_CARRIERS), case=False, na=False
    )
    mask_gen &= ~n.generators.carrier.str.contains("biomass", case=False)
    gens = n.generators[mask_gen]

    for carrier in gens.carrier.unique():
        co2_fac = n.carriers.at[carrier, "co2_emissions"] if carrier in n.carriers.index else 0
        if pd.isna(co2_fac) or co2_fac == 0:
            c_low = carrier.lower()
            if "lignite" in c_low:   co2_fac = 0.406
            elif "coal"  in c_low:   co2_fac = 0.34
            elif "oil"   in c_low:   co2_fac = 0.28
            elif "gas"   in c_low or "ccgt" in c_low: co2_fac = 0.202
        if co2_fac <= 0:
            continue

        idx = gens[gens.carrier == carrier].index
        avail = n.generators_t.p.columns.intersection(idx)
        if avail.empty:
            continue
        eff = n.generators.loc[avail, "efficiency"].replace(0, 1.0)
        p_el = n.generators_t.p[avail].multiply(weight, axis=0).sum(axis=0)
        emissions = (p_el / eff) * co2_fac

        for gen_name, val in emissions.items():
            if val > 0.01:
                data.append({
                    "Country": get_country(n.generators.at[gen_name, "bus"]),
                    "Carrier": carrier,
                    "Type":    "Stromerzeugung",
                    "Amount":  val / 1e6,
                })

    # Links (Boiler)
    mask_link = (
        n.links.carrier.str.contains("boiler", case=False) &
        n.links.carrier.str.contains("gas|oil", case=False)
    )
    links = n.links[mask_link]

    for carrier in links.carrier.unique():
        co2_fac = 0.202 if "gas" in carrier.lower() else 0.28
        idx   = links[links.carrier == carrier].index
        avail = n.links_t.p0.columns.intersection(idx)
        if avail.empty:
            continue
        p_in = n.links_t.p0[avail].multiply(weight, axis=0).sum(axis=0)
        emissions = p_in * co2_fac

        for link_name, val in emissions.items():
            if val > 0.01:
                bus = n.links.at[link_name, "bus0"]
                data.append({
                    "Country": get_country(bus),
                    "Carrier": carrier,
                    "Type":    "Wärmeerzeugung",
                    "Amount":  val / 1e6,
                })

    return pd.DataFrame(data)


def plot_de_breakdown(df_gross, save_dir: str, year: int):
    df_de = df_gross[df_gross["Country"] == "DE"]
    if df_de.empty:
        return

    def map_cat(row):
        c = row["Carrier"].lower()
        if "rural"          in c: return "Dezentral (Ländlich)"
        if "urban decentral" in c: return "Dezentral (Städtisch)"
        if "urban central"  in c: return "Fernwärme"
        if "ccgt" in c or "ocgt" in c: return "Strom (Kraftwerke)"
        if "oil"            in c: return "Öl-Kessel"
        return "Industrie/Sonstiges"

    df_de = df_de.copy()
    df_de["Category"] = df_de.apply(map_cat, axis=1)
    stats = df_de.groupby("Category")["Amount"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(stats)))
    wedges, texts, autotexts = ax.pie(
        stats, labels=stats.index, autopct="%1.1f%%",
        startangle=90, colors=colors, pctdistance=0.85,
    )
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    fig.gca().add_artist(centre_circle)
    total = stats.sum()
    ax.text(0, 0, f"Gesamt\n{total:.1f} Mt", ha="center", va="center",
            fontsize=12, fontweight="bold")
    ax.set_title(f"Quellen der CO₂-Emissionen in Deutschland ({year})", fontsize=14)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "de_emissions_breakdown.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Plot gespeichert: {out}")


# =============================================================================
# TEIL 2: CAPTURE & STORAGE
# =============================================================================

def analyze_capture_storage(n):
    print("... Analysiere Capture & Storage ...")
    weight  = n.snapshot_weightings.generators
    cap_data   = []
    store_data = []

    # Capture
    mask_cc = n.links.carrier.str.contains("CC|DAC", case=False, regex=True)
    mask_cc &= ~n.links.carrier.str.contains("CCGT", case=False)
    candidates = n.links[mask_cc]

    for idx, row in candidates.iterrows():
        co2_port = None
        for col in ["bus0", "bus1", "bus2", "bus3", "bus4"]:
            if col in n.links.columns:
                if "co2" in str(row[col]).lower() and "atm" not in str(row[col]).lower():
                    co2_port = col
                    break

        if co2_port:
            port_num = co2_port.replace("bus", "p")
            try:
                flow = n.links_t[port_num][idx].multiply(weight, axis=0).sum()
            except Exception:
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
                    "Country":    get_country(row.bus0),
                    "Technology": tech_group,
                    "Amount":     amount,
                })

    # Storage
    seq_links = n.links[
        n.links.carrier.str.contains("sequestered", case=False) &
        ~n.links.carrier.str.contains("pipeline", case=False)
    ]
    for idx, row in seq_links.iterrows():
        avail = n.links_t.p0.columns
        if idx not in avail:
            continue
        flow   = n.links_t.p0[idx].multiply(weight, axis=0).sum()
        amount = flow / 1e6
        if amount > 0.01:
            store_data.append({
                "Country": get_country(row.bus1),
                "Amount":  amount,
            })

    return pd.DataFrame(cap_data), pd.DataFrame(store_data)


def plot_co2_flows(df_cap, df_store, save_dir: str, year: int):
    if df_cap.empty and df_store.empty:
        return

    cap_agg   = df_cap.groupby(["Country", "Technology"])["Amount"].sum().unstack().fillna(0) \
                if not df_cap.empty else pd.DataFrame()
    store_agg = df_store.groupby("Country")["Amount"].sum() \
                if not df_store.empty else pd.Series(dtype=float)

    countries = sorted(set(cap_agg.index if not cap_agg.empty else []) |
                       set(store_agg.index if not store_agg.empty else []))
    if not countries:
        return

    if not cap_agg.empty:
        cap_agg = cap_agg.reindex(countries).fillna(0)
    if not store_agg.empty:
        store_agg = store_agg.reindex(countries).fillna(0)

    total = (cap_agg.sum(axis=1) if not cap_agg.empty else pd.Series(0, index=countries)) + \
            (store_agg if not store_agg.empty else pd.Series(0, index=countries))
    countries = total.sort_values(ascending=False).index[:10].tolist()

    if not cap_agg.empty:
        cap_agg = cap_agg.loc[[c for c in countries if c in cap_agg.index]]
    if not store_agg.empty:
        store_agg = store_agg.reindex(countries).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        "Direct Air Capture (DAC)": "#1f77b4",
        "Biomasse-CCS (BECCS)":     "#2ca02c",
        "Industrie/Process":        "#7f7f7f",
        "Gas-CCS":                  "#d62728",
    }

    bottom = np.zeros(len(countries))
    if not cap_agg.empty:
        for tech in cap_agg.columns:
            vals = cap_agg.reindex(countries).fillna(0)[tech].values
            ax.bar(countries, vals, bottom=bottom,
                   label=tech, color=colors.get(tech, "grey"),
                   edgecolor="white", width=0.6)
            bottom += vals

    if not store_agg.empty:
        ax.bar(countries, -store_agg.reindex(countries).fillna(0).values,
               color="#8c564b", label="Endlagerung (Sequestrierung)",
               edgecolor="white", width=0.6, hatch="//")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"CO₂-Infrastruktur Europa {year}: Einfang (oben) vs. Speicherung (unten)",
                 fontsize=14)
    ax.set_ylabel("CO₂-Menge [Mio. Tonnen]", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"co2_infrastructure_{year}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Plot gespeichert: {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    config   = PlottingConfig()
    networks = config.get_networks()

    # FIX 2: defensiver Zugriff — "both" ist kein Key in networks-Dict
    # Keys bei "both"-Modus: "average" und "dunkelflaute"
    # Im ARO-Modus: Key = run_key oder beliebiger Scenario-Key
    if isinstance(networks, dict):
        sel = config.SCENARIO_SELECTION
        if sel in networks:
            paths = networks[sel]
        elif "average" in networks:
            paths = networks["average"]          # "both"-Modus: Basisjahr nehmen
        else:
            paths = list(networks.values())[0]   # erster verfügbarer Key
    else:
        paths = networks

    if not paths:
        print("❌ Keine Netzwerkpfade verfügbar.")
        return

    # FIX 4: Ziel-Netz über Snapshot-Jahr finden, nicht über Dateinamen
    # Im ARO-Modus gibt es nur ein Netz — das wird immer genommen
    target_file = None
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            n_tmp = pypsa.Network(path)
            year  = _parse_year(n_tmp)
            if year == TARGET_YEAR or len(paths) == 1:
                target_file = path
                break
        except Exception:
            continue

    # Fallback: erstes vorhandene Netz
    if target_file is None:
        target_file = next((p for p in paths if os.path.isfile(p)), None)

    if target_file is None:
        print(f"❌ Kein Netzwerk für Jahr {TARGET_YEAR} gefunden.")
        return

    print(f"Lade Netzwerk: {target_file}")
    n    = pypsa.Network(target_file)
    year = _parse_year(n, fallback=TARGET_YEAR)

    # FIX 3: PLOT_OUTPUT_PATH statt BASE_SAVE_PATH + network_name
    save_dir = os.path.join(config.PLOT_OUTPUT_PATH, "emissions")

    # --- TEIL 1 ---
    df_gross = analyze_gross_emissions(n)

    print("\n🏆 TOP EMITTENTEN (Brutto, Mt):")
    top_emitters = df_gross.groupby("Country")["Amount"].sum().sort_values(ascending=False)
    print(top_emitters.head(8))

    plot_de_breakdown(df_gross, save_dir, year)

    # --- TEIL 2 ---
    df_cap, df_store = analyze_capture_storage(n)

    total_dac     = df_cap[df_cap["Technology"] == "Direct Air Capture (DAC)"]["Amount"].sum() \
                    if not df_cap.empty else 0.0
    total_capture = df_cap["Amount"].sum() if not df_cap.empty else 0.0

    print(f"\n🟦 DIRECT AIR CAPTURE (DAC)")
    print(f"Gesamt DAC Europa: {total_dac:.2f} Mt CO₂")
    if total_capture > 0:
        print(f"Anteil an gesamtem Capture: {100 * total_dac / total_capture:.1f} %")

    if not df_cap.empty:
        print("\n--- CAPTURE NACH LAND (Top 5) ---")
        print(df_cap.groupby("Country")["Amount"].sum().sort_values(ascending=False).head())

    if not df_store.empty:
        print("\n--- STORAGE NACH LAND (Top 5) ---")
        print(df_store.groupby("Country")["Amount"].sum().sort_values(ascending=False).head())

    plot_co2_flows(df_cap, df_store, save_dir, year)


if __name__ == "__main__":
    main()