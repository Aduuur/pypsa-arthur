#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot: CO2-Bilanz nach Primärenergie (Source-Based) - FIXED
==========================================================
Berechnet die Netto-Emissionen basierend auf:
(+) Fossile Primärenergie (Gas/Öl/Kohle, die ins System kommen)
(-) CO2-Sequestrierung (Was dauerhaft aus dem System entfernt wird)

Vermeidet Doppelzählung von Biogas/Synfuels.

ARO-Fix: year-Parsing defensiv – wenn kein YYYY im Pfad (z.B. Dispatch-Netz),
wird 2050 als Fallback verwendet, damit .group() nicht auf None crasht.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import pypsa
from config_final import PlottingConfig

# Welche Komponenten stellen fossile Primärenergie bereit?
FOSSIL_SOURCES = ["coal", "lignite", "oil", "gas", "oil primary"]

# Was ist Sequestrierung? (Endlagerung)
SINK_CARRIER = "co2 sequestered"


def _parse_year(path: str, fallback: int = 2050) -> int:
    """Extrahiert das Jahr aus einem Netzwerk-Pfad.

    Unterstützt sowohl Planungsnetze ('base_s_24___2050.nc')
    als auch ARO Dispatch-Netze ('dispatch_..._worst_case_std.nc').
    Im letzten Fall gibt es kein YYYY im Namen → Fallback wird genutzt.
    """
    m = re.search(r"___(\d{4})\.nc$", path) or re.search(r"_(\d{4})\.nc", path)
    return int(m.group(1)) if m else fallback


def get_emissions_source_based(n: pypsa.Network, country_code: str) -> dict:
    """
    Bilanz: Fossil In - Sequestered Out.
    """
    weight = n.snapshot_weightings.generators

    fossil_in = 0.0
    sequestered = 0.0

    # Wir machen hier die Berechnung für "ALL" (Europa)
    if country_code == "ALL":
        # A) Fossile Erzeugung (Quelle)
        # Suche Generatoren, die Fossilien "erzeugen" (ins Modell bringen)
        gens = n.generators[n.generators.carrier.isin(FOSSIL_SOURCES)]

        for carrier in gens.carrier.unique():
            # Faktor
            co2_fac = n.carriers.at[carrier, "co2_emissions"] if carrier in n.carriers.index else 0
            # Fallback
            if co2_fac == 0:
                c_low = carrier.lower()
                if "lignite" in c_low:
                    co2_fac = 0.406
                elif "coal" in c_low:
                    co2_fac = 0.34
                elif "oil" in c_low:
                    co2_fac = 0.28
                elif "gas" in c_low:
                    co2_fac = 0.202

            if co2_fac > 0:
                idx = gens[gens.carrier == carrier].index
                # p ist die Menge an Brennstoff (MWh_th), die ins System kommt
                amount = n.generators_t.p[idx].multiply(weight, axis=0).sum(axis=0).sum()
                fossil_in += amount * co2_fac

        # B) Sequestrierung (Senke)
        # Link "co2 sequestered" - KORRIGIERT
        seq_links = n.links[n.links.carrier == SINK_CARRIER]

        if not seq_links.empty:
            # p0 ist Input in den Speicher (Tonnen CO2)
            stored = n.links_t.p0[seq_links.index].multiply(weight, axis=0).sum(axis=0).sum()
            sequestered += stored

        return {
            "Gross": fossil_in / 1e6,
            "Sequestered": sequestered / 1e6,
            "Net": (fossil_in - sequestered) / 1e6
        }

    else:
        # Für DE nehmen wir die Verbrauchssicht (wie vorher)
        return get_emissions_consumption_based(n, country_code)


def get_emissions_consumption_based(n, country_code):
    """
    Berechnet Emissionen basierend auf Verbrennung (Kraftwerke + Boiler).
    Zieht lokale Sequestrierung ab.
    """
    gross = 0.0
    seq = 0.0
    weight = n.snapshot_weightings.generators

    # 1. Verbrennung (Gen + Link)
    targets = ["coal", "lignite", "oil", "gas", "CCGT", "OCGT", "waste"]

    # Generatoren
    if country_code != "ALL":
        gens = n.generators[n.generators.bus.str.startswith(country_code)]
    else:
        gens = n.generators

    for c in gens.carrier.unique():
        if any(t in c for t in targets):
            # Faktor
            co2 = n.carriers.at[c, "co2_emissions"] if c in n.carriers.index else 0
            if co2 == 0:  # Fallback
                if "gas" in c or "CCGT" in c:
                    co2 = 0.202
                elif "coal" in c:
                    co2 = 0.34
                else:
                    co2 = 0.4

            idx = gens[gens.carrier == c].index
            eff = n.generators.loc[idx, "efficiency"].replace(0, 1.0)
            p = n.generators_t.p[idx].multiply(weight, axis=0).sum(axis=0).sum()
            gross += (p / eff.mean()) * co2

    # Boiler (Links)
    if country_code != "ALL":
        links = n.links[(n.links.bus0.str.startswith(country_code)) | (n.links.bus1.str.startswith(country_code))]
    else:
        links = n.links

    emit_links = links[
        links.carrier.str.contains("boiler", case=False) & links.carrier.str.contains("gas|oil", case=False)]
    for c in emit_links.carrier.unique():
        co2 = 0.202 if "gas" in c else 0.28
        idx = emit_links[emit_links.carrier == c].index
        p = n.links_t.p0[idx].multiply(weight, axis=0).sum(axis=0).sum()  # p0 input
        gross += p * co2

    # 2. Sequestrierung (Lokal)
    seq_links = links[links.carrier == SINK_CARRIER]
    if not seq_links.empty:
        seq += n.links_t.p0[seq_links.index].multiply(weight, axis=0).sum(axis=0).sum()

    return {"Gross": gross / 1e6, "Sequestered": seq / 1e6, "Net": (gross - seq) / 1e6}


def plot_final(df, config, network_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    years = df.index

    # DE Netto (eigentlich "Local Net")
    ax.plot(years, df["DE_Net"], 'o-', color="black", linewidth=2, label="DE (Lokale Bilanz)")

    # ALL Netto (Das ist der Proof für Net Zero!)
    ax.plot(years, df["ALL_Net"], 's--', color="blue", linewidth=2, label="Europa (System-Bilanz)")

    # Werte anschreiben
    for y in years:
        # DE
        val_de = df.loc[y, "DE_Net"]
        ax.annotate(f"{val_de:.0f}", (y, val_de), xytext=(0, 10), textcoords="offset points", ha="center",
                    fontweight="bold")
        # ALL
        val_all = df.loc[y, "ALL_Net"]
        ax.annotate(f"{val_all:.0f}", (y, val_all), xytext=(0, -15), textcoords="offset points", ha="center",
                    color="blue")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("CO₂-Bilanz: Deutschland vs. Europa (Net-Zero Check)", fontsize=14)
    ax.set_ylabel("Netto-Emissionen [Mio. t CO₂]", fontsize=12)
    ax.set_xlabel("Jahr", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "emissions")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "co2_proof_final.png"), dpi=300)
    print(f"✅ Plot: {save_dir}/co2_proof_final.png")


def main():
    config = PlottingConfig()
    networks = config.get_networks()
    if isinstance(networks, dict):
        paths = networks[config.SCENARIO_SELECTION]
    else:
        paths = networks

    ref = paths[0]
    name = os.path.basename(os.path.dirname(os.path.dirname(ref)))

    res = []
    for path in paths:
        if not os.path.isfile(path):
            continue

        # ARO-Fix: defensives year-parsing – kein Crash wenn kein YYYY im Pfad
        y = _parse_year(path, fallback=2050)
        print(f"Jahr {y}...")
        n = pypsa.Network(path)

        # DE nach alter Logik (Verbrennung), ALL nach neuer Logik (Primärenergie)
        de = get_emissions_consumption_based(n, "DE")
        all_sys = get_emissions_source_based(n, "ALL")

        res.append({
            "Year": y,
            "DE_Gross": de["Gross"],
            "DE_Net": de["Net"],
            "ALL_Net": all_sys["Net"]
        })

    df = pd.DataFrame(res).set_index("Year").sort_index()
    print(df.round(1))
    plot_final(df, config, name)


if __name__ == "__main__":
    main()
