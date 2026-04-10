#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot: CO2-Bilanz nach Primärenergie (Source-Based)
====================================================
Berechnet Netto-Emissionen: fossile Primärenergie minus CO2-Sequestrierung.

Funktioniert sowohl im normalen als auch im ARO-Modus.
Ausgabe in config.PLOT_OUTPUT_PATH/emissions/.

ARO-Fix: Jahr wird aus n.snapshots[0].year gelesen, nicht aus Dateinamen.
"""

import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pypsa

# FIX: war config_final
from master_config import PlottingConfig

FOSSIL_SOURCES = ["coal", "lignite", "oil", "gas", "oil primary"]
SINK_CARRIER   = "co2 sequestered"


def _parse_year(n: pypsa.Network, path: str, fallback: int = 2050) -> int:
    """Jahr zuverlässig aus Netzwerk lesen — funktioniert auch für Dispatch-Netze."""
    try:
        return int(n.snapshots[0].year)
    except Exception:
        pass
    m = re.search(r"___(\d{4})\.nc$", path) or re.search(r"_(\d{4})\.nc", path)
    return int(m.group(1)) if m else fallback


def get_emissions_source_based(n: pypsa.Network, country_code: str) -> dict:
    weight = n.snapshot_weightings.generators

    fossil_in    = 0.0
    sequestered  = 0.0

    if country_code == "ALL":
        gens = n.generators[n.generators.carrier.isin(FOSSIL_SOURCES)]
        for carrier in gens.carrier.unique():
            co2_fac = n.carriers.at[carrier, "co2_emissions"] if carrier in n.carriers.index else 0
            if co2_fac == 0:
                c_low = carrier.lower()
                if "lignite" in c_low: co2_fac = 0.406
                elif "coal"   in c_low: co2_fac = 0.34
                elif "oil"    in c_low: co2_fac = 0.28
                elif "gas"    in c_low: co2_fac = 0.202
            if co2_fac > 0:
                idx    = gens[gens.carrier == carrier].index
                amount = n.generators_t.p[idx].multiply(weight, axis=0).sum(axis=0).sum()
                fossil_in += amount * co2_fac

        seq_links = n.links[n.links.carrier == SINK_CARRIER]
        if not seq_links.empty:
            stored = n.links_t.p0[seq_links.index].multiply(weight, axis=0).sum(axis=0).sum()
            sequestered += stored

        return {
            "Gross":       fossil_in  / 1e6,
            "Sequestered": sequestered / 1e6,
            "Net":         (fossil_in - sequestered) / 1e6,
        }
    else:
        return get_emissions_consumption_based(n, country_code)


def get_emissions_consumption_based(n, country_code):
    gross  = 0.0
    seq    = 0.0
    weight = n.snapshot_weightings.generators
    targets = ["coal", "lignite", "oil", "gas", "CCGT", "OCGT", "waste"]

    gens = (
        n.generators[n.generators.bus.str.startswith(country_code)]
        if country_code != "ALL"
        else n.generators
    )
    for c in gens.carrier.unique():
        if not any(t in c for t in targets):
            continue
        co2 = n.carriers.at[c, "co2_emissions"] if c in n.carriers.index else 0
        if co2 == 0:
            if "gas" in c or "CCGT" in c: co2 = 0.202
            elif "coal" in c:             co2 = 0.34
            else:                          co2 = 0.4
        idx = gens[gens.carrier == c].index
        eff = n.generators.loc[idx, "efficiency"].replace(0, 1.0)
        p   = n.generators_t.p[idx].multiply(weight, axis=0).sum(axis=0).sum()
        gross += (p / eff.mean()) * co2

    links = (
        n.links[
            n.links.bus0.str.startswith(country_code) |
            n.links.bus1.str.startswith(country_code)
        ]
        if country_code != "ALL"
        else n.links
    )
    emit_links = links[
        links.carrier.str.contains("boiler", case=False) &
        links.carrier.str.contains("gas|oil", case=False)
    ]
    for c in emit_links.carrier.unique():
        co2 = 0.202 if "gas" in c else 0.28
        idx = emit_links[emit_links.carrier == c].index
        p   = n.links_t.p0[idx].multiply(weight, axis=0).sum(axis=0).sum()
        gross += p * co2

    seq_links = links[links.carrier == SINK_CARRIER]
    if not seq_links.empty:
        seq += n.links_t.p0[seq_links.index].multiply(weight, axis=0).sum(axis=0).sum()

    return {"Gross": gross / 1e6, "Sequestered": seq / 1e6, "Net": (gross - seq) / 1e6}


def plot_final(df, save_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    years = df.index

    ax.plot(years, df["DE_Net"],  "o-",  color="black", linewidth=2, label="DE (Lokale Bilanz)")
    ax.plot(years, df["ALL_Net"], "s--", color="blue",  linewidth=2, label="Europa (System-Bilanz)")

    for y in years:
        val_de  = df.loc[y, "DE_Net"]
        val_all = df.loc[y, "ALL_Net"]
        ax.annotate(f"{val_de:.0f}",  (y, val_de),  xytext=(0,  10), textcoords="offset points", ha="center", fontweight="bold")
        ax.annotate(f"{val_all:.0f}", (y, val_all), xytext=(0, -15), textcoords="offset points", ha="center", color="blue")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("CO₂-Bilanz: Deutschland vs. Europa (Net-Zero Check)", fontsize=14)
    ax.set_ylabel("Netto-Emissionen [Mio. t CO₂]", fontsize=12)
    ax.set_xlabel("Jahr", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "co2_proof_final.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Plot: {out}")


def main():
    config   = PlottingConfig()
    networks = config.get_networks()

    if isinstance(networks, dict):
        paths = networks.get(config.SCENARIO_SELECTION, list(networks.values())[0])
    else:
        paths = networks

    # FIX: PLOT_OUTPUT_PATH statt BASE_SAVE_PATH + network_name
    save_dir = os.path.join(config.PLOT_OUTPUT_PATH, "emissions")

    res = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            n = pypsa.Network(path)
            # FIX: Jahr aus Snapshots lesen — sicher auch für Dispatch-Netze
            y = _parse_year(n, path, fallback=2050)
            print(f"Jahr {y}...")

            de      = get_emissions_consumption_based(n, "DE")
            all_sys = get_emissions_source_based(n, "ALL")
            res.append({
                "Year":     y,
                "DE_Gross": de["Gross"],
                "DE_Net":   de["Net"],
                "ALL_Net":  all_sys["Net"],
            })
        except Exception as e:
            print(f"Fehler {path}: {e}")

    if not res:
        print("Keine Netzwerke verarbeitet.")
        return

    df = pd.DataFrame(res).set_index("Year").sort_index()
    print(df.round(1))
    plot_final(df, save_dir)


if __name__ == "__main__":
    main()