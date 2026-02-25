#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot: Strompreis (marginal price) je Land – volle und begrenzte Skala
=====================================================================

Erstellt für jedes Land (und ALL) zwei Preisverläufe:
1️⃣ komplette Skala
2️⃣ begrenzt auf 0–100 €/MWh

Verwendet dieselbe Speicherlogik wie das Gas/H2-Dispatch-Skript.
"""

import os
import re
import pypsa
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config_final import PlottingConfig


# =====================================================================
# --- Hilfsfunktionen ---
# =====================================================================

def get_country_price_series(n: pypsa.Network, country: str) -> pd.Series:
    """
    Liefert den stündlichen marginalen Preis (€/MWh) für den Hauptstrombus eines Landes.
    - Bei 'ALL' wird der Mittelwert über alle elektrischen Busse (AC + low voltage) gebildet.
    - Für Länder (z. B. 'DE') wird, falls vorhanden, der Hauptbus 'DE0 0' gewählt.
      Andernfalls der erste elektrische Bus (AC oder low voltage) des Landes.
    """

    price = n.buses_t.marginal_price

    # === Spezialfall: gesamtes Netz (ALL) ===
    if country == "ALL":
        elec_buses = n.buses.index[n.buses.carrier.isin(["AC", "low voltage"])]
        if elec_buses.empty:
            print("⚠️ Keine elektrischen Busse im gesamten Netz gefunden.")
            return pd.Series(dtype=float)
        return price[elec_buses].mean(axis=1)

    # === Busse des Landes filtern (nur elektrische Spannungsebenen) ===
    country_buses = [
        b for b in n.buses.index
        if b.startswith(country)
        and n.buses.carrier[b] in ["AC"]
    ]

    if not country_buses:
        print(f"⚠️ Keine elektrischen Busse für {country} gefunden.")
        return pd.Series(dtype=float)

    # === Hauptstrombus identifizieren (z. B. 'DE0 0') ===
    main_bus_candidates = [b for b in country_buses if re.match(rf"{country}\d*\s0$", b)]
    if main_bus_candidates:
        main_bus = main_bus_candidates[0]
    else:
        # Fallback: erster Bus des Landes
        main_bus = country_buses[0]

    print(f"   → Verwende Hauptstrombus für {country}: {main_bus}")

    # === Preiszeitreihe dieses Busses ===
    return price[main_bus]


def plot_price_series(price_series: pd.Series, country: str, year: int,
                      config: PlottingConfig, save_folder: str, scenario_title: str):
    """
    Erstellt und speichert zwei Preisplots:
    1. komplette Skala
    2. begrenzte Skala (0–100 €/MWh)
    """

    if price_series.empty:
        print(f"⚠️ Keine Preisdaten für {country} {year}.")
        return

    mean_price = price_series.mean()
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(price_series.index, price_series.values,
            color="steelblue", lw=1.3, label=f"{country} Preisverlauf")
    ax.axhline(y=mean_price, color="r", linestyle="--",
               label=f"Mittelwert: {mean_price:.1f} €/MWh")

    ax.set_title(f"Strompreisverlauf {country} – {year} ({scenario_title}) – volle Skala",
                 fontsize=config.FONT_SIZES["title"])
    ax.set_xlabel("Datum", fontsize=config.FONT_SIZES["label"])
    ax.set_ylabel("Preis [€/MWh]", fontsize=config.FONT_SIZES["label"])
    ax.legend(fontsize=config.FONT_SIZES["legend"])
    ax.grid(alpha=0.4)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.setp(ax.get_xticklabels(), rotation=0)

    # === Plot 1: Volle Skala ===
    folder_full = os.path.join(save_folder, "price_chronological_full")
    os.makedirs(folder_full, exist_ok=True)
    fpath_full = os.path.join(folder_full, f"{country}_{year}_price_full.png")
    plt.tight_layout()
    plt.savefig(fpath_full, dpi=300)
    print(f"✅ Gespeichert: {fpath_full}")

    # === Plot 2: Begrenzte Skala ===
    ax.set_ylim(0, 100)
    ax.set_title(f"Strompreisverlauf {country} – {year} ({scenario_title}) – 0–100 €/MWh",
                 fontsize=config.FONT_SIZES["title"])

    folder_cap = os.path.join(save_folder, "price_chronological_capped")
    os.makedirs(folder_cap, exist_ok=True)
    fpath_cap = os.path.join(folder_cap, f"{country}_{year}_price_capped.png")
    plt.tight_layout()
    plt.savefig(fpath_cap, dpi=300)
    print(f"✅ Gespeichert: {fpath_cap}")

    plt.close(fig)


# =====================================================================
# --- Hauptfunktion ---
# =====================================================================

def main():
    config = PlottingConfig()
    networks = config.get_networks()

    if not isinstance(networks, list) or not networks:
        print("FEHLER: Erwartet Liste von Netzwerkdateien aus config.get_networks().")
        return

    scenario_title = config.SCENARIO_SELECTION.replace("_", " ").title()

    for path in networks:
        if not os.path.isfile(path):
            print(f"⚠️ Datei nicht gefunden: {path}")
            continue

        # Jahr aus Dateiname extrahieren
        m = re.search(r"_(\d{4})\.nc$", path)
        if not m:
            continue
        year = int(m.group(1))

        print(f"\n📂 Lade Netzwerk {year}: {path}")
        n = pypsa.Network(path)

        # Speicherordner analog zur Gas/H₂-Logik
        network_root = os.path.dirname(os.path.dirname(path))
        network_name = os.path.basename(network_root)
        save_folder = os.path.join(config.BASE_SAVE_PATH, network_name)

        for country in config.get_countries():
            print(f"→ {country}: Preisverlauf wird berechnet …")
            price_series = get_country_price_series(n, country)
            plot_price_series(price_series, country, year, config, save_folder, scenario_title)


if __name__ == "__main__":
    main()
