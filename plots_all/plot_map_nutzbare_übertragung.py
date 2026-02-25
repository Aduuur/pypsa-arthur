#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting: Nutzbare Kapazitäten (NTC Proxy)
==========================================

"""

import pypsa
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys
import re
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from config_final import PlottingConfig

# =============================================================================
# EINSTELLUNGEN
# =============================================================================

SCALE_FACTOR = 1.5e3

COLOR_BORDER_LINE = "#2a6fdb"  # Blau
COLOR_NODE = "black"

# Codes, die wir komplett ignorieren wollen
IGNORE_CODES = ["EU"]

# Sicherheitsfaktoren (NTC Proxy)
DEFAULT_S_MAX_PU_AC = 0.7
DEFAULT_P_MAX_PU_DC = 1.0


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def get_country(bus_name):
    """Extrahiert Ländercode. Gibt 'EU' zurück, wenn es ein EU-Knoten ist."""
    s = str(bus_name)
    if "EU" in s: return "EU"
    return s[:2]


def get_border_name(bus0, bus1):
    c0 = get_country(bus0)
    c1 = get_country(bus1)

    # Fehlerhafte oder EU-Knoten ignorieren
    if c0 in IGNORE_CODES or c1 in IGNORE_CODES: return None
    if c0 == "EU" or c1 == "EU": return None

    # Grenz-Name erstellen
    return "-".join(sorted([c0, c1]))


def calculate_usable_capacities(n):
    """
    Berechnet AC (skaliert) + DC (skaliert).
    Logik:
    - Internationale Grenzen: IMMER behalten (AC & DC).
    - Inlands-Grenzen: NUR behalten, wenn es DC ist (Insel-Anbindung).
    """
    connections = {}

    # --- 1. AC Lines ---
    col_ac = "s_nom_opt" if "s_nom_opt" in n.lines.columns else "s_nom"
    has_s_max = "s_max_pu" in n.lines.columns

    for idx, row in n.lines.iterrows():
        c0 = get_country(row.bus0)
        c1 = get_country(row.bus1)

        # Filter: Inländische AC-Leitungen ignorieren wir für den Map-Plot
        # (Sonst sieht man vor lauter Linien in DE nichts mehr)
        if c0 == c1:
            continue

        # Filter: EU Knoten weg
        if c0 in IGNORE_CODES or c1 in IGNORE_CODES: continue

        pair = tuple(sorted([row.bus0, row.bus1]))
        raw_cap = row[col_ac]
        if raw_cap < 1: continue

        factor = row.s_max_pu if has_s_max else DEFAULT_S_MAX_PU_AC
        usable_cap = raw_cap * factor

        if pair not in connections: connections[pair] = 0.0
        connections[pair] += usable_cap

    # --- 2. DC Links ---
    dc_links = n.links[n.links.carrier == "DC"]
    col_dc = "p_nom_opt" if "p_nom_opt" in n.links.columns else "p_nom"
    has_p_max = "p_max_pu" in n.links.columns

    for idx, row in dc_links.iterrows():
        if "-reversed" in str(idx): continue

        c0 = get_country(row.bus0)
        c1 = get_country(row.bus1)

        # Filter: EU Knoten weg
        if c0 in IGNORE_CODES or c1 in IGNORE_CODES: continue

        # HIER IST DER UNTERSCHIED:
        # Wir erlauben DC auch im Inland (c0 == c1)!
        # Das rettet Mallorca (ES-ES) und Nordirland (GB-GB).

        pair = tuple(sorted([row.bus0, row.bus1]))
        raw_cap = row[col_dc]
        if raw_cap < 1: continue

        factor = row.p_max_pu if has_p_max else DEFAULT_P_MAX_PU_DC
        usable_cap = raw_cap * factor

        if pair not in connections: connections[pair] = 0.0
        connections[pair] += usable_cap

    return connections


def print_table_from_dict(connections):
    """Gibt eine aggregierte Tabelle aus."""
    border_sums = {}
    for (b0, b1), cap in connections.items():
        # Name bauen
        border = get_border_name(b0, b1)
        # Fallback für Inlands-DC (z.B. ES0-ES1)
        if not border and get_country(b0) == get_country(b1):
            border = f"{get_country(b0)} (Internal DC)"

        if border:
            if border not in border_sums: border_sums[border] = 0.0
            border_sums[border] += cap

    sorted_borders = sorted(border_sums.items())
    print("-" * 60)
    for i in range(0, len(sorted_borders), 2):
        b1, c1 = sorted_borders[i]
        str1 = f"{b1}: {c1 / 1e3:.2f} GW"
        if i + 1 < len(sorted_borders):
            b2, c2 = sorted_borders[i + 1]
            str2 = f"{b2}: {c2 / 1e3:.2f} GW"
            print(f"{str1:<30} | {str2}")
        else:
            print(f"{str1}")
    print("-" * 60 + "\n")


# =============================================================================
# PLOTTING FUNKTION
# =============================================================================

def plot_ntc_map(n, year, scenario_name, output_dir):
    print(f"  -> Erstelle Karte für {year}...")

    # 1. Daten berechnen
    connections = calculate_usable_capacities(n)
    print_table_from_dict(connections)

    # 2. Liniensegmente erstellen
    segments = []
    linewidths = []

    bus_x = n.buses.x
    bus_y = n.buses.y

    for (b0, b1), cap in connections.items():
        if b0 not in bus_x or b1 not in bus_x: continue

        p0 = (bus_x[b0], bus_y[b0])
        p1 = (bus_x[b1], bus_y[b1])
        segments.append([p0, p1])

        width = cap / SCALE_FACTOR
        linewidths.append(width)

    # 3. Plot Setup
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.EqualEarth()})

    # --- KNOTEN FILTERN ---
    # Wir wollen nur echte Knoten plotten, keine "EU"-Hilfsknoten
    bus_sizes = pd.Series(0.0, index=n.buses.index)
    bus_colors = pd.Series("white", index=n.buses.index)

    # Zeige nur Knoten, die Teil unserer Verbindungen sind
    active_buses = set()
    for (b0, b1) in connections.keys():
        active_buses.add(b0)
        active_buses.add(b1)

    for b in active_buses:
        # Extra Check: Keine EU-Namen
        if "EU" in str(b): continue
        if b not in bus_sizes.index: continue

        bus_sizes[b] = 0.2
        bus_colors[b] = COLOR_NODE

    # Basiskarte ohne PyPSA-Linien
    n.plot(
        ax=ax,
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        line_widths=0,
        link_widths=0,
        geomap=True
    )

    # Unsere berechneten Linien hinzufügen
    lc = LineCollection(
        segments,
        linewidths=linewidths,
        colors=COLOR_BORDER_LINE,
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=2
    )
    ax.add_collection(lc)

    ax.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5, alpha=0.5)

    ax.set_title(f"Nutzbare Übertragungskapazität {year}\n", fontsize=18)

    # Legende
    legend_elements = [
        Line2D([0], [0], color=COLOR_BORDER_LINE, lw=3, label='Nutzbare Kapazität'),
        Line2D([0], [0], color='gray', lw=(5000 / SCALE_FACTOR), label='Ref: 5 GW'),
        Line2D([0], [0], color='gray', lw=(10000 / SCALE_FACTOR), label='Ref: 10 GW'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=11)

    fname = f"Final_NTC_Map_{scenario_name}_{year}.png"
    save_path = os.path.join(output_dir, fname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     Gespeichert: {fname}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = PlottingConfig()
    output_dir = os.path.join(cfg.PLOT_OUTPUT_PATH, "grid_maps_final")
    os.makedirs(output_dir, exist_ok=True)

    if "CONDA_PREFIX" in os.environ:
        proj_lib = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
        if os.path.exists(proj_lib):
            os.environ["PROJ_LIB"] = proj_lib

    networks_data = cfg.get_networks()
    if networks_data is None: return

    networks_dict = {}
    if isinstance(networks_data, list):
        networks_dict[cfg.SCENARIO_SELECTION] = networks_data
    elif isinstance(networks_data, dict):
        networks_dict = networks_data
    else:
        return

    for scenario_name, file_list in networks_dict.items():
        print(f"\n=== Bearbeite Szenario: {scenario_name} ===")
        year_map = {}
        for f in file_list:
            match = re.search(r"20\d{2}", f)
            if match:
                year_map[int(match.group(0))] = f

        for year in sorted(year_map.keys()):
            path = year_map[year]
            try:
                n = pypsa.Network(path)
                plot_ntc_map(n, year, scenario_name, output_dir)
            except Exception as e:
                print(f"Fehler bei Jahr {year}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()