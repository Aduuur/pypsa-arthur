#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nutzbare Übertragungskapazität (NTC-Proxy)
==========================================
Funktioniert im ARO-Modus (Dispatch-Netze) und im normalen Modus.
Ausgabe in config.PLOT_OUTPUT_PATH/grid_maps/.

ARO-Fix: Jahr aus n.snapshots[0].year statt Dateinamen-Regex.
"""

import os
import re
import pypsa
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

# FIX: war config_final
from master_config import get_planning_year, PlottingConfig

# =============================================================================
# EINSTELLUNGEN
# =============================================================================
SCALE_FACTOR         = 1.5e3
COLOR_BORDER_LINE    = "#2a6fdb"
COLOR_NODE           = "black"
IGNORE_CODES         = ["EU"]
DEFAULT_S_MAX_PU_AC  = 0.7
DEFAULT_P_MAX_PU_DC  = 1.0


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def _parse_year(n: pypsa.Network, path: str, fallback: int = 2050) -> int:
    """Jahr aus Netz-Snapshots lesen — funktioniert auch für Dispatch-Netze."""
    try:
        return get_planning_year(n, path)
    except Exception:
        pass
    m = re.search(r"___(\d{4})\.nc$", path) or re.search(r"_(\d{4})\.nc", path)
    return int(m.group(1)) if m else fallback


def get_country(bus_name: str) -> str:
    s = str(bus_name)
    if "EU" in s: return "EU"
    return s[:2]


def get_border_name(bus0: str, bus1: str):
    c0, c1 = get_country(bus0), get_country(bus1)
    if c0 in IGNORE_CODES or c1 in IGNORE_CODES: return None
    return "-".join(sorted([c0, c1]))


def calculate_usable_capacities(n) -> dict:
    connections = {}

    # --- AC Lines ---
    col_ac   = "s_nom_opt" if "s_nom_opt" in n.lines.columns else "s_nom"
    has_smax = "s_max_pu" in n.lines.columns

    for idx, row in n.lines.iterrows():
        c0, c1 = get_country(row.bus0), get_country(row.bus1)
        if c0 == c1: continue
        if c0 in IGNORE_CODES or c1 in IGNORE_CODES: continue
        pair     = tuple(sorted([row.bus0, row.bus1]))
        raw_cap  = row[col_ac]
        if raw_cap < 1: continue
        factor   = row.s_max_pu if has_smax else DEFAULT_S_MAX_PU_AC
        connections[pair] = connections.get(pair, 0.0) + raw_cap * factor

    # --- DC Links ---
    dc_links = n.links[n.links.carrier == "DC"]
    col_dc   = "p_nom_opt" if "p_nom_opt" in n.links.columns else "p_nom"
    has_pmax = "p_max_pu" in n.links.columns

    for idx, row in dc_links.iterrows():
        if "-reversed" in str(idx): continue
        c0, c1 = get_country(row.bus0), get_country(row.bus1)
        if c0 in IGNORE_CODES or c1 in IGNORE_CODES: continue
        pair    = tuple(sorted([row.bus0, row.bus1]))
        raw_cap = row[col_dc]
        if raw_cap < 1: continue
        factor  = row.p_max_pu if has_pmax else DEFAULT_P_MAX_PU_DC
        connections[pair] = connections.get(pair, 0.0) + raw_cap * factor

    return connections


def print_table_from_dict(connections: dict) -> None:
    border_sums: dict = {}
    for (b0, b1), cap in connections.items():
        border = get_border_name(b0, b1)
        if not border and get_country(b0) == get_country(b1):
            border = f"{get_country(b0)} (Internal DC)"
        if border:
            border_sums[border] = border_sums.get(border, 0.0) + cap

    sorted_borders = sorted(border_sums.items())
    print("-" * 60)
    for i in range(0, len(sorted_borders), 2):
        b1, c1 = sorted_borders[i]
        s1 = f"{b1}: {c1/1e3:.2f} GW"
        if i + 1 < len(sorted_borders):
            b2, c2 = sorted_borders[i + 1]
            print(f"{s1:<30} | {b2}: {c2/1e3:.2f} GW")
        else:
            print(s1)
    print("-" * 60 + "\n")


# =============================================================================
# PLOT
# =============================================================================

def plot_ntc_map(n, year: int, scenario_name: str, output_dir: str) -> None:
    print(f"  -> Erstelle Karte für {year}...")
    connections = calculate_usable_capacities(n)
    print_table_from_dict(connections)

    segments   = []
    linewidths = []
    bus_x = n.buses.x
    bus_y = n.buses.y

    for (b0, b1), cap in connections.items():
        if b0 not in bus_x or b1 not in bus_x: continue
        segments.append([(bus_x[b0], bus_y[b0]), (bus_x[b1], bus_y[b1])])
        linewidths.append(cap / SCALE_FACTOR)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.EqualEarth()})

    bus_sizes  = pd.Series(0.0,     index=n.buses.index)
    bus_colors = pd.Series("white", index=n.buses.index)
    active_buses = {b for pair in connections for b in pair}
    for b in active_buses:
        if "EU" in str(b) or b not in bus_sizes.index: continue
        bus_sizes[b]  = 0.2
        bus_colors[b] = COLOR_NODE

    n.plot(ax=ax, bus_sizes=bus_sizes, bus_colors=bus_colors,
           line_widths=0, link_widths=0, geomap=True)

    lc = LineCollection(
        segments, linewidths=linewidths,
        colors=COLOR_BORDER_LINE, alpha=0.9,
        transform=ccrs.PlateCarree(), zorder=2,
    )
    ax.add_collection(lc)
    ax.add_feature(cfeature.BORDERS,   edgecolor="gray", linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, edgecolor="gray", linewidth=0.5, alpha=0.5)
    ax.set_title(f"Nutzbare Übertragungskapazität {year}\n", fontsize=18)

    legend_elements = [
        Line2D([0], [0], color=COLOR_BORDER_LINE, lw=3, label="Nutzbare Kapazität"),
        Line2D([0], [0], color="gray", lw=5000/SCALE_FACTOR, label="Ref: 5 GW"),
        Line2D([0], [0], color="gray", lw=10000/SCALE_FACTOR, label="Ref: 10 GW"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=11)

    fname     = f"Final_NTC_Map_{scenario_name}_{year}.png"
    save_path = os.path.join(output_dir, fname)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"     Gespeichert: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = PlottingConfig()

    # FIX: PLOT_OUTPUT_PATH statt hardcodiertem Pfad
    output_dir = os.path.join(cfg.PLOT_OUTPUT_PATH, "grid_maps")
    os.makedirs(output_dir, exist_ok=True)

    if "CONDA_PREFIX" in os.environ:
        proj_lib = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
        if os.path.exists(proj_lib):
            os.environ["PROJ_LIB"] = proj_lib

    networks_data = cfg.get_networks()
    if networks_data is None:
        return

    networks_dict = (
        {cfg.SCENARIO_SELECTION: networks_data}
        if isinstance(networks_data, list)
        else networks_data
    )

    for scenario_name, file_list in networks_dict.items():
        print(f"\n=== Bearbeite Szenario: {scenario_name} ===")
        for path in file_list:
            if not os.path.isfile(path):
                print(f"  Datei fehlt: {path}")
                continue
            try:
                n    = pypsa.Network(path)
                # FIX: Jahr aus Snapshots statt Dateinamen-Regex
                year = _parse_year(n, path, fallback=2050)
                plot_ntc_map(n, year, scenario_name, output_dir)
            except Exception as e:
                print(f"Fehler: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()