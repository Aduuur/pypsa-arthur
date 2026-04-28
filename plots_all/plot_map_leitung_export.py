#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grenzüberschreitende Stromflüsse — Jahres- und Dunkelflautekarte
=================================================================
Funktioniert im ARO-Modus (Dispatch-Netze) und im normalen Modus.
Ausgabe in config.PLOT_OUTPUT_PATH/border_flows/.

ARO-Fix: Jahr aus n.snapshots[0].year statt Dateinamen-Regex.
         Dunkelflaute-Fenster aus config.DARK_SKY_START/END.
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

# FIX: war config_final
from master_config import get_planning_year, PlottingConfig

# =============================================================================
# EINSTELLUNGEN
# =============================================================================
SCALE_FACTOR    = 1.5e3
FLOW_THRESHOLD_MW = 100
COLOR_FLOW      = "#1f77b4"
COLOR_NODE      = "black"
IGNORE_COUNTRIES = ["EU", "co", "H2", "gas"]

MANUAL_CENTROIDS = {
    "FR": (2.0,  47.0),
    "NO": (8.5,  61.0),
    "SE": (16.0, 62.0),
    "GB": (-1.5, 54.5),
    "IT": (12.5, 43.0),
}


# =============================================================================
# HELPER
# =============================================================================

def _parse_year(n: pypsa.Network, path: str, fallback: int = 2050) -> int:
    """Jahr aus Netz-Snapshots lesen — funktioniert auch für Dispatch-Netze."""
    try:
        return get_planning_year(n, path)
    except Exception:
        pass
    m = re.search(r"___(\d{4})\.nc$", path) or re.search(r"_(\d{4})\.nc", path)
    return int(m.group(1)) if m else fallback


def get_country_centroids(n):
    countries = n.buses.index.str[:2]
    df = pd.DataFrame({"x": n.buses.x, "y": n.buses.y, "country": countries})
    mask = df["country"].str.match(r"^[A-Z]{2}$") & ~df["country"].isin(IGNORE_COUNTRIES)
    centroids = df[mask].groupby("country")[["x", "y"]].mean()
    for c, (x, y) in MANUAL_CENTROIDS.items():
        if c in centroids.index:
            centroids.at[c, "x"] = x
            centroids.at[c, "y"] = y
    return centroids


def aggregate_border_flows(n, snapshots):
    flow_lines = n.lines_t.p0.loc[snapshots].mean(axis=0).fillna(0)
    flow_links = pd.Series(dtype=float)
    is_dc = n.links.carrier == "DC"
    if any(is_dc):
        col       = "p0" if "p0" in n.links_t else "p_dispatch"
        valid_cols = n.links[is_dc].index.intersection(n.links_t[col].columns)
        flow_links = n.links_t[col].loc[snapshots, valid_cols].mean(axis=0).fillna(0)

    border_flows = {}

    def process(components_df, flow_series):
        for name, flow in flow_series.items():
            if name not in components_df.index: continue
            bus0, bus1 = components_df.at[name, "bus0"], components_df.at[name, "bus1"]
            c0, c1 = bus0[:2], bus1[:2]
            if c0 == c1: continue
            if c0 in IGNORE_COUNTRIES or c1 in IGNORE_COUNTRIES: continue
            if not (c0.isalpha() and c1.isalpha()): continue
            if c0 < c1:
                key, val = (c0, c1), flow
            else:
                key, val = (c1, c0), -flow
            border_flows[key] = border_flows.get(key, 0.0) + val

    process(n.lines, flow_lines)
    process(n.links,  flow_links)
    return border_flows


def calculate_net_flows_per_country(border_flows):
    balance = {}
    for (c0, c1), flow in border_flows.items():
        if abs(flow) < FLOW_THRESHOLD_MW: continue
        balance[c0] = balance.get(c0, 0.0) + flow
        balance[c1] = balance.get(c1, 0.0) - flow
    return balance


def print_border_flows_table(border_flows, title, hours):
    print(f"\n{title}")
    print("=" * 85)
    country_balance = calculate_net_flows_per_country(border_flows)
    print("\n  Länder-Nettofluss (Import/Export, ohne Transit):")
    print("  " + "-" * 65)
    for country, balance in sorted(country_balance.items(), key=lambda x: x[1], reverse=True):
        bg = balance / 1000
        bt = (balance * hours) / 1_000_000
        sym = f"+{bg:>6.2f} GW  (+{bt:>7.2f} TWh)  (Netto-Export)" if balance > 0 \
              else f"{bg:>7.2f} GW  ({bt:>8.2f} TWh)  (Netto-Import)"
        print(f"    {country}:  {sym}")

    print("\n  Alle Grenzüberschreitenden Flüsse:")
    print("  " + "-" * 65)
    for (c0, c1), flow in sorted(border_flows.items(), key=lambda x: abs(x[1]), reverse=True):
        if abs(flow) < FLOW_THRESHOLD_MW: continue
        fg = flow / 1000
        ft = (abs(flow) * hours) / 1_000_000
        d  = "→" if flow > 0 else "←"
        src, dst = (c0, c1) if flow > 0 else (c1, c0)
        print(f"    {src:>2} {d} {dst:<2}  {abs(fg):>6.2f} GW  ({ft:>6.2f} TWh)")
    print("=" * 85)


def plot_border_map(border_flows, centroids, title, output_path):
    print(f"  -> Zeichne: {title}")
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": ccrs.EqualEarth()})
    ax.add_feature(cfeature.BORDERS,   edgecolor="gray", linewidth=1.2, alpha=0.6)
    ax.add_feature(cfeature.COASTLINE, edgecolor="gray", linewidth=1.2, alpha=0.6)
    ax.set_extent([-10, 30, 36, 70])

    for (c0, c1), net_flow in border_flows.items():
        if abs(net_flow) < FLOW_THRESHOLD_MW: continue
        if c0 not in centroids.index or c1 not in centroids.index: continue
        p0, p1 = centroids.loc[c0], centroids.loc[c1]
        width  = abs(net_flow) / SCALE_FACTOR * 1.5
        sx, sy, ex, ey = p0.x, p0.y, p1.x, p1.y
        if net_flow < 0:
            sx, sy, ex, ey = ex, ey, sx, sy
        ax.plot([sx, ex], [sy, ey], color=COLOR_FLOW, linewidth=width,
                alpha=0.8, transform=ccrs.PlateCarree(), solid_capstyle="round")
        if width > 0.5:
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            dx, dy = ex - sx, ey - sy
            ax.annotate("", xy=(mx + dx * 0.01, my + dy * 0.01), xytext=(mx, my),
                        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2.5, mutation_scale=25))

    ax.scatter(centroids.x, centroids.y, transform=ccrs.PlateCarree(), s=40, c=COLOR_NODE, zorder=5)
    ax.set_title(title, fontsize=19, fontweight="normal")

    legend_elements = [
        Line2D([0], [0], color=COLOR_FLOW, lw=(2000 / SCALE_FACTOR * 1.5), label="Netto: 2 GW"),
        Line2D([0], [0], color=COLOR_FLOW, lw=(10000 / SCALE_FACTOR * 1.5), label="Netto: 10 GW"),
        Line2D([0], [0], color="black", marker=">", markersize=18, lw=0, label="Flussrichtung"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=16)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"     Gespeichert: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = PlottingConfig()

    # FIX: PLOT_OUTPUT_PATH statt hardcodiertem Pfad
    output_dir = os.path.join(cfg.PLOT_OUTPUT_PATH, "border_flows")
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

    # Dunkelflaute-Fenster aus master_config
    df_start = cfg.DARK_SKY_START
    df_end   = cfg.DARK_SKY_END

    for scenario_name, file_list in networks_dict.items():
        print(f"\n=== Bearbeite Szenario: {scenario_name} ===")
        for path in file_list:
            if not os.path.isfile(path):
                continue
            try:
                n    = pypsa.Network(path)
                # FIX: Jahr aus Snapshots, nicht aus Dateinamen
                year = _parse_year(n, path, fallback=2050)
                centroids = get_country_centroids(n)

                # 1. Jahresmittel
                flows_yr = aggregate_border_flows(n, n.snapshots)
                print_border_flows_table(
                    flows_yr,
                    f"📊 Jahresmittel {year} - Grenzüberschreitende Flüsse",
                    8760,
                )
                plot_border_map(
                    flows_yr, centroids,
                    f"Netto-Länderaustausch {year}",
                    os.path.join(output_dir, f"BorderFlow_Yearly_{scenario_name}_{year}.png"),
                )

                # 2. Dunkelflaute (aus master_config)
                snaps_df = n.snapshots[
                    (n.snapshots >= df_start) & (n.snapshots <= df_end)
                ]
                if not snaps_df.empty:
                    flows_df = aggregate_border_flows(n, snaps_df)
                    print_border_flows_table(
                        flows_df,
                        f"🌑 Dunkelflaute {year} - Grenzüberschreitende Flüsse",
                        len(snaps_df),
                    )
                    plot_border_map(
                        flows_df, centroids,
                        f"Netto-Länderaustausch DF {year}",
                        os.path.join(output_dir, f"BorderFlow_DF_{scenario_name}_{year}.png"),
                    )
                else:
                    print(f"  ⚠️  Dunkelflaute-Fenster {df_start}–{df_end} nicht im Netz.")

            except Exception as e:
                print(f"Fehler: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()