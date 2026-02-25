import pypsa
import matplotlib

matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import re
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from config_final import PlottingConfig

# =============================================================================
# EINSTELLUNGEN
# =============================================================================

DF_START_DAY = 7
DF_START_MONTH = 1
DF_DURATION_DAYS = 21

# Skalierung: Kleinerer Wert = Dickere Linien
# Vorher: 5e3. Jetzt: 1.5e3 (Massiv dicker)
SCALE_FACTOR = 1.5e3

FLOW_THRESHOLD_MW = 100
COLOR_FLOW = "#1f77b4"
COLOR_NODE = "black"
IGNORE_COUNTRIES = ["EU", "co", "H2", "gas"]

# Manuelle Korrektur für Mittelpunkte (Lon, Lat)
# Damit FR nicht im Mittelmeer liegt und NO nicht im Nordpol
MANUAL_CENTROIDS = {
    "FR": (2.0, 47.0),  # Zentralfrankreich
    "NO": (8.5, 61.0),  # Süd-Norwegen (wo der Strom ist)
    "SE": (16.0, 62.0),  # Etwas zentrierter
    "GB": (-1.5, 54.5),  # England/Midlands
    "IT": (12.5, 43.0)  # Italien zentrierter
}


# =============================================================================
# HELPER
# =============================================================================

def get_country_centroids(n):
    countries = n.buses.index.str[:2]
    df = pd.DataFrame({"x": n.buses.x, "y": n.buses.y, "country": countries})
    mask = df["country"].str.match(r'^[A-Z]{2}$') & ~df["country"].isin(IGNORE_COUNTRIES)

    # Berechnete Mittelwerte
    centroids = df[mask].groupby("country")[["x", "y"]].mean()

    # Manuelle Überschreibung anwenden
    for c, (x, y) in MANUAL_CENTROIDS.items():
        if c in centroids.index:
            centroids.at[c, "x"] = x
            centroids.at[c, "y"] = y

    return centroids


def aggregate_border_flows(n, snapshots):
    """Aggregiert Grenzflüsse zwischen Ländern.

    Wichtig: DC-Links haben oft bidirektionale Paare (A→B und B→A),
    die gegengerechnet werden müssen für den Netto-Fluss.
    """
    flow_lines = n.lines_t.p0.loc[snapshots].mean(axis=0).fillna(0)

    flow_links = pd.Series(dtype=float)
    is_dc = n.links.carrier == "DC"
    if any(is_dc):
        col = "p0" if "p0" in n.links_t else "p_dispatch"
        valid_cols = n.links[is_dc].index.intersection(n.links_t[col].columns)
        flow_links = n.links_t[col].loc[snapshots, valid_cols].mean(axis=0).fillna(0)

    border_flows = {}

    def process_components(components_df, flow_series):
        for name, flow in flow_series.items():
            if name not in components_df.index: continue
            bus0, bus1 = components_df.at[name, "bus0"], components_df.at[name, "bus1"]
            c0, c1 = bus0[:2], bus1[:2]

            if c0 == c1: continue
            if c0 in IGNORE_COUNTRIES or c1 in IGNORE_COUNTRIES: continue
            if not (c0.isalpha() and c1.isalpha()): continue

            # Sortiere IMMER alphabetisch für konsistente Aggregation
            # Positiver Wert = Fluss von c0 nach c1
            if c0 < c1:
                key = (c0, c1)
                val = flow  # Fluss von c0→c1 ist positiv
            else:
                key = (c1, c0)
                val = -flow  # Fluss von c1→c0, aber Key ist (c0,c1), also negativ

            # Aggregiere: DC-Gegenflüsse werden automatisch verrechnet
            if key not in border_flows:
                border_flows[key] = 0.0
            border_flows[key] += val

    # AC-Leitungen verarbeiten
    process_components(n.lines, flow_lines)

    # DC-Links verarbeiten (hin- und Rückfluss werden gegengerechnet)
    process_components(n.links, flow_links)

    return border_flows


def calculate_net_flows_per_country(border_flows):
    """Berechnet Netto-Im/Export pro Land (ohne Transit)."""
    country_balance = {}

    for (c0, c1), flow in border_flows.items():
        if abs(flow) < FLOW_THRESHOLD_MW:
            continue

        # c0 → c1: flow positiv bedeutet c0 exportiert, c1 importiert
        if c0 not in country_balance:
            country_balance[c0] = 0.0
        if c1 not in country_balance:
            country_balance[c1] = 0.0

        country_balance[c0] += flow  # Export ist positiv
        country_balance[c1] -= flow  # Import ist negativ

    return country_balance


def print_border_flows_table(border_flows, title, hours):
    """Gibt die Grenzflüsse als formatierte Tabelle aus.

    Args:
        border_flows: Dict mit Grenzflüssen in MW (Durchschnitt)
        title: Titel der Ausgabe
        hours: Anzahl Stunden für TWh-Berechnung (8760 für Jahr, 504 für 21 Tage)
    """
    print(f"\n{title}")
    print("=" * 85)

    # Netto-Bilanz pro Land berechnen
    country_balance = calculate_net_flows_per_country(border_flows)

    print("\n  Länder-Nettofluss (Import/Export, ohne Transit):")
    print("  " + "-" * 65)

    # Nach absolutem Wert sortieren
    sorted_countries = sorted(country_balance.items(), key=lambda x: x[1], reverse=True)

    for country, balance in sorted_countries:
        balance_gw = balance / 1000  # MW → GW
        balance_twh = (balance * hours) / 1_000_000  # MW × h → TWh
        if balance > 0:
            print(f"    {country}:  +{balance_gw:>6.2f} GW  (+{balance_twh:>7.2f} TWh)  (Netto-Export)")
        else:
            print(f"    {country}:  {balance_gw:>7.2f} GW  ({balance_twh:>8.2f} TWh)  (Netto-Import)")

    # Alle Grenzflüsse einzeln
    print("\n  Alle Grenzüberschreitenden Flüsse:")
    print("  " + "-" * 65)
    sorted_flows = sorted(border_flows.items(), key=lambda x: abs(x[1]), reverse=True)

    for (c0, c1), flow in sorted_flows:
        if abs(flow) < FLOW_THRESHOLD_MW:
            continue

        flow_gw = flow / 1000  # MW → GW
        flow_twh = (abs(flow) * hours) / 1_000_000  # MW × h → TWh
        direction = "→" if flow > 0 else "←"
        abs_flow_gw = abs(flow_gw)

        if flow > 0:
            print(f"    {c0:>2} {direction} {c1:<2}  {abs_flow_gw:>6.2f} GW  ({flow_twh:>6.2f} TWh)")
        else:
            print(f"    {c1:>2} {direction} {c0:<2}  {abs_flow_gw:>6.2f} GW  ({flow_twh:>6.2f} TWh)")

    print("=" * 85)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_border_map(border_flows, centroids, title, output_path):
    print(f"  -> Zeichne: {title}")
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": ccrs.EqualEarth()})

    # Ländergrenzen dicker
    ax.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=1.2, alpha=0.6)
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=1.2, alpha=0.6)
    ax.set_extent([-10, 30, 36, 70])

    for (c0, c1), net_flow in border_flows.items():
        if abs(net_flow) < FLOW_THRESHOLD_MW: continue
        if c0 not in centroids.index or c1 not in centroids.index: continue

        p0, p1 = centroids.loc[c0], centroids.loc[c1]
        width = abs(net_flow) / SCALE_FACTOR

        # Import-Linien 1.5x dicker
        width = width * 1.5

        start_x, start_y, end_x, end_y = p0.x, p0.y, p1.x, p1.y
        if net_flow < 0:
            start_x, start_y, end_x, end_y = end_x, end_y, start_x, start_y

        # Linie zeichnen
        ax.plot([start_x, end_x], [start_y, end_y], color=COLOR_FLOW,
                linewidth=width, alpha=0.8, transform=ccrs.PlateCarree(), solid_capstyle='round')

        # Pfeil zeichnen (Mitte) - fett und größer
        mx, my = (start_x + end_x) / 2, (start_y + end_y) / 2
        dx, dy = end_x - start_x, end_y - start_y

        # Nur Pfeil wenn Linie sichtbar ist
        if width > 0.5:
            # Pfeil fett und größer (lw=2.5, mutation_scale=25)
            ax.annotate("", xy=(mx + dx * 0.01, my + dy * 0.01), xytext=(mx, my),
                        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2.5, mutation_scale=25))

    # Knotenpunkte
    ax.scatter(centroids.x, centroids.y, transform=ccrs.PlateCarree(), s=40, c=COLOR_NODE, zorder=5)

    # Titel größer, NICHT fett
    ax.set_title(title, fontsize=19, fontweight='normal')

    # Legende größer
    legend_elements = [
        Line2D([0], [0], color=COLOR_FLOW, lw=(2000 / SCALE_FACTOR * 1.5), label='Netto: 2 GW'),
        Line2D([0], [0], color=COLOR_FLOW, lw=(10000 / SCALE_FACTOR * 1.5), label='Netto: 10 GW'),
        Line2D([0], [0], color='black', marker='>', markersize=18, lw=0, label='Flussrichtung')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=16)

    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.90
    )
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"     Gespeichert: {os.path.basename(output_path)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = PlottingConfig()
    output_dir = os.path.join(cfg.PLOT_OUTPUT_PATH, "border_flows_clean")
    os.makedirs(output_dir, exist_ok=True)

    if "CONDA_PREFIX" in os.environ:
        proj_lib = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
        if os.path.exists(proj_lib): os.environ["PROJ_LIB"] = proj_lib

    networks_data = cfg.get_networks()
    if networks_data is None: return

    networks_dict = {}
    if isinstance(networks_data, list):
        networks_dict[cfg.SCENARIO_SELECTION] = networks_data
    elif isinstance(networks_data, dict):
        networks_dict = networks_data

    for scenario_name, file_list in networks_dict.items():
        print(f"\n=== Bearbeite Szenario: {scenario_name} ===")
        year_map = {int(re.search(r"20\d{2}", f).group(0)): f for f in file_list if re.search(r"20\d{2}", f)}

        for year in sorted(year_map.keys()):
            path = year_map[year]
            try:
                n = pypsa.Network(path)
                centroids = get_country_centroids(n)

                # 1. Jahresmittel
                flows_yr = aggregate_border_flows(n, n.snapshots)
                print_border_flows_table(flows_yr, f"📊 Jahresmittel {year} - Grenzüberschreitende Flüsse", 8760)
                plot_border_map(flows_yr, centroids,
                                f"Netto-Länderaustausch {year}",
                                os.path.join(output_dir, f"BorderFlow_Yearly_{scenario_name}_{year}.png"))

                # 2. Dunkelflaute
                sim_year = n.snapshots[0].year
                start = f"{sim_year}-{DF_START_MONTH:02d}-{DF_START_DAY:02d}"
                end = str(pd.Timestamp(start) + pd.Timedelta(days=DF_DURATION_DAYS))
                snaps_df = n.snapshots[(n.snapshots >= start) & (n.snapshots <= end)]

                if not snaps_df.empty:
                    flows_df = aggregate_border_flows(n, snaps_df)
                    df_hours = len(snaps_df)  # Tatsächliche Anzahl Stunden in der DF-Periode
                    print_border_flows_table(flows_df, f"🌑 Dunkelflaute {year} - Grenzüberschreitende Flüsse", df_hours)
                    plot_border_map(flows_df, centroids,
                                    f"Netto-Länderaustausch DF {year}",
                                    os.path.join(output_dir, f"BorderFlow_DF_{scenario_name}_{year}.png"))

            except Exception as e:
                print(f"Fehler bei Jahr {year}: {e}")


if __name__ == "__main__":
    main()