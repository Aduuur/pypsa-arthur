#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import numpy as np

from config_final import PlottingConfig

# =====================================================================
# PARAMETER
# =====================================================================
START = "2005-01-07"
END = "2005-01-28"
SAVE_STEM = "price_delta_map_Jan7_3weeks_final"

# Modell-Länder
MODEL_COUNTRIES = [
    "DE", "FR", "PL", "CZ", "AT", "IT", "ES", "BE", "NL",
    "DK", "SE", "PT", "FI", "LT", "LV", "EE",
    "NO", "GB", "CH"
]

# FALLBACK: Name -> ISO Mapping für Natural Earth Probleme
NAME_TO_ISO = {
    "France": "FR",
    "Norway": "NO",
    "United Kingdom": "GB",
    "Germany": "DE",
    "Spain": "ES",
}


# =====================================================================
# FUNKTIONEN
# =====================================================================
def extract_year(path):
    return int(path.split("___")[-1].split(".")[0])


def normalize_country_code(code):
    if code is None:
        return None
    code = code.upper()
    if code in MODEL_COUNTRIES:
        return code
    for pref in MODEL_COUNTRIES:
        if code.startswith(pref):
            return pref
    return None


def get_country_iso(rec):
    """Robustes ISO-A2 Mapping mit Fallback"""
    iso_a2 = rec.attributes.get("ISO_A2", "").strip()
    name = rec.attributes.get("NAME", "").strip()

    # 1. Primär: ISO_A2
    if iso_a2 in MODEL_COUNTRIES:
        return iso_a2

    # 2. Fallback: Name -> ISO
    if name in NAME_TO_ISO:
        return NAME_TO_ISO[name]

    # 3. ISO_A3 Fallback (für -99 etc.)
    iso_a3 = rec.attributes.get("ISO_A3", "")
    if iso_a3 == "FRA": return "FR"
    if iso_a3 == "NOR": return "NO"

    return None


def get_total_electric_load(network, start, end):
    """
    Hilfsfunktion: Berechnet die GESAMTE elektrische Last.
    Summe aus:
      1. Statischer Last (loads_t.p_set)
      2. Wärmepumpen & Heizstäbe (links_t.p0)
    """
    # 1. Statische Last (Licht, Industrie etc.)
    loads_t = network.loads_t.p_set.loc[start:end]
    loads_t = loads_t.reindex(columns=network.loads.index).fillna(network.loads.p_set)
    total_load_by_bus = loads_t.groupby(network.loads.bus, axis=1).sum()

    # 2. Sektorkopplung (Wärmepumpen etc.) addieren
    if "carrier" in network.links.columns:
        mask = network.links.carrier.str.contains("heat pump|resistive heater", case=False, regex=True)
        heat_links = network.links.index[mask]

        if not heat_links.empty:
            link_consumption = network.links_t.p0.loc[start:end, heat_links]
            link_bus_map = network.links.loc[heat_links, "bus0"]
            link_load_by_bus = link_consumption.groupby(link_bus_map, axis=1).sum()
            total_load_by_bus = total_load_by_bus.add(link_load_by_bus, fill_value=0)
            print(f"   -> Habe {len(heat_links)} Wärmepumpen/Heizstäbe zur Last addiert.")

    return total_load_by_bus


def get_load_weighted_prices(network, start, end):
    """
    1. Berechnet den lastgewichteten Durchschnittspreis (VWAP).
    Summe(Preis * Last) / Summe(Last) über den gesamten Zeitraum.
    """
    total_load = get_total_electric_load(network, start, end)
    prices = network.buses_t.marginal_price.loc[start:end]

    common_buses = prices.columns.intersection(total_load.columns)
    p_aligned = prices[common_buses]
    l_aligned = total_load[common_buses]

    revenue = p_aligned * l_aligned

    group_key = lambda x: x[:2]
    total_revenue_country = revenue.groupby(group_key, axis=1).sum().sum(axis=0)
    total_load_country = l_aligned.groupby(group_key, axis=1).sum().sum(axis=0)

    return total_revenue_country / total_load_country.replace(0, np.nan)


def get_simple_average_prices(network, start, end):
    """
    2. Berechnet den einfachen zeitlichen Durchschnittspreis (Time-Weighted).
    Einfacher Mean der marginal_price Series.
    """
    prices = network.buses_t.marginal_price.loc[start:end]
    group_key = lambda x: x[:2]
    mean_prices_country = prices.groupby(group_key, axis=1).mean()
    return mean_prices_country.mean(axis=0)


# =====================================================================
# PLOT-FUNKTION
# =====================================================================
def plot_delta_map(delta_series, year, suffix, cfg, records, title_override=None,
                   vmin=-200, vmax=200, tick_step=50):
    """
    Erstellt die Karte.
    vmin, vmax: Grenzen der Farbskala
    tick_step: Abstand der Ticks auf der Colorbar
    """
    delta_dict_iso = {}
    for bus_code, val in delta_series.items():
        iso = normalize_country_code(bus_code)
        if iso in MODEL_COUNTRIES:
            delta_dict_iso[iso] = float(val)

    if title_override:
        main_title = title_override
    elif "weighted" in suffix:
        main_title = "Differenz der lastgewichteten Strompreise (VWAP)"
    else:
        main_title = "Differenz der mittleren Strompreise (Time-Weighted)"

    TITLE = f"{main_title}\nDunkelflaute vs. Referenzszenario {year} (07. – 28. Jan.)"

    fig = plt.figure(figsize=(10, 9))
    projection = ccrs.LambertConformal(central_longitude=12, central_latitude=54)
    ax = plt.axes(projection=projection, frameon=False)
    ax.set_extent([-12, 35, 35, 72], crs=ccrs.PlateCarree())
    ax.set_title(TITLE, fontsize=18, pad=15, linespacing=1.4)
    ax.spines['geo'].set_visible(False)

    cmap = plt.colormaps["RdYlBu_r"]

    def get_color(val):
        if val is None or np.isnan(val):
            return "#e0e0e0"
        norm = (val - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1)
        return cmap(norm)

    print(f"\n=== SHAPEFILE MAPPING {year} ({suffix.strip('_')}) ===")
    print(f"   Skala: {vmin} bis {vmax}")

    colored_countries = []
    for rec in records:
        iso = get_country_iso(rec)
        if not iso or iso not in MODEL_COUNTRIES:
            continue

        geom = rec.geometry
        val = delta_dict_iso.get(iso, None)
        color = get_color(val)

        if val is not None and not np.isnan(val):
            colored_countries.append((iso, val))

        ax.add_geometries([geom], ccrs.PlateCarree(),
                          facecolor=color, edgecolor="white",
                          linewidth=0.8, alpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02, fraction=0.046)
    cb.set_label("Preisdifferenz (EUR/MWh)", fontsize=16, labelpad=10)

    ticks = np.arange(vmin, vmax + 0.1, tick_step)
    cb.set_ticks(ticks)
    tick_labels = [f"{int(t):+}" if t != 0 else "0" for t in ticks]
    cb.set_ticklabels(tick_labels, fontsize=14)
    cb.outline.set_visible(False)

    plt.tight_layout(pad=0.5)

    save_name = f"A_{SAVE_STEM}_{year}{suffix}.png"
    save_path = os.path.join(cfg.PLOT_OUTPUT_PATH, save_name)
    os.makedirs(cfg.PLOT_OUTPUT_PATH, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"✅ {len(colored_countries)} Länder gespeichert: {save_path}")


# =====================================================================
# MAIN — alles was vorher Top-Level war, ist jetzt hier drin
# =====================================================================
def main():
    cfg = PlottingConfig()
    networks = cfg.get_networks()

    # ← FIX: kein Top-Level raise mehr, sauberes return statt crash
    if not isinstance(networks, dict) or "average" not in networks or "dunkelflaute" not in networks:
        print("⚠️ plot_delta_prices_map übersprungen: benötigt mode='both' "
              "mit Keys 'average' und 'dunkelflaute'.")
        return

    base_paths = networks["average"]
    df_paths = networks["dunkelflaute"]

    base_years = {extract_year(p): p for p in base_paths}
    df_years = {extract_year(p): p for p in df_paths}

    common_years = sorted(set(base_years.keys()) & set(df_years.keys()))
    print("\nGemeinsame Planungsjahre:", common_years)

    # Shapefile einmalig laden
    print("\nLade Natural Earth 10m (bessere ISO-Codes)...")
    shp_path = shpreader.natural_earth(resolution='10m',
                                       category='cultural',
                                       name='admin_0_countries')
    records = list(shpreader.Reader(shp_path).records())

    # Hauptschleife
    for year in common_years:
        print(f"\n{'=' * 60}")
        print(f"=== Bearbeite Jahr {year} ===")

        n_avg = pypsa.Network(base_years[year])
        n_df = pypsa.Network(df_years[year])
        n_avg.snapshots = pd.to_datetime(n_avg.snapshots)
        n_df.snapshots = pd.to_datetime(n_df.snapshots)

        print("Berechne Preise...")

        # 1. LASTGEWICHTET (VWAP)
        avg_vwap = get_load_weighted_prices(n_avg, START, END)
        df_vwap = get_load_weighted_prices(n_df, START, END)
        delta_vwap = df_vwap - avg_vwap

        # 2. ZEITGEWICHTET (SIMPLE AVERAGE)
        avg_simple = get_simple_average_prices(n_avg, START, END)
        df_simple = get_simple_average_prices(n_df, START, END)
        delta_simple = df_simple - avg_simple

        print(f"\nWeighted Price Delta {year} (Auszug):")
        for c, v in delta_vwap.items():
            iso = normalize_country_code(c)
            if iso in ["DE", "FR", "ES"]:
                print(f"{iso:2s}: {v:6.1f}")

        plot_delta_map(
            delta_vwap, year, "_weighted_total", cfg, records,
            title_override="Differenz lastgewichtete Preise (VWAP)",
            vmin=-200, vmax=200, tick_step=50
        )
        plot_delta_map(
            delta_simple, year, "_simple_average", cfg, records,
            title_override="Differenz mittlere Marktwerte (Time-Weighted)",
            vmin=-50, vmax=50, tick_step=10
        )

    print("\nALLE PLOTS FERTIG!")


if __name__ == "__main__":
    main()
