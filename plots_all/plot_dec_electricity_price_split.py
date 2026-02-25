#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot: Strompreise (Schattenpreise) - DUNKELFLAUTEN-SPEZIAL

Features:
1. Statistische Analyse: Vergleich Preise in DF vs. Rest des Jahres.
2. Split-Plot:
   - Links: Jahresverlauf linear (0-300 €) -> Zeigt günstige Sommer 2050.
   - Rechts: Januar logarithmisch -> Zeigt Investitions-Spikes (2035) vs. Betriebs-Plateau (2050).
"""

import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pypsa
from config_final import PlottingConfig

# --- KONFIGURATION ---

TARGET_COUNTRIES = ["DE", "FR", "GB", "NO", "SE", "DK", "IT", "ES"]
COMPARE_YEARS = [2025, 2035, 2050]

# Dunkelflaute Zeitraum (Monat-Tag) für Analyse & Plot
DF_START_MD = "01-07"
DF_END_MD = "01-28"

# Plot-Einstellungen
ROLLING_WINDOW_YEAR = 24  # Glättung für Jahresübersicht
ROLLING_WINDOW_ZOOM = 3  # Geringe Glättung für Zoom (Spikes sichtbar lassen)

# Farben (Deine Palette)
YEAR_COLORS = {
    2025: "#ff7f0e",  # Orange
    2030: "#d62728",  # Rot
    2035: "#5F9EA0",  # Teal (Türkis)
    2040: "#1f77b4",  # Blau
    2045: "#9467bd",  # Lila
    2050: "#8c564b"  # Braun
}


def get_price_series(n, country):
    """Holt durchschnittliche Marktpreise (gewichtet nach Buses)."""
    if country == "ALL":
        buses = n.buses[n.buses.carrier == "AC"].index
    else:
        buses = n.buses[(n.buses.index.str.startswith(country)) & (n.buses.carrier == "AC")].index

    if len(buses) == 0: return pd.Series(dtype=float)
    return n.buses_t.marginal_price[buses].mean(axis=1)


def analyze_df_impact(series, year, country):
    """
    Berechnet Statistiken: Dunkelflaute vs. Rest des Jahres.
    """
    if series.empty: return

    # Maske erstellen (Datum-String Vergleich für Unabhängigkeit vom Jahr)
    mask_df = (series.index.strftime('%m-%d') >= DF_START_MD) & \
              (series.index.strftime('%m-%d') <= DF_END_MD)

    prices_df = series[mask_df]
    prices_rest = series[~mask_df]

    avg_df = prices_df.mean()
    avg_rest = prices_rest.mean()
    max_df = prices_df.max()

    # Faktor berechnen (Vermeidung von Division durch Null)
    factor = avg_df / avg_rest if avg_rest > 1.0 else (avg_df / 1.0)

    print(f"\n{'=' * 50}")
    print(f"🌪️ KNAPPHEITS-ANALYSE {country} {year}")
    print(f"   Zeitraum: {DF_START_MD} bis {DF_END_MD}")
    print(f"{'=' * 50}")
    print(f"  Ø Preis in Dunkelflaute:  {avg_df:.2f} €/MWh")
    print(f"  Ø Preis Rest des Jahres:  {avg_rest:.2f} €/MWh")
    print(f"  -> Faktor (Knappheit):    {factor:.1f}x höher")
    print(f"  Maximalpreis in DF:       {max_df:.2f} €/MWh")
    print(f"{'=' * 50}\n")


def plot_split_view(data_dict, country, config, network_name):
    """
    Erstellt den Kombi-Plot:
    Links: Ganzes Jahr (Linear, Cutoff 300€)
    Rechts: Januar/Februar (Logarithmisch, bis 10.000€)
    """
    if not data_dict: return

    # Layout: 2 Subplots nebeneinander
    # width_ratios=[1.3, 1] gibt dem Jahresplot etwas mehr Platz
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.3, 1]})

    # Dummy Jahr 2020 für einheitliche X-Achse (Schaltjahr, deckt alles ab)
    plot_start_year = "2020-01-01"
    plot_end_year = "2020-12-31"

    # Fokus-Bereich für rechten Plot
    focus_start = "2020-01-01"
    focus_end = "2020-02-15"

    sorted_years = sorted(data_dict.keys())

    # ==========================
    # PLOT 1: Linke Seite (Jahr, Linear)
    # ==========================
    for year in sorted_years:
        if year not in COMPARE_YEARS: continue
        series = data_dict[year]

        # Mapping auf 2020
        dummy_idx = pd.date_range(start=plot_start_year + " 00:00", periods=len(series), freq="h")
        # Falls Längen nicht passen (Schaltjahr Problematik), abschneiden
        min_len = min(len(series), len(dummy_idx))
        series_dummy = pd.Series(series.values[:min_len], index=dummy_idx[:min_len])

        # Glättung
        smooth = series_dummy.rolling(window=ROLLING_WINDOW_YEAR, center=True).mean()

        ax1.plot(smooth.index, smooth.values,
                 color=YEAR_COLORS.get(year, "black"),
                 linewidth=1.5,
                 label=f"{year}")

    # Styling Links
    ax1.set_title(f"Jahresverlauf {country} (Detailansicht)", fontsize=23)
    ax1.set_ylabel("Schattenpreis [€/MWh]", fontsize=19)
    ax1.set_ylim(0, 250)  # Cutoff wie gewünscht

    # X-Achse: Jeden 2. Monat
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(title="Planungsjahr", loc="upper right", fontsize=19, title_fontsize=18)

    # ==========================
    # PLOT 2: Rechte Seite (Januar, Logarithmisch)
    # ==========================
    for year in sorted_years:
        if year not in COMPARE_YEARS: continue
        series = data_dict[year]

        # Mapping auf 2020
        dummy_idx = pd.date_range(start=plot_start_year + " 00:00", periods=len(series), freq="h")
        min_len = min(len(series), len(dummy_idx))
        series_dummy = pd.Series(series.values[:min_len], index=dummy_idx[:min_len])

        # Slice auf Fokus-Zeitraum
        subset = series_dummy[focus_start:focus_end]

        # Log-Vorbereitung: Werte <= 1 auf 1 setzen, damit Log-Scale funktioniert
        # Wir nehmen 1€ als Boden, alles darunter ist visuell irrelevant im Log-Plot
        subset_log = subset.copy()
        subset_log[subset_log < 1] = 1

        # Geringe Glättung
        smooth_focus = subset_log.rolling(window=ROLLING_WINDOW_ZOOM, center=True).mean()

        ax2.plot(smooth_focus.index, smooth_focus.values,
                 color=YEAR_COLORS.get(year, "black"),
                 linewidth=1.5,
                 label=f"{year}")

    # Dunkelflaute markieren (Grau)
    df_start_plot = pd.to_datetime(f"2020-{DF_START_MD} 00:00")
    df_end_plot = pd.to_datetime(f"2020-{DF_END_MD} 23:00")
    ax2.axvspan(df_start_plot, df_end_plot, color='gray', alpha=0.2, label="Dunkelflaute")

    # Styling Rechts
    ax2.set_title(f"Fokus: Dunkelflaute (Log-Skala)", fontsize=23)

    # Logarithmische Skala & Limits
    ax2.set_yscale('log')
    ax2.set_ylim(10, 10000)  # Von 10€ bis 10.000€

    # Manuelle Y-Ticks für bessere Lesbarkeit
    ax2.set_yticks([10, 100, 1000, 10000])
    ax2.set_yticklabels(["10", "100", "1.000", "10.000"])

    # X-Achse
    ax2.set_xlim(pd.to_datetime(focus_start), pd.to_datetime(focus_end))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))

    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.grid(True, linestyle="--", alpha=0.5, which="both")  # both für Log-Grid-Linien

    plt.tight_layout()

    # Speichern
    save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "prices")
    os.makedirs(save_dir, exist_ok=True)
    fname = f"prices_split_view_{country}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=300)
    print(f"  ✅ Split-Plot gespeichert: {fname}")
    plt.close(fig)


def main():
    config = PlottingConfig()
    networks = config.get_networks()

    # Szenario-Auswahl: Priorisiere 'dunkelflaute'
    if isinstance(networks, dict):
        if 'dunkelflaute' in networks:
            paths = networks['dunkelflaute']
            print("ℹ️ Nutze 'dunkelflaute' Szenarien.")
        else:
            print("⚠️ WARNUNG: Kein 'dunkelflaute' Key gefunden. Nutze erstes verfügbares.")
            paths = list(networks.values())[0]
    else:
        paths = networks

    # Name extrahieren
    if not paths: return
    ref_path = paths[0]
    network_name = os.path.basename(os.path.dirname(os.path.dirname(ref_path)))

    all_data = {c: {} for c in TARGET_COUNTRIES}

    for path in paths:
        if not os.path.isfile(path): continue
        m = re.search(r"_(\d{4})\.nc$", path)
        if not m: continue
        year = int(m.group(1))

        if year not in COMPARE_YEARS: continue

        print(f"Lade {year}...")
        n = pypsa.Network(path)
        # Zeitindex sicherstellen
        try:
            n.snapshots = pd.to_datetime(n.snapshots)
        except:
            pass

        for country in TARGET_COUNTRIES:
            prices = get_price_series(n, country)
            if not prices.empty:
                all_data[country][year] = prices

                # --- ANALYSE ---
                analyze_df_impact(prices, year, country)

    print("\nErstelle Plots...")
    for country in TARGET_COUNTRIES:
        if not all_data[country]: continue

        # Der neue Split-Plot
        plot_split_view(all_data[country], country, config, network_name)


if __name__ == "__main__":
    main()