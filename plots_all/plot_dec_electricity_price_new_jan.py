#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot: Strompreise (Marginal Costs) – MIT ANALYSE & JANUAR-ZUSATZPLOTS
=====================================================================
Erstellt:
- Volljahresplot
- Volljahresplot (Detailansicht)
- Januarplot
- Januarplot (Detailansicht)

Für alle Länder, die in config_final.COUNTRIES_TO_PLOT gesetzt sind.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pypsa

from config_final import PlottingConfig

# --- Konstante Parameter ---
COMPARE_YEARS = [2025, 2035, 2050]
ROLLING_WINDOW = 24

# Dunkelflaute (nur für Analyse)
DF_START = "01-07"
DF_END = "01-28"

YEAR_COLORS = {
    2025: "#ff7f0e",
    2030: "#d62728",
    2035: "#5F9EA0",
    2040: "#1f77b4",
    2045: "#9467bd",
    2050: "#8c564b"
}


# -------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------

def get_price_series(n, country):
    """Holt durchschnittliche Marktpreise pro Land."""
    if country == "ALL":
        buses = n.buses[n.buses.carrier == "AC"].index
    else:
        buses = n.buses[(n.buses.index.str.startswith(country)) &
                        (n.buses.carrier == "AC")].index

    if len(buses) == 0:
        return pd.Series(dtype=float)
    return n.buses_t.marginal_price[buses].mean(axis=1)


def analyze_prices(series, year, country):
    """Konsolenanalyse für Dunkelflaute."""
    mask = (series.index.strftime('%m-%d') >= DF_START) & \
           (series.index.strftime('%m-%d') <= DF_END)

    df_series = series[mask]
    rest = series[~mask]

    if df_series.empty:
        return

    print(f"\n📊 PREISANALYSE {country} {year}")
    print(f"  Ø DF: {df_series.mean():.2f} €/MWh")
    print(f"  Ø Rest: {rest.mean():.2f} €/MWh")
    print(f"  Max DF: {df_series.max():.2f} €/MWh ({df_series.idxmax()})")


# -------------------------------------------------------
# Plotfunktionen
# -------------------------------------------------------

def plot_price_series(ax, series_dict, config, zoomed, title):
    """Gemeinsame Plotlogik für Volljahr und Januar."""
    fs = config.FONT_SIZES

    for year, series in sorted(series_dict.items()):
        if year not in COMPARE_YEARS:
            continue

        smoothed = series.rolling(window=ROLLING_WINDOW,
                                  center=True).mean()

        dummy_index = pd.date_range(
            start="2020-01-01 00:00",
            periods=len(smoothed),
            freq="H"
        )

        # Längen-Anpassung
        if len(smoothed) != len(dummy_index):
            L = min(len(smoothed), len(dummy_index))
            smoothed = smoothed.iloc[:L]
            dummy_index = dummy_index[:L]

        ax.plot(dummy_index,
                smoothed.values,
                color=YEAR_COLORS.get(year, "black"),
                linewidth=2.2,
                label=f"{year}")

    # Y-Limits setzen
    if zoomed:
        ax.set_ylim(0, 250)

    # Beschriftungen
    ax.set_title(title, fontsize=fs["title"], pad=10)
    ax.set_ylabel("Preis [€/MWh]", fontsize=fs["label"])
    ax.tick_params(axis='both', labelsize=fs["tick"])

    # X-Achse
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax.grid(True, linestyle="--", alpha=0.5)

    legend = ax.legend(
        title="Planungsjahr",
        fontsize=fs["legend"],
        title_fontsize=fs["legend"] + 2,
        loc="upper right"
    )
    legend.get_frame().set_alpha(0.95)


def plot_january(series_dict, country, config, network_name, year, zoomed):
    """Erzeugt Januarplots (01–31.01.)."""

    fs = config.FONT_SIZES

    for y, series in series_dict.items():
        if y != year:
            continue

        # Filter auf 01–31 Januar
        mask = (series.index.month == 1)
        series_jan = series[mask]

        if series_jan.empty:
            return

        # Dummy-Achse für sauberes Overlay
        dummy_idx = pd.date_range("2020-01-01", periods=len(series_jan), freq="H")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot
        ax.plot(dummy_idx,
                series_jan.rolling(window=ROLLING_WINDOW,
                                   center=True).mean(),
                color=YEAR_COLORS.get(year, "black"),
                linewidth=2.2,
                label=f"{year}")

        # Y-Achsenlimit
        if zoomed:
            ax.set_ylim(0, 250)

        # Titel
        if zoomed:
            title = f"Strompreisentwicklung in {country} – {year} (Januar – Detailansicht)"
        else:
            title = f"Strompreisentwicklung in {country} – {year} (Januar)"

        ax.set_title(title, fontsize=fs["title"])
        ax.set_ylabel("Preis [€/MWh]", fontsize=fs["label"])
        ax.tick_params(axis='both', labelsize=fs["tick"])

        ax.grid(True, linestyle="--", alpha=0.5)

        ax.legend(
            title="Planungsjahr",
            fontsize=fs["legend"],
            title_fontsize=fs["legend"] + 2
        )

        # SPEICHERN
        save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "prices")
        os.makedirs(save_dir, exist_ok=True)

        suf = "_jan_detail" if zoomed else "_jan"
        fname = f"prices_{country}_{year}{suf}.png"

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname), dpi=300)
        plt.close(fig)

        print(f"  📁 Januar-Plot gespeichert: {fname}")


def plot_full_year(series_dict, country, config, network_name):
    """Erzeugt Volljahresplot + Detailansicht."""

    fs = config.FONT_SIZES

    # ---------- NORMAL ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    title = f"Strompreisentwicklung in {country} – Jahr"
    plot_price_series(ax, series_dict, config, zoomed=False, title=title)

    save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "prices")
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"prices_{country}_full.png"), dpi=300)
    plt.close(fig)

    # ---------- DETAIL ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    title = f"Strompreisentwicklung in {country} – Jahr (Detailansicht)"
    plot_price_series(ax, series_dict, config, zoomed=True, title=title)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"prices_{country}_full_detail.png"), dpi=300)
    plt.close(fig)


def plot_january_comparison(series_dict, country, config, network_name, zoomed=False):
    """
    Vergleichsplot Januar für alle Jahre in COMPARE_YEARS.
    zoomed=False  -> normal
    zoomed=True   -> Detail (0–250 €/MWh)
    """

    fs = config.FONT_SIZES

    fig, ax = plt.subplots(figsize=(12, 6))

    for year, series in sorted(series_dict.items()):
        if year not in COMPARE_YEARS:
            continue

        # Januar filtern
        mask = (series.index.month == 1)
        s_jan = series[mask]

        if s_jan.empty:
            continue

        smoothed = s_jan.rolling(window=ROLLING_WINDOW, center=True).mean()
        dummy_idx = pd.date_range("2020-01-01", periods=len(smoothed), freq="H")

        ax.plot(dummy_idx,
                smoothed.values,
                linewidth=2.2,
                color=YEAR_COLORS.get(year, "black"),
                label=f"{year}")

    # Titel
    if zoomed:
        title = f"Strompreisentwicklung in {country} – Januar (Detailansicht, Vergleich)"
        ax.set_ylim(0, 250)
    else:
        title = f"Strompreisentwicklung in {country} – Januar (Vergleich)"

    ax.set_title(title, fontsize=fs["title"])
    ax.set_ylabel("Preis [€/MWh]", fontsize=fs["label"])
    ax.tick_params(axis='both', labelsize=fs["tick"])

    ax.grid(True, linestyle="--", alpha=0.5)

    ax.legend(
        title="Planungsjahr",
        fontsize=fs["legend"],
        title_fontsize=fs["legend"] + 2,
        loc="upper right"
    )

    # Speichern
    save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "prices")
    os.makedirs(save_dir, exist_ok=True)

    suffix = "_jan_comparison_detail.png" if zoomed else "_jan_comparison.png"
    fname = f"prices_{country}{suffix}"

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=300)
    plt.close(fig)

    print(f"  📁 Januar-Vergleich gespeichert: {fname}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    config = PlottingConfig()
    networks = config.get_networks()
    countries = config.get_countries()

    # Referenzpfad bestimmen
    ref_path = networks[0]
    network_name = os.path.basename(os.path.dirname(os.path.dirname(ref_path)))

    all_data = {c: {} for c in countries}

    # Netzwerke laden
    for path in networks:
        if not os.path.isfile(path):
            continue
        m = re.search(r"_(\d{4})\.nc$", path)
        if not m:
            continue

        year = int(m.group(1))
        if year not in COMPARE_YEARS:
            continue

        print(f"\nLade {path} ...")
        n = pypsa.Network(path)

        # Zeitindex sicherstellen
        try:
            n.snapshots = pd.to_datetime(n.snapshots)
        except Exception:
            pass

        # Preisdaten pro Land einsammeln
        for country in countries:
            series = get_price_series(n, country)
            if not series.empty:
                all_data[country][year] = series
                analyze_prices(series, year, country)

    # Plots erzeugen
    for country in countries:
        if not all_data[country]:
            continue

        print(f"\n📈 Erzeuge Plots für {country} ...")

        # Volljahr
        plot_full_year(all_data[country], country, config, network_name)


        # Einzelplots Januar pro Jahr
        for year in all_data[country].keys():
            if year in COMPARE_YEARS:
                plot_january(all_data[country], country, config, network_name, year, zoomed=False)
                plot_january(all_data[country], country, config, network_name, year, zoomed=True)

        # NEU: Januar-Vergleichsplot
        plot_january_comparison(all_data[country], country, config, network_name, zoomed=False)
        plot_january_comparison(all_data[country], country, config, network_name, zoomed=True)


if __name__ == "__main__":
    main()
