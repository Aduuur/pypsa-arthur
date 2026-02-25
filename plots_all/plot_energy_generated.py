#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot: Jährlich erzeugte Energie pro Technologie (TWh)
=====================================================

Erstellt gestapelte Balkenplots der jährlichen Stromerzeugung (TWh)
nach Technologie für alle Planungshorizonte in einem Diagramm,
getrennt nach Land oder Gesamtnetz ("ALL").
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pypsa

from config_final import PlottingConfig
from ariadne_generated_energy import get_generation_twh


# =====================================================================
# --- Hilfsfunktionen ---
# =====================================================================

def collect_generation_data_all_years(network_paths, country):
    """Aggregiert die Erzeugung pro Technologie über alle Jahre hinweg."""
    all_data = {}
    for path in network_paths:
        if not os.path.isfile(path):
            continue
        m = re.search(r"_(\d{4})\.nc$", path)
        if not m:
            continue
        year = int(m.group(1))
        n = pypsa.Network(path)
        print(f"🔹 Berechne Erzeugung für {country} ({year})")

        if country == "ALL":
            s = get_generation_twh(n, "")
        else:
            s = get_generation_twh(n, country)
        all_data[year] = s
    df = pd.DataFrame(all_data).fillna(0)
    return df


def get_contrasting_color(hex_color):
    """Berechnet kontrastierende Textfarbe (schwarz/weiß) zu Balkenfarbe."""
    r, g, b = mcolors.to_rgb(hex_color)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "black" if brightness > 0.6 else "white"


def plot_generation_all_years(df, country, config: PlottingConfig, network_paths):
    """Erstellt einen Plot mit allen Planungshorizonten nebeneinander."""
    df_plot = df.copy()

    # Nettoimporte berechnen
    if "Import" in df_plot.index and "Export" in df_plot.index:
        df_plot.loc["Nettoimporte"] = (df_plot.loc["Import"] - df_plot.loc["Export"]).clip(lower=0)

    drop_labels = ["Import", "Export", "Nettohandel", "Verfügbare Nettoenergie"]
    df_plot = df_plot.loc[~df_plot.index.isin(drop_labels)]

    color_list = [config.CARRIER_COLORS.get(k, config.DEFAULT_COLOR) for k in df_plot.index]

    # === Plot ===
    fig, ax = plt.subplots(figsize=(11, 6))
    df_plot.T.plot(kind="bar", stacked=True, color=color_list, width=0.75, ax=ax, edgecolor="none")

    title_country = "Gesamtnetz" if country == "ALL" else country
    ax.set_title(f"Stromerzeugung und Nettoimporte in {title_country}",
                 fontsize=config.FONT_SIZES["title"], fontweight="bold", pad=15)
    ax.set_ylabel("Elektrizität [TWh]", fontsize=config.FONT_SIZES["label"])
    ax.set_xlabel("Jahr", fontsize=config.FONT_SIZES["label"])
    ax.tick_params(axis="x", rotation=0, labelsize=config.FONT_SIZES["tick"])
    ax.tick_params(axis="y", labelsize=config.FONT_SIZES["tick"])
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    # Zahlen in Balken (>10 TWh)
    cumulative = df_plot.cumsum()
    for i, year in enumerate(df_plot.columns):
        for carrier, value in df_plot[year].items():
            if value > 10:
                bottom = cumulative.loc[carrier, year] - value
                color = config.CARRIER_COLORS.get(carrier, config.DEFAULT_COLOR)
                text_color = get_contrasting_color(color)
                ax.text(
                    i, bottom + value / 2,
                    f"{value:.0f}",
                    ha="center", va="center",
                    fontsize=config.FONT_SIZES["bar_label"],
                    color=text_color, fontweight="bold"
                )

    # Legende
    handles, labels = ax.get_legend_handles_labels()
    labels = [lbl.replace("Net Imports", "Nettoimporte") for lbl in labels]
    ax.legend(
        handles, labels, title="Technologie",
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=config.FONT_SIZES["legend"], title_fontsize=config.FONT_SIZES["legend"]
    )

    plt.tight_layout()

    # === Speicherort ===
    first_network_path = network_paths[0]
    network_name = os.path.basename(os.path.dirname(os.path.dirname(first_network_path)))
    save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "generated_energy")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"energy_generated_{country}_all_years.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Plot gespeichert: {save_path}")


# =====================================================================
# --- Hauptablauf ---
# =====================================================================

def main():
    config = PlottingConfig()
    network_paths = config.get_networks()

    for country in config.get_countries():  # enthält ALL + Länder
        df = collect_generation_data_all_years(network_paths, country)
        if df.empty:
            print(f"⚠️ Keine Erzeugungsdaten für {country} gefunden.")
            continue
        plot_generation_all_years(df, country, config, network_paths)


if __name__ == "__main__":
    main()
