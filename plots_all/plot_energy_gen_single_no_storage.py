#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot_annual_generation (ohne Speicher - Final)
========================================================
- Fasst Solar (Utility + Rooftop) zusammen.
- Übersetzt Carriers in der Legende (Wind Onshore/Offshore beibehalten).
- OHNE Speicher (PHS, Batterie) in der Erzeugungsstatistik.
- MIT Laufwasser und Speicherwasserkraft.
"""

import os
import re
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from config_final import PlottingConfig

# =====================================================================
# --- Konfiguration: Carriers (Nur Stromerzeuger) ---
# =====================================================================
ELECTRICITY_CARRIERS = [
    # === Erneuerbare ===
    "onwind", "offwind-ac", "offwind-dc", "solar", "solar rooftop", "solar-hsat",
    "ror", "hydro",

    # === Konventionelle ===
    "coal", "lignite", "oil", "nuclear", "OCGT", "CCGT",

    # === Wasserstoff (Rückverstromung) ===
    "H2 OCGT", "H2 turbine", "H2 Fuel Cell",

    # === Biomasse / Abfall ===
    "urban central solid biomass CHP", "urban central solid biomass CHP CC",
    "biogas", "waste"
]

# --- ÜBERSETZUNGSTABELLE ---
CARRIER_TRANSLATION = {
    "solar": "Photovoltaik",
    "onwind": "Wind Onshore",
    "offwind-ac": "Wind Offshore (AC)",
    "offwind-dc": "Wind Offshore (DC)",
    "ror": "Laufwasser",
    "hydro": "Wasserkraft",  # <--- HIER: Übersetzung hinzugefügt
    "nuclear": "Kernkraft",
    "lignite": "Braunkohle",
    "coal": "Steinkohle",
    "oil": "Öl",
    "CCGT": "Erdgas (GuD)",
    "OCGT": "Erdgas (Gasturbine)",
    "urban central solid biomass CHP": "Biomasse (KWK)",
    "urban central solid biomass CHP CC": "Biomasse mit CCS",
    "biogas": "Biogas",
    "waste": "Müllverbrennung",
    "H2 turbine": "H2-Turbine",
    "H2 OCGT": "H2-Gasturbine",
    "H2 Fuel Cell": "Brennstoffzelle"
}


# =====================================================================
# --- Hilfsfunktionen ---
# =====================================================================
def scenario_label_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.dirname(path))).replace("_", " ")


def get_annual_generation(n: pypsa.Network, country_code: str) -> pd.Series:
    """
    Berechnet die Jahresenergie (TWh) pro Technologie.
    Fasst Solar zusammen und übersetzt Namen.
    WICHTIG: Speicher werden NICHT gezählt!
    """
    parts = []
    weight = n.snapshot_weightings.generators

    # --- 1. Generators ---
    gens = n.generators
    if country_code != "ALL":
        gens = gens[gens.bus.str.startswith(country_code)]
    else:
        gens = gens[~gens.bus.str.contains("EU", case=False, na=False)]

    # Filter: Nur Strom-Carrier, keine Wärme
    gens = gens[gens.carrier.isin(ELECTRICITY_CARRIERS)]
    gens = gens[~gens.carrier.str.contains("boiler|heater|heat", case=False, na=False)]

    if not gens.empty:
        p_gen = (n.generators_t.p[gens.index].multiply(weight, axis=0).sum(axis=0))
        parts.append(p_gen.groupby(gens.carrier).sum())

    # --- 2. Links (Nur Erzeuger, KEINE Speicher) ---
    links = n.links
    if country_code != "ALL":
        links = links[(links.bus0.str.startswith(country_code)) | (links.bus1.str.startswith(country_code))]

    links = links[links.carrier.isin(ELECTRICITY_CARRIERS)]
    links = links[~links.carrier.str.contains("heat|boiler", case=False)]
    links = links[links.carrier != "nuclear"]

    if not links.empty:
        p_links = n.links_t.p1[links.index].multiply(weight, axis=0).sum(axis=0).abs()
        parts.append(p_links.groupby(links.carrier).sum())

    # --- 3. Storage Units (NUR Wasserkraft, KEINE Batterien/PHS) ---
    storage = n.storage_units

    # EU-Busse immer ausschließen
    storage = storage[~storage.bus.str.contains("EU", case=False, na=False)]

    if country_code != "ALL":
        storage = storage[storage.bus.str.startswith(country_code)]

    # NUR Wasserkraft zählen, keine Batterien oder PHS
    storage = storage[storage.carrier.isin(["hydro", "ror"])]

    if not storage.empty:
        p_storage = n.storage_units_t.p[storage.index].multiply(weight, axis=0).sum(axis=0)
        # Nur positive Werte (Erzeugung/Entladung), keine Ladung
        p_storage = p_storage[p_storage > 0]
        parts.append(p_storage.groupby(storage.carrier).sum())

    if not parts:
        return pd.Series(dtype=float)

    # Zusammenfügen (Rohdaten)
    raw_data = pd.concat(parts).groupby(level=0).sum() / 1e6  # TWh

    # --- ZUSAMMENFASSEN & ÜBERSETZEN ---
    clean_data = {}

    # Solar zusammenfassen
    solar_sum = 0.0
    for c in ["solar", "solar rooftop", "solar-hsat"]:
        if c in raw_data:
            solar_sum += raw_data[c]
            raw_data = raw_data.drop(c)  # Aus Rohdaten entfernen

    if solar_sum > 0:
        clean_data["Photovoltaik"] = solar_sum

    # Rest übersetzen
    for c, val in raw_data.items():
        # Nutze Übersetzungstabelle oder Originalnamen
        new_name = CARRIER_TRANSLATION.get(c, c)
        if new_name not in clean_data: clean_data[new_name] = 0.0
        clean_data[new_name] += val

    return pd.Series(clean_data)[pd.Series(clean_data) > 0.01]


def print_console_statistics(df: pd.DataFrame, country: str):
    target_years = [2025, 2050]
    print(f"\n{'#' * 60}")
    print(f"📋 STROMERZEUGUNG (TWh): {country}")
    print(f"{'#' * 60}")

    for year in target_years:
        if year not in df.columns: continue
        print(f"\n>>> JAHR {year} (Werte in TWh) <<<")
        print(f"{'-' * 50}")

        col_data = df[year]
        col_data = col_data[col_data > 0].sort_values(ascending=False)

        if col_data.empty:
            print("  (Keine Erzeugung > 0.01 TWh gefunden)")
        else:
            total = col_data.sum()
            for carrier, val in col_data.items():
                share = (val / total) * 100
                print(f"{carrier:40s}: {val:8.2f} TWh ({share:4.1f}%)")

            print(f"{'-' * 50}")
            print(f"{'GESAMT ERZEUGUNG':40s}: {total:8.2f} TWh")
            print(f"{'-' * 50}")
    print("\n")


# =====================================================================
# --- Plotfunktion ---
# =====================================================================
def plot_annual_generation(df: pd.DataFrame, years, country, config: PlottingConfig, scenario: str, ref_path: str):
    if df.empty:
        print(f"⚠️ Keine Daten für {country}")
        return

    # Sortier-Reihenfolge (Deutsch, unten -> oben)
    order = [
        "Kernkraft",
        "Braunkohle", "Steinkohle", "Öl",
        "Erdgas (GuD)", "Erdgas (Gasturbine)",
        "Biomasse (KWK)", "Biomasse mit CCS", "Müllverbrennung", "Biogas",
        "H2-Turbine", "H2-Gasturbine",
        "Laufwasser", "Wasserkraft",
        "Wind Onshore", "Wind Offshore (AC)", "Wind Offshore (DC)",
        "Photovoltaik"
    ]

    df = df[years]
    df = df.reindex(order, axis=0).dropna(how="all")

    # Farben zuweisen (Mapping von deutschen Namen auf Config-Farben)
    color_map_reverse = {v: k for k, v in CARRIER_TRANSLATION.items()}
    color_map_reverse["Photovoltaik"] = "solar"

    colors = []
    for c_de in df.index:
        c_en = color_map_reverse.get(c_de, c_de)

        if c_en in config.CARRIER_COLORS:
            colors.append(config.CARRIER_COLORS[c_en])
        elif "Biomasse" in c_de or "Biogas" in c_de:
            colors.append("#2D5016")
        elif "Öl" in c_de:
            colors.append("#8B4513")
        else:
            colors.append(config.DEFAULT_COLOR)

    # === Plot ===
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(years))
    bar_width = 0.8
    bottom = pd.Series(0.0, index=x_pos)

    for tech, color in zip(df.index, colors):
        vals = df.loc[tech, years].fillna(0).values
        ax.bar(x_pos, vals, bottom=bottom, color=color, width=bar_width,
               edgecolor="white", linewidth=0.5, label=tech)

        # Beschriftung (> 2%)
        threshold = df.sum().max() * 0.02
        for x, h, b in zip(x_pos, vals, bottom):
            if h > threshold:
                r, g, b_col = mcolors.to_rgb(color)
                brightness = (r * 299 + g * 587 + b_col * 114) / 1000
                text_col = "black" if brightness > 0.6 else "white"

                ax.text(x, b + h / 2, f"{int(round(h))}", ha="center", va="center",
                        color=text_col, fontsize=8, fontweight="bold")
        bottom += vals

    ax.set_xticks(x_pos)
    ax.set_xticklabels(years)
    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)

    ax.set_ylabel("Elektrizität [TWh]", fontsize=config.FONT_SIZES.get("label", 11))
    ax.set_xlabel("Jahr", fontsize=config.FONT_SIZES.get("label", 11))

    title_country = "Gesamtnetz" if country == "ALL" else country
    ax.set_title(f"Stromerzeugung in {title_country}",
                 fontsize=config.FONT_SIZES.get("title", 14),
                 fontweight="normal")

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Legende
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              title="Technologien", loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=config.FONT_SIZES.get("legend", 9))

    fig.tight_layout()

    save_dir = os.path.join(config.BASE_SAVE_PATH,
                            os.path.basename(os.path.dirname(os.path.dirname(ref_path))),
                            "generated_energy")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"energy_generated_{country}.png")

    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Plot gespeichert: {save_path}")


# =====================================================================
# --- Hauptablauf ---
# =====================================================================
def main():
    config = PlottingConfig()
    networks = config.get_networks()
    scenario = config.SCENARIO_SELECTION.replace("_", " ")

    network_paths = networks[config.SCENARIO_SELECTION] if isinstance(networks, dict) else networks
    ref_path = network_paths[0]

    all_years = []
    all_data = {c: {} for c in config.get_countries()}

    for path in network_paths:
        if not os.path.isfile(path): continue
        m = re.search(r"_(\d{4})\.nc$", path)
        year = int(m.group(1)) if m else 2050  # ARO: kein Jahr im Namen
        all_years.append(year)

        print(f"📂 Lade Netzwerk {year}...")
        n = pypsa.Network(path)

        for country in config.get_countries():
            all_data[country][year] = get_annual_generation(n, country)

    years = sorted(set(all_years))

    for country, year_dict in all_data.items():
        if not year_dict: continue

        df = pd.DataFrame(year_dict).fillna(0)

        if country in ["DE", "ALL"]:
            print_console_statistics(df, country)

        plot_annual_generation(df, years, country, config, scenario, ref_path)


if __name__ == "__main__":
    main()