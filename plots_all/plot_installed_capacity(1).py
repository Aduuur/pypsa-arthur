#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot_installed_capacities (Final v10 – NUCLEAR GENERATOR ONLY)
===================================================================================
- Kernkraft nur über generatoren modelliert
- Whitelist für kleine Technologien aktiv.
"""

import os
import re
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config_final import PlottingConfig

# =====================================================================
# --- Konfiguration ---
# =====================================================================

ELECTRICITY_CARRIERS = [
    "onwind", "offwind-ac", "offwind-dc",
    "solar", "solar rooftop", "solar-hsat",
    "ror", "hydro",
    "coal", "lignite", "oil", "nuclear",
    "OCGT", "CCGT",
    "H2 turbine", "H2 OCGT", "H2 Fuel Cell",
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
    "battery discharger", "home battery discharger",
    "PHS"
]

STORAGE_DISCHARGE_CARRIERS = [
    "battery discharger", "home battery discharger", "PHS"
]

CARRIER_TRANSLATION = {
    "solar": "Photovoltaik",
    "onwind": "Wind Onshore",
    "offwind-ac": "Wind Offshore (AC)",
    "offwind-dc": "Wind Offshore (DC)",
    "ror": "Laufwasser",
    "hydro": "Wasserkraft",
    "nuclear": "Kernkraft",
    "lignite": "Braunkohle",
    "coal": "Steinkohle",
    "oil": "Öl",
    "CCGT": "Erdgas (GuD)",
    "OCGT": "Erdgas (Gasturbine)",
    "PHS": "Pumpspeicher",
    "battery discharger": "Batteriespeicher",
    "home battery discharger": "Heimspeicher",
    "H2 turbine": "H2-Turbine",
    "H2 OCGT": "H2-Gasturbine",
    "H2 Fuel Cell": "Brennstoffzelle"
}


# =====================================================================
# --- HILFSFUNKTION: Länder-Erkennung ---
# =====================================================================
def get_country_codes(network):
    all_buses = network.buses.index
    prefixes = set(b[:2] for b in all_buses)
    countries = [p for p in prefixes if p != "EU" and p.isalpha() and p.isupper() and len(p) == 2]
    return sorted(list(countries))


# =====================================================================
# --- Speicher (GWh) ---
# =====================================================================
def get_storage_capacity_gwh(n: pypsa.Network, country_code: str, tech_type: str) -> float:
    total = 0.0
    valid_countries = get_country_codes(n)
    regex_countries = "^(" + "|".join(valid_countries) + ")"

    if tech_type == "battery":
        pattern = "battery"
    elif tech_type == "h2":
        pattern = "H2|hydrogen"
    else:
        return 0.0

    if not n.stores.empty:
        mask = n.stores.carrier.str.contains(pattern, case=False, na=False)
        if country_code != "ALL":
            mask &= n.stores.bus.str.startswith(country_code)
        else:
            mask &= n.stores.bus.str.contains(regex_countries)
        total += n.stores.loc[mask, "e_nom_opt"].sum() / 1000.0

    if tech_type == "battery" and not n.storage_units.empty:
        mask = n.storage_units.carrier.str.contains(pattern, case=False, na=False)
        if country_code != "ALL":
            mask &= n.storage_units.bus.str.startswith(country_code)
        else:
            mask &= n.storage_units.bus.str.contains(regex_countries)

        if "e_nom_opt" in n.storage_units.columns:
            total += n.storage_units.loc[mask, "e_nom_opt"].sum() / 1000.0
        else:
            total += (
                             n.storage_units.loc[mask, "p_nom_opt"]
                             * n.storage_units.loc[mask, "max_hours"]
                     ).sum() / 1000.0

    return total


# =====================================================================
# --- Installierte Leistung (GW) ---
# =====================================================================
def get_installed_capacity(n: pypsa.Network, country_code: str) -> pd.Series:
    parts = []
    pcol = "p_nom_opt"

    valid_countries = get_country_codes(n)
    regex_countries = "^(" + "|".join(valid_countries) + ")"

    # -----------------------------------------------------------
    # 1. Generatoren (Hier ist meistens Nuclear drin)
    # -----------------------------------------------------------
    gens = n.generators.copy()  # Copy to avoid warnings
    if country_code != "ALL":
        gens = gens[gens.bus.str.startswith(country_code)]
    else:
        gens = gens[gens.bus.str.contains(regex_countries)]

    gens = gens[gens.carrier.isin(ELECTRICITY_CARRIERS)]

    if not gens.empty:
        # Hier landet die Kernkraft (als Generator).
        # Wir übernehmen den Wert 1:1 (elektrisch).
        parts.append(gens.groupby("carrier")[pcol].sum())

    # -----------------------------------------------------------
    # 2. Links
    # -----------------------------------------------------------
    links = n.links.copy()
    if country_code != "ALL":
        links = links[links.bus0.str.startswith(country_code) | links.bus1.str.startswith(country_code)]
    else:
        links = links[links.bus1.str.contains(regex_countries)]

    links = links[links.carrier.isin(ELECTRICITY_CARRIERS)]

    # --- WICHTIG: NUCLEAR BEI LINKS RAUSWERFEN ---
    # Da wir wissen, dass die Generatoren die echten Daten enthalten,
    # ignorieren wir hier alles, was "nuclear" heißt, um Doppelzählung zu vermeiden
    links = links[links.carrier != "nuclear"]

    if not links.empty:
        CORRECTION_MAP = {
            "OCGT": 0.39, "Erdgas (Gasturbine)": 0.39,
            "CCGT": 0.58, "Erdgas (GuD)": 0.58
        }

        cap_final = links[pcol].copy()

        for tech in links.carrier.unique():
            # Checken ob Tech korrigiert werden muss
            needs_correction = False
            correction_factor = 0.4

            # Direkter Match im Dictionary
            if tech in CORRECTION_MAP:
                needs_correction = True
                correction_factor = CORRECTION_MAP[tech]
            # Oder Substring Match (nur noch für Gas)
            elif any(x in tech for x in ["OCGT", "CCGT"]):
                needs_correction = True
                if "CCGT" in tech:
                    correction_factor = 0.58
                else:
                    correction_factor = 0.39

            if needs_correction:
                mask = links.carrier == tech
                if "efficiency" in links.columns:
                    eff_vals = links.loc[mask, "efficiency"]
                    cap_final.loc[mask] *= eff_vals


        parts.append(cap_final.groupby(links.carrier).sum())

    # 3. StorageUnits
    sus = n.storage_units
    if country_code != "ALL":
        sus = sus[sus.bus.str.startswith(country_code)]
    else:
        sus = sus[sus.bus.str.contains(regex_countries)]

    sus = sus[sus.carrier.isin(STORAGE_DISCHARGE_CARRIERS)]
    if not sus.empty:
        parts.append(sus.groupby("carrier")[pcol].sum())

    # 4. StorageUnits (Wasserkraft) - separat behandeln
    hydro_sus = n.storage_units

    # EU-Busse immer ausschließen
    hydro_sus = hydro_sus[~hydro_sus.bus.str.contains("EU", case=False, na=False)]

    if country_code != "ALL":
        hydro_sus = hydro_sus[hydro_sus.bus.str.startswith(country_code)]

    # NUR Wasserkraft
    hydro_sus = hydro_sus[hydro_sus.carrier.isin(["hydro", "ror"])]

    if not hydro_sus.empty:
        parts.append(hydro_sus.groupby("carrier")[pcol].sum())

    if not parts:
        return pd.Series(dtype=float)

    raw = pd.concat(parts).groupby(level=0).sum() / 1000.0

    clean = {}

    # Solar
    solar_sum = raw.filter(regex="solar").sum()
    raw = raw.drop(raw.index[raw.index.str.contains("solar")], errors="ignore")
    if solar_sum > 0:
        clean["Photovoltaik"] = solar_sum

    # Biomasse
    biomass_sum = raw.filter(regex="biomass").sum()
    raw = raw.drop(raw.index[raw.index.str.contains("biomass")], errors="ignore")
    if biomass_sum > 0:
        clean["Biomasse"] = biomass_sum

    for c, v in raw.items():
        name = CARRIER_TRANSLATION.get(c, c)
        clean[name] = clean.get(name, 0.0) + v

    s = pd.Series(clean)
    return s[s > 0.01]


# =====================================================================
# --- Plot ---
# =====================================================================
def plot_installed_capacities(df, years, country, config, scenario, ref_path):
    if df.empty:
        return

    total_capacity = df.sum(axis=0).max()
    MIN_SHARE = 0.01  # 1 %

    # --- HIER ÄNDERN: Laufwasser und Wasserkraft hinzufügen ---
    ALWAYS_KEEP = [
        "Braunkohle", "Steinkohle", "Biomasse", "Öl", "Kernkraft",
        "Laufwasser", "Wasserkraft",  "Erdgas (GuD)", "Erdgas (Gasturbine)",
        "Pumpspeicher", "Batteriespeicher", "Heimspeicher"
    ]
    # ----------------------------------------------------------

    mask = (df.max(axis=1) >= MIN_SHARE * total_capacity) | df.index.isin(ALWAYS_KEEP)

    df_plot = df[mask].copy()

    mask = (df.max(axis=1) >= MIN_SHARE * total_capacity) | df.index.isin(ALWAYS_KEEP)

    df_plot = df[mask].copy()

    if df_plot.empty:
        print(f"⚠️ Keine signifikanten Technologien für {country}")
        return

    order = [
        "Kernkraft",
        "Photovoltaik",
        "Wind Onshore", "Wind Offshore (AC)", "Wind Offshore (DC)",
        "Laufwasser", "Wasserkraft",
        "Biomasse",
        "Braunkohle", "Steinkohle", "Öl",
        "Erdgas (GuD)", "Erdgas (Gasturbine)",
        "Pumpspeicher", "Batteriespeicher", "Heimspeicher", "H2-Turbine", "H2-Gasturbine"
    ]

    df_plot = df_plot.reindex(order).dropna(how="all")
    years = [y for y in years if y in df_plot.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(years))
    bottom = np.zeros(len(years))

    TEXT_SHARE = 0.015
    text_threshold = TEXT_SHARE * total_capacity

    for tech in df_plot.index:
        if tech == "Biomasse":
            color = config.CARRIER_COLORS.get("biomass", "#6a3d9a")
        else:
            color_key = next(
                (k for k, v in CARRIER_TRANSLATION.items() if v == tech),
                None
            )
            color = config.CARRIER_COLORS.get(color_key, config.DEFAULT_COLOR)

        vals = df_plot.loc[tech, years].values
        ax.bar(
            x, vals, bottom=bottom, color=color, label=tech,
            edgecolor="white", linewidth=0.5
        )

        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= text_threshold and v > 0.5:
                ax.text(
                    x[i], b + v / 2, f"{int(round(v))}",
                    ha="center", va="center", color="white",
                    fontsize=config.FONT_SIZES.get("bar", 9), fontweight="bold"
                )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Kapazität (GW)", fontsize=config.FONT_SIZES.get("label", 11) + 3)
    ax.set_xlabel("Jahr", fontsize=config.FONT_SIZES.get("label", 11) + 3)
    ax.set_title(f"Installierte Leistung in {country}", fontsize=config.FONT_SIZES.get("title", 14) + 3)

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=config.FONT_SIZES.get("ticks", 9) + 3)
    ax.set_axisbelow(True)

    # Legende
    handles, labels = ax.get_legend_handles_labels()
    legend_keep = []

    for h, l in zip(handles, labels):
        is_big = (l in df.index and df.loc[l].max() >= MIN_SHARE * total_capacity)
        is_whitelist = (l in ALWAYS_KEEP and l in df.index and df.loc[l].sum() > 0.1)

        if is_big or is_whitelist:
            legend_keep.append((h, l))

    if legend_keep:
        handles_f, labels_f = zip(*legend_keep)
        ax.legend(
            handles_f[::-1], labels_f[::-1],
            title="Technologien",
            fontsize=config.FONT_SIZES.get("legend", 9) + 3,
            title_fontsize=config.FONT_SIZES.get("legend_title", 10) + 3,
            loc="center left", bbox_to_anchor=(1.02, 0.5)
        )

    fig.tight_layout()
    save_dir = os.path.join(
        config.BASE_SAVE_PATH,
        os.path.basename(os.path.dirname(os.path.dirname(ref_path))),
        "plots_installed_capacities"
    )
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"Plot_installed_capacities_{country}.png"), dpi=300)
    plt.close(fig)


def print_console_summary(df, h2_store, bat_store, country):
    print("\n" + "#" * 70)
    print(f"📊 INSTALLIERTE KAPAZITÄTEN & SPEICHER – {country}")
    print("#" * 70)
    for year in df.columns:
        print(f"\n>>> PLANJAHR {year}")
        print("-" * 60)
        col = df[year][df[year] > 0].sort_values(ascending=False)
        for tech, val in col.items():
            print(f"{tech:35s}: {val:8.2f} GW")
        print("-" * 60)
        print(f"{'SUMME KAPAZITÄT':35s}: {col.sum():8.2f} GW")
        print(f"{'Batteriespeicher':35s}: {bat_store.get(year, 0.0):8.2f} GWh")
    print("#" * 70 + "\n")


# =====================================================================
# --- Main ---
# =====================================================================
def main():
    config = PlottingConfig()
    networks = config.get_networks()

    if isinstance(networks, dict):
        network_paths = networks[config.SCENARIO_SELECTION]
    else:
        network_paths = networks

    if not network_paths:
        print(" Keine Netzwerke gefunden.")
        return

    all_data = {c: {} for c in config.get_countries()}
    h2_store = {c: {} for c in config.get_countries()}
    bat_store = {c: {} for c in config.get_countries()}
    years = []

    for path in network_paths:
        try:
            m = re.search(r"_(\d{4})\.nc$", path)
            if not m: continue
            year = int(m.group(1))
            years.append(year)

            print(f"📂 Lade Netzwerk {year}")
            n = pypsa.Network(path)

            for c in config.get_countries():
                all_data[c][year] = get_installed_capacity(n, c)
                h2_store[c][year] = get_storage_capacity_gwh(n, c, "h2")
                bat_store[c][year] = get_storage_capacity_gwh(n, c, "battery")
        except Exception as e:
            print(f"Fehler beim Laden von {path}: {e}")

    for c, data in all_data.items():
        df = pd.DataFrame(data).fillna(0)
        if c in ["DE", "ALL", "PL"]:  # Added FR for check
            print_console_summary(df, h2_store[c], bat_store[c], c)

        plot_installed_capacities(
            df, sorted(list(set(years))), c, config,
            config.SCENARIO_SELECTION, network_paths[0]
        )


if __name__ == "__main__":
    main()