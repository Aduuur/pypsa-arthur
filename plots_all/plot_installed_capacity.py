#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot_installed_capacities (final version 6)
===========================================

Korrektionen:
- NUR Stromerzeuger (keine Wärmekessel, keine Brennstoffe)
- Biomasse CHP (erzeugt Strom)
- Oil separat für Gen und Links
- Debug-Output für Oil (Gen vs Link)
- Speicher nur als Entladung
"""

import os
import re
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config_final import PlottingConfig

# =====================================================================
# --- Stromerzeuger ONLY ---
# =====================================================================
ELECTRICITY_CARRIERS = [
    # === Erneuerbare (direkt Strom) ===
    "onwind", "offwind-ac", "offwind-dc", "solar", "solar rooftop", "solar-hsat", "ror",

    # === Konventionelle Wärmekraftwerke (Strom) ===
    "coal", "lignite", "oil", "nuclear", "OCGT", "CCGT",

    # === Wasserstoff (Strom) ===
    "H2 OCGT", "H2 turbine", "H2 Fuel Cell",

    # === Biomasse NUR CHP (erzeugt auch Strom!) ===
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC"
]

STORAGE_DISCHARGE_CARRIERS = [
    "battery discharger", "home battery discharger", "PHS"
]


# =====================================================================
# --- Hilfsfunktionen ---
# =====================================================================
def scenario_label_from_path(path: str) -> str:
    """Liest den Szenarionamen aus dem Pfad."""
    return os.path.basename(os.path.dirname(os.path.dirname(path))).replace("_", " ")


def get_installed_capacity(n: pypsa.Network, country_code: str, debug: bool = False) -> pd.Series:
    """Aggregiert installierte Leistung (GW) nach Technologie."""
    parts = []
    pcol = "p_nom_opt"

    # ===== DEBUGGING: Alle Oil-Träger anzeigen =====
    if debug:
        print(f"\n{'=' * 80}")
        print(f"DEBUG OIL für {country_code}")
        print(f"{'=' * 80}")

        all_oil_gens = n.generators[n.generators.carrier.str.contains("oil", case=False, na=False)]
        print(f"\n🔸 ALLE Oil Generatoren (ungefilter):")
        for carrier in sorted(all_oil_gens.carrier.unique()):
            cap = all_oil_gens[all_oil_gens.carrier == carrier]['p_nom_opt'].sum()
            print(f"  {carrier:45s}: {cap:>10.1f} MW")

        all_oil_links = n.links[n.links.carrier.str.contains("oil", case=False, na=False)]
        print(f"\n🔸 ALLE Oil Links (ungefilter):")
        for carrier in sorted(all_oil_links.carrier.unique()):
            cap = all_oil_links[all_oil_links.carrier == carrier]['p_nom_opt'].sum()
            print(f"  {carrier:45s}: {cap:>10.1f} MW")

    # === Generatoren (NUR STROM!) ===
    gens = n.generators
    if country_code != "ALL":
        gens = gens[gens.bus.str.startswith(country_code)]
    else:
        gens = gens[~gens.bus.str.contains("EU", case=False, na=False)]

    # Filtere: nur ELECTRICITY_CARRIERS
    gens = gens[gens.carrier.isin(ELECTRICITY_CARRIERS)]

    # WICHTIG: Exclude Wärme-Oil! (nur wenn "boiler" oder "heater" im Namen)
    gens_filtered = gens[~gens.carrier.str.contains("boiler|heater|heat", case=False, na=False)]

    if debug and country_code in ["DE", "ALL"]:
        oil_gens_gen = gens_filtered[gens_filtered.carrier == "oil"]
        if not oil_gens_gen.empty:
            print(f"\n✓ Oil GENERATOREN (Strom, nach Filter):")
            print(f"  Anzahl: {len(oil_gens_gen)}")
            print(f"  Gesamt: {oil_gens_gen['p_nom_opt'].sum():>10.1f} MW")

    if not gens_filtered.empty:
        gen_capacity = gens_filtered.groupby("carrier")[pcol].sum()
        parts.append(gen_capacity)

    # === Links (NUR Stromerzeuger!) ===
    links = n.links
    if country_code != "ALL":
        links = links[
            (links.bus0.str.startswith(country_code)) |
            (links.bus1.str.startswith(country_code))
            ]
    else:
        links = links[
            (~links.bus0.str.contains("EU", case=False, na=False)) &
            (~links.bus1.str.contains("EU", case=False, na=False))
            ]

    # Nur Stromerzeuger Links
    links = links[links.carrier.isin(ELECTRICITY_CARRIERS)]

    # WICHTIG: Exclude Wärme-Oil + andere Heat Links
    links_filtered = links[~links.carrier.str.contains(
        "heat pump|heater|pipeline|inverter|retrofit|boiler",
        case=False, na=False
    )]

    # Exclude nuclear Links (nur Generatoren!)
    links_filtered = links_filtered[~(links_filtered.carrier == "nuclear")]

    if debug and country_code in ["DE", "ALL"]:
        oil_links_gen = links_filtered[links_filtered.carrier == "oil"]
        if not oil_links_gen.empty:
            print(f"\n✓ Oil LINKS (Strom, nach Filter):")
            print(f"  Anzahl: {len(oil_links_gen)}")
            print(f"  Gesamt: {oil_links_gen['p_nom_opt'].sum():>10.1f} MW")

    if not links_filtered.empty:
        link_capacity = links_filtered.groupby("carrier")[pcol].sum()
        parts.append(link_capacity)

    # === Storage Units (Entladung = Stromerzeuung) ===
    sus = n.storage_units
    if country_code != "ALL":
        sus = sus[sus.bus.str.startswith(country_code)]
    else:
        sus = sus[~sus.bus.str.contains("EU", case=False, na=False)]

    sus = sus[sus.carrier.isin(STORAGE_DISCHARGE_CARRIERS)]

    if not sus.empty:
        sus_capacity = sus.groupby("carrier")[pcol].sum()
        parts.append(sus_capacity)

    if not parts:
        return pd.Series(dtype=float)

    result = pd.concat(parts).groupby(level=0).sum() / 1000.0  # MW → GW

    # Entferne Technologien mit sehr kleiner Kapazität
    result = result[result > 0.01]  # Nur > 0.01 GW (10 MW) anzeigen

    if debug and country_code in ["DE", "ALL"]:
        oil_total = result.get("oil", 0)
        print(f"\n📊 FINAL Oil im Plot: {oil_total:.2f} GW ({oil_total * 1000:.0f} MW)")
        print(f"{'=' * 80}\n")

    return result


# =====================================================================
# --- Plotfunktion ---
# =====================================================================
def plot_installed_capacities(df: pd.DataFrame, years, country, config: PlottingConfig, scenario: str, ref_path: str):
    """Zeichnet Balkenplot über die Jahre."""
    if df.empty:
        print(f"⚠️ Keine Daten für {country}")
        return

    # Reihenfolge für Konsistenz (NUR STROMERZEUGER!)
    order = [
        # Kernenergie
        "nuclear",

        # Erneuerbare
        "solar", "solar rooftop", "solar-hsat",
        "onwind", "offwind-ac", "offwind-dc",
        "ror",

        # Biomasse CHP (erzeugt Strom)
        "urban central solid biomass CHP",
        "urban central solid biomass CHP CC",

        # Konventionelle (Kohle, Gas, Oil)
        "lignite", "coal", "oil",
        "CCGT", "OCGT",

        # Wasserstoff
        "H2 turbine", "H2 OCGT", "H2 Fuel Cell",

        # Speicher (Entladung)
        "PHS", "battery discharger", "home battery discharger"
    ]

    df = df[years]
    df = df.reindex(order, axis=0).dropna(how="all")

    # Farben laden
    colors = []
    for c in df.index:
        if c in config.CARRIER_COLORS:
            colors.append(config.CARRIER_COLORS[c])
        elif "biomass" in c.lower() or "CHP" in c:
            colors.append("#2D5016")  # Dunkelgrün für Biomasse CHP
        elif "battery" in c.lower():
            colors.append("#FF6B6B")  # Rot für Batterien
        elif c == "oil":
            colors.append("#8B4513")  # Braun für Oil
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

        # Beschriftungen auf den Balken (nur wenn signifikant)
        for x, h, b in zip(x_pos, vals, bottom):
            if h > (df.sum().max() * 0.02):
                ax.text(x, b + h / 2, f"{int(round(h))}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
        bottom += vals

    ax.set_xticks(x_pos)
    ax.set_xticklabels(years)
    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    ax.set_ylabel("Kapazität (GW)", fontsize=config.FONT_SIZES.get("label", 11))
    ax.set_xlabel("Jahr", fontsize=config.FONT_SIZES.get("label", 11))
    ax.set_title(f"Stromerzeugende Kapazität in {country} – {scenario}",
                 fontsize=config.FONT_SIZES.get("title", 14))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Legende
    ax.legend(title="Technologien", loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=config.FONT_SIZES.get("legend", 9))

    fig.tight_layout()
    save_dir = os.path.join(config.BASE_SAVE_PATH,
                            os.path.basename(os.path.dirname(os.path.dirname(ref_path))),
                            "plots_installed_capacities")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"Plot_installed_capacities_{country}.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Plot gespeichert: {save_path}")


# =====================================================================
# --- Hauptablauf ---
# =====================================================================
def main():
    config = PlottingConfig()
    networks = config.get_networks()

    # Szenario-Namen direkt aus der Konfiguration lesen
    scenario = config.SCENARIO_SELECTION.replace("_", " ")

    network_paths = networks[config.SCENARIO_SELECTION] if isinstance(networks, dict) else networks
    ref_path = network_paths[0]

    all_years = []
    all_data = {c: {} for c in config.get_countries()}

    for path in network_paths:
        if not os.path.isfile(path):
            continue
        m = re.search(r"_(\d{4})\.nc$", path)
        if not m:
            continue
        year = int(m.group(1))
        all_years.append(year)
        print(f"📂 Lade Netzwerk {year}: {path}")
        n = pypsa.Network(path)
        for country in config.get_countries():
            # DEBUG: Zeige Oil-Details für DE
            debug_mode = (country == "DE")
            all_data[country][year] = get_installed_capacity(n, country, debug=debug_mode)

    years = sorted(set(all_years))

    for country, year_dict in all_data.items():
        if not year_dict:
            continue
        df = pd.DataFrame(year_dict).fillna(0)
        plot_installed_capacities(df, years, country, config, scenario, ref_path)


if __name__ == "__main__":
    main()
