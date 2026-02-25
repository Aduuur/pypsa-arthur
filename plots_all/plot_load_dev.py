#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot_load_evolution_single (REDUCED VERSION)
============================================================
- Erzeugt einzelne Plots für die Lastentwicklung jedes Landes.
- OHNE Speicherbeladung.
- OHNE Verkehr (da im Szenario deaktiviert).
- FOKUS: Grundlast, Wärme, Netzverluste.
"""

import os
import re
import pypsa
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config_final import PlottingConfig

# =====================================================================
# --- Konfiguration: Farben & Namen
# =====================================================================

LOAD_CATEGORIES = [
    "Grundlast",
    "Wärme (Wärmepumpen/Heizstäbe)",
    "Netzverluste"
]

LOAD_COLORS = {
    "Grundlast": "#333333",  # Dunkelgrau
    "Wärme (Wärmepumpen/Heizstäbe)": "#d95f02",  # Orange
    "Netzverluste": "#7f7f7f",  # Grau
}


# =====================================================================
# --- Rechenlogik ---
# =====================================================================

def get_electric_bus_ids(n: pypsa.Network, country_code: str):
    if country_code == "ALL":
        country_buses = n.buses
    else:
        country_buses = n.buses[n.buses.index.str.startswith(country_code)]

    def is_elec(carrier):
        c = str(carrier).lower()
        # Wir schließen alles aus, was kein Strom ist
        if any(x in c for x in
               ["heat", "water", "gas", "h2", "hydrogen", "oil", "biomass", "co2", "tank", "pit", "coal", "lignite"]):
            return False
        return True

    return country_buses[country_buses.carrier.map(is_elec)].index


def get_annual_load_breakdown(n: pypsa.Network, country_code: str) -> pd.Series:
    elec_buses = get_electric_bus_ids(n, country_code)
    weights = n.snapshot_weightings.generators
    data = {k: 0.0 for k in LOAD_CATEGORIES}

    # --- 1. Grundlast (Klassisch) ---
    load_ids = n.loads[(n.loads.bus.isin(elec_buses)) & (n.loads.carrier == "electricity")].index
    if not load_ids.empty:
        p_col = "p" if "p" in n.loads_t else "p_set"
        val = n.loads_t[p_col][load_ids].multiply(weights, axis=0).sum().sum()
        data["Grundlast"] += val

    # --- 2. Sektorkopplung (Nur Wärme) ---
    links_out = n.links[n.links.bus0.isin(elec_buses)]

    for name, row in links_out.iterrows():
        c = row.carrier.lower()

        # Wir filtern H2, Verkehr und Speicher hier direkt raus, indem wir sie gar nicht erst abfragen
        # Nur Wärme interessiert uns
        if not ("heat pump" in c or "resistive heater" in c):
            continue

        # Zeitreihe holen
        p_col = "p0" if "p0" in n.links_t else "p"
        if name not in n.links_t[p_col]: continue

        val = (n.links_t[p_col][name] * weights).sum()

        # Zuordnung
        if "heat pump" in c or "resistive heater" in c:
            data["Wärme (Wärmepumpen/Heizstäbe)"] += val

    # --- 3. Verluste ---
    # Verteilnetz
    dist = n.links[(n.links.bus0.isin(elec_buses)) & (n.links.carrier.str.contains("distribution grid"))]
    if not dist.empty:
        valid_cols = dist.index.intersection(n.links_t.p0.columns)
        if not valid_cols.empty:
            p0 = n.links_t.p0[valid_cols]
            eff = dist.loc[valid_cols, "efficiency"]
            data["Netzverluste"] += p0.multiply(1 - eff, axis=1).multiply(weights, axis=0).sum().sum()

    # AC/DC Intern
    lines = n.lines[(n.lines.bus0.isin(elec_buses)) & (n.lines.bus1.isin(elec_buses))]
    if not lines.empty:
        data["Netzverluste"] += (n.lines_t.p0[lines.index] + n.lines_t.p1[lines.index]).multiply(weights,
                                                                                                 axis=0).sum().sum()

    # DC Intern (Links)
    dc_links = n.links[(n.links.carrier == "DC") & (n.links.bus0.isin(elec_buses)) & (n.links.bus1.isin(elec_buses))]
    if not dc_links.empty:
        valid_cols = dc_links.index.intersection(n.links_t.p0.columns)
        if not valid_cols.empty:
            # p0 + p1 (p1 ist negativ)
            loss = (n.links_t.p0[valid_cols] + n.links_t.p1[valid_cols]).multiply(weights, axis=0).sum().sum()
            data["Netzverluste"] += loss

    return pd.Series(data) / 1e6  # TWh


# =====================================================================
# --- PLOTTING ---
# =====================================================================

def plot_load_single(df, years, country, config, ref_path):
    if df.empty or df.sum().sum() == 0: return

    # Reihenfolge im Stack (Unten -> Oben)
    order = [
        "Grundlast",
        "Wärme (Wärmepumpen/Heizstäbe)",
        "Netzverluste"
    ]

    df_plot = df.reindex(order).dropna(how="all").fillna(0)
    total_max = df_plot.sum(axis=0).max()

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(years))
    bottom = np.zeros(len(years))

    for cat in df_plot.index:
        color = LOAD_COLORS.get(cat, "#333333")
        vals = df_plot.loc[cat, years].values
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.5, label=cat)

        # Zahlen in Balken (ab 3% Anteil)
        threshold = 0.03 * total_max
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= threshold:
                ax.text(x[i], b + v / 2, f"{int(round(v))}", ha="center", va="center",
                        color="white", fontweight="bold", fontsize=10)
        bottom += vals

    ax.set_title(f"Entwicklung der Stromnachfrage: {country}", fontsize=16, pad=15)
    ax.set_ylabel("Energie [TWh/a]", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Legende
    handles, labels = ax.get_legend_handles_labels()
    # Invertieren, damit Reihenfolge wie im Stack (oben=oben)
    ax.legend(handles[::-1], labels[::-1], title="Sektoren", loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()

    save_dir = os.path.join(config.BASE_SAVE_PATH,
                            os.path.basename(os.path.dirname(os.path.dirname(ref_path))),
                            "load_analysis_single_reduced")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"Load_Evolution_Reduced_{country}.png"), dpi=300)
    plt.close(fig)


# =====================================================================
# --- MAIN ---
# =====================================================================

def main():
    config = PlottingConfig()
    networks = config.get_networks()
    if isinstance(networks, dict): networks = networks[config.SCENARIO_SELECTION]

    data_all_countries = {c: {} for c in config.get_countries()}
    years = []

    for path in networks:
        try:
            m = re.search(r"_(\d{4})\.nc$", path)
            if not m: continue
            year = int(m.group(1))
            years.append(year)

            print(f"📂 Verarbeite Jahr {year}...")
            n = pypsa.Network(path)

            for c in config.get_countries():
                data_all_countries[c][year] = get_annual_load_breakdown(n, c)
        except Exception as e:
            print(f"Fehler bei {path}: {e}")

    years = sorted(list(set(years)))

    for c in config.get_countries():
        print(f"📊 Erstelle Plot für {c}...")
        df_country = pd.DataFrame(data_all_countries[c])
        plot_load_single(df_country, years, c, config, networks[0])


if __name__ == "__main__":
    main()