#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose-Skript: Fuel-Switch im Wärmesektor (DE 2050)
Features:
- Unterscheidung Groß-WP vs. Dezentrale WP
- High Contrast Colors (Blau vs. Orange/Rot)
- Zeitraum: 1. Jan - 15. Feb
- Fix: get_installed_heat_capacities NameError behoben
"""

import os
import re
import pypsa
import pandas as pd
import matplotlib

# Setze Backend auf Agg (für Server ohne Display), verhindert Qt-Fehler
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config_final import PlottingConfig

# ==========================================
# KONFIGURATION ZEITRAUM
# ==========================================
REFERENCE_YEAR = 2005
START_DATE = f"{REFERENCE_YEAR}-01-07"
END_DATE = f"{REFERENCE_YEAR}-01-28"


def analyze_heat_sector(n, country):
    print(f"\n🔍 Analysiere Wärmesektor für {country} ({START_DATE} bis {END_DATE})...")

    # Prüfen ob Zeitraum im Modell
    if START_DATE not in n.links_t.p0.index or END_DATE not in n.links_t.p0.index:
        print("WARNUNG: Gewünschter Zeitraum nicht vollständig im Modell! Nutze verfügbaren Ausschnitt.")
        # Fallback auf Intersection
        valid_start = max(pd.Timestamp(START_DATE), n.links_t.p0.index[0])
        valid_end = min(pd.Timestamp(END_DATE), n.links_t.p0.index[-1])
        time_slice = slice(valid_start, valid_end)
    else:
        time_slice = slice(START_DATE, END_DATE)

    # 1. Identifiziere Links
    links_in_country = n.links[n.links.bus0.str.startswith(country)]

    # A) Wärmepumpen unterscheiden
    mask_large_hp = (
            links_in_country.carrier.str.contains("heat pump", case=False) &
            links_in_country.carrier.str.contains("central", case=False)
    )
    mask_small_hp = (
            links_in_country.carrier.str.contains("heat pump", case=False) &
            ~links_in_country.carrier.str.contains("central", case=False)
    )

    large_hp_links = links_in_country[mask_large_hp].index
    small_hp_links = links_in_country[mask_small_hp].index

    # B) Heizstäbe
    res_links = links_in_country[
        links_in_country.carrier.str.contains("resistive heater", case=False)
    ].index

    # C) Backup-Kessel (Output bus1 im Land)
    links_heat_output = n.links[n.links.bus1.str.startswith(country)]
    boiler_links = links_heat_output[
        links_heat_output.carrier.str.contains("boiler", case=False)
    ].index

    print(f"Gefunden:")
    print(f"  - Großwärmepumpen (Fernwärme): {len(large_hp_links)}")
    print(f"  - Dezentrale Wärmepumpen:      {len(small_hp_links)}")
    print(f"  - Heizstäbe:                   {len(res_links)}")
    print(f"  - Kessel (Boiler):             {len(boiler_links)}")

    # 2. Daten extrahieren (Strom p0)
    def get_power_consumption(link_indices):
        if len(link_indices) == 0:
            return pd.Series(0.0, index=n.links_t.p0.loc[time_slice].index)
        return n.links_t.p0[link_indices].loc[time_slice].sum(axis=1) / 1000  # MW -> GW

    p_elec = pd.DataFrame(index=n.links_t.p0.loc[time_slice].index)
    p_elec["Groß-WP"] = get_power_consumption(large_hp_links)
    p_elec["Dezentrale WP"] = get_power_consumption(small_hp_links)
    p_elec["Heizstäbe"] = get_power_consumption(res_links)

    print(f"\n📊 Stromverbrauch (Mittelwert im Zeitraum):")
    print(f"  Groß-WP:       {p_elec['Groß-WP'].mean():.2f} GW")
    print(f"  Dezentrale WP: {p_elec['Dezentrale WP'].mean():.2f} GW")

    # 3. Wärmeerzeugung (p1)
    def get_heat_generation(link_indices):
        if len(link_indices) == 0:
            return pd.Series(0.0, index=n.links_t.p0.loc[time_slice].index)
        # p1 ist meist negativ (outflow), daher abs()
        raw = n.links_t.p1[link_indices].loc[time_slice].sum(axis=1) / 1000
        return raw.abs()

    q_heat = pd.DataFrame(index=p_elec.index)

    if len(large_hp_links) > 0:
        q_heat["Groß-WP (Fernw.)"] = get_heat_generation(large_hp_links)
    if len(small_hp_links) > 0:
        q_heat["Dezentrale WP"] = get_heat_generation(small_hp_links)
    if len(res_links) > 0:
        q_heat["Heizstäbe"] = get_heat_generation(res_links)

    for link in boiler_links:
        carrier = n.links.at[link, "carrier"]
        if "gas" in carrier:
            label = "Gas-Kessel"
        elif "oil" in carrier:
            label = "Öl-Kessel"
        elif "biomass" in carrier:
            label = "Biomasse-Kessel"
        else:
            label = "Sonstige Kessel"

        val = n.links_t.p1[link].loc[time_slice] / 1000
        val = val.abs()

        if label not in q_heat:
            q_heat[label] = val
        else:
            q_heat[label] += val

    return p_elec, q_heat

def summarize_heat_energy(q_heat, country):
    """
    Berechnet Wärmeenergie [TWh] und Spitzenleistung [GW_th]
    Nur für Diagnosezwecke (DF-Zeitraum)
    """
    print(f"\n🔥 Wärmebereitstellung während DF ({country}):")

    dt_hours = 1.0  # PyPSA: stündliche Auflösung

    for col in q_heat.columns:
        energy_twh = (q_heat[col].sum() * dt_hours) / 1000
        peak_gw = q_heat[col].max()

        print(f"  {col:20s}: {energy_twh:6.2f} TWh | Peak: {peak_gw:6.1f} GW_th")


# === HIER WAR DER FEHLER: DIESE FUNKTION FEHLTE ===
def get_installed_heat_capacities(n, country):
    """
    Liefert installierte Wärmekapazitäten [GW_th] nach Technologie
    """
    links = n.links.copy()
    if country != "ALL":
        mask_country = (
                links.bus0.str.startswith(country) |
                links.bus1.str.startswith(country)
        )
        links = links[mask_country]

    def filter_cap(mask):
        subset = links[mask]
        if subset.empty: return 0.0
        # Bevorzugt p_nom_opt, sonst p_nom
        col = "p_nom_opt" if "p_nom_opt" in subset.columns else "p_nom"
        return subset[col].sum() / 1000  # MW -> GW

    caps = {}
    caps["Groß-WP"] = filter_cap(
        links.carrier.str.contains("heat pump", case=False) &
        links.carrier.str.contains("central", case=False)
    )
    caps["Dezentrale WP"] = filter_cap(
        links.carrier.str.contains("heat pump", case=False) &
        ~links.carrier.str.contains("central", case=False)
    )
    caps["Heizstäbe"] = filter_cap(
        links.carrier.str.contains("resistive heater", case=False)
    )
    caps["Gas-Kessel"] = filter_cap(links.carrier.str.contains("gas boiler", case=False))
    caps["Öl-Kessel"] = filter_cap(links.carrier.str.contains("oil boiler", case=False))
    caps["Biomasse"] = filter_cap(links.carrier.str.contains("biomass boiler", case=False))

    caps["TOTAL"] = sum(caps.values())
    return caps


def plot_diagnosis(p_elec, q_heat, country, target_year, scenario_name):
    config = PlottingConfig()
    font_sizes = {k: v + 4 for k, v in config.FONT_SIZES.items()}

    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    # ==========================================
    # FARBPALETTE (High Contrast)
    # ==========================================
    colors = {
        "Groß-WP": "#1f77b4",  # Dunkelblau
        "Groß-WP (Fernw.)": "#1f77b4",
        "Dezentrale WP": "#00bfff",  # Cyan/Hellblau
        "Heizstäbe": "#d62728",  # Rot
        "Gas-Kessel": "#ff7f0e",  # Orange
        "Öl-Kessel": "#222222",  # Schwarz
        "Biomasse-Kessel": "#2ca02c",  # Grün
        "Sonstige Kessel": "#7f7f7f"  # Grau
    }

    # --- PLOT 1: Stromverbrauch ---
    ax1 = axes[0]
    elec_cols_prio = ["Groß-WP", "Dezentrale WP", "Heizstäbe"]
    elec_cols = [c for c in elec_cols_prio if c in p_elec.columns]

    if not p_elec.empty and p_elec.sum().sum() > 0:
        c_list_elec = [colors.get(c, "#000000") for c in elec_cols]
        ax1.stackplot(p_elec.index, *[p_elec[c] for c in elec_cols],
                      labels=elec_cols, colors=c_list_elec, alpha=0.9)
        ax1.legend(loc="upper left", fontsize=font_sizes['legend'], frameon=True, framealpha=0.9)
    else:
        ax1.text(0.5, 0.5, "Kein relevanter Stromverbrauch", ha='center', va='center', transform=ax1.transAxes)

    ax1.set_ylabel("Stromverbrauch [GW]", fontsize=font_sizes['label'])
    ax1.set_title(f"Stromverbrauch für Wärme ({country} {target_year})", fontsize=font_sizes['title'])
    ax1.grid(True, alpha=0.4, linestyle='--')

    # --- PLOT 2: Wärmebereitstellung ---
    ax2 = axes[1]
    plot_cols = [c for c in q_heat.columns if q_heat[c].sum() > 0.01]

    if len(plot_cols) > 0:
        # Sortierung: Grundlast unten, Backup oben
        priority = ["Groß-WP (Fernw.)", "Dezentrale WP", "Biomasse-Kessel", "Heizstäbe", "Gas-Kessel", "Öl-Kessel"]
        plot_cols.sort(key=lambda x: priority.index(x) if x in priority else 99)

        c_list_heat = [colors.get(c, "#777777") for c in plot_cols]
        ax2.stackplot(q_heat.index, *[q_heat[c] for c in plot_cols],
                      labels=plot_cols, colors=c_list_heat, alpha=0.9)
        ax2.legend(loc="upper left", fontsize=font_sizes['legend'], ncol=2, frameon=True, framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, "Keine Wärmeerzeugung gefunden", ha='center', va='center', transform=ax2.transAxes)

    ax2.set_ylabel("Wärmeerzeugung [GW$_{th}$]", fontsize=font_sizes['label'])
    ax2.set_title("Wärmebereitstellung nach Technologie", fontsize=font_sizes['title'])
    ax2.grid(True, alpha=0.4, linestyle='--')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))

    plt.tight_layout()

    plot_dir = get_heat_plot_dir(scenario_name)
    fname = f"diagnose_fuelswitch_HIGH_CONTRAST_{scenario_name}_{country}_{target_year}_NUR_DF.png"
    out_path = os.path.join(plot_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot gespeichert (High Contrast): {out_path}")
    plt.close()


def get_heat_plot_dir(scenario_name):
    base_dir = "/shared_endata/atlite_cutouts/results/PyPSA-eur results/final_lots"
    if "dunkelflaute" in scenario_name.lower():
        run_dir = "start_new_myopic_28_bigrun_dunkelflaute_transmission_limited"
    else:
        run_dir = "start_new_myopic_29_bigrun_transmission_limited"

    heat_dir = os.path.join(base_dir, run_dir, "heat_diagnosis")
    os.makedirs(heat_dir, exist_ok=True)
    return heat_dir


def main():
    config = PlottingConfig()
    networks = config.get_networks()
    countries = config.get_countries()

    if not isinstance(networks, dict):
        networks = {config.SCENARIO_SELECTION: networks}

    for scenario_name, paths in networks.items():
        for path in paths:
            match = re.search(r'_(\d{4})\.nc$', path)
            if not match: continue
            target_year = match.group(1)

            if not os.path.exists(path):
                print(f"Fehlt: {path}")
                continue

            print(f"\n📂 Lade {path} ...")
            n = pypsa.Network(path)

            # Statistik (Hier trat der Fehler auf)
            if int(target_year) in [2030, 2040, 2050]:
                print(f"--- Kapazitäten {target_year} ---")
                try:
                    caps = get_installed_heat_capacities(n, "DE")
                    for k, v in caps.items(): print(f"{k:15s}: {v:.2f} GW")
                except NameError as e:
                    print(f"❌ FEHLER: {e}")

            for country in countries:
                if country == "ALL": continue
                try:
                    p_elec, q_heat = analyze_heat_sector(n, country)
                    if country == "DE":
                        summarize_heat_energy(q_heat, country)

                    plot_diagnosis(p_elec, q_heat, country, target_year, scenario_name)
                except Exception as e:
                    print(f"Fehler {country}: {e}")


if __name__ == "__main__":
    main()