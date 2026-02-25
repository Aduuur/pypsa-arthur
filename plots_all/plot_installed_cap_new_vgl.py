#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot_scenario_comparison_installed_capacity
============================================================
Vergleicht installierte Kapazitäten zwischen zwei Szenarien.

"""

import os
import re
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config_final import PlottingConfig

# =====================================================================
# --- KONFIGURATION ---
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

COUNTRY_NAMES = {
    "ALL": "Europa (Gesamt)",
    "DE": "Deutschland",
    "FR": "Frankreich",
    "ES": "Spanien",
    "IT": "Italien",
    "GB": "Großbritannien",
    "PL": "Polen",
    "SE": "Schweden",
    "NO": "Norwegen",
    "NL": "Niederlande",
    "BE": "Belgien",
    "AT": "Österreich",
    "CH": "Schweiz",
    "DK": "Dänemark",
    "CZ": "Tschechien"
}

# Schriftgrößen (+4 gegenüber Standard)
FONT_SIZES = {
    "title": 20,
    "subtitle": 16,
    "axis_label": 16,
    "ticks": 15,
    "bar_text": 13,
    "legend": 14,
    "legend_title": 15
}


# =====================================================================
# --- HILFSFUNKTIONEN ---
# =====================================================================

def get_country_codes(network):
    all_buses = network.buses.index
    prefixes = set(b[:2] for b in all_buses)
    countries = [p for p in prefixes if p != "EU" and p.isalpha() and p.isupper() and len(p) == 2]
    return sorted(list(countries))


def get_installed_capacity(n: pypsa.Network, country_code: str) -> pd.Series:
    parts = []
    pcol = "p_nom_opt"

    valid_countries = get_country_codes(n)
    # Regex für Länderfilterung (Bus-Namen)
    regex_countries = "^(" + "|".join(valid_countries) + ")"

    # 1. Generatoren
    gens = n.generators.copy()
    if country_code != "ALL":
        gens = gens[gens.bus.str.startswith(country_code)]
    else:
        gens = gens[gens.bus.str.contains(regex_countries)]
    gens = gens[gens.carrier.isin(ELECTRICITY_CARRIERS)]

    if not gens.empty:
        parts.append(gens.groupby("carrier")[pcol].sum())

    # 2. Links
    links = n.links.copy()
    if country_code != "ALL":
        links = links[links.bus0.str.startswith(country_code) | links.bus1.str.startswith(country_code)]
    else:
        links = links[links.bus1.str.contains(regex_countries)]
    links = links[links.carrier.isin(ELECTRICITY_CARRIERS)]

    # Nuclear aus Links entfernen
    links = links[links.carrier != "nuclear"]

    if not links.empty:
        CORRECTION_MAP = {
            "OCGT": 0.39, "Erdgas (Gasturbine)": 0.39,
            "CCGT": 0.58, "Erdgas (GuD)": 0.58
        }
        cap_final = links[pcol].copy()
        for tech in links.carrier.unique():
            needs_correction = False
            correction_factor = 1.0
            if tech in CORRECTION_MAP:
                needs_correction = True
                correction_factor = CORRECTION_MAP[tech]
            elif any(x in tech for x in ["OCGT", "CCGT"]):
                needs_correction = True
                correction_factor = 0.58 if "CCGT" in tech else 0.39

            if needs_correction:
                mask = links.carrier == tech
                if "efficiency" in links.columns:
                    eff_vals = links.loc[mask, "efficiency"]
                    if eff_vals.mean() < 0.9:
                        cap_final.loc[mask] *= eff_vals
                    else:
                        cap_final.loc[mask] *= correction_factor
                else:
                    cap_final.loc[mask] *= correction_factor

        parts.append(cap_final.groupby(links.carrier).sum())

    # 3. Storage
    sus = n.storage_units
    if country_code != "ALL":
        sus = sus[sus.bus.str.startswith(country_code)]
    else:
        sus = sus[sus.bus.str.contains(regex_countries)]
    sus = sus[sus.carrier.isin(STORAGE_DISCHARGE_CARRIERS)]
    if not sus.empty:
        parts.append(sus.groupby("carrier")[pcol].sum())

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

    # Aggregation Solar (regex=False wo möglich oder Pattern sicherstellen)
    # Hier nutzen wir filter(regex=...), das erwartet Regex.
    solar_sum = raw.filter(regex="solar").sum()
    # Beim Drop regex=True lassen, aber Pattern ist simple
    raw = raw.drop(raw.index[raw.index.str.contains("solar")], errors="ignore")
    if solar_sum > 0: clean["Photovoltaik"] = solar_sum

    biomass_sum = raw.filter(regex="biomass").sum()
    raw = raw.drop(raw.index[raw.index.str.contains("biomass")], errors="ignore")
    if biomass_sum > 0: clean["Biomasse"] = biomass_sum

    for c, v in raw.items():
        name = CARRIER_TRANSLATION.get(c, c)
        clean[name] = clean.get(name, 0.0) + v

    s = pd.Series(clean)
    return s[s > 0.01]


# =====================================================================
# --- PLOTTING ---
# =====================================================================

def plot_scenario_comparison(df_avg, df_df, years, country, config):
    all_techs = df_avg.index.union(df_df.index)
    df_avg = df_avg.reindex(all_techs, fill_value=0)
    df_df = df_df.reindex(all_techs, fill_value=0)

    ALWAYS_KEEP = [
        "Braunkohle", "Steinkohle", "Biomasse", "Öl", "Kernkraft",
        "Laufwasser", "Wasserkraft", "Erdgas (GuD)", "Erdgas (Gasturbine)",
        "Pumpspeicher", "Batteriespeicher", "Heimspeicher"
    ]
    total_max = max(df_avg.sum().max(), df_df.sum().max())
    MIN_SHARE = 0.01

    keep_mask = (
            (df_avg.max(axis=1) >= MIN_SHARE * total_max) |
            (df_df.max(axis=1) >= MIN_SHARE * total_max) |
            (df_avg.index.isin(ALWAYS_KEEP))
    )
    df_avg = df_avg[keep_mask].copy()
    df_df = df_df[keep_mask].copy()

    order = [
        "Kernkraft", "Photovoltaik",
        "Wind Onshore", "Wind Offshore (AC)", "Wind Offshore (DC)",
        "Laufwasser", "Wasserkraft", "Biomasse",
        "Braunkohle", "Steinkohle", "Öl",
        "Erdgas (GuD)", "Erdgas (Gasturbine)",
        "Pumpspeicher", "Batteriespeicher", "Heimspeicher",
        "H2-Turbine", "H2-Gasturbine"    ]
    common_index = df_avg.index
    final_order = [t for t in order if t in common_index]
    final_order.extend([t for t in common_index if t not in final_order])

    df_avg = df_avg.reindex(final_order)
    df_df = df_df.reindex(final_order)
    techs_ordered = df_avg.index.tolist()

    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(len(years))
    width = 0.35
    bottom_avg = np.zeros(len(years))
    bottom_df = np.zeros(len(years))
    TEXT_THRESHOLD = 0.025 * total_max

    for tech in techs_ordered:
        if tech == "Biomasse":
            color = config.CARRIER_COLORS.get("biomass", "#6a3d9a")
        else:
            k = next((k for k, v in CARRIER_TRANSLATION.items() if v == tech), tech)
            color = config.CARRIER_COLORS.get(k, config.DEFAULT_COLOR)

        vals_avg = df_avg.loc[tech, years].values
        vals_df = df_df.loc[tech, years].values

        ax.bar(x - width / 2, vals_avg, width, bottom=bottom_avg,
               color=color, edgecolor="white", linewidth=0.5, label=tech)
        ax.bar(x + width / 2, vals_df, width, bottom=bottom_df,
               color=color, edgecolor="white", linewidth=0.5, alpha=0.95, hatch="///")

        for i in range(len(years)):
            if vals_avg[i] >= TEXT_THRESHOLD:
                ax.text(x[i] - width / 2, bottom_avg[i] + vals_avg[i] / 2, f"{int(round(vals_avg[i]))}",
                        ha="center", va="center", color="white",
                        fontsize=FONT_SIZES["bar_text"], fontweight="bold")
            if vals_df[i] >= TEXT_THRESHOLD:
                ax.text(x[i] + width / 2, bottom_df[i] + vals_df[i] / 2, f"{int(round(vals_df[i]))}",
                        ha="center", va="center", color="white",
                        fontsize=FONT_SIZES["bar_text"], fontweight="bold")

        bottom_avg += vals_avg
        bottom_df += vals_df

    c_name = COUNTRY_NAMES.get(country, country)
    ax.set_title(f"Installierte Kapazität: {c_name}\n(Links: Basisszenario | Rechts: Dunkelflaute)",
                 fontsize=FONT_SIZES["title"], pad=20)

    ax.set_ylabel("Kapazität (GW)", fontsize=FONT_SIZES["axis_label"])
    ax.set_xlabel("Jahr", fontsize=FONT_SIZES["axis_label"])

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=FONT_SIZES["ticks"])
    ax.tick_params(axis='y', labelsize=FONT_SIZES["ticks"])

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_order = techs_ordered[::-1]
    legend_handles = [by_label[t] for t in legend_order if t in by_label]
    legend_labels = [t for t in legend_order if t in by_label]

    ax.legend(legend_handles, legend_labels,
              title="Technologie",
              fontsize=FONT_SIZES["legend"],
              title_fontsize=FONT_SIZES["legend_title"],
              loc="upper left", bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    return fig


# =====================================================================
# --- MAIN ---
# =====================================================================

def main():
    config = PlottingConfig()
    config.SCENARIO_SELECTION = "both"
    networks = config.get_networks()

    if not isinstance(networks, dict) or 'average' not in networks:
        print("❌ Fehler: SCENARIO_SELECTION muss 'both' sein.")
        return

    countries = config.get_countries()
    data_avg = {c: {} for c in countries}
    data_df = {c: {} for c in countries}
    years_avg, years_df = [], []

    print("\n--- Lade Szenario: AVERAGE ---")
    for path in networks['average']:
        m = re.search(r"_(\d{4})\.nc$", path)
        if not m: continue
        year = int(m.group(1))
        years_avg.append(year)
        try:
            print(f"Lade {year}...")
            n = pypsa.Network(path)
            for c in countries:
                data_avg[c][year] = get_installed_capacity(n, c)
        except Exception as e:
            print(f"Err {path}: {e}")

    print("\n--- Lade Szenario: DUNKELFLAUTE ---")
    for path in networks['dunkelflaute']:
        m = re.search(r"_(\d{4})\.nc$", path)
        if not m: continue
        year = int(m.group(1))
        years_df.append(year)
        try:
            print(f"Lade {year}...")
            n = pypsa.Network(path)
            for c in countries:
                data_df[c][year] = get_installed_capacity(n, c)
        except Exception as e:
            print(f"Err {path}: {e}")

    common_years = sorted(list(set(years_avg) & set(years_df)))
    if not common_years:
        print("Keine gemeinsamen Jahre!")
        return

    save_dir = os.path.join(config.BASE_SAVE_PATH, "scenario_comparison", "installed_capacities")
    os.makedirs(save_dir, exist_ok=True)

    diff_records = []

    print(f"\nGeneriere Plots & Analyse für: {common_years}")

    for c in countries:
        df_a = pd.DataFrame(data_avg[c]).fillna(0)
        df_d = pd.DataFrame(data_df[c]).fillna(0)

        # Berechnungen für Statistik
        all_techs = df_a.index.union(df_d.index)
        df_a_sync = df_a.reindex(all_techs, fill_value=0)
        df_d_sync = df_d.reindex(all_techs, fill_value=0)
        diff_df = df_d_sync - df_a_sync

        if c != "ALL":
            for year in common_years:
                if year in diff_df.columns:
                    for tech, val in diff_df[year].items():
                        if abs(val) > 0.05:  # Schwelle 50 MW
                            diff_records.append({
                                "Land": c,
                                "Jahr": year,
                                "Technologie": tech,
                                "Differenz_GW": val
                            })

        if df_a.empty and df_d.empty: continue

        # Plot
        fig = plot_scenario_comparison(df_a, df_d, common_years, c, config)
        fname = f"Compare_Capacity_{c}.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- STATISTIK AUSGABE ---
    print("\n" + "=" * 80)
    print("📊 VERGLEICHS-STATISTIK (Dunkelflaute vs. Basisszenario)")
    print("=" * 80)

    if diff_records:
        stats_df = pd.DataFrame(diff_records)

        # A) Gesamtsystem
        total_diff_gw = stats_df["Differenz_GW"].sum()
        total_abs_diff_gw = stats_df["Differenz_GW"].abs().sum()
        print(f"\n>>> EU-WEITES GESAMTSYSTEM:")
        print(f"    Netto-Kapazitätsänderung: {total_diff_gw:+.2f} GW")
        print(f"    Absolutes Umbau-Volumen:  {total_abs_diff_gw:.2f} GW")

        # B) Pro Land: Top 10 Zuwachs & Top 10 Rückgang
        unique_countries = sorted(stats_df["Land"].unique())

        for country in unique_countries:
            c_name = COUNTRY_NAMES.get(country, country)
            print(f"\n" + "-" * 60)
            print(f"📍 {c_name} ({country})")
            print("-" * 60)

            c_data = stats_df[stats_df["Land"] == country]

            # Plus: Dunkelflaute > Basis
            plus = c_data[c_data["Differenz_GW"] > 0].sort_values("Differenz_GW", ascending=False).head(10)
            # Minus: Dunkelflaute < Basis
            minus = c_data[c_data["Differenz_GW"] < 0].sort_values("Differenz_GW", ascending=True).head(10)

            if not plus.empty:
                print(f"  [+] Zuwachs in Dunkelflaute (Top 10):")
                for _, r in plus.iterrows():
                    print(f"      {r['Jahr']} | {r['Technologie']:<28} | {r['Differenz_GW']:>+8.2f} GW")
            else:
                print("  [+] Keine signifikanten Zuwächse.")

            if not minus.empty:
                print(f"\n  [-] Rückgang in Dunkelflaute (Top 10):")
                for _, r in minus.iterrows():
                    print(f"      {r['Jahr']} | {r['Technologie']:<28} | {r['Differenz_GW']:>+8.2f} GW")
            else:
                print("  [-] Keine signifikanten Rückgänge.")

    else:
        print("Keine signifikanten Unterschiede (> 50 MW) gefunden.")

    print("\n" + "=" * 80)
    print(f"Plots gespeichert unter:\n{save_dir}")


if __name__ == "__main__":
    main()