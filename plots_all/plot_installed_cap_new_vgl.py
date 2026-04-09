#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot_scenario_comparison_installed_capacity
============================================================
Vergleicht installierte Kapazitaeten zwischen zwei Szenarien.

ARO-Modus: robustes Portfolio (average) vs. Worst-Case-Dispatch (dunkelflaute)
Klassischer Modus: 'both' aus config (average vs. dunkelflaute)
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
    "oil": "Oel",
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
    "GB": "Grossbritannien",
    "PL": "Polen",
    "SE": "Schweden",
    "NO": "Norwegen",
    "NL": "Niederlande",
    "BE": "Belgien",
    "AT": "Oesterreich",
    "CH": "Schweiz",
    "DK": "Daenemark",
    "CZ": "Tschechien"
}

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

def _extract_year_from_network(n: pypsa.Network, path: str) -> int | None:
    """
    Extrahiert das Jahr robust:
    1. Aus n.snapshots (funktioniert auch bei ARO-Dispatch-Pfaden)
    2. Fallback: Regex auf den Dateinamen
    """
    try:
        return int(n.snapshots[0].year)
    except Exception:
        pass
    m = re.search(r"_(\d{4})[\._ ]", os.path.basename(path))
    if m:
        return int(m.group(1))
    return None


def get_country_codes(network):
    all_buses = network.buses.index
    prefixes = set(b[:2] for b in all_buses)
    countries = [p for p in prefixes if p != "EU" and p.isalpha() and p.isupper() and len(p) == 2]
    return sorted(list(countries))


def get_installed_capacity(n: pypsa.Network, country_code: str) -> pd.Series:
    parts = []
    pcol = "p_nom_opt"

    valid_countries = get_country_codes(n)
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
    hydro_sus = hydro_sus[~hydro_sus.bus.str.contains("EU", case=False, na=False)]
    if country_code != "ALL":
        hydro_sus = hydro_sus[hydro_sus.bus.str.startswith(country_code)]
    hydro_sus = hydro_sus[hydro_sus.carrier.isin(["hydro", "ror"])]
    if not hydro_sus.empty:
        parts.append(hydro_sus.groupby("carrier")[pcol].sum())

    if not parts:
        return pd.Series(dtype=float)

    raw = pd.concat(parts).groupby(level=0).sum() / 1000.0
    clean = {}

    solar_sum = raw.filter(regex="solar").sum()
    raw = raw.drop(raw.index[raw.index.str.contains("solar")], errors="ignore")
    if solar_sum > 0:
        clean["Photovoltaik"] = solar_sum

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
# --- PLOTTING ---
# =====================================================================

def plot_scenario_comparison(df_avg, df_df, years, country, config):
    all_techs = df_avg.index.union(df_df.index)
    df_avg = df_avg.reindex(all_techs, fill_value=0)
    df_df = df_df.reindex(all_techs, fill_value=0)

    ALWAYS_KEEP = [
        "Braunkohle", "Steinkohle", "Biomasse", "Oel", "Kernkraft",
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
        "Braunkohle", "Steinkohle", "Oel",
        "Erdgas (GuD)", "Erdgas (Gasturbine)",
        "Pumpspeicher", "Batteriespeicher", "Heimspeicher",
        "H2-Turbine", "H2-Gasturbine"
    ]
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
    ax.set_title(
        f"Installierte Kapazitaet: {c_name}\n(Links: Robustes Portfolio | Rechts: Worst-Case Dispatch)",
        fontsize=FONT_SIZES["title"], pad=20
    )
    ax.set_ylabel("Kapazitaet (GW)", fontsize=FONT_SIZES["axis_label"])
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
# --- MAIN (standalone + ARO-kompatibel via run_analysis.py) ---
# =====================================================================

def main(aro_network=None, aro_robust_network=None):
    """
    Parameters
    ----------
    aro_network : str or None
        Im ARO-Modus: Pfad zum Worst-Case-Dispatch-Netzwerk (= 'dunkelflaute'-Seite).
    aro_robust_network : str or None
        Im ARO-Modus: Pfad zum robusten Portfolio-Netzwerk (= 'average'-Seite).
        Falls None aber aro_network gesetzt, wird nur ein Balken gezeigt.
    """
    config = PlottingConfig()

    # ----------------------------------------------------------------
    # ARO-Modus
    # ----------------------------------------------------------------
    if aro_network is not None:
        paths_avg = [aro_robust_network] if aro_robust_network and os.path.isfile(aro_robust_network) else []
        paths_df  = [aro_network]        if os.path.isfile(aro_network) else []

        if not paths_df:
            print(f"\u26a0\ufe0f  ARO-Netzwerk nicht gefunden: {aro_network}")
            return
        if not paths_avg:
            print("\u26a0\ufe0f  Kein robustes Portfolio angegeben — Vergleich nicht moeglich, uebersprungen.")
            return

        countries = config.get_countries()
        data_avg = {c: {} for c in countries}
        data_df  = {c: {} for c in countries}
        years_avg, years_df = [], []

        for path in paths_avg:
            try:
                n = pypsa.Network(path)
                year = _extract_year_from_network(n, path)
                if year is None:
                    continue
                years_avg.append(year)
                for c in countries:
                    data_avg[c][year] = get_installed_capacity(n, c)
            except Exception as e:
                print(f"Err {path}: {e}")

        for path in paths_df:
            try:
                n = pypsa.Network(path)
                year = _extract_year_from_network(n, path)
                if year is None:
                    continue
                years_df.append(year)
                for c in countries:
                    data_df[c][year] = get_installed_capacity(n, c)
            except Exception as e:
                print(f"Err {path}: {e}")

        common_years = sorted(set(years_avg) & set(years_df))
        if not common_years:
            print("\u26a0\ufe0f  Keine gemeinsamen Jahre gefunden (ARO-Modus).")
            return

        save_dir = os.path.join(config.BASE_SAVE_PATH, "scenario_comparison", "installed_capacities")
        os.makedirs(save_dir, exist_ok=True)

        for c in countries:
            df_a = pd.DataFrame(data_avg[c]).fillna(0)
            df_d = pd.DataFrame(data_df[c]).fillna(0)
            if df_a.empty and df_d.empty:
                continue
            fig = plot_scenario_comparison(df_a, df_d, common_years, c, config)
            fname = f"Compare_Capacity_ARO_{c}.png"
            fig.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"\u2705 {fname}")
        return

    # ----------------------------------------------------------------
    # Klassischer Standalone-Modus (mode='both')
    # ----------------------------------------------------------------
    config.SCENARIO_SELECTION = "both"
    networks = config.get_networks()

    if not isinstance(networks, dict) or 'average' not in networks:
        print("\u274c Fehler: SCENARIO_SELECTION muss 'both' sein.")
        return

    countries = config.get_countries()
    data_avg = {c: {} for c in countries}
    data_df  = {c: {} for c in countries}
    years_avg, years_df = [], []

    print("\n--- Lade Szenario: AVERAGE ---")
    for path in networks['average']:
        try:
            n = pypsa.Network(path)
            year = _extract_year_from_network(n, path)
            if year is None:
                continue
            years_avg.append(year)
            print(f"Lade {year}...")
            for c in countries:
                data_avg[c][year] = get_installed_capacity(n, c)
        except Exception as e:
            print(f"Err {path}: {e}")

    print("\n--- Lade Szenario: DUNKELFLAUTE ---")
    for path in networks['dunkelflaute']:
        try:
            n = pypsa.Network(path)
            year = _extract_year_from_network(n, path)
            if year is None:
                continue
            years_df.append(year)
            print(f"Lade {year}...")
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
    print(f"\nGeneriere Plots & Analyse fuer: {common_years}")

    for c in countries:
        df_a = pd.DataFrame(data_avg[c]).fillna(0)
        df_d = pd.DataFrame(data_df[c]).fillna(0)

        all_techs = df_a.index.union(df_d.index)
        df_a_sync = df_a.reindex(all_techs, fill_value=0)
        df_d_sync = df_d.reindex(all_techs, fill_value=0)
        diff_df = df_d_sync - df_a_sync

        if c != "ALL":
            for year in common_years:
                if year in diff_df.columns:
                    for tech, val in diff_df[year].items():
                        if abs(val) > 0.05:
                            diff_records.append({
                                "Land": c,
                                "Jahr": year,
                                "Technologie": tech,
                                "Differenz_GW": val
                            })

        if df_a.empty and df_d.empty:
            continue

        fig = plot_scenario_comparison(df_a, df_d, common_years, c, config)
        fname = f"Compare_Capacity_{c}.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("\n" + "=" * 80)
    print("\U0001f4ca VERGLEICHS-STATISTIK (Dunkelflaute vs. Basisszenario)")
    print("=" * 80)

    if diff_records:
        stats_df = pd.DataFrame(diff_records)
        total_diff_gw = stats_df["Differenz_GW"].sum()
        total_abs_diff_gw = stats_df["Differenz_GW"].abs().sum()
        print(f"\n>>> EU-WEITES GESAMTSYSTEM:")
        print(f"    Netto-Kapazitaetsaenderung: {total_diff_gw:+.2f} GW")
        print(f"    Absolutes Umbau-Volumen:   {total_abs_diff_gw:.2f} GW")

        for country in sorted(stats_df["Land"].unique()):
            c_name = COUNTRY_NAMES.get(country, country)
            print(f"\n" + "-" * 60)
            print(f"\U0001f4cd {c_name} ({country})")
            print("-" * 60)
            c_data = stats_df[stats_df["Land"] == country]
            plus  = c_data[c_data["Differenz_GW"] > 0].sort_values("Differenz_GW", ascending=False).head(10)
            minus = c_data[c_data["Differenz_GW"] < 0].sort_values("Differenz_GW", ascending=True).head(10)
            if not plus.empty:
                print(f"  [+] Zuwachs in Dunkelflaute (Top 10):")
                for _, r in plus.iterrows():
                    print(f"      {r['Jahr']} | {r['Technologie']:<28} | {r['Differenz_GW']:>+8.2f} GW")
            else:
                print("  [+] Keine signifikanten Zuwachse.")
            if not minus.empty:
                print(f"\n  [-] Rueckgang in Dunkelflaute (Top 10):")
                for _, r in minus.iterrows():
                    print(f"      {r['Jahr']} | {r['Technologie']:<28} | {r['Differenz_GW']:>+8.2f} GW")
            else:
                print("  [-] Keine signifikanten Rueckgaenge.")
    else:
        print("Keine signifikanten Unterschiede (> 50 MW) gefunden.")

    print("\n" + "=" * 80)
    print(f"Plots gespeichert unter:\n{save_dir}")


if __name__ == "__main__":
    main()
