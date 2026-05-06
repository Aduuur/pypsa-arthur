#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot_scenario_comparison_installed_capacity
============================================================
Vergleicht installierte Kapazitaeten zwischen zwei Szenarien.

ARO-Modus: robustes Portfolio (links) vs. Basisjahr-Planungsnetz (rechts)
Klassischer Modus: 'both' aus config (average vs. dunkelflaute)

FIXES:
  #1 — Import war `from config_final import PlottingConfig` → jetzt korrekt
       `from master_config import PlottingConfig`
  #4 — Speicherpfad im ARO-Modus nutzt jetzt `config.PLOT_OUTPUT_PATH`
       (run-spezifisch) statt `config.BASE_SAVE_PATH` (globaler Root)
"""

import os
import re
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# FIX #1: war `from config_final import PlottingConfig` — config_final existiert nicht.
# PlottingConfig ist in master_config definiert.
from master_config import PlottingConfig

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
    "PHS",
    "OCGT methanol", "biomass",  # FIX: fehlende AC-Erzeuger
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
        from master_config import get_planning_year
        return get_planning_year(n, getattr(n, '_source_path', ''))
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

def plot_scenario_comparison(df_robust, df_basis, years, country, config,
                             label_left="Robustes Portfolio",
                             label_right="Basisjahr"):
    all_techs = df_robust.index.union(df_basis.index)
    df_robust = df_robust.reindex(all_techs, fill_value=0)
    df_basis  = df_basis.reindex(all_techs, fill_value=0)

    ALWAYS_KEEP = [
        "Braunkohle", "Steinkohle", "Biomasse", "Oel", "Kernkraft",
        "Laufwasser", "Wasserkraft", "Erdgas (GuD)", "Erdgas (Gasturbine)",
        "Pumpspeicher", "Batteriespeicher", "Heimspeicher"
    ]
    total_max = max(df_robust.sum().max(), df_basis.sum().max())
    MIN_SHARE = 0.01

    keep_mask = (
        (df_robust.max(axis=1) >= MIN_SHARE * total_max) |
        (df_basis.max(axis=1)  >= MIN_SHARE * total_max) |
        (df_robust.index.isin(ALWAYS_KEEP))
    )
    df_robust = df_robust[keep_mask].copy()
    df_basis  = df_basis[keep_mask].copy()

    order = [
        "Kernkraft", "Photovoltaik",
        "Wind Onshore", "Wind Offshore (AC)", "Wind Offshore (DC)",
        "Laufwasser", "Wasserkraft", "Biomasse",
        "Braunkohle", "Steinkohle", "Oel",
        "Erdgas (GuD)", "Erdgas (Gasturbine)",
        "Pumpspeicher", "Batteriespeicher", "Heimspeicher",
        "H2-Turbine", "H2-Gasturbine"
    ]
    common_index = df_robust.index
    final_order = [t for t in order if t in common_index]
    final_order.extend([t for t in common_index if t not in final_order])

    df_robust = df_robust.reindex(final_order)
    df_basis  = df_basis.reindex(final_order)
    techs_ordered = df_robust.index.tolist()

    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(len(years))
    width = 0.35
    bottom_robust = np.zeros(len(years))
    bottom_basis  = np.zeros(len(years))
    TEXT_THRESHOLD = 0.025 * total_max

    for tech in techs_ordered:
        if tech == "Biomasse":
            color = config.CARRIER_COLORS.get("biomass", "#6a3d9a")
        else:
            k = next((k for k, v in CARRIER_TRANSLATION.items() if v == tech), tech)
            color = config.CARRIER_COLORS.get(k, config.DEFAULT_COLOR)

        vals_robust = df_robust.loc[tech, years].values
        vals_basis  = df_basis.loc[tech, years].values

        ax.bar(x - width / 2, vals_robust, width, bottom=bottom_robust,
               color=color, edgecolor="white", linewidth=0.5, label=tech)
        ax.bar(x + width / 2, vals_basis, width, bottom=bottom_basis,
               color=color, edgecolor="white", linewidth=0.5, alpha=0.95, hatch="///")

        for i in range(len(years)):
            if vals_robust[i] >= TEXT_THRESHOLD:
                ax.text(x[i] - width / 2, bottom_robust[i] + vals_robust[i] / 2,
                        f"{int(round(vals_robust[i]))}",
                        ha="center", va="center", color="white",
                        fontsize=FONT_SIZES["bar_text"], fontweight="bold")
            if vals_basis[i] >= TEXT_THRESHOLD:
                ax.text(x[i] + width / 2, bottom_basis[i] + vals_basis[i] / 2,
                        f"{int(round(vals_basis[i]))}",
                        ha="center", va="center", color="white",
                        fontsize=FONT_SIZES["bar_text"], fontweight="bold")

        bottom_robust += vals_robust
        bottom_basis  += vals_basis

    c_name = COUNTRY_NAMES.get(country, country)
    ax.set_title(
        f"Installierte Kapazitaet: {c_name}\n"
        f"(Links: {label_left} | Rechts: {label_right})",
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

def main(aro_network=None, aro_robust_network=None, aro_basis_network=None):
    """
    Parameters
    ----------
    aro_network : str or None
        Veraltet / ungenutzt im ARO-Modus. Wird ignoriert wenn aro_robust_network
        und aro_basis_network gesetzt sind. Bleibt fuer Rueckwaertskompatibilitaet
        als Trigger fuer den ARO-Zweig erhalten (prueft ob ARO-Modus aktiv).
    aro_robust_network : str or None
        Pfad zum robusten Portfolio-Netzwerk (linke Balken).
    aro_basis_network : str or None
        Pfad zum Basisjahr-Planungsnetz (rechte Balken, Vergleichsreferenz).
        Frueherer Name: aro_network (war faelschlicherweise Worst-Case-Dispatch).
    """
    config = PlottingConfig()

    # ----------------------------------------------------------------
    # ARO-Modus: wird aktiviert wenn aro_robust_network ODER aro_network gesetzt.
    # Fuer den Vergleich werden robust (links) und basis (rechts) benoetigt.
    # ----------------------------------------------------------------
    aro_mode = (aro_robust_network is not None) or (aro_network is not None)

    if aro_mode:
        # Rueckwaertskompatibilitaet: frueheres aro_network war der Dispatch-Pfad
        # und wurde als rechte Seite missbraucht. Jetzt ist aro_basis_network
        # die korrekte rechte Seite. Falls aro_basis_network nicht gesetzt,
        # Fallback auf aro_network (altes Verhalten, zeigt Warnung).
        path_robust = aro_robust_network
        path_basis  = aro_basis_network

        if path_basis is None and aro_network is not None:
            print(
                "⚠️  aro_basis_network nicht gesetzt — Fallback auf aro_network.\n"
                "   Bitte run_analysis.py aktualisieren: aro_basis_network=<Basisjahr-Pfad>."
            )
            path_basis = aro_network

        paths_robust = [path_robust] if path_robust and os.path.isfile(path_robust) else []
        paths_basis  = [path_basis]  if path_basis  and os.path.isfile(path_basis)  else []

        if not paths_robust:
            print("⚠️  Kein robustes Portfolio angegeben — Vergleich nicht moeglich, uebersprungen.")
            return
        if not paths_basis:
            print("⚠️  Kein Basisjahr-Planungsnetz angegeben — Vergleich nicht moeglich, uebersprungen.")
            return

        countries = config.get_countries()
        data_robust = {c: {} for c in countries}
        data_basis  = {c: {} for c in countries}
        years_robust, years_basis = [], []

        for path in paths_robust:
            try:
                n = pypsa.Network(path)
                year = _extract_year_from_network(n, path)
                if year is None:
                    continue
                years_robust.append(year)
                for c in countries:
                    data_robust[c][year] = get_installed_capacity(n, c)
            except Exception as e:
                print(f"Err {path}: {e}")

        for path in paths_basis:
            try:
                n = pypsa.Network(path)
                year = _extract_year_from_network(n, path)
                if year is None:
                    continue
                years_basis.append(year)
                for c in countries:
                    data_basis[c][year] = get_installed_capacity(n, c)
            except Exception as e:
                print(f"Err {path}: {e}")

        # Falls die Jahreszahlen verschieden sind (z.B. Robust 2050, Basis 2050),
        # versuchen wir gemeinsame Jahre; bei keinem Treffer nehmen wir alle
        # verfuegbaren Jahre und haengen sie nebeneinander.
        common_years = sorted(set(years_robust) & set(years_basis))
        if not common_years:
            all_years = sorted(set(years_robust) | set(years_basis))
            print(
                f"⚠️  Keine gemeinsamen Jahre (Robust: {years_robust}, Basis: {years_basis}).\n"
                f"   Plotte alle verfuegbaren Jahre nebeneinander: {all_years}"
            )
            common_years = all_years

        # FIX #4: im ARO-Modus PLOT_OUTPUT_PATH (run-spezifisch) statt
        # BASE_SAVE_PATH (globaler Root) verwenden, damit Plots in den
        # richtigen Run-Unterordner geschrieben werden.
        save_dir = os.path.join(
            config.PLOT_OUTPUT_PATH,          # ← FIX: war config.BASE_SAVE_PATH
            "scenario_comparison",
            "installed_capacities",
        )
        os.makedirs(save_dir, exist_ok=True)

        for c in countries:
            df_r = pd.DataFrame(data_robust[c]).reindex(columns=common_years).fillna(0)
            df_b = pd.DataFrame(data_basis[c]).reindex(columns=common_years).fillna(0)
            if df_r.empty and df_b.empty:
                continue
            fig = plot_scenario_comparison(
                df_r, df_b, common_years, c, config,
                label_left="Robustes Portfolio",
                label_right="Basisjahr",
            )
            fname = f"Compare_Capacity_ARO_{c}.png"
            fig.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"✅ {fname}")
        return

    # ----------------------------------------------------------------
    # Klassischer Standalone-Modus (mode='both')
    # ----------------------------------------------------------------
    config.SCENARIO_SELECTION = "both"
    networks = config.get_networks()

    if not isinstance(networks, dict) or 'average' not in networks:
        print("❌ Fehler: SCENARIO_SELECTION muss 'both' sein.")
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

    # FIX #4: auch im klassischen Modus run-spezifischen Pfad nutzen
    save_dir = os.path.join(
        config.PLOT_OUTPUT_PATH,              # ← FIX: war config.BASE_SAVE_PATH
        "scenario_comparison",
        "installed_capacities",
    )
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

        fig = plot_scenario_comparison(
            df_a, df_d, common_years, c, config,
            label_left="Durchschnitt",
            label_right="Dunkelflaute",
        )
        fname = f"Compare_Capacity_{c}.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("\n" + "=" * 80)
    print("📊 VERGLEICHS-STATISTIK (Dunkelflaute vs. Basisszenario)")
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
            print(f"📍 {c_name} ({country})")
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