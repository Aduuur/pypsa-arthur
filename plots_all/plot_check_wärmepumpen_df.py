#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose-Skript: Fuel-Switch im Wärmesektor (DE 2050)
Features:
- Unterscheidung Groß-WP vs. Dezentrale WP
- High Contrast Colors (Blau vs. Orange/Rot)
- Zeitraum: 7. Jan - 28. Jan
"""

import os
import re
import pypsa
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config_final import PlottingConfig

REFERENCE_YEAR = 2005
START_DATE = f"{REFERENCE_YEAR}-01-07"
END_DATE   = f"{REFERENCE_YEAR}-01-28"


def _get_valid_timeslice(n, start: str, end: str):
    """
    Gibt (valid_start, valid_end) als Timestamps zurück,
    geclipt auf den tatsächlichen Snapshot-Bereich des Netzes.
    """
    idx = n.snapshots
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)
    valid_start = max(ts_start, idx[0])
    valid_end   = min(ts_end,   idx[-1])
    if ts_start < idx[0] or ts_end > idx[-1]:
        print("WARNUNG: Gewünschter Zeitraum nicht vollständig im Modell! Nutze verfügbaren Ausschnitt.")
    return valid_start, valid_end


def _safe_p(df_t, indices, valid_start, valid_end):
    """
    Gibt Series/DataFrame für indices im Zeitraum [valid_start, valid_end] zurück.
    FIX: Nutzt .loc[start:end] statt slice()-Objekt auf DatetimeIndex.
    Gibt 0-Series zurück wenn keine Spalten vorhanden.
    """
    avail = [i for i in indices if i in df_t.columns]
    # FIX: Basis-Index sicher über .loc[start:end] auf dem Zeitreihen-DataFrame
    base_idx = df_t.loc[valid_start:valid_end].index
    if len(avail) == 0:
        return pd.Series(0.0, index=base_idx)
    return df_t[avail].loc[valid_start:valid_end].sum(axis=1) / 1000  # MW -> GW


def analyze_heat_sector(n, country):
    print(f"\n🔍 Analysiere Wärmesektor für {country} ({START_DATE} bis {END_DATE})...")

    valid_start, valid_end = _get_valid_timeslice(n, START_DATE, END_DATE)

    # Identifiziere Links
    links_in_country = n.links[n.links.bus0.str.startswith(country)]

    large_hp_links = links_in_country[
        links_in_country.carrier.str.contains("heat pump", case=False) &
        links_in_country.carrier.str.contains("central", case=False)
    ].index

    small_hp_links = links_in_country[
        links_in_country.carrier.str.contains("heat pump", case=False) &
        ~links_in_country.carrier.str.contains("central", case=False)
    ].index

    res_links = links_in_country[
        links_in_country.carrier.str.contains("resistive heater", case=False)
    ].index

    # Kessel: bus1 im Land (Wärmeoutput)
    links_heat_output = n.links[n.links.bus1.str.startswith(country)]
    boiler_links = links_heat_output[
        links_heat_output.carrier.str.contains("boiler", case=False)
    ].index

    print(f"Gefunden:")
    print(f"  - Großwärmepumpen (Fernwärme): {len(large_hp_links)}")
    print(f"  - Dezentrale Wärmepumpen:      {len(small_hp_links)}")
    print(f"  - Heizstäbe:                   {len(res_links)}")
    print(f"  - Kessel (Boiler):             {len(boiler_links)}")

    # Stromverbrauch (p0)
    # FIX: _safe_p bekommt valid_start/valid_end statt slice-Objekt
    p0 = n.links_t.p0
    p_elec = pd.DataFrame(index=_safe_p(p0, large_hp_links, valid_start, valid_end).index)
    p_elec["Groß-WP"]      = _safe_p(p0, large_hp_links, valid_start, valid_end).values
    p_elec["Dezentrale WP"] = _safe_p(p0, small_hp_links,  valid_start, valid_end).values
    p_elec["Heizstäbe"]    = _safe_p(p0, res_links,        valid_start, valid_end).values

    print(f"\n📊 Stromverbrauch (Mittelwert im Zeitraum):")
    print(f"  Groß-WP:       {p_elec['Groß-WP'].mean():.2f} GW")
    print(f"  Dezentrale WP: {p_elec['Dezentrale WP'].mean():.2f} GW")

    # Wärmeerzeugung (p1)
    p1 = n.links_t.p1
    q_heat = pd.DataFrame(index=p_elec.index)

    if len(large_hp_links) > 0:
        q_heat["Groß-WP (Fernw.)"] = _safe_p(p1, large_hp_links, valid_start, valid_end).abs().values
    if len(small_hp_links) > 0:
        q_heat["Dezentrale WP"]     = _safe_p(p1, small_hp_links,  valid_start, valid_end).abs().values
    if len(res_links) > 0:
        q_heat["Heizstäbe"]         = _safe_p(p1, res_links,        valid_start, valid_end).abs().values

    for link in boiler_links:
        carrier = n.links.at[link, "carrier"]
        if "gas"     in carrier: label = "Gas-Kessel"
        elif "oil"   in carrier: label = "Öl-Kessel"
        elif "biomass" in carrier: label = "Biomasse-Kessel"
        else: label = "Sonstige Kessel"

        if link not in p1.columns:
            continue
        val = p1[link].loc[valid_start:valid_end].abs() / 1000
        # FIX: q_heat.get() funktioniert nicht für DataFrames — direkte Spaltenprüfung
        if label in q_heat.columns:
            q_heat[label] = q_heat[label].values + val.values
        else:
            q_heat[label] = val.values

    return p_elec, q_heat


def summarize_heat_energy(q_heat, country):
    print(f"\n🔥 Wärmebereitstellung während DF ({country}):")
    dt_hours = 1.0
    all_labels = [
        "Groß-WP (Fernw.)", "Dezentrale WP", "Heizstäbe",
        "Gas-Kessel", "Öl-Kessel", "Biomasse-Kessel", "Sonstige Kessel"
    ]
    for col in all_labels:
        if col in q_heat.columns:
            energy_twh = (q_heat[col].sum() * dt_hours) / 1000
            peak_gw    = q_heat[col].max()
        else:
            energy_twh = 0.0
            peak_gw    = 0.0
        print(f"  {col:20s}: {energy_twh:6.2f} TWh | Peak: {peak_gw:6.1f} GW_th")


def get_installed_heat_capacities(n, country):
    links = n.links.copy()
    if country != "ALL":
        links = links[
            links.bus0.str.startswith(country) |
            links.bus1.str.startswith(country)
        ]

    def filter_cap(mask):
        subset = links[mask]
        if subset.empty: return 0.0
        col = "p_nom_opt" if "p_nom_opt" in subset.columns else "p_nom"
        return subset[col].sum() / 1000

    caps = {
        "Groß-WP":      filter_cap(links.carrier.str.contains("heat pump", case=False) & links.carrier.str.contains("central", case=False)),
        "Dezentrale WP": filter_cap(links.carrier.str.contains("heat pump", case=False) & ~links.carrier.str.contains("central", case=False)),
        "Heizstäbe":    filter_cap(links.carrier.str.contains("resistive heater", case=False)),
        "Gas-Kessel":   filter_cap(links.carrier.str.contains("gas boiler",      case=False)),
        "Öl-Kessel":    filter_cap(links.carrier.str.contains("oil boiler",      case=False)),
        "Biomasse":     filter_cap(links.carrier.str.contains("biomass boiler",  case=False)),
    }
    caps["TOTAL"] = sum(caps.values())
    return caps


def plot_diagnosis(p_elec, q_heat, country, target_year, scenario_name, save_dir):
    config = PlottingConfig()
    font_sizes = {k: v + 4 for k, v in config.FONT_SIZES.items()}

    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    colors = {
        "Groß-WP": "#1f77b4",
        "Groß-WP (Fernw.)": "#1f77b4",
        "Dezentrale WP": "#00bfff",
        "Heizstäbe": "#d62728",
        "Gas-Kessel": "#ff7f0e",
        "Öl-Kessel": "#222222",
        "Biomasse-Kessel": "#2ca02c",
        "Sonstige Kessel": "#7f7f7f"
    }

    # PLOT 1: Stromverbrauch
    ax1 = axes[0]
    elec_cols = [c for c in ["Groß-WP", "Dezentrale WP", "Heizstäbe"] if c in p_elec.columns]
    if not p_elec.empty and len(elec_cols) > 0 and p_elec[elec_cols].sum().sum() > 0:
        ax1.stackplot(
            p_elec.index,
            *[p_elec[c] for c in elec_cols],
            labels=elec_cols,
            colors=[colors.get(c, "#000000") for c in elec_cols],
            alpha=0.9
        )
        ax1.legend(loc="upper left", fontsize=font_sizes['legend'], frameon=True, framealpha=0.9)
    else:
        ax1.text(0.5, 0.5, "Kein relevanter Stromverbrauch",
                 ha='center', va='center', transform=ax1.transAxes,
                 fontsize=font_sizes.get('label', 12))
    ax1.set_ylabel("Stromverbrauch [GW]", fontsize=font_sizes['label'])
    ax1.set_title(f"Stromverbrauch für Wärme ({country} {target_year})", fontsize=font_sizes['title'])
    ax1.grid(True, alpha=0.4, linestyle='--')

    # PLOT 2: Wärmeerzeugung
    ax2 = axes[1]
    plot_cols = [c for c in q_heat.columns if q_heat[c].sum() > 0.01]
    if len(plot_cols) > 0:
        priority = ["Groß-WP (Fernw.)", "Dezentrale WP", "Biomasse-Kessel",
                    "Heizstäbe", "Gas-Kessel", "Öl-Kessel"]
        plot_cols.sort(key=lambda x: priority.index(x) if x in priority else 99)
        ax2.stackplot(
            q_heat.index,
            *[q_heat[c] for c in plot_cols],
            labels=plot_cols,
            colors=[colors.get(c, "#777777") for c in plot_cols],
            alpha=0.9
        )
        ax2.legend(loc="upper left", fontsize=font_sizes['legend'],
                   ncol=2, frameon=True, framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, "Keine Wärmeerzeugung gefunden",
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=font_sizes.get('label', 12))

    ax2.set_ylabel("Wärmeerzeugung [GW$_{th}$]", fontsize=font_sizes['label'])
    ax2.set_title("Wärmebereitstellung nach Technologie", fontsize=font_sizes['title'])
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))

    plt.tight_layout()

    fname = f"diagnose_fuelswitch_{scenario_name}_{country}_{target_year}.png"
    out_path = os.path.join(save_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot gespeichert: {out_path}")
    plt.close()


def main():
    config = PlottingConfig()
    networks = config.get_networks()
    countries = config.get_countries()

    save_dir = os.path.join(config.BASE_SAVE_PATH, "heat_diagnosis")
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Speichere Plots nach: {save_dir}")

    if not isinstance(networks, dict):
        networks = {config.SCENARIO_SELECTION: networks}

    for scenario_name, paths in networks.items():
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            match = re.search(r'_(\d{4})\.nc$', path)
            if not match:
                continue
            target_year = match.group(1)

            if not os.path.exists(path):
                print(f"Fehlt: {path}")
                continue

            print(f"\n📂 Lade {path} ...")
            n = pypsa.Network(path)

            if int(target_year) in [2030, 2040, 2050]:
                print(f"--- Kapazitäten {target_year} ---")
                try:
                    caps = get_installed_heat_capacities(n, "DE")
                    for k, v in caps.items():
                        print(f"{k:15s}: {v:.2f} GW")
                except Exception as e:
                    print(f"❌ FEHLER get_installed_heat_capacities: {e}")

            for country in countries:
                if country == "ALL":
                    continue
                try:
                    p_elec, q_heat = analyze_heat_sector(n, country)
                    if country == "DE":
                        summarize_heat_energy(q_heat, country)
                    # FIX: plot_diagnosis wird IMMER aufgerufen (auch wenn Werte 0).
                    # Vorher wurde bei leerem q_heat kein Plot erstellt.
                    plot_diagnosis(p_elec, q_heat, country, target_year, scenario_name, save_dir)
                except Exception as e:
                    print(f"Fehler {country}: {e}")
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    main()
