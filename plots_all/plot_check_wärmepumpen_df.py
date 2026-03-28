#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose-Skript: Fuel-Switch im Wärmesektor (DE 2050)
Features:
- Unterscheidung Groß-WP vs. Dezentrale WP
- High Contrast Colors (Blau vs. Orange/Rot)
- Zeitraum: 1. Jan - 15. Feb
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

# ==========================================
# KONFIGURATION ZEITRAUM
# ==========================================
REFERENCE_YEAR = 2005
START_DATE = f"{REFERENCE_YEAR}-01-07"
END_DATE   = f"{REFERENCE_YEAR}-01-28"


def _get_valid_timeslice(n, start: str, end: str):
    """
    FIX: links_t.p0 hat einen eigenen Snapshot-Index.
    Ableiten aus n.snapshots ist zuverlässiger als direktes .loc[] auf String-Datum.
    Gibt (time_slice, warned) zurück.
    """
    idx = n.snapshots
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)

    valid_start = max(ts_start, idx[0])
    valid_end   = min(ts_end,   idx[-1])

    warned = (ts_start < idx[0]) or (ts_end > idx[-1])
    if warned:
        print("WARNUNG: Gewünschter Zeitraum nicht vollständig im Modell! Nutze verfügbaren Ausschnitt.")

    return slice(valid_start, valid_end), warned


def _safe_p(df_t, indices, time_slice):
    """
    FIX: Gibt 0-Series zurück, falls keine der indices in df_t.columns sind.
    Verhindert nan durch fehlende Spalten nach .loc[].
    """
    avail = [i for i in indices if i in df_t.columns]
    base_idx = df_t.loc[time_slice].index
    if len(avail) == 0:
        return pd.Series(0.0, index=base_idx)
    return df_t[avail].loc[time_slice].sum(axis=1) / 1000  # MW -> GW


def analyze_heat_sector(n, country):
    print(f"\n🔍 Analysiere Wärmesektor für {country} ({START_DATE} bis {END_DATE})...")

    # FIX: robuster Timeslice aus n.snapshots (nicht links_t.p0.index, der evtl. leer ist)
    time_slice, _ = _get_valid_timeslice(n, START_DATE, END_DATE)

    # 1. Identifiziere Links
    links_in_country = n.links[n.links.bus0.str.startswith(country)]

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

    res_links = links_in_country[
        links_in_country.carrier.str.contains("resistive heater", case=False)
    ].index

    links_heat_output = n.links[n.links.bus1.str.startswith(country)]
    boiler_links = links_heat_output[
        links_heat_output.carrier.str.contains("boiler", case=False)
    ].index

    print(f"Gefunden:")
    print(f"  - Großwärmepumpen (Fernwärme): {len(large_hp_links)}")
    print(f"  - Dezentrale Wärmepumpen:      {len(small_hp_links)}")
    print(f"  - Heizstäbe:                   {len(res_links)}")
    print(f"  - Kessel (Boiler):             {len(boiler_links)}")

    # 2. Stromverbrauch (p0) – FIX: _safe_p statt direktes loc[]
    p_elec = pd.DataFrame(index=n.snapshots[time_slice])
    p_elec["Groß-WP"]      = _safe_p(n.links_t.p0, large_hp_links, time_slice)
    p_elec["Dezentrale WP"] = _safe_p(n.links_t.p0, small_hp_links, time_slice)
    p_elec["Heizstäbe"]    = _safe_p(n.links_t.p0, res_links,      time_slice)

    print(f"\n📊 Stromverbrauch (Mittelwert im Zeitraum):")
    print(f"  Groß-WP:       {p_elec['Groß-WP'].mean():.2f} GW")
    print(f"  Dezentrale WP: {p_elec['Dezentrale WP'].mean():.2f} GW")

    # 3. Wärmeerzeugung (p1)
    q_heat = pd.DataFrame(index=p_elec.index)

    if len(large_hp_links) > 0:
        q_heat["Groß-WP (Fernw.)"] = _safe_p(n.links_t.p1, large_hp_links, time_slice).abs()
    if len(small_hp_links) > 0:
        q_heat["Dezentrale WP"]    = _safe_p(n.links_t.p1, small_hp_links, time_slice).abs()
    if len(res_links) > 0:
        q_heat["Heizstäbe"]        = _safe_p(n.links_t.p1, res_links,      time_slice).abs()

    for link in boiler_links:
        carrier = n.links.at[link, "carrier"]
        if "gas"    in carrier: label = "Gas-Kessel"
        elif "oil"  in carrier: label = "Öl-Kessel"
        elif "biomass" in carrier: label = "Biomasse-Kessel"
        else: label = "Sonstige Kessel"

        if link not in n.links_t.p1.columns:
            continue
        val = n.links_t.p1[link].loc[time_slice].abs() / 1000
        q_heat[label] = q_heat.get(label, 0) + val

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
        mask_country = (
            links.bus0.str.startswith(country) |
            links.bus1.str.startswith(country)
        )
        links = links[mask_country]

    def filter_cap(mask):
        subset = links[mask]
        if subset.empty: return 0.0
        col = "p_nom_opt" if "p_nom_opt" in subset.columns else "p_nom"
        return subset[col].sum() / 1000  # MW -> GW

    caps = {}
    caps["Groß-WP"]      = filter_cap(
        links.carrier.str.contains("heat pump", case=False) &
        links.carrier.str.contains("central", case=False)
    )
    caps["Dezentrale WP"] = filter_cap(
        links.carrier.str.contains("heat pump", case=False) &
        ~links.carrier.str.contains("central", case=False)
    )
    caps["Heizstäbe"]    = filter_cap(links.carrier.str.contains("resistive heater", case=False))
    caps["Gas-Kessel"]   = filter_cap(links.carrier.str.contains("gas boiler",      case=False))
    caps["Öl-Kessel"]    = filter_cap(links.carrier.str.contains("oil boiler",      case=False))
    caps["Biomasse"]     = filter_cap(links.carrier.str.contains("biomass boiler",  case=False))
    caps["TOTAL"]        = sum(v for k, v in caps.items() if k != "TOTAL")
    return caps


def plot_diagnosis(p_elec, q_heat, country, target_year, scenario_name, save_dir):
    """FIX: save_dir wird jetzt von main() übergeben (kein hardcoded /shared_endata)."""
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

    # --- PLOT 1: Stromverbrauch ---
    ax1 = axes[0]
    elec_cols = [c for c in ["Groß-WP", "Dezentrale WP", "Heizstäbe"] if c in p_elec.columns]
    if not p_elec.empty and p_elec[elec_cols].sum().sum() > 0:
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
                 ha='center', va='center', transform=ax1.transAxes)
    ax1.set_ylabel("Stromverbrauch [GW]", fontsize=font_sizes['label'])
    ax1.set_title(f"Stromverbrauch für Wärme ({country} {target_year})", fontsize=font_sizes['title'])
    ax1.grid(True, alpha=0.4, linestyle='--')

    # --- PLOT 2: Wärmeerzeugung ---
    ax2 = axes[1]
    plot_cols = [c for c in q_heat.columns if q_heat[c].sum() > 0.01]
    if len(plot_cols) > 0:
        priority = ["Groß-WP (Fernw.)", "Dezentrale WP", "Biomasse-Kessel", "Heizstäbe", "Gas-Kessel", "Öl-Kessel"]
        plot_cols.sort(key=lambda x: priority.index(x) if x in priority else 99)
        ax2.stackplot(
            q_heat.index,
            *[q_heat[c] for c in plot_cols],
            labels=plot_cols,
            colors=[colors.get(c, "#777777") for c in plot_cols],
            alpha=0.9
        )
        ax2.legend(loc="upper left", fontsize=font_sizes['legend'], ncol=2, frameon=True, framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, "Keine Wärmeerzeugung gefunden",
                 ha='center', va='center', transform=ax2.transAxes)

    ax2.set_ylabel("Wärmeerzeugung [GW$_{th}$]", fontsize=font_sizes['label'])
    ax2.set_title("Wärmebereitstellung nach Technologie", fontsize=font_sizes['title'])
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))

    plt.tight_layout()

    fname = f"diagnose_fuelswitch_HIGH_CONTRAST_{scenario_name}_{country}_{target_year}_NUR_DF.png"
    out_path = os.path.join(save_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot gespeichert (High Contrast): {out_path}")
    plt.close()


def main():
    config = PlottingConfig()
    networks = config.get_networks()
    countries = config.get_countries()

    # FIX: save_dir aus config.BASE_SAVE_PATH (kein hardcoded /shared_endata)
    save_dir = os.path.join(config.BASE_SAVE_PATH, "heat_diagnosis")
    os.makedirs(save_dir, exist_ok=True)

    if not isinstance(networks, dict):
        networks = {config.SCENARIO_SELECTION: networks}

    for scenario_name, paths in networks.items():
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            match = re.search(r'_(\d{4})\.nc$', path)
            if not match: continue
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
                if country == "ALL": continue
                try:
                    p_elec, q_heat = analyze_heat_sector(n, country)
                    if country == "DE":
                        summarize_heat_energy(q_heat, country)
                    plot_diagnosis(p_elec, q_heat, country, target_year, scenario_name, save_dir)
                except Exception as e:
                    print(f"Fehler {country}: {e}")


if __name__ == "__main__":
    main()
