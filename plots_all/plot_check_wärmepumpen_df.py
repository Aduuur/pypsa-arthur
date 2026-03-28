#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose-Skript: Fuel-Switch im Wärmesektor
Features:
- Unterscheidung Groß-WP vs. Dezentrale WP
- Filterung über n.links.index (carrier + country-Tag), NICHT bus0.startswith
- Zeitreihen-Lookup direkt über links_t.p0 / links_t.p1
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
    idx = n.snapshots
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)
    valid_start = max(ts_start, idx[0])
    valid_end   = min(ts_end,   idx[-1])
    if ts_start < idx[0] or ts_end > idx[-1]:
        print("WARNUNG: Gewünschter Zeitraum nicht vollständig im Modell!")
    return valid_start, valid_end


def _links_for_country(n, country):
    """
    Gibt alle Links für ein Land zurück.
    Strategie (robuster als bus0.startswith):
      1. Versuche n.links[n.links.index.str.contains(country)] -- Link-Name enthält Landcode
      2. Fallback: bus0 oder bus1 startswith country
    Gibt den gefilterten links-DataFrame zurück.
    """
    # Strategie 1: Link-Index enthält Länderkürzel (PyPSA-Eur Standard)
    by_name = n.links[n.links.index.str.startswith(country + " ") |
                      n.links.index.str.startswith(country + "0") |
                      n.links.index.str.startswith(country + "1")]
    if not by_name.empty:
        return by_name
    # Fallback: bus0 oder bus1
    return n.links[
        n.links.bus0.str.startswith(country) |
        n.links.bus1.str.startswith(country)
    ]


def _safe_p(df_t, indices, valid_start, valid_end):
    """
    Summiert Zeitreihen für alle indices die in df_t.columns vorhanden sind.
    Gibt 0-Series zurück wenn keine vorhanden.
    Nutzt .loc[start:end] (DatetimeIndex-kompatibel).
    """
    avail = [i for i in indices if i in df_t.columns]
    base_idx = df_t.loc[valid_start:valid_end].index
    if not avail:
        return pd.Series(0.0, index=base_idx)
    return df_t[avail].loc[valid_start:valid_end].sum(axis=1) / 1000  # MW -> GW


def _debug_links_t(n, links_idx, label):
    """Gibt aus, wie viele der gefundenen Links tatsächlich in links_t.p0/p1 vorhanden sind."""
    in_p0 = [l for l in links_idx if l in n.links_t.p0.columns]
    in_p1 = [l for l in links_idx if l in n.links_t.p1.columns]
    print(f"  [{label}] {len(links_idx)} Links gefunden, "
          f"davon {len(in_p0)} in links_t.p0, {len(in_p1)} in links_t.p1")
    if not in_p0 and not in_p1 and len(links_idx) > 0:
        print(f"    ⚠️  KEINE Zeitreihen! Beispiel-Link: '{links_idx[0]}'")
        print(f"    bus0='{n.links.at[links_idx[0], 'bus0']}' | "
              f"carrier='{n.links.at[links_idx[0], 'carrier']}'")


def analyze_heat_sector(n, country):
    print(f"\n🔍 Analysiere Wärmesektor für {country} ({START_DATE} bis {END_DATE})...")
    valid_start, valid_end = _get_valid_timeslice(n, START_DATE, END_DATE)

    links = _links_for_country(n, country)

    large_hp_links = links[
        links.carrier.str.contains("heat pump", case=False) &
        links.carrier.str.contains("central", case=False)
    ].index

    small_hp_links = links[
        links.carrier.str.contains("heat pump", case=False) &
        ~links.carrier.str.contains("central", case=False)
    ].index

    res_links = links[
        links.carrier.str.contains("resistive heater", case=False)
    ].index

    # Kessel: carrier enthält "boiler" -- bus1 Strategie als zusätzlicher Fallback
    boiler_links = links[
        links.carrier.str.contains("boiler", case=False)
    ].index
    if len(boiler_links) == 0:
        # Fallback: alle Links deren bus1 im Land liegt
        boiler_links = n.links[
            n.links.bus1.str.startswith(country) &
            n.links.carrier.str.contains("boiler", case=False)
        ].index

    print(f"  Großwärmepumpen (Fernwärme): {len(large_hp_links)}")
    print(f"  Dezentrale Wärmepumpen:      {len(small_hp_links)}")
    print(f"  Heizstäbe:                   {len(res_links)}")
    print(f"  Kessel (Boiler):             {len(boiler_links)}")

    # Diagnose: tatsächlich in links_t vorhanden?
    _debug_links_t(n, large_hp_links,  "Groß-WP")
    _debug_links_t(n, small_hp_links,  "Dez-WP")
    _debug_links_t(n, res_links,       "Heizstab")
    _debug_links_t(n, boiler_links,    "Boiler")

    p0 = n.links_t.p0
    p1 = n.links_t.p1

    # Stromverbrauch: primär aus p0
    # Falls p0 leer: elektrischer Input = Wärmeoutput (p1) / COP (efficiency)
    def get_elec_input(link_idx):
        in_p0 = [l for l in link_idx if l in p0.columns]
        if in_p0:
            return _safe_p(p0, in_p0, valid_start, valid_end)
        # Fallback via p1 / efficiency
        in_p1 = [l for l in link_idx if l in p1.columns]
        if in_p1:
            result = pd.Series(0.0, index=p1.loc[valid_start:valid_end].index)
            for l in in_p1:
                eff = n.links.at[l, "efficiency"] if "efficiency" in n.links.columns else 1.0
                if eff == 0:
                    eff = 1.0
                result += p1[l].loc[valid_start:valid_end].abs() / eff / 1000
            return result
        return pd.Series(0.0, index=p1.loc[valid_start:valid_end].index
                         if not p1.empty else pd.DatetimeIndex([]))

    base_idx_series = get_elec_input(large_hp_links
                                     if len(large_hp_links) > 0 else small_hp_links
                                     if len(small_hp_links) > 0 else res_links)
    if len(base_idx_series) == 0:
        # Letzter Fallback: direkt aus Snapshot-Index
        base_idx = n.snapshots[
            (n.snapshots >= valid_start) & (n.snapshots <= valid_end)
        ]
        base_idx_series = pd.Series(0.0, index=base_idx)

    p_elec = pd.DataFrame(index=base_idx_series.index)
    p_elec["Groß-WP"]      = get_elec_input(large_hp_links).reindex(p_elec.index, fill_value=0.0)
    p_elec["Dezentrale WP"] = get_elec_input(small_hp_links).reindex(p_elec.index, fill_value=0.0)
    p_elec["Heizstäbe"]    = get_elec_input(res_links).reindex(p_elec.index, fill_value=0.0)

    print(f"  Groß-WP Mittelwert:     {p_elec['Groß-WP'].mean():.3f} GW")
    print(f"  Dezentrale WP Mittelwert: {p_elec['Dezentrale WP'].mean():.3f} GW")

    # Wärmeerzeugung aus p1
    q_heat = pd.DataFrame(index=p_elec.index)

    if len(large_hp_links) > 0:
        q_heat["Groß-WP (Fernw.)"] = _safe_p(p1, large_hp_links, valid_start, valid_end).abs().reindex(p_elec.index, fill_value=0.0)
    if len(small_hp_links) > 0:
        q_heat["Dezentrale WP"]     = _safe_p(p1, small_hp_links,  valid_start, valid_end).abs().reindex(p_elec.index, fill_value=0.0)
    if len(res_links) > 0:
        q_heat["Heizstäbe"]         = _safe_p(p1, res_links,        valid_start, valid_end).abs().reindex(p_elec.index, fill_value=0.0)

    for link in boiler_links:
        carrier = n.links.at[link, "carrier"]
        if "gas"     in carrier: label = "Gas-Kessel"
        elif "oil"   in carrier: label = "Öl-Kessel"
        elif "biomass" in carrier: label = "Biomasse-Kessel"
        else: label = "Sonstige Kessel"

        if link not in p1.columns:
            continue
        val = p1[link].loc[valid_start:valid_end].abs() / 1000
        if label in q_heat.columns:
            q_heat[label] = q_heat[label].values + val.reindex(p_elec.index, fill_value=0.0).values
        else:
            q_heat[label] = val.reindex(p_elec.index, fill_value=0.0).values

    return p_elec, q_heat


def summarize_heat_energy(q_heat, country):
    print(f"\n🔥 Wärmebereitstellung während DF ({country}):")
    all_labels = [
        "Groß-WP (Fernw.)", "Dezentrale WP", "Heizstäbe",
        "Gas-Kessel", "Öl-Kessel", "Biomasse-Kessel", "Sonstige Kessel"
    ]
    for col in all_labels:
        if col in q_heat.columns:
            energy_twh = q_heat[col].sum() / 1000
            peak_gw    = q_heat[col].max()
        else:
            energy_twh = 0.0
            peak_gw    = 0.0
        print(f"  {col:20s}: {energy_twh:6.2f} TWh | Peak: {peak_gw:6.1f} GW_th")


def get_installed_heat_capacities(n, country):
    links = _links_for_country(n, country)

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

    ax1 = axes[0]
    elec_cols = [c for c in ["Groß-WP", "Dezentrale WP", "Heizstäbe"] if c in p_elec.columns]
    if not p_elec.empty and elec_cols and p_elec[elec_cols].sum().sum() > 0:
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

    ax2 = axes[1]
    plot_cols = [c for c in q_heat.columns if q_heat[c].sum() > 0.01]
    if plot_cols:
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

            # Einmalige Diagnose: zeige Beispiel-Links und ob Zeitreihen vorhanden
            print(f"  links_t.p0 Spalten: {len(n.links_t.p0.columns)} | "
                  f"links_t.p1 Spalten: {len(n.links_t.p1.columns)}")
            hp_all = n.links[n.links.carrier.str.contains("heat pump", case=False)]
            print(f"  Alle WP-Links im Netz: {len(hp_all)}")
            if not hp_all.empty:
                ex = hp_all.index[0]
                print(f"  Beispiel WP-Link: '{ex}' | bus0='{n.links.at[ex,'bus0']}' | "
                      f"carrier='{n.links.at[ex,'carrier']}'")
                print(f"  In links_t.p0: {ex in n.links_t.p0.columns} | "
                      f"In links_t.p1: {ex in n.links_t.p1.columns}")

            if int(target_year) in [2030, 2040, 2050]:
                try:
                    caps = get_installed_heat_capacities(n, "DE")
                    print(f"--- Kapazitäten {target_year} (DE) ---")
                    for k, v in caps.items():
                        print(f"  {k:15s}: {v:.2f} GW")
                except Exception as e:
                    print(f"❌ FEHLER get_installed_heat_capacities: {e}")

            for country in countries:
                if country == "ALL":
                    continue
                try:
                    p_elec, q_heat = analyze_heat_sector(n, country)
                    if country == "DE":
                        summarize_heat_energy(q_heat, country)
                    plot_diagnosis(p_elec, q_heat, country, target_year, scenario_name, save_dir)
                except Exception as e:
                    print(f"Fehler {country}: {e}")
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    main()
