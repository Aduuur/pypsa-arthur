#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose-Skript: Fuel-Switch im Wärmesektor
- Unterstützt PyPSA perfect-foresight (MultiIndex snapshots) UND myopic (DatetimeIndex)
- Filterung über n.links.index (Link-Name enthält Länderkürzel)
- Zeitraum: REFERENCE_YEAR Jan 7-28
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


def _resolve_snapshots(n):
    """
    Gibt den tatsächlichen Zeitreihen-Index zurück der in links_t verwendet wird.
    - Bei myopic / single-year: n.snapshots ist DatetimeIndex -> direkt verwenden
    - Bei perfect-foresight: n.snapshots ist MultiIndex (period, timestep)
      -> links_t.p0.index ist der innere Timestep-Index (DatetimeIndex)
    Gibt (index, is_multiindex) zurück.
    """
    snap = n.snapshots
    if isinstance(snap, pd.MultiIndex):
        # Perfect-foresight: innerer Index aus links_t direkt lesen
        if not n.links_t.p0.empty:
            return n.links_t.p0.index, True
        if not n.links_t.p1.empty:
            return n.links_t.p1.index, True
        # Fallback: zweite Ebene des MultiIndex
        return snap.get_level_values(1), True
    return snap, False


def _get_valid_mask(ts_index, start: str, end: str):
    """
    Gibt eine Boolean-Maske für ts_index im Bereich [start, end] zurück.
    Funktioniert für DatetimeIndex und MultiIndex-innere-Ebene.
    """
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)

    # Falls ts_index ein MultiIndex ist: nach innerem Level filtern
    if isinstance(ts_index, pd.MultiIndex):
        inner = ts_index.get_level_values(-1)
    else:
        inner = ts_index

    # Sicherstellen dass inner ein DatetimeIndex ist
    inner = pd.DatetimeIndex(inner)
    mask = (inner >= ts_start) & (inner <= ts_end)

    if not mask.any():
        print(f"WARNUNG: Kein Zeitraum {start}--{end} im Modell gefunden!")
        print(f"  Verfügbarer Bereich: {inner[0]} -- {inner[-1]}")
    return mask


def _safe_p(df_t, indices, mask):
    """
    Summiert Zeitreihen für indices im Bereich der Boolean-Maske.
    Gibt 0-Series zurück wenn keine Spalten vorhanden.
    MW -> GW Konversion.
    """
    avail = [i for i in indices if i in df_t.columns]
    base_idx = df_t.index[mask]
    if not avail:
        return pd.Series(0.0, index=base_idx)
    sub = df_t[avail].loc[mask]
    return sub.sum(axis=1) / 1000  # MW -> GW


def _links_for_country(n, country):
    """
    Filtert Links für ein Land über den Link-Index (PyPSA-Eur-Standard: Link-Name startet mit Länderkürzel).
    Fallback: bus0 oder bus1 startswith country.
    """
    by_name = n.links[
        n.links.index.str.startswith(country + " ") |
        n.links.index.str.startswith(country + "0") |
        n.links.index.str.startswith(country + "1")
    ]
    if not by_name.empty:
        return by_name
    return n.links[
        n.links.bus0.str.startswith(country) |
        n.links.bus1.str.startswith(country)
    ]


def _debug_links_t(n, links_idx, label):
    in_p0 = [l for l in links_idx if l in n.links_t.p0.columns]
    in_p1 = [l for l in links_idx if l in n.links_t.p1.columns]
    print(f"  [{label}] {len(links_idx)} Links, {len(in_p0)} in p0, {len(in_p1)} in p1")


def analyze_heat_sector(n, country, ts_index, mask):
    print(f"\n🔍 {country}...")

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
    boiler_links = links[
        links.carrier.str.contains("boiler", case=False)
    ].index
    if len(boiler_links) == 0:
        boiler_links = n.links[
            n.links.bus1.str.startswith(country) &
            n.links.carrier.str.contains("boiler", case=False)
        ].index

    print(f"  WP-zentral: {len(large_hp_links)} | WP-dez: {len(small_hp_links)} "
          f"| Heizstab: {len(res_links)} | Boiler: {len(boiler_links)}")
    _debug_links_t(n, large_hp_links, "Groß-WP")
    _debug_links_t(n, small_hp_links, "Dez-WP")
    _debug_links_t(n, res_links,      "Heizstab")
    _debug_links_t(n, boiler_links,   "Boiler")

    p0 = n.links_t.p0
    p1 = n.links_t.p1
    base_idx = ts_index[mask]

    # Timestamps für Plot-Index: innerer Level falls MultiIndex
    if isinstance(base_idx, pd.MultiIndex):
        plot_idx = pd.DatetimeIndex(base_idx.get_level_values(-1))
    else:
        plot_idx = pd.DatetimeIndex(base_idx)

    def get_series(df, idx_list):
        avail = [l for l in idx_list if l in df.columns]
        if not avail:
            return pd.Series(0.0, index=plot_idx)
        sub = df[avail].loc[mask].sum(axis=1) / 1000
        sub.index = plot_idx
        return sub

    p_elec = pd.DataFrame(index=plot_idx)
    p_elec["Groß-WP"]      = get_series(p0, large_hp_links)
    p_elec["Dezentrale WP"] = get_series(p0, small_hp_links)
    p_elec["Heizstäbe"]    = get_series(p0, res_links)

    print(f"  Groß-WP Ø: {p_elec['Groß-WP'].mean():.3f} GW | "
          f"Dez-WP Ø: {p_elec['Dezentrale WP'].mean():.3f} GW | "
          f"Heizstab Ø: {p_elec['Heizstäbe'].mean():.3f} GW")

    q_heat = pd.DataFrame(index=plot_idx)
    if len(large_hp_links) > 0:
        q_heat["Groß-WP (Fernw.)"] = get_series(p1, large_hp_links).abs()
    if len(small_hp_links) > 0:
        q_heat["Dezentrale WP"]     = get_series(p1, small_hp_links).abs()
    if len(res_links) > 0:
        q_heat["Heizstäbe"]         = get_series(p1, res_links).abs()

    for link in boiler_links:
        carrier = n.links.at[link, "carrier"]
        if "gas"      in carrier: label = "Gas-Kessel"
        elif "oil"    in carrier: label = "Öl-Kessel"
        elif "biomass" in carrier: label = "Biomasse-Kessel"
        else: label = "Sonstige Kessel"
        if link not in p1.columns:
            continue
        val = p1[[link]].loc[mask].iloc[:, 0].abs() / 1000
        val.index = plot_idx
        if label in q_heat.columns:
            q_heat[label] += val
        else:
            q_heat[label] = val

    return p_elec, q_heat


def summarize_heat_energy(q_heat, country):
    print(f"\n🔥 Wärme {country}:")
    for col in ["Groß-WP (Fernw.)", "Dezentrale WP", "Heizstäbe",
                "Gas-Kessel", "Öl-Kessel", "Biomasse-Kessel", "Sonstige Kessel"]:
        if col in q_heat.columns:
            print(f"  {col:20s}: {q_heat[col].sum()/1000:6.2f} TWh | "
                  f"Peak: {q_heat[col].max():6.1f} GW_th")
        else:
            print(f"  {col:20s}:   0.00 TWh | Peak:    0.0 GW_th")


def get_installed_heat_capacities(n, country):
    links = _links_for_country(n, country)

    def filter_cap(mask):
        subset = links[mask]
        if subset.empty: return 0.0
        col = "p_nom_opt" if "p_nom_opt" in subset.columns else "p_nom"
        return subset[col].sum() / 1000

    caps = {
        "Groß-WP":       filter_cap(links.carrier.str.contains("heat pump", case=False) & links.carrier.str.contains("central", case=False)),
        "Dezentrale WP":  filter_cap(links.carrier.str.contains("heat pump", case=False) & ~links.carrier.str.contains("central", case=False)),
        "Heizstäbe":     filter_cap(links.carrier.str.contains("resistive heater", case=False)),
        "Gas-Kessel":    filter_cap(links.carrier.str.contains("gas boiler",       case=False)),
        "Öl-Kessel":     filter_cap(links.carrier.str.contains("oil boiler",        case=False)),
        "Biomasse":      filter_cap(links.carrier.str.contains("biomass boiler",   case=False)),
    }
    caps["TOTAL"] = sum(caps.values())
    return caps


def plot_diagnosis(p_elec, q_heat, country, target_year, scenario_name, save_dir):
    config = PlottingConfig()
    font_sizes = {k: v + 4 for k, v in config.FONT_SIZES.items()}
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    colors = {
        "Groß-WP": "#1f77b4", "Groß-WP (Fernw.)": "#1f77b4",
        "Dezentrale WP": "#00bfff", "Heizstäbe": "#d62728",
        "Gas-Kessel": "#ff7f0e", "Öl-Kessel": "#222222",
        "Biomasse-Kessel": "#2ca02c", "Sonstige Kessel": "#7f7f7f"
    }

    ax1 = axes[0]
    elec_cols = [c for c in ["Groß-WP", "Dezentrale WP", "Heizstäbe"] if c in p_elec.columns]
    if not p_elec.empty and elec_cols and p_elec[elec_cols].abs().sum().sum() > 1e-6:
        ax1.stackplot(p_elec.index, *[p_elec[c] for c in elec_cols],
                      labels=elec_cols,
                      colors=[colors.get(c, "#000") for c in elec_cols], alpha=0.9)
        ax1.legend(loc="upper left", fontsize=font_sizes['legend'], frameon=True)
    else:
        ax1.text(0.5, 0.5, "Kein relevanter Stromverbrauch",
                 ha='center', va='center', transform=ax1.transAxes)
    ax1.set_ylabel("Stromverbrauch [GW]", fontsize=font_sizes['label'])
    ax1.set_title(f"Stromverbrauch für Wärme ({country} {target_year})", fontsize=font_sizes['title'])
    ax1.grid(True, alpha=0.4, linestyle='--')

    ax2 = axes[1]
    plot_cols = [c for c in q_heat.columns if q_heat[c].abs().sum() > 0.01]
    if plot_cols:
        priority = ["Groß-WP (Fernw.)", "Dezentrale WP", "Biomasse-Kessel", "Heizstäbe", "Gas-Kessel", "Öl-Kessel"]
        plot_cols.sort(key=lambda x: priority.index(x) if x in priority else 99)
        ax2.stackplot(q_heat.index, *[q_heat[c] for c in plot_cols],
                      labels=plot_cols,
                      colors=[colors.get(c, "#777") for c in plot_cols], alpha=0.9)
        ax2.legend(loc="upper left", fontsize=font_sizes['legend'], ncol=2, frameon=True)
    else:
        ax2.text(0.5, 0.5, "Keine Wärmeerzeugung gefunden",
                 ha='center', va='center', transform=ax2.transAxes)
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

            # Snapshot-Index auflösen (DatetimeIndex vs. MultiIndex)
            ts_index, is_multi = _resolve_snapshots(n)
            print(f"  Snapshot-Typ: {'MultiIndex (perfect-foresight)' if is_multi else 'DatetimeIndex (myopic)'}")
            print(f"  links_t.p0: {len(n.links_t.p0.columns)} Spalten | "
                  f"links_t.p1: {len(n.links_t.p1.columns)} Spalten")

            # Zeitraum-Maske einmal berechnen
            mask = _get_valid_mask(ts_index, START_DATE, END_DATE)
            n_steps = mask.sum()
            print(f"  Zeitschritte im Fenster ({START_DATE} bis {END_DATE}): {n_steps}")

            if n_steps == 0:
                print(f"  ⚠️ Keine Zeitschritte im Fenster — überspringe {path}")
                continue

            # WP-Diagnose
            hp_all = n.links[n.links.carrier.str.contains("heat pump", case=False)]
            print(f"  Alle WP-Links: {len(hp_all)}")
            if not hp_all.empty:
                ex = hp_all.index[0]
                print(f"  Beispiel: '{ex}' | bus0='{n.links.at[ex,'bus0']}' | "
                      f"p0: {ex in n.links_t.p0.columns} | p1: {ex in n.links_t.p1.columns}")

            if int(target_year) in [2030, 2040, 2050]:
                try:
                    caps = get_installed_heat_capacities(n, "DE")
                    print(f"  --- Kapazitäten DE {target_year} ---")
                    for k, v in caps.items():
                        print(f"    {k:15s}: {v:.2f} GW")
                except Exception as e:
                    print(f"❌ Kapazitäten: {e}")

            for country in countries:
                if country == "ALL":
                    continue
                try:
                    p_elec, q_heat = analyze_heat_sector(n, country, ts_index, mask)
                    if country == "DE":
                        summarize_heat_energy(q_heat, country)
                    plot_diagnosis(p_elec, q_heat, country, target_year, scenario_name, save_dir)
                except Exception as e:
                    print(f"Fehler {country}: {e}")
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    main()
