#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose-Skript: Fuel-Switch im Wärmesektor
- Analysezeitraum wird aus config.DARK_SKY_START / DARK_SKY_END gelesen
- Unterstützt PyPSA perfect-foresight (MultiIndex snapshots) UND myopic (DatetimeIndex)
- Filterung über n.links.index (Link-Name enthält Länderkürzel)
- p0 wird mit .abs() gelesen: in PyPSA ist p0 des Input-Bus negativ (Entnahme aus Bus)
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


def _resolve_snapshots(n):
    snap = n.snapshots
    if isinstance(snap, pd.MultiIndex):
        if not n.links_t.p0.empty:
            return n.links_t.p0.index, True
        if not n.links_t.p1.empty:
            return n.links_t.p1.index, True
        return snap.get_level_values(1), True
    return snap, False


def _get_valid_mask(ts_index, start: str, end: str):
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)
    if isinstance(ts_index, pd.MultiIndex):
        inner = pd.DatetimeIndex(ts_index.get_level_values(-1))
    else:
        inner = pd.DatetimeIndex(ts_index)
    mask = (inner >= ts_start) & (inner <= ts_end)
    if not mask.any():
        print(f"  ⚠️  Kein Zeitraum {start}–{end} im Modell!")
        print(f"     Verfügbar: {inner[0]} – {inner[-1]}")
    return mask


def _links_for_country(n, country):
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
    in_p0 = sum(1 for l in links_idx if l in n.links_t.p0.columns)
    in_p1 = sum(1 for l in links_idx if l in n.links_t.p1.columns)
    print(f"  [{label}] {len(links_idx)} Links, {in_p0} in p0, {in_p1} in p1")


def _cop_check(n, links_idx, label, p0, p1, mask):
    """
    COP-Konsistenz für eine Gruppe von Links.
    Gibt effektiven COP = sum(|p1|) / sum(|p0|) aus.
    Zeigt außerdem: wird p1 überhaupt befüllt? Welcher bus1?
    """
    if len(links_idx) == 0:
        return
    print(f"\n  🔎 COP-Check [{label}] ({len(links_idx)} Links)")

    # Zeige bus0/bus1 für ersten Link
    ex = links_idx[0]
    bus0 = n.links.at[ex, "bus0"] if "bus0" in n.links.columns else "?"
    bus1 = n.links.at[ex, "bus1"] if "bus1" in n.links.columns else "?"
    eff  = n.links.at[ex, "efficiency"] if "efficiency" in n.links.columns else float("nan")
    print(f"     Beispiel-Link: '{ex}'")
    print(f"     bus0={bus0}  bus1={bus1}  efficiency={eff:.3f}")

    # Alle Links die auch in p0/p1 vorhanden sind
    in_p0 = [l for l in links_idx if l in p0.columns]
    in_p1 = [l for l in links_idx if l in p1.columns]
    print(f"     In links_t.p0: {len(in_p0)}/{len(links_idx)}")
    print(f"     In links_t.p1: {len(in_p1)}/{len(links_idx)}")

    if not in_p0:
        print("     ❌ Kein p0 verfügbar -> überspringe")
        return

    sum_p0 = p0[in_p0].loc[mask].abs().sum().sum()
    print(f"     Sum |p0| im Fenster: {sum_p0/1000:.2f} GW*h")

    if in_p1:
        sum_p1 = p1[in_p1].loc[mask].abs().sum().sum()
        print(f"     Sum |p1| im Fenster: {sum_p1/1000:.2f} GW*h")
        if sum_p0 > 0:
            eff_cop = sum_p1 / sum_p0
            print(f"     => Effektiver COP (p1/p0): {eff_cop:.3f}")
            if eff_cop < 0.5:
                print(f"     ⚠️  COP < 0.5 -> p1 wird möglicherweise am falschen Bus gelesen!")
            elif eff_cop > 5.0:
                print(f"     ⚠️  COP > 5.0 -> Einheitenproblem oder falscher Bus?")
            else:
                print(f"     ✅  COP plausibel")
    else:
        print("     ❌ Kein p1 verfügbar -> Wärmeerzeugung wird 0 sein!")

    # Variabilitäts-Check für p0
    ts_p0 = p0[in_p0].loc[mask].abs().sum(axis=1)
    std_p0  = ts_p0.std()
    mean_p0 = ts_p0.mean()
    cv = std_p0 / mean_p0 if mean_p0 > 0 else 0
    print(f"     p0 Variationskoeffizient CV = {cv:.3f}  (std={std_p0/1000:.3f} GW, mean={mean_p0/1000:.3f} GW)")
    if cv < 0.02:
        print(f"     ⚠️  CV sehr klein -> WP läuft konstant auf Nennleistung (kein echter Dispatch?)")
    else:
        print(f"     ✅  WP zeigt sinnvolle Variabilität")


def analyze_heat_sector(n, country, ts_index, mask):
    print(f"  🔍 {country}")
    links = _links_for_country(n, country)

    large_hp = links[
        links.carrier.str.contains("heat pump", case=False) &
        links.carrier.str.contains("central", case=False)
    ].index
    small_hp = links[
        links.carrier.str.contains("heat pump", case=False) &
        ~links.carrier.str.contains("central", case=False)
    ].index
    res = links[links.carrier.str.contains("resistive heater", case=False)].index
    boilers = links[links.carrier.str.contains("boiler", case=False)].index
    if len(boilers) == 0:
        boilers = n.links[
            n.links.bus1.str.startswith(country) &
            n.links.carrier.str.contains("boiler", case=False)
        ].index

    _debug_links_t(n, large_hp, "Groß-WP")
    _debug_links_t(n, small_hp, "Dez-WP")
    _debug_links_t(n, res,      "Heizstab")
    _debug_links_t(n, boilers,  "Boiler")

    p0 = n.links_t.p0
    p1 = n.links_t.p1

    # COP + Variabilitäts-Check nur für DE (weniger Output-Spam)
    if country == "DE":
        _cop_check(n, large_hp, "Groß-WP",  p0, p1, mask)
        _cop_check(n, small_hp, "Dez-WP",    p0, p1, mask)
        _cop_check(n, res,      "Heizstab",  p0, p1, mask)

    # Plot-Index: immer DatetimeIndex
    if isinstance(ts_index, pd.MultiIndex):
        plot_idx = pd.DatetimeIndex(ts_index.get_level_values(-1)[mask])
    else:
        plot_idx = pd.DatetimeIndex(ts_index[mask])

    def get_series(df, idx_list):
        avail = [l for l in idx_list if l in df.columns]
        if not avail:
            return pd.Series(0.0, index=plot_idx)
        sub = df[avail].loc[mask].sum(axis=1) / 1000  # MW -> GW
        sub.index = plot_idx
        return sub

    p_elec = pd.DataFrame(index=plot_idx)
    p_elec["Groß-WP"]      = get_series(p0, large_hp).abs()
    p_elec["Dezentrale WP"] = get_series(p0, small_hp).abs()
    p_elec["Heizstäbe"]    = get_series(p0, res).abs()

    print(f"    Groß-WP Ø {p_elec['Groß-WP'].mean():.3f} GW | "
          f"Dez-WP Ø {p_elec['Dezentrale WP'].mean():.3f} GW | "
          f"Heizstab Ø {p_elec['Heizstäbe'].mean():.3f} GW")

    q_heat = pd.DataFrame(index=plot_idx)
    if len(large_hp) > 0:
        q_heat["Groß-WP (Fernw.)"] = get_series(p1, large_hp).abs()
    if len(small_hp) > 0:
        q_heat["Dezentrale WP"]     = get_series(p1, small_hp).abs()
    if len(res) > 0:
        q_heat["Heizstäbe"]         = get_series(p1, res).abs()

    for link in boilers:
        carrier = n.links.at[link, "carrier"]
        if "gas"       in carrier: label = "Gas-Kessel"
        elif "oil"     in carrier: label = "Öl-Kessel"
        elif "biomass" in carrier: label = "Biomasse-Kessel"
        else:                      label = "Sonstige Kessel"
        if link not in p1.columns:
            continue
        val = p1[[link]].loc[mask].iloc[:, 0].abs() / 1000
        val.index = plot_idx
        q_heat[label] = q_heat[label] + val if label in q_heat.columns else val

    return p_elec, q_heat


def summarize_heat_energy(q_heat, country):
    print(f"\n  🔥 Wärme {country}:")
    for col in ["Groß-WP (Fernw.)", "Dezentrale WP", "Heizstäbe",
                "Gas-Kessel", "Öl-Kessel", "Biomasse-Kessel", "Sonstige Kessel"]:
        if col in q_heat.columns:
            print(f"    {col:20s}: {q_heat[col].sum()/1000:6.2f} TWh | Peak: {q_heat[col].max():6.1f} GW")
        else:
            print(f"    {col:20s}:   0.00 TWh | Peak:    0.0 GW")


def get_installed_heat_capacities(n, country):
    links = _links_for_country(n, country)

    def cap(mask):
        sub = links[mask]
        if sub.empty: return 0.0
        col = "p_nom_opt" if "p_nom_opt" in sub.columns else "p_nom"
        return sub[col].sum() / 1000

    caps = {
        "Groß-WP":       cap(links.carrier.str.contains("heat pump", case=False) & links.carrier.str.contains("central", case=False)),
        "Dezentrale WP":  cap(links.carrier.str.contains("heat pump", case=False) & ~links.carrier.str.contains("central", case=False)),
        "Heizstäbe":     cap(links.carrier.str.contains("resistive heater", case=False)),
        "Gas-Kessel":    cap(links.carrier.str.contains("gas boiler",      case=False)),
        "Öl-Kessel":     cap(links.carrier.str.contains("oil boiler",       case=False)),
        "Biomasse":      cap(links.carrier.str.contains("biomass boiler",  case=False)),
    }
    caps["TOTAL"] = sum(caps.values())
    return caps


def plot_diagnosis(p_elec, q_heat, country, target_year, scenario_name, save_dir,
                  start_date, end_date):
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
    ax1.set_title(f"Stromverbrauch für Wärme ({country} {target_year}, {start_date}–{end_date})",
                  fontsize=font_sizes['title'])
    ax1.grid(True, alpha=0.4, linestyle='--')

    ax2 = axes[1]
    plot_cols = [c for c in q_heat.columns if q_heat[c].abs().sum() > 0.01]
    if plot_cols:
        priority = ["Groß-WP (Fernw.)", "Dezentrale WP", "Biomasse-Kessel",
                    "Heizstäbe", "Gas-Kessel", "Öl-Kessel"]
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
    print(f"  ✅ {out_path}")
    plt.close()


def main():
    config = PlottingConfig()
    networks  = config.get_networks()
    countries = config.get_countries()

    start_date = config.DARK_SKY_START
    end_date   = config.DARK_SKY_END
    print(f"📅 Analysezeitraum: {start_date} bis {end_date}")

    save_dir = os.path.join(config.BASE_SAVE_PATH, "heat_diagnosis")
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Plots → {save_dir}")

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

            print(f"\n📂 {path}")
            n = pypsa.Network(path)

            ts_index, is_multi = _resolve_snapshots(n)
            print(f"  Snapshots: {'MultiIndex (perfect-foresight)' if is_multi else 'DatetimeIndex (myopic)'}")
            print(f"  links_t.p0: {len(n.links_t.p0.columns)} | links_t.p1: {len(n.links_t.p1.columns)}")

            mask = _get_valid_mask(ts_index, start_date, end_date)
            n_steps = int(mask.sum())
            print(f"  Zeitschritte im Fenster: {n_steps}")
            if n_steps == 0:
                print(f"  ⚠️  Kein Zeitfenster → überspringe")
                continue

            hp_all = n.links[n.links.carrier.str.contains("heat pump", case=False)]
            if not hp_all.empty:
                ex = hp_all.index[0]
                print(f"  Beispiel WP: '{ex}' | p0: {ex in n.links_t.p0.columns}")

            if int(target_year) in [2030, 2040, 2050]:
                try:
                    caps = get_installed_heat_capacities(n, "DE")
                    print(f"  Kapazitäten DE {target_year}:")
                    for k, v in caps.items():
                        print(f"    {k:15s}: {v:.2f} GW")
                except Exception as e:
                    print(f"  ❌ Kapazitäten: {e}")

            for country in countries:
                if country == "ALL":
                    continue
                try:
                    p_elec, q_heat = analyze_heat_sector(n, country, ts_index, mask)
                    if country == "DE":
                        summarize_heat_energy(q_heat, country)
                    plot_diagnosis(p_elec, q_heat, country, target_year,
                                   scenario_name, save_dir, start_date, end_date)
                except Exception as e:
                    print(f"  Fehler {country}: {e}")
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    main()
