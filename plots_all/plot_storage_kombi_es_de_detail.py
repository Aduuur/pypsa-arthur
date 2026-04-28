#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse der Speicher (Batterie DE & Wasser FR) - WINTER-DETAIL MIT VERBESSERTER ANALYSE
=======================================================================================

Erstellt einen kombinierten Plot für den Analyse-Zeitraum aus der Config:
- Oben: Batteriespeicher in BATTERY_COUNTRY (default: DE)
- Unten: Wasserspeicher in HYDRO_COUNTRY (default: FR)

Plus: Detaillierte Konsolen-Analyse der Batteriezyklen.
Zeitraum, Länder und Simulationsjahre kommen aus PlottingConfig.
"""

import os
import re
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config_final import PlottingConfig, fill_leap_day

# ------------------------------------------------------------
# Länder-Konfiguration (einzige "hardcoded" Werte — fachlich begründet)
# ------------------------------------------------------------
BATTERY_COUNTRY = "DE"
HYDRO_COUNTRY   = "FR"


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def get_battery_data_detailed(n, country):
    """
    Holt detaillierte Batterie-Daten mit Effizienz-Korrektur.
    Returns: charge_p0, discharge_net, soc_norm, e_nom,
             charge_eff, discharge_eff, n_batteries, soc_abs
    """
    charge_links = n.links.index[
        n.links.carrier.str.contains("charger", case=False) &
        n.links.bus0.str.startswith(country)
    ]
    charge_p0  = (n.links_t.p0[charge_links].sum(axis=1)
                  if not charge_links.empty
                  else pd.Series(0.0, index=n.snapshots))
    charge_eff = (n.links.loc[charge_links, "efficiency"].mean()
                  if not charge_links.empty else 0.95)

    discharge_links = n.links.index[
        n.links.carrier.str.contains("discharger", case=False) &
        n.links.bus0.str.startswith(country)
    ]
    discharge_p0   = (n.links_t.p0[discharge_links].sum(axis=1)
                      if not discharge_links.empty
                      else pd.Series(0.0, index=n.snapshots))
    discharge_eff  = (n.links.loc[discharge_links, "efficiency"].mean()
                      if not discharge_links.empty else 0.95)
    discharge_net  = discharge_p0 * discharge_eff

    stores  = n.stores.index[
        n.stores.carrier.str.contains("battery", case=False) &
        n.stores.bus.str.startswith(country)
    ]
    soc_abs = (n.stores_t.e[stores].sum(axis=1)
               if not stores.empty
               else pd.Series(0.0, index=n.snapshots))
    e_nom   = n.stores.loc[stores, "e_nom_opt"].sum() if not stores.empty else 0.0
    soc_norm = (soc_abs / e_nom
                if e_nom > 1e-3
                else pd.Series(0.0, index=soc_abs.index))
    n_batteries = len(stores)

    return charge_p0, discharge_net, soc_norm, e_nom, charge_eff, discharge_eff, n_batteries, soc_abs


def get_hydro_data(n, country):
    """Holt Wasserspeicher-Daten (Pumpen, Turbinieren, SOC) für ein Land."""
    gen  = pd.Series(0.0, index=n.snapshots)
    pump = pd.Series(0.0, index=n.snapshots)
    soc  = pd.Series(0.0, index=n.snapshots)
    e_nom = 0.0

    # a) StorageUnits (PHS + Hydro)
    su_idx = n.storage_units.index[
        n.storage_units.carrier.isin(["PHS", "hydro"]) &
        n.storage_units.bus.str.startswith(country)
    ]
    if not su_idx.empty:
        gen  += n.storage_units_t.p_dispatch[su_idx].clip(lower=0).sum(axis=1)
        if "p_store" in n.storage_units_t:
            pump += n.storage_units_t.p_store[su_idx].sum(axis=1).clip(lower=0)
        else:
            pump += -n.storage_units_t.p_dispatch[su_idx].clip(upper=0).sum(axis=1)
        soc  += n.storage_units_t.state_of_charge[su_idx].sum(axis=1)
        if "e_nom_opt" in n.storage_units.columns:
            e_nom += n.storage_units.loc[su_idx, "e_nom_opt"].sum()
        else:
            e_nom += (n.storage_units.loc[su_idx, "p_nom"] *
                      n.storage_units.loc[su_idx, "max_hours"]).sum()

    # b) Links (Pumpen für PHS)
    if not su_idx.empty:
        phs_buses  = n.storage_units.loc[su_idx, "bus"].unique()
        pump_links = n.links.index[
            n.links.bus1.isin(phs_buses) &
            n.links.carrier.str.contains("PHS", case=False)
        ]
        if not pump_links.empty:
            pump += n.links_t.p1[pump_links].sum(axis=1).clip(lower=0)

    # c) Stores (Hydro Reservoirs)
    store_idx = n.stores.index[
        n.stores.carrier.isin(["PHS", "hydro"]) &
        n.stores.bus.str.startswith(country)
    ]
    if not store_idx.empty:
        soc   += n.stores_t.e[store_idx].sum(axis=1)
        e_nom += n.stores.loc[store_idx, "e_nom_opt"].sum()

    soc_norm = (soc / e_nom if e_nom > 1e-3
                else pd.Series(0.0, index=soc.index))
    return pump, gen, soc_norm


def _slice_to_period(series, start, end):
    """Schneidet eine Series auf [start, end] zu."""
    return series[(series.index >= start) & (series.index <= end)]


def analyze_battery_cycles_improved(charge_gross, discharge_net, soc_norm, soc_abs,
                                     e_nom, charge_eff, discharge_eff, n_batteries,
                                     start_date, end_date):
    """
    Detaillierte Analyse der Batteriezyklen mit Effizienz-Korrektur.
    Gibt sauber auf, wenn der Zeitraum zu wenige Schritte enthält.
    """
    print("\n" + "=" * 80)
    print("🔋 DETAILLIERTE BATTERIE-ZYKLUS-ANALYSE")
    print("=" * 80)
    print(f"\n📊 GRUNDDATEN:")
    print(f"   Zeitraum:           {start_date} bis {end_date}")
    print(f"   Anzahl Zeitschritte: {len(charge_gross)}")

    if len(charge_gross) < 2:
        print("⚠️  Zu wenige Zeitschritte – Analyse übersprungen.")
        return

    timestep       = charge_gross.index[1] - charge_gross.index[0]
    timestep_hours = timestep.total_seconds() / 3600
    days           = max((charge_gross.index[-1] - charge_gross.index[0]).days, 1)

    print(f"   Zeitauflösung:       {timestep} ({timestep_hours:.1f} h)")
    print(f"   Installierte Kap.:   {e_nom / 1000:.1f} GWh ({e_nom:.0f} MWh)")
    print(f"   Batteriestandorte:   {n_batteries}")
    print(f"   Charger-Effizienz:   {charge_eff * 100:.1f}%")
    print(f"   Discharger-Effizienz:{discharge_eff * 100:.1f}%")

    charge_net = charge_gross * charge_eff

    # --- Leistungsstatistiken ---
    print(f"\n⚡ LEISTUNGSSTATISTIKEN:")
    for label, s in [("Laden (vom Netz)", charge_gross), ("Entladen (ins Netz)", discharge_net)]:
        active = s[s > 100]
        print(f"   {label}:")
        print(f"      Max:            {s.max():.1f} MW")
        print(f"      Mittel (aktiv): {active.mean():.1f} MW" if not active.empty else "      Mittel (aktiv): –")
        print(f"      P95:            {s.quantile(0.95):.1f} MW")
        pct = 100 * (s > 100).sum() / len(s)
        print(f"      Stunden >100 MW:{(s > 100).sum()} ({pct:.1f}%)")

    # --- Füllstand ---
    print(f"\n📈 FÜLLSTAND (STATE OF CHARGE):")
    print(f"   Min:      {soc_norm.min() * 100:.1f}%  ({soc_abs.min():.0f} MWh)")
    print(f"   Max:      {soc_norm.max() * 100:.1f}%  ({soc_abs.max():.0f} MWh)")
    print(f"   Mittel:   {soc_norm.mean() * 100:.1f}%  ({soc_abs.mean():.0f} MWh)")
    print(f"   Median:   {soc_norm.median() * 100:.1f}%  ({soc_abs.median():.0f} MWh)")
    print(f"   Std.Abw.: {soc_norm.std() * 100:.1f}%")

    # --- Zyklen ---
    soc_change  = soc_norm.diff().abs()
    full_cycles = soc_change.sum()
    print(f"\n🔄 ZYKLEN-AKTIVITÄT:")
    print(f"   Vollzyklus-Äquivalent:     {full_cycles:.1f}")
    print(f"   Zyklen pro Tag:            {full_cycles / days:.2f}")
    active_changes = soc_change[soc_change > 0]
    if not active_changes.empty:
        print(f"   Ø Zyklentiefe:             {active_changes.mean() * 100:.1f}%")

    # --- Energiebilanz ---
    energy_in     = (charge_gross * timestep_hours).sum() / 1000
    energy_stored = (charge_net   * timestep_hours).sum() / 1000
    energy_out    = (discharge_net * timestep_hours).sum() / 1000
    soc_delta_gwh = (soc_abs.iloc[-1] - soc_abs.iloc[0]) / 1000
    roundtrip_eff       = (energy_out / energy_in * 100) if energy_in > 0 else 0
    theoretical_roundtrip = charge_eff * discharge_eff * 100

    print(f"\n⚡ ENERGIEBILANZ:")
    print(f"   Vom Netz bezogen:        {energy_in:.1f} GWh")
    print(f"   In Batterie geladen:     {energy_stored:.1f} GWh (nach Ladeverlusten)")
    print(f"   Ins Netz entladen:       {energy_out:.1f} GWh (nach Entladeverlusten)")
    print(f"   Nettoänderung SOC:       {soc_delta_gwh:+.1f} GWh")
    print(f"   Gesamtverluste:          {energy_in - energy_out:.1f} GWh")
    print(f"   System-Roundtrip:        {roundtrip_eff:.1f}% (gemessen)")
    print(f"   Theoretischer Roundtrip: {theoretical_roundtrip:.1f}%")

    # --- Tägliche Muster ---
    charge_daily    = charge_gross.resample("D").sum() * timestep_hours / 1000
    discharge_daily = discharge_net.resample("D").sum() * timestep_hours / 1000
    print(f"\n📅 TÄGLICHE MUSTER:")
    print(f"   Laden  — Max: {charge_daily.max():.2f} GWh/Tag | Ø: {charge_daily.mean():.2f} GWh/Tag")
    print(f"   Entladen — Max: {discharge_daily.max():.2f} GWh/Tag | Ø: {discharge_daily.mean():.2f} GWh/Tag")

    # --- Netto-Betriebszustand ---
    threshold = 1000
    net_state = charge_gross - (discharge_net / max(discharge_eff, 1e-6))
    n_ch  = (net_state >  threshold).sum()
    n_dis = (net_state < -threshold).sum()
    n_bal = ((net_state >= -threshold) & (net_state <= threshold)).sum()
    total = len(net_state)
    print(f"\n🎯 NETTO-BETRIEBSZUSTAND (Schwelle ±{threshold} MW):")
    print(f"   Überwiegend Laden:    {n_ch}  ({100 * n_ch / total:.1f}%)")
    print(f"   Überwiegend Entladen: {n_dis} ({100 * n_dis / total:.1f}%)")
    print(f"   Ausgewogen/Gering:    {n_bal} ({100 * n_bal / total:.1f}%)")
    print(f"\n💡 {n_batteries} Standorte aggregiert — systemweite Tendenz: "
          f"{100 * n_ch / total:.0f}% Laden, {100 * n_dis / total:.0f}% Entladen.")

    # --- SOC-Grenzbereiche ---
    low_soc  = (soc_norm < 0.2).sum()
    high_soc = (soc_norm > 0.8).sum()
    print(f"\n⚠️  SOC-GRENZBEREICHE:")
    print(f"   Kritisch niedrig (<20%): {low_soc}  ({100 * low_soc / len(soc_norm):.1f}%)")
    print(f"   Fast voll        (>80%): {high_soc} ({100 * high_soc / len(soc_norm):.1f}%)")

    # --- Längste Perioden ---
    threshold_period = 5000
    print(f"\n⏱️  LÄNGSTE PERIODEN (Netto-Leistung > {threshold_period} MW):")
    for label, cond in [("Ladeperiode", net_state > threshold_period),
                         ("Entladeperiode", net_state < -threshold_period)]:
        arr = cond.astype(int)
        grp = arr.diff().ne(0).cumsum()
        lengths = arr.groupby(grp).sum()
        longest = lengths[lengths > 0].max() if (lengths > 0).any() else 0
        print(f"   Längste Netto-{label}: {longest} h ({longest / 24:.1f} Tage)")

    # --- Extremereignisse ---
    print(f"\n🌩️  EXTREMEREIGNISSE:")
    print(f"   Max. Ladeleistung:    {charge_gross.max():.0f} MW  am {charge_gross.idxmax()}")
    print(f"   Max. Entladeleistung: {discharge_net.max():.0f} MW  am {discharge_net.idxmax()}")
    print(f"   Niedrigster SoC:      {soc_norm.min() * 100:.1f}%  am {soc_norm.idxmin()}")
    print(f"   Höchster SoC:         {soc_norm.max() * 100:.1f}%  am {soc_norm.idxmax()}")
    print("\n" + "=" * 80 + "\n")


def plot_storage_detail(n, year, save_dir, start_date, end_date,
                         battery_country=BATTERY_COUNTRY,
                         hydro_country=HYDRO_COUNTRY):
    """
    Kombi-Plot: Batterien (battery_country) oben, Wasserkraft (hydro_country) unten.
    start_date / end_date kommen aus der Config bzw. werden dynamisch aus dem Netz abgeleitet.
    """
    # --- Rohdaten holen ---
    (de_charge, de_discharge, de_soc, de_e_nom,
     charge_eff, discharge_eff, n_batteries, de_soc_abs) = get_battery_data_detailed(n, battery_country)
    es_pump_full, es_gen_full, es_soc_full = get_hydro_data(n, hydro_country)

    # --- Zeitraum zuschneiden ---
    de_charge_period    = _slice_to_period(de_charge,    start_date, end_date)
    de_discharge_period = _slice_to_period(de_discharge, start_date, end_date)
    de_soc_period       = _slice_to_period(de_soc,       start_date, end_date)
    de_soc_abs_period   = _slice_to_period(de_soc_abs,   start_date, end_date)
    es_pump             = _slice_to_period(es_pump_full, start_date, end_date)
    es_gen              = _slice_to_period(es_gen_full,  start_date, end_date)
    es_soc              = _slice_to_period(es_soc_full,  start_date, end_date)

    # --- Fallback: wenn Zeitraum leer → gesamtes Netz nehmen ---
    if len(de_charge_period) == 0:
        print(f"⚠️  Keine Snapshots im Zeitraum {start_date} – {end_date}.")
        print(f"   Verfügbarer Bereich: {de_charge.index.min()} – {de_charge.index.max()}")
        print(f"   → Nutze gesamten verfügbaren Zeitraum.")
        de_charge_period    = de_charge
        de_discharge_period = de_discharge
        de_soc_period       = de_soc
        de_soc_abs_period   = de_soc_abs
        es_pump             = es_pump_full
        es_gen              = es_gen_full
        es_soc              = es_soc_full
        start_date = str(de_charge.index.min())
        end_date   = str(de_charge.index.max())

    # --- Batterie-Analyse Konsole ---
    analyze_battery_cycles_improved(
        de_charge_period, de_discharge_period, de_soc_period, de_soc_abs_period,
        de_e_nom, charge_eff, discharge_eff, n_batteries,
        start_date, end_date
    )

    # --- Resampling ---
    de_charge_period = fill_leap_day(de_charge_period.to_frame()).iloc[:,0]
    de_discharge_period = fill_leap_day(de_discharge_period.to_frame()).iloc[:,0]
    de_soc_period = fill_leap_day(de_soc_period.to_frame()).iloc[:,0]
    es_pump = fill_leap_day(es_pump.to_frame()).iloc[:,0]
    es_gen = fill_leap_day(es_gen.to_frame()).iloc[:,0]
    es_soc = fill_leap_day(es_soc.to_frame()).iloc[:,0]
    resample = "4h"
    de_charge_rs    = de_charge_period.resample(resample).mean()
    de_discharge_rs = de_discharge_period.resample(resample).mean()
    de_soc_rs       = de_soc_period.resample(resample).mean()
    es_pump_rs      = es_pump.resample(resample).mean()
    es_gen_rs       = es_gen.resample(resample).mean()
    es_soc_rs       = es_soc.resample(resample).mean()

    # --- Plot ---
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Titel zeigt tatsächlichen Zeitraum
    start_str = pd.Timestamp(start_date).strftime("%d.%m.%Y")
    end_str   = pd.Timestamp(end_date).strftime("%d.%m.%Y")
    fig.suptitle(
        f"Speicherbewirtschaftung {start_str} – {end_str}  |  Szenario {year}",
        fontsize=20, y=0.96
    )

    # -- Subplot 1: Batterie --
    ax1   = axes[0]
    ax1_r = ax1.twinx()
    l1, = ax1.plot(de_charge_rs.index,   de_charge_rs,   color="#ace37f", label="Laden (+)",    linewidth=2)
    l2, = ax1.plot(de_discharge_rs.index, -de_discharge_rs, color="#5f8c3f", label="Entladen (-)", linewidth=2)
    l3   = ax1_r.fill_between(de_soc_rs.index, 0, de_soc_rs, color="gray", alpha=0.25, label="Füllstand (SoC)")
    ax1.set_ylabel("Leistung [MW]", fontsize=14)
    ax1_r.set_ylabel("Füllstand [0–1]", fontsize=14)
    ax1_r.set_ylim(0, 1.1)
    ax1.set_title(f"Kurzzeitspeicherung: Batterien in {battery_country}", fontsize=16, pad=10)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]],
               loc="upper right", fontsize=12, framealpha=0.9)

    # -- Subplot 2: Wasserkraft --
    ax2   = axes[1]
    ax2_r = ax2.twinx()
    l4, = ax2.plot(es_pump_rs.index,  es_pump_rs,  color="#00ced1", label="Pumpen (+)",      linewidth=2)
    l5, = ax2.plot(es_gen_rs.index,  -es_gen_rs,   color="#007f82", label="Turbinieren (-)", linewidth=2)
    l6   = ax2_r.fill_between(es_soc_rs.index, 0, es_soc_rs, color="gray", alpha=0.25, label="Füllstand (SoC)")
    ax2.set_ylabel("Leistung [MW]", fontsize=14)
    ax2_r.set_ylabel("Füllstand [0–1]", fontsize=14)
    ax2_r.set_ylim(0, 1.1)
    ax2.set_title(f"Saisonale Speicherung: Wasserkraft (Reservoir & PHS) in {hydro_country}",
                  fontsize=16, pad=10)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.legend([l4, l5, l6], [l.get_label() for l in [l4, l5, l6]],
               loc="upper right", fontsize=12, framealpha=0.9)

    # X-Achse
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax2.set_xlabel("Datum", fontsize=14)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = f"storage_detail_{battery_country}_{hydro_country}_{year}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot gespeichert: {filepath}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    config   = PlottingConfig()
    networks = config.get_networks()

    # Netzwerk-Liste normalisieren
    if isinstance(networks, dict):
        if config.SCENARIO_SELECTION == "both":
            # bei 'both': erstes Szenario nehmen (average)
            networks = list(networks.values())[0]
        else:
            networks = networks[config.SCENARIO_SELECTION]

    if not networks:
        print("⚠️  Keine Netzwerke gefunden.")
        return

    # Zeitraum aus Config holen — dynamischer Fallback auf gesamtes Netz
    cfg_start = getattr(config, "START", None)
    cfg_end   = getattr(config, "END",   None)

    for path in networks:
        if not os.path.isfile(path):
            print(f"⚠️  Datei nicht gefunden: {path}")
            continue

        try:
            m = re.search(r"_(\d{4})\.nc$", path)
            year = int(m.group(1)) if m else 2050  # ARO: kein Jahr im Namen -> Planungsjahr

            n = pypsa.Network(path)
            n.snapshots = pd.to_datetime(n.snapshots)

            # Zeitraum bestimmen: Config-Wert wenn vorhanden, sonst gesamtes Netz
            start_date = str(cfg_start) if cfg_start else str(n.snapshots.min())
            end_date   = str(cfg_end)   if cfg_end   else str(n.snapshots.max())

            print(f"\n📂 Analysiere Speicher-Detail für {year}  ({start_date} – {end_date})")

            save_dir = os.path.join(
                config.BASE_SAVE_PATH,
                os.path.basename(os.path.dirname(os.path.dirname(path))),
                "plots_storage_analysis"
            )
            os.makedirs(save_dir, exist_ok=True)

            plot_storage_detail(n, year, save_dir, start_date, end_date)

        except Exception as e:
            print(f"❌ Fehler bei {path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
