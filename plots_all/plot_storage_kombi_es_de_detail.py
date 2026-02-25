#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse der Speicher (Batterie DE & Wasser FR) - WINTER-DETAIL MIT VERBFRSERTER ANALYSE
=======================================================================================

Erstellt einen kombinierten Plot für Jan - 15. Feb:
- Oben: Batteriespeicher in DE (Kurzzeitspeicher)
- Unten: Wasserspeicher in FR (Saisonaler Speicher)

Plus: Korrigierte und detaillierte Konsolen-Analyse der Batteriezyklen
"""

import os
import re
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config_final import PlottingConfig


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def get_battery_data_detailed(n, country):
    """
    Holt detaillierte Batterie-Daten mit Effizienz-Korrektur.
    Returns: charge, discharge, soc_norm, e_nom, charge_efficiency, discharge_efficiency
    """
    # Ladeleistung (vom Netz in Batterie) - p0 ist Leistung am bus0 (Netz)
    charge_links = n.links.index[
        n.links.carrier.str.contains("charger", case=False) & n.links.bus0.str.startswith(country)]

    charge_p0 = n.links_t.p0[charge_links].sum(axis=1) if not charge_links.empty else pd.Series(0, index=n.snapshots)

    # Effizienz der Charger (typisch ~0.95)
    charge_eff = n.links.loc[charge_links, "efficiency"].mean() if not charge_links.empty else 0.95

    # Tatsächlich in Batterie gespeicherte Energie (nach Verlusten)
    charge_net = charge_p0 * charge_eff

    # Entladeleistung (von Batterie ins Netz) - p0 ist Leistung am bus0 (Batterie)
    discharge_links = n.links.index[
        n.links.carrier.str.contains("discharger", case=False) & n.links.bus0.str.startswith(country)]

    discharge_p0 = n.links_t.p0[discharge_links].sum(axis=1) if not discharge_links.empty else pd.Series(0,
                                                                                                         index=n.snapshots)

    # Effizienz der Discharger
    discharge_eff = n.links.loc[discharge_links, "efficiency"].mean() if not discharge_links.empty else 0.95

    # Tatsächlich ans Netz abgegebene Energie (nach Verlusten)
    discharge_net = discharge_p0 * discharge_eff

    # Füllstand
    stores = n.stores.index[
        n.stores.carrier.str.contains("battery", case=False) & n.stores.bus.str.startswith(country)]
    soc = n.stores_t.e[stores].sum(axis=1) if not stores.empty else pd.Series(0, index=n.snapshots)
    e_nom = n.stores.loc[stores, "e_nom_opt"].sum()

    soc_norm = soc / e_nom if e_nom > 1e-3 else pd.Series(0, index=soc.index)

    # Zusätzlich: Anzahl der Batterie-Standorte
    n_batteries = len(stores)

    return charge_p0, discharge_net, soc_norm, e_nom, charge_eff, discharge_eff, n_batteries, soc


def get_hydro_data(n, country):
    """Holt Wasserspeicher-Daten (Pumpen, Turbinieren, SOC) für ein Land."""
    gen = pd.Series(0.0, index=n.snapshots)
    pump = pd.Series(0.0, index=n.snapshots)
    soc = pd.Series(0.0, index=n.snapshots)
    e_nom = 0.0

    # a) StorageUnits (PHS + Hydro)
    su_idx = n.storage_units.index[
        n.storage_units.carrier.isin(["PHS", "hydro"]) & n.storage_units.bus.str.startswith(country)]

    if not su_idx.empty:
        gen += n.storage_units_t.p_dispatch[su_idx].clip(lower=0).sum(axis=1)
        if 'p_store' in n.storage_units_t:
            pump += n.storage_units_t.p_store[su_idx].sum(axis=1).clip(lower=0)
        else:
            pump += -n.storage_units_t.p_dispatch[su_idx].clip(upper=0).sum(axis=1)

        soc += n.storage_units_t.state_of_charge[su_idx].sum(axis=1)

        if "e_nom_opt" in n.storage_units.columns:
            e_nom += n.storage_units.loc[su_idx, "e_nom_opt"].sum()
        else:
            e_nom += (n.storage_units.loc[su_idx, "p_nom"] * n.storage_units.loc[su_idx, "max_hours"]).sum()

    # b) Links (Pumpen für PHS)
    if not su_idx.empty:
        phs_buses = n.storage_units.loc[su_idx, "bus"].unique()
        pump_links = n.links.index[
            n.links.bus1.isin(phs_buses) &
            n.links.carrier.str.contains("PHS", case=False)
            ]
        if not pump_links.empty:
            pump += n.links_t.p1[pump_links].sum(axis=1).clip(lower=0)

    # c) Stores (Hydro Reservoirs)
    store_idx = n.stores.index[
        n.stores.carrier.isin(["PHS", "hydro"]) & n.stores.bus.str.startswith(country)]

    if not store_idx.empty:
        soc += n.stores_t.e[store_idx].sum(axis=1)
        e_nom += n.stores.loc[store_idx, "e_nom_opt"].sum()

    soc_norm = soc / e_nom if e_nom > 1e-3 else pd.Series(0, index=soc.index)

    return pump, gen, soc_norm


def analyze_battery_cycles_improved(charge_gross, discharge_net, soc_norm, soc_abs, e_nom,
                                    charge_eff, discharge_eff, n_batteries,
                                    start_date, end_date):
    """
    Verbesserte detaillierte Analyse der Batteriezyklen mit korrekter Effizienz-Berechnung.
    """
    print("\n" + "=" * 80)
    print("🔋 DETAILLIERTE BATTERIE-ZYKLUS-ANALYSE (VERBFRSERT)")
    print("=" * 80)

    # Grundstatistiken
    print(f"\n📊 GRUNDDATEN:")
    print(f"   Zeitraum: {start_date} bis {end_date}")
    print(f"   Anzahl Zeitschritte: {len(charge_gross)}")
    timestep = (charge_gross.index[1] - charge_gross.index[0])
    timestep_hours = timestep.total_seconds() / 3600
    print(f"   Zeitauflösung: {timestep} ({timestep_hours:.1f}h)")
    print(f"   Installierte Kapazität: {e_nom / 1000:.1f} GWh ({e_nom:.0f} MWh)")
    print(f"   Anzahl Batteriestandorte: {n_batteries}")
    print(f"   Mittlere Charger-Effizienz: {charge_eff * 100:.1f}%")
    print(f"   Mittlere Discharger-Effizienz: {discharge_eff * 100:.1f}%")

    # Berechne Netto-Energie (in/aus Batterie)
    charge_net = charge_gross * charge_eff  # Was tatsächlich in Batterie gespeichert wird

    # Leistungsstatistiken
    print(f"\n⚡ LEISTUNGSSTATISTIKEN:")
    print(f"   Laden (vom Netz):")
    print(f"      Max:    {charge_gross.max():.1f} MW")
    print(f"      Mittel: {charge_gross[charge_gross > 100].mean():.1f} MW (wenn aktiv)")
    print(f"      P95:    {charge_gross.quantile(0.95):.1f} MW")
    print(
        f"      Stunden >100 MW: {(charge_gross > 100).sum()} ({100 * (charge_gross > 100).sum() / len(charge_gross):.1f}%)")

    print(f"   Entladen (ins Netz, nach Verlusten):")
    print(f"      Max:    {discharge_net.max():.1f} MW")
    print(f"      Mittel: {discharge_net[discharge_net > 100].mean():.1f} MW (wenn aktiv)")
    print(f"      P95:    {discharge_net.quantile(0.95):.1f} MW")
    print(
        f"      Stunden >100 MW: {(discharge_net > 100).sum()} ({100 * (discharge_net > 100).sum() / len(discharge_net):.1f}%)")

    # Füllstand-Analyse
    print(f"\n📈 FÜLLSTAND (STATE OF CHARGE):")
    print(f"   Min:    {soc_norm.min() * 100:.1f}% ({soc_abs.min():.0f} MWh)")
    print(f"   Max:    {soc_norm.max() * 100:.1f}% ({soc_abs.max():.0f} MWh)")
    print(f"   Mittel: {soc_norm.mean() * 100:.1f}% ({soc_abs.mean():.0f} MWh)")
    print(f"   Median: {soc_norm.median() * 100:.1f}% ({soc_abs.median():.0f} MWh)")
    print(f"   Std.Abw.: {soc_norm.std() * 100:.1f}%")

    # Vollzyklen aus SOC-Änderungen
    soc_change = soc_norm.diff().abs()
    full_cycles = soc_change.sum()
    days = (charge_gross.index[-1] - charge_gross.index[0]).days
    print(f"\n🔄 ZYKLEN-AKTIVITÄT:")
    print(f"   Vollzyklus-Äquivalent: {full_cycles:.1f}")
    print(f"   Zyklen pro Tag: {full_cycles / days:.2f}")
    print(f"   Durchschnittliche Zyklentiefe: {soc_change[soc_change > 0].mean() * 100:.1f}%")

    # KORRIGIERTE Energiebilanz
    energy_in = (charge_gross * timestep_hours).sum() / 1000  # GWh vom Netz
    energy_stored = (charge_net * timestep_hours).sum() / 1000  # GWh in Batterie
    energy_out = (discharge_net * timestep_hours).sum() / 1000  # GWh ins Netz

    # Berechne aus tatsächlicher SOC-Änderung
    soc_delta = soc_abs.iloc[-1] - soc_abs.iloc[0]
    soc_delta_gwh = soc_delta / 1000

    # Roundtrip berechnet sich: Energie raus / Energie rein
    # ABER: Wir müssen die gespeicherte Energie nutzen
    roundtrip_eff = (energy_out / energy_in * 100) if energy_in > 0 else 0

    # Theoretische Roundtrip-Effizienz
    theoretical_roundtrip = charge_eff * discharge_eff * 100

    print(f"\n⚡ ENERGIEBILANZ:")
    print(f"   Vom Netz bezogen:     {energy_in:.1f} GWh")
    print(f"   In Batterie geladen:  {energy_stored:.1f} GWh (nach Ladeverlusten)")
    print(f"   Ins Netz entladen:    {energy_out:.1f} GWh (nach Entladeverlusten)")
    print(f"   Nettoänderung SOC:    {soc_delta_gwh:+.1f} GWh")
    print(f"   Gesamtverluste:       {energy_in - energy_out:.1f} GWh")
    print(f"   System-Roundtrip:     {roundtrip_eff:.1f}% (gemessen)")
    print(f"   Theoretischer Roundtrip: {theoretical_roundtrip:.1f}% (Link-Effizienzen)")

    # Tägliche Muster
    print(f"\n📅 TÄGLICHE MUSTER:")
    charge_daily = charge_gross.resample('D').sum() * timestep_hours / 1000
    discharge_daily = discharge_net.resample('D').sum() * timestep_hours / 1000

    print(f"   Laden pro Tag:")
    print(f"      Max:    {charge_daily.max():.2f} GWh/Tag")
    print(f"      Mittel: {charge_daily.mean():.2f} GWh/Tag")
    print(f"   Entladen pro Tag:")
    print(f"      Max:    {discharge_daily.max():.2f} GWh/Tag")
    print(f"      Mittel: {discharge_daily.mean():.2f} GWh/Tag")

    # Netto-Zustand (überwiegend laden oder entladen?)
    threshold = 1000  # MW
    net_state = charge_gross - (discharge_net / discharge_eff)  # Brutto-Vergleich

    mainly_charging = (net_state > threshold).sum()
    mainly_discharging = (net_state < -threshold).sum()
    balanced = ((net_state >= -threshold) & (net_state <= threshold)).sum()

    print(f"\n🎯 NETTO-BETRIEBSZUSTAND (Schwelle ±{threshold} MW):")
    print(f"   Überwiegend Laden:     {mainly_charging} h ({100 * mainly_charging / len(net_state):.1f}%)")
    print(f"   Überwiegend Entladen:  {mainly_discharging} h ({100 * mainly_discharging / len(net_state):.1f}%)")
    print(f"   Ausgewogen/Gering:     {balanced} h ({100 * balanced / len(net_state):.1f}%)")

    print(f"\n💡 INTERPRETATION:")
    print(f"   Da {n_batteries} Batteriestandorte aggregiert sind, können gleichzeitig")
    print(f"   verschiedene Standorte laden bzw. entladen. Der Netto-Zustand zeigt die")
    print(f"   systemweite Tendenz: {100 * mainly_charging / len(net_state):.0f}% der Zeit überwiegt Laden,")
    print(f"   {100 * mainly_discharging / len(net_state):.0f}% überwiegt Entladen.")

    # SOC-Grenzbereiche
    low_soc = (soc_norm < 0.2).sum()
    high_soc = (soc_norm > 0.8).sum()
    print(f"\n⚠️  SOC-GRENZBEREICHE:")
    print(f"   Kritisch niedrig (<20%): {low_soc} h ({100 * low_soc / len(soc_norm):.1f}%)")
    print(f"   Fast voll (>80%): {high_soc} h ({100 * high_soc / len(soc_norm):.1f}%)")

    # Längste Perioden (mit höherem Threshold)
    threshold_period = 5000  # MW - sinnvoller für aggregierte Betrachtung
    print(f"\n⏱️  LÄNGSTE PERIODEN (Netto-Leistung >{threshold_period} MW):")

    # Netto-Laden
    net_charging = (net_state > threshold_period).astype(int)
    charge_groups = net_charging.diff().ne(0).cumsum()
    charge_lengths = net_charging.groupby(charge_groups).sum()
    longest_charge = charge_lengths[charge_lengths > 0].max() if len(charge_lengths[charge_lengths > 0]) > 0 else 0
    print(f"   Längste Netto-Ladeperiode: {longest_charge} h ({longest_charge / 24:.1f} Tage)")

    # Netto-Entladen
    net_discharging = (net_state < -threshold_period).astype(int)
    discharge_groups = net_discharging.diff().ne(0).cumsum()
    discharge_lengths = net_discharging.groupby(discharge_groups).sum()
    longest_discharge = discharge_lengths[discharge_lengths > 0].max() if len(
        discharge_lengths[discharge_lengths > 0]) > 0 else 0
    print(f"   Längste Netto-Entladeperiode: {longest_discharge} h ({longest_discharge / 24:.1f} Tage)")

    # Extremereignis-Analyse
    print(f"\n🌩️  EXTREMEREIGNISSE:")
    max_charge_idx = charge_gross.idxmax()
    max_discharge_idx = discharge_net.idxmax()
    min_soc_idx = soc_norm.idxmin()
    max_soc_idx = soc_norm.idxmax()

    print(f"   Maximale Ladeleistung:    {charge_gross.max():.0f} MW am {max_charge_idx}")
    print(f"   Maximale Entladeleistung: {discharge_net.max():.0f} MW am {max_discharge_idx}")
    print(f"   Niedrigster Füllstand:    {soc_norm.min() * 100:.1f}% am {min_soc_idx}")
    print(f"   Höchster Füllstand:       {soc_norm.max() * 100:.1f}% am {max_soc_idx}")

    print("\n" + "=" * 80 + "\n")


def plot_winter_detail(n, year, save_dir):
    """
    Erstellt den Kombi-Plot für Jan - 15. Feb:
    Oben: DE Batterien
    Unten: FR Wasserkraft
    """

    # Daten holen mit verbesserter Funktion
    (de_charge, de_discharge, de_soc, de_e_nom,
     charge_eff, discharge_eff, n_batteries, de_soc_abs) = get_battery_data_detailed(n, "DE")

    es_pump, es_gen, es_soc = get_hydro_data(n, "FR")

    # WICHTIG: Wetterjahr 2005, nicht Simulationsjahr!
    weather_year = 2005
    start_date = f"{weather_year}-01-01"
    end_date = f"{weather_year}-01-31 23:59:59"

    mask = (de_charge.index >= start_date) & (de_charge.index <= end_date)

    de_charge_period = de_charge[mask]
    de_discharge_period = de_discharge[mask]
    de_soc_period = de_soc[mask]
    de_soc_abs_period = de_soc_abs[mask]

    es_pump = es_pump[mask]
    es_gen = es_gen[mask]
    es_soc = es_soc[mask]

    # VERBFRSERTE Batterie-Analyse
    analyze_battery_cycles_improved(
        de_charge_period, de_discharge_period, de_soc_period, de_soc_abs_period,
        de_e_nom, charge_eff, discharge_eff, n_batteries,
        start_date, end_date
    )

    # Resampling für Plot
    resample = "4h"

    de_charge_rs = de_charge_period.resample(resample).mean()
    de_discharge_rs = de_discharge_period.resample(resample).mean()
    de_soc_rs = de_soc_period.resample(resample).mean()

    es_pump_rs = es_pump.resample(resample).mean()
    es_gen_rs = es_gen.resample(resample).mean()
    es_soc_rs = es_soc.resample(resample).mean()

    # Plotting Settings
    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"Speicherbewirtschaftung im Winter (Januar, Szenario {year})",
                 fontsize=20, y=0.96)

    # --- PLOT 1: DEUTSCHLAND BATTERIE ---
    ax1 = axes[0]
    ax1_r = ax1.twinx()

    l1, = ax1.plot(de_charge_rs.index, de_charge_rs, color="#ace37f",
                   label="Laden (+)", linewidth=2)
    l2, = ax1.plot(de_discharge_rs.index, -de_discharge_rs, color="#5f8c3f",
                   label="Entladen (-)", linewidth=2)

    l3 = ax1_r.fill_between(de_soc_rs.index, 0, de_soc_rs, color="gray",
                            alpha=0.25, label="Füllstand (SoC)")

    ax1.set_ylabel("Leistung [MW]", fontsize=14)
    ax1_r.set_ylabel("Füllstand [0-1]", fontsize=14)
    ax1.set_title("Kurzzeitspeicherung: Batterien in Deutschland", fontsize=16, pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(0, color="black", linewidth=0.8)

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=12, framealpha=0.9)
    ax1_r.set_ylim(0, 1.1)

    # --- PLOT 2: SPANIEN WASSERKRAFT ---
    ax2 = axes[1]
    ax2_r = ax2.twinx()

    l4, = ax2.plot(es_pump_rs.index, es_pump_rs, color="#00ced1",
                   label="Pumpen (+)", linewidth=2)
    l5, = ax2.plot(es_gen_rs.index, -es_gen_rs, color="#007f82",
                   label="Turbinieren (-)", linewidth=2)

    l6 = ax2_r.fill_between(es_soc_rs.index, 0, es_soc_rs, color="gray",
                            alpha=0.25, label="Füllstand (SoC)")

    ax2.set_ylabel("Leistung [MW]", fontsize=14)
    ax2_r.set_ylabel("Füllstand [0-1]", fontsize=14)
    ax2.set_title("Saisonale Speicherung: Wasserkraft (Reservoir & PHS) in Frankreich", fontsize=16, pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(0, color="black", linewidth=0.8)

    lines2 = [l4, l5, l6]
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc="upper right", fontsize=12, framealpha=0.9)
    ax2_r.set_ylim(0, 1.1)

    # X-Achse
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax2.set_xlabel("Datum", fontsize=14)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Speichern
    filename = f"storage_winter_detail_DE_FR_{year}_1h.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Winter-Detail-Plot gespeichert: {filepath}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    config = PlottingConfig()
    networks = config.get_networks()

    if isinstance(networks, dict):
        if config.SCENARIO_SELECTION == "both":
            networks = list(networks.values())[0]
        else:
            networks = networks[config.SCENARIO_SELECTION]

    for path in networks:
        if not os.path.isfile(path):
            continue

        try:
            year = int(re.search(r"_(\d{4})\.nc$", path).group(1))

            if year != 2050:
                continue

            n = pypsa.Network(path)
            print(f"\n📂 Analysiere Winter-Detail für {year}...")

            save_dir = os.path.join(config.BASE_SAVE_PATH,
                                    os.path.basename(os.path.dirname(os.path.dirname(path))),
                                    "plots_storage_analysis")
            os.makedirs(save_dir, exist_ok=True)

            plot_winter_detail(n, year, save_dir)

        except Exception as e:
            print(f"❌ Fehler: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()