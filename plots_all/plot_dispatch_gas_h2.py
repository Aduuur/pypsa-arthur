#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geglätteter Jahres-Dispatch von Gas- & H₂-Kraftwerken (pro Land + ALL)

- Gas: OCGT, CCGT
- H₂ : H2 turbine, H2 Fuel Cell

Erkennt das elektrische Ende jedes Links (bus0/bus1) und summiert nur
die positive Einspeisung ins Stromnetz. Funktioniert für einzelne Länder
und für 'ALL' (gesamtes Netz).
"""

import os
import re
import pypsa
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config_final import PlottingConfig

# ================================
# Definition der Technologiegruppen
# ================================
GAS_TECHS = ["OCGT", "CCGT"]
H2_TECHS  = ["H2 turbine", "H2 Fuel Cell"]


# =====================================================================
# --- Helper-Funktionen ---
# =====================================================================
def detect_electric_buses(n: pypsa.Network):
    carriers = set(n.buses.carrier.unique())
    if "electricity" in carriers:
        return {"electricity"}
    if "AC" in carriers:
        return {"AC"}
    # Fallback: alles mit 'electr' im Namen
    return {c for c in carriers if "electr" in str(c).lower()}


def _series_positive_electric_injection(n: pypsa.Network, link_name: str, electric_carriers):
    """Liefert die (>=0) Einspeisung dieses Links ins elektrische Netz in MW."""
    row = n.links.loc[link_name]
    b0, b1 = row.bus0, row.bus1
    c0, c1 = n.buses.at[b0, "carrier"], n.buses.at[b1, "carrier"]

    # Strom am elektrischen Ende ist positiv, wenn er AUS dem Link IN den Bus fließt
    if c1 in electric_carriers and c0 not in electric_carriers:
        # elektrisches Ende = bus1 → Einspeisung = -p1
        return (-n.links_t.p1[link_name]).clip(lower=0)
    if c0 in electric_carriers and c1 not in electric_carriers:
        # elektrisches Ende = bus0 → Einspeisung = +p0
        return (n.links_t.p0[link_name]).clip(lower=0)
    # beide Seiten elektrisch → auf das Ende mit größerer Einspeisung projizieren
    if c0 in electric_carriers and c1 in electric_carriers:
        p0pos = n.links_t.p0[link_name].clip(lower=0)
        p1pos = (-n.links_t.p1[link_name]).clip(lower=0)
        return pd.concat([p0pos, p1pos], axis=1).max(axis=1)
    # kein elektrisches Ende
    return pd.Series(0.0, index=n.snapshots)


def get_dispatch_by_tech_group(n: pypsa.Network, country: str):
    """
    Gibt DataFrame [MW] mit Spalten 'Gas' und 'H2' zurück (stündlich).
    - filtert auf Links, deren elektrisches Ende im Land liegt (oder ALL)
    - nimmt nur positive Einspeisung
    """
    elec = detect_electric_buses(n)
    if n.links.empty:
        return pd.DataFrame({"Gas": 0.0, "H2": 0.0}, index=n.snapshots)

    # Kandidaten: nur Träger aus Gas/H2
    cand = n.links[n.links.carrier.isin(GAS_TECHS + H2_TECHS)].copy()
    if cand.empty:
        return pd.DataFrame({"Gas": 0.0, "H2": 0.0}, index=n.snapshots)

    # Länderfilter: elektrisches Ende muss im Land liegen (oder ALL = kein Filter)
    def electric_end_in_country(row):
        b0, b1 = row.bus0, row.bus1
        c0 = n.buses.at[b0, "carrier"]
        c1 = n.buses.at[b1, "carrier"]
        if country == "ALL":
            return True
        if c1 in elec and b1.startswith(country):
            return True
        if c0 in elec and b0.startswith(country):
            return True
        return False

    cand = cand[cand.apply(electric_end_in_country, axis=1)]
    if cand.empty:
        return pd.DataFrame({"Gas": 0.0, "H2": 0.0}, index=n.snapshots)

    # Summieren
    gas_sum = pd.Series(0.0, index=n.snapshots)
    h2_sum  = pd.Series(0.0, index=n.snapshots)

    # Logging: was ist „H2“?
    print(f"🔎 H2 zählt als: {H2_TECHS}")
    print(f"🔎 Gas zählt als: {GAS_TECHS}")

    gas_links = cand[cand.carrier.isin(GAS_TECHS)].index.tolist()
    h2_links  = cand[cand.carrier.isin(H2_TECHS)].index.tolist()
    print(f"   {country}: {len(gas_links)} Gas-Links, {len(h2_links)} H₂-Links (elektrisches Ende im Land/ALL)")

    for name in gas_links:
        gas_sum = gas_sum.add(_series_positive_electric_injection(n, name, elec), fill_value=0.0)
    for name in h2_links:
        h2_sum  = h2_sum.add(_series_positive_electric_injection(n, name, elec), fill_value=0.0)

    # MW → GW
    return pd.DataFrame({"Gas": gas_sum / 1e3, "H2": h2_sum / 1e3})


# =====================================================================
# --- Plot ---
# =====================================================================
def plot_single(country, year, df, config: PlottingConfig, save_folder: str, scenario_title: str):
    if df is None or df.empty or (df.sum().sum() == 0):
        print(f"⚠️ Keine Gas/H₂-Einspeisung gefunden für {country} ({year}).")
        return

    # 7-Tage gleitendes Mittel
    roll = df.rolling(window=7 * 24, min_periods=1, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(roll.index, roll["Gas"], color="#d95f02", lw=2, label="Gas")
    ax.plot(roll.index, roll["H2"], color="#7570b3", lw=2, label="H₂")

    ax.set_title(
        f"Geglätteter Einsatz von Gas- & H₂-Kraftwerken in {country} – {year} ({scenario_title})",
        fontsize=config.FONT_SIZES["title"], fontweight="normal"
    )
    ax.set_ylabel("Durchschnittliche Erzeugung (GW)", fontsize=config.FONT_SIZES["label"])
    ax.set_xlabel("Datum", fontsize=config.FONT_SIZES["label"])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.setp(ax.get_xticklabels(), rotation=0)

    ax.legend(fontsize=config.FONT_SIZES["legend"])
    plt.tight_layout()

    save_dir = os.path.join(save_folder, "gas_h2__dispatch_chronological")  # <== Dein Wunschname
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"{country}_{year}_dispatch_gas_h2.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✅ gespeichert: {fname}")


# =====================================================================
# --- Hauptablauf ---
# =====================================================================
def main():
    config = PlottingConfig()
    networks = config.get_networks()  # Liste von .nc-Pfaden (Single-Szenario)

    if not isinstance(networks, list) or not networks:
        print("FEHLER: Erwartet Liste von Netzwerkdateien aus config_final.get_networks().")
        return

    scenario_title = config.SCENARIO_SELECTION.replace("_", " ").title()

    for path in networks:
        if not os.path.isfile(path):
            print(f"⚠️ Datei nicht gefunden: {path}")
            continue

        m = re.search(r"_(\d{4})\.nc$", path)
        if not m:
            continue
        year = int(m.group(1))

        print(f"\n📂 Lade Netzwerk {year}: {path}")
        n = pypsa.Network(path)

        network_root = os.path.dirname(os.path.dirname(path))
        network_name = os.path.basename(network_root)
        save_folder = os.path.join(config.BASE_SAVE_PATH, network_name)

        # Alle gewünschten Länder inkl. ALL
        for country in config.get_countries():
            df = get_dispatch_by_tech_group(n, country)
            if not df.empty:
                print(f"   {country} {year}: Gas mean={df['Gas'].mean():.3f} GW, H2 mean={df['H2'].mean():.3f} GW")
            plot_single(country, year, df, config, save_folder, scenario_title)


if __name__ == "__main__":
    main()
