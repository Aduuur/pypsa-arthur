#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot: Residuallast vs. Erneuerbare Einspeisung (Gap-Analyse) - V3 (Ohne Speicher-Last)
=====================================================================================
Fokus: Visualisierung der Lücke ("Gap") zwischen REINER Last und VRE-Einspeisung.
Änderung: Speicher-Beladung wird NICHT zur Last gezählt
"""

import os
import re
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config_final import PlottingConfig

plt.switch_backend("Agg")


# =====================================================================
# --- HILFSFUNKTIONEN ---
# =====================================================================

def get_electric_bus_ids(n: pypsa.Network, country_code: str):
    if country_code == "ALL":
        country_buses = n.buses
    else:
        country_buses = n.buses[n.buses.index.str.startswith(country_code)]

    def is_elec(carrier):
        c = str(carrier).lower()
        if any(x in c for x in
               ["heat", "water", "gas", "h2", "hydrogen", "oil", "biomass", "co2", "tank", "pit", "coal", "lignite"]):
            return False
        if c in ["electricity", "ac", "dc", "low voltage", "high voltage", "residential", "services", "transport"]:
            return True
        return False

    return country_buses[country_buses.carrier.map(is_elec)].index


def get_vre_generation(n: pypsa.Network, country_code: str):
    """Holt Wind- und Solar-Einspeisung."""
    elec_buses = get_electric_bus_ids(n, country_code)

    # Generatoren
    gens = n.generators[n.generators.bus.isin(elec_buses)]

    # Filter für VRE
    vre_carriers = ["onwind", "offwind-ac", "offwind-dc", "solar", "solar rooftop"]
    gens_vre = gens[gens.carrier.isin(vre_carriers)]

    if gens_vre.empty:
        return pd.DataFrame(0.0, index=n.snapshots, columns=["Wind", "Solar"])

    p_gen = n.generators_t.p[gens_vre.index].copy()
    p_gen.columns = gens_vre.carrier.values

    # Gruppieren
    grouped = p_gen.T.groupby(level=0).sum().T

    # Zusammenfassen
    res = pd.DataFrame(index=n.snapshots)
    res["Wind"] = 0.0
    res["Solar"] = 0.0

    for c in grouped.columns:
        if "wind" in c:
            res["Wind"] += grouped[c]
        elif "solar" in c:
            res["Solar"] += grouped[c]

    return res / 1e3  # GW


def get_total_load(n: pypsa.Network, country_code: str):
    """Berechnet die REINE elektrische Last (ohne Speicher-Beladung)."""
    elec_buses = get_electric_bus_ids(n, country_code)
    total_demand = pd.Series(0.0, index=n.snapshots)

    # 1. Klassische Last (Stromverbrauch)
    load_ids = n.loads[(n.loads.bus.isin(elec_buses)) & (n.loads.carrier == "electricity")].index
    if "p" in n.loads_t:
        valid = load_ids.intersection(n.loads_t.p.columns)
        if not valid.empty: total_demand += n.loads_t.p[valid].sum(axis=1)
    else:
        valid = load_ids.intersection(n.loads_t.p_set.columns)
        if not valid.empty: total_demand += n.loads_t.p_set[valid].sum(axis=1)

    # --- HIER GEÄNDERT: Speicher-Ladung wird NICHT mehr addiert ---
    # sus = n.storage_units[n.storage_units.bus.isin(elec_buses)]
    # if not sus.empty:
    #     total_demand += n.storage_units_t.p_store[sus.index].sum(axis=1)

    # 3. Links Eingang (Sektorkopplung: P2H, Elektrolyse, EV-Charger)
    # Alles was Strom verbraucht (p0 > 0)
    links_out = n.links[n.links.bus0.isin(elec_buses)]

    # Wir wollen Links, die Strom VERBRAUCHEN (z.B. Heat Pumps, Elektrolyse)
    # ABER: AC/DC Transmission und reine Batterielader schließen wir aus,
    # wenn wir die "Netto-Last" sehen wollen.

    exclude = ["DC", "AC", "distribution", "transmission", "battery charger", "charger"]
    # "charger" entfernt Batterie-Ladung via Links (neues PyPSA-Eur Modell)

    valid_links = []
    for name, row in links_out.iterrows():
        if not any(ex in row.carrier for ex in exclude):
            valid_links.append(name)

    if valid_links:
        total_demand += n.links_t.p0[valid_links].clip(lower=0).sum(axis=1)

    return total_demand / 1e3  # GW


# =====================================================================
# --- PLOTTING ---
# =====================================================================
def plot_residual_load(load, vre, year, country, config, net_name, period_label, start=None, end=None, resample="1D"):
    # 1. Slice
    if start and end:
        if start not in load.index:
            data_year = str(load.index[0].year)
            start = start.replace(str(year), data_year)
            end = end.replace(str(year), data_year)

        load = load.loc[start:end]
        vre = vre.loc[start:end]

    if load.empty: return

    # 2. Resample
    if resample != "1h":
        load = load.resample(resample).mean()
        vre = vre.resample(resample).mean()

    # 3. Berechne Residual-Last
    total_vre = vre["Wind"] + vre["Solar"]
    # Da wir Speicherladung entfernt haben, ist 'load' jetzt niedriger.
    # Gap zeigt jetzt die Lücke zur *Verbrauchs*-Last.
    residual = load - total_vre

    # 4. Plot
    save_dir = os.path.join(config.BASE_SAVE_PATH, net_name, "plots_residuallast_clean")  # Neuer Ordner
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Last Kurve
    ax.plot(load.index, load, color="black", linewidth=2, label="Elektrische Last")

    # VRE Area (Gestapelt)
    ax.stackplot(vre.index, vre["Wind"], vre["Solar"],
                 labels=["Wind", "Solar"],
                 colors=["#235ebc", "#f9d002"], alpha=0.8)

    # Residual Gap (Wo Last > VRE)
    ax.fill_between(load.index, total_vre, load,
                    where=(load > total_vre),
                    color="gray", alpha=0.3, hatch="//",
                    label="Residuallast (Gap)")

    # Styling
    ax.set_title(f"Residuallast vs. Erneuerbare Einspeisung - {country} {year} ({period_label})", fontsize=22)
    ax.set_ylabel("Leistung [GW]", fontsize=18)

    # Formatierung der Datumsachse
    if period_label == "Year":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    # ---> HIER: Schriftgröße der Achsen-Ticks (Zahlen/Datum) ändern <---
    ax.tick_params(axis='x', labelsize=16)  # X-Achse (Datum) größer
    ax.tick_params(axis='y', labelsize=16)  # Y-Achse (GW Werte) auch größer

    ax.grid(True, alpha=0.3)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 3, 1, 2]

    legend_size = 18
    if len(handles) < 4:
        ax.legend(loc="upper right")
    else:
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right")



    plt.tight_layout()
    filename = f"residual_gap_{country}_{year}_{period_label}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close(fig)
    print(f"   -> Plot erstellt: {filename}")


# =====================================================================
# --- MAIN ---
# =====================================================================
def main():
    config = PlottingConfig()
    networks = config.get_networks()

    if isinstance(networks, dict):
        if config.SCENARIO_SELECTION == "both":
            networks = networks.get('dunkelflaute', list(networks.values())[0])
        else:
            networks = networks[config.SCENARIO_SELECTION]

    for path in networks:
        if not os.path.isfile(path): continue
        m = re.search(r"_(\d{4})\.nc$", path)
        year = int(m.group(1)) if m else 2050  # ARO: kein Jahr im Namen

        if year not in [2025, 2030, 2035, 2040, 2045, 2050]: continue

        print(f"\n📂 Netz {year}: {os.path.basename(path)}")
        n = pypsa.Network(path)
        try:
            n.snapshots = pd.to_datetime(n.snapshots)
        except:
            pass

        weather_year = n.snapshots[0].year
        net_name = os.path.basename(os.path.dirname(os.path.dirname(path)))

        jan_feb_start = f"{weather_year}-01-01"
        jan_feb_end = f"{weather_year}-02-28"

        countries_to_plot = config.get_countries()
        if "ALL" not in countries_to_plot: countries_to_plot.append("ALL")

        for country in countries_to_plot:
            if country not in ["GB","NO", "FR", "DK", "ES"]: continue  # Fokus auf DE/ALL für Test

            print(f"   ... analysiere {country}")

            load = get_total_load(n, country)
            vre = get_vre_generation(n, country)

            # 1. Zoom
            plot_residual_load(load, vre, year, country, config, net_name,
                               period_label="Januar-Februar",
                               start=jan_feb_start, end=jan_feb_end,
                               resample="6h")

            # 2. Jahr
            plot_residual_load(load, vre, year, country, config, net_name,
                               period_label="Jahr",
                               start=None, end=None,
                               resample="1D")


if __name__ == "__main__":
    main()