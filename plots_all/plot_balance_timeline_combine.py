#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot: Strombilanz (Erzeugung & Verbrauch) – Detail & Jahresverlauf
==================================================================

Erstellt für jedes Land (inkl. Gesamtnetz "ALL") und jeden Planungshorizont:
1️⃣ Detailansicht (10. Januar – 15. Februar) mit stündlicher Auflösung
2️⃣ Jahresverlauf (01. Januar – 31. Dezember) mit Tagesmittelwerten

Zeigt Erzeugung (positiv) und Verbrauch (invertiert) nach Technologie/Sektor.
"""

import os
import re
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config_final import PlottingConfig

# =====================================================================
# --- Zeiträume ---
# =====================================================================
DETAIL_START = "2005-01-10"
DETAIL_END   = "2005-02-15"
YEAR_START   = "2005-01-01"
YEAR_END     = "2005-12-31"

# =====================================================================
# --- Helper: elektrische Busse + Filter ---
# =====================================================================
LINK_BLACKLIST_GEN = re.compile(
    r"distribution|AC|DC|heat|boiler|pump|battery charger|storage charger|"
    r"pipeline|hydrogen network|H2 pipeline|industry|resistive|grid|EV|V2G|"
    r"urban central heat vent|Electrolysis|Haber-Bosch|methanation",
    re.IGNORECASE,
)

LINK_BLACKLIST_CONS = re.compile(
    r"distribution|AC line|DC line|heat|boiler|storage charger|pipeline|industry|grid|urban central heat vent",
    re.IGNORECASE,
)

def detect_electric_buses(n: pypsa.Network):
    carriers = n.buses.carrier.unique()
    if "electricity" in carriers:
        return {"electricity"}
    elif "AC" in carriers:
        return {"AC"}
    else:
        return set(c for c in carriers if "electr" in c.lower())

# =====================================================================
# --- Erzeugung & Verbrauchszeitreihen ---
# =====================================================================
def get_generation_timeseries(n: pypsa.Network, country_code: str):
    electric_carriers = detect_electric_buses(n)

    # Generatoren
    gens = n.generators if country_code == "ALL" else n.generators[n.generators.bus.str.startswith(country_code)]
    gens = gens[gens.bus.map(lambda b: n.buses.at[b, "carrier"] in electric_carriers)]
    gen_ts = pd.DataFrame(index=n.snapshots)

    if not gens.empty:
        p = n.generators_t.p[gens.index].copy()
        p.columns = gens.carrier.values
        gen_ts = p.groupby(axis=1, level=0).sum()

    # Links
    links = n.links.copy() if country_code == "ALL" else n.links[
        (n.links.bus0.str.startswith(country_code) | n.links.bus1.str.startswith(country_code))
    ]
    links = links[~links.carrier.str.contains(LINK_BLACKLIST_GEN, na=False)]

    link_ts = pd.DataFrame(index=n.snapshots)
    for name, row in links.iterrows():
        carrier = row.carrier
        b0, b1 = row.bus0, row.bus1
        c0, c1 = n.buses.at[b0, "carrier"], n.buses.at[b1, "carrier"]

        if c1 in electric_carriers and c0 not in electric_carriers:
            s = (-n.links_t.p1[name]).clip(lower=0)
        elif c0 in electric_carriers and c1 not in electric_carriers:
            s = (n.links_t.p0[name]).clip(lower=0)
        else:
            continue

        if s.max() > 0:
            link_ts[carrier] = link_ts.get(carrier, 0) + s

    total = pd.concat([gen_ts, link_ts], axis=1).groupby(axis=1, level=0).sum()
    return total.loc[:, total.max() > 0.5]


def get_consumption_timeseries(n: pypsa.Network, country_code: str):
    electric_carriers = detect_electric_buses(n)
    cons_ts = pd.DataFrame(index=n.snapshots)

    links = n.links.copy() if country_code == "ALL" else n.links[
        (n.links.bus0.str.startswith(country_code) | n.links.bus1.str.startswith(country_code))
    ]
    links = links[~links.carrier.str.contains(LINK_BLACKLIST_CONS, na=False)]

    for name, row in links.iterrows():
        carrier = row.carrier
        b0, b1 = row.bus0, row.bus1
        c0, c1 = n.buses.at[b0, "carrier"], n.buses.at[b1, "carrier"]

        if c0 in electric_carriers and c1 not in electric_carriers:
            s = (n.links_t.p0[name]).clip(lower=0)
        elif c1 in electric_carriers and c0 not in electric_carriers:
            s = (n.links_t.p1[name]).clip(lower=0)
        else:
            continue

        if s.max() > 0:
            cons_ts[carrier] = cons_ts.get(carrier, 0) + s

    if hasattr(n, "loads") and not n.loads.empty:
        loads = n.loads if country_code == "ALL" else n.loads[n.loads.bus.str.startswith(country_code)]
        if not loads.empty:
            p = n.loads_t.p[loads.index].copy()
            p.columns = ["load"] * len(p.columns)
            cons_ts = pd.concat([cons_ts, p], axis=1)

    total = cons_ts.groupby(axis=1, level=0).sum()
    return total.loc[:, total.max() > 0.5]

# =====================================================================
# --- Plotfunktion ---
# =====================================================================
def plot_balance(gen, cons, year_label, country, config: PlottingConfig, network_path: str):
    """Erzeugt Strombilanz-Plot mit getrennten Bereichen für Erzeugung & Verbrauch."""
    if gen.empty and cons.empty:
        print(f"⚠️ Keine Daten für {country} ({year_label})")
        return

    title_country = "Gesamtnetz" if country == "ALL" else country
    network_name = os.path.basename(os.path.dirname(os.path.dirname(network_path)))
    save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "balance_timeline_combined")
    os.makedirs(save_dir, exist_ok=True)

    # Reihenfolge
    gen_order = [
        "solar", "onwind", "offwind-ac", "offwind-dc",
        "ror", "hydro", "PHS", "H2 Fuel Cell", "H2 turbine",
        "CCGT", "OCGT", "coal", "lignite", "nuclear", "battery discharger"
    ]
    gen = gen[[c for c in gen_order if c in gen.columns] + [c for c in gen.columns if c not in gen_order]]

    cons_order = [
        "load", "battery charger", "H2 Electrolysis", "methanation",
        "Haber-Bosch", "heat pump", "resistive heater", "EV charger", "export"
    ]
    cons = cons[[c for c in cons_order if c in cons.columns] + [c for c in cons.columns if c not in cons_order]]

    # === 1️⃣ Detailansicht ===
    g = gen.loc[DETAIL_START:DETAIL_END]
    c = cons.loc[DETAIL_START:DETAIL_END]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})

    g.plot.area(ax=ax1, color=[config.CARRIER_COLORS.get(c, config.DEFAULT_COLOR) for c in g.columns],
                linewidth=0, alpha=0.95)
    (-c).plot.area(ax=ax2, color=[config.CARRIER_COLORS.get(c, config.DEFAULT_COLOR) for c in c.columns],
                   linewidth=0, alpha=0.95)

    ax1.set_ylabel("Erzeugung [GW]", fontsize=config.FONT_SIZES["label"])
    ax2.set_ylabel("Verbrauch [GW]", fontsize=config.FONT_SIZES["label"])
    ax2.invert_yaxis()

    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    ax1.set_title(f"Strombilanz – {title_country} ({year_label})",
                  fontsize=config.FONT_SIZES["title"], fontweight="normal")

    ax2.set_xlim(pd.to_datetime(DETAIL_START), pd.to_datetime(DETAIL_END))
    ax2.tick_params(axis="x", labelsize=config.FONT_SIZES["tick"])
    ax1.tick_params(axis="y", labelsize=config.FONT_SIZES["tick"])
    ax2.tick_params(axis="y", labelsize=config.FONT_SIZES["tick"])

    handles = [mpatches.Patch(color=config.CARRIER_COLORS.get(c, config.DEFAULT_COLOR), label=c)
               for c in list(g.columns) + list(c.columns)]
    fig.legend(handles=handles, bbox_to_anchor=(1.02, 0.5), loc="center left",
               fontsize=config.FONT_SIZES["legend"], title="Technologie / Sektor")

    fig.tight_layout()
    save_path_detail = os.path.join(save_dir, f"balance_timeline_detail_{country}_{year_label}.png")
    plt.savefig(save_path_detail, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Detailplot gespeichert: {save_path_detail}")

    # === 2️⃣ Jahresverlauf ===
    g_year = gen.loc[YEAR_START:YEAR_END].resample("1D").mean()
    c_year = cons.loc[YEAR_START:YEAR_END].resample("1D").mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})

    g_year.plot.area(ax=ax1, color=[config.CARRIER_COLORS.get(c, config.DEFAULT_COLOR) for c in g_year.columns],
                     linewidth=0, alpha=0.95)
    (-c_year).plot.area(ax=ax2, color=[config.CARRIER_COLORS.get(c, config.DEFAULT_COLOR) for c in c_year.columns],
                        linewidth=0, alpha=0.95)

    ax1.set_ylabel("Erzeugung [GW]", fontsize=config.FONT_SIZES["label"])
    ax2.set_ylabel("Verbrauch [GW]", fontsize=config.FONT_SIZES["label"])
    ax2.invert_yaxis()
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_title(f"Jahresverlauf Strombilanz – {title_country} ({year_label})",
                  fontsize=config.FONT_SIZES["title"], fontweight="normal")

    fig.tight_layout()
    save_path_year = os.path.join(save_dir, f"balance_timeline_year_{country}_{year_label}.png")
    plt.savefig(save_path_year, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Jahresplot gespeichert: {save_path_year}")

# =====================================================================
# --- Hauptablauf ---
# =====================================================================
def main():
    config = PlottingConfig()
    networks = config.get_networks()

    for path in networks:
        if not os.path.isfile(path):
            continue
        m = re.search(r"_(\d{4})\.nc$", path)
        if not m:
            continue
        year = int(m.group(1))

        print(f"\n📂 Lade Netzwerk {year}: {path}")
        n = pypsa.Network(path)

        for country in config.get_countries():
            gen = get_generation_timeseries(n, country)
            cons = get_consumption_timeseries(n, country)
            plot_balance(gen, cons, year, country, config, path)

if __name__ == "__main__":
    main()
