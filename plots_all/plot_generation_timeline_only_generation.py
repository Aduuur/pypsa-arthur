#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot: Zeitverlauf der Stromerzeugung (Detail & Jahresverlauf)
=============================================================

Erstellt für jedes Land (inkl. Gesamtnetz "ALL") und jeden Planungshorizont:
1️⃣ Detailansicht (10. Januar – 15. Februar) mit stündlicher Auflösung
2️⃣ Jahresverlauf (01. Januar – 31. Dezember) mit Tagesmittelwerten

Zeigt die Stromerzeugung (positive Einspeisung) nach Technologie.
Verbrauch (z. B. Elektrolyse, Batterieladung) wird ausgeschlossen.
"""

import os
import re
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from config_final import PlottingConfig

# =====================================================================
# --- Linkfilter & Bus-Erkennung ---
# =====================================================================
LINK_BLACKLIST_PAT = re.compile(
    r"distribution|AC|DC|heat|boiler|pump|battery charger|storage charger|"
    r"pipeline|hydrogen network|H2 pipeline|industry|resistive|"
    r"grid|EV|V2G|urban central heat vent",
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
# --- Datenaufbereitung ---
# =====================================================================
def get_generation_timeseries(n: pypsa.Network, country_code: str):
    """Berechnet die zeitliche Erzeugung (MW → GW) pro Technologie."""
    electric_carriers = detect_electric_buses(n)

    # Generatoren
    if country_code == "ALL":
        gens = n.generators.copy()
    else:
        gens = n.generators[n.generators.bus.str.startswith(country_code)]

    gens = gens[gens.bus.map(lambda b: n.buses.at[b, "carrier"] in electric_carriers)]
    gen_ts = pd.DataFrame(index=n.snapshots)

    if not gens.empty:
        p = n.generators_t.p[gens.index].copy()
        p.columns = gens.carrier.values
        gen_ts = p.groupby(axis=1, level=0).sum()

    # Links
    if country_code == "ALL":
        links = n.links[~n.links.carrier.str.contains(LINK_BLACKLIST_PAT)]
    else:
        links = n.links[
            (n.links.bus0.str.startswith(country_code) | n.links.bus1.str.startswith(country_code))
            & ~n.links.carrier.str.contains(LINK_BLACKLIST_PAT)
        ]

    links = links[~links.carrier.str.contains("Electrolysis", case=False, na=False)]
    link_ts = pd.DataFrame(index=n.snapshots)

    for name, row in links.iterrows():
        carrier = row.carrier
        bus0_carr = n.buses.at[row.bus0, "carrier"]
        bus1_carr = n.buses.at[row.bus1, "carrier"]

        if bus1_carr in electric_carriers:
            s = (-n.links_t.p1[name]).clip(lower=0)
        elif bus0_carr in electric_carriers:
            s = (n.links_t.p0[name]).clip(lower=0)
        else:
            continue

        if s.max() > 0:
            link_ts[carrier] = link_ts.get(carrier, 0) + s

    total = pd.concat([gen_ts, link_ts], axis=1).groupby(axis=1, level=0).sum()
    total = total.loc[:, total.max() > 0.5]  # Filter kleine Werte
    return total / 1e3  # MW → GW

# =====================================================================
# --- Plotfunktion ---
# =====================================================================
def plot_generation(df, year_label, country, config: PlottingConfig, network_path: str):
    """Erstellt zwei Plots: Detail (Jan–Feb) + Jahresverlauf (Tagesmittel)."""
    if df.empty:
        print(f"⚠️ Keine Erzeugung für {country} ({year_label})")
        return

    # FIX: Zeitgrenzen dynamisch aus dem Netz-Index ableiten (kein hardcoded globaler Name)
    sim_year = df.index[0].year
    DETAIL_START = f"{sim_year}-01-10"
    DETAIL_END   = f"{sim_year}-02-15"
    YEAR_START   = f"{sim_year}-01-01"
    YEAR_END     = f"{sim_year}-12-31"

    # Reihenfolge der Technologien
    order = [
         "H2 Fuel Cell", "H2 turbine", "H2 OCGT", "battery discharger",
        "solar", "solar-hsat", "onwind", "offwind-ac", "offwind-dc",
        "hydro", "PHS", "biogas", "waste CHP", "OCGT", "CCGT",
        "urban central gas CHP", "urban central H2 CHP",
        "urban central H2 retrofit CHP", "waste CHP CC",
        "coal", "lignite", "solid biomass", "ror", "nuclear"
    ]
    ordered_cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    df = df[ordered_cols[::-1]]

    title_country = "Gesamtnetz" if country == "ALL" else country
    network_name = os.path.basename(os.path.dirname(os.path.dirname(network_path)))
    save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "generation_timeline_only_generation")
    os.makedirs(save_dir, exist_ok=True)

    # === 1️⃣ Detailansicht: Januar–Februar (stundenweise) ===
    df_detail = df.loc[DETAIL_START:DETAIL_END]
    if df_detail.empty:
        print(f"⚠️ Keine Daten im Detailzeitraum für {country} ({year_label}), skip.")
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    df_detail.plot.area(
        ax=ax,
        color=[config.CARRIER_COLORS.get(c, config.DEFAULT_COLOR) for c in df_detail.columns],
        linewidth=0,
        alpha=0.95,
    )
    ax.set_xlim(pd.to_datetime(DETAIL_START), pd.to_datetime(DETAIL_END))
    ax.set_ylabel("Elektrische Leistung [GW]", fontsize=config.FONT_SIZES["label"])
    ax.set_xlabel("")
    ax.set_title(
        f"Zeitverlauf der Stromerzeugung – {title_country} ({year_label})",
        fontsize=config.FONT_SIZES["title"],
        fontweight="normal",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=config.FONT_SIZES["tick"])
    ax.tick_params(axis="y", labelsize=config.FONT_SIZES["tick"])
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=config.FONT_SIZES["legend"],
        title="Technologie",
    )
    fig.tight_layout()
    save_path_detail = os.path.join(save_dir, f"generation_timeline_detail_{country}_{year_label}.png")
    plt.savefig(save_path_detail, dpi=300)
    plt.close(fig)
    print(f"✅ Detailplot gespeichert: {save_path_detail}")

    # === 2️⃣ Jahresverlauf: Tagesmittel ===
    df_year = df.loc[YEAR_START:YEAR_END].resample("1D").mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    df_year.plot.area(
        ax=ax,
        color=[config.CARRIER_COLORS.get(c, config.DEFAULT_COLOR) for c in df_year.columns],
        linewidth=0,
        alpha=0.95,
    )
    ax.set_xlim(pd.to_datetime(YEAR_START), pd.to_datetime(YEAR_END))
    ax.set_ylabel("Elektrische Leistung [GW]", fontsize=config.FONT_SIZES["label"])
    ax.set_xlabel("")
    ax.set_title(
        f"Jahresverlauf der Stromerzeugung – {title_country} ({year_label})",
        fontsize=config.FONT_SIZES["title"],
        fontweight="normal",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=config.FONT_SIZES["tick"])
    ax.tick_params(axis="y", labelsize=config.FONT_SIZES["tick"])
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=config.FONT_SIZES["legend"],
        title="Technologie",
    )
    fig.tight_layout()
    save_path_year = os.path.join(save_dir, f"generation_timeline_year_{country}_{year_label}.png")
    plt.savefig(save_path_year, dpi=300)
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
            print(f"⚠️ Datei nicht gefunden: {path}")
            continue

        m = re.search(r"_(\d{4})\.nc$", path)
        if not m:
            continue
        year = int(m.group(1))

        print(f"\n📂 Lade Netzwerk {year}: {path}")
        n = pypsa.Network(path)

        for country in config.get_countries():
            df = get_generation_timeseries(n, country)
            plot_generation(df, year, country, config, path)


if __name__ == "__main__":
    main()
