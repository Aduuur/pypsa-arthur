#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot: Zeitverlauf des Stromverbrauchs (Detail & Jahresverlauf)
==============================================================

Erstellt für jedes Land (inkl. Gesamtnetz "ALL") und jeden Planungshorizont:
1️⃣ Detailansicht (10. Januar – 15. Februar) mit stündlicher Auflösung
2️⃣ Jahresverlauf (01. Januar – 31. Dezember) mit Tagesmittelwerten

Zeigt den Stromverbrauch nach Sektor/Technologie.
Erzeugung wird ignoriert (negative Werte).
"""

import os
import re
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from config_final import PlottingConfig

# =====================================================================
# --- Zeiträume ---
# =====================================================================
DETAIL_START = "2005-01-10"
DETAIL_END   = "2005-02-15"
YEAR_START   = "2005-01-01"
YEAR_END     = "2005-12-31"

# =====================================================================
# --- Linkfilter & Bus-Erkennung ---
# =====================================================================
LINK_BLACKLIST_PAT = re.compile(
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
# --- Verbrauchs-Zeitreihe ---
# =====================================================================
def get_consumption_timeseries(n: pypsa.Network, country_code: str):
    """Berechnet Stromverbrauch (MW → GW) für alle elektrischen Verbraucher."""
    electric_carriers = detect_electric_buses(n)
    cons_ts = pd.DataFrame(index=n.snapshots)

    for name, row in n.links.iterrows():
        if not (row.bus0.startswith(country_code) or row.bus1.startswith(country_code) or country_code == "ALL"):
            continue
        if re.search(LINK_BLACKLIST_PAT, row.carrier or ""):
            continue

        c0 = n.buses.at[row.bus0, "carrier"]
        c1 = n.buses.at[row.bus1, "carrier"]
        carrier = row.carrier

        # Verbrauch = Strom fließt AUS elektrischem Bus
        if c0 in electric_carriers and not c1 in electric_carriers:
            s = (n.links_t.p0[name]).clip(lower=0)
        elif c1 in electric_carriers and not c0 in electric_carriers:
            s = (n.links_t.p1[name]).clip(lower=0)
        else:
            continue

        if s.max() > 0:
            cons_ts[carrier] = cons_ts.get(carrier, 0) + s

    # klassische Last hinzufügen
    if hasattr(n, "loads") and not n.loads.empty:
        if country_code == "ALL":
            loads = n.loads.copy()
        else:
            loads = n.loads[n.loads.bus.str.startswith(country_code)]

        if not loads.empty:
            p = n.loads_t.p[loads.index].copy()
            p.columns = ["load"] * len(p.columns)
            cons_ts = pd.concat([cons_ts, p], axis=1)

    return cons_ts.groupby(axis=1, level=0).sum() / 1e3  # MW → GW

# =====================================================================
# --- Plotfunktion ---
# =====================================================================
def plot_consumption(df, year_label, country, config: PlottingConfig, network_path: str):
    """Erstellt zwei Plots: Detail (Jan–Feb) + Jahresverlauf (Tagesmittel)."""
    if df.empty:
        print(f"⚠️ Kein Verbrauch für {country} ({year_label})")
        return

    # gewünschte Reihenfolge (unten nach oben)
    order = [
        "load", "battery charger", "H2 Electrolysis", "methanation", "Haber-Bosch",
        "heat pump", "resistive heater", "EV charger", "export"
    ]
    ordered_cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    df = df[ordered_cols]

    title_country = "Gesamtnetz" if country == "ALL" else country
    network_name = os.path.basename(os.path.dirname(os.path.dirname(network_path)))
    save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "consumption_timeline_only_consumption")
    os.makedirs(save_dir, exist_ok=True)

    # === 1️⃣ Detailansicht: Januar–Februar (stundenweise) ===
    df_detail = df.loc[DETAIL_START:DETAIL_END]
    fig, ax = plt.subplots(figsize=(13, 5))
    (-df_detail).plot.area(
        ax=ax,
        color=[config.CARRIER_COLORS.get(c, config.DEFAULT_COLOR) for c in df_detail.columns],
        linewidth=0,
        alpha=0.95
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlim(pd.to_datetime(DETAIL_START), pd.to_datetime(DETAIL_END))
    ax.set_ylabel("Elektrischer Verbrauch [GW]", fontsize=config.FONT_SIZES["label"])
    ax.set_xlabel("")
    ax.set_title(
        f"Zeitverlauf des Stromverbrauchs – {title_country} ({year_label})",
        fontsize=config.FONT_SIZES["title"],
        fontweight="normal",  # nicht fett
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=config.FONT_SIZES["tick"])
    ax.tick_params(axis="y", labelsize=config.FONT_SIZES["tick"])

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=config.FONT_SIZES["legend"],
        title="Sektor / Technologie",
    )

    fig.tight_layout()
    save_path_detail = os.path.join(save_dir, f"consumption_timeline_detail_{country}_{year_label}.png")
    plt.savefig(save_path_detail, dpi=300)
    plt.close(fig)
    print(f"✅ Detailplot gespeichert: {save_path_detail}")

    # === 2️⃣ Jahresverlauf: Tagesmittel ===
    df_year = df.loc[YEAR_START:YEAR_END].resample("1D").mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    (-df_year).plot.area(
        ax=ax,
        color=[config.CARRIER_COLORS.get(c, config.DEFAULT_COLOR) for c in df_year.columns],
        linewidth=0,
        alpha=0.95
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlim(pd.to_datetime(YEAR_START), pd.to_datetime(YEAR_END))
    ax.set_ylabel("Elektrischer Verbrauch [GW]", fontsize=config.FONT_SIZES["label"])
    ax.set_xlabel("")
    ax.set_title(
        f"Jahresverlauf des Stromverbrauchs – {title_country} ({year_label})",
        fontsize=config.FONT_SIZES["title"],
        fontweight="normal",  # nicht fett
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=config.FONT_SIZES["tick"])
    ax.tick_params(axis="y", labelsize=config.FONT_SIZES["tick"])

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=config.FONT_SIZES["legend"],
        title="Sektor / Technologie",
    )

    fig.tight_layout()
    save_path_year = os.path.join(save_dir, f"consumption_timeline_year_{country}_{year_label}.png")
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
            df = get_consumption_timeseries(n, country)
            plot_consumption(df, year, country, config, path)

if __name__ == "__main__":
    main()
