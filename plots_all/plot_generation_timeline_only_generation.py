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
import matplotlib.dates as mdates
from config_final import PlottingConfig, fill_leap_day

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
    """Erstellt drei Plots: Detail (Jan-Feb), Dunkelflaute (7d), Jahresverlauf."""
    if df.empty:
        print(f"  Keine Erzeugung fuer {country} ({year_label})")
        return

    sim_year = df.index[0].year
    DETAIL_START = f"{sim_year}-01-10"
    DETAIL_END   = f"{sim_year}-02-15"
    YEAR_START   = f"{sim_year}-01-01"
    YEAR_END     = f"{sim_year}-12-31"

    # ── Deutsche Labels ──────────────────────────────────────────
    LABELS_DE = {
        "nuclear": "Kernkraft", "ror": "Laufwasser", "hydro": "Wasserkraft",
        "PHS": "Pumpspeicher", "onwind": "Wind Onshore",
        "offwind-ac": "Wind Offshore (AC)", "offwind-dc": "Wind Offshore (DC)",
        "solar": "Photovoltaik", "solar rooftop": "Solar Aufdach",
        "solar-hsat": "Solar (Tracker)", "CCGT": "Erdgas (GuD)",
        "OCGT": "Erdgas (GT)", "coal": "Steinkohle", "lignite": "Braunkohle",
        "oil": "Oel", "biomass": "Biomasse",
        "solid biomass": "Festbiomasse", "biogas": "Biogas",
        "urban central solid biomass CHP": "Biomasse-KWK",
        "urban central solid biomass CHP CC": "Biomasse-KWK CC",
        "urban central gas CHP": "Gas-KWK",
        "H2 turbine": "H2-Turbine", "H2 Fuel Cell": "Brennstoffzelle",
        "OCGT methanol": "OCGT Methanol", "CCGT methanol": "CCGT Methanol",
        "battery discharger": "Batteriespeicher",
        "home battery discharger": "Heimspeicher",
    }

    # ── Stapelreihenfolge (unten → oben) ─────────────────────────
    STACK_ORDER = [
        "nuclear", "ror", "hydro", "PHS",
        "coal", "lignite", "oil",
        "solid biomass", "biomass", "biogas",
        "urban central solid biomass CHP", "urban central solid biomass CHP CC",
        "urban central gas CHP",
        "CCGT", "OCGT", "OCGT methanol", "CCGT methanol",
        "onwind", "offwind-ac", "offwind-dc",
        "solar", "solar rooftop", "solar-hsat",
        "H2 turbine", "H2 Fuel Cell",
        "battery discharger", "home battery discharger",
    ]

    # ── Daten vorbereiten ────────────────────────────────────────
    df = df.clip(lower=0)

    # Carrier mit < 0.5% Anteil entfernen
    total = df.sum().sum()
    if total > 0:
        keep = [c for c in df.columns if df[c].sum() / total > 0.005]
        df = df[keep]

    # Sortieren nach STACK_ORDER
    ordered = [c for c in STACK_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    df = df[ordered + rest]

    title_country = "Europa (Gesamt)" if country == "ALL" else country
    save_dir = os.path.join(config.PLOT_OUTPUT_PATH, "generation_timeline_only_generation")
    os.makedirs(save_dir, exist_ok=True)

    def _get_colors(cols):
        return [config.CARRIER_COLORS.get(c, config.DEFAULT_COLOR) for c in cols]

    def _get_labels(cols):
        return [LABELS_DE.get(c, c) for c in cols]

    def _make_legend(ax, cols):
        """Legende mit deutschen Labels, nur sichtbare Carrier."""
        handles, raw_labels = ax.get_legend_handles_labels()
        # Mapping: raw label -> deutsch
        label_map = {c: LABELS_DE.get(c, c) for c in cols}
        new_labels = [label_map.get(l, l) for l in raw_labels]
        ax.legend(
            handles[::-1], new_labels[::-1],
            bbox_to_anchor=(1.02, 1), loc="upper left",
            fontsize=8, frameon=True, framealpha=0.9,
            title="Technologie", title_fontsize=9,
        )

    def _style_ax(ax, ylabel="Elektrische Leistung [GW]"):
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_ylim(bottom=0)

    # ═════════════════════════════════════════════════════════════
    # 1. Detailansicht: Januar-Februar
    # ═════════════════════════════════════════════════════════════
    df_detail = df.loc[DETAIL_START:DETAIL_END]
    if df_detail.empty:
        print(f"  Keine Daten im Detailzeitraum fuer {country} ({year_label})")
        return
    df_detail = fill_leap_day(df_detail)

    fig, ax = plt.subplots(figsize=(14, 6))
    df_detail.plot.area(ax=ax, color=_get_colors(df_detail.columns),
                        linewidth=0, alpha=0.9)
    ax.set_xlim(pd.to_datetime(DETAIL_START), pd.to_datetime(DETAIL_END))
    ax.set_title(f"Stromerzeugung Detail (Jan-Feb) -- {title_country} ({year_label})",
                 fontsize=13, fontweight="normal")
    _style_ax(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d. %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    _make_legend(ax, df_detail.columns)
    plt.savefig(os.path.join(save_dir,
        f"generation_timeline_detail_{country}_{year_label}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Detail gespeichert: {country}_{year_label}")

    # ═════════════════════════════════════════════════════════════
    # 2. Dunkelflaute (7 Tage, nur wenn _from_ im Dateinamen)
    # ═════════════════════════════════════════════════════════════
    try:
        import re as _re_df
        _df_m = _re_df.search(r"_from_\d{4}_(\d{2})_(\d{2})", os.path.basename(network_path))
        if _df_m:
            df_start = f"{sim_year}-{_df_m.group(1)}-{_df_m.group(2)}"
            df_end = str((pd.Timestamp(df_start) + pd.Timedelta(days=6)).date())
            df_df = df.loc[df_start:df_end]
            if not df_df.empty and df_df.sum().sum() > 0:
                fig, ax = plt.subplots(figsize=(14, 6))
                df_df.plot.area(ax=ax, color=_get_colors(df_df.columns),
                                linewidth=0, alpha=0.9)
                ax.set_xlim(pd.to_datetime(df_start), pd.to_datetime(df_end))
                ax.set_title(f"Stromerzeugung Dunkelflaute -- {title_country} ({year_label})",
                             fontsize=13, fontweight="normal")
                _style_ax(ax)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d. %b"))
                ax.xaxis.set_major_locator(mdates.DayLocator())
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
                _make_legend(ax, df_df.columns)
                plt.savefig(os.path.join(save_dir,
                    f"generation_timeline_dunkelflaute_{country}_{year_label}.png"),
                    dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"  DF-Detail gespeichert: {country}_{year_label}")
    except Exception as _e:
        print(f"  Kein DF-Plot fuer {country}: {_e}")

    # ═════════════════════════════════════════════════════════════
    # 3. Jahresverlauf (Tagesmittel)
    # ═════════════════════════════════════════════════════════════
    df_year = df.loc[YEAR_START:YEAR_END].resample("1D").mean()
    df_year = df_year.clip(lower=0)
    df_year = fill_leap_day(df_year)

    fig, ax = plt.subplots(figsize=(14, 5))
    df_year.plot.area(ax=ax, color=_get_colors(df_year.columns),
                      linewidth=0, alpha=0.9)
    ax.set_xlim(pd.to_datetime(YEAR_START), pd.to_datetime(YEAR_END))
    ax.set_title(f"Jahresverlauf der Stromerzeugung -- {title_country} ({year_label})",
                 fontsize=13, fontweight="normal")
    _style_ax(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9)
    _make_legend(ax, df_year.columns)
    plt.savefig(os.path.join(save_dir,
        f"generation_timeline_year_{country}_{year_label}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Jahresplot gespeichert: {country}_{year_label}")


def main():
    config = PlottingConfig()
    networks = config.get_networks()

    for path in networks:
        if not os.path.isfile(path):
            print(f"⚠️ Datei nicht gefunden: {path}")
            continue

        m = re.search(r"_(\d{4})\.nc$", path)
        year = int(m.group(1)) if m else 2050  # ARO: kein Jahr im Namen

        print(f"\n📂 Lade Netzwerk {year}: {path}")
        n = pypsa.Network(path)

        for country in config.get_countries():
            df = get_generation_timeseries(n, country)
            plot_generation(df, year, country, config, path)


if __name__ == "__main__":
    main()
