import os
import re
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from config_final import PlottingConfig

# =====================================================================
# EINSTELLUNGEN
# =====================================================================

# Zeitraum für Detail-Plot (Dunkelflaute)
DF_START_MONTH = 1
DF_START_DAY = 7
DF_DURATION_DAYS = 21

# Schwellenwert für Legende (%)
LEGEND_THRESHOLD = 0.1

LINK_BLACKLIST_PAT = re.compile(
    r"distribution|heat|boiler|pump|battery charger|storage charger|"
    r"pipeline|hydrogen network|H2 pipeline|industry|resistive|"
    r"grid|EV|V2G|urban central heat vent|DAC|process",
    re.IGNORECASE,
)

CARRIER_TRANSLATION = {
    "solar": "Photovoltaik",
    "solar rooftop": "Photovoltaik",
    "solar-hsat": "Photovoltaik",
    "onwind": "Wind Onshore",
    "offwind-ac": "Wind Offshore",
    "offwind-dc": "Wind Offshore",
    "ror": "Laufwasser",
    "hydro": "Wasserkraft",  # Reservoir Hydro
    "PHS": "Pumpspeicher",  # Pumped Hydro
    "nuclear": "Kernkraft",
    "lignite": "Braunkohle",
    "coal": "Steinkohle",
    "oil": "Öl",
    "CCGT": "Erdgas (GuD)",
    "OCGT": "Erdgas (Gasturbine)",
    "OCGT methanol": "Erdgas (Gasturbine)",
    "battery discharger": "Batteriespeicher",
    "home battery discharger": "Batteriespeicher",
    "H2 turbine": "H2-Turbine",
    "H2 OCGT": "H2-Gasturbine",
    "H2 Fuel Cell": "Brennstoffzelle",
    "urban central solid biomass CHP": "Biomasse-KWK",
    "urban central solid biomass CHP CC": "Biomasse-KWK",
    "urban central gas CHP": "Gas-KWK",
    "urban central gas CHP CC": "Gas-KWK",
    "Net Import": "Nettohandel"
}


# =====================================================================
# HELPER
# =====================================================================

def detect_electric_buses(n: pypsa.Network):
    carriers = n.buses.carrier.unique()
    if "electricity" in carriers:
        return {"electricity"}
    if "AC" in carriers:
        return {"AC"}
    return set(c for c in carriers if "electr" in c.lower())


def consolidate_carriers(df):
    df_renamed = pd.DataFrame(index=df.index)
    for col in df.columns:
        german_name = CARRIER_TRANSLATION.get(col, col)
        if german_name in df_renamed.columns:
            df_renamed[german_name] += df[col]
        else:
            df_renamed[german_name] = df[col]
    return df_renamed


def get_generation_timeseries(n: pypsa.Network, country_code: str):
    """
    Sammelt Erzeugungsdaten von Generators, Links UND StorageUnits.
    """
    electric_carriers = detect_electric_buses(n)

    # --- 1. GENERATORS (Wind, Solar, Laufwasser, Gas, etc.) ---
    gens = n.generators[n.generators.bus.str.startswith(country_code)]
    # Filter auf elektrische Busse
    gens = gens[gens.bus.map(lambda b: n.buses.at[b, "carrier"] in electric_carriers)]

    if not gens.empty:
        p = n.generators_t.p[gens.index].copy()
        p.columns = gens.carrier.values
        gen_ts = p.groupby(axis=1, level=0).sum()
    else:
        gen_ts = pd.DataFrame(index=n.snapshots)

    # --- 2. STORAGE UNITS (Wasserkraft Reservoirs, PHS) ---
    # Das fehlte vorher! Hydro Reservoirs sind StorageUnits.
    sus = n.storage_units[n.storage_units.bus.str.startswith(country_code)]
    sus = sus[sus.bus.map(lambda b: n.buses.at[b, "carrier"] in electric_carriers)]

    if not sus.empty:
        # p_dispatch ist die Erzeugung (Turbinieren)
        p_su = n.storage_units_t.p_dispatch[sus.index].copy()
        p_su.columns = sus.carrier.values
        su_ts = p_su.groupby(axis=1, level=0).sum()
    else:
        su_ts = pd.DataFrame(index=n.snapshots)

    # --- 3. LINKS (Batterien Discharger, Interkonnektoren die als Generatoren wirken) ---
    links = n.links[
        (n.links.bus0.str.startswith(country_code) | n.links.bus1.str.startswith(country_code))
        & ~n.links.carrier.str.contains(LINK_BLACKLIST_PAT)
        ]
    # AC/DC Links filtern wir hier raus, die machen wir separat über Import/Export Funktion
    links = links[~links.carrier.str.contains("Electrolysis|DC|AC", case=False, na=False)]

    link_data = {}
    for name, row in links.iterrows():
        carrier = row.carrier
        b0_loc = str(row.bus0)[:2]
        b1_loc = str(row.bus1)[:2]
        c0 = n.buses.at[row.bus0, "carrier"]
        c1 = n.buses.at[row.bus1, "carrier"]

        # Logik: Wir suchen Links, die in das Stromnetz einspeisen (wie Generatoren)
        val = None

        # Fall 1: Link endet im Land (bus1) -> Einspeisung ist p1 (positiv definiert?)
        # PyPSA Konvention: p1 ist Leistung am Bus1. Wenn positiv -> Einspeisung in Bus.
        # Aber bei Dischargern ist oft p0 positiv (Entnahme aus Store) und p1 = -p0 * eff (negativ am Bus? oder positiv?)
        # Wir prüfen p1. Normalerweise: p1 ist negativ bei Entnahme aus Link.
        # Aber wir nehmen hier an: Wir wollen POSITIVE Erzeugung sehen.

        if b1_loc == country_code and c1 in electric_carriers:
            # Energie kommt am Bus1 an. In PyPSA ist p1 oft negativ (Entnahme aus Link).
            # Wir nehmen -p1 und clippen auf positiv.
            val = (-n.links_t.p1[name]).clip(lower=0)

        # Fall 2: Link startet im Land (bus0) -> Einspeisung in Bus0?
        # Unwahrscheinlich für "Generation", meistens Verbraucher (P2G). Ignorieren wir hier meistens.
        elif b0_loc == country_code and c0 in electric_carriers:
            continue
        else:
            continue

        if val is not None:
            s = val
            if s.sum() > 0:
                if carrier in link_data:
                    link_data[carrier] += s
                else:
                    link_data[carrier] = s

    link_ts = pd.DataFrame(link_data, index=n.snapshots)

    # --- ZUSAMMENFÜHREN ---
    total = pd.concat([gen_ts, su_ts, link_ts], axis=1).groupby(axis=1, level=0).sum()
    total = total.loc[:, total.sum() > 0.1]
    return total / 1e3  # MW -> GW


def get_cross_border_flow(n: pypsa.Network, country_code: str):
    net_import = pd.Series(0.0, index=n.snapshots)
    for idx, line in n.lines.iterrows():
        c0 = line.bus0[:2]
        c1 = line.bus1[:2]
        if c0 == country_code and c1 != country_code:
            net_import -= n.lines_t.p0[idx]
        elif c1 == country_code and c0 != country_code:
            net_import += n.lines_t.p0[idx]

    dc_links = n.links[n.links.carrier == "DC"]
    for idx, link in dc_links.iterrows():
        c0 = link.bus0[:2]
        c1 = link.bus1[:2]
        if c0 == country_code and c1 != country_code:
            net_import -= n.links_t.p0[idx]
        elif c1 == country_code and c0 != country_code:
            net_import += -n.links_t.p1[idx]
    return net_import / 1e3


def filter_by_contribution(df, threshold_pct=0.7):
    abs_sums = df.abs().sum()
    total = abs_sums.sum()
    if total == 0:
        return df
    contributions = (abs_sums / total * 100)
    keep_cols = [col for col in df.columns if contributions[col] >= threshold_pct or col == "Nettohandel"]
    return df[keep_cols].copy()


# =====================================================================
# PLOTTING
# =====================================================================

def plot_difference_with_trade(
        df_base, df_df, imp_base, imp_df,
        year, country, config,
        base_name, df_name,
        period_label, period_slice
):
    """Erstellt den Differenz-Plot mit fester Geometrie."""

    # 1. Daten Slicen & Resampling
    d_base = df_base.loc[period_slice]
    d_df = df_df.loc[period_slice]
    i_base = imp_base.loc[period_slice]
    i_df = imp_df.loc[period_slice]

    if d_base.empty or d_df.empty: return
    if period_label == "year":
        d_base = d_base.resample("1D").mean()
        d_df = d_df.resample("1D").mean()
        i_base = i_base.resample("1D").mean()
        i_df = i_df.resample("1D").mean()

    d_base, d_df = d_base.align(d_df, join="outer", axis=1, fill_value=0)
    diff_gen = d_df - d_base
    diff_imp = i_df - i_base
    diff_gen = consolidate_carriers(diff_gen)
    diff_gen["Nettohandel"] = diff_imp
    diff_gen = filter_by_contribution(diff_gen, threshold_pct=LEGEND_THRESHOLD)

    cols = list(diff_gen.columns)
    if "Nettohandel" in cols:
        cols.remove("Nettohandel")
        cols = ["Nettohandel"] + sorted(cols)
    diff_gen = diff_gen[cols]

    # --- PLOT SETUP ---
    fig, ax = plt.subplots(figsize=(14, 7))

    pos = diff_gen.clip(lower=0)
    neg = diff_gen.clip(upper=0)

    colors = []
    for c in diff_gen.columns:
        if c == "Nettohandel":
            colors.append("#800080")
        elif c == "Sonstige":
            colors.append("#808080")
        else:
            orig_carrier = None
            for k, v in CARRIER_TRANSLATION.items():
                if v == c:
                    orig_carrier = k
                    break
            if orig_carrier:
                colors.append(config.CARRIER_COLORS.get(orig_carrier, config.DEFAULT_COLOR))
            else:
                colors.append(config.DEFAULT_COLOR)

    x = diff_gen.index
    ax.stackplot(x, pos.T, colors=colors, labels=diff_gen.columns, alpha=0.9)
    ax.stackplot(x, neg.T, colors=colors, alpha=0.9)
    ax.axhline(0, color="black", lw=1.5, zorder=5)

    # Titel & Labels
    title_period = "Januar" if period_label == "Januar" else "Jahr"
    ax.set_title(
        f"Delta Dispatch: {country} {year}\n"
        f"(Dunkelflaute minus Referenzwetter) - {title_period}",
        fontsize=18
    )
    ax.set_ylabel("Leistungsdifferenz [GW]\n(Positiv = Mehr in Dunkelflaute)", fontsize=16)

    # X-Achse
    if period_label == "Januar":
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    ax.tick_params(axis='both', labelsize=14)

    # Legende
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        reversed(handles), reversed(labels),
        loc='upper left', bbox_to_anchor=(1.02, 1),
        title="Technologie", fontsize=14, title_fontsize=14
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # --- FESTES LAYOUT ---
    plt.subplots_adjust(left=0.10, right=0.80, top=0.88, bottom=0.12)

    # Speichern
    path = os.path.join(config.PLOT_OUTPUT_PATH, "dispatch_difference")
    os.makedirs(path, exist_ok=True)
    fname = f"DiffDispatch_{country}_{year}_{period_label}.png"
    plt.savefig(os.path.join(path, fname), dpi=300)
    plt.close()
    print(f"    Plot gespeichert: {fname}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    cfg = PlottingConfig()
    if "CONDA_PREFIX" in os.environ:
        proj_lib = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
        if os.path.exists(proj_lib):
            os.environ["PROJ_LIB"] = proj_lib

    cfg.SCENARIO_SELECTION = "both"
    networks = cfg.get_networks()
    keys = list(networks.keys())
    if len(keys) != 2:
        print("Fehler: Modus 'both' muss genau 2 Szenarien liefern.")
        return

    base_key = keys[0]
    df_key = keys[1]
    map_base = {int(re.search(r"20\d{2}", f).group(0)): f for f in networks[base_key] if re.search(r"20\d{2}", f)}
    map_df = {int(re.search(r"20\d{2}", f).group(0)): f for f in networks[df_key] if re.search(r"20\d{2}", f)}

    years = sorted(list(set(map_base.keys()) & set(map_df.keys())))
    countries = [c for c in cfg.get_countries() if c != "ALL"]

    for year in years:
        print(f"\n=== Bearbeite Jahr {year} ===")
        try:
            n_base = pypsa.Network(map_base[year])
            n_df = pypsa.Network(map_df[year])
            sim_year = n_base.snapshots[0].year

            slice_januar = slice(f"{sim_year}-01-01", f"{sim_year}-01-31")
            slice_year = slice(f"{sim_year}-01-01", f"{sim_year}-12-31")

            for c in countries:
                print(f"  Land: {c}")
                gen_base = get_generation_timeseries(n_base, c)
                gen_df = get_generation_timeseries(n_df, c)
                imp_base = get_cross_border_flow(n_base, c)
                imp_df = get_cross_border_flow(n_df, c)

                if gen_base.empty or gen_df.empty: continue

                plot_difference_with_trade(
                    gen_base, gen_df, imp_base, imp_df,
                    year, c, cfg, base_key, df_key, "Januar", slice_januar
                )
                plot_difference_with_trade(
                    gen_base, gen_df, imp_base, imp_df,
                    year, c, cfg, base_key, df_key, "year", slice_year
                )

        except Exception as e:
            print(f"Fehler in {year}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()