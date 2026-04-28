#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot: Physikalischer Dispatch (V17)
=======================================================

"""

import os
import re
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from config_final import PlottingConfig, fill_leap_day

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


# =====================================================================
# --- 1. IMPORT/EXPORT ---
# =====================================================================
def get_physical_net_export(n: pypsa.Network, country_code: str):
    if country_code == "ALL": return pd.Series(0.0, index=n.snapshots)
    internal_buses = n.buses.index[n.buses.index.str.startswith(country_code)]
    net_export = pd.Series(0.0, index=n.snapshots)

    # AC Lines
    lines_out = n.lines[(n.lines.bus0.isin(internal_buses)) & (~n.lines.bus1.isin(internal_buses))].index
    if not lines_out.empty: net_export += n.lines_t.p0[lines_out].sum(axis=1)
    lines_in = n.lines[(~n.lines.bus0.isin(internal_buses)) & (n.lines.bus1.isin(internal_buses))].index
    if not lines_in.empty: net_export += n.lines_t.p1[lines_in].sum(axis=1)

    # DC Links
    dc_links = n.links[n.links.carrier == "DC"]
    links_out = dc_links[(dc_links.bus0.isin(internal_buses)) & (~dc_links.bus1.isin(internal_buses))].index
    if not links_out.empty: net_export += n.links_t.p0[links_out].sum(axis=1)
    links_in = dc_links[(~dc_links.bus0.isin(internal_buses)) & (dc_links.bus1.isin(internal_buses))].index
    if not links_in.empty: net_export += n.links_t.p1[links_in].sum(axis=1)

    return net_export / 1e3


# =====================================================================
# --- 2. ERZEUGUNG ---
# =====================================================================
def get_generation_timeseries(n: pypsa.Network, country_code: str):
    print(f"   -> Berechne Erzeugung für {country_code}...")
    elec_buses = get_electric_bus_ids(n, country_code)
    total_list = []

    # A) GENERATOREN
    # Blacklist geleert, damit Gas/Kohle nicht gefiltert werden
    gen_blacklist = ["unsustainable", "bioliquids"]

    gens = n.generators[n.generators.bus.isin(elec_buses)]
    gens = gens[~gens.carrier.str.contains("|".join(gen_blacklist), case=False)]

    if not gens.empty:
        p_gen = n.generators_t.p[gens.index].copy()
        p_gen.columns = gens.carrier.values
        total_list.append(p_gen.T.groupby(level=0).sum().T)

    # B) STORAGE UNITS
    sus = n.storage_units[n.storage_units.bus.isin(elec_buses)]
    if not sus.empty:
        p_su = n.storage_units_t.p_dispatch[sus.index].copy()
        p_su.columns = sus.carrier.values
        total_list.append(p_su.T.groupby(level=0).sum().T)

    # C) LINKS
    links_in = n.links[n.links.bus1.isin(elec_buses)]
    ignore_carriers = ["distribution grid", "transmission", "AC", "DC",
                       "home battery charger", "BEV charger"]

    ts_dict = {}
    for name, row in links_in.iterrows():
        carrier = str(row.carrier)
        if any(x in carrier for x in ignore_carriers): continue
        val = None
        if name in n.links_t.p1:
            val = -n.links_t.p1[name]
        elif name in n.links_t.p0:
            val = n.links_t.p0[name] * row.efficiency
        if val is not None:
            val = val.clip(lower=0)
            if val.sum() > 1:
                if carrier not in ts_dict: ts_dict[carrier] = 0.0
                ts_dict[carrier] += val

    if ts_dict: total_list.append(pd.DataFrame(ts_dict))
    if not total_list: return pd.DataFrame(index=n.snapshots)

    total = pd.concat(total_list, axis=1)
    total = total.T.groupby(level=0).sum().T

    cleaned = pd.DataFrame(index=total.index)
    for col in total.columns:
        c = col.lower()
        if "water" in c or "heat" in c: continue
        new_name = col

        # --- KONSISTENTE NAMENSGEBUNG & TRENNUNG ---
        if "lignite" in c:
            new_name = "Braunkohle"
        elif "coal" in c:
            new_name = "Steinkohle"
        elif "biomass" in c or "biogas" in c:
            new_name = "Biomasse"
        elif "solar" in c:
            new_name = "Photovoltaik"
        elif "offwind" in c:
            new_name = "Wind Offshore"
        elif "onwind" in c:
            new_name = "Wind Onshore"
        elif "nuclear" in c:
            new_name = "Kernkraft"

        # --- HIER IST DIE TRENNUNG ---
        elif "ccgt" in c:
            new_name = "Erdgas (GuD)"
        elif "ocgt" in c:
            new_name = "Erdgas (Gasturbine)"
        elif "gas" in c:
            new_name = "Erdgas (Sonstige)"
        # -----------------------------

        elif "oil" in c:
            new_name = "Öl"
        elif "hydro" in c:
            new_name = "Wasserkraft"
        elif "ror" in c:
            new_name = "Laufwasser"
        elif "phs" in c:
            new_name = "Pumpspeicher"
        elif "battery" in c and ("discharg" in c or "inverter" in c):
            new_name = "Batteriespeicher"
        elif "v2g" in c:
            new_name = "V2G (Auto)"
        elif "h2" in c and ("turbine" in c or "fuel" in c):
            new_name = "H2-Rückverstromung"

        if new_name not in cleaned: cleaned[new_name] = 0.0
        cleaned[new_name] += total[col]

    return cleaned / 1e3


# =====================================================================
# --- 3. VERBRAUCH ---
# =====================================================================
def get_domestic_demand_series(n: pypsa.Network, country_code: str):
    elec_buses = get_electric_bus_ids(n, country_code)
    total_demand = pd.Series(0.0, index=n.snapshots)

    # 1. Last
    load_ids = n.loads[(n.loads.bus.isin(elec_buses)) & (n.loads.carrier == "electricity")].index
    if "p" in n.loads_t:
        valid = load_ids.intersection(n.loads_t.p.columns)
        if not valid.empty: total_demand += n.loads_t.p[valid].sum(axis=1)
    else:
        valid = load_ids.intersection(n.loads_t.p_set.columns)
        if not valid.empty: total_demand += n.loads_t.p_set[valid].sum(axis=1)

    # 2. Speicher
    sus = n.storage_units[n.storage_units.bus.isin(elec_buses)]
    if not sus.empty: total_demand += n.storage_units_t.p_store[sus.index].sum(axis=1)

    # 3. Links Eingang
    links_out = n.links[n.links.bus0.isin(elec_buses)]
    exclude = ["DC", "AC", "distribution", "biomass", "fossil", "gas", "oil", "bioliquid"]
    valid_links = []
    for name, row in links_out.iterrows():
        if "battery" in row.carrier and "charger" in row.carrier:
            valid_links.append(name)
        elif not any(ex in row.carrier for ex in exclude):
            valid_links.append(name)
    if valid_links: total_demand += n.links_t.p0[valid_links].clip(lower=0).sum(axis=1)

    # 4. Interne Verluste
    dist_links = n.links[(n.links.bus0.isin(elec_buses)) & (n.links.carrier.str.contains("distribution grid"))]
    if not dist_links.empty:
        valid = dist_links.index.intersection(n.links_t.p0.columns)
        if not valid.empty:
            p0 = n.links_t.p0[valid]
            eff = dist_links.loc[valid, "efficiency"]
            total_demand += p0.multiply(1 - eff, axis=1).sum(axis=1)

    internal_lines = n.lines[(n.lines.bus0.isin(elec_buses)) & (n.lines.bus1.isin(elec_buses))]
    if not internal_lines.empty:
        total_demand += (n.lines_t.p0[internal_lines.index] + n.lines_t.p1[internal_lines.index]).sum(axis=1)

    internal_dc = n.links[(n.links.bus0.isin(elec_buses)) & (n.links.bus1.isin(elec_buses)) & (n.links.carrier == "DC")]
    if not internal_dc.empty:
        valid_p1 = internal_dc.index.intersection(n.links_t.p1.columns)
        if not valid_p1.empty:
            total_demand += (n.links_t.p0[valid_p1] + n.links_t.p1[valid_p1]).sum(axis=1)

    return total_demand / 1e3


# =====================================================================
# --- PLOTTING ---
# =====================================================================
def plot_dispatch(df_gen, s_load, s_export_net, year, country, config, net_name,
                  period_label, start=None, end=None, resample="1D"):
    if df_gen.empty: return

    # 1. Slice
    if start and end:
        if start not in df_gen.index or end not in df_gen.index:
            data_year = str(df_gen.index[0].year)
            start = start.replace(str(year), data_year)
            end = end.replace(str(year), data_year)
        df_gen = df_gen.loc[start:end]
        s_load = s_load.loc[start:end]
        s_export_net = s_export_net.loc[start:end]
        if df_gen.empty: return

    # 2. Resample
    df_gen = fill_leap_day(df_gen)
    s_load = fill_leap_day(s_load.to_frame()).iloc[:,0]
    s_export_net = fill_leap_day(s_export_net.to_frame()).iloc[:,0]
    if resample != "1h":
        df_res = df_gen.resample(resample).mean()
        load_res = s_load.resample(resample).mean()
        export_res = s_export_net.resample(resample).mean()
    else:
        df_res = df_gen
        load_res = s_load
        export_res = s_export_net

    # 3. Import
    s_import_stack = (-export_res).clip(lower=0)
    if s_import_stack.max() > 0.01:
        df_res = df_res.copy()
        df_res["Import"] = s_import_stack

    # 4. Sortierung
    desired_order = [
        "Kernkraft", "Laufwasser", "Wasserkraft", "Pumpspeicher", "Biomasse", "Müll",
        "Braunkohle", "Steinkohle", "Öl",
        "Erdgas (GuD)", "Erdgas (Gasturbine)", "Erdgas (Sonstige)",
        "H2-Rückverstromung", "Batteriespeicher", "V2G (Auto)",
        "Wind Offshore", "Wind Onshore", "Photovoltaik",
        "Import"
    ]
    cols = [c for c in desired_order if c in df_res.columns] + [c for c in df_res.columns if c not in desired_order]
    df_res = df_res[cols]

    # 5. Farben
    colors = []
    for c in df_res.columns:
        if c == "Import":
            col = "#404040"
        elif c == "Biomasse":
            col = config.CARRIER_COLORS.get("biomass", "#baa741")
        elif c == "Wasserkraft":
            col = config.CARRIER_COLORS.get("hydro", "#298c81")
        elif c == "Laufwasser":
            col = config.CARRIER_COLORS.get("ror", "#4adbc8")
        elif c == "Pumpspeicher":
            col = config.CARRIER_COLORS.get("PHS", "#51dbcc")
        elif c == "Photovoltaik":
            col = config.CARRIER_COLORS.get("solar", "#f9d002")
        elif c == "Wind Onshore":
            col = config.CARRIER_COLORS.get("onwind", "#235ebc")
        elif c == "Wind Offshore":
            col = config.CARRIER_COLORS.get("offwind-ac", "#6895dd")
        elif c == "Erdgas (GuD)":
            col = config.CARRIER_COLORS.get("CCGT", "#a85522")
        elif c == "Erdgas (Gasturbine)":
            col = config.CARRIER_COLORS.get("OCGT", "#d68452")
        elif c == "Erdgas (Sonstige)":
            col = "#a85522"
        elif c == "Steinkohle":
            col = config.CARRIER_COLORS.get("coal", "#545454")
        elif c == "Braunkohle":
            col = config.CARRIER_COLORS.get("lignite", "#826837")
        elif c == "Kernkraft":
            col = config.CARRIER_COLORS.get("nuclear", "#ff8c00")
        elif c == "Batteriespeicher":
            col = config.CARRIER_COLORS.get("battery", "#ace37f")
        elif c == "V2G (Auto)":
            col = config.CARRIER_COLORS.get("V2G", "#e5ffa8")
        elif c == "H2-Rückverstromung":
            col = config.CARRIER_COLORS.get("H2 Fuel Cell", "#bf13a0")
        elif c == "Öl":
            col = config.CARRIER_COLORS.get("oil", "#7a7a7a")
        else:
            col = config.DEFAULT_COLOR
        colors.append(col)

    # 6. Filter für Legende
    total_generation = df_res.sum().sum()
    percentage_threshold = 0.001  # 0,1%

    # --- HIER GEÄNDERT: "Erdgas (Sonstige)" aus Whitelist entfernt ---
    ALWAYS_SHOW = [
        "Erdgas (GuD)", "Erdgas (Gasturbine)",
        "Steinkohle", "Braunkohle", "Kernkraft", "Öl"
    ]

    labels_filtered = []
    for col in df_res.columns:
        col_total = df_res[col].sum()

        # --- HIER GEÄNDERT: Expliziter Ausschluss für "Erdgas (Sonstige)" ---
        if col == "Erdgas (Sonstige)":
            labels_filtered.append("")  # Niemals anzeigen
        # Normale Logik für den Rest
        elif (col_total / total_generation >= percentage_threshold) or (col in ALWAYS_SHOW):
            labels_filtered.append(col)
        else:
            labels_filtered.append("")

    # 7. Plotting
    save_dir = os.path.join(config.BASE_SAVE_PATH, net_name, "plots_dispatch_final")
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Oben: Stack
    ax1.stackplot(df_res.index, df_res.T, labels=labels_filtered, colors=colors, alpha=0.9)

    # Linie
    label_text = "Inlandsverbrauch\n(Last, Speicher, Verluste)"
    ax1.plot(load_res.index, load_res, color="black", linewidth=1.8, label=label_text)

    # Titel
    if period_label == "Year":
        t_label = "im Jahresverlauf"
    elif period_label == "Januar":
        t_label = "(Dunkelflaute)"
    else:
        t_label = f"({period_label})"

    ax1.set_title(f"Dispatch {t_label} für {country} {year}", fontsize=15)

    # Y-Achse
    unit = "Leistung [GW]"
    if resample == "1D":
        unit += " (Tagesmittel)"
    elif resample == "6h":
        unit += " (6h-Mittel)"
    ax1.set_ylabel(unit, fontsize=12)

    # Legende bereinigen
    handles, labels = ax1.get_legend_handles_labels()
    final_handles = []
    final_labels = []
    for h, l in zip(handles, labels):
        if l != "":
            final_handles.append(h)
            final_labels.append(l)

    ax1.legend(final_handles[::-1], final_labels[::-1], bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

    # Unten: Netto-Export
    ax2.fill_between(export_res.index, export_res, 0, where=(export_res > 0), color="green", alpha=0.5,
                     label="Netto-Export")
    ax2.fill_between(export_res.index, export_res, 0, where=(export_res < 0), color="red", alpha=0.5,
                     label="Netto-Import")
    ax2.plot(export_res.index, export_res, color="black", linewidth=0.8)
    ax2.axhline(0, color="black", linewidth=1)

    ax2.set_ylabel("Netto-Fluss [GW]", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # X-Achse
    if period_label == "Year":
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    else:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    plt.tight_layout()
    filename = f"dispatch_{country}_{year}_{period_label}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close(fig)
    print(f"   -> Plot erstellt: {filename}")


# =====================================================================
# --- MAIN ---
# =====================================================================
def main():
    config = PlottingConfig()
    networks = config.get_networks()
    if isinstance(networks, dict): networks = list(networks.values())[0]

    for path in networks:
        if not os.path.isfile(path): continue
        m = re.search(r"_(\d{4})\.nc$", path)
        if not m: continue
        planning_year = int(m.group(1))

        print(f"\n📂 Netz {planning_year}: {os.path.basename(path)}")
        n = pypsa.Network(path)
        n.snapshots = pd.to_datetime(n.snapshots)

        weather_year = n.snapshots[0].year
        net_name = os.path.basename(os.path.dirname(os.path.dirname(path)))

        # --- FIX: Datum ISO Format (YYYY-MM-DD) ---
        jan_start = f"{weather_year}-01-01"
        jan_end = f"{weather_year}-01-31"

        july_start = f"{weather_year}-07-01"
        july_end = f"{weather_year}-07-08"

        for country in config.get_countries():
            if country == "ALL": continue

            df_gen = get_generation_timeseries(n, country)
            s_load = get_domestic_demand_series(n, country)
            s_export = get_physical_net_export(n, country)

            # 1. Jahr
            plot_dispatch(df_gen, s_load, s_export, planning_year, country, config, net_name,
                          period_label="Year", start=None, end=None, resample="1D")

            # 2. Januar
            plot_dispatch(df_gen, s_load, s_export, planning_year, country, config, net_name,
                          period_label="Januar", start=jan_start, end=jan_end, resample="6h")

            if country == "ES":
                plot_dispatch(df_gen, s_load, s_export, planning_year, country, config, net_name,
                              period_label="Juli-Woche", start=july_start, end=july_end, resample="1h")


if __name__ == "__main__":
    main()