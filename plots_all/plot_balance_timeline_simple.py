#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot: Strombilanz (einfach) – mit Nettoimporten (korrigierte Version)

Wesentliche Änderungen:
- Robustere Erkennung elektrischer Busse (case-insensitive, Fallback).
- Nutzt n.buses.country falls vorhanden (statt bus_name[:2]).
- Prüft Timeseries-Attribute (generators_t, links_t, lines_t, storage_units_t, stores_t).
- Berechnet Net-Imports konsistent (Konvention: links_t.p0 = Flow von bus0 -> bus1).
- Berechnet Verluste als (p0 + p1) und teilt grenzüberschreitende Verluste 50/50 auf.
- Korrekte Umrechnung in GWh: energy [GWh] = sum(power [MW] * hours_per_snapshot) / 1e3.
"""

import os
import re
from datetime import timedelta
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config_final import PlottingConfig

# =====================================================================
# --- Zeiträume & Konfiguration ---
# =====================================================================
DETAIL_START = "2005-01-10"
DETAIL_END = "2005-02-15"
YEAR_START = "2005-01-01"
YEAR_END = "2005-12-31"

# =====================================================================
# --- Hilfsfunktionen ---
# =====================================================================
def hours_per_snapshot(n: pypsa.Network) -> float:
    """Bestimme die Stunden pro Snapshot (falls möglich), fallback auf 1.0."""
    snaps = list(n.snapshots)
    if len(snaps) >= 2:
        delta = snaps[1] - snaps[0]
        if isinstance(delta, timedelta):
            hours = delta.total_seconds() / 3600.0
            if hours > 0:
                return hours
    return 1.0


def detect_electric_buses(n: pypsa.Network):
    """Erkenne elektrische Bus-Carrier robust (lowercase)."""
    if "carrier" in n.buses.columns:
        carriers = n.buses["carrier"].dropna().astype(str).str.lower().unique()
    else:
        carriers = []
    electric_candidates = set(carriers).intersection(
        {"electricity", "ac", "dc", "low voltage", "lv", "hv"}
    )
    if electric_candidates:
        return set(electric_candidates)
    return set(c for c in carriers if "electr" in c)


def bus_country_map(n: pypsa.Network):
    """Gebe Serie bus -> country falls vorhanden, ansonsten None."""
    if "country" in n.buses.columns:
        return n.buses["country"].astype(str)
    return None


def power_series_sum(df: pd.DataFrame | None) -> pd.Series:
    """Sichere Summe über Achse=1 für evtl. leere DataFrames."""
    if df is None or df.empty:
        return pd.Series(0.0, index=df.index if df is not None else [])
    return df.sum(axis=1).astype(float)


def _select_timeseries_attr(t_obj, attr_names):
    """
    Wähle das erste vorhandene, nicht-leere DataFrame-Attribut aus t_obj in der
    Reihenfolge attr_names und gib es zurück. Falls keines passt, gib None zurück.
    """
    if t_obj is None:
        return None
    for name in attr_names:
        if hasattr(t_obj, name):
            attr = getattr(t_obj, name)
            # Pandas DataFrame
            if isinstance(attr, pd.DataFrame) and not attr.empty:
                return attr
            # xarray DataArray (optional)
            try:
                import xarray as xr
                if isinstance(attr, xr.DataArray) and attr.size > 0:
                    # Konvertiere zu DataFrame (time x columns)
                    try:
                        df = attr.to_pandas()
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            return df
                    except Exception:
                        pass
            except Exception:
                pass
    return None

# =====================================================================
# --- Nettoimporte ---
# =====================================================================
def get_net_imports_fixed(n: pypsa.Network, country: str) -> pd.Series:
    """Nettoimporte als Zeitreihe (MW). Liefert Series indexed by snapshots."""
    snaps = n.snapshots
    net_imports = pd.Series(0.0, index=snaps, dtype=float)
    if country == "ALL":
        return net_imports

    bus_country = bus_country_map(n)
    if not hasattr(n, "links") or n.links.empty:
        return net_imports

    links = n.links.copy()
    if bus_country is not None:
        links = links.join(bus_country, on="bus0").rename(columns={"country": "bus0_country"})
        links = links.join(bus_country, on="bus1").rename(columns={"country": "bus1_country"})
        trade_links = links[(links["bus0_country"] == country) ^ (links["bus1_country"] == country)]
    else:
        trade_links = links[(links.bus0.astype(str).str.startswith(country)) ^ (links.bus1.astype(str).str.startswith(country))]

    p0df = _select_timeseries_attr(getattr(n, "links_t", None), ["p0", "p"])
    # p1 may be separate
    p1df = _select_timeseries_attr(getattr(n, "links_t", None), ["p1", "p"])

    for name, row in trade_links.iterrows():
        # choose p0 as canonical flow bus0 -> bus1
        if p0df is None:
            continue
        if name not in p0df.columns:
            continue
        series_p0 = p0df[name]
        if bus_country is not None:
            if row["bus1_country"] == country:
                net_imports = net_imports.add(series_p0, fill_value=0.0)
            else:
                net_imports = net_imports.subtract(series_p0, fill_value=0.0)
        else:
            if row.bus1.startswith(country):
                net_imports = net_imports.add(series_p0, fill_value=0.0)
            else:
                net_imports = net_imports.subtract(series_p0, fill_value=0.0)
    return net_imports


# =====================================================================
# --- Erzeugung ---
# =====================================================================
def get_total_generation_with_imports(n: pypsa.Network, country: str):
    """
    Gibt Zeitreihe der Erzeugungsleistung (MW) und Komponenten-Summen zurück.
    gen_sum: pd.Series (MW)
    gen_components: dict mit Summen (in MWh-equivalent: placeholder keys)
    """
    snaps = n.snapshots
    gen_sum = pd.Series(0.0, index=snaps, dtype=float)
    components = {}

    electric_carriers = detect_electric_buses(n)
    if "carrier" in n.buses.columns:
        electric_buses = n.buses.index[n.buses["carrier"].astype(str).str.lower().isin(electric_carriers)]
    else:
        electric_buses = n.buses.index

    # --- GENERATORS ---
    if hasattr(n, "generators") and not n.generators.empty:
        gens = n.generators if country == "ALL" else n.generators[n.generators.bus.str.startswith(country)]
        gens = gens[gens.bus.isin(electric_buses)]
        p_attr = _select_timeseries_attr(getattr(n, "generators_t", None), ["p", "p_dispatch", "p_set"])
        if p_attr is not None:
            available = [c for c in gens.index if c in p_attr.columns]
            if available:
                gen_series = p_attr[available].sum(axis=1)
                gen_sum = gen_sum.add(gen_series, fill_value=0.0)
                components["Generators_MW_sum"] = gen_series.sum()
        else:
            components["Generators_no_timeseries_p_nom_MW"] = gens["p_nom"].sum() if "p_nom" in gens.columns else 0.0

    # --- LINKS: Beiträge (z.B. P2G Entladung) ---
    if hasattr(n, "links") and not n.links.empty:
        links_scope = n.links if country == "ALL" else n.links[
            n.links.bus0.str.startswith(country) | n.links.bus1.str.startswith(country)
        ]
        p0df = _select_timeseries_attr(getattr(n, "links_t", None), ["p0", "p"])
        p1df = _select_timeseries_attr(getattr(n, "links_t", None), ["p1", "p"])
        link_gen = {}
        for name, row in links_scope.iterrows():
            b0, b1 = row.bus0, row.bus1
            if b0 not in n.buses.index or b1 not in n.buses.index:
                continue
            c0 = str(n.buses.at[b0, "carrier"]) if "carrier" in n.buses.columns else ""
            c1 = str(n.buses.at[b1, "carrier"]) if "carrier" in n.buses.columns else ""
            c0l, c1l = c0.lower(), c1.lower()
            contribution = pd.Series(0.0, index=snaps)
            # Entladung in elektr. Seite: positive flow into electric bus
            if (c1l in electric_carriers) and (c0l not in electric_carriers):
                if p0df is not None and name in p0df.columns:
                    contribution = p0df[name].clip(lower=0)
            elif (c0l in electric_carriers) and (c1l not in electric_carriers):
                if p1df is not None and name in p1df.columns:
                    contribution = p1df[name].clip(lower=0)
            if contribution.sum() > 0:
                gen_sum = gen_sum.add(contribution, fill_value=0.0)
                carrier_key = (row.carrier.split()[0] if isinstance(row.carrier, str) else "link")
                link_gen[carrier_key] = link_gen.get(carrier_key, 0.0) + contribution.sum()
        components["Links_MW_sum"] = link_gen

    # --- StorageUnits (Discharge) ---
    if hasattr(n, "storage_units") and not n.storage_units.empty:
        sus = n.storage_units if country == "ALL" else n.storage_units[n.storage_units.bus.str.startswith(country)]
        sus = sus[sus.bus.isin(electric_buses)]
        p_attr = _select_timeseries_attr(getattr(n, "storage_units_t", None), ["p", "p_dispatch"])
        if p_attr is not None:
            available = [c for c in sus.index if c in p_attr.columns]
            if available:
                discharge = p_attr[available].clip(lower=0).sum(axis=1)
                gen_sum = gen_sum.add(discharge, fill_value=0.0)
                components["StorageUnits_discharge_MW_sum"] = discharge.sum()

    # --- Stores (discharge) ---
    if hasattr(n, "stores") and not n.stores.empty:
        if hasattr(n, "stores_t"):
            stores = n.stores if country == "ALL" else n.stores[n.stores.bus.str.startswith(country)]
            stores = stores[stores.bus.isin(electric_buses)]
            p_attr = _select_timeseries_attr(getattr(n, "stores_t", None), ["p"])
            if p_attr is not None:
                available = [c for c in stores.index if c in p_attr.columns]
                if available:
                    discharge = (-p_attr[available]).clip(lower=0).sum(axis=1)
                    gen_sum = gen_sum.add(discharge, fill_value=0.0)
                    components["Stores_discharge_MW_sum"] = discharge.sum()

    # --- Net Imports ---
    if country != "ALL":
        net_imports = get_net_imports_fixed(n, country)
        gen_sum = gen_sum.add(net_imports, fill_value=0.0)
        components["NetImports_MW_sum"] = net_imports.sum()

    return gen_sum, components

# =====================================================================
# --- Verbrauch (inkl. korrekter Verlustzuweisung) ---
# =====================================================================
def get_total_consumption(n: pypsa.Network, country: str):
    """Gibt Zeitreihe der Leistungsaufnahme (MW) zurück (cons_sum) und gibt debug-prints."""
    snaps = n.snapshots
    cons_sum = pd.Series(0.0, index=snaps, dtype=float)

    electric_carriers = detect_electric_buses(n)
    if "carrier" in n.buses.columns:
        electric_buses = n.buses.index[n.buses["carrier"].astype(str).str.lower().isin(electric_carriers)]
    else:
        electric_buses = n.buses.index

    # --- Loads ---
    if hasattr(n, "loads") and not n.loads.empty and hasattr(n, "loads_t"):
        loads = n.loads if country == "ALL" else n.loads[n.loads.bus.str.startswith(country)]
        loads = loads[loads.bus.isin(electric_buses)]
        p_attr = _select_timeseries_attr(getattr(n, "loads_t", None), ["p"])
        if p_attr is not None:
            available = [c for c in loads.index if c in p_attr.columns]
            if available:
                load_ts = p_attr[available].sum(axis=1)
                cons_sum = cons_sum.add(load_ts, fill_value=0.0)

    # --- Links: Verbrauch (z.B. G2P, Laden) ---
    if hasattr(n, "links") and not n.links.empty and hasattr(n, "links_t"):
        links_scope = n.links if country == "ALL" else n.links[
            n.links.bus0.str.startswith(country) | n.links.bus1.str.startswith(country)
        ]
        p0df = _select_timeseries_attr(getattr(n, "links_t", None), ["p0", "p"])
        p1df = _select_timeseries_attr(getattr(n, "links_t", None), ["p1", "p"])
        link_cons = {}
        for name, row in links_scope.iterrows():
            b0, b1 = row.bus0, row.bus1
            if b0 not in n.buses.index or b1 not in n.buses.index:
                continue
            c0 = str(n.buses.at[b0, "carrier"]) if "carrier" in n.buses.columns else ""
            c1 = str(n.buses.at[b1, "carrier"]) if "carrier" in n.buses.columns else ""
            c0l, c1l = c0.lower(), c1.lower()
            consumed = pd.Series(0.0, index=snaps)
            if (c0l in electric_carriers) and (c1l not in electric_carriers):
                if p0df is not None and name in p0df.columns:
                    consumed = p0df[name].clip(lower=0)
            elif (c1l in electric_carriers) and (c0l not in electric_carriers):
                if p1df is not None and name in p1df.columns:
                    consumed = p1df[name].clip(lower=0)
            if consumed.sum() > 0:
                cons_sum = cons_sum.add(consumed, fill_value=0.0)
                carrier_key = (row.carrier.split()[0] if isinstance(row.carrier, str) else "link")
                link_cons[carrier_key] = link_cons.get(carrier_key, 0.0) + consumed.sum()

    # --- StorageUnits charging ---
    if hasattr(n, "storage_units") and not n.storage_units.empty and hasattr(n, "storage_units_t"):
        sus = n.storage_units if country == "ALL" else n.storage_units[n.storage_units.bus.str.startswith(country)]
        sus = sus[sus.bus.isin(electric_buses)]
        p_attr = _select_timeseries_attr(getattr(n, "storage_units_t", None), ["p", "p_set"])
        if p_attr is not None:
            available = [c for c in sus.index if c in p_attr.columns]
            if available:
                charge = (-p_attr[available]).clip(lower=0).sum(axis=1)
                cons_sum = cons_sum.add(charge, fill_value=0.0)

    # --- Stores charging ---
    if hasattr(n, "stores") and not n.stores.empty and hasattr(n, "stores_t"):
        stores = n.stores if country == "ALL" else n.stores[n.stores.bus.str.startswith(country)]
        stores = stores[stores.bus.isin(electric_buses)]
        p_attr = _select_timeseries_attr(getattr(n, "stores_t", None), ["p"])
        if p_attr is not None:
            available = [c for c in stores.index if c in p_attr.columns]
            if available:
                charge = p_attr[available].clip(lower=0).sum(axis=1)
                cons_sum = cons_sum.add(charge, fill_value=0.0)

    # --- Netzverluste (Lines + DC-Links/Transformers) ---
    grid_losses_series = pd.Series(0.0, index=snaps, dtype=float)
    bus_country = bus_country_map(n)

    # Lines (AC)
    if hasattr(n, "lines") and not n.lines.empty and hasattr(n, "lines_t"):
        p0df = _select_timeseries_attr(getattr(n, "lines_t", None), ["p0", "p"])
        p1df = _select_timeseries_attr(getattr(n, "lines_t", None), ["p1", "p"])
        for name, line in n.lines.iterrows():
            if p0df is None or p1df is None:
                continue
            if name not in p0df.columns or name not in p1df.columns:
                continue
            losses = p0df[name].fillna(0) + p1df[name].fillna(0)
            if bus_country is not None:
                country0 = bus_country.get(line.bus0, "")
                country1 = bus_country.get(line.bus1, "")
            else:
                country0 = str(line.bus0)[:2]
                country1 = str(line.bus1)[:2]
            if country == "ALL":
                grid_losses_series = grid_losses_series.add(losses, fill_value=0.0)
            elif country0 == country and country1 == country:
                grid_losses_series = grid_losses_series.add(losses, fill_value=0.0)
            elif (country0 == country) ^ (country1 == country):
                grid_losses_series = grid_losses_series.add(0.5 * losses, fill_value=0.0)

    # Links (DC/transformer)
    if hasattr(n, "links") and not n.links.empty and hasattr(n, "links_t"):
        p0df = _select_timeseries_attr(getattr(n, "links_t", None), ["p0", "p"])
        p1df = _select_timeseries_attr(getattr(n, "links_t", None), ["p1", "p"])
        for name, link in n.links.iterrows():
            if p0df is None or p1df is None:
                continue
            if name not in p0df.columns or name not in p1df.columns:
                continue
            b0, b1 = link.bus0, link.bus1
            if b0 not in n.buses.index or b1 not in n.buses.index:
                continue
            c0 = str(n.buses.at[b0, "carrier"]) if "carrier" in n.buses.columns else ""
            c1 = str(n.buses.at[b1, "carrier"]) if "carrier" in n.buses.columns else ""
            if c0.lower() in electric_carriers and c1.lower() in electric_carriers:
                losses = p0df[name].fillna(0) + p1df[name].fillna(0)
                if bus_country is not None:
                    country0 = bus_country.get(b0, "")
                    country1 = bus_country.get(b1, "")
                else:
                    country0 = str(b0)[:2]
                    country1 = str(b1)[:2]
                if country == "ALL":
                    grid_losses_series = grid_losses_series.add(losses, fill_value=0.0)
                elif country0 == country and country1 == country:
                    grid_losses_series = grid_losses_series.add(losses, fill_value=0.0)
                elif (country0 == country) ^ (country1 == country):
                    grid_losses_series = grid_losses_series.add(0.5 * losses, fill_value=0.0)

    cons_sum = cons_sum.add(grid_losses_series, fill_value=0.0)

    # Debug-Ausgabe (korrekte Umrechnung in GWh)
    hps = hours_per_snapshot(n)
    def to_gwh(series_mw: pd.Series) -> float:
        return (series_mw.sum() * hps) / 1e3

    print(f"\n🔍 Verbrauchskomponenten für {'ALL' if country=='ALL' else country}:")
    print(f"   Gesamt Verbrauch (inkl. Verluste): {to_gwh(cons_sum):12.2f} GWh")

    return cons_sum

# =====================================================================
# --- Plot & Hauptablauf ---
# =====================================================================
def plot_simple_balance(gen_mw: pd.Series, cons_mw: pd.Series, year: int, country: str, config: PlottingConfig, network_path: str):
    title_country = "Gesamtnetz" if country == "ALL" else country
    network_name = os.path.basename(os.path.dirname(os.path.dirname(network_path)))
    save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "balance_timeline_simple_imports")
    os.makedirs(save_dir, exist_ok=True)

    # Detailansicht
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(gen_mw.loc[DETAIL_START:DETAIL_END].index, gen_mw.loc[DETAIL_START:DETAIL_END],
                    color="#74a9cf", alpha=0.7, label="Erzeugung inkl. Import (MW)")
    ax.plot(cons_mw.loc[DETAIL_START:DETAIL_END].index, cons_mw.loc[DETAIL_START:DETAIL_END],
            color="black", linewidth=1.3, label="Verbrauch inkl. Verluste (MW)")
    ax.set_title(f"Strombilanz (Detail) – {title_country} ({year})", fontsize=config.FONT_SIZES.get("title", 14))
    ax.legend()
    plt.savefig(os.path.join(save_dir, f"balance_simple_imports_detail_{country}_{year}.png"), dpi=300)
    plt.close(fig)

    # Jahresverlauf: resample daily mean (MW)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(gen_mw.resample("D").mean().index, gen_mw.resample("D").mean(),
                    color="#74a9cf", alpha=0.7, label="Erzeugung inkl. Import (MW)")
    ax.plot(cons_mw.resample("D").mean().index, cons_mw.resample("D").mean(), color="black", linewidth=1.3,
            label="Verbrauch inkl. Verluste (MW)")
    ax.set_title(f"Strombilanz (Jahresverlauf) – {title_country} ({year})", fontsize=config.FONT_SIZES.get("title", 14))
    ax.legend()
    plt.savefig(os.path.join(save_dir, f"balance_simple_imports_year_{country}_{year}.png"), dpi=300)
    plt.close(fig)
    print(f"✅ Plots für {country} ({year}) gespeichert.")


def main():
    config = PlottingConfig()
    networks = config.get_networks()

    for path in networks:
        if not os.path.isfile(path):
            continue
        m = re.search(r"_(\d{4})\.nc$", path)
        if not m:
            print(f"Warnung: Jahr nicht im Dateinamen erkannt: {path}")
            continue
        year = int(m.group(1))

        print(f"\n📂 Lade Netzwerk {year}: {path}")
        n = pypsa.Network(path)

        for country in config.get_countries():
            print(f"\n{'=' * 60}\n🌍 Analysiere {country} ({year})\n{'=' * 60}")
            gen_mw, gen_components = get_total_generation_with_imports(n, country)
            cons_mw = get_total_consumption(n, country)

            # Energie-Summen in GWh (korrekt)
            hps = hours_per_snapshot(n)
            total_gen_gwh = (gen_mw.sum() * hps) / 1e3
            total_cons_gwh = (cons_mw.sum() * hps) / 1e3

            print(f"\n📊 ERZEUGUNG - Detailansicht für {country}:")
            print(f"   GESAMT ERZEUGUNG:        {total_gen_gwh:12.2f} GWh")
            print(f"   GESAMT VERBRAUCH:        {total_cons_gwh:12.2f} GWh")

            plot_simple_balance(gen_mw, cons_mw, year, country, config, path)

            balance_gwh = total_gen_gwh - total_cons_gwh
            print(f"\n⚖️ Bilanzcheck für {country} ({year}): Differenz = {balance_gwh:.2f} GWh")
            if abs(balance_gwh) < 5.0:
                print("✅ Bilanz geschlossen\n")
            else:
                print("⚠️ Bilanzabweichung – bitte prüfen\n")


if __name__ == "__main__":
    main()