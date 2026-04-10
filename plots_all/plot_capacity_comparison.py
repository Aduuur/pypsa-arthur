#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_capacity_comparison.py
============================
Vergleich installierter elektrischer Kapazitäten:
  Linke Balken  = Robustes Portfolio  (ARO-Ergebnis)
  Rechte Balken = Basisjahr-Szenario  (deterministischer Referenz-Run)

Schema wie im Bild:
  x-Achse = Jahre (alle vorhandenen Planungsjahre)
  Je Jahr  = zwei gestapelte Balken nebeneinander (links solid, rechts schraffiert)
  Carrier  = gestapelt von unten (Grundlast) nach oben (Speicher)

Output-Struktur:
  <out_dir>/
    ALL_capacity_comparison.png     ← Europa gesamt
    DE_capacity_comparison.png      ← Deutschland
    FR_capacity_comparison.png      ← ...
    ...

Aufruf (standalone):
  python plot_capacity_comparison.py

Aufruf (aus run_analysis.py / aro_analysis.py):
  from plot_capacity_comparison import run_capacity_comparison
  run_capacity_comparison(
      networks_robust=["/pfad/aro_robust__std.nc"],
      networks_basis=["/pfad/base_s_24___2050.nc"],
      out_dir=Path("/pfad/plots/scenario_comparison"),
      label_robust="Robustes Portfolio",
      label_basis="Basisjahr",
  )
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pypsa

# ---------------------------------------------------------------------------
# Carrier-Definition (nur Strom-Sektor)
# ---------------------------------------------------------------------------

# Reihenfolge von unten (Grundlast) nach oben (Speicher) — genau wie im Bild
CARRIER_ORDER = [
    "nuclear",
    "solar", "solar rooftop", "solar-hsat",
    "onwind",
    "offwind-ac", "offwind-dc", "offwind-float",
    "ror", "hydro",
    "biomass", "biogas",
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
    "lignite", "coal", "oil",
    "CCGT", "OCGT",
    "H2 turbine", "H2 OCGT", "H2 Fuel Cell",
    "PHS",
    "battery discharger",
    "home battery discharger",
]

# Deutsche Anzeigenamen
CARRIER_LABELS: Dict[str, str] = {
    "nuclear":                             "Kernkraft",
    "solar":                               "Photovoltaik",
    "solar rooftop":                       "PV Dach",
    "solar-hsat":                          "PV HSAT",
    "onwind":                              "Wind Onshore",
    "offwind-ac":                          "Wind Offshore (AC)",
    "offwind-dc":                          "Wind Offshore (DC)",
    "offwind-float":                       "Wind Offshore (Float)",
    "ror":                                 "Laufwasser",
    "hydro":                               "Wasserkraft",
    "biomass":                             "Biomasse",
    "biogas":                              "Biogas",
    "urban central solid biomass CHP":     "Biomasse KWK",
    "urban central solid biomass CHP CC":  "Biomasse KWK CC",
    "lignite":                             "Braunkohle",
    "coal":                                "Steinkohle",
    "oil":                                 "Öl",
    "CCGT":                                "Erdgas (GuD)",
    "OCGT":                                "Erdgas (Gasturbine)",
    "H2 turbine":                          "H2-Turbine",
    "H2 OCGT":                             "H2-Gasturbine",
    "H2 Fuel Cell":                        "Brennstoffzelle",
    "PHS":                                 "Pumpspeicher",
    "battery discharger":                  "Batteriespeicher",
    "home battery discharger":             "Heimspeicher",
}

# Farben passend zum Bild
CARRIER_COLORS: Dict[str, str] = {
    "nuclear":                             "#ff8c00",
    "solar":                               "#f9d002",
    "solar rooftop":                       "#ffea80",
    "solar-hsat":                          "#FFF080",
    "onwind":                              "#235ebc",
    "offwind-ac":                          "#6895dd",
    "offwind-dc":                          "#74c6f2",
    "offwind-float":                       "#15a0bf",
    "ror":                                 "#3dbfb0",
    "hydro":                               "#298c81",
    "biomass":                             "#baa741",
    "biogas":                              "#e3d37d",
    "urban central solid biomass CHP":     "#a09030",
    "urban central solid biomass CHP CC":  "#8a7a28",
    "lignite":                             "#826837",
    "coal":                                "#545454",
    "oil":                                 "#c9c9c9",
    "CCGT":                                "#a85522",
    "OCGT":                                "#e0986c",
    "H2 turbine":                          "#991f83",
    "H2 OCGT":                             "#c251ae",
    "H2 Fuel Cell":                        "#bf13a0",
    "PHS":                                 "#51dbcc",
    "battery discharger":                  "#ace37f",
    "home battery discharger":             "#80c944",
}

# Carrier die als Links modelliert sind (elektrische Einspeisung über bus1)
LINK_ELECTRICITY_CARRIERS = {
    "OCGT", "CCGT", "oil", "nuclear",
    "H2 Fuel Cell", "H2 turbine", "H2 OCGT",
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
}

# Strings die auf Nicht-Strom-Busse hinweisen
NON_ELECTRICITY_BUS_KW = {"heat", "H2", "gas", "biogas", "co2", "CO2",
                           "biomass", "methanol", "ammonia"}

COUNTRY_NAMES: Dict[str, str] = {
    "ALL": "Europa (Gesamt)",
    "DE": "Deutschland",    "FR": "Frankreich",   "ES": "Spanien",
    "IT": "Italien",        "GB": "Großbritannien","PL": "Polen",
    "SE": "Schweden",       "NO": "Norwegen",      "NL": "Niederlande",
    "BE": "Belgien",        "AT": "Österreich",    "CH": "Schweiz",
    "DK": "Dänemark",       "CZ": "Tschechien",    "PT": "Portugal",
    "FI": "Finnland",       "LT": "Litauen",       "LV": "Lettland",
    "EE": "Estland",
}

# Schriftgrößen
FS = {
    "title":     20,
    "axis":      15,
    "tick":      13,
    "bar_label": 10,
    "legend":    11,
    "leg_title": 12,
}


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _year_from_network(n: pypsa.Network, path: str) -> Optional[int]:
    try:
        return int(n.snapshots[0].year)
    except Exception:
        pass
    m = re.search(r"_(\d{4})\.nc$", path)
    return int(m.group(1)) if m else None


def _is_elec_bus(bus: str) -> bool:
    bl = bus.lower()
    return not any(kw.lower() in bl for kw in NON_ELECTRICITY_BUS_KW)


def _valid_countries(n: pypsa.Network) -> set:
    return {b[:2] for b in n.buses.index
            if len(b) >= 2 and b[:2].isalpha() and b[:2].isupper() and b[:2] != "EU"}


def _bus_mask(n: pypsa.Network, country: str) -> pd.Index:
    """Bus-Indizes die zum Land (oder ganz Europa) gehören."""
    if country == "ALL":
        vc = _valid_countries(n)
        return n.buses.index[n.buses.index.str[:2].isin(vc)]
    return n.buses.index[n.buses.index.str.startswith(country)]


# ---------------------------------------------------------------------------
# Kapazitäts-Extraktion
# ---------------------------------------------------------------------------

def _cap_col(df: pd.DataFrame) -> Optional[str]:
    if "p_nom_opt" in df.columns:
        return "p_nom_opt"
    if "p_nom" in df.columns and df["p_nom"].sum() > 0:
        return "p_nom"
    return None


def _from_generators(n: pypsa.Network, buses: pd.Index) -> pd.Series:
    gens = n.generators[n.generators.bus.isin(buses)].copy()
    gens = gens[gens.carrier.isin(set(CARRIER_ORDER) - LINK_ELECTRICITY_CARRIERS)]
    col = _cap_col(gens)
    if col is None or gens.empty:
        return pd.Series(dtype=float)

    # Solar aggregieren
    raw = gens.groupby("carrier")[col].sum() / 1e3
    return raw[raw > 0.01]


def _from_storage_units(n: pypsa.Network, buses: pd.Index) -> pd.Series:
    if n.storage_units.empty:
        return pd.Series(dtype=float)
    sus = n.storage_units.copy()
    # FIX: carrier-Spalte prüfen bevor gefiltert wird
    if "carrier" not in sus.columns:
        return pd.Series(dtype=float)
    if "bus" not in sus.columns:
        return pd.Series(dtype=float)
    sus = sus[sus.bus.isin(buses)]
    sus = sus[~sus.bus.str.contains("EU", case=False, na=False)]
    sus = sus[sus.bus.apply(_is_elec_bus)]
    if sus.empty or "carrier" not in sus.columns:
        return pd.Series(dtype=float)
    sus = sus[sus["carrier"].isin(set(CARRIER_ORDER))]
    col = _cap_col(sus)
    if col is None or sus.empty:
        return pd.Series(dtype=float)
    raw = sus.groupby("carrier")[col].sum() / 1e3
    return raw[raw > 0.01]


def _from_links(n: pypsa.Network, buses: pd.Index) -> pd.Series:
    lks = n.links.copy()
    lks = lks[lks.carrier.isin(LINK_ELECTRICITY_CARRIERS)]
    lks = lks[lks.bus1.isin(buses) | lks.bus1.str[:2].isin(
        {b[:2] for b in buses} if len(buses) < 5000 else {"--"}
    )]
    lks = lks[lks.bus1.apply(_is_elec_bus)]
    col = _cap_col(lks)
    if col is None or lks.empty:
        return pd.Series(dtype=float)

    cap = lks[col].copy()
    if "efficiency" in lks.columns:
        eff = lks["efficiency"].fillna(1.0)
        mask = (eff > 0.01) & (eff < 0.99)
        cap[mask] = cap[mask] * eff[mask]

    lks = lks.copy()
    lks["_c"] = cap / 1e3
    raw = lks.groupby("carrier")["_c"].sum()
    return raw[raw > 0.01]


def get_capacity(n: pypsa.Network, country: str) -> pd.Series:
    """
    Installierte elektrische Kapazität [GW] für ein Land oder Europa gesamt.
    Kombiniert Generatoren + Storage Units + Links (nur Strom-Sektor).
    """
    buses = _bus_mask(n, country)
    parts = [
        _from_generators(n, buses),
        _from_storage_units(n, buses),
        _from_links(n, buses),
    ]
    non_empty = [p for p in parts if not p.empty]
    if not non_empty:
        return pd.Series(dtype=float)

    combined = pd.concat(non_empty).groupby(level=0).sum()

    # Solar-Varianten zusammenfassen → "solar"
    solar_sum = combined.filter(regex="^solar").sum()
    combined = combined.drop(combined.index[combined.index.str.startswith("solar")],
                             errors="ignore")
    if solar_sum > 0.01:
        combined["solar"] = solar_sum

    # Biomasse zusammenfassen → "biomass"
    bio_sum = combined.filter(regex="biomass|biogas").sum()
    combined = combined.drop(
        combined.index[combined.index.str.contains("biomass|biogas")], errors="ignore"
    )
    if bio_sum > 0.01:
        combined["biomass"] = bio_sum

    return combined[combined > 0.01]


# ---------------------------------------------------------------------------
# Plot-Funktion (ein Land, alle Jahre)
# ---------------------------------------------------------------------------

def plot_comparison(
    data_robust: Dict[int, pd.Series],   # {year: Series[carrier -> GW]}
    data_basis:  Dict[int, pd.Series],
    years: List[int],
    country: str,
    out_path: Path,
    label_robust: str = "Robustes Portfolio",
    label_basis:  str = "Basisjahr",
    carrier_colors: Optional[Dict[str, str]] = None,
    dpi: int = 300,
) -> None:
    """
    Erstellt den Vergleichsplot nach dem Schema im Bild:
      - x-Achse: Jahre
      - Je Jahr: linker Balken (robust, solid) + rechter Balken (basis, schraffiert)
      - Carrier gestapelt, beschriftet wenn groß genug
    """
    colors = {**CARRIER_COLORS, **(carrier_colors or {})}
    c_name = COUNTRY_NAMES.get(country, country)

    # Alle vorkommenden Carrier (geordnet)
    all_carr_set = set()
    for d in [data_robust, data_basis]:
        for s in d.values():
            all_carr_set.update(s.index)

    carriers = [c for c in CARRIER_ORDER if c in all_carr_set]
    carriers += sorted(c for c in all_carr_set if c not in carriers)

    # Minimale Anzeige-Schwelle (1% des größten Balkens)
    _vals_robust = [sum(data_robust.get(y, pd.Series()).get(c, 0) for c in carriers) for y in years]
    _vals_basis  = [sum(data_basis.get(y, pd.Series()).get(c, 0) for c in carriers) for y in years]
    _all_vals = _vals_robust + _vals_basis
    total_max = max(_all_vals) if _all_vals else 1.0
    TEXT_THR = 0.025 * total_max   # Beschriftung ab 2,5% der Gesamthöhe

    n_years = len(years)
    fig_w   = max(14, n_years * 2.5 + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 10))

    x     = np.arange(n_years)
    width = 0.35   # Balkenbreite (wie im Bild)

    # Stapel aufbauen
    bottom_r = np.zeros(n_years)
    bottom_b = np.zeros(n_years)

    legend_handles: List[mpatches.Patch] = []

    for carrier in carriers:
        vals_r = np.array([data_robust.get(y, pd.Series()).get(carrier, 0.0)
                           for y in years])
        vals_b = np.array([data_basis.get(y, pd.Series()).get(carrier, 0.0)
                           for y in years])

        if vals_r.sum() < 0.01 and vals_b.sum() < 0.01:
            continue

        color = colors.get(carrier, "#a9a9a9")
        label = CARRIER_LABELS.get(carrier, carrier)

        # Linke Balken (robust, solid)
        ax.bar(x - width / 2, vals_r, width,
               bottom=bottom_r, color=color,
               edgecolor="white", linewidth=0.5)

        # Rechte Balken (basis, schraffiert wie im Bild)
        ax.bar(x + width / 2, vals_b, width,
               bottom=bottom_b, color=color,
               edgecolor="white", linewidth=0.5,
               hatch="///", alpha=0.92)

        # Beschriftungen
        for i in range(n_years):
            if vals_r[i] >= TEXT_THR:
                ax.text(x[i] - width / 2,
                        bottom_r[i] + vals_r[i] / 2,
                        f"{int(round(vals_r[i]))}",
                        ha="center", va="center",
                        color="white", fontsize=FS["bar_label"],
                        fontweight="bold")
            if vals_b[i] >= TEXT_THR:
                ax.text(x[i] + width / 2,
                        bottom_b[i] + vals_b[i] / 2,
                        f"{int(round(vals_b[i]))}",
                        ha="center", va="center",
                        color="white", fontsize=FS["bar_label"],
                        fontweight="bold")

        bottom_r += vals_r
        bottom_b += vals_b

        legend_handles.append(mpatches.Patch(facecolor=color, label=label))

    # Achsen & Titel
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=FS["tick"])
    ax.set_xlabel("Jahr", fontsize=FS["axis"])
    ax.set_ylabel("Kapazität (GW)", fontsize=FS["axis"])
    ax.tick_params(axis="y", labelsize=FS["tick"])
    ax.set_title(
        f"Installierte Kapazität: {c_name}\n"
        f"(Links: {label_robust} | Rechts: {label_basis})",
        fontsize=FS["title"], pad=16,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Legende (Carrier, umgekehrte Reihenfolge = oben = Speicher)
    ax.legend(
        handles=legend_handles[::-1],
        title="Technologie",
        fontsize=FS["legend"],
        title_fontsize=FS["leg_title"],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {out_path.name}")


# ---------------------------------------------------------------------------
# Haupt-API
# ---------------------------------------------------------------------------

def run_capacity_comparison(
    networks_robust: List[str],
    networks_basis:  List[str],
    out_dir: Path,
    countries: Optional[List[str]] = None,
    label_robust: str = "Robustes Portfolio",
    label_basis:  str = "Basisjahr",
    carrier_colors: Optional[Dict[str, str]] = None,
    dpi: int = 300,
) -> None:
    """
    Lädt die Netzwerke und erstellt Vergleichsplots für alle angegebenen Länder.

    Parameters
    ----------
    networks_robust : Liste von .nc-Pfaden des robusten Portfolios
                      (bei myopischen Runs mehrere Jahre)
    networks_basis  : Liste von .nc-Pfaden des Basisszenarios
    out_dir         : Ausgabeordner
    countries       : Länderliste; None = alle im Netz vorhandenen + ALL
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Netzwerke laden ----
    def load_nets(paths: List[str]) -> Dict[int, pypsa.Network]:
        nets: Dict[int, pypsa.Network] = {}
        for p in paths:
            if not Path(p).is_file():
                print(f"  ⚠️  Datei nicht gefunden: {p}")
                continue
            try:
                n = pypsa.Network(p)
                year = _year_from_network(n, p)
                if year is None:
                    print(f"  ⚠️  Jahr nicht erkennbar: {p}")
                    continue
                nets[year] = n
                print(f"  📂 Geladen: {Path(p).name}  → Jahr {year}")
            except Exception as e:
                print(f"  ❌ Ladefehler {p}: {e}")
        return nets

    print("\n[1/2] Lade robuste Netze...")
    nets_r = load_nets(networks_robust)
    print("[2/2] Lade Basis-Netze...")
    nets_b = load_nets(networks_basis)

    if not nets_r:
        print("❌ Keine robusten Netzwerke geladen.")
        return
    if not nets_b:
        print("❌ Keine Basis-Netzwerke geladen.")
        return

    # ---- Gemeinsame Jahre + Länderliste ----
    years_r = sorted(nets_r.keys())
    years_b = sorted(nets_b.keys())
    # Gemeinsame Jahre bevorzugen; wenn leer, alle Jahre nebeneinander
    common_years = sorted(set(years_r) & set(years_b))
    all_years    = sorted(set(years_r) | set(years_b))
    years = common_years if common_years else all_years

    if not common_years:
        print(f"  ⚠️  Keine gemeinsamen Jahre (Robust: {years_r}, Basis: {years_b}).")
        print(f"       Plotte alle verfügbaren Jahre: {all_years}")

    # Länder aus dem ersten verfügbaren Netzwerk ermitteln
    ref_net = next(iter(nets_r.values()))
    vc = _valid_countries(ref_net)
    if countries is None:
        plot_countries = ["ALL"] + sorted(vc)
    else:
        plot_countries = countries

    # ---- Kapazitäten extrahieren ----
    print(f"\n📊 Extrahiere Kapazitäten für {len(plot_countries)} Länder × "
          f"{len(years)} Jahre...")

    data_robust: Dict[str, Dict[int, pd.Series]] = {c: {} for c in plot_countries}
    data_basis:  Dict[str, Dict[int, pd.Series]] = {c: {} for c in plot_countries}

    for year, n in nets_r.items():
        for c in plot_countries:
            data_robust[c][year] = get_capacity(n, c)

    for year, n in nets_b.items():
        for c in plot_countries:
            data_basis[c][year] = get_capacity(n, c)

    # ---- Plots erstellen ----
    print(f"\n🎨 Erstelle Plots → {out_dir}")
    for country in plot_countries:
        fname = f"{country}_capacity_comparison.png"
        plot_comparison(
            data_robust=data_robust[country],
            data_basis=data_basis[country],
            years=years,
            country=country,
            out_path=out_dir / fname,
            label_robust=label_robust,
            label_basis=label_basis,
            carrier_colors=carrier_colors,
            dpi=dpi,
        )

    print(f"\n✅ Fertig. {len(plot_countries)} Plots in: {out_dir}")


# ---------------------------------------------------------------------------
# Standalone main() — über master_config konfiguriert
# ---------------------------------------------------------------------------

def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from master_config import MasterConfig, AROPlottingConfig

    master  = MasterConfig()
    aro_cfg = AROPlottingConfig(master=master)

    run_conf = aro_cfg.get_current_run_config()

    # Robustes Netz — bevorzugt robust_network_std, dann robust_network
    robust_path = (
        run_conf.get("robust_network_std")
        or run_conf.get("robust_network")
    )
    if not robust_path or not Path(robust_path).is_file():
        print(f"❌ Robustes Netz nicht gefunden: {robust_path}")
        print("   Tipp: run_conf['robust_network'] in master_config.py setzen.")
        return

    # Basis-Netz — Referenznetzwerk (deterministischer Run)
    basis_path = run_conf.get("reference_network")
    if not basis_path or not Path(basis_path).is_file():
        # Fallback: aus scenarios.registry
        basis_path = master.get_reference_network_path(aro_cfg.SELECTED_RUN)
    if not basis_path or not Path(basis_path).is_file():
        print(f"❌ Basisnetz nicht gefunden: {basis_path}")
        print("   Tipp: run_conf['reference_network'] in master_config.py setzen.")
        return

    out_dir = aro_cfg.get_plot_output_dir("capacity_comparison")

    print(f"Robustes Netz : {Path(robust_path).name}")
    print(f"Basisjahr-Netz: {Path(basis_path).name}")
    print(f"Output        : {out_dir}")

    run_capacity_comparison(
        networks_robust=[robust_path],
        networks_basis=[basis_path],
        out_dir=out_dir,
        countries=None,   # None = alle Länder automatisch
        label_robust="Robustes Portfolio",
        label_basis="Basisjahr",
        carrier_colors=aro_cfg.CARRIER_COLORS,
        dpi=300,
    )


if __name__ == "__main__":
    main()