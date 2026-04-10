#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_cross_run_comparison.py
============================
Generischer Vergleich beliebiger PyPSA-Runs (normal und/oder ARO).

Unterstützte Kombinationen:
  - normal  vs. normal
  - normal  vs. aro
  - aro     vs. aro
  - N Runs gleichzeitig

Erzeugte Plots (pro Land):
  1. Installierte Kapazität [GW]          -- gestapelte Balken je Run,
     analog zum angehängten Schema:       je Run ein Balken pro Jahr
     bei single-year-Runs (2050):         je Run ein Balken nebeneinander
  2. Jährliche Stromerzeugung [TWh]       -- Balken gestapelt + Linie Nachfrage
  3. Speicherkapazitäten [GW / GWh]       -- grouped bars Leistung + Energie
  4. Systemkosten [Mrd. EUR/a]            -- horizontal grouped bar
  5. CO2-Emissionen [Mt CO2]              -- grouped bar
  6. Stromerzeugungsmix-Anteil [%]        -- 100%-Stapelbalken
  7. Curtailment [TWh + %]                -- grouped bar
  8. Mittlere Strom-Grenzpreise [EUR/MWh] -- grouped bar je Land
  9. Delta-Kapazität (run_B - run_A) [GW] -- Divergenzbalken

FIXES:
  #3 — bottoms-Array in plot_installed_capacity() war 1D und wurde über alle
       Runs akkumuliert (Run 1 startete auf Höhe von Run 0). Jetzt 2D:
       bottoms[run_idx, year_idx] — jeder Run hat seinen eigenen Stack.
  #7 — COMPARE_SPECS war leer. Jetzt mit sinnvollen Defaults aus master_config
       (REF_RUN_NAME vs. ARO_RUN_NAME) vorbelegt. Anpassen nach Bedarf.

Aufruf:
  python plot_cross_run_comparison.py
  python plot_cross_run_comparison.py \\
      --runs Basisrun-rcp45-2028:Basis big-aro-run-2:"ARO Robust" \\
      --countries ALL DE FR --year 2050
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")

from master_config import MasterConfig, MASTER_CONFIG, ARO_RUN_NAME, REF_RUN_NAME
from run_loader import RunLoader, RunSpec


# ============================================================================
# ███  USER CONFIGURATION  ███
# ============================================================================
# FIX #7: COMPARE_SPECS war leer → Skript brach ohne CLI-Argumente sofort ab.
# Jetzt mit Basis-Default aus master_config befüllt.
# Anpassen auf eigene Run-Keys, oder leer lassen und --runs CLI nutzen.
#
# Format: {"key": "<run-key-aus-master_config>", "label": "<Anzeigename>"}
# Für ARO-Run wird standardmäßig nur das robuste Netz geladen (kein Dispatch).
# Setze "aro_include_scenarios": True um alle Worst-Case-Netze zu laden.
#
# Alternativ: direkter Pfad ohne master_config-Eintrag:
# {"label": "Manuell", "paths": ["/abs/path/to/2050.nc"]}

COMPARE_SPECS: List[Dict] = [
    {"key": REF_RUN_NAME, "label": "Basisjahr 2050"},
    {"key": ARO_RUN_NAME, "label": "Robustes Portfolio", "aro_include_scenarios": False},
    # Weitere Runs nach Bedarf ergänzen:
    # {"key": "ein-weiterer-run", "label": "Run C"},
    # {"label": "Manuell", "paths": ["/pfad/zu/netz.nc"]},
]

# Länder ("ALL" = ganz Europa)
COMPARE_COUNTRIES: List[str] = ["ALL", "DE", "FR", "ES"]

# Falls Runs nur ein Jahr haben (2050): wird als "single-year" erkannt.
# Alternativ kann ein Ziel-Jahr erzwungen werden:
TARGET_YEAR: Optional[int] = None  # z.B. 2050 oder None für auto

# Ausgabepfad (None = plots_base/cross_run_comparison aus master_config)
OUT_DIR: Optional[str] = None

# Hatch-Muster je Run-Index (für gestapelte Balken-Vergleich)
HATCH_PATTERNS = ["", "///", "...", "xxx", "\\\\\\\\", "+++"]

# ============================================================================
# CARRIER / TECHNOLOGIE DEFINITIONEN
# ============================================================================

ELECTRICITY_GENERATORS = [
    "onwind", "offwind-ac", "offwind-dc", "offwind-float",
    "solar", "solar rooftop", "solar-hsat",
    "ror", "hydro", "nuclear",
    "coal", "lignite", "oil",
    "OCGT", "CCGT",
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
    "H2 turbine", "H2 OCGT", "H2 Fuel Cell",
]

STORAGE_CARRIERS_POWER = [
    "battery discharger", "home battery discharger", "PHS",
    "H2 Electrolysis",
]

STORAGE_CARRIERS_ENERGY = [
    "battery", "home battery", "PHS", "H2 storage",
]

VRE_CURTAILMENT_CARRIERS = ["onwind", "offwind-ac", "offwind-dc", "solar", "solar rooftop", "solar-hsat"]

CARRIER_TRANSLATION: Dict[str, str] = {
    "solar": "Photovoltaik",
    "solar rooftop": "PV Dach",
    "solar-hsat": "PV HSAT",
    "onwind": "Wind Onshore",
    "offwind-ac": "Wind Offshore (AC)",
    "offwind-dc": "Wind Offshore (DC)",
    "offwind-float": "Wind Offshore (Float)",
    "ror": "Laufwasser",
    "hydro": "Wasserkraft",
    "nuclear": "Kernkraft",
    "lignite": "Braunkohle",
    "coal": "Steinkohle",
    "oil": "Öl",
    "CCGT": "Erdgas (GuD)",
    "OCGT": "Erdgas (Gasturbine)",
    "PHS": "Pumpspeicher",
    "battery": "Batteriespeicher",
    "battery discharger": "Batteriespeicher",
    "home battery": "Heimspeicher",
    "home battery discharger": "Heimspeicher",
    "H2 storage": "H2-Speicher",
    "H2 Electrolysis": "Elektrolyse",
    "H2 turbine": "H2-Turbine",
    "H2 OCGT": "H2-Gasturbine",
    "H2 Fuel Cell": "Brennstoffzelle",
    "urban central solid biomass CHP": "Biomasse KWK",
    "urban central solid biomass CHP CC": "Biomasse KWK CC",
}

# Stapelung Reihenfolge (unten = Grundlast)
STACK_ORDER = [
    "Kernkraft", "Braunkohle", "Steinkohle", "Öl",
    "Erdgas (GuD)", "Erdgas (Gasturbine)",
    "Biomasse KWK", "Biomasse KWK CC",
    "Laufwasser", "Wasserkraft",
    "Wind Onshore", "Wind Offshore (AC)", "Wind Offshore (DC)", "Wind Offshore (Float)",
    "Photovoltaik", "PV Dach", "PV HSAT",
    "Pumpspeicher", "Batteriespeicher", "Heimspeicher",
    "H2-Turbine", "H2-Gasturbine", "Brennstoffzelle",
    "Elektrolyse", "H2-Speicher",
]

COUNTRY_NAMES: Dict[str, str] = {
    "ALL": "Europa (Gesamt)", "DE": "Deutschland", "FR": "Frankreich",
    "ES": "Spanien", "IT": "Italien", "GB": "Großbritannien",
    "PL": "Polen", "SE": "Schweden", "NO": "Norwegen",
    "NL": "Niederlande", "BE": "Belgien", "AT": "Österreich",
    "CH": "Schweiz", "DK": "Dänemark", "CZ": "Tschechien",
    "PT": "Portugal", "FI": "Finnland",
}

# ============================================================================
# EXTRACTION HELPERS
# ============================================================================

def _get_cc(master: MasterConfig) -> Dict[str, str]:
    return master.carrier_colors


def _get_fs(master: MasterConfig) -> Dict[str, int]:
    return master.font_sizes


def _country_bus_mask(n: pypsa.Network, country: str) -> pd.Index:
    """Bus-Indizes die zum Land gehören."""
    if country == "ALL":
        valid = set(b[:2] for b in n.buses.index
                    if len(b) >= 2 and b[:2].isalpha() and b[:2].isupper() and b[:2] != "EU")
        return n.buses.index[n.buses.index.str[:2].isin(valid)]
    return n.buses.index[n.buses.index.str.startswith(country)]


def _translate(carrier: str) -> str:
    if "solar" in carrier.lower():
        return "Photovoltaik"
    if "biomass" in carrier.lower():
        return "Biomasse KWK"
    return CARRIER_TRANSLATION.get(carrier, carrier)


def extract_installed_capacity(n: pypsa.Network, country: str) -> pd.Series:
    """
    Installierte elektrische Kapazität [GW] pro Technologie.
    Quellen: generators (p_nom_opt) + links (p_nom_opt * eff) + storage_units.
    """
    pcol = "p_nom_opt" if "p_nom_opt" in n.generators.columns else "p_nom"
    parts: List[pd.Series] = []

    valid_buses = _country_bus_mask(n, country)

    # --- Generatoren ---
    gens = n.generators.copy()
    gens = gens[gens.bus.isin(valid_buses)]
    gens = gens[gens.carrier.isin(ELECTRICITY_GENERATORS)]
    pcol_g = "p_nom_opt" if "p_nom_opt" in gens.columns else "p_nom"
    if not gens.empty:
        parts.append(gens.groupby("carrier")[pcol_g].sum())

    # --- Links (OCGT, CCGT, H2-Turbinen, CHP) ---
    lks = n.links.copy()
    lks_bus_mask = lks.bus1.isin(valid_buses) if country != "ALL" else \
        lks.bus1.str[:2].apply(lambda x: len(x) == 2 and x.isalpha() and x.isupper() and x != "EU")
    lks = lks[lks_bus_mask]
    lks = lks[lks.carrier.isin(ELECTRICITY_GENERATORS)]
    pcol_l = "p_nom_opt" if "p_nom_opt" in lks.columns else "p_nom"
    if not lks.empty:
        eff_col = "efficiency" if "efficiency" in lks.columns else None
        cap = lks[pcol_l].copy()
        if eff_col:
            eff = lks[eff_col].fillna(1.0)
            needs_corr = eff < 0.99
            cap[needs_corr] = cap[needs_corr] * eff[needs_corr]
        parts.append(cap.groupby(lks.carrier).sum())

    # --- Storage Units ---
    sus = n.storage_units.copy()
    sus = sus[sus.bus.isin(valid_buses)]
    sus = sus[~sus.bus.str.contains("EU", case=False, na=False)]
    sus = sus[sus.carrier.isin(STORAGE_CARRIERS_POWER + ["hydro", "ror"])]
    pcol_s = "p_nom_opt" if "p_nom_opt" in sus.columns else "p_nom"
    if not sus.empty:
        parts.append(sus.groupby("carrier")[pcol_s].sum())

    if not parts:
        return pd.Series(dtype=float)

    raw = pd.concat(parts).groupby(level=0).sum() / 1000.0  # MW -> GW

    clean: Dict[str, float] = {}
    for c, v in raw.items():
        name = _translate(c)
        clean[name] = clean.get(name, 0.0) + v

    s = pd.Series(clean)
    return s[s > 0.01]


def extract_storage_capacity(n: pypsa.Network, country: str) -> Tuple[pd.Series, pd.Series]:
    """Rückgabe: (power_GW, energy_GWh) je Technologie."""
    valid_buses = _country_bus_mask(n, country)
    pow_parts: List[pd.Series] = []
    ene_parts: List[pd.Series] = []

    sus = n.storage_units.copy()
    sus = sus[sus.bus.isin(valid_buses)]
    sus = sus[~sus.bus.str.contains("EU", case=False, na=False)]
    pcol = "p_nom_opt" if "p_nom_opt" in sus.columns else "p_nom"
    ecol = "e_nom_opt" if "e_nom_opt" in sus.columns else "e_nom"
    if not sus.empty:
        pow_parts.append(sus.groupby("carrier")[pcol].sum() / 1000.0)
        ene_parts.append(sus.groupby("carrier")[ecol].sum() / 1000.0)

    stores = n.stores.copy()
    if "bus" in stores.columns:
        stores = stores[stores.bus.isin(valid_buses)]
    ecol_st = "e_nom_opt" if "e_nom_opt" in stores.columns else "e_nom"
    if not stores.empty:
        ene_parts.append(stores.groupby("carrier")[ecol_st].sum() / 1000.0)

    def _agg(parts):
        if not parts:
            return pd.Series(dtype=float)
        raw = pd.concat(parts).groupby(level=0).sum()
        clean: Dict[str, float] = {}
        for c, v in raw.items():
            name = _translate(c)
            clean[name] = clean.get(name, 0.0) + v
        s = pd.Series(clean)
        return s[s > 0.01]

    return _agg(pow_parts), _agg(ene_parts)


def extract_annual_generation(n: pypsa.Network, country: str) -> Tuple[pd.Series, float]:
    """Jährliche Stromerzeugung [TWh] + Nachfrage [TWh]."""
    valid_buses = _country_bus_mask(n, country)
    gen_parts: List[pd.Series] = []

    gens = n.generators.copy()
    gens = gens[gens.bus.isin(valid_buses)]
    gens = gens[gens.carrier.isin(ELECTRICITY_GENERATORS)]
    avail_g = n.generators_t.p.columns.intersection(gens.index)
    if not avail_g.empty:
        g_sum = n.generators_t.p[avail_g].sum().rename(gens.loc[avail_g, "carrier"])
        gen_parts.append(g_sum.groupby(level=0).sum())

    sus = n.storage_units.copy()
    sus = sus[sus.bus.isin(valid_buses)]
    sus = sus[~sus.bus.str.contains("EU", na=False)]
    avail_s = n.storage_units_t.p.columns.intersection(sus.index)
    if not avail_s.empty:
        su_p = n.storage_units_t.p[avail_s].clip(lower=0).sum()
        su_p = su_p.rename(sus.loc[avail_s, "carrier"])
        gen_parts.append(su_p.groupby(level=0).sum())

    lks = n.links.copy()
    lks_mask = lks.bus1.isin(valid_buses) if country != "ALL" else \
        lks.bus1.str[:2].apply(lambda x: len(x) == 2 and x.isalpha() and x.isupper() and x != "EU")
    lks = lks[lks_mask]
    lks = lks[lks.carrier.isin(ELECTRICITY_GENERATORS)]
    avail_l = n.links_t.p1.columns.intersection(lks.index) if hasattr(n, "links_t") else []
    if len(avail_l) > 0:
        l_sum = n.links_t.p1[avail_l].abs().sum().rename(lks.loc[avail_l, "carrier"])
        gen_parts.append(l_sum.groupby(level=0).sum())

    if not gen_parts:
        return pd.Series(dtype=float), 0.0

    raw = pd.concat(gen_parts).groupby(level=0).sum() / 1e6
    clean: Dict[str, float] = {}
    for c, v in raw.items():
        name = _translate(c)
        clean[name] = clean.get(name, 0.0) + v
    gen = pd.Series(clean)
    gen = gen[gen > 0.01]

    lds = n.loads.copy()
    lds_elec = lds[lds.bus.isin(valid_buses)]
    lds_elec = lds_elec[~lds_elec.bus.str.contains("heat", case=False, na=False)]
    avail_d = n.loads_t.p.columns.intersection(lds_elec.index)
    demand = float(n.loads_t.p[avail_d].sum().sum() / 1e6) if not avail_d.empty else 0.0

    return gen, demand


def extract_system_cost(n: pypsa.Network, country: str) -> float:
    """Annualisierte Systemkosten [Mrd. EUR/a]."""
    cost = 0.0
    valid_buses = _country_bus_mask(n, country)

    gens = n.generators.copy()
    gens_c = gens[gens.bus.isin(valid_buses)]
    pcol = "p_nom_opt" if "p_nom_opt" in gens_c.columns else "p_nom"
    if not gens_c.empty:
        capex = (gens_c[pcol] * gens_c.get("capital_cost", pd.Series(0, index=gens_c.index))).sum()
        avail = n.generators_t.p.columns.intersection(gens_c.index)
        opex = 0.0
        if not avail.empty:
            mc = gens_c.loc[avail, "marginal_cost"] if "marginal_cost" in gens_c.columns else pd.Series(0, index=avail)
            opex = (n.generators_t.p[avail] * mc).sum().sum()
        cost += capex + opex

    lks = n.links.copy()
    lks_c = lks[lks.bus1.isin(valid_buses)] if country != "ALL" else lks
    pcol_l = "p_nom_opt" if "p_nom_opt" in lks_c.columns else "p_nom"
    if not lks_c.empty:
        capex_l = (lks_c[pcol_l] * lks_c.get("capital_cost", pd.Series(0, index=lks_c.index))).sum()
        cost += capex_l

    return cost / 1e9


def extract_co2_emissions(n: pypsa.Network, country: str) -> float:
    """CO2-Emissionen [Mt CO2/a]."""
    valid_buses = _country_bus_mask(n, country)
    co2_total = 0.0
    EMISSION_FACTORS: Dict[str, float] = {
        "coal": 0.34, "lignite": 0.40, "oil": 0.28,
        "OCGT": 0.19, "CCGT": 0.19, "gas": 0.19,
    }

    gens = n.generators.copy()
    gens_c = gens[gens.bus.isin(valid_buses)]
    avail = n.generators_t.p.columns.intersection(gens_c.index)
    if not avail.empty:
        for c, ef in EMISSION_FACTORS.items():
            mask = gens_c.index[gens_c.carrier.str.contains(c, case=False)].intersection(avail)
            if len(mask) > 0:
                co2_total += n.generators_t.p[mask].sum().sum() * ef / 1e6

    return co2_total


def extract_curtailment(n: pypsa.Network, country: str) -> Tuple[pd.Series, pd.Series]:
    """Rückgabe: (curtailed_TWh, curtailment_rate_pct) je VRE-Technologie."""
    valid_buses = _country_bus_mask(n, country)
    gens = n.generators.copy()
    gens_vre = gens[
        gens.bus.isin(valid_buses) &
        gens.carrier.isin(VRE_CURTAILMENT_CARRIERS)
    ]

    curtailed: Dict[str, float] = {}
    rate: Dict[str, float] = {}

    avail_p = n.generators_t.p.columns
    avail_pmax = n.generators_t.p_max_pu.columns if hasattr(n.generators_t, "p_max_pu") else pd.Index([])

    pcol = "p_nom_opt" if "p_nom_opt" in gens_vre.columns else "p_nom"

    for carrier in gens_vre.carrier.unique():
        subset = gens_vre[gens_vre.carrier == carrier]
        idx_p    = subset.index.intersection(avail_p)
        idx_pmax = subset.index.intersection(avail_pmax)

        if idx_p.empty:
            continue

        actual  = n.generators_t.p[idx_p].sum().sum()
        if not idx_pmax.empty:
            p_nom = subset.loc[idx_pmax, pcol]
            potential = (n.generators_t.p_max_pu[idx_pmax] * p_nom).sum().sum()
        else:
            potential = actual

        curt_twh = max(0.0, (potential - actual) / 1e6)
        rate_pct  = (curt_twh / (potential / 1e6) * 100.0) if potential > 0 else 0.0
        name = _translate(carrier)
        curtailed[name] = curtailed.get(name, 0.0) + curt_twh
        rate[name]      = rate.get(name, 0.0) + rate_pct

    return pd.Series(curtailed), pd.Series(rate)


def extract_marginal_prices(n: pypsa.Network, country: str) -> float:
    """Zeitlich gemittelter Grenzpreis [EUR/MWh] auf AC-Strombussen."""
    valid_buses = _country_bus_mask(n, country)
    elec_buses = valid_buses[
        ~valid_buses.str.contains("heat", case=False) &
        ~valid_buses.str.contains("H2", case=False) &
        ~valid_buses.str.contains("gas", case=False)
    ]
    if not hasattr(n, "buses_t") or not hasattr(n.buses_t, "marginal_price"):
        return np.nan
    avail = n.buses_t.marginal_price.columns.intersection(elec_buses)
    if avail.empty:
        return np.nan
    return float(n.buses_t.marginal_price[avail].mean().mean())


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def _carrier_color(tech: str, cc: Dict[str, str], default: str = "#aaaaaa") -> str:
    if tech in cc:
        return cc[tech]
    for eng, de in CARRIER_TRANSLATION.items():
        if de == tech and eng in cc:
            return cc[eng]
    tech_lower = tech.lower()
    for k, v in cc.items():
        if k.lower() in tech_lower or tech_lower in k.lower():
            return v
    return default


def _sort_techs(techs: List[str]) -> List[str]:
    ordered = [t for t in STACK_ORDER if t in techs]
    rest    = [t for t in techs if t not in ordered]
    return ordered + rest


def _save(fig: plt.Figure, out_dir: Path, fname: str, dpi: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path}")


def _stacked_bar_for_year(
    ax: plt.Axes,
    data_per_run: Dict[str, pd.Series],
    cc: Dict[str, str],
    hatch_patterns: List[str],
    unit: str = "GW",
    bar_label_threshold: float = 5.0,
    fs: Dict[str, int] = None,
    demand_per_run: Optional[Dict[str, float]] = None,
) -> List[mpatches.Patch]:
    fs = fs or {"bar_label": 9, "tick": 11, "label": 13, "title": 16}
    runs = list(data_per_run.keys())
    n_runs = len(runs)
    x = np.arange(n_runs)
    width = 0.6

    all_techs: set = set()
    for s in data_per_run.values():
        all_techs.update(s.index.tolist())
    techs = _sort_techs(list(all_techs))

    total_max = max((s.sum() for s in data_per_run.values()), default=1.0)
    TEXT_THR = bar_label_threshold if bar_label_threshold > 0 else 0.02 * total_max

    bottoms = np.zeros(n_runs)
    handles: List[mpatches.Patch] = []
    seen_techs: List[str] = []

    for tech in techs:
        vals = np.array([data_per_run[r].get(tech, 0.0) for r in runs])
        if vals.max() < 0.001:
            continue
        color = _carrier_color(tech, cc)
        for i, (v, hatch) in enumerate(zip(vals, hatch_patterns[:n_runs])):
            ax.bar(x[i], v, width, bottom=bottoms[i], color=color,
                   hatch=hatch, edgecolor="white", linewidth=0.4)
            if v >= TEXT_THR:
                ax.text(x[i], bottoms[i] + v / 2, f"{int(round(v))}",
                        ha="center", va="center", color="white",
                        fontsize=fs["bar_label"], fontweight="bold")
        bottoms += vals
        if tech not in seen_techs:
            seen_techs.append(tech)
            handles.append(mpatches.Patch(facecolor=color, label=tech))

    if demand_per_run:
        demands = [demand_per_run.get(r, 0.0) for r in runs]
        ax.plot(x, demands, "D--", color="#d63031", linewidth=1.8,
                markersize=6, zorder=5, label="Nachfrage")
        handles.append(mpatches.Patch(facecolor="#d63031", label="Nachfrage"))

    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=25, ha="right", fontsize=fs["tick"])
    ax.set_ylabel(unit, fontsize=fs["label"])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    return handles


# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def plot_installed_capacity(
    specs: List[RunSpec],
    country: str,
    out_dir: Path,
    cc: Dict[str, str],
    fs: Dict[str, int],
    target_year: Optional[int] = None,
) -> None:
    """
    Installierte Kapazität: Pro Run ein gestapelter Balken, alle Runs nebeneinander.
    Bei mehrjährigen Runs: x-Achse = Jahr, Balken je Run daneben.

    FIX #3: bottoms war 1D (n_years,) → über alle Runs akkumuliert.
    Jetzt 2D (n_runs, n_years): jeder Run hat seinen eigenen Stack.
    """
    c_name = COUNTRY_NAMES.get(country, country)

    all_years: set = set()
    run_data: Dict[str, Dict[Union[int, str], pd.Series]] = {}

    for spec in specs:
        nets = spec.get_all_networks()
        run_data[spec.label] = {}
        for tag, n in nets.items():
            if target_year is not None and isinstance(tag, int) and tag != target_year:
                continue
            cap = extract_installed_capacity(n, country)
            run_data[spec.label][tag] = cap
            if isinstance(tag, int):
                all_years.add(tag)

    years = sorted(all_years) if all_years else ["2050"]
    n_runs = len(specs)
    n_years = len(years)

    fig, ax = plt.subplots(figsize=(max(10, n_runs * n_years * 1.5 + 3), 10))

    all_techs: set = set()
    for rd in run_data.values():
        for s in rd.values():
            all_techs.update(s.index.tolist())
    techs = _sort_techs(list(all_techs))

    x_base = np.arange(n_years)
    width_total = 0.8
    bar_w = width_total / n_runs
    offsets = np.linspace(-width_total / 2 + bar_w / 2, width_total / 2 - bar_w / 2, n_runs)

    total_max = 0.0
    for rd in run_data.values():
        for s in rd.values():
            total_max = max(total_max, s.sum())
    TEXT_THR = 0.02 * total_max

    handles: List[mpatches.Patch] = []
    seen_techs: List[str] = []
    legend_run_patches: List[mpatches.Patch] = []

    # FIX #3: 2D-Array statt 1D — Dimension (n_runs × n_years)
    # Jeder Run akkumuliert seinen eigenen Stapel unabhängig von den anderen.
    bottoms = np.zeros((n_runs, n_years))

    for tech in techs:
        color = _carrier_color(tech, cc)
        for run_idx, spec in enumerate(specs):
            hatch = HATCH_PATTERNS[run_idx % len(HATCH_PATTERNS)]
            for y_idx, yr in enumerate(years):
                cap = run_data[spec.label].get(yr, pd.Series(dtype=float))
                v = float(cap.get(tech, 0.0))
                xpos = x_base[y_idx] + offsets[run_idx]
                ax.bar(
                    xpos, v, bar_w * 0.95,
                    bottom=bottoms[run_idx, y_idx],   # ← FIX: run-spezifischer bottom
                    color=color, hatch=hatch,
                    edgecolor="white", linewidth=0.3,
                )
                if v >= TEXT_THR:
                    ax.text(
                        xpos,
                        bottoms[run_idx, y_idx] + v / 2,  # ← FIX: run-spezifischer bottom
                        f"{int(round(v))}",
                        ha="center", va="center",
                        color="white",
                        fontsize=fs.get("bar_label", 9),
                        fontweight="bold",
                    )
                bottoms[run_idx, y_idx] += v              # ← FIX: nur diesen Run akkumulieren

        if tech not in seen_techs:
            seen_techs.append(tech)
            handles.append(mpatches.Patch(facecolor=color, label=tech))

    for run_idx, spec in enumerate(specs):
        hatch = HATCH_PATTERNS[run_idx % len(HATCH_PATTERNS)]
        legend_run_patches.append(
            mpatches.Patch(facecolor="#888888", hatch=hatch,
                           label=spec.label, edgecolor="white")
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels([str(y) for y in years], fontsize=fs.get("tick", 13))
    ax.set_xlabel("Jahr", fontsize=fs.get("label", 15))
    ax.set_ylabel("Kapazität [GW]", fontsize=fs.get("label", 15))

    run_labels = " | ".join([f"{HATCH_PATTERNS[i] or '—'} {s.label}" for i, s in enumerate(specs)])
    ax.set_title(
        f"Installierte Kapazität: {c_name}\n({run_labels})",
        fontsize=fs.get("title", 18), pad=14,
    )

    legend1 = ax.legend(
        handles=handles[::-1], title="Technologie",
        fontsize=fs.get("legend", 11),
        title_fontsize=fs.get("legend", 12),
        loc="upper left", bbox_to_anchor=(1.0, 1.0),
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=legend_run_patches, title="Run",
        fontsize=fs.get("legend", 11),
        title_fontsize=fs.get("legend", 12),
        loc="upper left", bbox_to_anchor=(1.0, 0.5),
    )

    plt.tight_layout()
    _save(fig, out_dir, f"01_installed_capacity_{country}.png")


def plot_annual_generation(
    specs: List[RunSpec],
    country: str,
    out_dir: Path,
    cc: Dict[str, str],
    fs: Dict[str, int],
    target_year: Optional[int] = None,
) -> None:
    c_name = COUNTRY_NAMES.get(country, country)

    gen_data: Dict[str, pd.Series] = {}
    demand_data: Dict[str, float] = {}

    for spec in specs:
        nets = spec.get_all_networks()
        for tag, n in nets.items():
            if target_year is not None and isinstance(tag, int) and tag != target_year:
                continue
            gen, dem = extract_annual_generation(n, country)
            key = f"{spec.label}\n({tag})"
            gen_data[key] = gen
            demand_data[key] = dem
            break

    if not gen_data:
        print(f"  [plot_annual_generation] Keine Daten für {country}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    ax = axes[0]
    handles = _stacked_bar_for_year(
        ax, gen_data, cc, HATCH_PATTERNS,
        unit="Stromerzeugung [TWh]",
        bar_label_threshold=20.0,
        fs=fs, demand_per_run=demand_data,
    )
    ax.set_title(f"Jährl. Stromerzeugung — {c_name}", fontsize=fs.get("title", 16))

    ax2 = axes[1]
    all_techs: set = set()
    for s in gen_data.values():
        all_techs.update(s.index.tolist())
    techs = _sort_techs(list(all_techs))
    runs_sorted = list(gen_data.keys())
    x = np.arange(len(runs_sorted))
    bottoms = np.zeros(len(runs_sorted))

    for tech in techs:
        vals = np.array([gen_data[r].get(tech, 0.0) for r in runs_sorted])
        totals = np.array([gen_data[r].sum() for r in runs_sorted])
        totals = np.where(totals > 0, totals, 1.0)
        pcts = vals / totals * 100.0
        if pcts.max() < 0.1:
            continue
        color = _carrier_color(tech, cc)
        for i, (p, h) in enumerate(zip(pcts, HATCH_PATTERNS[:len(runs_sorted)])):
            ax2.bar(x[i], p, 0.6, bottom=bottoms[i],
                    color=color, hatch=h, edgecolor="white", linewidth=0.4)
            if p > 3.0:
                ax2.text(x[i], bottoms[i] + p / 2, f"{p:.0f}%",
                         ha="center", va="center", color="white",
                         fontsize=fs.get("bar_label", 9), fontweight="bold")
        bottoms += pcts

    ax2.set_xticks(x)
    ax2.set_xticklabels(runs_sorted, rotation=25, ha="right", fontsize=fs.get("tick", 11))
    ax2.set_ylabel("Anteil [%]", fontsize=fs.get("label", 13))
    ax2.set_ylim(0, 105)
    ax2.set_title(f"Erzeugungsmix [%] — {c_name}", fontsize=fs.get("title", 16))
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    handles_legend = [mpatches.Patch(facecolor=_carrier_color(t, cc), label=t)
                      for t in techs if any(gen_data[r].get(t, 0) > 0 for r in runs_sorted)]
    fig.legend(
        handles=handles_legend[::-1], title="Technologie",
        fontsize=fs.get("legend", 10),
        loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout()
    _save(fig, out_dir, f"02_annual_generation_{country}.png")


def plot_storage_capacities(
    specs: List[RunSpec],
    country: str,
    out_dir: Path,
    cc: Dict[str, str],
    fs: Dict[str, int],
    target_year: Optional[int] = None,
) -> None:
    c_name = COUNTRY_NAMES.get(country, country)

    pow_data: Dict[str, pd.Series] = {}
    ene_data: Dict[str, pd.Series] = {}

    for spec in specs:
        nets = spec.get_all_networks()
        for tag, n in nets.items():
            if target_year is not None and isinstance(tag, int) and tag != target_year:
                continue
            pw, en = extract_storage_capacity(n, country)
            pow_data[spec.label] = pw
            ene_data[spec.label] = en
            break

    if not pow_data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, data_dict, unit, title in zip(
        axes,
        [pow_data, ene_data],
        ["Leistung [GW]", "Energie [GWh]"],
        ["Speicher-Leistungskapazität", "Speicher-Energiekapazität"],
    ):
        all_techs = set(t for s in data_dict.values() for t in s.index)
        techs = _sort_techs(list(all_techs))
        runs_list = list(data_dict.keys())
        n_r = len(runs_list)
        x = np.arange(n_r)
        bottoms = np.zeros(n_r)

        for tech in techs:
            vals = np.array([data_dict[r].get(tech, 0.0) for r in runs_list])
            if vals.max() < 0.001:
                continue
            color = _carrier_color(tech, cc)
            for i, (v, hatch) in enumerate(zip(vals, HATCH_PATTERNS[:n_r])):
                ax.bar(x[i], v, 0.6, bottom=bottoms[i],
                       color=color, hatch=hatch, edgecolor="white", linewidth=0.4,
                       label=tech if i == 0 else "_nolegend_")
                if v > 0.5:
                    ax.text(x[i], bottoms[i] + v / 2, f"{v:.0f}",
                            ha="center", va="center", color="white",
                            fontsize=fs.get("bar_label", 9), fontweight="bold")
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(runs_list, rotation=25, ha="right", fontsize=fs.get("tick", 11))
        ax.set_ylabel(unit, fontsize=fs.get("label", 13))
        ax.set_title(f"{title} — {c_name}", fontsize=fs.get("title", 16))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  fontsize=fs.get("legend", 10), loc="upper right")

    plt.tight_layout()
    _save(fig, out_dir, f"03_storage_capacities_{country}.png")


def plot_system_costs_and_co2(
    specs: List[RunSpec],
    country: str,
    out_dir: Path,
    cc: Dict[str, str],
    fs: Dict[str, int],
    target_year: Optional[int] = None,
) -> None:
    c_name = COUNTRY_NAMES.get(country, country)

    costs: Dict[str, float] = {}
    co2s: Dict[str, float] = {}

    for spec in specs:
        nets = spec.get_all_networks()
        for tag, n in nets.items():
            if target_year is not None and isinstance(tag, int) and tag != target_year:
                continue
            costs[spec.label] = extract_system_cost(n, country)
            co2s[spec.label]  = extract_co2_emissions(n, country)
            break

    if not costs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    run_labels = list(costs.keys())
    x = np.arange(len(run_labels))
    bar_colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"][:len(run_labels)]
    hatches = HATCH_PATTERNS[:len(run_labels)]

    for ax, data_dict, unit, title, _ in zip(
        axes,
        [costs, co2s],
        ["Systemkosten [Mrd. EUR/a]", "CO₂-Emissionen [Mt CO₂/a]"],
        ["Annualisierte Systemkosten", "CO₂-Emissionen"],
        ["#2ecc71", "#e67e22"],
    ):
        vals = np.array([data_dict.get(r, 0.0) for r in run_labels])
        bars = ax.bar(x, vals, 0.6,
                      color=[bar_colors[i] for i in range(len(run_labels))],
                      hatch=hatches,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{v:.1f}",
                        ha="center", va="bottom",
                        fontsize=fs.get("bar_label", 10), fontweight="bold")

        if len(vals) > 1 and vals[0] > 0:
            for i in range(1, len(vals)):
                delta_pct = (vals[i] - vals[0]) / vals[0] * 100
                ax.text(x[i], vals[i] * 0.5, f"{delta_pct:+.1f}%",
                        ha="center", va="center",
                        color="white", fontsize=fs.get("bar_label", 10),
                        fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=20, ha="right", fontsize=fs.get("tick", 11))
        ax.set_ylabel(unit, fontsize=fs.get("label", 13))
        ax.set_title(f"{title} — {c_name}", fontsize=fs.get("title", 16))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    _save(fig, out_dir, f"04_costs_co2_{country}.png")


def plot_curtailment(
    specs: List[RunSpec],
    country: str,
    out_dir: Path,
    cc: Dict[str, str],
    fs: Dict[str, int],
    target_year: Optional[int] = None,
) -> None:
    c_name = COUNTRY_NAMES.get(country, country)

    curt_data: Dict[str, pd.Series] = {}
    rate_data: Dict[str, pd.Series] = {}

    for spec in specs:
        nets = spec.get_all_networks()
        for tag, n in nets.items():
            if target_year is not None and isinstance(tag, int) and tag != target_year:
                continue
            curt, rate = extract_curtailment(n, country)
            curt_data[spec.label] = curt
            rate_data[spec.label] = rate
            break

    if not curt_data or all(s.sum() < 0.001 for s in curt_data.values()):
        print(f"  [plot_curtailment] Kein Curtailment für {country}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, data_dict, unit, title in zip(
        axes,
        [curt_data, rate_data],
        ["Curtailment [TWh/a]", "Curtailment-Rate [%]"],
        ["Abgeregelte Energie", "Curtailment-Rate"],
    ):
        all_techs = set(t for s in data_dict.values() for t in s.index)
        techs = _sort_techs(list(all_techs))
        runs_list = list(data_dict.keys())
        n_r = len(runs_list)
        x = np.arange(n_r)
        bottoms = np.zeros(n_r)

        for tech in techs:
            vals = np.array([data_dict[r].get(tech, 0.0) for r in runs_list])
            if vals.max() < 0.001:
                continue
            color = _carrier_color(tech, cc)
            for i, (v, hatch) in enumerate(zip(vals, HATCH_PATTERNS[:n_r])):
                ax.bar(x[i], v, 0.6, bottom=bottoms[i],
                       color=color, hatch=hatch, edgecolor="white", linewidth=0.4,
                       label=tech if i == 0 else "_nolegend_")
                if v > 0.5:
                    ax.text(x[i], bottoms[i] + v / 2, f"{v:.0f}",
                            ha="center", va="center", color="white",
                            fontsize=fs.get("bar_label", 9), fontweight="bold")
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(runs_list, rotation=20, ha="right", fontsize=fs.get("tick", 11))
        ax.set_ylabel(unit, fontsize=fs.get("label", 13))
        ax.set_title(f"{title} — {c_name}", fontsize=fs.get("title", 16))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  fontsize=fs.get("legend", 10))

    plt.tight_layout()
    _save(fig, out_dir, f"05_curtailment_{country}.png")


def plot_marginal_prices_comparison(
    specs: List[RunSpec],
    countries: List[str],
    out_dir: Path,
    fs: Dict[str, int],
    target_year: Optional[int] = None,
) -> None:
    data: Dict[str, Dict[str, float]] = {spec.label: {} for spec in specs}

    for spec in specs:
        nets = spec.get_all_networks()
        for tag, n in nets.items():
            if target_year is not None and isinstance(tag, int) and tag != target_year:
                continue
            for country in countries:
                price = extract_marginal_prices(n, country)
                data[spec.label][country] = price
            break

    valid_countries = [c for c in countries
                       if any(not np.isnan(data[r].get(c, np.nan)) for r in data)]
    if not valid_countries:
        print("  [plot_marginal_prices_comparison] Keine Preisdaten")
        return

    run_labels = list(data.keys())
    n_r = len(run_labels)
    n_c = len(valid_countries)
    x = np.arange(n_c)
    width = 0.8 / n_r
    offsets = np.linspace(-(n_r - 1) * width / 2, (n_r - 1) * width / 2, n_r)

    fig, ax = plt.subplots(figsize=(max(12, n_c * 1.2 + 2), 8))
    run_colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"][:n_r]

    for r_idx, (run_label, color) in enumerate(zip(run_labels, run_colors)):
        vals = [data[run_label].get(c, np.nan) for c in valid_countries]
        hatch = HATCH_PATTERNS[r_idx % len(HATCH_PATTERNS)]
        ax.bar(x + offsets[r_idx], vals, width * 0.95, color=color, hatch=hatch,
               edgecolor="white", linewidth=0.4, label=run_label)
        for i, v in enumerate(vals):
            if not np.isnan(v) and v > 1.0:
                ax.text(x[i] + offsets[r_idx], v + 0.5, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=fs.get("bar_label", 9))

    ax.set_xticks(x)
    ax.set_xticklabels([COUNTRY_NAMES.get(c, c) for c in valid_countries],
                       rotation=30, ha="right", fontsize=fs.get("tick", 11))
    ax.set_ylabel("Mittl. Grenzpreis [EUR/MWh]", fontsize=fs.get("label", 13))
    ax.set_title("Mittlere Strom-Grenzpreise je Land", fontsize=fs.get("title", 16))
    ax.legend(title="Run", fontsize=fs.get("legend", 11), title_fontsize=fs.get("legend", 12))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, out_dir, "06_marginal_prices_by_country.png")


def plot_delta_capacity(
    specs: List[RunSpec],
    country: str,
    out_dir: Path,
    cc: Dict[str, str],
    fs: Dict[str, int],
    target_year: Optional[int] = None,
    reference_idx: int = 0,
) -> None:
    """Divergenz-Balken: Delta-Kapazität relativ zu Referenz-Run."""
    if len(specs) < 2:
        return
    c_name = COUNTRY_NAMES.get(country, country)
    ref_spec = specs[reference_idx]
    compare_specs_list = [s for i, s in enumerate(specs) if i != reference_idx]

    def _get_cap(spec):
        nets = spec.get_all_networks()
        for tag, n in nets.items():
            if target_year is not None and isinstance(tag, int) and tag != target_year:
                continue
            return extract_installed_capacity(n, country)
        return pd.Series(dtype=float)

    cap_ref = _get_cap(ref_spec)

    fig, axes = plt.subplots(1, len(compare_specs_list),
                             figsize=(max(8, len(compare_specs_list) * 7), 9),
                             sharey=True)
    if len(compare_specs_list) == 1:
        axes = [axes]

    for ax, comp_spec in zip(axes, compare_specs_list):
        cap_comp = _get_cap(comp_spec)
        all_techs = cap_ref.index.union(cap_comp.index)
        delta = (cap_comp.reindex(all_techs, fill_value=0.0)
                 - cap_ref.reindex(all_techs, fill_value=0.0))
        delta = delta[delta.abs() > 0.05].sort_values()

        if delta.empty:
            ax.text(0.5, 0.5, "Keine signifikanten\nUnterschiede",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=fs.get("label", 13))
        else:
            colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in delta.values]
            ax.barh(range(len(delta)), delta.values, color=colors,
                    edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(delta)))
            ax.set_yticklabels(delta.index, fontsize=fs.get("tick", 10))
            ax.axvline(0, color="black", linewidth=1.0)
            for i, v in enumerate(delta.values):
                ax.text(v + (0.1 if v >= 0 else -0.1), i, f"{v:+.1f}",
                        ha="left" if v >= 0 else "right", va="center",
                        fontsize=fs.get("bar_label", 9))

        ax.set_xlabel("Δ Kapazität [GW]", fontsize=fs.get("label", 13))
        ax.set_title(f"{comp_spec.label} − {ref_spec.label}\n{c_name}",
                     fontsize=fs.get("title", 15))
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    _save(fig, out_dir, f"07_delta_capacity_{country}.png")


# ============================================================================
# STATISTICS REPORT
# ============================================================================

def print_statistics(
    specs: List[RunSpec],
    countries: List[str],
    target_year: Optional[int] = None,
) -> None:
    print("\n" + "=" * 80)
    print("📊 VERGLEICHS-STATISTIK")
    print("=" * 80)

    for country in countries:
        c_name = COUNTRY_NAMES.get(country, country)
        print(f"\n{'─' * 60}")
        print(f"📍 {c_name} ({country})")
        print(f"{'─' * 60}")

        rows = []
        for spec in specs:
            nets = spec.get_all_networks()
            for tag, n in nets.items():
                if target_year is not None and isinstance(tag, int) and tag != target_year:
                    continue
                gen, dem = extract_annual_generation(n, country)
                cost = extract_system_cost(n, country)
                co2  = extract_co2_emissions(n, country)
                price = extract_marginal_prices(n, country)
                rows.append({
                    "Run": spec.label, "Tag": tag,
                    "Erzeugung (TWh)": gen.sum(),
                    "Nachfrage (TWh)": dem,
                    "Systemkosten (Mrd.€)": cost,
                    "CO₂ (Mt)": co2,
                    "Grenzpreis (€/MWh)": price,
                })
                break

        if rows:
            df_stats = pd.DataFrame(rows).set_index("Run")
            print(df_stats.to_string(float_format="{:.1f}".format))

    print("\n" + "=" * 80)


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_comparison(
    specs: List[RunSpec],
    countries: List[str],
    out_dir: Path,
    master: MasterConfig,
    target_year: Optional[int] = None,
) -> None:
    cc = master.carrier_colors
    fs = master.font_sizes

    print(f"\n{'═' * 70}")
    print(f"  Cross-Run Vergleich: {len(specs)} Runs × {len(countries)} Länder")
    print(f"  Runs: {[s.label for s in specs]}")
    print(f"  Ziel-Ausgabepfad: {out_dir}")
    print(f"{'═' * 70}")

    print_statistics(specs, countries, target_year)

    for country in countries:
        c_dir = out_dir / country
        c_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n── Land: {COUNTRY_NAMES.get(country, country)} ──")

        print("  Plot 1: Installierte Kapazität")
        plot_installed_capacity(specs, country, c_dir, cc, fs, target_year)

        print("  Plot 2: Jährliche Stromerzeugung")
        plot_annual_generation(specs, country, c_dir, cc, fs, target_year)

        print("  Plot 3: Speicherkapazitäten")
        plot_storage_capacities(specs, country, c_dir, cc, fs, target_year)

        print("  Plot 4: Systemkosten & CO₂")
        plot_system_costs_and_co2(specs, country, c_dir, cc, fs, target_year)

        print("  Plot 5: Curtailment")
        plot_curtailment(specs, country, c_dir, cc, fs, target_year)

        print("  Plot 7: Delta-Kapazität")
        plot_delta_capacity(specs, country, c_dir, cc, fs, target_year)

    print("  Plot 6: Grenzpreise")
    plot_marginal_prices_comparison(specs, countries, out_dir, fs, target_year)

    print(f"\n{'═' * 70}")
    print(f"  ✅ Fertig. Plots in: {out_dir}")
    print(f"{'═' * 70}")


def main() -> None:
    master = MasterConfig()

    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument(
        "--runs", nargs="+", metavar="KEY:LABEL",
        help="Runs als 'key:Label' Paare, z.B. basis-run:Basis aro-v2:\"ARO v2\"",
    )
    ap.add_argument("--countries", nargs="+", default=None)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument(
        "--aro-scenarios", action="store_true",
        help="ARO-Runs: alle Worst-Case-Netze laden (Standard: nur robust)",
    )
    args = ap.parse_args()

    if args.runs:
        spec_dicts = []
        for r in args.runs:
            if ":" in r:
                key, label = r.split(":", 1)
            else:
                key, label = r, r
            spec_dicts.append({"key": key, "label": label,
                               "aro_include_scenarios": args.aro_scenarios})
        specs = RunLoader.from_spec_dicts(spec_dicts, master)
    elif COMPARE_SPECS:
        specs = RunLoader.from_spec_dicts(COMPARE_SPECS, master)
    else:
        print("❌ Keine Runs definiert. Bitte COMPARE_SPECS im Skript konfigurieren"
              " oder --runs CLI-Argument nutzen.")
        return

    countries = args.countries or COMPARE_COUNTRIES
    target_year = args.year or TARGET_YEAR

    if args.outdir:
        out_dir = Path(args.outdir)
    elif OUT_DIR:
        out_dir = Path(OUT_DIR)
    else:
        out_dir = Path(master.plots_base) / "cross_run_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_comparison(specs, countries, out_dir, master, target_year)


if __name__ == "__main__":
    main()