#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_aro_capacity_by_country.py
================================
Installierte elektrische Kapazitäten für ARO-Ergebnisse — ein Plot PRO LAND.

FIXES (alle Versionen):
  A — Links (OCGT, CCGT, H2-Turbinen, CHP etc.) fehlten, da PyPSA-Eur sie
      als Links modelliert, nicht als Generatoren.
  B — Kein Fallback-Schutz wenn p_nom_opt fehlt.
  C — Länderfilter für Links muss über bus1 (Stromnetz-Bus) gehen.
  D — Kein Sektor-Filter: Wärmepumpen, Heizstäbe, Boiler, CHP-Wärmeseite
      usw. wurden mitgeplottet weil keine Carrier-Whitelist für Strom existierte.
      In einem sektor-gekoppelten Netz dominieren Wärme-Carrier die Plots.
      Fix: ELECTRICITY_CARRIERS Whitelist — nur Strom-erzeugende/-speichernde
      Komponenten werden dargestellt.
  E — bus1-Filter für Links prüfte nicht ob der Bus tatsächlich ein Strom-Bus
      ist (kein "heat", "H2", "gas" im Namen). Wärmepumpen (bus1 = Wärme-Bus)
      wurden dadurch fälschlicherweise als Stromerzeugung gezählt.

Jeder Plot zeigt:
  x-Achse  = Energieträger (Carrier)
  Säulen   = grouped bar: robust + stress_01, stress_02, ...

Standalone-Nutzung::
    python plots_all/plot_aro_capacity_by_country.py [--run RUN_KEY] [--output DIR]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pypsa

# ---------------------------------------------------------------------------
# FIX D: Carrier-Whitelist — NUR diese Carrier werden dargestellt.
# Alles was nicht hier steht (Wärmepumpen, Heizstäbe, Boiler, H2-Elektrolyse
# als Verbraucher, EV-Lader etc.) wird ignoriert.
# ---------------------------------------------------------------------------
ELECTRICITY_CARRIERS: Set[str] = {
    # VRE
    "onwind", "offwind-ac", "offwind-dc", "offwind-float",
    "solar", "solar rooftop", "solar-hsat",
    # Laufwasser / Speicherwasser
    "ror", "hydro",
    # Kernkraft
    "nuclear",
    # Fossile Stromerzeugung
    "coal", "lignite", "oil",
    "OCGT", "CCGT", "gas",
    # Biomasse (Strom-Anteil)
    "biomass", "biogas",
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
    # H2-basierte Stromerzeugung
    "H2 turbine", "H2 OCGT", "H2 Fuel Cell",
    # Stromspeicher (Entladeleistung)
    "battery discharger", "home battery discharger",
    "PHS",
}

# Carrier die als Links modelliert sind und Strom INS Netz einspeisen.
# bus1 muss ein Strom-Bus sein (kein heat/H2/gas-Bus).
LINK_ELECTRICITY_CARRIERS: Set[str] = {
    "OCGT", "CCGT", "gas", "oil", "nuclear",
    "H2 Fuel Cell", "H2 turbine", "H2 OCGT",
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
    "battery discharger", "home battery discharger",  # FIX: Batterien fehlten
    "coal", "lignite", "biomass", "OCGT methanol",  # FIX: fehlende AC-Erzeuger
}

# FIX E: Strings im Bus-Namen die auf NICHT-Strom-Busse hinweisen.
# Links mit diesen Strings in bus1 werden als Nicht-Strom-Links behandelt.
NON_ELECTRICITY_BUS_KEYWORDS: Set[str] = {
    "heat", "H2", "gas", "biogas", "oil", "co2", "CO2",
    "biomass", "methanol", "ammonia", "Fischer",
}

# Reihenfolge in den Plots (von unten nach oben / links nach rechts)
CARRIER_ORDER = [
    "nuclear",
    "coal", "lignite", "oil",
    "OCGT", "CCGT", "gas",
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
    "biomass", "biogas",
    "ror", "hydro", "PHS",
    "H2 Fuel Cell", "H2 turbine", "H2 OCGT",
    "offwind-ac", "offwind-dc", "offwind-float",
    "onwind",
    "solar", "solar rooftop", "solar-hsat",
    "battery discharger", "home battery discharger",
]

DEFAULT_COLORS: Dict[str, str] = {
    "offwind-ac":                          "#6caedf",
    "offwind-dc":                          "#2e86c1",
    "offwind-float":                       "#15a0bf",
    "onwind":                              "#74c476",
    "solar":                               "#fdae6b",
    "solar rooftop":                       "#fd8d3c",
    "solar-hsat":                          "#FFF080",
    "ror":                                 "#9ecae1",
    "hydro":                               "#6baed6",
    "nuclear":                             "#9467bd",
    "OCGT":                                "#e7cb94",
    "CCGT":                                "#c49c94",
    "gas":                                 "#c5b0d5",
    "oil":                                 "#aec7e8",
    "coal":                                "#636363",
    "lignite":                             "#393b79",
    "H2 Fuel Cell":                        "#b5cf6b",
    "H2 turbine":                          "#991f83",
    "H2 OCGT":                             "#c251ae",
    "battery discharger":                  "#7b2d8b",
    "home battery discharger":             "#b05cc7",
    "PHS":                                 "#3182bd",
    "biomass":                             "#31a354",
    "biogas":                              "#74c476",
    "urban central solid biomass CHP":     "#baa741",
    "urban central solid biomass CHP CC":  "#a8963a",
}


# ---------------------------------------------------------------------------
# Szenario-Kurznamen  (stress_xy)
# ---------------------------------------------------------------------------

def _shorten_scenario_name(scenario_name: str, index: int) -> str:
    m = re.search(r'stress[_\-](\w+)', scenario_name, re.IGNORECASE)
    if m:
        return f"stress_{m.group(1)}"
    rcp_m = re.search(r'(rcp\d+|ssp\d+)', scenario_name, re.IGNORECASE)
    if rcp_m:
        return f"stress_{rcp_m.group(1).lower()}"
    return f"stress_{index:02d}"


def build_scenario_label_map(scenario_names: List[str]) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    stress_idx = 1
    for name in scenario_names:
        if name == "robust":
            label_map[name] = "robust"
        else:
            label_map[name] = _shorten_scenario_name(name, stress_idx)
            stress_idx += 1
    return label_map


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _country_from_bus(bus_name: str) -> str:
    return str(bus_name)[:2].strip()


def _is_electricity_bus(bus_name: str) -> bool:
    """
    FIX E: Prüft ob ein Bus-Name zu einem Strom-Bus gehört.
    Busse die Wärme, H2, Gas etc. enthalten sind keine Strom-Busse.
    """
    bus_lower = str(bus_name).lower()
    for kw in NON_ELECTRICITY_BUS_KEYWORDS:
        if kw.lower() in bus_lower:
            return False
    return True


def _get_capacity_col(comp: pd.DataFrame, component_name: str) -> Optional[str]:
    """
    FIX B: Gibt die korrekte Kapazitätsspalte zurück.
    Warnt wenn nur p_nom verfügbar und alle Werte 0 sind.
    """
    if "p_nom_opt" in comp.columns:
        return "p_nom_opt"
    if "p_nom" in comp.columns:
        if comp["p_nom"].sum() == 0:
            print(f"  [cap_by_country] Warnung: {component_name}.p_nom ist überall 0 "
                  f"— wurde das Netzwerk optimiert?")
            return None
        return "p_nom"
    return None


# ---------------------------------------------------------------------------
# Kapazitäts-Extraktion (Generatoren)
# ---------------------------------------------------------------------------

def _cap_generators(n: pypsa.Network) -> pd.DataFrame:
    """
    Installierte Kapazität [GW] aus n.generators.
    FIX D: Nur ELECTRICITY_CARRIERS werden berücksichtigt.
    """
    if n.generators.empty:
        return pd.DataFrame()

    cap_col = _get_capacity_col(n.generators, "generators")
    if cap_col is None:
        return pd.DataFrame()

    gens = n.generators.copy()
    # FIX D: Carrier-Whitelist
    gens = gens[gens.carrier.isin(ELECTRICITY_CARRIERS)]
    if gens.empty:
        return pd.DataFrame()

    df = gens[["bus", "carrier", cap_col]].copy()
    df["country"] = df["bus"].apply(_country_from_bus)
    df[cap_col] = df[cap_col] / 1e3  # MW → GW

    return (
        df[df[cap_col] > 0]
        .groupby(["country", "carrier"])[cap_col]
        .sum()
        .unstack(fill_value=0)
    )


# ---------------------------------------------------------------------------
# Kapazitäts-Extraktion (Storage Units)
# ---------------------------------------------------------------------------

def _cap_storage_units(n: pypsa.Network) -> pd.DataFrame:
    """
    Installierte Leistungskapazität [GW] aus n.storage_units.
    FIX D: Nur Strom-Speicher (PHS, battery, hydro, ror).
    FIX E: EU-Busse und Nicht-Strom-Busse werden gefiltert.
    """
    if n.storage_units.empty:
        return pd.DataFrame()

    cap_col = _get_capacity_col(n.storage_units, "storage_units")
    if cap_col is None:
        return pd.DataFrame()

    sus = n.storage_units.copy()
    # FIX D: Carrier-Whitelist
    sus = sus[sus.carrier.isin(ELECTRICITY_CARRIERS)]
    # FIX E: EU-Busse und Nicht-Strom-Busse raus
    sus = sus[~sus.bus.str.contains("EU", case=False, na=False)]
    sus = sus[sus.bus.apply(_is_electricity_bus)]
    if sus.empty:
        return pd.DataFrame()

    df = sus[["bus", "carrier", cap_col]].copy()
    df["country"] = df["bus"].apply(_country_from_bus)
    df[cap_col] = df[cap_col] / 1e3

    return (
        df[df[cap_col] > 0]
        .groupby(["country", "carrier"])[cap_col]
        .sum()
        .unstack(fill_value=0)
    )


# ---------------------------------------------------------------------------
# Kapazitäts-Extraktion (Links)
# ---------------------------------------------------------------------------

def _cap_links(n: pypsa.Network) -> pd.DataFrame:
    """
    FIX A+C+D+E: Installierte elektrische Kapazität [GW] aus Links.

    Nur Links bei denen:
      1. carrier in LINK_ELECTRICITY_CARRIERS  (FIX D)
      2. bus1 ist ein Strom-Bus (kein "heat"/"H2"/... im Namen)  (FIX E)

    Kapazität = p_nom_opt × efficiency (elektrisch, falls < 1).
    Land = bus1[:2].
    """
    if n.links.empty:
        return pd.DataFrame()

    cap_col = _get_capacity_col(n.links, "links")
    if cap_col is None:
        return pd.DataFrame()

    # FIX D: Carrier-Filter
    lks = n.links[n.links.carrier.isin(LINK_ELECTRICITY_CARRIERS)].copy()
    if lks.empty:
        return pd.DataFrame()

    # FIX E: bus1 muss ein Strom-Bus sein
    lks = lks[lks.bus1.apply(_is_electricity_bus)]
    if lks.empty:
        return pd.DataFrame()

    # Elektrische Kapazität = p_nom_opt × elektrischer Wirkungsgrad
    cap = lks[cap_col].copy()
    if "efficiency" in lks.columns:
        eff = lks["efficiency"].fillna(1.0)
        # Nur korrigieren wenn eff plausibel als elektrischer Wirkungsgrad
        needs_corr = (eff > 0.01) & (eff < 0.99)
        cap[needs_corr] = cap[needs_corr] * eff[needs_corr]

    lks = lks.copy()
    lks["_cap_gw"] = cap / 1e3
    lks["country"] = lks["bus1"].apply(_country_from_bus)   # FIX C: bus1

    return (
        lks[lks["_cap_gw"] > 0]
        .groupby(["country", "carrier"])["_cap_gw"]
        .sum()
        .unstack(fill_value=0)
    )


# ---------------------------------------------------------------------------
# Kombinierte Extraktion
# ---------------------------------------------------------------------------

def _merge_dfs(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Vereinigt mehrere [country × carrier] DataFrames additiv."""
    non_empty = [df for df in dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame()
    result = non_empty[0]
    for df in non_empty[1:]:
        result = result.add(df, fill_value=0)
    return result


def extract_electricity_capacity(n: pypsa.Network) -> pd.DataFrame:
    """
    Vollständige installierte elektrische Kapazität [GW] je Land und Carrier.
    Kombiniert Generatoren + Storage Units + Links.
    Nur Strom-Sektor (FIX D+E).
    """
    return _merge_dfs(
        _cap_generators(n),
        _cap_storage_units(n),
        _cap_links(n),
    )


# ---------------------------------------------------------------------------
# Farben
# ---------------------------------------------------------------------------

def _get_carrier_color(carrier: str, user_colors: Optional[Dict] = None) -> str:
    base = {**DEFAULT_COLORS, **(user_colors or {})}
    return base.get(carrier, "#a9a9a9")


# ---------------------------------------------------------------------------
# Plot 1: Grouped-bar je Land  (x = Carrier, Gruppen = Szenarien)
# ---------------------------------------------------------------------------

def plot_capacity_grouped_by_carrier(
    country: str,
    data: Dict[str, pd.Series],
    label_map: Dict[str, str],
    all_carriers: List[str],
    title: str,
    out_path: Path,
    user_colors: Optional[Dict] = None,
    save: bool = True,
) -> None:
    """Für EIN Land: je Carrier eine Gruppe, je Szenario eine Säule."""
    if not data:
        return

    scenario_labels = list(data.keys())
    n_scenarios     = len(scenario_labels)
    n_carriers      = len(all_carriers)
    if n_carriers == 0:
        return

    fig_w = max(10, n_carriers * (n_scenarios * 0.35 + 0.5) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    group_w   = 0.8
    bar_w     = group_w / n_scenarios
    x_centers = np.arange(n_carriers)
    alphas    = np.linspace(1.0, 0.45, n_scenarios)

    for s_idx, slabel in enumerate(scenario_labels):
        series  = data[slabel]
        vals    = np.array([series.get(c, 0.0) for c in all_carriers])
        offsets = (s_idx - n_scenarios / 2 + 0.5) * bar_w
        x_pos   = x_centers + offsets

        for xi, (v, col) in enumerate(
            zip(vals, [_get_carrier_color(c, user_colors) for c in all_carriers])
        ):
            ax.bar(
                x_pos[xi], v, bar_w * 0.92,
                color=col, alpha=float(alphas[s_idx]),
                label=slabel if xi == 0 else "_",
                linewidth=0.4, edgecolor="white",
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(all_carriers, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Installierte Kapazität [GW]", fontsize=11)
    ax.set_xlabel("Energieträger (Strom)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(facecolor="#555555", alpha=float(alphas[i]), label=slabel)
        for i, slabel in enumerate(scenario_labels)
    ]
    ax.legend(handles=legend_handles, title="Szenario",
              loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"  [cap_by_country] Gespeichert: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Gestapelter Balken je Land (Szenarien auf x-Achse)
# ---------------------------------------------------------------------------

def plot_capacity_stacked_per_country(
    country: str,
    data: Dict[str, pd.Series],
    all_carriers: List[str],
    title: str,
    out_path: Path,
    user_colors: Optional[Dict] = None,
    save: bool = True,
) -> None:
    """Für EIN Land: x-Achse = Szenarien, Carrier gestapelt."""
    if not data:
        return

    scenario_labels = list(data.keys())
    n_scenarios     = len(scenario_labels)
    fig_w           = max(6, n_scenarios * 1.1 + 3)
    fig, ax         = plt.subplots(figsize=(fig_w, 6))

    x       = np.arange(n_scenarios)
    bottom  = np.zeros(n_scenarios)
    handles = []

    for carrier in all_carriers:
        vals = np.array([data[sl].get(carrier, 0.0) for sl in scenario_labels])
        if vals.sum() < 0.001:
            continue
        color = _get_carrier_color(carrier, user_colors)
        ax.bar(x, vals, bottom=bottom, color=color, width=0.65,
               linewidth=0.3, edgecolor="white")
        bottom += vals
        handles.append(mpatches.Patch(facecolor=color, label=carrier))

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Installierte Kapazität [GW]", fontsize=11)
    ax.set_xlabel("Szenario", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(1.01, 1), fontsize=8, ncol=1, frameon=True)

    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"  [cap_by_country] Gespeichert: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Diff-Plot je Land  (Szenario − Robust)
# ---------------------------------------------------------------------------

def plot_capacity_diff_per_country(
    country: str,
    diff_data: Dict[str, pd.Series],
    all_carriers: List[str],
    title: str,
    out_path: Path,
    user_colors: Optional[Dict] = None,
    save: bool = True,
) -> None:
    """Diff-Balken (Szenario − Robust) für EIN Land."""
    if not diff_data:
        return

    scenario_labels = list(diff_data.keys())
    n_scenarios     = len(scenario_labels)

    relevant = [
        c for c in all_carriers
        if any(abs(diff_data[sl].get(c, 0.0)) > 0.01 for sl in scenario_labels)
    ]
    if not relevant:
        return

    fig_w   = max(8, len(relevant) * (n_scenarios * 0.35 + 0.5) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    group_w   = 0.8
    bar_w     = group_w / n_scenarios
    x_centers = np.arange(len(relevant))
    colors_sc = plt.cm.tab10(np.linspace(0, 0.8, n_scenarios))

    for s_idx, slabel in enumerate(scenario_labels):
        vals    = np.array([diff_data[slabel].get(c, 0.0) for c in relevant])
        offsets = (s_idx - n_scenarios / 2 + 0.5) * bar_w
        ax.bar(x_centers + offsets, vals, bar_w * 0.92,
               color=colors_sc[s_idx], label=slabel,
               linewidth=0.4, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_centers)
    ax.set_xticklabels(relevant, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("ΔKapazität [GW]  (Szenario − Robust)", fontsize=11)
    ax.set_xlabel("Energieträger (Strom)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Szenario", loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"  [cap_by_country] Gespeichert: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Haupt-API
# ---------------------------------------------------------------------------

def run_capacity_by_country(
    n_robust: Optional[pypsa.Network],
    scenario_networks: Dict[str, pypsa.Network],
    output_dir: Path,
    run_name: str = "",
    user_colors: Optional[Dict] = None,
    plot_diff: bool = True,
    save: bool = True,
) -> None:
    """
    Erstellt alle elektrischen Kapazitäts-nach-Land-Plots.

    Struktur:
      output_dir/
        DE/
          grouped_by_carrier.png   ← Carrier auf x, Szenarien nebeneinander
          stacked_by_scenario.png  ← Szenarien auf x, Carrier gestapelt
          diff_vs_robust.png       ← Δ Szenario − Robust je Carrier
        FR/ ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_networks: Dict[str, pypsa.Network] = {}
    if n_robust is not None:
        all_networks["robust"] = n_robust
    for sc_name, n in scenario_networks.items():
        if n is not None:
            all_networks[sc_name] = n

    if not all_networks:
        print("  [cap_by_country] Keine Netzwerke verfügbar.")
        return

    label_map = build_scenario_label_map(list(all_networks.keys()))

    # FIX A+D+E: Vollständige elektrische Kapazität (Generatoren + Storage + Links,
    # nur Strom-Sektor)
    cap_data: Dict[str, pd.DataFrame] = {}
    for full_name, n in all_networks.items():
        df = extract_electricity_capacity(n)
        if not df.empty:
            cap_data[full_name] = df
        else:
            print(f"  [cap_by_country] Warnung: keine elektrischen Kapazitäten "
                  f"in '{full_name}' gefunden.")

    if not cap_data:
        print("  [cap_by_country] Keine Kapazitätsdaten. Prüfe ob p_nom_opt "
              "im Netzwerk vorhanden ist (wurde das Netz optimiert?).")
        return

    all_countries: List[str] = sorted(
        set().union(*[set(df.index) for df in cap_data.values()])
    )
    all_carriers_set = set().union(*[set(df.columns) for df in cap_data.values()])

    # Nur Carrier aus der Whitelist (sollte durch extract_electricity_capacity
    # bereits gefiltert sein, aber als zweite Absicherung)
    all_carriers_set = all_carriers_set & ELECTRICITY_CARRIERS

    all_carriers: List[str] = (
        [c for c in CARRIER_ORDER if c in all_carriers_set]
        + sorted(c for c in all_carriers_set if c not in CARRIER_ORDER)
    )

    robust_df = cap_data.get("robust")

    print(f"  [cap_by_country] {len(all_countries)} Länder, "
          f"{len(all_carriers)} Strom-Carrier, {len(cap_data)} Netzwerke")
    print(f"  [cap_by_country] Carrier: {all_carriers}")

    for country in all_countries:
        country_dir = output_dir / country
        country_dir.mkdir(parents=True, exist_ok=True)

        country_data: Dict[str, pd.Series] = {}
        for full_name, df in cap_data.items():
            slabel = label_map[full_name]
            country_data[slabel] = (
                df.loc[country] if country in df.index
                else pd.Series(dtype=float)
            )

        if all(s.sum() < 0.001 for s in country_data.values()):
            continue

        plot_capacity_grouped_by_carrier(
            country=country,
            data=country_data,
            label_map=label_map,
            all_carriers=all_carriers,
            title=f"{country} — Installierte elektr. Kapazität je Carrier [{run_name}]",
            out_path=country_dir / "grouped_by_carrier.png",
            user_colors=user_colors,
            save=save,
        )

        plot_capacity_stacked_per_country(
            country=country,
            data=country_data,
            all_carriers=all_carriers,
            title=f"{country} — Installierte elektr. Kapazität nach Szenario [{run_name}]",
            out_path=country_dir / "stacked_by_scenario.png",
            user_colors=user_colors,
            save=save,
        )

        if plot_diff and robust_df is not None:
            robust_series = (
                robust_df.loc[country] if country in robust_df.index
                else pd.Series(dtype=float)
            )
            diff_data: Dict[str, pd.Series] = {}
            for full_name, df in cap_data.items():
                if full_name == "robust":
                    continue
                slabel = label_map[full_name]
                sc_series = (
                    df.loc[country] if country in df.index
                    else pd.Series(dtype=float)
                )
                idx = sc_series.index.union(robust_series.index)
                diff_data[slabel] = (
                    sc_series.reindex(idx, fill_value=0)
                    - robust_series.reindex(idx, fill_value=0)
                )

            plot_capacity_diff_per_country(
                country=country,
                diff_data=diff_data,
                all_carriers=all_carriers,
                title=f"{country} — ΔKapazität elektr. (Szenario − Robust) [{run_name}]",
                out_path=country_dir / "diff_vs_robust.png",
                user_colors=user_colors,
                save=save,
            )

    print(f"\n  [cap_by_country] Fertig. Unterordner je Land in: {output_dir}")


# ---------------------------------------------------------------------------
# Standalone main()
# ---------------------------------------------------------------------------

def main():
    import argparse
    sys.path.insert(0, str(Path(__file__).parent))
    from master_config import MasterConfig, AROPlottingConfig
    from aro_analysis import AROAnalyzer

    parser = argparse.ArgumentParser(description="ARO Kapazitäten nach Land (Strom)")
    parser.add_argument("--run",     default=None)
    parser.add_argument("--output",  default=None)
    parser.add_argument("--no-diff", action="store_true")
    args, _ = parser.parse_known_args()

    master  = MasterConfig()
    aro_cfg = AROPlottingConfig(master=master)
    if args.run:
        aro_cfg.SELECTED_RUN = args.run

    analyzer = AROAnalyzer(config=aro_cfg, auto_find_dispatch=True)
    out_dir  = (
        Path(args.output) if args.output
        else aro_cfg.get_plot_output_dir("capacity_by_country")
    )

    run_capacity_by_country(
        n_robust=analyzer.n_robust,
        scenario_networks=analyzer.scenario_networks,
        output_dir=out_dir,
        run_name=aro_cfg.SELECTED_RUN,
        user_colors=aro_cfg.CARRIER_COLORS,
        plot_diff=not args.no_diff,
        save=True,
    )
    print(f"\n✓ Kapazitäten nach Land gespeichert in: {out_dir}")


if __name__ == "__main__":
    main()