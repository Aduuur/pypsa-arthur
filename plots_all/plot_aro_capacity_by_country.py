#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_aro_capacity_by_country.py
================================
Kapazitäten für ARO-Ergebnisse — ein Plot PRO LAND.

Jeder Plot zeigt:
  x-Achse  = Energieträger (Carrier)
  Säulen   = grouped bar: robust + stress_01, stress_02, ...

Zusätzlich wird für jedes Land ein gestapelter Vergleichsplot
(Carrier gestapelt, Szenarien nebeneinander) gespeichert.

Szenario-Namen werden auf stress_xy gekürzt:
  'networks/cutout_rcp45_2050_dunkelflaute'  →  stress_01
  'cutout_heatwave_2080'                     →  stress_02
  ...falls schon 'stress' im Namen:          →  stress_01 (beibehalten)

Standalone-Nutzung::
    python plots_all/plot_aro_capacity_by_country.py [--run RUN_KEY] [--output DIR]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pypsa

# ---------------------------------------------------------------------------
# Carrier-Reihenfolge (von links nach rechts in grouped-bar / von unten nach oben)
# ---------------------------------------------------------------------------
CARRIER_ORDER = [
    "offwind-ac", "offwind-dc", "onwind", "solar", "solar rooftop",
    "ror",
    "nuclear",
    "OCGT", "CCGT", "gas", "oil",
    "coal", "lignite",
    "H2 Electrolysis", "H2 Fuel Cell",
    "battery", "battery charger", "battery discharger",
    "PHS", "hydro",
    "biomass", "biogas",
    "load",
]

DEFAULT_COLORS: Dict[str, str] = {
    "offwind-ac":        "#6caedf",
    "offwind-dc":        "#2e86c1",
    "onwind":            "#74c476",
    "solar":             "#fdae6b",
    "solar rooftop":     "#fd8d3c",
    "ror":               "#9ecae1",
    "nuclear":           "#9467bd",
    "OCGT":              "#e7cb94",
    "CCGT":              "#c49c94",
    "gas":               "#c5b0d5",
    "oil":               "#aec7e8",
    "coal":              "#636363",
    "lignite":           "#393b79",
    "H2 Electrolysis":   "#17becf",
    "H2 Fuel Cell":      "#b5cf6b",
    "battery":           "#8c6d31",
    "battery charger":   "#8c6d31",
    "battery discharger":"#bd9e39",
    "PHS":               "#3182bd",
    "hydro":             "#6baed6",
    "biomass":           "#31a354",
    "biogas":            "#74c476",
    "load":              "#d62728",
}


# ---------------------------------------------------------------------------
# Szenario-Kurznamen  (stress_xy)
# ---------------------------------------------------------------------------

def _shorten_scenario_name(scenario_name: str, index: int) -> str:
    """
    Kürzt einen langen Cutout/Szenario-Namen auf 'stress_XX'.

    Logik:
    1. Falls der Name bereits 'stress_\\d+' enthält → diesen Teil extrahieren.
    2. Falls der Name einen Bezeichner wie 'rcp45', 'rcp85', 'ssp2', 'ssp5' enthält
       und zusätzlich ein Jahr → stress_<rcp>_<year> (noch recht kurz).
    3. Fallback: stress_{index:02d}

    Der 'index' ist 1-basiert (stress_01, stress_02, ...).
    """
    # Schon ein stress_XX drin?
    m = re.search(r'stress[_\-](\w+)', scenario_name, re.IGNORECASE)
    if m:
        return f"stress_{m.group(1)}"

    # RCP/SSP + Jahr extrahieren (z.B. rcp45_2050 → stress_rcp45)
    rcp_m = re.search(r'(rcp\d+|ssp\d+)', scenario_name, re.IGNORECASE)
    year_m = re.search(r'(\d{4})', scenario_name)
    if rcp_m:
        tag = rcp_m.group(1).lower()
        return f"stress_{tag}"

    # Generischer Fallback
    return f"stress_{index:02d}"


def build_scenario_label_map(
    scenario_names: List[str],
    include_robust: bool = True,
) -> Dict[str, str]:
    """
    Erstellt ein Mapping {voller_name → Kurzlabel}.
    'robust' bleibt immer 'robust'.
    Alle anderen werden zu stress_01, stress_02, ... (oder abgeleitet).
    """
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
    """Extrahiert das 2-Buchstaben-Länderkürzel aus dem Bus-Namen."""
    return str(bus_name)[:2].strip()


def _capacity_by_country(
    n: pypsa.Network,
    component: str = "generators",
    capacity_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gibt DataFrame [country × carrier] mit Kapazitäten in GW zurück.
    Unterstützt generators, storage_units, links.
    """
    comp = getattr(n, component, None)
    if comp is None or comp.empty:
        return pd.DataFrame()

    if capacity_col is None:
        if "p_nom_opt" in comp.columns:
            capacity_col = "p_nom_opt"
        elif "p_nom" in comp.columns:
            capacity_col = "p_nom"
        else:
            return pd.DataFrame()

    if "carrier" not in comp.columns:
        return pd.DataFrame()

    df = comp[["bus", "carrier", capacity_col]].copy()
    df["country"] = df["bus"].apply(_country_from_bus)
    df[capacity_col] = df[capacity_col] / 1e3  # MW → GW

    pivot = (
        df[df[capacity_col] > 0]
        .groupby(["country", "carrier"])[capacity_col]
        .sum()
        .unstack(fill_value=0)
    )
    return pivot


def _reorder_carriers(df: pd.DataFrame, carrier_order: List[str]) -> pd.DataFrame:
    """Sortiert Carrier-Spalten nach CARRIER_ORDER (unbekannte hinten)."""
    ordered = [c for c in carrier_order if c in df.columns]
    rest    = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


def _get_carrier_color(carrier: str, user_colors: Optional[Dict] = None) -> str:
    base = {**DEFAULT_COLORS, **(user_colors or {})}
    return base.get(carrier, "#a9a9a9")


# ---------------------------------------------------------------------------
# Plot 1: Grouped-bar je Land  (x = Carrier, Gruppen = Szenarien)
# ---------------------------------------------------------------------------

def plot_capacity_grouped_by_carrier(
    country: str,
    data: Dict[str, pd.Series],   # {scenario_label: Series[carrier → GW]}
    label_map: Dict[str, str],    # voller Name → Kurzlabel  (für Legende)
    all_carriers: List[str],
    title: str,
    out_path: Path,
    user_colors: Optional[Dict] = None,
    save: bool = True,
) -> None:
    """
    Für EIN Land: je Carrier eine Gruppe, je Szenario eine Säule.
    Säulenfarbe = Carrier-Farbe (abgestuft per Szenario via Alpha).
    """
    if not data:
        return

    scenario_labels = list(data.keys())   # bereits Kurzlabels
    n_scenarios     = len(scenario_labels)
    n_carriers      = len(all_carriers)
    if n_carriers == 0:
        return

    fig_w = max(10, n_carriers * (n_scenarios * 0.35 + 0.5) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    group_w   = 0.8
    bar_w     = group_w / n_scenarios
    x_centers = np.arange(n_carriers)

    # Alpha-Stufen für Szenarien (robust dunkel, stress_* heller)
    alphas = np.linspace(1.0, 0.45, n_scenarios)

    for s_idx, slabel in enumerate(scenario_labels):
        series = data[slabel]
        vals   = np.array([series.get(c, 0.0) for c in all_carriers])
        offsets = (s_idx - n_scenarios / 2 + 0.5) * bar_w
        x_pos  = x_centers + offsets

        colors_bars = [
            _get_carrier_color(c, user_colors) for c in all_carriers
        ]
        for xi, (v, col) in enumerate(zip(vals, colors_bars)):
            ax.bar(
                x_pos[xi], v, bar_w * 0.92,
                color=col, alpha=float(alphas[s_idx]),
                label=slabel if xi == 0 else "_",  # Legende nur einmal
                linewidth=0.4, edgecolor="white",
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(all_carriers, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Kapazität [GW]", fontsize=11)
    ax.set_xlabel("Energieträger", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(axis="y", alpha=0.3)

    # Legende: Szenarien
    legend_handles = [
        mpatches.Patch(
            facecolor="#555555",
            alpha=float(alphas[i]),
            label=slabel,
        )
        for i, slabel in enumerate(scenario_labels)
    ]
    ax.legend(
        handles=legend_handles,
        title="Szenario",
        loc="upper right",
        fontsize=8,
        frameon=True,
    )

    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"  [cap_by_country] Gespeichert: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Gestapelter Balken je Land (ein Balken pro Szenario)
# ---------------------------------------------------------------------------

def plot_capacity_stacked_per_country(
    country: str,
    data: Dict[str, pd.Series],   # {scenario_label: Series[carrier → GW]}
    all_carriers: List[str],
    title: str,
    out_path: Path,
    user_colors: Optional[Dict] = None,
    save: bool = True,
) -> None:
    """
    Für EIN Land: x-Achse = Szenarien, Carrier gestapelt.
    """
    if not data:
        return

    scenario_labels = list(data.keys())
    n_scenarios     = len(scenario_labels)
    fig_w           = max(6, n_scenarios * 1.1 + 3)
    fig, ax         = plt.subplots(figsize=(fig_w, 6))

    x      = np.arange(n_scenarios)
    bottom = np.zeros(n_scenarios)

    handles = []
    for carrier in all_carriers:
        vals  = np.array([data[sl].get(carrier, 0.0) for sl in scenario_labels])
        if vals.sum() < 0.001:
            continue
        color = _get_carrier_color(carrier, user_colors)
        ax.bar(x, vals, bottom=bottom, color=color, width=0.65,
               label=carrier, linewidth=0.3, edgecolor="white")
        bottom += vals
        handles.append(mpatches.Patch(facecolor=color, label=carrier))

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Kapazität [GW]", fontsize=11)
    ax.set_xlabel("Szenario", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        handles=handles, loc="upper left",
        bbox_to_anchor=(1.01, 1), fontsize=8, ncol=1, frameon=True,
    )

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
    diff_data: Dict[str, pd.Series],   # {scenario_label: diff-Series}
    all_carriers: List[str],
    title: str,
    out_path: Path,
    user_colors: Optional[Dict] = None,
    save: bool = True,
) -> None:
    """Diff-Balken (Szenario − Robust) für EIN Land, Carrier auf x-Achse."""
    if not diff_data:
        return

    scenario_labels = list(diff_data.keys())
    n_scenarios     = len(scenario_labels)
    n_carriers      = len(all_carriers)
    if n_carriers == 0:
        return

    # Nur Carrier mit messbaren Änderungen
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
        series  = diff_data[slabel]
        vals    = np.array([series.get(c, 0.0) for c in relevant])
        offsets = (s_idx - n_scenarios / 2 + 0.5) * bar_w
        x_pos   = x_centers + offsets
        color   = colors_sc[s_idx]
        ax.bar(x_pos, vals, bar_w * 0.92, color=color,
               label=slabel, linewidth=0.4, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x_centers)
    ax.set_xticklabels(relevant, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("ΔKapazität [GW]  (Szenario − Robust)", fontsize=11)
    ax.set_xlabel("Energieträger", fontsize=11)
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
    Erstellt alle Kapazitäts-nach-Land-Plots — EIN Unterordner je Land.

    Struktur:
      output_dir/
        DE/
          grouped_by_carrier.png     ← Carrier auf x-Achse, Szenarien nebeneinander
          stacked_by_scenario.png    ← Szenarien auf x-Achse, Carrier gestapelt
          diff_vs_robust.png         ← Δ Szenario − Robust je Carrier
        FR/
          ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Netzwerke sammeln ---
    all_networks: Dict[str, pypsa.Network] = {}
    if n_robust is not None:
        all_networks["robust"] = n_robust
    for sc_name, n in scenario_networks.items():
        if n is not None:
            all_networks[sc_name] = n

    if not all_networks:
        print("  [cap_by_country] Keine Netzwerke verfügbar.")
        return

    # --- Szenario-Kurzlabels aufbauen ---
    label_map = build_scenario_label_map(list(all_networks.keys()))
    # Mapping: Kurzlabel → voller Name (für Ausgabe-Dateinamen)
    short_to_full = {v: k for k, v in label_map.items()}

    # --- Kapazitäten extrahieren ---
    # cap_data: {full_name: DataFrame[country × carrier]}
    cap_data: Dict[str, pd.DataFrame] = {}
    for full_name, n in all_networks.items():
        df = _capacity_by_country(n)
        if not df.empty:
            cap_data[full_name] = df

    if not cap_data:
        print("  [cap_by_country] Keine Kapazitätsdaten in keinem Netzwerk.")
        return

    # --- Gemeinsame Länder und Carrier ---
    all_countries: List[str] = sorted(
        set().union(*[set(df.index) for df in cap_data.values()])
    )
    all_carriers_set = set().union(*[set(df.columns) for df in cap_data.values()])
    # Carrier in CARRIER_ORDER-Reihenfolge sortiert
    all_carriers: List[str] = (
        [c for c in CARRIER_ORDER if c in all_carriers_set]
        + sorted(c for c in all_carriers_set if c not in CARRIER_ORDER)
    )

    robust_df = cap_data.get("robust")

    print(f"  [cap_by_country] {len(all_countries)} Länder, "
          f"{len(all_carriers)} Carrier, {len(cap_data)} Netzwerke")

    # --- Pro Land plotten ---
    for country in all_countries:
        country_dir = output_dir / country
        country_dir.mkdir(parents=True, exist_ok=True)

        # Daten je Kurzlabel sammeln: {kurzlabel → Series[carrier → GW]}
        country_data: Dict[str, pd.Series] = {}
        for full_name, df in cap_data.items():
            slabel = label_map[full_name]
            if country in df.index:
                country_data[slabel] = df.loc[country]
            else:
                country_data[slabel] = pd.Series(dtype=float)

        if all(s.sum() < 0.001 for s in country_data.values()):
            continue  # Land ohne nennenswerte Kapazität überspringen

        # Plot 1: Grouped-bar (Carrier auf x, Szenarien nebeneinander)
        plot_capacity_grouped_by_carrier(
            country=country,
            data=country_data,
            label_map=label_map,
            all_carriers=all_carriers,
            title=f"{country} — Kapazität je Carrier [{run_name}]",
            out_path=country_dir / "grouped_by_carrier.png",
            user_colors=user_colors,
            save=save,
        )

        # Plot 2: Gestapelter Balken (Szenarien auf x, Carrier gestapelt)
        plot_capacity_stacked_per_country(
            country=country,
            data=country_data,
            all_carriers=all_carriers,
            title=f"{country} — Kapazität nach Szenario [{run_name}]",
            out_path=country_dir / "stacked_by_scenario.png",
            user_colors=user_colors,
            save=save,
        )

        # Plot 3: Diff-Plot (Szenario − Robust)
        if plot_diff and robust_df is not None:
            robust_series = (
                robust_df.loc[country]
                if country in robust_df.index
                else pd.Series(dtype=float)
            )
            diff_data: Dict[str, pd.Series] = {}
            for full_name, df in cap_data.items():
                if full_name == "robust":
                    continue
                slabel = label_map[full_name]
                sc_series = df.loc[country] if country in df.index else pd.Series(dtype=float)
                # Diff auf gemeinsamen Carrier-Index
                idx = sc_series.index.union(robust_series.index)
                diff_data[slabel] = (
                    sc_series.reindex(idx, fill_value=0)
                    - robust_series.reindex(idx, fill_value=0)
                )

            plot_capacity_diff_per_country(
                country=country,
                diff_data=diff_data,
                all_carriers=all_carriers,
                title=f"{country} — ΔKapazität (Szenario − Robust) [{run_name}]",
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

    parser = argparse.ArgumentParser(description="ARO Kapazitäten nach Land")
    parser.add_argument("--run",    default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-diff", action="store_true")
    args, _ = parser.parse_known_args()

    master  = MasterConfig()
    aro_cfg = AROPlottingConfig(master=master)
    if args.run:
        aro_cfg.SELECTED_RUN = args.run

    analyzer = AROAnalyzer(config=aro_cfg, auto_find_dispatch=True)

    out_dir = Path(args.output) if args.output else aro_cfg.get_plot_output_dir("capacity_by_country")

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
