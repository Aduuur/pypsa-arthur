#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_aro_capacity_by_country.py
================================
Kapazitäten nach Land aufgesplittet für ARO-Ergebnisse.

Erzeugt je nach verfügbaren Netzwerken:
  1. Robustes Portfolio — Kapazität [GW] je Land und Carrier (gestapelter Balken)
  2. Alle Szenario-Dispatch-Netzwerke — gleiche Darstellung
  3. Diff: Szenario vs. robustes Portfolio (Abweichung in GW, Farbe = pos/neg)

Kann standalone oder über run_analysis.py / AROAnalyzer aufgerufen werden.

Standalone-Nutzung::

    python plots_all/plot_aro_capacity_by_country.py [--run RUN_KEY] [--output DIR]
"""

from __future__ import annotations

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
# Carrier-Reihenfolge (von unten nach oben im gestapelten Balken)
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

# Carrier → Farbe (Fallback: Grau)
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
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _country_from_bus(bus_name: str) -> str:
    """Extrahiert das 2-Buchstaben Länderkürzel aus dem Bus-Namen."""
    # Typisches PyPSA-EUR-Format: 'DE0 0', 'FR1 1', 'ES 0 0' etc.
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

    # Kapazitätsspalte bestimmen
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


def _get_colors(carriers: List[str], user_colors: Optional[Dict] = None) -> List[str]:
    base = {**DEFAULT_COLORS, **(user_colors or {})}
    return [base.get(c, "#a9a9a9") for c in carriers]


# ---------------------------------------------------------------------------
# Plot-Funktion: gestapelter Balken je Land
# ---------------------------------------------------------------------------

def plot_capacity_stacked(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    user_colors: Optional[Dict] = None,
    figsize_per_country: float = 0.9,
    ylabel: str = "Kapazität [GW]",
    save: bool = True,
) -> None:
    """Gestapeltes Balkendiagramm: Länder auf x-Achse, Carrier gestapelt."""
    if df.empty:
        print(f"  [cap_by_country] Keine Daten für: {title}")
        return

    df = _reorder_carriers(df, CARRIER_ORDER)
    n_countries = len(df)
    figwidth    = max(8, n_countries * figsize_per_country + 3)
    fig, ax     = plt.subplots(figsize=(figwidth, 7))

    colors  = _get_colors(df.columns.tolist(), user_colors)
    bottom  = np.zeros(n_countries)
    x       = np.arange(n_countries)

    for carrier, color in zip(df.columns, colors):
        vals = df[carrier].values
        ax.bar(x, vals, bottom=bottom, label=carrier, color=color, width=0.7)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(df.index.tolist(), rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Land", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.3)

    # Legende außerhalb
    handles = [
        mpatches.Patch(facecolor=_get_colors([c], user_colors)[0], label=c)
        for c in df.columns
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
        ncol=1,
        frameon=True,
    )

    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  [cap_by_country] Gespeichert: {out_path}")
    plt.close(fig)


def plot_capacity_diff(
    df_scenario: pd.DataFrame,
    df_robust: pd.DataFrame,
    scenario_name: str,
    out_path: Path,
    user_colors: Optional[Dict] = None,
    save: bool = True,
) -> None:
    """
    Diff-Plot: Kapazität Szenario minus robustes Portfolio je Land.
    Positive Balken = mehr Kapazität im Szenario, negativ = weniger.
    """
    if df_scenario.empty or df_robust.empty:
        return

    # Align auf gemeinsame Länder und Carrier
    all_countries = df_scenario.index.union(df_robust.index)
    all_carriers  = df_scenario.columns.union(df_robust.columns)
    s = df_scenario.reindex(index=all_countries, columns=all_carriers, fill_value=0)
    r = df_robust.reindex(index=all_countries, columns=all_carriers, fill_value=0)
    diff = s - r
    diff = diff.loc[:, (diff.abs() > 0.01).any()]  # Carrier ohne Änderung raus

    if diff.empty:
        return

    diff = _reorder_carriers(diff, CARRIER_ORDER)
    n_countries = len(diff)
    fig, ax     = plt.subplots(figsize=(max(8, n_countries * 0.9 + 3), 7))
    x           = np.arange(n_countries)

    pos_bottom = np.zeros(n_countries)
    neg_bottom = np.zeros(n_countries)

    colors = _get_colors(diff.columns.tolist(), user_colors)
    for carrier, color in zip(diff.columns, colors):
        vals = diff[carrier].values
        pos  = np.where(vals > 0, vals, 0)
        neg  = np.where(vals < 0, vals, 0)
        if pos.any():
            ax.bar(x, pos, bottom=pos_bottom, label=carrier, color=color, width=0.7)
            pos_bottom += pos
        if neg.any():
            ax.bar(x, neg, bottom=neg_bottom, color=color, width=0.7)
            neg_bottom += neg

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(diff.index.tolist(), rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Land", fontsize=11)
    ax.set_ylabel("ΔKapazität [GW]  (Szenario − Robust)", fontsize=11)
    ax.set_title(f"Kapazitätsabweichung: {scenario_name} vs. Robust", fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.3)

    handles = [
        mpatches.Patch(facecolor=_get_colors([c], user_colors)[0], label=c)
        for c in diff.columns
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=8, ncol=1, frameon=True)

    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  [cap_by_country] Gespeichert: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Haupt-API (wird von AROAnalyzer / run_analysis aufgerufen)
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
    Erstellt alle Kapazitäts-nach-Land-Plots für ARO-Ergebnisse.

    Parameter
    ----------
    n_robust          : robustes Portfolio-Netzwerk (oder None)
    scenario_networks : dict Szenario → pypsa.Network
    output_dir        : Ausgabeverzeichnis (wird angelegt falls nötig)
    run_name          : Für Plot-Titel
    user_colors       : Carrier-Farben aus master_config
    plot_diff         : Diff-Plot Szenario vs. robust (nur wenn n_robust vorhanden)
    save              : Plots speichern
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Robustes Portfolio
    df_robust: Optional[pd.DataFrame] = None
    if n_robust is not None:
        df_robust = _capacity_by_country(n_robust)
        if not df_robust.empty:
            plot_capacity_stacked(
                df=df_robust,
                title=f"Robustes Portfolio — Kapazität je Land [{run_name}]",
                out_path=output_dir / "robust_capacity_by_country.png",
                user_colors=user_colors,
                save=save,
            )
        else:
            print("  [cap_by_country] Robustes Portfolio: keine Kapazitätsdaten.")

    # 2. Alle Szenario-Dispatch-Netzwerke
    for scenario_name, n in scenario_networks.items():
        if n is None:
            print(f"  [cap_by_country] Szenario '{scenario_name}' nicht geladen — übersprungen.")
            continue

        df_sc = _capacity_by_country(n)
        if df_sc.empty:
            print(f"  [cap_by_country] Szenario '{scenario_name}': keine Kapazitätsdaten.")
            continue

        safe_name = scenario_name.replace("/", "_").replace(" ", "_")
        out_sc = output_dir / f"capacity_by_country_{safe_name}.png"
        plot_capacity_stacked(
            df=df_sc,
            title=f"Szenario: {scenario_name} — Kapazität je Land [{run_name}]",
            out_path=out_sc,
            user_colors=user_colors,
            save=save,
        )

        # Diff-Plot vs. Robust
        if plot_diff and df_robust is not None and not df_robust.empty:
            plot_capacity_diff(
                df_scenario=df_sc,
                df_robust=df_robust,
                scenario_name=scenario_name,
                out_path=output_dir / f"capacity_diff_{safe_name}_vs_robust.png",
                user_colors=user_colors,
                save=save,
            )


# ---------------------------------------------------------------------------
# Standalone main()
# ---------------------------------------------------------------------------

def main():
    """Entry-Point für run_analysis.py STANDALONE_SCRIPTS."""
    import argparse
    sys.path.insert(0, str(Path(__file__).parent))

    from master_config import MasterConfig, AROPlottingConfig
    from aro_analysis import AROAnalyzer

    parser = argparse.ArgumentParser(description="ARO Kapazitäten nach Land")
    parser.add_argument("--run",    default=None, help="ARO Run Key")
    parser.add_argument("--output", default=None, help="Output-Verzeichnis")
    parser.add_argument("--no-diff", action="store_true", help="Kein Diff-Plot")
    args, _ = parser.parse_known_args()

    master = MasterConfig()
    aro_cfg = AROPlottingConfig(master=master)
    if args.run:
        aro_cfg.SELECTED_RUN = args.run

    analyzer = AROAnalyzer(config=aro_cfg, auto_find_dispatch=True)

    out_dir = Path(args.output) if args.output else (
        aro_cfg.get_plot_output_dir("capacity_by_country")
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
