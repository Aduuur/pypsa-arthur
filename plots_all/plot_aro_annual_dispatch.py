#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_aro_annual_dispatch.py
============================
Jährlicher Dispatch je ARO-Szenario.

Erzeugt:
  1. Gestapelter Balken: jährliche Erzeugung [TWh] je Szenario und Carrier
     – getrennte Balken für Erzeugung (positiv) und Last/Speicher-Laden (negativ)
  2. Heatmap: Carrier × Szenario (Energie in TWh)
  3. Scatter: Kapazitätsfaktor (Onwind+Solar) vs. Systemkosten je Szenario

Kann standalone oder über run_analysis.py / AROAnalyzer aufgerufen werden.

Standalone-Nutzung::

    python plots_all/plot_aro_annual_dispatch.py [--run RUN_KEY] [--output DIR]
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
# Carrier-Reihenfolge und Farben (identisch zu capacity-Skript)
# ---------------------------------------------------------------------------
CARRIER_ORDER = [
    "offwind-ac", "offwind-dc", "onwind", "solar", "solar rooftop",
    "ror", "nuclear",
    "OCGT", "CCGT", "gas", "oil", "coal", "lignite",
    "H2 Electrolysis", "H2 Fuel Cell",
    "battery", "PHS", "hydro",
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
    "PHS":               "#3182bd",
    "hydro":             "#6baed6",
    "biomass":           "#31a354",
    "biogas":            "#74c476",
    "load":              "#d62728",
}


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _get_colors(
    carriers: List[str], user_colors: Optional[Dict] = None
) -> List[str]:
    base = {**DEFAULT_COLORS, **(user_colors or {})}
    return [base.get(c, "#a9a9a9") for c in carriers]


def _annual_dispatch(
    n: pypsa.Network,
    unit: float = 1e6,   # MWh → TWh
) -> pd.Series:
    """
    Berechnet die jährliche Erzeugung je Carrier [TWh].
    Berücksichtigt generators_t.p, storage_units_t.p_dispatch,
    und links_t.p0 (positive Seite = Erzeugung).
    Gibt nur positive Werte zurück (Erzeugung, kein Verbrauch).
    """
    records: Dict[str, float] = {}

    # --- Generatoren ---
    if not n.generators.empty and not n.generators_t.p.empty:
        gen_p = n.generators_t.p  # shape: (T, n_gen)
        # Aggregiere: sum über Zeit, dann nach Carrier gruppieren
        gen_energy = gen_p.sum()                    # Series: index=gen_name
        carrier_map = n.generators["carrier"]
        by_carrier = (
            gen_energy
            .groupby(carrier_map)
            .sum()
            / unit
        )
        for c, val in by_carrier.items():
            records[c] = records.get(c, 0.0) + val

    # --- Speicher (Storage Units) ---
    if not n.storage_units.empty:
        su = n.storage_units
        carrier_map_su = su["carrier"] if "carrier" in su.columns else None

        if carrier_map_su is not None and not n.storage_units_t.p_dispatch.empty:
            dispatch = n.storage_units_t.p_dispatch.sum() / unit
            by_carrier_su = dispatch.groupby(carrier_map_su).sum()
            for c, val in by_carrier_su.items():
                records[c] = records.get(c, 0.0) + max(0.0, val)

    # --- Links (z.B. H2 Fuel Cell, CCGT über Link) ---
    if not n.links.empty and not n.links_t.p0.empty and "carrier" in n.links.columns:
        link_p0   = n.links_t.p0.sum() / unit
        carrier_map_l = n.links["carrier"]
        by_carrier_l = link_p0.groupby(carrier_map_l).sum()
        # Nur positive Links = Erzeugung/Verbrauch-Seite
        for c, val in by_carrier_l.items():
            if val > 0:
                records[c] = records.get(c, 0.0) + val

    result = pd.Series(records, name="energy_TWh")
    return result[result > 0].sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Plot 1: Gestapelter Balken jährlicher Dispatch je Szenario
# ---------------------------------------------------------------------------

def plot_annual_dispatch_stacked(
    dispatch_dict: Dict[str, pd.Series],
    run_name: str,
    out_path: Path,
    user_colors: Optional[Dict] = None,
    cost_dict: Optional[Dict[str, float]] = None,
    save: bool = True,
) -> None:
    """
    Gestapelter Balken: x = Szenario, y = Energie [TWh], Farben = Carrier.
    Optional: Systemkosten als Linie auf Sekundärachse.
    """
    if not dispatch_dict:
        print("  [annual_dispatch] Keine Daten für gestapelten Balken.")
        return

    # Alle Carrier über alle Szenarien sammeln
    all_carriers: List[str] = []
    for s in dispatch_dict.values():
        for c in s.index:
            if c not in all_carriers:
                all_carriers.append(c)

    # Carrier sortieren
    ordered = [c for c in CARRIER_ORDER if c in all_carriers]
    rest    = [c for c in all_carriers if c not in ordered]
    all_carriers = ordered + rest

    scenarios  = list(dispatch_dict.keys())
    n_sc       = len(scenarios)
    figwidth   = max(10, n_sc * 0.9 + 4)

    fig, ax1 = plt.subplots(figsize=(figwidth, 7))
    x        = np.arange(n_sc)
    bottom   = np.zeros(n_sc)

    colors = _get_colors(all_carriers, user_colors)
    for carrier, color in zip(all_carriers, colors):
        vals = np.array(
            [dispatch_dict[sc].get(carrier, 0.0) for sc in scenarios]
        )
        ax1.bar(x, vals, bottom=bottom, label=carrier, color=color, width=0.7)
        bottom += vals

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("Szenario", fontsize=11)
    ax1.set_ylabel("Jährliche Erzeugung [TWh]", fontsize=11)
    ax1.set_title(
        f"Jährlicher Dispatch je Szenario [{run_name}]",
        fontsize=13, pad=10,
    )
    ax1.grid(axis="y", alpha=0.3)

    # Sekundärachse: Systemkosten
    if cost_dict:
        costs = [cost_dict.get(sc, np.nan) / 1e9 for sc in scenarios]  # → Mrd. €
        ax2 = ax1.twinx()
        ax2.plot(x, costs, "D--", color="#d62728", linewidth=1.8,
                 markersize=6, label="Systemkosten [Mrd. €/a]")
        ax2.set_ylabel("Systemkosten [Mrd. €/a]", fontsize=11, color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.legend(loc="upper right", fontsize=9)

    # Legende Carrier außerhalb
    handles = [
        mpatches.Patch(
            facecolor=_get_colors([c], user_colors)[0], label=c
        )
        for c in all_carriers
    ]
    ax1.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=8,
        ncol=1,
        frameon=True,
    )

    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  [annual_dispatch] Gespeichert: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Heatmap Carrier × Szenario
# ---------------------------------------------------------------------------

def plot_dispatch_heatmap(
    dispatch_dict: Dict[str, pd.Series],
    run_name: str,
    out_path: Path,
    save: bool = True,
) -> None:
    """Heatmap: Carrier (Zeilen) × Szenario (Spalten), Farbtiefe = TWh."""
    if not dispatch_dict:
        return

    df = pd.DataFrame(dispatch_dict).fillna(0)  # index=carrier, columns=scenario
    # Carrier mit Gesamterzeugung > 0.1 TWh behalten
    df = df.loc[df.sum(axis=1) > 0.1]
    # Sortieren nach Gesamterzeugung absteigend
    df = df.loc[df.sum(axis=1).sort_values(ascending=False).index]

    if df.empty:
        return

    n_carriers  = len(df)
    n_scenarios = len(df.columns)
    figheight   = max(5, n_carriers * 0.45 + 2)
    figwidth    = max(8, n_scenarios * 0.7 + 3)

    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    im = ax.imshow(df.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(n_scenarios))
    ax.set_xticklabels(df.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_carriers))
    ax.set_yticklabels(df.index.tolist(), fontsize=8)
    ax.set_title(f"Heatmap Jährlicher Dispatch [TWh] — {run_name}", fontsize=13, pad=10)

    # Werte in Zellen
    for i in range(n_carriers):
        for j in range(n_scenarios):
            val = df.values[i, j]
            if val > 0.1:
                ax.text(j, i, f"{val:.0f}",
                        ha="center", va="center", fontsize=6,
                        color="black" if val < df.values.max() * 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Energie [TWh/a]", shrink=0.8)
    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  [annual_dispatch] Gespeichert: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Kapazitätsfaktor vs. Kosten Scatter
# ---------------------------------------------------------------------------

def plot_cf_vs_cost_scatter(
    dispatch_dict: Dict[str, pd.Series],
    capacity_dict: Dict[str, pd.Series],
    cost_dict: Dict[str, float],
    run_name: str,
    out_path: Path,
    res_carriers: Optional[List[str]] = None,
    save: bool = True,
) -> None:
    """
    Scatter: x = Kapazitätsfaktor RES (onwind + solar), y = Systemkosten.
    Jeder Punkt = 1 Szenario.
    """
    if not dispatch_dict or not cost_dict:
        return

    if res_carriers is None:
        res_carriers = ["onwind", "offwind-ac", "offwind-dc", "solar", "solar rooftop", "ror"]

    records = []
    for sc, dispatch in dispatch_dict.items():
        cost = cost_dict.get(sc)
        if cost is None:
            continue

        cap_series = capacity_dict.get(sc, pd.Series(dtype=float))
        total_res_cap  = cap_series.loc[
            cap_series.index.intersection(res_carriers)
        ].sum()  # GW
        total_res_gen  = dispatch.loc[
            dispatch.index.intersection(res_carriers)
        ].sum()  # TWh/a

        # Kapazitätsfaktor [h/a] = Energie [MWh] / Kapazität [MW]
        # = TWh * 1e6 MWh/TWh  /  GW * 1e3 MW/GW
        cf = (total_res_gen * 1e6) / (total_res_cap * 1e3) if total_res_cap > 0 else np.nan
        records.append({
            "scenario": sc,
            "cf_h": cf,
            "cost_bn": cost / 1e9,
        })

    if not records:
        return

    df = pd.DataFrame(records).dropna(subset=["cf_h"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        df["cf_h"], df["cost_bn"],
        s=80, c=df["cost_bn"], cmap="RdYlGn_r", alpha=0.85, zorder=3
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["scenario"],
            (row["cf_h"], row["cost_bn"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=6, alpha=0.85,
        )

    plt.colorbar(scatter, ax=ax, label="Systemkosten [Mrd. €/a]")
    ax.set_xlabel("Kapazitätsfaktor RES [h/a]", fontsize=11)
    ax.set_ylabel("Systemkosten [Mrd. €/a]", fontsize=11)
    ax.set_title(
        f"RES Kapazitätsfaktor vs. Systemkosten je Szenario [{run_name}]",
        fontsize=13, pad=10,
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  [annual_dispatch] Gespeichert: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Haupt-API
# ---------------------------------------------------------------------------

def run_annual_dispatch(
    scenario_networks: Dict[str, pypsa.Network],
    output_dir: Path,
    run_name: str = "",
    user_colors: Optional[Dict] = None,
    cost_dict: Optional[Dict[str, float]] = None,
    save: bool = True,
) -> None:
    """
    Erstellt alle jährlichen-Dispatch-Plots für alle ARO-Szenarien.

    Parameter
    ----------
    scenario_networks : dict Szenario → pypsa.Network
    output_dir        : Ausgabeverzeichnis
    run_name          : Für Plot-Titel
    user_colors       : Carrier-Farben aus master_config
    cost_dict         : Szenario → Systemkosten [€/a] (aus aro_summary)
    save              : Plots speichern
    """
    if not scenario_networks:
        print("  [annual_dispatch] Keine Szenario-Netzwerke — übersprungen.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch und Kapazitäten für alle Szenarien berechnen
    dispatch_dict:  Dict[str, pd.Series] = {}
    capacity_dict:  Dict[str, pd.Series] = {}

    for sc_name, n in scenario_networks.items():
        if n is None:
            print(f"  [annual_dispatch] Szenario '{sc_name}' nicht geladen — übersprungen.")
            continue
        try:
            dispatch_dict[sc_name] = _annual_dispatch(n)
        except Exception as e:
            print(f"  [annual_dispatch] Dispatch-Berechnung für '{sc_name}' fehlgeschlagen: {e}")
            continue

        # Kapazitäten für Scatter-Plot (GW je Carrier)
        try:
            from plot_aro_capacity_by_country import _capacity_by_country
            df_cap = _capacity_by_country(n)
            if not df_cap.empty:
                capacity_dict[sc_name] = df_cap.sum()  # Summiere über Länder → Series[carrier]
        except Exception:
            capacity_dict[sc_name] = pd.Series(dtype=float)

    if not dispatch_dict:
        print("  [annual_dispatch] Alle Dispatch-Berechnungen fehlgeschlagen.")
        return

    # --- Plot 1: Gestapelter Balken ---
    plot_annual_dispatch_stacked(
        dispatch_dict=dispatch_dict,
        run_name=run_name,
        out_path=output_dir / "annual_dispatch_stacked.png",
        user_colors=user_colors,
        cost_dict=cost_dict,
        save=save,
    )

    # --- Plot 2: Heatmap ---
    plot_dispatch_heatmap(
        dispatch_dict=dispatch_dict,
        run_name=run_name,
        out_path=output_dir / "annual_dispatch_heatmap.png",
        save=save,
    )

    # --- Plot 3: Kapazitätsfaktor vs. Kosten Scatter ---
    if cost_dict:
        plot_cf_vs_cost_scatter(
            dispatch_dict=dispatch_dict,
            capacity_dict=capacity_dict,
            cost_dict=cost_dict,
            run_name=run_name,
            out_path=output_dir / "res_cf_vs_cost_scatter.png",
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

    parser = argparse.ArgumentParser(description="Jährlicher ARO Dispatch je Szenario")
    parser.add_argument("--run",    default=None, help="ARO Run Key")
    parser.add_argument("--output", default=None, help="Output-Verzeichnis")
    args, _ = parser.parse_known_args()

    master  = MasterConfig()
    aro_cfg = AROPlottingConfig(master=master)
    if args.run:
        aro_cfg.SELECTED_RUN = args.run

    analyzer = AROAnalyzer(config=aro_cfg, auto_find_dispatch=True)

    # Kosten aus aro_summary
    cost_dict = (
        analyzer.aro_summary
        .get("aro_final_evaluation", {})
        .get("all_costs", {})
    ) or {}

    out_dir = Path(args.output) if args.output else (
        aro_cfg.get_plot_output_dir("annual_dispatch")
    )

    run_annual_dispatch(
        scenario_networks=analyzer.scenario_networks,
        output_dir=out_dir,
        run_name=aro_cfg.SELECTED_RUN,
        user_colors=aro_cfg.CARRIER_COLORS,
        cost_dict=cost_dict,
        save=True,
    )
    print(f"\n✓ Jährlicher Dispatch gespeichert in: {out_dir}")


if __name__ == "__main__":
    main()
