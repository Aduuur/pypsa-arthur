#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_aro_annual_dispatch.py
============================
Jährlicher Dispatch-Vergleich über alle ARO-Szenarien.

Szenario-Labels werden auf 'stress_xy' gekürzt (gleiche Logik wie
plot_aro_capacity_by_country._shorten_scenario_name).

Plots:
  1. Gestapelter Balken: jährliche Erzeugung je Szenario (stress_xy)
  2. Heatmap: Erzeugung × Szenario  (Carrier-Zeilen, Szenario-Spalten)
  3. Scatter: Kapazitätsfaktor vs. Systemkosten je Szenario

Standalone::
    python plots_all/plot_aro_annual_dispatch.py [--run KEY] [--output DIR]
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

# reuse shorten logic from capacity module
sys.path.insert(0, str(Path(__file__).parent))
from plot_aro_capacity_by_country import _shorten_scenario_name, build_scenario_label_map


# ---------------------------------------------------------------------------
# Carrier-Reihenfolge (Erzeugung oben, Last unten)
# ---------------------------------------------------------------------------
GEN_CARRIER_ORDER = [
    "solar", "solar rooftop", "onwind", "offwind-ac", "offwind-dc",
    "ror", "hydro", "nuclear",
    "biomass", "biogas",
    "CCGT", "OCGT", "gas", "oil", "coal", "lignite",
    "H2 Fuel Cell",
    "battery discharger",
    "PHS",
]

DEFAULT_GEN_COLORS: Dict[str, str] = {
    "solar":             "#fdae6b",
    "solar rooftop":     "#fd8d3c",
    "onwind":            "#74c476",
    "offwind-ac":        "#6caedf",
    "offwind-dc":        "#2e86c1",
    "ror":               "#9ecae1",
    "hydro":             "#6baed6",
    "nuclear":           "#9467bd",
    "biomass":           "#31a354",
    "biogas":            "#74c476",
    "CCGT":              "#c49c94",
    "OCGT":              "#e7cb94",
    "gas":               "#c5b0d5",
    "oil":               "#aec7e8",
    "coal":              "#636363",
    "lignite":           "#393b79",
    "H2 Fuel Cell":      "#b5cf6b",
    "battery discharger":"#bd9e39",
    "PHS":               "#3182bd",
}


# ---------------------------------------------------------------------------
# Daten-Extraktion
# ---------------------------------------------------------------------------

def _extract_annual_generation(
    n: pypsa.Network,
) -> pd.Series:
    """
    Jährliche Erzeugung [TWh] je Carrier aus generators_t.p.
    Speicher werden separat als positiver Dispatch erfasst.
    """
    result: Dict[str, float] = {}

    # Generatoren
    if hasattr(n, "generators_t") and hasattr(n.generators_t, "p") and not n.generators_t.p.empty:
        gen_p = n.generators_t.p
        for gen_name in gen_p.columns:
            if gen_name in n.generators.index:
                carrier = n.generators.loc[gen_name, "carrier"]
                result[carrier] = result.get(carrier, 0.0) + gen_p[gen_name].clip(lower=0).sum()

    # Speicher (nur Entladung positiv)
    if hasattr(n, "storage_units_t") and hasattr(n.storage_units_t, "p") \
            and not n.storage_units_t.p.empty:
        sto_p = n.storage_units_t.p
        for su_name in sto_p.columns:
            if su_name in n.storage_units.index:
                carrier = n.storage_units.loc[su_name, "carrier"]
                result[carrier] = result.get(carrier, 0.0) + sto_p[su_name].clip(lower=0).sum()

    # MWh → TWh
    dt = n.snapshot_weightings.generators.values[0] if not n.snapshot_weightings.empty else 1.0
    return pd.Series({k: v * dt / 1e6 for k, v in result.items()})


def _extract_system_cost(
    n: pypsa.Network,
) -> float:
    """Gesamtsystemkosten [Mrd. EUR] — Summe aller Kapitalkosten + variable Kosten."""
    total = 0.0
    try:
        # Kapitalkosten
        for comp_name in ("generators", "storage_units", "links", "lines"):
            comp = getattr(n, comp_name)
            if comp.empty:
                continue
            if "capital_cost" in comp.columns and "p_nom_opt" in comp.columns:
                total += (comp["capital_cost"] * comp["p_nom_opt"]).sum()
        # Variable Kosten (approx über generators_t.p und marginal_cost)
        if not n.generators_t.p.empty:
            mc = n.generators.get("marginal_cost", pd.Series(0, index=n.generators.index))
            dt = n.snapshot_weightings.generators.values[0] if not n.snapshot_weightings.empty else 1.0
            total += (n.generators_t.p.multiply(mc, axis=1).sum().sum() * dt)
    except Exception:
        pass
    return total / 1e9  # EUR → Mrd. EUR


def _capacity_factor(
    n: pypsa.Network,
) -> float:
    """Mittlerer Kapazitätsfaktor der VRE-Generatoren."""
    vre_carriers = {"solar", "solar rooftop", "onwind", "offwind-ac", "offwind-dc", "ror"}
    try:
        mask = n.generators["carrier"].isin(vre_carriers)
        gens = n.generators[mask]
        if gens.empty or n.generators_t.p.empty:
            return float("nan")
        vre_cols = [g for g in gens.index if g in n.generators_t.p.columns]
        if not vre_cols:
            return float("nan")
        dt = n.snapshot_weightings.generators.values[0] if not n.snapshot_weightings.empty else 1.0
        p_nom_opt = gens.loc[vre_cols, "p_nom_opt"] if "p_nom_opt" in gens.columns else gens.loc[vre_cols, "p_nom"]
        total_gen  = n.generators_t.p[vre_cols].sum().sum() * dt
        total_cap  = p_nom_opt.sum() * len(n.snapshots) * dt
        return float(total_gen / total_cap) if total_cap > 0 else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Plot 1: Gestapelter Balken — jährliche Erzeugung je Szenario
# ---------------------------------------------------------------------------

def plot_annual_generation_stacked(
    gen_data: Dict[str, pd.Series],   # {scenario_label → Series[carrier → TWh]}
    all_carriers: List[str],
    title: str,
    out_path: Path,
    user_colors: Optional[Dict] = None,
    save: bool = True,
) -> None:
    """Gestapelter Balken: x = Szenarien (stress_xy), Carrier gestapelt."""
    if not gen_data:
        return

    slabels  = list(gen_data.keys())
    fig_w    = max(6, len(slabels) * 1.1 + 3)
    fig, ax  = plt.subplots(figsize=(fig_w, 6))
    x        = np.arange(len(slabels))
    bottom   = np.zeros(len(slabels))
    base_col = {**DEFAULT_GEN_COLORS, **(user_colors or {})}

    handles = []
    for carrier in all_carriers:
        vals = np.array([gen_data[sl].get(carrier, 0.0) for sl in slabels])
        if vals.sum() < 0.001:
            continue
        color = base_col.get(carrier, "#a9a9a9")
        ax.bar(x, vals, bottom=bottom, color=color, width=0.65, label=carrier,
               linewidth=0.3, edgecolor="white")
        bottom += vals
        handles.append(mpatches.Patch(facecolor=color, label=carrier))

    ax.set_xticks(x)
    ax.set_xticklabels(slabels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Jährl. Erzeugung [TWh]", fontsize=11)
    ax.set_xlabel("Szenario", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(1.01, 1), fontsize=8, frameon=True)
    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"  [annual_dispatch] Gespeichert: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Heatmap  Carrier × Szenario
# ---------------------------------------------------------------------------

def plot_generation_heatmap(
    gen_data: Dict[str, pd.Series],
    all_carriers: List[str],
    title: str,
    out_path: Path,
    save: bool = True,
) -> None:
    """Heatmap: Zeilen = Carrier, Spalten = Szenarien (stress_xy), Werte = TWh."""
    if not gen_data:
        return

    slabels = list(gen_data.keys())
    matrix  = pd.DataFrame(
        {sl: [gen_data[sl].get(c, 0.0) for c in all_carriers] for sl in slabels},
        index=all_carriers,
    )
    # Nur Carrier mit nennenswerter Erzeugung
    matrix = matrix.loc[matrix.max(axis=1) > 0.1]
    if matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(slabels) * 1.0 + 2), max(5, len(matrix) * 0.5 + 1)))
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(slabels)))
    ax.set_xticklabels(slabels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=8)
    ax.set_title(title, fontsize=12, pad=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("TWh", fontsize=9)

    # Werte eintragen wenn Zellen groß genug
    if len(slabels) * len(matrix) <= 80:
        for i in range(len(matrix)):
            for j in range(len(slabels)):
                val = matrix.values[i, j]
                if val > 0.1:
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=7, color="black")

    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"  [annual_dispatch] Gespeichert: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Scatter  Kapazitätsfaktor vs. Systemkosten
# ---------------------------------------------------------------------------

def plot_cf_vs_cost_scatter(
    cf_data: Dict[str, float],      # {scenario_label → CF}
    cost_data: Dict[str, float],    # {scenario_label → Mrd. EUR}
    title: str,
    out_path: Path,
    save: bool = True,
) -> None:
    """Scatter: x = Kapazitätsfaktor, y = Systemkosten, beschriftet mit stress_xy."""
    slabels = list(cf_data.keys())
    xs = [cf_data[sl] for sl in slabels]
    ys = [cost_data.get(sl, float("nan")) for sl in slabels]

    if all(np.isnan(xs)) or all(np.isnan(ys)):
        return

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(slabels)))
    fig, ax = plt.subplots(figsize=(8, 5))
    for sl, x, y, col in zip(slabels, xs, ys, colors):
        if not (np.isnan(x) or np.isnan(y)):
            ax.scatter(x, y, color=col, s=80, zorder=3)
            ax.annotate(sl, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Kapazitätsfaktor VRE [-]", fontsize=11)
    ax.set_ylabel("Systemkosten [Mrd. EUR]", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"  [annual_dispatch] Gespeichert: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Haupt-API
# ---------------------------------------------------------------------------

def run_annual_dispatch(
    n_robust: Optional[pypsa.Network],
    scenario_networks: Dict[str, pypsa.Network],
    output_dir: Path,
    run_name: str = "",
    user_colors: Optional[Dict] = None,
    cost_dict: Optional[Dict[str, float]] = None,
    save: bool = True,
) -> None:
    """
    Erstellt alle jährlichen Dispatch-Vergleichsplots.
    Szenario-Namen → stress_xy Kurzlabels.

    Parameters
    ----------
    cost_dict : dict, optional
        {cutout_name → Kosten [Mrd. EUR/a]} aus aro_final_evaluation.all_costs.
        Wird für den Scatter-Plot (CF vs. Systemkosten) genutzt.
        Falls None: Kosten werden aus den Netzwerken selbst berechnet.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Netzwerke zusammenstellen
    all_networks: Dict[str, pypsa.Network] = {}
    if n_robust is not None:
        all_networks["robust"] = n_robust
    for sc_name, n in scenario_networks.items():
        if n is not None:
            all_networks[sc_name] = n

    if not all_networks:
        print("  [annual_dispatch] Keine Netzwerke verfügbar.")
        return

    # Kurzlabels
    label_map = build_scenario_label_map(list(all_networks.keys()))

    # Daten extrahieren
    gen_data:  Dict[str, pd.Series] = {}
    cf_data:   Dict[str, float]     = {}
    cost_data: Dict[str, float]     = {}

    for full_name, n in all_networks.items():
        slabel = label_map[full_name]
        gen_data[slabel]  = _extract_annual_generation(n)
        cf_data[slabel]   = _capacity_factor(n)
        # Kosten: aus cost_dict bevorzugen (direkter ARO-Summary-Wert),
        # sonst aus Netzwerk berechnen
        if cost_dict and full_name in cost_dict:
            cost_data[slabel] = float(cost_dict[full_name])
        else:
            cost_data[slabel] = _extract_system_cost(n)

    # Carrier-Reihenfolge
    all_carriers_set = set().union(*[set(s.index) for s in gen_data.values()])
    all_carriers: List[str] = (
        [c for c in GEN_CARRIER_ORDER if c in all_carriers_set]
        + sorted(c for c in all_carriers_set if c not in GEN_CARRIER_ORDER)
    )

    # Plot 1: gestapelter Balken
    plot_annual_generation_stacked(
        gen_data=gen_data,
        all_carriers=all_carriers,
        title=f"Jährl. Erzeugung je Szenario [{run_name}]",
        out_path=output_dir / "annual_generation_by_scenario.png",
        user_colors=user_colors,
        save=save,
    )

    # Plot 2: Heatmap
    plot_generation_heatmap(
        gen_data=gen_data,
        all_carriers=all_carriers,
        title=f"Erzeugung [TWh] — Carrier × Szenario [{run_name}]",
        out_path=output_dir / "generation_heatmap.png",
        save=save,
    )

    # Plot 3: Scatter
    # Titel mit Hinweis ob Kosten aus Summary oder Netzwerk kommen
    cost_source = "ARO-Summary" if cost_dict else "berechnet aus Netzwerk"
    plot_cf_vs_cost_scatter(
        cf_data=cf_data,
        cost_data=cost_data,
        title=f"Kapazitätsfaktor vs. Systemkosten [{run_name}] (Kosten: {cost_source})",
        out_path=output_dir / "cf_vs_cost_scatter.png",
        save=save,
    )

    print(f"  [annual_dispatch] Fertig. Plots in: {output_dir}")


# ---------------------------------------------------------------------------
# Standalone main()
# ---------------------------------------------------------------------------

def main():
    import argparse
    sys.path.insert(0, str(Path(__file__).parent))
    from master_config import MasterConfig, AROPlottingConfig
    from aro_analysis import AROAnalyzer

    parser = argparse.ArgumentParser(description="ARO Annual Dispatch")
    parser.add_argument("--run",    default=None)
    parser.add_argument("--output", default=None)
    args, _ = parser.parse_known_args()

    master  = MasterConfig()
    aro_cfg = AROPlottingConfig(master=master)
    if args.run:
        aro_cfg.SELECTED_RUN = args.run

    analyzer = AROAnalyzer(config=aro_cfg, auto_find_dispatch=True)
    out_dir  = Path(args.output) if args.output else aro_cfg.get_plot_output_dir("annual_dispatch")

    cost_dict = (
        analyzer.aro_summary
        .get("aro_final_evaluation", {})
        .get("all_costs", {}) or {}
    )

    run_annual_dispatch(
        n_robust=analyzer.n_robust,
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
