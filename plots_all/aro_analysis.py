# aro_analysis.py
"""
ARO-spezifische Analyse und Visualisierung.
Unterstützt beliebig viele Szenarien generisch.
"""

import json
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

import matplotlib
matplotlib.use("Agg")

from master_config import AROPlottingConfig

sns.set_theme("paper", style="whitegrid")


def _safe_path(p: Optional[str]) -> Optional[Path]:
    if p is None:
        return None
    s = str(p).strip()
    return Path(s) if s else None


def _first_existing(paths: List[Optional[Path]]) -> Optional[Path]:
    for p in paths:
        if p is not None and p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Kapazitätsextraktion (Planungsnetz)
# ---------------------------------------------------------------------------

def _extract_capacity(n: pypsa.Network, label: str) -> pd.Series:
    """
    Extrahiert installierte Kapazitäten aus einem Planungsnetz.

    Priorität: p_nom_opt > p_nom_min (wenn >0) > p_nom.
    Gibt eine nach Carrier aggregierte Series zurück.
    """
    gens = n.generators.copy()
    if "carrier" not in gens.columns:
        print(f"  [{label}] WARNUNG: generators.carrier fehlt.")
        return pd.Series(dtype=float)

    if "p_nom_opt" in gens.columns:
        pcol = "p_nom_opt"
    elif "p_nom_min" in gens.columns and (gens["p_nom_min"] > 0).any():
        pcol = "p_nom_min"
    else:
        pcol = "p_nom"

    cap = gens.groupby("carrier")[pcol].sum()
    cap = cap[cap > 1e-3].drop(index=["load"], errors="ignore")  # < 1 kW ignorieren
    print(f"  [{label}] Kapazitätsquelle: {pcol} — {len(cap)} Carrier")
    return cap


def _extract_storage_capacity(n: pypsa.Network, label: str) -> pd.Series:
    """Speicherkapazitäten (StorageUnits + Links mit Speicher-Carrier)."""
    parts = []

    if not n.storage_units.empty and "carrier" in n.storage_units.columns:
        su = n.storage_units.copy()
        pcol = "p_nom_opt" if "p_nom_opt" in su.columns else "p_nom"
        s = su.groupby("carrier")[pcol].sum()
        parts.append(s[s > 1e-3])

    if parts:
        return pd.concat(parts).groupby(level=0).sum()
    return pd.Series(dtype=float)


class AROAnalyzer:
    """
    Generischer Analyzer für ARO-Ergebnisse.
    Lädt alle Szenario-Dispatch-Netzwerke und unterstützt beliebig viele Szenarien.
    """

    def __init__(self, config: AROPlottingConfig = None, auto_find_dispatch: bool = True):
        if config is None:
            config = AROPlottingConfig()

        self.config = config
        self.run_config = config.get_current_run_config()

        self.load_aro_summary()
        self.load_networks(auto_find_dispatch=auto_find_dispatch)

        print(f"ARO Analyzer initialisiert für: {self.run_config.get('name', '<unnamed>')}")
        print(f"  Szenarien: {len(self.scenarios)}: {self.scenarios[:5]}{'...' if len(self.scenarios)>5 else ''}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_aro_summary(self):
        """Lädt ARO Summary JSON."""
        summary_path = _safe_path(self.run_config.get("summary_json"))
        if summary_path is None:
            raise ValueError("run_config['summary_json'] ist None/leer.")
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary JSON nicht gefunden: {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as f:
            self.aro_summary = json.load(f)

        # Szenarien aus Config (bereits auto-befüllt durch AROPlottingConfig)
        self.scenarios: List[str] = list(self.run_config.get("scenarios", []))
        if not self.scenarios:
            # Fallback direkt aus Summary
            self.scenarios = list(
                self.aro_summary.get("aro_final_scenarios", [])
                or self.aro_summary.get("aro_final_evaluation", {}).get("all_costs", {}).keys()
            )

        iters = (
            self.aro_summary.get("aro_convergence", {}).get("iterations_run")
            or len(self.aro_summary.get("aro_history", []))
        )
        print(f"ARO Summary geladen: {summary_path}")
        print(f"  Iterationen: {iters}  |  Finale Szenarien: {len(self.scenarios)}")

    # ------------------------------------------------------------------
    # Dispatch-Pfad-Suche
    # ------------------------------------------------------------------

    def _dispatch_dir(self) -> Optional[Path]:
        base = Path(self.config.BASE_RESULTS_PATH)
        run_name = self.run_config.get("name", "")
        if not run_name:
            return None
        d = base / run_name / "networks" / "dispatch"
        return d if d.is_dir() else None

    def _dispatch_tmp_dir(self) -> Optional[Path]:
        base = Path(self.config.BASE_RESULTS_PATH)
        run_name = self.run_config.get("name", "")
        if not run_name:
            return None
        d = base / run_name / "networks" / "_dispatch_tmp" / "final"
        return d if d.is_dir() else None

    def _safe_name(self, scenario_name: str) -> str:
        return scenario_name.replace("/", "_").replace(" ", "_")

    def _auto_find_dispatch_for_scenario(self, scenario_name: str) -> Optional[Path]:
        safe = self._safe_name(scenario_name)
        d = self._dispatch_dir()
        if d is not None:
            for suffix in (f"dispatch_{safe}_std.nc", f"dispatch_{safe}_flat.nc"):
                p = d / suffix
                if p.is_file():
                    return p
            candidates = sorted(d.glob(f"dispatch_*{safe}*.nc"))
            std_cands = [c for c in candidates if "_std.nc" in c.name]
            if std_cands:
                return std_cands[0]
            if candidates:
                return candidates[0]
        tmp = self._dispatch_tmp_dir()
        if tmp is not None:
            short = (
                Path(scenario_name).stem
                if ("/" in scenario_name or "\\" in scenario_name)
                else scenario_name
            )
            for pattern in (f"dispatch_*{safe}*.nc", f"dispatch_*{short}*.nc"):
                candidates = sorted(tmp.glob(pattern))
                if candidates:
                    return candidates[0]
        return None

    def _auto_find_any_dispatch(self) -> Optional[Path]:
        worst = self.aro_summary.get("aro_final_evaluation", {}).get("worst_case_cutout")
        d = self._dispatch_dir()
        if d is not None:
            if worst:
                safe = self._safe_name(str(worst))
                for suffix in (
                    f"dispatch_{safe}_worst_case_std.nc",
                    f"dispatch_{safe}_worst_case_flat.nc",
                    f"dispatch_{safe}_std.nc",
                    f"dispatch_{safe}_flat.nc",
                ):
                    p = d / suffix
                    if p.is_file():
                        return p
                candidates = sorted(d.glob(f"dispatch_*{safe}*worst_case*.nc"))
                if candidates:
                    return candidates[0]
            wc_cands = sorted(d.glob("dispatch_*_worst_case_std.nc"))
            if wc_cands:
                return wc_cands[0]
            wc_cands = sorted(d.glob("dispatch_*_worst_case*.nc"))
            if wc_cands:
                return wc_cands[0]
            all_std = sorted(d.glob("dispatch_*_std.nc"))
            if all_std:
                return all_std[-1]
            all_nc = sorted(d.glob("dispatch_*.nc"))
            if all_nc:
                return all_nc[-1]
        tmp = self._dispatch_tmp_dir()
        if tmp is not None:
            candidates = sorted(tmp.glob("dispatch_*.nc"))
            if not candidates:
                return None
            if worst:
                for p in candidates:
                    if str(worst) in p.name:
                        return p
            return candidates[-1]
        return None

    def _find_basis_network(self) -> Optional[pypsa.Network]:
        """
        Sucht das Basisjahr-Planungsnetz für den Robust-vs-Basis-Vergleich.

        Suchpfade (Priorität):
          1. run_config["basis_network"]              — explizit in master_config gesetzt
          2. run_config["reference_network"]          — Alternativschlüssel
          3. config.REFERENCE_NETWORK_PATH            — Global definiertes Referenznetz
          4. config.get_reference_networks()[0]       — Erste aus Liste
          5. results/<run>/networks/base_s_*.nc       — Glob-Suche im Run-Verzeichnis
        """
        # 1+2: run_config-Schlüssel
        for key in ("basis_network", "reference_network", "base_network"):
            p = _safe_path(self.run_config.get(key))
            if p is not None and p.is_file():
                print(f"  [basis_network] Aus run_config['{key}']: {p.name}")
                return pypsa.Network(str(p))

        # 3: Globaler REFERENCE_NETWORK_PATH
        ref_path = getattr(self.config, "REFERENCE_NETWORK_PATH", None)
        if ref_path:
            p = Path(str(ref_path))
            if p.is_file():
                print(f"  [basis_network] REFERENCE_NETWORK_PATH: {p.name}")
                return pypsa.Network(str(p))

        # 4: get_reference_networks()
        try:
            ref_networks = self.config.get_reference_networks()
            if ref_networks:
                first = (
                    ref_networks[0]
                    if isinstance(ref_networks, list)
                    else next(iter(ref_networks.values()))[0]
                )
                p = Path(str(first))
                if p.is_file():
                    print(f"  [basis_network] get_reference_networks()[0]: {p.name}")
                    return pypsa.Network(str(p))
        except Exception:
            pass

        # 5: Glob-Suche
        try:
            base = Path(self.config.BASE_RESULTS_PATH)
            run_name = self.run_config.get("name", "")
            if run_name:
                run_net_dir = base / run_name / "networks"
                for pattern in ("base_s_*.nc", "*base*.nc", "*reference*.nc"):
                    hits = sorted(run_net_dir.glob(pattern))
                    # Dispatch-Netze ausschließen
                    hits = [h for h in hits if "dispatch" not in h.name]
                    if hits:
                        print(f"  [basis_network] Glob ({pattern}): {hits[0].name}")
                        return pypsa.Network(str(hits[0]))
        except Exception:
            pass

        print("  [basis_network] WARNUNG: Kein Basisjahr-Planungsnetz gefunden.")
        return None

    def load_networks(self, auto_find_dispatch: bool = True):
        """
        Lädt:
          - n_robust:          robustes Portfolio-Netzwerk
          - n_basis:           Basisjahr-Planungsnetz (für Vergleich)
          - scenario_networks: dict{szenario -> pypsa.Network} für ALLE Szenarien
          - n_worst_case:      Worst-Case-Dispatch (Rückwärtskompatibilität)
        """
        # ---- Robustes Portfolio ----
        portfolio_key = "robust_network"
        portfolio_path = _safe_path(self.run_config.get("robust_network"))
        if portfolio_path is None or not portfolio_path.exists():
            portfolio_key = "robust_network_std"
            portfolio_path = _safe_path(self.run_config.get("robust_network_std"))
        if portfolio_path and portfolio_path.exists():
            self.n_robust = pypsa.Network(str(portfolio_path))
            print(
                f"Robustes Portfolio geladen ({portfolio_key}): "
                f"{len(self.n_robust.generators)} Generatoren"
            )
        else:
            self.n_robust = None
            print(f"WARNUNG: Portfolio nicht gefunden: {portfolio_path}")

        # ---- Basisjahr-Planungsnetz ----
        self.n_basis: Optional[pypsa.Network] = self._find_basis_network()
        if self.n_basis is not None:
            print(f"Basisjahr-Netz geladen: {len(self.n_basis.generators)} Generatoren")
        else:
            print("WARNUNG: Basisjahr-Planungsnetz nicht verfügbar — Robust-vs-Basis-Plots werden übersprungen.")

        # ---- Alle Szenario-Dispatch-Netzwerke ----
        self.scenario_networks: Dict[str, pypsa.Network] = {}

        for scenario in self.scenarios:
            dispatch_paths = self.run_config.get("dispatch_paths", {})
            explicit = _safe_path(dispatch_paths.get(scenario))
            chosen = _first_existing([explicit])

            if chosen is None and auto_find_dispatch:
                chosen = self._auto_find_dispatch_for_scenario(scenario)

            if chosen and chosen.exists():
                try:
                    self.scenario_networks[scenario] = pypsa.Network(str(chosen))
                    print(f"  Szenario-Dispatch geladen: {scenario} ({chosen.name})")
                except Exception as e:
                    print(f"  WARNUNG: Konnte {scenario} nicht laden: {e}")
            else:
                print(f"  WARNUNG: Kein Dispatch-Netzwerk für Szenario '{scenario}' gefunden.")

        # ---- Worst-Case (Rückwärtskompatibilität) ----
        dispatch_std_path = _safe_path(self.run_config.get("worst_case_dispatch_std"))
        dispatch_path     = _safe_path(self.run_config.get("worst_case_dispatch"))
        chosen_wc = _first_existing([dispatch_std_path, dispatch_path])
        if chosen_wc is None and auto_find_dispatch:
            chosen_wc = self._auto_find_any_dispatch()

        if chosen_wc and chosen_wc.exists():
            self.n_worst_case = pypsa.Network(str(chosen_wc))
            print(f"Worst-Case Dispatch geladen: {len(self.n_worst_case.snapshots)} Snapshots ({chosen_wc.name})")
        elif self.scenario_networks:
            all_costs = self.aro_summary.get("aro_final_evaluation", {}).get("all_costs", {})
            if all_costs:
                worst_key = max(all_costs, key=lambda k: all_costs[k])
                self.n_worst_case = self.scenario_networks.get(worst_key)
                print(f"Worst-Case Dispatch: Fallback auf teuerstes Szenario '{worst_key}'")
            else:
                self.n_worst_case = next(iter(self.scenario_networks.values()))
                print("Worst-Case Dispatch: Fallback auf erstes Szenario-Netzwerk")
        else:
            self.n_worst_case = None
            print("Worst-Case Dispatch nicht verfügbar")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_aro_convergence(self, save=True):
        """ARO-Konvergenzplot."""
        if "aro_history" not in self.aro_summary:
            print("Keine ARO History verfügbar")
            return

        history = self.aro_summary["aro_history"]
        if not isinstance(history, list) or len(history) == 0:
            print("ARO History ist leer")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        def _get(h, *keys, default=np.nan):
            for k in keys:
                if k in h:
                    return h[k]
            return default

        iterations  = [int(_get(h, "iteration", default=i + 1)) for i, h in enumerate(history)]
        worst_costs = [float(_get(h, "worst_cost", "worst_case_cost", "worst_total_cost", default=np.nan)) for h in history]
        worst_in_set = [float(_get(h, "worst_in_set", "worst_cost_in_set", default=np.nan)) for h in history]

        axes[0, 0].plot(iterations, worst_costs, "o-", label="Worst-Case (alle)", linewidth=2, markersize=8)
        if not all(np.isnan(worst_in_set)):
            axes[0, 0].plot(iterations, worst_in_set, "s-", label="Worst-Case (im Set)", linewidth=2, markersize=8)
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Kosten [€/a]")
        axes[0, 0].set_title("ARO Konvergenz - Kosten")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        gaps = []
        for h in history:
            g = _get(h, "aro_gap", "gap", default=np.nan)
            try:
                gaps.append(float(g) * 100)
            except Exception:
                gaps.append(np.nan)

        convergence_tol = float(_get(history[0], "convergence_tol", default=0.0)) * 100
        axes[0, 1].plot(iterations, gaps, "o-", linewidth=2, markersize=8, color="red")
        if convergence_tol > 0:
            axes[0, 1].axhline(y=convergence_tol, color="green", linestyle="--",
                               label=f"Toleranz ({convergence_tol:.2f}%)")
            axes[0, 1].legend()
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Gap [%]")
        axes[0, 1].set_title("ARO Konvergenz - Gap")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale("log")

        scenarios_added = [str(_get(h, "worst_cutout", "worst_case_cutout", default="?")) for h in history]
        axes[1, 0].barh(range(len(scenarios_added)), [1] * len(scenarios_added), tick_label=scenarios_added)
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_title("Hinzugefügte Szenarien")
        axes[1, 0].grid(True, alpha=0.3, axis="x")

        final_eval = self.aro_summary.get("aro_final_evaluation", {})
        if isinstance(final_eval, dict):
            all_costs = final_eval.get("all_costs", {})
            if all_costs:
                costs_list     = list(all_costs.values())
                scenarios_list = list(all_costs.keys())
                final_set      = set(self.aro_summary.get("aro_final_scenarios", []))
                bar_colors     = ["red" if s in final_set else "steelblue" for s in scenarios_list]
                axes[1, 1].bar(range(len(costs_list)), costs_list, color=bar_colors, tick_label=scenarios_list)
                worst_total = final_eval.get("worst_case_total_cost")
                if worst_total is not None:
                    axes[1, 1].axhline(y=worst_total, color="red", linestyle="--", label="Worst-Case")
                    axes[1, 1].legend()
                axes[1, 1].set_xlabel("Szenario")
                axes[1, 1].set_ylabel("Kosten [€/a]")
                axes[1, 1].set_title(f"Kosten pro Szenario (finale Evaluation) — N={len(scenarios_list)}")
                plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.suptitle(f"ARO Konvergenzanalyse - {self.run_config.get('name', '')}", fontsize=16)
        plt.tight_layout()

        if save:
            p = self.config.get_plot_output_dir("aro_metrics") / "aro_convergence.png"
            plt.savefig(p, dpi=300, bbox_inches="tight")
            print(f"Gespeichert: {p}")
        plt.close()

    def plot_scenario_cost_comparison(self, save=True):
        """Kosten-Vergleich über alle Szenarien."""
        if "aro_final_evaluation" not in self.aro_summary:
            print("Keine finale Evaluation verfügbar")
            return

        final_eval  = self.aro_summary["aro_final_evaluation"]
        final_costs = final_eval.get("all_costs")
        if not isinstance(final_costs, dict) or not final_costs:
            print("aro_final_evaluation.all_costs fehlt/leer")
            return

        sorted_costs = dict(sorted(final_costs.items(), key=lambda x: x[1]))
        scenarios    = list(sorted_costs.keys())
        costs        = list(sorted_costs.values())
        final_set    = set(self.aro_summary.get("aro_final_scenarios", []))
        bar_colors   = ["red" if s in final_set else "steelblue" for s in scenarios]

        fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 0.6), 8))
        ax.bar(range(len(costs)), costs, color=bar_colors)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
        ax.set_xlabel("Szenario", fontsize=12)
        ax.set_ylabel("Gesamtkosten [€/a]", fontsize=12)
        ax.set_title(
            f"Kosten pro Szenario — {self.run_config.get('name', '')} — N={len(scenarios)}",
            fontsize=14,
        )

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="red",       label="Im finalen Set"),
            Patch(facecolor="steelblue", label="Nicht im finalen Set"),
        ])

        min_cost, max_cost = min(costs), max(costs)
        ax.annotate(f"Min: {min_cost/1e9:.2f} Mrd. €",
                    xy=(costs.index(min_cost), min_cost),
                    xytext=(costs.index(min_cost), min_cost * 1.05),
                    arrowprops=dict(arrowstyle="->", color="green"))
        ax.annotate(f"Max: {max_cost/1e9:.2f} Mrd. €",
                    xy=(costs.index(max_cost), max_cost),
                    xytext=(costs.index(max_cost), max_cost * 0.95),
                    arrowprops=dict(arrowstyle="->", color="red"))

        plt.tight_layout()
        if save:
            p = self.config.get_plot_output_dir("aro_metrics") / "scenario_cost_comparison.png"
            plt.savefig(p, dpi=300, bbox_inches="tight")
            print(f"Gespeichert: {p}")
        plt.close()

    # ------------------------------------------------------------------
    # NEUER HAUPTPLOT: Robust vs. Basisjahr
    # Ersetzt das alte Robust-vs-WorstCase im scenario_comparison-Ordner.
    # ------------------------------------------------------------------

    def plot_robust_vs_basis(
        self,
        save: bool = True,
        n_basis_override: Optional[pypsa.Network] = None,
        label_robust: str = "Robustes Portfolio (ARO)",
        label_basis:  str = "Basisjahr",
    ) -> None:
        """
        Vergleich: Installierte Kapazitäten – Robustes Portfolio vs. Basisjahr.

        Erstellt drei Sub-Plots:
          1. Grouped Bar Chart: absolute Kapazitäten je Carrier
          2. Delta-Chart: (Robust − Basis) je Carrier
          3. Donut-Chart: Anteile beider Portfolios (außen=ARO, innen=Basis)

        Das Basisjahr-Netz wird aus self.n_basis genommen (geladen via
        _find_basis_network() in load_networks). Optional kann ein eigenes
        Netz über n_basis_override übergeben werden.
        """
        n_robust = self.n_robust
        n_basis  = n_basis_override or self.n_basis

        if n_robust is None:
            print("[robust_vs_basis] Robustes Portfolio nicht verfügbar — übersprungen.")
            return
        if n_basis is None:
            print("[robust_vs_basis] Basisjahr-Netz nicht verfügbar — übersprungen.")
            # Fallback: nur Robust-Breakdown zeigen
            self.plot_robust_capacity_breakdown(save=save)
            return

        cap_robust = _extract_capacity(n_robust, label_robust)
        cap_basis  = _extract_capacity(n_basis,  label_basis)

        # Alle Carrier vereinen
        all_carriers = sorted(set(cap_robust.index) | set(cap_basis.index))
        rob = cap_robust.reindex(all_carriers, fill_value=0.0)
        bas = cap_basis.reindex(all_carriers, fill_value=0.0)
        delta = rob - bas

        # Carrier filtern: mindestens 1 MW in einem der beiden
        mask = (rob.abs() + bas.abs()) > 1.0
        rob   = rob[mask]
        bas   = bas[mask]
        delta = delta[mask]
        all_carriers = list(rob.index)

        if not all_carriers:
            print("[robust_vs_basis] Keine gemeinsamen Carrier mit Kapazität > 1 MW.")
            return

        colors_rob = [self.config.CARRIER_COLORS.get(c, "#a9a9a9") for c in all_carriers]
        colors_bas = [self._lighten(self.config.CARRIER_COLORS.get(c, "#a9a9a9"), 0.45) for c in all_carriers]
        delta_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]

        fig = plt.figure(figsize=(20, 14))
        gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)

        # ---- Sub-Plot 1: Grouped Bar ----
        ax1 = fig.add_subplot(gs[0, :])
        x   = np.arange(len(all_carriers))
        w   = 0.38
        bars_r = ax1.bar(x - w / 2, rob.values / 1e3,   width=w, label=label_robust, color=colors_rob, alpha=0.92)
        bars_b = ax1.bar(x + w / 2, bas.values / 1e3,   width=w, label=label_basis,  color=colors_bas, alpha=0.92,
                         edgecolor="grey", linewidth=0.6)
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_carriers, rotation=35, ha="right", fontsize=9)
        ax1.set_ylabel("Kapazität [GW]", fontsize=11)
        ax1.set_title(
            f"Installierte Kapazitäten: {label_robust} vs. {label_basis}",
            fontsize=13, fontweight="bold",
        )
        ax1.legend(fontsize=10)
        ax1.grid(axis="y", alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))

        # Wertebeschriftung (nur wenn > 0.1 GW)
        for bar, val in zip(bars_r, rob.values / 1e3):
            if val > 0.1:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=7, color="#333")
        for bar, val in zip(bars_b, bas.values / 1e3):
            if val > 0.1:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=7, color="#555")

        # ---- Sub-Plot 2: Delta ----
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.bar(all_carriers, delta.values / 1e3, color=delta_colors, alpha=0.88)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_xticklabels(all_carriers, rotation=35, ha="right", fontsize=9)
        ax2.set_ylabel("Δ Kapazität [GW]  (Robust − Basis)", fontsize=10)
        ax2.set_title("Kapazitätsdifferenz (ARO − Basisjahr)", fontsize=12)
        ax2.grid(axis="y", alpha=0.3)

        from matplotlib.patches import Patch
        ax2.legend(handles=[
            Patch(facecolor="#2ca02c", label="ARO höher"),
            Patch(facecolor="#d62728", label="ARO niedriger"),
        ], fontsize=9)

        # ---- Sub-Plot 3: Donut-Vergleich ----
        ax3 = fig.add_subplot(gs[1, 1])
        total_rob = rob.sum()
        total_bas = bas.sum()
        if total_rob > 0 and total_bas > 0:
            # Donut: äußerer Ring = Robust, innerer Ring = Basis
            # Nur Top-8 Carrier zeigen (Rest als "Sonstige")
            top_n = 8
            top_idx = rob.abs().nlargest(top_n).index.tolist()
            other_idx = [c for c in all_carriers if c not in top_idx]

            rob_top  = rob[top_idx].tolist()  + ([rob[other_idx].sum()]  if other_idx else [])
            bas_top  = bas[top_idx].tolist()  + ([bas[other_idx].sum()]  if other_idx else [])
            labels_d = top_idx + (["Sonstige"] if other_idx else [])
            cols_d   = [self.config.CARRIER_COLORS.get(c, "#a9a9a9") for c in top_idx]
            if other_idx:
                cols_d.append("#cccccc")

            kw = dict(wedgeprops=dict(width=0.42), startangle=90, labels=None)
            ax3.pie(rob_top, colors=cols_d, radius=1.0,   **kw)
            ax3.pie(bas_top, colors=cols_d, radius=0.58,  **kw)

            ax3.text(0,  0.14, "ARO",   ha="center", va="center", fontsize=11, fontweight="bold")
            ax3.text(0, -0.14, "Basis", ha="center", va="center", fontsize=10, color="#555")

            # Legende
            handles = [plt.Rectangle((0,0),1,1, fc=c) for c in cols_d]
            ax3.legend(handles, labels_d, loc="lower center",
                       bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
            ax3.set_title("Portfolio-Anteile (außen=ARO, innen=Basis)", fontsize=11)
        else:
            ax3.axis("off")
            ax3.text(0.3, 0.5, "Keine Daten für Donut.", fontsize=11)

        # Kennzahlen-Box
        total_delta_gw = (total_rob - total_bas) / 1e3
        pct_change     = ((total_rob - total_bas) / total_bas * 100) if total_bas > 0 else float("nan")
        info_text = (
            f"Gesamtkapazität\n"
            f"  ARO:   {total_rob/1e3:.1f} GW\n"
            f"  Basis: {total_bas/1e3:.1f} GW\n"
            f"  Delta: {total_delta_gw:+.1f} GW ({pct_change:+.1f}%)"
        )
        fig.text(0.72, 0.02, info_text, fontsize=9, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

        plt.suptitle(
            f"Robustes Portfolio vs. Basisjahr — {self.run_config.get('name', '')}",
            fontsize=15, fontweight="bold", y=0.98,
        )

        if save:
            p = self.config.get_plot_output_dir("scenario_comparison") / "robust_vs_basis_capacity.png"
            plt.savefig(p, dpi=300, bbox_inches="tight")
            print(f"Gespeichert: {p}")
        plt.close()
        print("[robust_vs_basis] Plot fertig.")

    @staticmethod
    def _lighten(hex_color: str, amount: float = 0.4) -> str:
        """Hellt eine Hex-Farbe auf (amount=1.0 → weiß)."""
        try:
            h = hex_color.lstrip("#")
            r, g, b = [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
            r = r + (1 - r) * amount
            g = g + (1 - g) * amount
            b = b + (1 - b) * amount
            return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        except Exception:
            return hex_color

    def plot_robust_capacity_breakdown(self, save=True):
        """Kapazitätsaufschlüsselung des robusten Portfolios (einzeln, ohne Vergleich)."""
        if self.n_robust is None:
            print("Robustes Portfolio nicht verfügbar")
            return

        cap = _extract_capacity(self.n_robust, "robust")
        if cap.empty:
            print("Keine positiven Kapazitäten gefunden.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = [self.config.CARRIER_COLORS.get(c, "#a9a9a9") for c in cap.index]

        cap.plot(kind="bar", ax=axes[0], color=colors)
        axes[0].set_title("Installierte Kapazität nach Technologie", fontsize=14)
        axes[0].set_ylabel("Kapazität [MW]", fontsize=12)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

        if len(cap) <= 20:
            axes[1].pie(cap.values, labels=cap.index, colors=colors,
                        autopct="%1.1f%%", startangle=90)
            axes[1].set_title("Kapazitätsanteile", fontsize=14)
        else:
            axes[1].axis("off")
            axes[1].text(0.0, 0.5, "Zu viele Carrier für Pie-Chart (>20).", fontsize=12)

        plt.suptitle(f"Robustes Portfolio - Kapazitäten - {self.run_config.get('name', '')}", fontsize=16)
        plt.tight_layout()
        if save:
            p = self.config.get_plot_output_dir("capacity") / "robust_capacity_breakdown.png"
            plt.savefig(p, dpi=300, bbox_inches="tight")
            print(f"Gespeichert: {p}")
        plt.close()

    def plot_scenario_capacity_comparison(self, save=True):
        """
        scenario_comparison/-Ordner:
          1. Robust vs. Basisjahr (Kapazitätsvergleich)  ← NEU, Hauptplot
          2. Dispatch-Performance je Klimaszenario        ← wie bisher
        """
        # --- 1. Robust vs. Basisjahr ---
        self.plot_robust_vs_basis(save=save)

        # --- 2. Dispatch-Performance je Szenario ---
        if not self.scenario_networks:
            print("Keine Szenario-Netzwerke geladen — Dispatch-Szenariovergleich übersprungen.")
            return

        all_costs = self.aro_summary.get("aro_final_evaluation", {}).get("all_costs", {})

        records = []
        for scenario, n in self.scenario_networks.items():
            if n is None:
                continue
            if "carrier" not in n.generators.columns:
                continue
            if hasattr(n, "generators_t") and hasattr(n.generators_t, "p") and not n.generators_t.p.empty:
                gen_sum = n.generators_t.p.sum()
                gen_df  = n.generators[["carrier"]].copy()
                gen_df["energy_MWh"] = gen_sum
                by_carrier = gen_df.groupby("carrier")["energy_MWh"].sum()
            else:
                pcol = "p_nom_opt" if "p_nom_opt" in n.generators.columns else "p_nom"
                by_carrier = n.generators.groupby("carrier")[pcol].sum()

            for carrier, val in by_carrier.items():
                records.append({
                    "scenario": scenario,
                    "carrier":  carrier,
                    "value":    val,
                    "cost":     all_costs.get(scenario, np.nan),
                })

        if not records:
            print("Keine Dispatch-Daten für Szenario-Vergleich.")
            return

        df = pd.DataFrame(records)
        top_carriers = (
            df.groupby("carrier")["value"].sum()
              .sort_values(ascending=False)
              .head(12)
              .index.tolist()
        )
        df_top = df[df["carrier"].isin(top_carriers)]
        pivot  = df_top.pivot_table(index="scenario", columns="carrier", values="value", aggfunc="sum").fillna(0)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        colors = [self.config.CARRIER_COLORS.get(c, "#a9a9a9") for c in pivot.columns]
        pivot.plot(kind="bar", stacked=True, ax=axes[0], color=colors)
        axes[0].set_title(f"Erzeugung pro Klimaszenario (robustes Portfolio) — N={len(pivot)}", fontsize=13)
        axes[0].set_xlabel("Szenario")
        axes[0].set_ylabel("MWh")
        axes[0].legend(loc="upper right", fontsize=8, ncol=2)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

        scenario_costs = df.groupby("scenario")["cost"].first().dropna()
        scenario_total = df.groupby("scenario")["value"].sum()
        common = scenario_costs.index.intersection(scenario_total.index)
        if len(common) > 0:
            axes[1].scatter(scenario_total[common], scenario_costs[common], s=80, alpha=0.8)
            for s in common:
                axes[1].annotate(s, (scenario_total[s], scenario_costs[s]), fontsize=7,
                                 xytext=(4, 4), textcoords="offset points")
            axes[1].set_xlabel("Gesamterzeugung [MWh]")
            axes[1].set_ylabel("Systemkosten [€/a]")
            axes[1].set_title("Kosten vs. Erzeugung (je Klimaszenario)", fontsize=13)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis("off")
            axes[1].text(0.1, 0.5, "Keine Kostendaten für Scatter.", fontsize=11)

        plt.suptitle(f"Dispatch-Szenariovergleich — {self.run_config.get('name', '')}", fontsize=15)
        plt.tight_layout()
        if save:
            p = self.config.get_plot_output_dir("scenario_comparison") / "scenario_dispatch_comparison.png"
            plt.savefig(p, dpi=300, bbox_inches="tight")
            print(f"Gespeichert: {p}")
        plt.close()

    def analyze_worst_case_dispatch(self, country="DE", save=True):
        """Worst-Case Dispatch Analyse für spezifisches Land."""
        if self.n_worst_case is None:
            print("Worst-Case Dispatch nicht verfügbar")
            return

        from country_analysis import CountryAnalyzer
        analyzer = CountryAnalyzer(
            network=self.n_worst_case,
            country=country,
            output_dir=str(self.config.get_plot_output_dir("worst_case") / country),
            carrier_colors=self.config.CARRIER_COLORS,
            default_color=self.config.DEFAULT_COLOR,
        )
        analyzer.summary_report()
        analyzer.generate_all_plots(save=save)
        print(f"Worst-Case Analyse für {country} abgeschlossen")

    def analyze_all_scenario_dispatches(self, countries: Optional[List[str]] = None, save=True):
        """
        Führt CountryAnalyzer für ALLE geladenen Szenario-Netzwerke durch.
        """
        if not self.scenario_networks:
            print("Keine Szenario-Netzwerke verfügbar.")
            return

        from country_analysis import CountryAnalyzer
        c_list = countries or self.config.COUNTRIES_TO_ANALYZE

        for scenario, n in self.scenario_networks.items():
            if n is None:
                continue
            print(f"\n  Analysiere Szenario: {scenario}")
            for country in c_list:
                try:
                    out_dir = self.config.get_plot_output_dir(f"scenarios/{scenario}") / country
                    analyzer = CountryAnalyzer(
                        network=n,
                        country=country,
                        output_dir=str(out_dir),
                        carrier_colors=self.config.CARRIER_COLORS,
                        default_color=self.config.DEFAULT_COLOR,
                    )
                    analyzer.summary_report()
                    analyzer.generate_all_plots(save=save)
                    print(f"    ✓ {country}")
                except Exception as e:
                    print(f"    ✗ {country}: {e}")

    def generate_all_aro_plots(self, save=True):
        """Generiert alle ARO-spezifischen Plots."""
        print(f"\n=== Generiere ARO-Plots für {self.run_config.get('name', '')} ===\n")

        plot_functions = [
            ("ARO Konvergenz",              self.plot_aro_convergence),
            ("Szenario-Kostenvergleich",     self.plot_scenario_cost_comparison),
            ("Robuste Kapazitäten",          self.plot_robust_capacity_breakdown),
            ("Robust vs. Basisjahr",          self.plot_robust_vs_basis),
            ("Szenario-Dispatch-Vergleich",   self.plot_scenario_capacity_comparison),
        ]

        for name, func in plot_functions:
            try:
                print(f"Erstelle: {name}")
                func(save=save)
            except Exception as e:
                print(f"FEHLER bei {name}: {e}")
                import traceback
                traceback.print_exc()

        print("\n=== ARO-Plots abgeschlossen ===")

    def summary_report(self):
        """Druckt ARO-Zusammenfassung."""
        print(f"\n{'=' * 60}")
        print(f"ARO ANALYSE-BERICHT: {self.run_config.get('name', '')}")
        print(f"{'=' * 60}")

        conv = self.aro_summary.get("aro_convergence", {})
        print(f"\nKonvergenz:")
        print(f"  Iterationen: {conv.get('iterations_run', '?')} / {conv.get('max_iter', '?')}")
        print(f"  Konvergenzgrund: {conv.get('reason', 'Unbekannt')}")

        final_eval = self.aro_summary.get("aro_final_evaluation", {})
        print(f"\nFinale Evaluation:")
        print(f"  Worst-Case Cutout: {final_eval.get('worst_case_cutout', '?')}")
        wc_cost = final_eval.get("worst_case_total_cost", 0)
        print(f"  Worst-Case Kosten: {wc_cost/1e9:.2f} Mrd. €/a")
        print(f"  Finale Gap: {final_eval.get('aro_gap', 0) * 100:.4f}%")

        print(f"\nFinale Szenarien ({len(self.scenarios)}):")
        all_costs = final_eval.get("all_costs", {})
        for s in self.scenarios:
            cost = all_costs.get(s, 0)
            loaded = "✓" if s in self.scenario_networks else "✗ (nicht geladen)"
            print(f"  {loaded}  {s}: {cost/1e9:.2f} Mrd. €/a")

        n_basis_info = (
            f"{len(self.n_basis.generators)} Generatoren"
            if self.n_basis is not None
            else "nicht gefunden"
        )
        print(f"\nBasisjahr-Netz: {n_basis_info}")

        d = self._dispatch_dir()
        tmp = self._dispatch_tmp_dir()
        if d:
            nc_files = list(d.glob("dispatch_*.nc"))
            print(f"\nDispatch-Verzeichnis: {d}  ({len(nc_files)} Dateien)")
        elif tmp:
            nc_files = list(tmp.glob("dispatch_*.nc"))
            print(f"\nDispatch-Verzeichnis (tmp-Fallback): {tmp}  ({len(nc_files)} Dateien)")
        else:
            print(f"\nWARNUNG: Kein Dispatch-Verzeichnis gefunden!")
            print(f"  Erwartet: results/<run>/networks/dispatch/")

        print(f"\nGeladene Dispatch-Netzwerke: {len(self.scenario_networks)} / {len(self.scenarios)}")


# CLI
def main():
    import argparse

    parser = argparse.ArgumentParser(description="ARO Ergebnisse analysieren")
    parser.add_argument("--run",       default=None, help="ARO Run Name (aus master_config.py)")
    parser.add_argument("--countries", nargs="+", default=["DE"], help="Länder für Detailanalyse")
    parser.add_argument("--output",    default=None, help="Output-Pfad (überschreibt Config)")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Alle Szenario-Dispatch-Netzwerke einzeln analysieren")
    args = parser.parse_args()

    config = AROPlottingConfig()
    if args.run:
        config.SELECTED_RUN = args.run
    if args.output:
        config.PLOT_OUTPUT_PATH = args.output

    analyzer = AROAnalyzer(config, auto_find_dispatch=True)
    analyzer.summary_report()
    analyzer.generate_all_aro_plots(save=True)

    if args.all_scenarios:
        analyzer.analyze_all_scenario_dispatches(countries=args.countries, save=True)
    else:
        for country in args.countries:
            print(f"\n=== Analysiere Worst-Case für {country} ===")
            analyzer.analyze_worst_case_dispatch(country=country, save=True)

    print("\n✓ Analyse abgeschlossen!")
    print(f"Plots gespeichert in: {config.PLOT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
