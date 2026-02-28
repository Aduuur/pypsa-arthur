# aro_analysis.py
"""
ARO-spezifische Analyse und Visualisierung
"""

import json
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

from master_config import AROPlottingConfig

# Matplotlib für Serverumgebung
import matplotlib
matplotlib.use("Agg")

sns.set_theme("paper", style="whitegrid")


def _safe_path(p: Optional[str]) -> Optional[Path]:
    if p is None:
        return None
    try:
        s = str(p).strip()
        if not s:
            return None
        return Path(s)
    except Exception:
        return None


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p is not None and p.is_file():
            return p
    return None


class AROAnalyzer:
    """
    Analyzer für ARO-Ergebnisse
    """

    def __init__(self, config: AROPlottingConfig = None, auto_find_dispatch: bool = True):
        if config is None:
            config = AROPlottingConfig()

        self.config = config
        self.run_config = config.get_current_run_config()

        # Lade ARO Summary JSON
        self.load_aro_summary()

        # Lade Netzwerke
        self.load_networks(auto_find_dispatch=auto_find_dispatch)

        print(f"ARO Analyzer initialisiert für: {self.run_config.get('name', '<unnamed>')}")

    def load_aro_summary(self):
        """Lädt ARO Summary JSON"""
        summary_path = _safe_path(self.run_config.get("summary_json"))
        if summary_path is None:
            raise ValueError("run_config['summary_json'] ist None/leer.")

        if not summary_path.exists():
            raise FileNotFoundError(f"Summary JSON nicht gefunden: {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as f:
            self.aro_summary = json.load(f)

        iters = self.aro_summary.get("aro_convergence", {}).get("iterations_run", None)
        if iters is None:
            # fallback: length of history
            hist = self.aro_summary.get("aro_history", [])
            iters = len(hist) if isinstance(hist, list) else "?"

        print(f"ARO Summary geladen: {summary_path}")
        print(f"  - Iterationen: {iters}")
        print(f"  - Finale Szenarien: {len(self.aro_summary.get('aro_final_scenarios', []))}")

    def _auto_find_dispatch(self) -> Optional[Path]:
        """
        Auto-discovery für Worst-Case Dispatch-Dateien.
        Erwartet (wie in deinen Ergebnissen):
          <aro_results_base>/<run_name>/networks/_dispatch_tmp/final/dispatch_*.nc
        """
        base = Path(self.config.BASE_RESULTS_PATH)
        run_name = self.run_config.get("name", "")
        if not run_name:
            return None

        cand_dir = base / run_name / "networks" / "_dispatch_tmp" / "final"
        if not cand_dir.is_dir():
            return None

        candidates = sorted(cand_dir.glob("dispatch_*.nc"))
        if not candidates:
            return None

        # best-effort: versuche worst_case_cutout aus summary
        worst = None
        final_eval = self.aro_summary.get("aro_final_evaluation", {})
        if isinstance(final_eval, dict):
            worst = final_eval.get("worst_case_cutout")

        if worst:
            for p in candidates:
                if str(worst) in p.name:
                    return p

        # fallback: nimm letztes (häufig rcp45/rcp26 beide da; zuletzt sortiert)
        return candidates[-1]

    def load_networks(self, auto_find_dispatch: bool = True):
        """Lädt robustes Portfolio und Worst-Case Dispatch"""
        # Robustes Portfolio
        portfolio_path = _safe_path(self.run_config.get("robust_network"))
        if portfolio_path is not None and portfolio_path.exists():
            self.n_robust = pypsa.Network(str(portfolio_path))
            print(f"Robustes Portfolio geladen: {len(self.n_robust.generators)} Generatoren")
        else:
            self.n_robust = None
            if portfolio_path is None:
                print("WARNUNG: robust_network ist None/leer.")
            else:
                print(f"WARNUNG: Portfolio nicht gefunden: {portfolio_path}")

        # Worst-Case Dispatch (Standard-Format wenn verfügbar)
        dispatch_std_path = _safe_path(self.run_config.get("worst_case_dispatch_std"))
        dispatch_path = _safe_path(self.run_config.get("worst_case_dispatch"))

        chosen = _first_existing([dispatch_std_path, dispatch_path])

        if chosen is None and auto_find_dispatch:
            chosen = self._auto_find_dispatch()

        if chosen is not None and chosen.exists():
            self.n_worst_case = pypsa.Network(str(chosen))
            print(f"Worst-Case Dispatch geladen: {len(self.n_worst_case.snapshots)} Snapshots ({chosen.name})")
        else:
            self.n_worst_case = None
            print("Worst-Case Dispatch nicht verfügbar")

    def plot_aro_convergence(self, save=True):
        """1. ARO-Konvergenzplot"""
        if "aro_history" not in self.aro_summary:
            print("Keine ARO History verfügbar")
            return

        history = self.aro_summary["aro_history"]
        if not isinstance(history, list) or len(history) == 0:
            print("ARO History ist leer")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Worst-Case Kosten pro Iteration (robust: key variants)
        def _get(h: Dict[str, Any], *keys, default=np.nan):
            for k in keys:
                if k in h:
                    return h[k]
            return default

        iterations = [int(_get(h, "iteration", default=i + 1)) for i, h in enumerate(history)]
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

        # Gap pro Iteration
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
            axes[0, 1].axhline(y=convergence_tol, color="green", linestyle="--", label=f"Toleranz ({convergence_tol:.2f}%)")
            axes[0, 1].legend()
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Gap [%]")
        axes[0, 1].set_title("ARO Konvergenz - Gap")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale("log")  # Log-Skala für Gap

        # Szenario-Hinzufügung pro Iteration
        scenarios_added = []
        for h in history:
            cutout = _get(h, "worst_cutout", "worst_case_cutout", default="?")
            scenarios_added.append(str(cutout))

        axes[1, 0].barh(range(len(scenarios_added)), [1] * len(scenarios_added), tick_label=scenarios_added)
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_title("Hinzugefügte Szenarien")
        axes[1, 0].grid(True, alpha=0.3, axis="x")

        # Kosten-Verteilung finale Evaluation (robust: key variants)
        final_eval = self.aro_summary.get("aro_final_evaluation", {})
        if isinstance(final_eval, dict):
            all_costs = final_eval.get("all_costs")
            if isinstance(all_costs, dict) and all_costs:
                costs_list = list(all_costs.values())
                scenarios_list = list(all_costs.keys())

                axes[1, 1].bar(range(len(costs_list)), costs_list, tick_label=scenarios_list)
                worst_total = final_eval.get("worst_case_total_cost")
                if worst_total is not None:
                    axes[1, 1].axhline(y=worst_total, color="red", linestyle="--", label="Worst-Case")
                    axes[1, 1].legend()
                axes[1, 1].set_xlabel("Szenario")
                axes[1, 1].set_ylabel("Kosten [€/a]")
                axes[1, 1].set_title("Kosten pro Szenario (finale Evaluation)")
                plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.suptitle(f"ARO Konvergenzanalyse - {self.run_config.get('name', '')}", fontsize=16)
        plt.tight_layout()

        if save:
            output_path = self.config.get_plot_output_dir("aro_metrics") / "aro_convergence.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Gespeichert: {output_path}")

        plt.close()

    def plot_scenario_cost_comparison(self, save=True):
        """2. Kosten-Vergleich über alle Szenarien"""
        if "aro_final_evaluation" not in self.aro_summary:
            print("Keine finale Evaluation verfügbar")
            return

        final_eval = self.aro_summary["aro_final_evaluation"]
        final_costs = final_eval.get("all_costs")
        if not isinstance(final_costs, dict) or not final_costs:
            print("aro_final_evaluation.all_costs fehlt/leer")
            return

        # Sortiere nach Kosten
        sorted_costs = dict(sorted(final_costs.items(), key=lambda x: x[1]))

        scenarios = list(sorted_costs.keys())
        costs = list(sorted_costs.values())

        # Markiere finale Szenarien
        final_scenarios = set(self.aro_summary.get("aro_final_scenarios", []))
        colors = ["red" if s in final_scenarios else "steelblue" for s in scenarios]

        fig, ax = plt.subplots(figsize=(14, 8))

        bars = ax.bar(range(len(costs)), costs, color=colors)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
        ax.set_xlabel("Szenario", fontsize=12)
        ax.set_ylabel("Gesamtkosten [€/a]", fontsize=12)
        ax.set_title(f'Kosten pro Szenario - {self.run_config.get("name","")}', fontsize=16)

        # Legende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="red", label="Im finalen Set"),
            Patch(facecolor="steelblue", label="Nicht im finalen Set"),
        ]
        ax.legend(handles=legend_elements)

        # Annotationen für Min/Max
        min_cost = min(costs)
        max_cost = max(costs)
        min_idx = costs.index(min_cost)
        max_idx = costs.index(max_cost)

        ax.annotate(
            f"Min: {min_cost / 1e9:.2f} Mrd. €",
            xy=(min_idx, min_cost),
            xytext=(min_idx, min_cost * 1.05),
            arrowprops=dict(arrowstyle="->", color="green"),
        )

        ax.annotate(
            f"Max: {max_cost / 1e9:.2f} Mrd. €",
            xy=(max_idx, max_cost),
            xytext=(max_idx, max_cost * 0.95),
            arrowprops=dict(arrowstyle="->", color="red"),
        )

        plt.tight_layout()

        if save:
            output_path = self.config.get_plot_output_dir("aro_metrics") / "scenario_cost_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Gespeichert: {output_path}")

        plt.close()

    def plot_robust_capacity_breakdown(self, save=True):
        """3. Kapazitätsaufschlüsselung des robusten Portfolios"""
        if self.n_robust is None:
            print("Robustes Portfolio nicht verfügbar")
            return

        # Kapazitäten pro Carrier (robust: p_nom_opt fallback)
        gens = self.n_robust.generators
        pcol = "p_nom_opt" if "p_nom_opt" in gens.columns else ("p_nom" if "p_nom" in gens.columns else None)
        if pcol is None or "carrier" not in gens.columns:
            print("Generators fehlen p_nom_opt/p_nom oder carrier.")
            return

        capacity_by_carrier = gens.groupby("carrier")[pcol].sum()
        capacity_by_carrier = capacity_by_carrier[capacity_by_carrier > 0].sort_values(ascending=False)

        # remove nonsensical
        capacity_by_carrier = capacity_by_carrier.drop(index=["load"], errors="ignore")

        if capacity_by_carrier.empty:
            print("Keine positiven Kapazitäten gefunden.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        colors = [self.config.CARRIER_COLORS.get(c, "#a9a9a9") for c in capacity_by_carrier.index]

        # Bar Chart
        capacity_by_carrier.plot(kind="bar", ax=axes[0], color=colors)
        axes[0].set_title("Installierte Kapazität nach Technologie", fontsize=14)
        axes[0].set_ylabel("Kapazität [MW]", fontsize=12)
        axes[0].set_xlabel("Technologie", fontsize=12)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Pie Chart (nur wenn nicht zu viele)
        if len(capacity_by_carrier) <= 20:
            axes[1].pie(
                capacity_by_carrier.values,
                labels=capacity_by_carrier.index,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
            )
            axes[1].set_title("Kapazitätsanteile", fontsize=14)
        else:
            axes[1].axis("off")
            axes[1].text(0.0, 0.5, "Zu viele Carrier für Pie-Chart\n(>20).", fontsize=12)

        plt.suptitle(f'Robustes Portfolio - Kapazitäten - {self.run_config.get("name","")}', fontsize=16)
        plt.tight_layout()

        if save:
            output_path = self.config.get_plot_output_dir("capacity") / "robust_capacity_breakdown.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Gespeichert: {output_path}")

        plt.close()

    def analyze_worst_case_dispatch(self, country="DE", save=True):
        """4. Worst-Case Dispatch Analyse für spezifisches Land"""
        if self.n_worst_case is None:
            print("Worst-Case Dispatch nicht verfügbar")
            return

        # Importiere Country Analyzer für Detailanalyse
        from country_analysis import CountryAnalyzer

        # Direktes Netzwerkobjekt nutzen (kein /tmp export nötig)
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

    def generate_all_aro_plots(self, save=True):
        """Generiert alle ARO-spezifischen Plots"""
        print(f"\n=== Generiere ARO-Plots für {self.run_config.get('name','')} ===\n")

        plot_functions = [
            ("ARO Konvergenz", self.plot_aro_convergence),
            ("Szenario-Kostenvergleich", self.plot_scenario_cost_comparison),
            ("Robuste Kapazitäten", self.plot_robust_capacity_breakdown),
        ]

        for name, func in plot_functions:
            try:
                print(f"Erstelle: {name}")
                func(save=save)
            except Exception as e:
                print(f"FEHLER bei {name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n=== ARO-Plots abgeschlossen ===")

    def summary_report(self):
        """Druckt ARO-Zusammenfassung"""
        print(f"\n{'=' * 60}")
        print(f"ARO ANALYSE-BERICHT: {self.run_config.get('name','')}")
        print(f"{'=' * 60}")

        conv = self.aro_summary.get("aro_convergence", {})
        print(f"\nKonvergenz:")
        print(f"  - Iterationen durchgeführt: {conv.get('iterations_run', '?')} / {conv.get('max_iter', '?')}")
        print(f"  - Konvergenzgrund: {conv.get('reason', 'Unbekannt')}")

        final_eval = self.aro_summary.get("aro_final_evaluation", {})
        print(f"\nFinale Evaluation:")
        print(f"  - Worst-Case Cutout: {final_eval.get('worst_case_cutout', '?')}")
        print(f"  - Worst-Case Kosten: {final_eval.get('worst_case_total_cost', 0) / 1e9:.2f} Mrd. €/a")
        print(f"  - Finale Gap: {final_eval.get('aro_gap', 0) * 100:.4f}%")

        print(f"\nFinale Szenarien ({len(self.aro_summary.get('aro_final_scenarios', []))}):")
        for s in self.aro_summary.get("aro_final_scenarios", []):
            cost = final_eval.get("all_costs", {}).get(s, 0)
            print(f"  - {s}: {cost / 1e9:.2f} Mrd. €/a")


# Hauptfunktion für CLI-Nutzung
def main():
    import argparse

    parser = argparse.ArgumentParser(description="ARO Ergebnisse analysieren")
    parser.add_argument("--run", default=None, help="ARO Run Name (aus master_config.py)")
    parser.add_argument("--countries", nargs="+", default=["DE"], help="Länder für Detailanalyse")
    parser.add_argument("--output", default=None, help="Output-Pfad (überschreibt Config)")

    args = parser.parse_args()

    config = AROPlottingConfig()

    if args.run:
        config.SELECTED_RUN = args.run

    if args.output:
        config.PLOT_OUTPUT_PATH = args.output

    analyzer = AROAnalyzer(config, auto_find_dispatch=True)

    analyzer.summary_report()
    analyzer.generate_all_aro_plots(save=True)

    for country in args.countries:
        print(f"\n=== Analysiere Worst-Case für {country} ===")
        analyzer.analyze_worst_case_dispatch(country=country, save=True)

    print("\n✓ Analyse abgeschlossen!")
    print(f"Plots gespeichert in: {config.PLOT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()