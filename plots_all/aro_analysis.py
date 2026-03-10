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

    def _auto_find_dispatch_for_scenario(self, scenario_name: str) -> Optional[Path]:
        """
        Sucht Dispatch-Netzwerk für ein bestimmtes Szenario.
        Erwartet: <aro_results_base>/<run>/ networks/_dispatch_tmp/final/dispatch_<scenario>*.nc
        """
        base = Path(self.config.BASE_RESULTS_PATH)
        run_name = self.run_config.get("name", "")
        if not run_name:
            return None
        cand_dir = base / run_name / "networks" / "_dispatch_tmp" / "final"
        if not cand_dir.is_dir():
            return None

        # Suche nach Szenario-Namen im Dateinamen
        candidates = sorted(cand_dir.glob(f"dispatch_*{scenario_name}*.nc"))
        if candidates:
            return candidates[0]

        # Breitere Suche (scenario_name kann Pfad-Teile enthalten)
        short = Path(scenario_name).stem if "/" in scenario_name or "\\" in scenario_name else scenario_name
        candidates = sorted(cand_dir.glob(f"dispatch_*{short}*.nc"))
        return candidates[0] if candidates else None

    def _auto_find_any_dispatch(self) -> Optional[Path]:
        """Fallback: letztes Dispatch-Netzwerk im final-Ordner."""
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

        # Bevorzuge worst_case_cutout aus summary
        worst = self.aro_summary.get("aro_final_evaluation", {}).get("worst_case_cutout")
        if worst:
            for p in candidates:
                if str(worst) in p.name:
                    return p
        return candidates[-1]

    def load_networks(self, auto_find_dispatch: bool = True):
        """
        Lädt:
          - n_robust: robustes Portfolio-Netzwerk
          - scenario_networks: dict{szenario -> pypsa.Network} für ALLE Szenarien
          - n_worst_case: Worst-Case-Dispatch (für Rückwärtskompatibilität)
        """
        # ---- Robustes Portfolio ----
        portfolio_path = _safe_path(self.run_config.get("robust_network"))
        if portfolio_path and portfolio_path.exists():
            self.n_robust = pypsa.Network(str(portfolio_path))
            print(f"Robustes Portfolio geladen: {len(self.n_robust.generators)} Generatoren")
        else:
            self.n_robust = None
            print(f"WARNUNG: Portfolio nicht gefunden: {portfolio_path}")

        # ---- Alle Szenario-Dispatch-Netzwerke ----
        self.scenario_networks: Dict[str, pypsa.Network] = {}

        for scenario in self.scenarios:
            # 1. Explizit in run_config (z.B. "dispatch_paths": {"cutout_rcp45": "/path/..."})
            dispatch_paths = self.run_config.get("dispatch_paths", {})
            explicit = _safe_path(dispatch_paths.get(scenario))
            chosen = _first_existing([explicit])

            # 2. Auto-Suche
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
            # Fallback: teuerste Szenario als worst case
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
                costs_list    = list(all_costs.values())
                scenarios_list = list(all_costs.keys())
                final_set     = set(self.aro_summary.get("aro_final_scenarios", []))
                bar_colors    = ["red" if s in final_set else "steelblue" for s in scenarios_list]
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

        sorted_costs   = dict(sorted(final_costs.items(), key=lambda x: x[1]))
        scenarios      = list(sorted_costs.keys())
        costs          = list(sorted_costs.values())
        final_set      = set(self.aro_summary.get("aro_final_scenarios", []))
        bar_colors     = ["red" if s in final_set else "steelblue" for s in scenarios]

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
            Patch(facecolor="red",      label="Im finalen Set"),
            Patch(facecolor="steelblue",label="Nicht im finalen Set"),
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

    def plot_robust_capacity_breakdown(self, save=True):
        """Kapazitätsaufschlüsselung des robusten Portfolios."""
        if self.n_robust is None:
            print("Robustes Portfolio nicht verfügbar")
            return

        gens = self.n_robust.generators
        pcol = "p_nom_opt" if "p_nom_opt" in gens.columns else ("p_nom" if "p_nom" in gens.columns else None)
        if pcol is None or "carrier" not in gens.columns:
            print("Generators fehlen p_nom_opt/p_nom oder carrier.")
            return

        cap = gens.groupby("carrier")[pcol].sum()
        cap = cap[cap > 0].sort_values(ascending=False).drop(index=["load"], errors="ignore")

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
        NEU: Vergleicht installierte Kapazitäten (aus n_robust) mit den
        Dispatch-Ergebnissen pro Szenario — zeigt wie das robuste Portfolio
        unter verschiedenen Klimaszenarien performt.
        """
        if not self.scenario_networks:
            print("Keine Szenario-Netzwerke geladen — plot_scenario_capacity_comparison übersprungen.")
            return

        all_costs = self.aro_summary.get("aro_final_evaluation", {}).get("all_costs", {})

        # Sammle Gesamt-Erzeugung pro Szenario und Carrier
        records = []
        for scenario, n in self.scenario_networks.items():
            if n is None:
                continue
            if "carrier" not in n.generators.columns:
                continue
            # Erzeugung: p_t (dispatch) summiert
            if hasattr(n, "generators_t") and hasattr(n.generators_t, "p") and not n.generators_t.p.empty:
                gen_sum = n.generators_t.p.sum()
                gen_df  = n.generators[["carrier"]].copy()
                gen_df["energy_MWh"] = gen_sum
                by_carrier = gen_df.groupby("carrier")["energy_MWh"].sum()
            else:
                # Fallback: p_nom_opt * capacity_factor (grobes Schätzverfahren)
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

        pivot = df_top.pivot_table(index="scenario", columns="carrier", values="value", aggfunc="sum").fillna(0)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Gestapeltes Balkendiagramm
        colors = [self.config.CARRIER_COLORS.get(c, "#a9a9a9") for c in pivot.columns]
        pivot.plot(kind="bar", stacked=True, ax=axes[0], color=colors)
        axes[0].set_title(f"Erzeugung/Kapazität pro Szenario — N={len(pivot)}", fontsize=13)
        axes[0].set_xlabel("Szenario")
        axes[0].set_ylabel("MWh / MW")
        axes[0].legend(loc="upper right", fontsize=8, ncol=2)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Kosten vs. Erzeugung Scatter
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
            axes[1].set_title("Kosten vs. Erzeugung (je Szenario)", fontsize=13)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis("off")
            axes[1].text(0.1, 0.5, "Keine Kostendaten für Scatter.", fontsize=11)

        plt.suptitle(f"Szenario-Vergleich — {self.run_config.get('name', '')}", fontsize=15)
        plt.tight_layout()
        if save:
            p = self.config.get_plot_output_dir("scenario_comparison") / "scenario_capacity_comparison.png"
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
        NEU: Führt CountryAnalyzer für ALLE geladenen Szenario-Netzwerke durch.
        Ermöglicht vollständige Auswertung aller ARO-Szenarien.
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
            ("ARO Konvergenz",             self.plot_aro_convergence),
            ("Szenario-Kostenvergleich",    self.plot_scenario_cost_comparison),
            ("Robuste Kapazitäten",         self.plot_robust_capacity_breakdown),
            ("Szenario-Kapazitätsvergleich",self.plot_scenario_capacity_comparison),  # NEU
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
