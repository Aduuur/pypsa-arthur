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
from master_config import AROPlottingConfig

# Matplotlib für Serverumgebung
import matplotlib

matplotlib.use('Agg')
sns.set_theme("paper", style="whitegrid")


class AROAnalyzer:
    """
    Analyzer für ARO-Ergebnisse
    """

    def __init__(self, config: AROPlottingConfig = None):
        if config is None:
            config = AROPlottingConfig()

        self.config = config
        self.run_config = config.get_current_run_config()

        # Lade ARO Summary JSON
        self.load_aro_summary()

        # Lade Netzwerke
        self.load_networks()

        print(f"ARO Analyzer initialisiert für: {self.run_config['name']}")

    def load_aro_summary(self):
        """Lädt ARO Summary JSON"""
        summary_path = Path(self.run_config['summary_json'])

        if not summary_path.exists():
            raise FileNotFoundError(f"Summary JSON nicht gefunden: {summary_path}")

        with open(summary_path, 'r') as f:
            self.aro_summary = json.load(f)

        print(f"ARO Summary geladen: {summary_path}")
        print(f"  - Iterationen: {self.aro_summary.get('aro_convergence', {}).get('iterations_run', '?')}")
        print(f"  - Finale Szenarien: {len(self.aro_summary.get('aro_final_scenarios', []))}")

    def load_networks(self):
        """Lädt robustes Portfolio und Worst-Case Dispatch"""
        # Robustes Portfolio
        portfolio_path = Path(self.run_config['robust_network'])
        if portfolio_path.exists():
            self.n_robust = pypsa.Network(str(portfolio_path))
            print(f"Robustes Portfolio geladen: {len(self.n_robust.generators)} Generatoren")
        else:
            self.n_robust = None
            print(f"WARNUNG: Portfolio nicht gefunden: {portfolio_path}")

        # Worst-Case Dispatch (Standard-Format wenn verfügbar)
        dispatch_std_path = Path(self.run_config.get('worst_case_dispatch_std', ''))
        if dispatch_std_path.exists():
            self.n_worst_case = pypsa.Network(str(dispatch_std_path))
            print(f"Worst-Case Dispatch geladen: {len(self.n_worst_case.snapshots)} Snapshots")
        else:
            self.n_worst_case = None
            print("Worst-Case Dispatch nicht verfügbar")

    def plot_aro_convergence(self, save=True):
        """1. ARO-Konvergenzplot"""
        if 'aro_history' not in self.aro_summary:
            print("Keine ARO History verfügbar")
            return

        history = self.aro_summary['aro_history']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Worst-Case Kosten pro Iteration
        iterations = [h['iteration'] for h in history]
        worst_costs = [h['worst_cost'] for h in history]
        worst_in_set = [h['worst_in_set'] for h in history]

        axes[0, 0].plot(iterations, worst_costs, 'o-', label='Worst-Case (alle)', linewidth=2, markersize=8)
        axes[0, 0].plot(iterations, worst_in_set, 's-', label='Worst-Case (im Set)', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Kosten [€/a]')
        axes[0, 0].set_title('ARO Konvergenz - Kosten')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Gap pro Iteration
        gaps = [h['aro_gap'] * 100 for h in history]
        convergence_tol = history[0].get('convergence_tol', 0) * 100

        axes[0, 1].plot(iterations, gaps, 'o-', linewidth=2, markersize=8, color='red')
        axes[0, 1].axhline(y=convergence_tol, color='green', linestyle='--', label=f'Toleranz ({convergence_tol:.2f}%)')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Gap [%]')
        axes[0, 1].set_title('ARO Konvergenz - Gap')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')  # Log-Skala für Gap

        # Szenario-Hinzufügung pro Iteration
        scenarios_added = []
        for h in history:
            cutout = h.get('worst_cutout', '?')
            scenarios_added.append(cutout)

        axes[1, 0].barh(range(len(scenarios_added)), [1] * len(scenarios_added), tick_label=scenarios_added)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_title('Hinzugefügte Szenarien')
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        # Kosten-Verteilung finale Evaluation
        if 'aro_final_evaluation' in self.aro_summary:
            final_costs = self.aro_summary['aro_final_evaluation']['all_costs']
            costs_list = list(final_costs.values())
            scenarios_list = list(final_costs.keys())

            axes[1, 1].bar(range(len(costs_list)), costs_list, tick_label=scenarios_list)
            axes[1, 1].axhline(y=self.aro_summary['aro_final_evaluation']['worst_case_total_cost'],
                               color='red', linestyle='--', label='Worst-Case')
            axes[1, 1].set_xlabel('Szenario')
            axes[1, 1].set_ylabel('Kosten [€/a]')
            axes[1, 1].set_title('Kosten pro Szenario (finale Evaluation)')
            axes[1, 1].legend()
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.suptitle(f"ARO Konvergenzanalyse - {self.run_config['name']}", fontsize=16)
        plt.tight_layout()

        if save:
            output_path = self.config.get_plot_output_dir("aro_metrics") / "aro_convergence.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Gespeichert: {output_path}")

        plt.close()

    def plot_scenario_cost_comparison(self, save=True):
        """2. Kosten-Vergleich über alle Szenarien"""
        if 'aro_final_evaluation' not in self.aro_summary:
            print("Keine finale Evaluation verfügbar")
            return

        final_costs = self.aro_summary['aro_final_evaluation']['all_costs']

        # Sortiere nach Kosten
        sorted_costs = dict(sorted(final_costs.items(), key=lambda x: x[1]))

        scenarios = list(sorted_costs.keys())
        costs = list(sorted_costs.values())

        # Markiere finale Szenarien
        final_scenarios = set(self.aro_summary.get('aro_final_scenarios', []))
        colors = ['red' if s in final_scenarios else 'steelblue' for s in scenarios]

        fig, ax = plt.subplots(figsize=(14, 8))

        bars = ax.bar(range(len(costs)), costs, color=colors)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.set_xlabel('Szenario', fontsize=12)
        ax.set_ylabel('Gesamtkosten [€/a]', fontsize=12)
        ax.set_title(f'Kosten pro Szenario - {self.run_config["name"]}', fontsize=16)

        # Legende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Im finalen Set'),
            Patch(facecolor='steelblue', label='Nicht im finalen Set')
        ]
        ax.legend(handles=legend_elements)

        # Annotationen für Min/Max
        min_cost = min(costs)
        max_cost = max(costs)
        min_idx = costs.index(min_cost)
        max_idx = costs.index(max_cost)

        ax.annotate(f'Min: {min_cost / 1e9:.2f} Mrd. €',
                    xy=(min_idx, min_cost),
                    xytext=(min_idx, min_cost * 1.05),
                    arrowprops=dict(arrowstyle='->', color='green'))

        ax.annotate(f'Max: {max_cost / 1e9:.2f} Mrd. €',
                    xy=(max_idx, max_cost),
                    xytext=(max_idx, max_cost * 0.95),
                    arrowprops=dict(arrowstyle='->', color='red'))

        plt.tight_layout()

        if save:
            output_path = self.config.get_plot_output_dir("aro_metrics") / "scenario_cost_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Gespeichert: {output_path}")

        plt.close()

    def plot_robust_capacity_breakdown(self, save=True):
        """3. Kapazitätsaufschlüsselung des robusten Portfolios"""
        if self.n_robust is None:
            print("Robustes Portfolio nicht verfügbar")
            return

        # Kapazitäten pro Carrier
        capacity_by_carrier = self.n_robust.generators.groupby('carrier')['p_nom_opt'].sum()
        capacity_by_carrier = capacity_by_carrier[capacity_by_carrier > 0].sort_values(ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar Chart
        colors = [self.config.CARRIER_COLORS.get(c, '#a9a9a9') for c in capacity_by_carrier.index]
        capacity_by_carrier.plot(kind='bar', ax=axes[0], color=colors)
        axes[0].set_title('Installierte Kapazität nach Technologie', fontsize=14)
        axes[0].set_ylabel('Kapazität [MW]', fontsize=12)
        axes[0].set_xlabel('Technologie', fontsize=12)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Pie Chart
        axes[1].pie(capacity_by_carrier.values,
                    labels=capacity_by_carrier.index,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90)
        axes[1].set_title('Kapazitätsanteile', fontsize=14)

        plt.suptitle(f'Robustes Portfolio - Kapazitäten - {self.run_config["name"]}', fontsize=16)
        plt.tight_layout()

        if save:
            output_path = self.config.get_plot_output_dir("capacity") / "robust_capacity_breakdown.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Gespeichert: {output_path}")

        plt.close()

    def analyze_worst_case_dispatch(self, country="DE", save=True):
        """4. Worst-Case Dispatch Analyse für spezifisches Land"""
        if self.n_worst_case is None:
            print("Worst-Case Dispatch nicht verfügbar")
            return

        # Importiere Country Analyzer für Detailanalyse
        from country_analysis import CountryAnalyzer

        # Erstelle temporäres Netzwerk-File
        temp_network_path = Path("/tmp/worst_case_temp.nc")
        self.n_worst_case.export_to_netcdf(str(temp_network_path))

        # Nutze bestehenden Analyzer
        analyzer = CountryAnalyzer(
            network_path=str(temp_network_path),
            country=country,
            output_dir=str(self.config.get_plot_output_dir("worst_case") / country)
        )

        # Generiere alle Plots
        analyzer.generate_all_plots(save=save)

        # Cleanup
        if temp_network_path.exists():
            temp_network_path.unlink()

        print(f"Worst-Case Analyse für {country} abgeschlossen")

    def generate_all_aro_plots(self, save=True):
        """Generiert alle ARO-spezifischen Plots"""
        print(f"\n=== Generiere ARO-Plots für {self.run_config['name']} ===\n")

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
        print(f"ARO ANALYSE-BERICHT: {self.run_config['name']}")
        print(f"{'=' * 60}")

        conv = self.aro_summary.get('aro_convergence', {})
        print(f"\nKonvergenz:")
        print(f"  - Iterationen durchgeführt: {conv.get('iterations_run', '?')} / {conv.get('max_iter', '?')}")
        print(f"  - Konvergenzgrund: {conv.get('reason', 'Unbekannt')}")

        final_eval = self.aro_summary.get('aro_final_evaluation', {})
        print(f"\nFinale Evaluation:")
        print(f"  - Worst-Case Cutout: {final_eval.get('worst_case_cutout', '?')}")
        print(f"  - Worst-Case Kosten: {final_eval.get('worst_case_total_cost', 0) / 1e9:.2f} Mrd. €/a")
        print(f"  - Finale Gap: {final_eval.get('aro_gap', 0) * 100:.4f}%")

        print(f"\nFinale Szenarien ({len(self.aro_summary.get('aro_final_scenarios', []))}):")
        for s in self.aro_summary.get('aro_final_scenarios', []):
            cost = final_eval.get('all_costs', {}).get(s, 0)
            print(f"  - {s}: {cost / 1e9:.2f} Mrd. €/a")


# Hauptfunktion für CLI-Nutzung
def main():
    import argparse

    parser = argparse.ArgumentParser(description='ARO Ergebnisse analysieren')
    parser.add_argument('--run', default=None, help='ARO Run Name (aus config_aro.py)')
    parser.add_argument('--countries', nargs='+', default=["DE"], help='Länder für Detailanalyse')
    parser.add_argument('--output', default=None, help='Output-Pfad (überschreibt Config)')

    args = parser.parse_args()

    # Config erstellen
    config = AROPlottingConfig()

    if args.run:
        config.SELECTED_RUN = args.run

    if args.output:
        config.PLOT_OUTPUT_PATH = args.output

    # Analyzer erstellen
    analyzer = AROAnalyzer(config)

    # Summary
    analyzer.summary_report()

    # ARO-Plots
    analyzer.generate_all_aro_plots(save=True)

    # Worst-Case Analyse pro Land
    for country in args.countries:
        print(f"\n=== Analysiere Worst-Case für {country} ===")
        analyzer.analyze_worst_case_dispatch(country=country, save=True)

    print(f"\n✓ Analyse abgeschlossen!")
    print(f"Plots gespeichert in: {config.PLOT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
