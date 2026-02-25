"""
Batch-Verarbeitung für mehrere Länder und Szenarien
"""

from country_analysis_myopic import MyopicCountryAnalyzer
from pathlib import Path
import yaml
from typing import Dict, List


class BatchMyopicAnalyzer:
    """Batch-Verarbeitung für Myopic-Analysen"""

    def __init__(self, config_file: str = "batch_config.yaml"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """Lädt Konfiguration aus YAML-Datei"""
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Standard-Konfiguration erstellen
            default_config = {
                'scenarios': {
                    'base_scenario': {
                        'base_path': 'results/networks/base_s_37_',
                        'description': 'Base scenario with 37 nodes'
                    }
                },
                'countries': ['DE', 'FR', 'ES', 'IT'],
                'years': [2030, 2040, 2050],
                'output_base_dir': 'batch_results',
                'create_detailed_plots': False,
                'create_comparison_plots': True
            }

            with open(self.config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)

            print(f"Standard-Konfiguration erstellt: {self.config_file}")
            print("Bitte anpassen und erneut ausführen.")
            return default_config

    def run_batch_analysis(self):
        """Führt Batch-Analyse für alle konfigurierten Szenarien und Länder durch"""
        results = {}

        for scenario_name, scenario_config in self.config['scenarios'].items():
            print(f"\n{'=' * 60}")
            print(f"SZENARIO: {scenario_name}")
            print(f"Beschreibung: {scenario_config.get('description', 'Keine Beschreibung')}")
            print(f"{'=' * 60}")

            scenario_results = {}

            for country in self.config['countries']:
                print(f"\nAnalysiere {country} für Szenario {scenario_name}...")

                try:
                    output_dir = Path(self.config['output_base_dir']) / scenario_name / f"analysis_{country}"

                    analyzer = MyopicCountryAnalyzer(
                        network_base_path=scenario_config['base_path'],
                        country=country,
                        years=self.config.get('years'),
                        output_dir=str(output_dir)
                    )

                    # Myopic-Plots erstellen
                    analyzer.generate_all_myopic_plots(save=True)

                    # Optional: Detailplots für einzelne Jahre
                    if self.config.get('create_detailed_plots', False):
                        for year in analyzer.years:
                            analyzer.analyze_single_year(year, create_detailed_plots=True)

                    scenario_results[country] = analyzer
                    print(f"✓ {country} erfolgreich analysiert")

                except Exception as e:
                    print(f"✗ Fehler bei {country}: {e}")
                    scenario_results[country] = None

            results[scenario_name] = scenario_results

        # Vergleichsplots zwischen Ländern erstellen
        if self.config.get('create_comparison_plots', True):
            self.create_comparison_plots(results)

        return results

    def create_comparison_plots(self, results: Dict):
        """Erstellt Vergleichsplots zwischen Ländern und Szenarien"""
        print(f"\n{'=' * 40}")
        print("ERSTELLE VERGLEICHSPLOTS")
        print(f"{'=' * 40}")

        # Implementierung folgt...
        # Hier könnten Sie Plots erstellen, die verschiedene Länder
        # oder Szenarien miteinander vergleichen

        pass


def main():
    batch_analyzer = BatchMyopicAnalyzer()
    batch_analyzer.run_batch_analysis()


if __name__ == "__main__":
    main()