"""
Batch-Verarbeitung für mehrere Länder und Szenarien.
Verwendet master_config.py als Single Source of Truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from master_config import MasterConfig, PlottingConfig, AROPlottingConfig
from country_analysis_myopic import MyopicCountryAnalyzer


class BatchMyopicAnalyzer:
    """
    Batch-Verarbeitung für myopische/normale Analysen über mehrere
    Szenarien und Länder. Liest Konfiguration aus master_config.py.
    """

    def __init__(
        self,
        master: Optional[MasterConfig] = None,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        output_base_dir: Optional[str] = None,
        create_detailed_plots: bool = False,
        create_comparison_plots: bool = True,
    ):
        self.master   = master or MasterConfig()
        self.plot_cfg = PlottingConfig(master=self.master)

        self.countries    = countries or self.plot_cfg.get_countries()
        self.years        = years
        self.output_base  = Path(output_base_dir or self.plot_cfg.BASE_SAVE_PATH) / "batch_results"
        self.create_detailed_plots  = create_detailed_plots
        self.create_comparison_plots = create_comparison_plots

    def _get_all_normal_scenarios(self) -> Dict[str, List[str]]:
        """
        Gibt alle Szenarien mit run_type='normal' aus dem Registry zurück.
        Format: {scenario_key: [network_paths]}
        """
        reg = self.master.scenarios_registry
        result = {}
        for key, entry in reg.items():
            if self.master.get_run_type(key) == "normal":
                if isinstance(entry, dict):
                    result[key] = list(entry.get("networks", []))
                elif isinstance(entry, list):
                    result[key] = list(entry)
        return result

    def run_batch_analysis(self) -> Dict:
        """Führt Batch-Analyse für alle normalen Szenarien und Länder durch."""
        scenarios = self._get_all_normal_scenarios()

        if not scenarios:
            print("Keine normalen (non-ARO) Szenarien im registry gefunden.")
            return {}

        results = {}

        for scenario_name, network_paths in scenarios.items():
            print(f"\n{'=' * 60}")
            print(f"SZENARIO: {scenario_name}  ({len(network_paths)} Netzwerke)")
            print(f"{'=' * 60}")

            scenario_results = {}

            for country in self.countries:
                print(f"\n  Analysiere {country} für Szenario {scenario_name}...")

                # Versuche MyopicCountryAnalyzer (multi-year)
                # Fallback auf CountryAnalyzer pro Netzwerk
                try:
                    # Gemeinsamer Basispfad ermitteln (für MyopicCountryAnalyzer)
                    if network_paths:
                        import re
                        # Versuche Basispfad herauszufiltern (alles vor dem Jahr)
                        base_path = re.sub(r"_?(?:base_s_\d+_+)?\d{4}\.nc$", "", network_paths[0])
                        output_dir = self.output_base / scenario_name / f"analysis_{country}"

                        analyzer = MyopicCountryAnalyzer(
                            network_base_path=base_path,
                            country=country,
                            years=self.years,
                            output_dir=str(output_dir),
                        )
                        analyzer.generate_all_myopic_plots(save=True)

                        if self.create_detailed_plots:
                            for year in (self.years or analyzer.years):
                                analyzer.analyze_single_year(year, create_detailed_plots=True)

                        scenario_results[country] = analyzer
                        print(f"  ✓ {country}")

                except Exception as e:
                    print(f"  ✗ {country} (MyopicCountryAnalyzer): {e}")
                    # Fallback: einzelne Netzwerke per CountryAnalyzer
                    from country_analysis import CountryAnalyzer
                    for net_path in network_paths:
                        if not Path(net_path).is_file():
                            continue
                        try:
                            import re
                            m = re.search(r"(\d{4})\.nc$", net_path)
                            tag = m.group(1) if m else Path(net_path).stem
                            out_dir = self.output_base / scenario_name / f"analysis_{country}" / tag
                            ca = CountryAnalyzer(
                                network_path=net_path,
                                country=country,
                                output_dir=str(out_dir),
                                carrier_colors=self.plot_cfg.CARRIER_COLORS,
                                default_color=self.plot_cfg.DEFAULT_COLOR,
                            )
                            ca.summary_report()
                            ca.generate_all_plots(save=True)
                            print(f"  ✓ {country} ({tag}) [Fallback]")
                        except Exception as e2:
                            print(f"  ✗ {country} ({net_path}): {e2}")

            results[scenario_name] = scenario_results

        if self.create_comparison_plots:
            self.create_comparison_plots_fn(results)

        return results

    def create_comparison_plots_fn(self, results: Dict):
        """Erstellt Vergleichsplots zwischen Ländern und Szenarien."""
        print(f"\n{'=' * 40}")
        print("ERSTELLE VERGLEICHSPLOTS")
        print(f"{'=' * 40}")
        # Erweiterbar: z.B. installierte Kapazitäten über Szenarien vergleichen
        pass


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default=None, help="Komma-getrennte Länder")
    ap.add_argument("--years",     default=None, help="Komma-getrennte Jahre")
    ap.add_argument("--output",    default=None, help="Output-Verzeichnis")
    ap.add_argument("--detailed",  action="store_true")
    args = ap.parse_args()

    countries = [x.strip() for x in args.countries.split(",")] if args.countries else None
    years     = [int(x)    for x in args.years.split(",")]     if args.years     else None

    analyzer = BatchMyopicAnalyzer(
        countries=countries,
        years=years,
        output_base_dir=args.output,
        create_detailed_plots=args.detailed,
    )
    analyzer.run_batch_analysis()


if __name__ == "__main__":
    main()
