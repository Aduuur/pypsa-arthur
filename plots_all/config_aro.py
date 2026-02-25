# config_aro.py
import os
from pathlib import Path


class AROPlottingConfig:
    """
    Konfiguration für ARO-Ergebnisse-Plots
    """

    def __init__(self):
        # ============================================================
        # PFADE ANPASSEN
        # ============================================================

        # Basis-Pfad zu deinen ARO-Ergebnissen
        self.BASE_RESULTS_PATH = "/home/endata/pypsa-eur/results/aro_runs"

        # Output-Pfad für Plots
        self.PLOT_OUTPUT_PATH = "/shared_endata/aro_analysis/plots"

        # ============================================================
        # ARO-SZENARIO DEFINITION
        # ============================================================

        # Definiere deine ARO-Runs
        self.ARO_RUNS = {
            "aro_run_1": {
                "name": "ARO 5 Szenarien",
                "summary_json": f"{self.BASE_RESULTS_PATH}/aro_run_1/aro_summary.json",
                "robust_network": f"{self.BASE_RESULTS_PATH}/aro_run_1/robust_portfolio.nc",
                "worst_case_dispatch": f"{self.BASE_RESULTS_PATH}/aro_run_1/worst_case_dispatch.nc",
                "worst_case_dispatch_std": f"{self.BASE_RESULTS_PATH}/aro_run_1/worst_case_dispatch_std.nc",
                "scenarios": ["1971", "1984", "1997", "2010", "2020"]
            },
            "aro_run_2": {
                "name": "ARO Klimaprojektionen",
                "summary_json": f"{self.BASE_RESULTS_PATH}/aro_run_2/aro_summary.json",
                "robust_network": f"{self.BASE_RESULTS_PATH}/aro_run_2/robust_portfolio.nc",
                "worst_case_dispatch": f"{self.BASE_RESULTS_PATH}/aro_run_2/worst_case_dispatch.nc",
                "scenarios": ["GCM1_RCP45", "GCM2_RCP45", "GCM3_RCP85"]
            }
        }

        # Aktuell ausgewählter Run
        self.SELECTED_RUN = "aro_run_1"

        # ============================================================
        # PLOT-EINSTELLUNGEN
        # ============================================================

        self.COUNTRIES_TO_ANALYZE = ["ALL", "DE", "FR", "ES", "CH"]

        self.FONT_SIZES = {
            "title": 20,
            "label": 15,
            "tick": 13,
            "legend": 11,
        }

        # Technologie-Farben (PyPSA-Eur Standard + eigene)
        self.CARRIER_COLORS = {
            # Wind
            'onwind': "#235ebc",
            'offwind-ac': "#6895dd",
            'offwind-dc': "#74c6f2",

            # Solar
            'solar': "#f9d002",
            'solar rooftop': '#ffea80',

            # Konventionell
            'OCGT': '#e0986c',
            'CCGT': '#a85522',
            'gas': '#e05b09',
            'nuclear': '#ff8c00',
            'coal': '#545454',
            'lignite': '#826837',

            # Wasserkraft
            'hydro': '#298c81',
            'ror': '#3dbfb0',
            'PHS': '#51dbcc',

            # Speicher & H2
            'battery': '#ace37f',
            'H2': '#bf13a0',
            'H2 Fuel Cell': '#c251ae',

            # Biomasse
            'biomass': '#baa741',
            'biogas': '#e3d37d',

            # Sonstige
            'load': '#1f77b4',
            'other': '#a9a9a9',
        }

        # ============================================================
        # ARO-SPEZIFISCHE PLOT-TYPEN
        # ============================================================

        self.ARO_PLOTS = {
            "convergence": True,  # ARO-Konvergenz-Plot
            "scenario_comparison": True,  # Kosten über Szenarien
            "worst_case_analysis": True,  # Worst-Case Deep-Dive
            "robustness_metrics": True,  # Robustheitskennzahlen
            "capacity_comparison": True,  # Robust vs. Deterministisch
        }

    def get_current_run_config(self):
        """Gibt Config für ausgewählten Run zurück"""
        return self.ARO_RUNS[self.SELECTED_RUN]

    def get_plot_output_dir(self, plot_type="general"):
        """Erstellt Outputordner für spezifischen Plot-Typ"""
        output_dir = Path(self.PLOT_OUTPUT_PATH) / self.SELECTED_RUN / plot_type
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
