#!/usr/bin/env python3
# master_config.py
# =============================================================================
# Zentrale Master-Konfigurationsdatei für PyPSA-Eur Auswertung
# =============================================================================

from __future__ import annotations

import os
import json
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _mkdir(p: Union[str, Path]) -> Path:
    p = _as_path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------------------------------------------------------
# RUN NAMES
# ARO_RUN_NAME   : der eigentliche ARO-Lauf (robustes Portfolio + Dispatch-Netze)
# REF_RUN_NAME   : der deterministische Referenz-Run (Basisrun-rcp45-2028)
#                  wird für Vergleichsplots (plot_installed_cap_new_vgl etc.) genutzt
# RUN_NAME       : aktiv selektierter Run (für normale Plot-Skripte)
# -----------------------------------------------------------------------------
ARO_RUN_NAME: str = _env("ARO_RUN_NAME", "Referenzrun-rcp45")       # type: ignore[assignment]
REF_RUN_NAME: str = _env("REF_RUN_NAME", "Referenzrun-rcp45") # type: ignore[assignment]
RUN_NAME: str = _env("RUN_NAME", ARO_RUN_NAME)                   # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Master configuration block
# -----------------------------------------------------------------------------
# Globaler Plot-Output-Override fuer per-Szenario Auswertung
_PLOT_OUTPUT_OVERRIDE: Optional[str] = None

MASTER_CONFIG: Dict[str, Any] = {
    # =======================================================================
    # GLOBAL PATHS
    # =======================================================================
    "paths": {
        "pypsa_results_base": _env("PYPSA_RESULTS_BASE", "/home/endata/PycharmProjects/pypsa-ee/results"),
        "aro_results_base":   _env("ARO_RESULTS_BASE",   "/home/endata/PycharmProjects/pypsa-ee/results"),
        # plots_base ist die EINZIGE Output-Wurzel für alle Plots.
        # Jeder Run landet als Unterordner: plots_base/<run_key>/
        "plots_base":         _env("PLOTS_BASE",         "/mnt/endata/MA_Arthur/PyPSA-results/plots"),
        "aro_plots_base":     _env("ARO_PLOTS_BASE",     "/mnt/endata/MA_Arthur/PyPSA-results/plots"),
    },

    # =======================================================================
    # COUNTRY SETTINGS
    # =======================================================================
    "countries": {
        "default_countries_to_plot": [
            "ALL", "DE", "FR", "ES", "CH", "DK", "SE", "NO", "IT", "GB",
            "NL", "PL", "BE", "FI", "AT", "PT", "CZ", "LT", "LV", "EE",
        ],
        "names": {
            "DE": "Deutschland", "FR": "Frankreich", "ES": "Spanien",
            "IT": "Italien",     "NL": "Niederlande","BE": "Belgien",
            "AT": "Österreich",  "CH": "Schweiz",    "PL": "Polen",
            "DK": "Dänemark",    "SE": "Schweden",   "NO": "Norwegen",
            "FI": "Finnland",    "GB": "Großbritannien","IE": "Irland",
            "PT": "Portugal",    "CZ": "Tschechien", "LT": "Litauen",
            "LV": "Lettland",    "EE": "Estland",    "ALL": "Gesamtnetz",
        },
    },

    # =======================================================================
    # DARK SKY / DUNKELFLAUTE ANALYSIS PERIOD
    # =======================================================================
    "planning_year": int(_env("PLANNING_YEAR", "2050")),
    "dark_sky_period": {
        "reference_year": int(_env("DARK_SKY_YEAR", "2030")),
        "start_mmdd":     _env("DARK_SKY_START", "01-07"),
        "end_mmdd":       _env("DARK_SKY_END",   "01-28"),
    },

    "dispatch_windows": {},

    # =======================================================================
    # PLOT STYLE
    # =======================================================================
    "plotting": {
        "matplotlib_style": _env("PLOT_STYLE", "seaborn-v0_8-whitegrid"),
        "dpi":              int(_env("PLOT_DPI",    "300")),
        "bbox_inches":      _env("PLOT_BBOX",   "tight"),
        "save_format":      _env("PLOT_FORMAT", "png"),
        "figsizes":   {"default": (12, 8), "large": (16, 12), "wide": (14, 6)},
        "font_sizes": {"title": 20, "label": 15, "tick": 13, "bar_label": 10, "legend": 11},
        "default_color": "#a9a9a9",
    },

    # =======================================================================
    # TECHNOLOGY / CARRIER COLOR PALETTE
    # =======================================================================
    "colors": {
        "carriers": {
            "onwind": "#235ebc", "onshore wind": "#235ebc",
            "offwind": "#004E8A", "offwind-ac": "#6895dd",
            "offwind-dc": "#74c6f2", "offshore wind": "#6895dd",
            "offwind-float": "#15a0bf",
            "solar": "#f9d002", "solar rooftop": "#ffea80", "solar-hsat": "#FFF080",
            "hydro": "#298c81", "ror": "#3dbfb0", "run of river": "#3dbfb0", "PHS": "#51dbcc",
            "gas": "#e05b09", "OCGT": "#e0986c", "CCGT": "#a85522",
            "coal": "#545454", "lignite": "#826837", "oil": "#c9c9c9",
            "oil primary": "#7a7a7a", "nuclear": "#ff8c00",
            "biomass": "#baa741", "Biomasse": "#baa741", "solid biomass": "#baa741",
            "municipal solid waste": "#91ba41", "biogas": "#e3d37d", "waste": "#e3d37d",
            "H2": "#bf13a0", "hydrogen": "#bf13a0", "H2 storage": "#bf13a0",
            "H2 Electrolysis": "#ff29d9", "H2 Fuel Cell": "#c251ae", "H2 turbine": "#991f83",
            "battery": "#ace37f", "home battery": "#80c944",
            "battery charger": "#76c7a3", "battery discharger": "#7b2d8b",
            "load": "#1f77b4", "heat pump": "#ff9966", "resistive heater": "#ff7f50",
            "EV charger": "#9999ff", "export": "#808080",
            "geothermal": "#ba91b1", "other": "#000000",
        },
    },

    # =======================================================================
    # SCENARIO REGISTRY (deterministic / myopic / normal)
    # =======================================================================
    "scenarios": {
        "selection": _env("SCENARIO_SELECTION", RUN_NAME),

        "registry": {
            # -----------------------------------------------------------------
            # ARO-Run (robustes Portfolio-Optimierungsergebnis)
            # -----------------------------------------------------------------
            ARO_RUN_NAME: {
                "run_type": "aro",
                "description": "ARO-Lauf mit robustem Portfolio (big-aro-run-2)",
                # Robustes Netz für Kapazitätsvergleich (wird von run_loader genutzt
                # wenn nur aro_include_scenarios=False -> "robust"-Tag)
                "networks": [
                    f"/home/endata/PycharmProjects/pypsa-ee/results/{ARO_RUN_NAME}/networks/aro_robust__std.nc",
                ],
            },
            # -----------------------------------------------------------------
            # Deterministischer Referenz-Run (Basisrun-rcp45-2028)
            # Wird als Vergleichsnetz in plot_installed_cap_new_vgl etc. genutzt.
            # Das n_roth / aro_robust_network in den Plots zeigt auf diesen Run.
            # -----------------------------------------------------------------
            REF_RUN_NAME: {
                "run_type": "normal",
                "description": "Deterministischer Referenz-Run rcp4.5 / 2028",
                "networks": [
                    f"/home/endata/PycharmProjects/pypsa-ee/results/{REF_RUN_NAME}/networks/base_s_24___2050.nc",
                ],
            },
        },

        # FIX #2: both_mapping muss auf tatsächlich existierende Registry-Keys
        # zeigen. Früher standen hier "new_avg" / "dunkelflaute_neu_2" — die
        # nicht im registry waren, was get_networks("both") immer leer ließ.
        #
        # Bedeutung im Vergleichsmodus (SCENARIO_SELECTION = "both"):
        #   "average"      → linke Säule  = deterministischer Basisrun
        #   "dunkelflaute" → rechte Säule = ARO-Robustes-Portfolio
        #
        # Im ARO-Worst-Case-Vergleich (via _inject_aro_dispatch_network) werden
        # diese Keys temporär auf "__aro_basis_reference__" / "__aro_worst_case_dispatch__"
        # umgebogen — das bleibt unverändert.
        "both_mapping": {
            "average":      REF_RUN_NAME,   # deterministischer Basisrun (rechte Seite Vergleich)
            "dunkelflaute": ARO_RUN_NAME,   # robustes ARO-Portfolio  (linke Seite Vergleich)
        },
    },

    # =======================================================================
    # AVAILABLE PLOTS
    # =======================================================================
    "plots": {
        "selection": _env("PLOTS_TO_RUN", "all"),
        "available": {
            "installed_capacity": "plot_installed_capacity",
            "energy_generated":   "plot_energy_generated",
            "storage":            "plot_storage",
            "price":              "plot_price",
            "losses":             "plot_losses",
            "demand_vs_generation": "plot_demand_vs_generation",
            "gas_h2_usage":       "plot_gas_h2_usage",
            "capacity_expansion": "plot_capacity_expansion",
            "demand_profile":     "plot_demand_profile",
            "generation_profile": "plot_generation_profile",
        },
    },

    # =======================================================================
    # ARO RUN REGISTRY
    # =======================================================================
    "aro": {
        # Aktiv selektierter ARO-Run (= der Run dessen Dispatch-Netze geladen werden)
        "selected_run": _env("ARO_SELECTED_RUN", ARO_RUN_NAME),
        "countries_to_analyze": ["ALL", "DE", "FR", "ES", "CH"],
        "plot_toggles": {
            "convergence":          True,
            "scenario_comparison":  True,
            "worst_case_analysis":  False,
            "robustness_metrics":   True,
            "capacity_comparison":  True,
        },

        "runs": {
            ARO_RUN_NAME: {
                "name":                    ARO_RUN_NAME,
                "summary_json":            f"{{aro_results_base}}/{ARO_RUN_NAME}/aro_summary.json",
                # Robustes Portfolio-Netzwerk (das ARO-Ergebnis selbst)
                "robust_network":          f"{{aro_results_base}}/{ARO_RUN_NAME}/networks/aro_robust.nc",
                "robust_network_std":      f"{{aro_results_base}}/{ARO_RUN_NAME}/networks/aro_robust__std.nc",
                # Deterministisches Referenznetz für Vergleichsplots (n_roth / aro_robust_network
                # in plot_installed_cap_new_vgl.py und aro_analysis.py).
                # Bezieht sich IMMER auf Basisrun-rcp45-2028.
                "reference_network":       f"{{pypsa_results_base}}/{REF_RUN_NAME}/networks/base_s_24___2050.nc",
                "worst_case_dispatch":     None,
                "worst_case_dispatch_std": None,
                "dispatch_paths":          {},
                # Optional: run-spezifische Dunkelflautenfenster pro Dispatch/Szenario.
                # Beispiel:
                # "dispatch_windows": {
                #     "__default__": {"reference_year": 2028, "start_mmdd": "01-07", "end_mmdd": "01-28"},
                #     "cutout_2012": {"reference_year": 2012, "start_mmdd": "01-15", "end_mmdd": "01-29"},
                # },
                "dispatch_windows":       {},
                "scenarios": [],
            },
        },
    },
}


# -----------------------------------------------------------------------------
# Dataclasses wrapping the dict
# -----------------------------------------------------------------------------

def get_dunkelflaute_window(network_path: str, duration_days: int = 7) -> tuple[str, str]:
    """
    Leitet das Dunkelflaute-Fenster aus dem Dispatch-Dateinamen ab.

    Dateiname-Muster: ...stress_04_from_2040_12_12...
    → Monat/Tag = 12-12, Jahr aus n.snapshots[0].year
    → start = YYYY-12-12, end = start + 7 Tage

    Fallback: erste 7 Tage des Netzwerks wenn kein Datum im Namen.
    """
    import re
    from pathlib import Path
    import pandas as pd

    fname = Path(network_path).name

    # Datum aus Dateiname parsen: _from_YYYY_MM_DD
    m = re.search(r"_from_\d{4}_(\d{2})_(\d{2})", fname)

    # Jahr aus Snapshots lesen
    try:
        import pypsa
        n = pypsa.Network(network_path)
        snap_year = int(n.snapshots[0].year)
    except Exception:
        snap_year = 2028  # Fallback

    if m:
        month, day = int(m.group(1)), int(m.group(2))
        start = pd.Timestamp(year=snap_year, month=month, day=day)
    else:
        # Fallback: erste Snapshots
        try:
            start = pd.Timestamp(n.snapshots[0])
        except Exception:
            start = pd.Timestamp(f"{snap_year}-01-07")

    end = start + pd.Timedelta(days=duration_days - 1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def get_planning_year(n=None, network_path: str = "", fallback: int = 2050) -> int:
    """
    Gibt das Planungsjahr zurueck (z.B. 2050), NICHT das Wetterjahr (2028).
    
    Prioritaet:
    1. Jahreszahl im Dateinamen: base_s_24___2050.nc -> 2050
    2. fallback-Parameter (default: 2050)
    
    n.snapshots[0].year wird NICHT verwendet da das das Wetterjahr ist.
    """
    import re
    from pathlib import Path
    if network_path:
        m = re.search(r"_(\d{4})\.nc$", Path(network_path).name)
        if m:
            return int(m.group(1))
    return fallback


@dataclass
class MasterConfig:
    raw: Dict[str, Any] = field(default_factory=lambda: MASTER_CONFIG)

    # ---------- PATHS ----------
    @property
    def paths(self) -> Dict[str, str]:
        return self.raw["paths"]

    @property
    def pypsa_results_base(self) -> str:
        return self.paths["pypsa_results_base"]

    @property
    def aro_results_base(self) -> str:
        return self.paths["aro_results_base"]

    @property
    def plots_base(self) -> str:
        return self.paths["plots_base"]

    @property
    def aro_plots_base(self) -> str:
        return self.paths.get("aro_plots_base", self.paths["plots_base"])

    # ---------- UNIFIED OUTPUT DIR ----------
    def get_run_output_dir(self, run_key: str, sub: Optional[str] = None) -> Path:
        """
        Gibt den kanonischen Output-Ordner für einen Run zurück und legt ihn an:

          <plots_base>/<run_key>/           <- Root des Runs
          <plots_base>/<run_key>/<sub>/     <- falls sub angegeben

        Das ist die EINZIGE Funktion die Output-Pfade erzeugt.
        Sowohl ARO- als auch Normal-Runs landen hier.
        """
        p = Path(self.plots_base) / run_key
        if sub:
            p = p / sub
        return _mkdir(p)

    # ---------- DARK SKY PERIOD ----------
    @property
    def dark_sky_period(self) -> Dict[str, Any]:
        return self.raw["dark_sky_period"]

    @property
    def dark_sky_reference_year(self) -> int:
        return int(self.dark_sky_period["reference_year"])

    @property
    def dark_sky_start(self) -> str:
        return f"{self.dark_sky_reference_year}-{self.dark_sky_period['start_mmdd']}"

    @property
    def dark_sky_end(self) -> str:
        return f"{self.dark_sky_reference_year}-{self.dark_sky_period['end_mmdd']}"

    @property
    def planning_year(self) -> int:
        return int(self.raw.get("planning_year", 2050))

    @property
    def dispatch_windows(self) -> Dict[str, Any]:
        return dict(self.raw.get("dispatch_windows", {}))

    @staticmethod
    def _format_period(period: Dict[str, Any]) -> Dict[str, Any]:
        ref = int(period["reference_year"])
        start = f"{ref}-{period['start_mmdd']}"
        end = f"{ref}-{period['end_mmdd']}"
        return {
            "reference_year": ref,
            "start_mmdd": str(period["start_mmdd"]),
            "end_mmdd": str(period["end_mmdd"]),
            "start": start,
            "end": end,
        }

    def get_dispatch_window(
        self,
        dispatch_tag: Optional[str] = None,
        run_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        global_default = self._format_period(self.dark_sky_period)
        windows = self.dispatch_windows
        tag = str(dispatch_tag) if dispatch_tag is not None else None
        rk = str(run_key) if run_key is not None else None

        if rk and rk in self.aro_runs:
            run_conf = self.aro_runs.get(rk, {})
            run_windows = run_conf.get("dispatch_windows", {}) or {}
            if tag and tag in run_windows:
                return self._format_period(run_windows[tag])
            if "__default__" in run_windows:
                return self._format_period(run_windows["__default__"])

        if rk and rk in windows and isinstance(windows[rk], dict):
            run_windows = windows[rk]
            if tag and tag in run_windows:
                return self._format_period(run_windows[tag])
            if "__default__" in run_windows:
                return self._format_period(run_windows["__default__"])

        if tag and tag in windows:
            return self._format_period(windows[tag])

        return global_default

    # ---------- COUNTRIES ----------
    @property
    def country_names(self) -> Dict[str, str]:
        return self.raw["countries"]["names"]

    @property
    def default_countries_to_plot(self) -> List[str]:
        return list(self.raw["countries"]["default_countries_to_plot"])

    # ---------- PLOTTING ----------
    @property
    def plotting(self) -> Dict[str, Any]:
        return self.raw["plotting"]

    @property
    def font_sizes(self) -> Dict[str, int]:
        return dict(self.plotting["font_sizes"])

    @property
    def default_color(self) -> str:
        return self.plotting["default_color"]

    @property
    def carrier_colors(self) -> Dict[str, str]:
        return dict(self.raw["colors"]["carriers"])

    # ---------- SCENARIOS ----------
    @property
    def scenario_selection(self) -> str:
        return self.raw["scenarios"]["selection"]

    @property
    def scenarios_registry(self) -> Dict[str, Any]:
        return self.raw["scenarios"]["registry"]

    # ---------- PLOTS ----------
    @property
    def plots_selection(self) -> Union[str, List[str]]:
        sel = self.raw["plots"]["selection"]
        if sel == "all":
            return "all"
        if isinstance(sel, str) and "," in sel:
            return [s.strip() for s in sel.split(",") if s.strip()]
        return sel

    @property
    def available_plots(self) -> Dict[str, str]:
        return dict(self.raw["plots"]["available"])

    # ---------- ARO ----------
    @property
    def aro_selected_run(self) -> str:
        return self.raw["aro"]["selected_run"]

    @property
    def aro_runs(self) -> Dict[str, Any]:
        return self.raw["aro"]["runs"]

    @property
    def aro_plot_toggles(self) -> Dict[str, bool]:
        return dict(self.raw["aro"]["plot_toggles"])

    @property
    def aro_countries_to_analyze(self) -> List[str]:
        return list(self.raw["aro"]["countries_to_analyze"])

    # ---------- Helpers ----------
    def get_run_type(self, run_key: Optional[str] = None) -> str:
        key = run_key or self.scenario_selection
        reg_entry = self.scenarios_registry.get(key, {})
        if isinstance(reg_entry, dict):
            t = reg_entry.get("run_type", "").lower()
            if t in ("aro", "normal"):
                return t
        if key in self.aro_runs:
            return "aro"
        return "normal"

    def resolve_template(self, s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        return s.format(**self.paths)

    def get_reference_network_path(self, aro_run_key: Optional[str] = None) -> Optional[str]:
        """
        Gibt den Pfad zum deterministischen Referenznetz zurück.
        Verwendet den `reference_network`-Eintrag im ARO-Run-Config.
        Fallback: Netzwerk des REF_RUN_NAME aus scenarios.registry.

        Dies ist das Netz das in Vergleichsplots als "n_roth" / Basisrun
        genutzt wird – immer Basisrun-rcp45-2028, nicht das robuste Portfolio.
        """
        rk = aro_run_key or self.aro_selected_run
        run_conf = self.aro_runs.get(rk, {})

        # 1. Explizit im ARO-Run-Eintrag konfiguriert
        ref_raw = run_conf.get("reference_network")
        if ref_raw:
            resolved = self.resolve_template(ref_raw)
            if resolved:
                return resolved

        # 2. Fallback: erstes Netzwerk des REF_RUN_NAME aus scenarios.registry
        ref_entry = self.scenarios_registry.get(REF_RUN_NAME, {})
        if isinstance(ref_entry, dict):
            nets = ref_entry.get("networks", [])
            if nets:
                return nets[0]
        elif isinstance(ref_entry, list) and ref_entry:
            return ref_entry[0]

        return None

    def get_networks(
        self,
        run_key: Optional[str] = None,
    ) -> Union[List[str], Dict[str, List[str]], None]:
        sel = run_key or self.scenario_selection
        reg = self.scenarios_registry

        if sel == "all":
            out: Dict[str, List[str]] = {}
            for k, v in reg.items():
                if isinstance(v, dict):
                    out[k] = list(v.get("networks", []))
                elif isinstance(v, list):
                    out[k] = list(v)
            return out

        if sel == "both":
            mapping = self.raw["scenarios"]["both_mapping"]
            out = {}
            for out_key, scenario_key in mapping.items():
                entry = reg.get(scenario_key, [])
                if isinstance(entry, dict):
                    out[out_key] = list(entry.get("networks", []))
                else:
                    out[out_key] = list(entry)
            return out

        entry = reg.get(sel)
        if entry is None:
            return None
        if isinstance(entry, dict):
            return list(entry.get("networks", []))
        return list(entry)

    def get_countries(self, override: Optional[List[str]] = None) -> List[str]:
        countries = override if override is not None else self.default_countries_to_plot
        if "ALL" in countries:
            return ["ALL"] + [c for c in countries if c != "ALL"]
        return countries

    def get_plots_to_run(self) -> List[str]:
        sel = self.plots_selection
        if sel == "all":
            return list(self.available_plots.keys())
        if isinstance(sel, list):
            return [p for p in sel if p in self.available_plots]
        if isinstance(sel, str):
            return [sel] if sel in self.available_plots else []
        return []

    def get_aro_scenarios_for_run(self, run_key: str) -> List[str]:
        run_conf = self.aro_runs.get(run_key, {})
        scenarios = run_conf.get("scenarios", [])
        if scenarios:
            return list(scenarios)

        summary_path = self.resolve_template(run_conf.get("summary_json"))
        # Fallback: aro_summary_iter1.json wenn aro_summary.json nicht existiert
        if summary_path and not Path(summary_path).is_file():
            fallback_path = summary_path.replace("aro_summary.json", "aro_summary_iter1.json")
            if Path(fallback_path).is_file():
                print(f"  [summary] aro_summary.json nicht gefunden, nutze: {Path(fallback_path).name}")
                summary_path = fallback_path
        if summary_path and Path(summary_path).is_file():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                from_json = summary.get("aro_final_scenarios", [])
                if from_json:
                    return list(from_json)
                all_costs = summary.get("aro_final_evaluation", {}).get("all_costs", {})
                if all_costs:
                    return list(all_costs.keys())
            except Exception as e:
                print(f"WARNUNG: Szenarien konnten nicht aus {summary_path} geladen werden: {e}")
        return []


# -----------------------------------------------------------------------------
# Backwards-compatible wrappers
# -----------------------------------------------------------------------------
class PlottingConfig:
    """Rückwärtskompatibel für ältere plot_* Skripte."""

    def __init__(self, master: Optional[MasterConfig] = None):
        self.master = master or MasterConfig()
        self.BASE_NETWORK_PATH  = self.master.pypsa_results_base
        self.BASE_SAVE_PATH     = self.master.plots_base
        self.SCENARIO_SELECTION = self.master.scenario_selection
        self.COUNTRIES_TO_PLOT  = self.master.get_countries(self.master.default_countries_to_plot)
        self.FONT_SIZES         = self.master.font_sizes
        self.CARRIER_COLORS     = self.master.carrier_colors
        self.DEFAULT_COLOR      = self.master.default_color
        self.SCENARIOS          = self.master.scenarios_registry
        self.PLOTS_TO_RUN       = self.master.plots_selection
        self.AVAILABLE_PLOTS    = self.master.available_plots
        self.DARK_SKY_START     = self.master.dark_sky_start
        self.DARK_SKY_END       = self.master.dark_sky_end
        self.PLANNING_YEAR      = self.master.planning_year
        # FIX: "both" und "all" sind Vergleichsmodi, keine Run-Keys.
        # Ordner wie plots_base/both/ sind sinnlos — stattdessen den
        # tatsächlichen Run-Ordner (ARO oder erster Registry-Eintrag) nutzen.
        _sel = self.SCENARIO_SELECTION
        if _sel in ("both", "all"):
            _sel = (self.master.aro_selected_run
                    or next(iter(self.master.scenarios_registry), _sel))
        self.PLOT_OUTPUT_PATH   = str(self.master.get_run_output_dir(_sel))
        if _PLOT_OUTPUT_OVERRIDE is not None:
            self.PLOT_OUTPUT_PATH = _PLOT_OUTPUT_OVERRIDE

    def get_networks(self):
        return self.master.get_networks()

    def get_countries(self):
        return self.master.get_countries(self.COUNTRIES_TO_PLOT)

    def get_plots_to_run(self):
        return self.master.get_plots_to_run()


class AROPlottingConfig:
    """Rückwärtskompatibel für ARO-Skripte."""

    def __init__(self, master: Optional[MasterConfig] = None):
        self.master = master or MasterConfig()
        self.BASE_RESULTS_PATH    = self.master.aro_results_base
        self._plot_output_path_override: Optional[str] = None
        self.ARO_RUNS             = self.master.aro_runs
        self.SELECTED_RUN         = self.master.aro_selected_run
        self.COUNTRIES_TO_ANALYZE = self.master.aro_countries_to_analyze
        self.FONT_SIZES           = self.master.font_sizes
        self.CARRIER_COLORS       = self.master.carrier_colors
        self.DEFAULT_COLOR        = self.master.default_color
        self.ARO_PLOTS            = self.master.aro_plot_toggles
        self.DARK_SKY_START       = self.master.dark_sky_start
        self.DARK_SKY_END         = self.master.dark_sky_end
        self.PLANNING_YEAR        = self.master.planning_year
        # Pfad zum Referenznetz (Basisrun-rcp45-2028) für Vergleichsplots
        self.REFERENCE_NETWORK_PATH: Optional[str] = (
            self.master.get_reference_network_path(self.SELECTED_RUN)
        )

    @property
    def PLOT_OUTPUT_PATH(self) -> str:
        if self._plot_output_path_override is not None:
            return self._plot_output_path_override
        return str(self.master.get_run_output_dir(self.SELECTED_RUN))

    @PLOT_OUTPUT_PATH.setter
    def PLOT_OUTPUT_PATH(self, value: str) -> None:
        self._plot_output_path_override = value

    def get_current_run_config(self) -> Dict[str, Any]:
        cfg = dict(self.ARO_RUNS[self.SELECTED_RUN])
        for k in ["summary_json", "robust_network", "robust_network_std",
                  "worst_case_dispatch", "worst_case_dispatch_std",
                  "reference_network"]:
            if k in cfg:
                cfg[k] = self.master.resolve_template(cfg[k])
        raw_dispatch_paths = dict(cfg.get("dispatch_paths", {}) or {})
        cfg["dispatch_paths"] = {
            sc: self.master.resolve_template(v) for sc, v in raw_dispatch_paths.items()
        }
        if not cfg.get("scenarios"):
            cfg["scenarios"] = self.master.get_aro_scenarios_for_run(self.SELECTED_RUN)
        cfg["resolved_dispatch_windows"] = {
            sc: self.master.get_dispatch_window(sc, self.SELECTED_RUN)
            for sc in cfg.get("scenarios", [])
        }
        cfg["resolved_run_default_window"] = self.master.get_dispatch_window(None, self.SELECTED_RUN)
        return cfg

    def get_plot_output_dir(self, plot_type: str = "general") -> Path:
        if self._plot_output_path_override is not None:
            base = Path(self._plot_output_path_override)
            return _mkdir(base / plot_type)
        return self.master.get_run_output_dir(self.SELECTED_RUN, sub=plot_type)


# =============================================================================
# Validation + helpers
# =============================================================================

def fill_leap_day(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Füllt fehlende Stunden (z.B. 29.02. bei Nicht-Schaltjahr-Daten) in einem
    stündlichen Zeitindex per forward-fill auf. Gibt den DataFrame unverändert
    zurück wenn keine Lücken vorhanden sind.
    """
    import pandas as _pd
    if df is None or df.empty:
        return df
    idx = _pd.to_datetime(df.index)
    full = _pd.date_range(start=idx[0], end=idx[-1], freq="h")
    if len(full) == len(idx):
        return df
    df2 = df.copy()
    df2.index = idx
    df2 = df2.reindex(full).ffill().bfill()
    return df2


def _now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _exists_file(p: Optional[str]) -> bool:
    return (p is not None) and Path(p).is_file()


def _exists_dir(p: Optional[str]) -> bool:
    return (p is not None) and Path(p).is_dir()


def validate_config(master: Optional[MasterConfig] = None, strict: bool = False) -> Dict[str, Any]:
    master = master or MasterConfig()
    report: Dict[str, Any] = {"ok": True, "errors": [], "warnings": [], "infos": []}

    def err(msg):  report["ok"] = False; report["errors"].append(msg)
    def warn(msg): report["warnings"].append(msg)
    def info(msg): report["infos"].append(msg)

    for k, required in [("pypsa_results_base", False), ("aro_results_base", False),
                        ("plots_base", True), ("aro_plots_base", False)]:
        v = master.paths.get(k)
        if v is None:
            if required:
                err(f"paths.{k} fehlt.")
            else:
                warn(f"paths.{k} ist nicht gesetzt.")
        elif not _exists_dir(v):
            if required:
                warn(f"Verzeichnis existiert noch nicht (wird angelegt): paths.{k}={v}")
            else:
                info(f"Verzeichnis existiert noch nicht: paths.{k}={v}")

    plotting = master.plotting
    if not isinstance(plotting.get("dpi"), int) or plotting["dpi"] <= 0:
        err(f"plotting.dpi ungültig: {plotting.get('dpi')}")
    if "font_sizes" not in plotting or not isinstance(plotting["font_sizes"], dict):
        err("plotting.font_sizes fehlt oder ist nicht dict.")

    colors = master.carrier_colors
    if not colors:
        err("colors.carriers ist leer.")

    if "ALL" not in master.default_countries_to_plot:
        warn("countries.default_countries_to_plot enthält kein 'ALL'.")

    # FIX #2 Validation: both_mapping auf Registry-Existenz prüfen
    both_mapping = master.raw["scenarios"].get("both_mapping", {})
    reg = master.scenarios_registry
    for mapping_label, registry_key in both_mapping.items():
        if registry_key not in reg:
            warn(
                f"scenarios.both_mapping['{mapping_label}'] = '{registry_key}' "
                f"existiert nicht im scenarios.registry. "
                f"get_networks('both') wird für diesen Key leer zurückgeben."
            )

    for k, v in reg.items():
        if isinstance(v, dict):
            rt = v.get("run_type", "")
            if rt not in ("normal", "aro", ""):
                warn(f"scenarios.registry['{k}'].run_type='{rt}' unbekannt.")
        elif not isinstance(v, list):
            warn(f"scenarios.registry['{k}'] sollte dict oder list sein.")

    sel = master.scenario_selection
    if sel not in ("both", "all") and sel not in reg:
        warn(f"scenarios.selection='{sel}' ist nicht im scenarios.registry vorhanden.")

    networks = master.get_networks()
    def check_paths(paths_list, ctx):
        missing = sum(1 for p in paths_list if not Path(p).is_file())
        if missing:
            warn(f"{ctx}: {missing}/{len(paths_list)} Netzwerkdateien fehlen.")

    if isinstance(networks, list):
        check_paths(networks, f"scenario '{sel}'")
    elif isinstance(networks, dict):
        for k, v in networks.items():
            check_paths(v, f"scenario '{k}'")

    aro_sel = master.aro_selected_run
    aro_runs = master.aro_runs
    if aro_sel not in aro_runs:
        warn(f"aro.selected_run='{aro_sel}' ist nicht in aro.runs.")
    else:
        cfg = dict(aro_runs[aro_sel])
        for key in ["summary_json", "robust_network", "reference_network",
                    "worst_case_dispatch", "worst_case_dispatch_std"]:
            if key in cfg:
                resolved = master.resolve_template(cfg[key])
                if resolved is None and key not in ("worst_case_dispatch", "worst_case_dispatch_std"):
                    warn(f"ARO '{aro_sel}': '{key}' ist None.")
                elif resolved and key in ("summary_json", "robust_network", "reference_network") \
                        and not Path(resolved).is_file():
                    warn(f"ARO '{aro_sel}': Datei fehlt: {key}={resolved}")

        scenarios = master.get_aro_scenarios_for_run(aro_sel)
        if not scenarios:
            warn(f"ARO '{aro_sel}': Keine Szenarien gefunden (wird aus aro_summary.json geladen).")
        else:
            info(f"ARO '{aro_sel}': {len(scenarios)} Szenarien: {scenarios[:5]}{'...' if len(scenarios)>5 else ''}")

        # Referenznetz validieren
        ref_path = master.get_reference_network_path(aro_sel)
        if ref_path and not Path(ref_path).is_file():
            warn(f"ARO '{aro_sel}': Referenznetz fehlt: {ref_path}")
        elif ref_path:
            info(f"ARO '{aro_sel}': Referenznetz OK: {ref_path}")

    try:
        import pandas as pd
        pd.Timestamp(master.dark_sky_start)
        pd.Timestamp(master.dark_sky_end)
        info(f"dark_sky_period: {master.dark_sky_start} -- {master.dark_sky_end}")
    except Exception as e:
        err(f"dark_sky_period ungültig: {e}")

    if strict and not report["ok"]:
        raise ValueError("MasterConfig validation failed:\n" + "\n".join(report["errors"]))

    info("Validation finished.")
    return report


def make_run_output_dir(master: Optional[MasterConfig] = None, run_name: str = "run") -> Path:
    """
    Rückwärtskompatible Funktion — delegiert an get_run_output_dir().
    """
    master = master or MasterConfig()
    return master.get_run_output_dir(run_name)


def write_report(out_dir: Path, report: Dict[str, Any], name: str = "report") -> None:
    out_dir = _mkdir(out_dir)
    (out_dir / f"{name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = [f"# Analysis report: {name}\n", f"- ok: **{report.get('ok')}**\n"]
    for section in ("errors", "warnings", "infos"):
        items = report.get(section, [])
        if items:
            md.append(f"## {section.capitalize()}\n")
            md.extend(f"- {i}\n" for i in items)
    (out_dir / f"{name}.md").write_text("".join(md), encoding="utf-8")