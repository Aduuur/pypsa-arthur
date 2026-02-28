# master_config.py
# =============================================================================
# Zentrale Master-Konfigurationsdatei für PyPSA-Eur Auswertung:
# - gemeinsame Settings (Farben, Länder, Plotstyle, Output)
# - Myopic / deterministische Netzwerkserien
# - ARO-Runs (Summary JSON, robust network, worst-case dispatch, Szenarien)
#
# Ziel:
#   1) Alles an einer Stelle editierbar
#   2) Möglichst generisch und vollständig
#   3) Backwards-Kompatibilität für vorhandene Skripte
# =============================================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import datetime
from typing import Tuple


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Simple environment override helper."""
    return os.environ.get(key, default)


def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _mkdir(p: Union[str, Path]) -> Path:
    p = _as_path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------------------------------------------------------
# RUN NAME
# -----------------------------------------------------------------------------
# Define the run name once, and reuse it "wildcard-style" throughout the config.
# You can override it via ENV: RUN_NAME="compare-robust-vol2"
RUN_NAME: str = _env("RUN_NAME", "compare-robust-vol2")  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Master configuration block
# -----------------------------------------------------------------------------
MASTER_CONFIG: Dict[str, Any] = {
    # =======================================================================
    # GLOBAL PATHS (overridable via ENV)
    # =======================================================================
    "paths": {
        # base directory for PyPSA-Eur results (deterministic/myopic)
        "pypsa_results_base": _env("PYPSA_RESULTS_BASE", "/home/endata/PycharmProjects/pypsa-ee/results"),

        # base directory for ARO results (summary json + robust/worst-case networks)
        "aro_results_base": _env("ARO_RESULTS_BASE", "/home/endata/PycharmProjects/pypsa-ee/results"),

        # global plot output base
        "plots_base": _env("PLOTS_BASE", "/mnt/endata/MA_Arthur/PyPSA-results/plots"),

        # optional: separate directory for ARO analysis plots
        "aro_plots_base": _env("ARO_PLOTS_BASE", "/mnt/endata/MA_Arthur/aro_analysis/plots"),
    },

    # =======================================================================
    # COUNTRY SETTINGS
    # =======================================================================
    "countries": {
        # preferred order for plotting (ALL typically first)
        "default_countries_to_plot": [
            "ALL", "DE", "FR", "ES", "CH", "DK", "SE", "NO", "IT", "GB",
            "NL", "PL", "BE", "FI", "AT", "PT", "CZ", "LT", "LV", "EE"
        ],

        # display names (for titles/legends/reports)
        "names": {
            "DE": "Deutschland",
            "FR": "Frankreich",
            "ES": "Spanien",
            "IT": "Italien",
            "NL": "Niederlande",
            "BE": "Belgien",
            "AT": "Österreich",
            "CH": "Schweiz",
            "PL": "Polen",
            "DK": "Dänemark",
            "SE": "Schweden",
            "NO": "Norwegen",
            "FI": "Finnland",
            "GB": "Großbritannien",
            "IE": "Irland",
            "PT": "Portugal",
            "CZ": "Tschechien",
            "LT": "Litauen",
            "LV": "Lettland",
            "EE": "Estland",
            "ALL": "Gesamtnetz",
        },
    },

    # =======================================================================
    # PLOT STYLE / OUTPUT SETTINGS
    # =======================================================================
    "plotting": {
        "matplotlib_style": _env("PLOT_STYLE", "seaborn-v0_8-whitegrid"),
        "dpi": int(_env("PLOT_DPI", "300")),
        "bbox_inches": _env("PLOT_BBOX", "tight"),
        "save_format": _env("PLOT_FORMAT", "png"),

        "figsizes": {
            "default": (12, 8),
            "large": (16, 12),
            "wide": (14, 6),
        },

        "font_sizes": {
            "title": 20,
            "label": 15,
            "tick": 13,
            "bar_label": 10,
            "legend": 11,
        },

        # global default color if carrier unknown
        "default_color": "#a9a9a9",
    },

    # =======================================================================
    # TECHNOLOGY / CARRIER COLOR PALETTE
    # =======================================================================
    # One unified palette. Add aliases + grouped names here to avoid duplicates
    # in plotting logic.
    "colors": {
        "carriers": {
            # --- Wind ---
            "onwind": "#235ebc",
            "onshore wind": "#235ebc",
            "offwind": "#004E8A",
            "offwind-ac": "#6895dd",
            "offwind-dc": "#74c6f2",
            "offshore wind": "#6895dd",
            "offwind-float": "#15a0bf",

            # --- Solar ---
            "solar": "#f9d002",
            "solar rooftop": "#ffea80",
            "solar-hsat": "#FFF080",

            # --- Hydro ---
            "hydro": "#298c81",
            "ror": "#3dbfb0",
            "run of river": "#3dbfb0",
            "PHS": "#51dbcc",

            # --- Fossil / conventional ---
            "gas": "#e05b09",
            "OCGT": "#e0986c",
            "CCGT": "#a85522",
            "coal": "#545454",
            "lignite": "#826837",
            "oil": "#c9c9c9",
            "oil primary": "#7a7a7a",
            "nuclear": "#ff8c00",

            # --- Biomass (detailed) ---
            "biomass": "#baa741",
            "Biomasse": "#baa741",
            "solid biomass": "#baa741",
            "municipal solid waste": "#91ba41",
            "solid biomass import": "#d5ca8d",
            "solid biomass transport": "#baa741",
            "solid biomass for industry": "#7a6d26",
            "solid biomass for industry CC": "#47411c",
            "solid biomass for industry co2 from atmosphere": "#736412",
            "solid biomass for industry co2 to stored": "#47411c",
            "urban central solid biomass CHP": "#9d9042",
            "urban central solid biomass CHP CC": "#6c5d28",
            "biomass boiler": "#8A9A5B",
            "residential rural biomass boiler": "#a1a066",
            "residential urban decentral biomass boiler": "#b0b87b",
            "services rural biomass boiler": "#c6cf98",
            "services urban decentral biomass boiler": "#dde5b5",
            "biogas": "#e3d37d",
            "waste": "#e3d37d",
            "unsustainable solid biomass": "#998622",
            "unsustainable bioliquids": "#32CD32",
            "electrobiofuels": "#ff0000",
            "BioSNG": "#123456",
            "BioSNG CC": "#45233b",
            "solid biomass to hydrogen": "#654321",
            "biomass to liquid": "#32CD32",

            # --- Hydrogen / storage ---
            "H2": "#bf13a0",
            "hydrogen": "#bf13a0",
            "H2 storage": "#bf13a0",
            "H2 Electrolysis": "#ff29d9",
            "H2 Fuel Cell": "#c251ae",
            "H2 turbine": "#991f83",

            "battery": "#ace37f",
            "home battery": "#80c944",
            "battery charger": "#76c7a3",
            "battery discharger": "#2a9d8f",
            "home battery discharger": "#8ecae6",
            "V2G": "#e5ffa8",

            # --- Sector coupling / demand / exports ---
            "load": "#1f77b4",
            "methanation": "#a349a4",
            "Haber-Bosch": "#b565b0",
            "heat pump": "#ff9966",
            "resistive heater": "#ff7f50",
            "EV charger": "#9999ff",
            "export": "#808080",

            # --- Other ---
            "geothermal": "#ba91b1",
            "other": "#000000",
        },
    },


    # =======================================================================
    # SCENARIO REGISTRY (deterministic/myopic sequences)
    # =======================================================================
    # You can reference these keys from scripts, and choose selection below.
    "scenarios": {
        # which scenario to load by default in "PlottingConfig"
        # -> set to RUN_NAME by default
        "selection": _env("SCENARIO_SELECTION", RUN_NAME),  # or "both"

        # scenario registry: name -> list of network paths (or dict for multi)
        "registry": {
            # -------------------------------------------------------------------
            # RUN_NAME
            # -------------------------------------------------------------------
            RUN_NAME: [
                f"/home/endata/PycharmProjects/pypsa-ee/results/{RUN_NAME}/networks/base_s_24___2050.nc",
            ],
        },

        # convenience mode: if selection == "both", return this mapping
        "both_mapping": {
            "average": "new_avg",
            "dunkelflaute": "dunkelflaute_neu_2",
        },
    },


    # =======================================================================
    # AVAILABLE PLOTS (generic registry, optional)
    # =======================================================================
    "plots": {
        "selection": _env("PLOTS_TO_RUN", "all"),  # "all" or comma-separated list via env
        "available": {
            "installed_capacity": "plot_installed_capacity",
            "energy_generated": "plot_energy_generated",
            "storage": "plot_storage",
            "price": "plot_price",
            "losses": "plot_losses",
            "demand_vs_generation": "plot_demand_vs_generation",
            "price_setting_unit": "plot_price_setting_unit",
            "gas_h2_usage": "plot_gas_h2_usage",
            "capacity_expansion": "plot_capacity_expansion",
            "demand_profile": "plot_demand_profile",
            "generation_profile": "plot_generation_profile",
        },
    },

    # =======================================================================
    # ARO RUN REGISTRY
    # =======================================================================
    "aro": {
        # which ARO run is "active" (used by AROPlottingConfig wrapper)
        "selected_run": _env("ARO_SELECTED_RUN", RUN_NAME),

        # countries to deep-dive for ARO worst-case / robust analyses
        "countries_to_analyze": ["ALL", "DE", "FR", "ES", "CH"],

        # which ARO plots to generate
        "plot_toggles": {
            "convergence": True,
            "scenario_comparison": True,
            "worst_case_analysis": False,
            "robustness_metrics": True,
            "capacity_comparison": True,
        },

        # ARO runs (you can add unlimited runs)
        "runs": {
            RUN_NAME: {
                "name": RUN_NAME,
                "summary_json": f"{{aro_results_base}}/{RUN_NAME}/results/aro_summary.json",
                "robust_network": f"{{aro_results_base}}/{RUN_NAME}/networks/aro_robust.nc",
                "robust_network_std": f"{{aro_results_base}}/{RUN_NAME}/networks/aro_robust__std.nc",
                "worst_case_dispatch": None,
                "worst_case_dispatch_std": None,
                "scenarios": [],
            },
        },
    },
}


# -----------------------------------------------------------------------------
# Dataclasses wrapping the dict (safer access + easy extension)
# -----------------------------------------------------------------------------
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
        return self.paths["aro_plots_base"]

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
        # allow env override: "installed_capacity,storage"
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
    def resolve_template(self, s: Optional[str]) -> Optional[str]:
        """Resolve template placeholders like {aro_results_base} in run configs."""
        if s is None:
            return None
        return s.format(**self.paths)

    def get_networks(self) -> Union[List[str], Dict[str, List[str]], None]:
        """
        For deterministic/myopic runs:
        - if selection == "both": returns dict with keys from both_mapping
        - else: returns list of network paths
        """
        sel = self.scenario_selection
        reg = self.scenarios_registry

        if sel == "both":
            mapping = self.raw["scenarios"]["both_mapping"]
            out: Dict[str, List[str]] = {}
            for out_key, scenario_key in mapping.items():
                out[out_key] = list(reg.get(scenario_key, []))
            return out

        if sel in reg:
            return list(reg[sel])

        return None

    def get_countries(self, override: Optional[List[str]] = None) -> List[str]:
        """
        Countries to plot/analyze; ensures 'ALL' is first if present.
        """
        countries = override if override is not None else self.default_countries_to_plot
        if "ALL" in countries:
            return ["ALL"] + [c for c in countries if c != "ALL"]
        return countries

    def get_plots_to_run(self) -> List[str]:
        """
        Returns list of plot keys to run.
        """
        sel = self.plots_selection
        if sel == "all":
            return list(self.available_plots.keys())
        if isinstance(sel, list):
            return [p for p in sel if p in self.available_plots]
        if isinstance(sel, str):
            return [sel] if sel in self.available_plots else []
        return []


# -----------------------------------------------------------------------------
# Backwards-compatible wrappers
# -----------------------------------------------------------------------------
class PlottingConfig:
    """
    Backwards-compatible wrapper for older scripts.
    Use this in the existing plot_* scripts that currently do:
        from config_final import PlottingConfig
    Change to:
        from master_config import PlottingConfig
    """
    def __init__(self, master: Optional[MasterConfig] = None):
        self.master = master or MasterConfig()

        self.BASE_NETWORK_PATH = self.master.pypsa_results_base
        self.BASE_SAVE_PATH = self.master.plots_base
        self.PLOT_OUTPUT_PATH = str(_mkdir(Path(self.BASE_SAVE_PATH) / "plots_combined"))

        # Controls (keep old attribute names)
        self.SCENARIO_SELECTION = self.master.scenario_selection
        self.COUNTRIES_TO_PLOT = self.master.get_countries(self.master.default_countries_to_plot)
        self.FONT_SIZES = self.master.font_sizes
        self.CARRIER_COLORS = self.master.carrier_colors
        self.DEFAULT_COLOR = self.master.default_color

        self.SCENARIOS = self.master.scenarios_registry

        self.PLOTS_TO_RUN = self.master.plots_selection
        self.AVAILABLE_PLOTS = self.master.available_plots

    def get_networks(self):
        return self.master.get_networks()

    def get_countries(self):
        return self.master.get_countries(self.COUNTRIES_TO_PLOT)

    def get_plots_to_run(self):
        return self.master.get_plots_to_run()


class AROPlottingConfig:
    """
    Backwards-compatible wrapper for ARO scripts.
    Replace imports:
        from config_aro import AROPlottingConfig
    with:
        from master_config import AROPlottingConfig
    """
    def __init__(self, master: Optional[MasterConfig] = None):
        self.master = master or MasterConfig()

        self.BASE_RESULTS_PATH = self.master.aro_results_base
        self.PLOT_OUTPUT_PATH = self.master.aro_plots_base

        self.ARO_RUNS = self.master.aro_runs
        self.SELECTED_RUN = self.master.aro_selected_run

        self.COUNTRIES_TO_ANALYZE = self.master.aro_countries_to_analyze
        self.FONT_SIZES = self.master.font_sizes

        # unify palette
        self.CARRIER_COLORS = self.master.carrier_colors
        self.DEFAULT_COLOR = self.master.default_color

        self.ARO_PLOTS = self.master.aro_plot_toggles

    def get_current_run_config(self) -> Dict[str, Any]:
        cfg = dict(self.ARO_RUNS[self.SELECTED_RUN])

        # resolve templates in paths
        for k in ["summary_json", "robust_network", "worst_case_dispatch", "worst_case_dispatch_std"]:
            if k in cfg:
                cfg[k] = self.master.resolve_template(cfg[k])

        return cfg

    def get_plot_output_dir(self, plot_type: str = "general") -> Path:
        output_dir = Path(self.PLOT_OUTPUT_PATH) / self.SELECTED_RUN / plot_type
        return _mkdir(output_dir)



# =============================================================================
# Validation + helpers
# =============================================================================


def _now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _exists_file(p: Optional[str]) -> bool:
    return (p is not None) and Path(p).is_file()


def _exists_dir(p: Optional[str]) -> bool:
    return (p is not None) and Path(p).is_dir()


def validate_config(master: Optional[MasterConfig] = None, strict: bool = False) -> Dict[str, Any]:
    """
    Validiert die MasterConfig.
    - strict=False: nur Warnungen, keine Exception
    - strict=True: wirft ValueError bei kritischen Problemen
    Returns: dict(report) mit errors/warnings/infos
    """
    master = master or MasterConfig()
    report: Dict[str, Any] = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "infos": [],
    }

    def err(msg: str):
        report["ok"] = False
        report["errors"].append(msg)

    def warn(msg: str):
        report["warnings"].append(msg)

    def info(msg: str):
        report["infos"].append(msg)

    # --- Paths ---
    paths = master.paths
    for k in ["pypsa_results_base", "aro_results_base", "plots_base", "aro_plots_base"]:
        v = paths.get(k)
        if v is None:
            err(f"paths.{k} fehlt.")
        else:
            # results_base kann auch existieren ohne dir; check dir
            if k.endswith("_base") and not _exists_dir(v):
                warn(f"Verzeichnis existiert nicht: paths.{k}={v}")
            if k.endswith("_plots_base") and not _exists_dir(v):
                warn(f"Plot-Ausgabeordner existiert nicht (wird angelegt): paths.{k}={v}")

    # --- Plot settings sanity ---
    plotting = master.plotting
    if "dpi" in plotting and (not isinstance(plotting["dpi"], int) or plotting["dpi"] <= 0):
        err(f"plotting.dpi ungültig: {plotting.get('dpi')}")
    if "save_format" in plotting and not isinstance(plotting["save_format"], str):
        err("plotting.save_format muss str sein.")
    if "font_sizes" not in plotting or not isinstance(plotting["font_sizes"], dict):
        err("plotting.font_sizes fehlt oder ist nicht dict.")

    # --- Colors ---
    colors = master.carrier_colors
    if not colors:
        err("colors.carriers ist leer.")
    # suspicious duplicates by case/whitespace
    normalized = {}
    for k, v in colors.items():
        nk = " ".join(k.strip().split()).lower()
        if nk in normalized and normalized[nk] != k:
            warn(f"Carrier-Name-Dublette (case/whitespace): '{normalized[nk]}' vs '{k}'")
        normalized[nk] = k
        if not isinstance(v, str) or not v.startswith("#") or len(v) not in (4, 7):
            warn(f"Farbwert sieht nicht nach Hex aus: {k}={v}")

    # --- Countries ---
    c_list = master.default_countries_to_plot
    if "ALL" not in c_list:
        warn("countries.default_countries_to_plot enthält kein 'ALL' (optional, aber meist sinnvoll).")

    # --- Scenario registry ---
    sel = master.scenario_selection
    reg = master.scenarios_registry
    if sel != "both" and sel not in reg:
        warn(f"scenarios.selection='{sel}' ist nicht im scenarios.registry vorhanden.")
    if sel == "both":
        mapping = master.raw["scenarios"].get("both_mapping", {})
        for out_key, scenario_key in mapping.items():
            if scenario_key not in reg:
                warn(f"both_mapping: '{scenario_key}' fehlt im scenarios.registry (für '{out_key}').")

    # --- Scenario paths existence (soft) ---
    networks = master.get_networks()
    def check_paths(paths_list: List[str], ctx: str):
        missing = 0
        for p in paths_list:
            if not Path(p).is_file():
                missing += 1
        if missing > 0:
            warn(f"{ctx}: {missing}/{len(paths_list)} Netzwerkdateien fehlen am Pfad (oder nicht gemountet).")

    if isinstance(networks, list):
        check_paths(networks, f"scenario '{sel}'")
    elif isinstance(networks, dict):
        for k, v in networks.items():
            check_paths(v, f"scenario both.{k}")

    # --- ARO runs ---
    aro_sel = master.aro_selected_run
    aro_runs = master.aro_runs
    if aro_sel not in aro_runs:
        warn(f"aro.selected_run='{aro_sel}' ist nicht in aro.runs vorhanden.")
    else:
        cfg = dict(aro_runs[aro_sel])
        # resolve templates and check
        for key in ["summary_json", "robust_network", "worst_case_dispatch", "worst_case_dispatch_std"]:
            if key in cfg:
                resolved = master.resolve_template(cfg[key])
                if resolved is None and key != "worst_case_dispatch_std":
                    warn(f"ARO '{aro_sel}': '{key}' ist None.")
                else:
                    # hard requirement: summary_json + robust_network should exist
                    if key in ("summary_json", "robust_network") and resolved and not Path(resolved).is_file():
                        warn(f"ARO '{aro_sel}': Datei fehlt: {key}={resolved}")
                    if key == "worst_case_dispatch" and resolved and not Path(resolved).is_file():
                        warn(f"ARO '{aro_sel}': worst_case_dispatch fehlt: {resolved}")

        scenarios = cfg.get("scenarios", [])
        # scenarios optional; can be read from aro_summary.json (scenario_names)
        if scenarios is not None and (not isinstance(scenarios, list)):
            warn(f"ARO '{aro_sel}': 'scenarios' ist nicht list.")

    # --- strict mode ---
    if strict and not report["ok"]:
        raise ValueError("MasterConfig validation failed:\n" + "\n".join(report["errors"]))

    info("Validation finished.")
    return report


def make_run_output_dir(master: Optional[MasterConfig] = None, run_name: str = "run") -> Path:
    """
    Erstellt einen eindeutigen Output-Ordner: <plots_base>/analysis_runs/<run_name>_<timestamp>
    """
    master = master or MasterConfig()
    base = Path(master.plots_base) / "analysis_runs"
    out = base / f"{run_name}_{_now_stamp()}"
    return _mkdir(out)


def write_report(out_dir: Path, report: Dict[str, Any], name: str = "report") -> None:
    out_dir = _mkdir(out_dir)
    # json
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # md (quick)
    md = []
    md.append(f"# Analysis report: {name}\n")
    md.append(f"- ok: **{report.get('ok')}**\n")
    if report.get("errors"):
        md.append("## Errors\n")
        for e in report["errors"]:
            md.append(f"- {e}\n")
    if report.get("warnings"):
        md.append("## Warnings\n")
        for w in report["warnings"]:
            md.append(f"- {w}\n")
    if report.get("infos"):
        md.append("## Infos\n")
        for i in report["infos"]:
            md.append(f"- {i}\n")
    (out_dir / f"{name}.md").write_text("".join(md), encoding="utf-8")