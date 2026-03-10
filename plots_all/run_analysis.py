#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified runner für ARO- und normale (myopische/deterministische) Läufe.

Verwendet master_config.py als Single Source of Truth.

Usage:
  python plots_all/run_analysis.py --mode auto
      -> erkennt run_type aus master_config automatisch

  python plots_all/run_analysis.py --mode aro --aro-run compare-robust-vol2
  python plots_all/run_analysis.py --mode normal --scenario new_avg --countries ALL,DE
  python plots_all/run_analysis.py --mode all-scenarios --aro-run compare-robust-vol2
      -> alle ARO-Dispatch-Szenarien einzeln auswerten
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from master_config import (
    MasterConfig,
    AROPlottingConfig,
    PlottingConfig,
    validate_config,
    make_run_output_dir,
    write_report,
)

from aro_analysis import AROAnalyzer
from country_analysis import CountryAnalyzer


# Am Anfang der Datei ergänzen:
import importlib
import importlib.util
import sys

# Alle standalone plot_*.py Skripte die per Dispatcher ausgeführt werden sollen.
# Key = interner Name, Value = Dateiname (relativ zu plots_all/).
STANDALONE_SCRIPTS: Dict[str, str] = {
    "dispatch_timeline":       "plot_generation_timeline_with_load_line_v2.py",
    "balance_timeline":        "plot_balance_timeline_simple.py",
    "installed_cap_vgl":       "plot_installed_cap_new_vgl.py",
    "energy_gen_no_storage":   "plot_energy_gen_single_no_storage.py",
    "generation_timeline_res": "plot_generation_timeline_res_last.py",
    "generation_timeline_gen": "plot_generation_timeline_only_generation.py",
    "storage_v2":              "plot_storage_v2.py",
    "storage_kombi_es_de":     "plot_storage_kombi_es_de_detail.py",
    "storage_kombi_fr_de":     "plot_storage_kombi_fr_de.py",
    "marginal_prices":         "plot_marginal_prices.py",
    "marginal_prices_check":   "plot_marginal_prices_check.py",
    "dec_price_normal":        "plot_dec_electircity_price_normal.py",
    "dec_price_new_jan":       "plot_dec_electricity_price_new_jan.py",
    "dec_price_split":         "plot_dec_electricity_price_split.py",
    "dec_delta_dispatch":      "plot_dec_new_delta_dispatch_with_trade.py",
    "delta_prices_map":        "plot_delta_prices_map.py",
    "dispatch_gas_h2":         "plot_dispatch_gas_h2.py",
    "consumption_timeline":    "plot_consumption_timeline.py",
    "map_leitung_export":      "plot_map_leitung_export.py",
    "map_nutzbare_uebertrag":  "plot_map_nutzbare_übertragung.py",
    "co2_emissionen":          "plot_jaehrliche_co2_emissionen.py",
    "co2_emissionen_analyse":  "plot_jaehrliche_co2_emissionen_analyse.py",
    "check_waermepumpen":      "plot_check_wärmepumpen_df.py",
}

# Welche Skripte bei ARO-Runs (auf Worst-Case-Netz) ausgeführt werden:
ARO_APPLICABLE_SCRIPTS: set = {
    "dispatch_timeline",
    "balance_timeline",
    "energy_gen_no_storage",
    "generation_timeline_res",
    "generation_timeline_gen",
    "storage_v2",
    "marginal_prices",
    "dec_price_normal",
}


def _load_and_run_script(
    script_key: str,
    scripts_dir: Path,
    override_scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Lädt ein standalone plot_*.py als Modul und ruft dessen main() auf.
    Setzt optional SCENARIO_SELECTION in master_config für den Aufruf.
    """
    result: Dict[str, Any] = {"script": script_key, "ok": False, "error": None}

    filename = STANDALONE_SCRIPTS.get(script_key)
    if filename is None:
        result["error"] = f"Unbekanntes Script: {script_key}"
        return result

    script_path = scripts_dir / filename
    if not script_path.is_file():
        result["error"] = f"Datei nicht gefunden: {script_path}"
        return result

    # Szenario temporär überschreiben wenn nötig
    if override_scenario:
        from master_config import MASTER_CONFIG
        _orig = MASTER_CONFIG["scenarios"]["selection"]
        MASTER_CONFIG["scenarios"]["selection"] = override_scenario

    try:
        # Dynamisch als Modul laden (isoliert)
        spec = importlib.util.spec_from_file_location(f"_standalone_{script_key}", script_path)
        mod = importlib.util.module_from_spec(spec)
        # Plots_all im sys.path damit config_final etc. gefunden werden
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        spec.loader.exec_module(mod)
        if hasattr(mod, "main"):
            mod.main()
            result["ok"] = True
        else:
            result["error"] = "Kein main() gefunden"
    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    finally:
        if override_scenario:
            MASTER_CONFIG["scenarios"]["selection"] = _orig

    return result


def run_standalone_scripts(
    scripts_dir: Path,
    script_keys: Optional[List[str]] = None,
    aro_only: bool = False,
    override_scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Führt alle (oder ausgewählte) standalone plot_*.py Skripte aus.

    Args:
        scripts_dir:       Pfad zum plots_all/ Verzeichnis
        script_keys:       Auswahl (None = alle)
        aro_only:          Nur ARO-kompatible Skripte ausführen
        override_scenario: Szenario in master_config temporär überschreiben
    """
    report: Dict[str, Any] = {"ok": True, "scripts": {}}

    keys = script_keys or list(STANDALONE_SCRIPTS.keys())
    if aro_only:
        keys = [k for k in keys if k in ARO_APPLICABLE_SCRIPTS]

    for key in keys:
        print(f"  Starte standalone: {key} ({STANDALONE_SCRIPTS.get(key, '?')})")
        res = _load_and_run_script(key, scripts_dir, override_scenario=override_scenario)
        report["scripts"][key] = res
        if res["ok"]:
            print(f"  ✓ {key}")
        else:
            print(f"  ✗ {key}: {res['error']}")
            report["ok"] = False

    return report


def _parse_csv(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _year_from_path(path: str) -> Optional[int]:
    m = re.search(r"___(\d{4})\.nc$", path) or re.search(r"_(\d{4})\.nc$", path)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------
# ARO runner
# ------------------------------------------------------------------

def run_aro(
    master: MasterConfig,
    out_dir: Path,
    aro_run: Optional[str] = None,
    countries: Optional[List[str]] = None,
    all_scenarios: bool = False,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"mode": "aro", "ok": True, "steps": [], "warnings": [], "errors": []}

    aro_cfg = AROPlottingConfig(master=master)
    if aro_run is not None:
        aro_cfg.SELECTED_RUN = aro_run

    run_key  = aro_cfg.SELECTED_RUN
    run_conf = aro_cfg.get_current_run_config()

    aro_out = out_dir / "aro" / run_key
    aro_out.mkdir(parents=True, exist_ok=True)
    aro_cfg.PLOT_OUTPUT_PATH = str(aro_out)

    report["steps"].append({"run_key": run_key, "scenarios": run_conf.get("scenarios", [])})

    try:
        analyzer = AROAnalyzer(config=aro_cfg, auto_find_dispatch=True)
        report["steps"].append({"AROAnalyzer": "initialized",
                                 "scenario_networks_loaded": len(analyzer.scenario_networks)})
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"AROAnalyzer init failed: {e}")
        return report

    # Summary
    try:
        analyzer.summary_report()
        report["steps"].append({"summary_report": "ok"})
    except Exception as e:
        report["warnings"].append(f"summary_report failed: {e}")

    toggles = aro_cfg.ARO_PLOTS

    # Standard ARO-Plots
    for toggle_key, method_name in [
        ("convergence",         "plot_aro_convergence"),
        ("scenario_comparison", "plot_scenario_cost_comparison"),
        ("capacity_comparison", "plot_robust_capacity_breakdown"),
    ]:
        if toggles.get(toggle_key, True):
            try:
                getattr(analyzer, method_name)()
                report["steps"].append({method_name: "ok"})
            except Exception as e:
                report["warnings"].append(f"{method_name} failed: {e}")

    # Neuer Szenario-Kapazitätsvergleich
    try:
        analyzer.plot_scenario_capacity_comparison()
        report["steps"].append({"plot_scenario_capacity_comparison": "ok"})
    except Exception as e:
        report["warnings"].append(f"plot_scenario_capacity_comparison failed: {e}")

    # Worst-Case-Analyse (pro Land)
    if toggles.get("worst_case_analysis", False):
        c_list = countries or aro_cfg.COUNTRIES_TO_ANALYZE
        for c in c_list:
            try:
                analyzer.analyze_worst_case_dispatch(country=c)
                report["steps"].append({"analyze_worst_case_dispatch": c})
            except Exception as e:
                report["warnings"].append(f"worst_case_analysis failed for {c}: {e}")

    # Alle Szenarien einzeln auswerten
    if all_scenarios:
        c_list = countries or aro_cfg.COUNTRIES_TO_ANALYZE
        try:
            analyzer.analyze_all_scenario_dispatches(countries=c_list, save=True)
            report["steps"].append({"analyze_all_scenario_dispatches": "ok"})
        except Exception as e:
            report["warnings"].append(f"analyze_all_scenario_dispatches failed: {e}")

    # Am Ende von run_aro(), nach den ARO-spezifischen Plots:

    # Standalone Skripte auf Worst-Case-Netz (oder normalem Szenario) ausführen
    scripts_dir = Path(__file__).parent
    print("\n=== Standalone plot_*.py Skripte (ARO-kompatibel) ===")
    standalone_report = run_standalone_scripts(
        scripts_dir=scripts_dir,
        aro_only=True,  # nur Dispatch/Timeline Skripte
        override_scenario=aro_cfg.SELECTED_RUN,
    )
    report["standalone"] = standalone_report

    return report


# ------------------------------------------------------------------
# Normal/myopic runner
# ------------------------------------------------------------------

def run_normal(
    master: MasterConfig,
    out_dir: Path,
    scenario: Optional[str] = None,
    countries: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"mode": "normal", "ok": True, "steps": [], "warnings": [], "errors": []}

    if scenario is not None:
        master.raw["scenarios"]["selection"] = scenario

    plot_cfg = PlottingConfig(master=master)
    networks = plot_cfg.get_networks()

    if networks is None:
        report["ok"] = False
        report["errors"].append(f"Keine Netzwerke für Szenario '{plot_cfg.SCENARIO_SELECTION}'.")
        return report

    c_list = countries or plot_cfg.get_countries()
    normal_out = out_dir / "normal" / plot_cfg.SCENARIO_SELECTION
    normal_out.mkdir(parents=True, exist_ok=True)

    def analyze_one(net_path: str, base_out: Path):
        y = _year_from_path(net_path)
        if years is not None and y is not None and y not in years:
            return
        if not Path(net_path).is_file():
            report["warnings"].append(f"Network file missing: {net_path}")
            return
        tag = str(y) if y is not None else Path(net_path).stem
        per_net_out = base_out / tag
        per_net_out.mkdir(parents=True, exist_ok=True)

        for c in c_list:
            try:
                analyzer = CountryAnalyzer(
                    network_path=net_path,
                    country=c,
                    output_dir=str(per_net_out / c),
                    carrier_colors=plot_cfg.CARRIER_COLORS,
                    default_color=plot_cfg.DEFAULT_COLOR,
                )
                analyzer.summary_report()
                analyzer.generate_all_plots(save=True)
                report["steps"].append({"country_analysis": {"network": net_path, "country": c}})
            except Exception as e:
                report["warnings"].append(f"CountryAnalyzer failed for {c} in {net_path}: {e}")

    if isinstance(networks, list):
        for p in networks:
            analyze_one(p, normal_out)
    elif isinstance(networks, dict):
        for branch, paths in networks.items():
            branch_out = normal_out / branch
            branch_out.mkdir(parents=True, exist_ok=True)
            for p in paths:
                analyze_one(p, branch_out)
    # Am Ende von run_normal(), nach CountryAnalyzer:

    scripts_dir = Path(__file__).parent
    print("\n=== Standalone plot_*.py Skripte ===")
    standalone_report = run_standalone_scripts(
        scripts_dir=scripts_dir,
        aro_only=False,  # alle Skripte
        override_scenario=plot_cfg.SCENARIO_SELECTION,
    )
    report["standalone"] = standalone_report

    return report


# ------------------------------------------------------------------
# Auto-detection
# ------------------------------------------------------------------

def run_auto(
    master: MasterConfig,
    out_dir: Path,
    aro_run: Optional[str] = None,
    scenario: Optional[str] = None,
    countries: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    all_scenarios: bool = False,
) -> Dict[str, Any]:
    """
    Erkennt run_type aus master_config und delegiert entsprechend.
    Bei 'all'-Selektion werden alle Szenarien iteriert, gemischt ARO/normal.
    """
    sel = aro_run or scenario or master.scenario_selection
    run_type = master.get_run_type(sel)
    print(f"[auto] Erkannter run_type für '{sel}': {run_type}")

    if run_type == "aro":
        return run_aro(master, out_dir, aro_run=sel, countries=countries, all_scenarios=all_scenarios)
    else:
        return run_normal(master, out_dir, scenario=sel, countries=countries, years=years)


def run_all_scenarios(
    master: MasterConfig,
    out_dir: Path,
    countries: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    all_scenarios: bool = False,
) -> Dict[str, Any]:
    """
    Iteriert über ALLE Einträge im scenarios.registry und ruft je nach
    run_type die passende Pipeline auf.
    """
    report: Dict[str, Any] = {"mode": "all", "ok": True, "sub_reports": {}, "warnings": [], "errors": []}

    reg = master.scenarios_registry
    for key in reg:
        run_type = master.get_run_type(key)
        print(f"\n[all] Verarbeite Szenario '{key}' (run_type={run_type})")
        try:
            if run_type == "aro":
                sub = run_aro(master, out_dir, aro_run=key, countries=countries, all_scenarios=all_scenarios)
            else:
                sub = run_normal(master, out_dir, scenario=key, countries=countries, years=years)
            report["sub_reports"][key] = sub
            if not sub.get("ok", True):
                report["ok"] = False
            report["warnings"].extend(sub.get("warnings", []))
            report["errors"].extend(sub.get("errors", []))
        except Exception as e:
            report["ok"] = False
            report["errors"].append(f"Szenario '{key}' fehlgeschlagen: {e}")

    return report


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["auto", "aro", "normal", "all", "all-scenarios", "validate"],
        default="auto",
        help=(
            "auto: run_type aus Config lesen | "
            "aro: ARO-Pipeline | "
            "normal: normale/myopische Pipeline | "
            "all: alle Szenarien aus registry | "
            "all-scenarios: ARO + alle Dispatch-Szenarien einzeln | "
            "validate: nur Validierung"
        ),
    )
    ap.add_argument(
        "--standalone",
        nargs="*",
        metavar="SCRIPT_KEY",
        help=(
                "Standalone plot_*.py Skripte ausführen. "
                "Ohne Argumente: alle. Mit Keys: nur diese. "
                "Verfügbare Keys: " + ", ".join(STANDALONE_SCRIPTS.keys())
        ),
    )
    ap.add_argument(
        "--no-standalone",
        action="store_true",
        help="Standalone Skripte deaktivieren (nur CountryAnalyzer / ARO-Plots)"
    )

    ap.add_argument("--aro-run",    default=None, help="ARO Run Key überschreiben")
    ap.add_argument("--scenario",   default=None, help="Scenario Key überschreiben")
    ap.add_argument("--countries",  default=None, help="Komma-getrennte Länder (z.B. ALL,DE,FR)")
    ap.add_argument("--years",      default=None, help="Komma-getrennte Jahre (z.B. 2050)")
    ap.add_argument("--all-scenarios", action="store_true",
                    help="Alle ARO Dispatch-Szenarien einzeln per CountryAnalyzer auswerten")
    ap.add_argument("--strict",     action="store_true", help="Strikte Validierung")
    ap.add_argument("--run-name",   default="analysis", help="Prefix für Output-Ordner")
    args = ap.parse_args()

    master    = MasterConfig()
    v         = validate_config(master, strict=args.strict)
    out_dir   = make_run_output_dir(master, run_name=args.run_name)
    write_report(out_dir, v, name="validation")

    if args.mode == "validate":
        print(f"[OK={v['ok']}] Validierungsbericht: {out_dir}")
        return

    countries = _parse_csv(args.countries)
    years     = _parse_int_list(args.years)

    report: Dict[str, Any] = {
        "ok": True, "validation_ok": v["ok"],
        "out_dir": str(out_dir),
        "warnings": [], "errors": [],
    }

    if args.mode == "auto":
        sub = run_auto(master, out_dir,
                       aro_run=args.aro_run, scenario=args.scenario,
                       countries=countries, years=years, all_scenarios=args.all_scenarios)
        report.update(sub)

    elif args.mode == "aro":
        sub = run_aro(master, out_dir, aro_run=args.aro_run,
                      countries=countries, all_scenarios=args.all_scenarios)
        report["aro"] = sub
        if not sub.get("ok", True): report["ok"] = False
        report["warnings"].extend(sub.get("warnings", []))
        report["errors"].extend(sub.get("errors", []))

    elif args.mode in ("normal", "myopic"):
        sub = run_normal(master, out_dir, scenario=args.scenario,
                         countries=countries, years=years)
        report["normal"] = sub
        if not sub.get("ok", True): report["ok"] = False
        report["warnings"].extend(sub.get("warnings", []))
        report["errors"].extend(sub.get("errors", []))

    elif args.mode in ("all", "all-scenarios"):
        sub = run_all_scenarios(master, out_dir, countries=countries, years=years,
                                all_scenarios=(args.mode == "all-scenarios" or args.all_scenarios))
        report["all"] = sub
        if not sub.get("ok", True): report["ok"] = False
        report["warnings"].extend(sub.get("warnings", []))
        report["errors"].extend(sub.get("errors", []))

    write_report(out_dir, report, name="report")
    print(f"[OK={report['ok']}] Bericht gespeichert: {out_dir}")


if __name__ == "__main__":
    main()
