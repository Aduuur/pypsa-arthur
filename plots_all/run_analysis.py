#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified runner für ARO- und normale (myopische/deterministische) Läufe.

Verwendung:
  python plots_all/run_analysis.py --mode auto
  python plots_all/run_analysis.py --mode aro --aro-run compare-robust-vol2
  python plots_all/run_analysis.py --mode normal --scenario new_avg --countries ALL,DE
  python plots_all/run_analysis.py --mode standalone              # nur plot_*.py Skripte
  python plots_all/run_analysis.py --mode aro --no-standalone     # ohne standalone Skripte
  python plots_all/run_analysis.py --mode aro --standalone dispatch_timeline co2_emissionen
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# -----------------------------------------------------------------------
# Standalone-Script-Registry
# -----------------------------------------------------------------------
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

# Skripte die sinnvoll auf einem einzelnen Netz (ARO Worst-Case) laufen
ARO_APPLICABLE_SCRIPTS: set = {
    "dispatch_timeline",
    "balance_timeline",
    "energy_gen_no_storage",
    "generation_timeline_res",
    "generation_timeline_gen",
    "storage_v2",
    "marginal_prices",
    "dec_price_normal",
    "dec_price_new_jan",
    "dispatch_gas_h2",
    "consumption_timeline",
    "co2_emissionen",
}

SCRIPTS_DIR = Path(__file__).parent


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

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


def _inject_aro_dispatch_network(aro_dispatch_path: str) -> dict:
    """
    Temporarily override the network path in MASTER_CONFIG so standalone
    scripts pick up the ARO worst-case dispatch network instead of the
    planning network (base_s_24___2050.nc).

    Returns a dict with the original values so the caller can restore them
    in a finally block via _restore_network_config().

    The function patches the following common keys used by standalone scripts:
      MASTER_CONFIG["networks"]["2050"]          (most scripts)
      MASTER_CONFIG["networks"]["network_2050"]  (alt key variant)
      MASTER_CONFIG["network_path"]              (simple key variant)
    All patches are only applied when the key already exists; unknown schemas
    are left untouched so the injection is always safe.
    """
    from master_config import MASTER_CONFIG

    saved: dict = {}
    p = str(aro_dispatch_path)

    # Variant A: nested dict under "networks"
    networks = MASTER_CONFIG.get("networks", {})
    if isinstance(networks, dict):
        for k in list(networks.keys()):
            # patch any key that looks like a year-keyed path (2050, "2050", etc.)
            if str(k) == "2050" or k == "network_2050":
                saved[("networks", k)] = networks[k]
                networks[k] = p

    # Variant B: flat "network_path" key
    if "network_path" in MASTER_CONFIG:
        saved[("network_path",)] = MASTER_CONFIG["network_path"]
        MASTER_CONFIG["network_path"] = p

    # Variant C: flat "network_2050" key at top level
    if "network_2050" in MASTER_CONFIG:
        saved[("network_2050",)] = MASTER_CONFIG["network_2050"]
        MASTER_CONFIG["network_2050"] = p

    return saved


def _restore_network_config(saved: dict) -> None:
    """Restore MASTER_CONFIG keys previously saved by _inject_aro_dispatch_network."""
    from master_config import MASTER_CONFIG

    networks = MASTER_CONFIG.get("networks", {})
    for key_tuple, original in saved.items():
        if len(key_tuple) == 2 and key_tuple[0] == "networks":
            if isinstance(networks, dict):
                networks[key_tuple[1]] = original
        elif len(key_tuple) == 1:
            MASTER_CONFIG[key_tuple[0]] = original


# -----------------------------------------------------------------------
# Standalone Dispatcher
# -----------------------------------------------------------------------

def _load_and_run_script(
    script_key: str,
    override_scenario: Optional[str] = None,
    aro_dispatch_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and execute a standalone plot_*.py script.

    Parameters
    ----------
    script_key : str
        Registry key from STANDALONE_SCRIPTS.
    override_scenario : str, optional
        If set, temporarily overrides MASTER_CONFIG["scenarios"]["selection"].
    aro_dispatch_path : str, optional
        If set (ARO mode), temporarily patches the network path in MASTER_CONFIG
        so the script reads the worst-case dispatch network instead of the
        planning network.  Without this, all time-series based plots produce
        0 GWh generation / empty marginal prices because the planning network
        has no dispatch solution stored in generators_t, buses_t, etc.
    """
    result: Dict[str, Any] = {"script": script_key, "ok": False, "error": None}

    filename = STANDALONE_SCRIPTS.get(script_key)
    if filename is None:
        result["error"] = f"Unbekannter Script-Key: {script_key}"
        return result

    script_path = SCRIPTS_DIR / filename
    if not script_path.is_file():
        result["error"] = f"Datei nicht gefunden: {script_path}"
        return result

    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))

    from master_config import MASTER_CONFIG
    _orig_sel = None
    _network_backup: dict = {}

    if override_scenario is not None:
        _orig_sel = MASTER_CONFIG["scenarios"]["selection"]
        MASTER_CONFIG["scenarios"]["selection"] = override_scenario

    # --- ARO fix: redirect network path to worst-case dispatch -----------
    if aro_dispatch_path is not None:
        if Path(aro_dispatch_path).is_file():
            _network_backup = _inject_aro_dispatch_network(aro_dispatch_path)
            print(f"     [ARO] Netzwerk → {Path(aro_dispatch_path).name}")
        else:
            print(f"     [ARO] Warnung: Worst-Case-Dispatch nicht gefunden: {aro_dispatch_path}")
    # ----------------------------------------------------------------------

    try:
        spec = importlib.util.spec_from_file_location(f"_standalone_{script_key}", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "main"):
            mod.main()
            result["ok"] = True
        else:
            result["error"] = "Kein main() gefunden"
    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    finally:
        if _orig_sel is not None:
            MASTER_CONFIG["scenarios"]["selection"] = _orig_sel
        if _network_backup:
            _restore_network_config(_network_backup)

    return result


def run_standalone_scripts(
    script_keys: Optional[List[str]] = None,
    aro_only: bool = False,
    override_scenario: Optional[str] = None,
    aro_dispatch_path: Optional[str] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"ok": True, "scripts": {}}

    keys = script_keys or list(STANDALONE_SCRIPTS.keys())
    if aro_only:
        keys = [k for k in keys if k in ARO_APPLICABLE_SCRIPTS]

    for key in keys:
        print(f"  [standalone] {key}  ({STANDALONE_SCRIPTS.get(key, '?')})")
        res = _load_and_run_script(
            key,
            override_scenario=override_scenario,
            aro_dispatch_path=aro_dispatch_path,
        )
        report["scripts"][key] = res
        if res["ok"]:
            print(f"  ✓ {key}")
        else:
            print(f"  ✗ {key}: {res['error']}")
            # Einzelner Fehler stoppt nicht den Rest

    return report


# -----------------------------------------------------------------------
# ARO runner
# -----------------------------------------------------------------------

def run_aro(
    master: MasterConfig,
    out_dir: Path,
    aro_run: Optional[str] = None,
    countries: Optional[List[str]] = None,
    all_scenarios: bool = False,
    run_standalone: bool = True,
    standalone_keys: Optional[List[str]] = None,
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
        report["steps"].append({
            "AROAnalyzer": "initialized",
            "scenario_networks_loaded": len(analyzer.scenario_networks),
        })
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"AROAnalyzer init failed: {e}")
        return report

    try:
        analyzer.summary_report()
        report["steps"].append({"summary_report": "ok"})
    except Exception as e:
        report["warnings"].append(f"summary_report failed: {e}")

    toggles = aro_cfg.ARO_PLOTS

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

    try:
        analyzer.plot_scenario_capacity_comparison()
        report["steps"].append({"plot_scenario_capacity_comparison": "ok"})
    except Exception as e:
        report["warnings"].append(f"plot_scenario_capacity_comparison failed: {e}")

    if toggles.get("worst_case_analysis", False):
        c_list = countries or aro_cfg.COUNTRIES_TO_ANALYZE
        for c in c_list:
            try:
                analyzer.analyze_worst_case_dispatch(country=c)
                report["steps"].append({"analyze_worst_case_dispatch": c})
            except Exception as e:
                report["warnings"].append(f"worst_case_analysis failed for {c}: {e}")

    if all_scenarios:
        c_list = countries or aro_cfg.COUNTRIES_TO_ANALYZE
        try:
            analyzer.analyze_all_scenario_dispatches(countries=c_list, save=True)
            report["steps"].append({"analyze_all_scenario_dispatches": "ok"})
        except Exception as e:
            report["warnings"].append(f"analyze_all_scenario_dispatches failed: {e}")

    # ------------------------------------------------------------------
    # Standalone Skripte
    # ------------------------------------------------------------------
    if run_standalone:
        # Resolve the worst-case dispatch path so standalone scripts get a
        # network with an actual dispatch solution (generators_t.p, buses_t.
        # marginal_price, etc.) instead of the empty planning network.
        wc_dispatch_path: Optional[str] = None
        try:
            wc_dispatch_path = analyzer.get_worst_case_dispatch_path()
        except Exception:
            pass

        # Fallback: inspect scenario_networks for the worst-case key
        if not wc_dispatch_path:
            try:
                wc_key = analyzer.worst_case_cutout  # attribute set during init
                if wc_key and wc_key in analyzer.scenario_networks:
                    n = analyzer.scenario_networks[wc_key]
                    if hasattr(n, "_source_path"):
                        wc_dispatch_path = n._source_path
                    elif hasattr(analyzer, "_dispatch_paths") and wc_key in analyzer._dispatch_paths:
                        wc_dispatch_path = analyzer._dispatch_paths[wc_key]
            except Exception:
                pass

        # Second fallback: search dispatch dir directly
        if not wc_dispatch_path:
            try:
                dispatch_dir = Path(analyzer.config.get_dispatch_dir())
                worst_cutout = analyzer.worst_case_cutout or ""
                safe = worst_cutout.replace("/", "_").replace(" ", "_")
                for candidate in (
                    f"dispatch_{safe}_worst_case_std.nc",
                    f"dispatch_{safe}_std.nc",
                ):
                    p = dispatch_dir / candidate
                    if p.is_file():
                        wc_dispatch_path = str(p)
                        break
                if not wc_dispatch_path:
                    # any *_worst_case_std.nc in dispatch dir
                    candidates = sorted(dispatch_dir.glob("*_worst_case_std.nc"))
                    if candidates:
                        wc_dispatch_path = str(candidates[0])
            except Exception:
                pass

        if wc_dispatch_path:
            print(f"\n  [ARO] Worst-Case-Dispatch für Standalone-Skripte: "
                  f"{Path(wc_dispatch_path).name}")
        else:
            print("\n  [ARO] Warnung: Kein Worst-Case-Dispatch gefunden – "
                  "Standalone-Skripte laufen ohne Dispatch-Netz (Ergebnisse werden leer sein)")

        print("\n=== Standalone plot_*.py Skripte (ARO-kompatibel) ===")
        standalone_report = run_standalone_scripts(
            script_keys=standalone_keys,
            aro_only=(standalone_keys is None),   # ohne explizite Auswahl: nur ARO_APPLICABLE
            override_scenario=run_key,
            aro_dispatch_path=wc_dispatch_path,
        )
        report["standalone"] = standalone_report

    return report


# -----------------------------------------------------------------------
# Normal/myopic runner
# -----------------------------------------------------------------------

def run_normal(
    master: MasterConfig,
    out_dir: Path,
    scenario: Optional[str] = None,
    countries: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    run_standalone: bool = True,
    standalone_keys: Optional[List[str]] = None,
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
                ca = CountryAnalyzer(
                    network_path=net_path,
                    country=c,
                    output_dir=str(per_net_out / c),
                    carrier_colors=plot_cfg.CARRIER_COLORS,
                    default_color=plot_cfg.DEFAULT_COLOR,
                )
                ca.summary_report()
                ca.generate_all_plots(save=True)
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

    # Standalone Skripte
    if run_standalone:
        print("\n=== Standalone plot_*.py Skripte ===")
        standalone_report = run_standalone_scripts(
            script_keys=standalone_keys,
            aro_only=False,
            override_scenario=plot_cfg.SCENARIO_SELECTION,
        )
        report["standalone"] = standalone_report

    return report


# -----------------------------------------------------------------------
# Auto-detection
# -----------------------------------------------------------------------

def run_auto(
    master: MasterConfig,
    out_dir: Path,
    aro_run: Optional[str] = None,
    scenario: Optional[str] = None,
    countries: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    all_scenarios: bool = False,
    run_standalone: bool = True,
    standalone_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    sel = aro_run or scenario or master.scenario_selection
    run_type = master.get_run_type(sel)
    print(f"[auto] Erkannter run_type für '{sel}': {run_type}")

    if run_type == "aro":
        return run_aro(master, out_dir, aro_run=sel, countries=countries,
                       all_scenarios=all_scenarios,
                       run_standalone=run_standalone, standalone_keys=standalone_keys)
    else:
        return run_normal(master, out_dir, scenario=sel, countries=countries,
                          years=years,
                          run_standalone=run_standalone, standalone_keys=standalone_keys)


def run_all_scenarios(
    master: MasterConfig,
    out_dir: Path,
    countries: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    all_scenarios: bool = False,
    run_standalone: bool = True,
    standalone_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"mode": "all", "ok": True, "sub_reports": {}, "warnings": [], "errors": []}

    for key in master.scenarios_registry:
        run_type = master.get_run_type(key)
        print(f"\n[all] Verarbeite Szenario '{key}' (run_type={run_type})")
        try:
            if run_type == "aro":
                sub = run_aro(master, out_dir, aro_run=key, countries=countries,
                              all_scenarios=all_scenarios,
                              run_standalone=run_standalone, standalone_keys=standalone_keys)
            else:
                sub = run_normal(master, out_dir, scenario=key, countries=countries,
                                 years=years,
                                 run_standalone=run_standalone, standalone_keys=standalone_keys)
            report["sub_reports"][key] = sub
            if not sub.get("ok", True):
                report["ok"] = False
            report["warnings"].extend(sub.get("warnings", []))
            report["errors"].extend(sub.get("errors", []))
        except Exception as e:
            report["ok"] = False
            report["errors"].append(f"Szenario '{key}' fehlgeschlagen: {e}")

    return report


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument(
        "--mode",
        choices=["auto", "aro", "normal", "myopic", "all", "all-scenarios", "validate", "standalone"],
        default="auto",
        help=(
            "auto: run_type aus Config | aro: ARO-Pipeline | "
            "normal/myopic: normale Pipeline | all: alle Szenarien | "
            "all-scenarios: ARO + alle Dispatch-Szenarien | "
            "standalone: nur plot_*.py Skripte | validate: nur Validierung"
        ),
    )
    ap.add_argument("--aro-run",       default=None, help="ARO Run Key überschreiben")
    ap.add_argument("--scenario",      default=None, help="Scenario Key überschreiben")
    ap.add_argument("--countries",     default=None, help="Komma-getrennte Länder (z.B. ALL,DE,FR)")
    ap.add_argument("--years",         default=None, help="Komma-getrennte Jahre (z.B. 2050)")
    ap.add_argument("--all-scenarios", action="store_true",
                    help="Alle ARO Dispatch-Szenarien einzeln per CountryAnalyzer auswerten")
    ap.add_argument("--standalone",    nargs="*", metavar="KEY",
                    help=(
                        "Standalone plot_*.py Skripte ausführen. "
                        "Ohne Argumente: alle. Mit Keys: nur diese. "
                        "Verfügbare Keys: " + ", ".join(STANDALONE_SCRIPTS.keys())
                    ))
    ap.add_argument("--no-standalone", action="store_true",
                    help="Standalone Skripte komplett deaktivieren")
    ap.add_argument("--strict",        action="store_true", help="Strikte Validierung")
    ap.add_argument("--run-name",      default="analysis", help="Prefix für Output-Ordner")
    args = ap.parse_args()

    master  = MasterConfig()
    v       = validate_config(master, strict=args.strict)
    out_dir = make_run_output_dir(master, run_name=args.run_name)
    write_report(out_dir, v, name="validation")

    if args.mode == "validate":
        print(f"[OK={v['ok']}] Validierungsbericht: {out_dir}")
        return

    countries      = _parse_csv(args.countries)
    years          = _parse_int_list(args.years)
    run_standalone = not args.no_standalone
    # args.standalone ist None wenn Flag nicht gesetzt,
    # [] wenn --standalone ohne Keys, ["k1","k2"] mit Keys
    standalone_keys = args.standalone if args.standalone is not None else None

    # --mode standalone: nur Standalone-Skripte, kein CountryAnalyzer / ARO
    if args.mode == "standalone":
        print("\n=== Nur Standalone Skripte ===")
        rep = run_standalone_scripts(
            script_keys=standalone_keys,
            override_scenario=args.scenario or args.aro_run,
        )
        write_report(out_dir, rep, name="report")
        print(f"[OK={rep['ok']}] Bericht: {out_dir}")
        return

    report: Dict[str, Any] = {
        "ok": True,
        "validation_ok": v["ok"],
        "out_dir": str(out_dir),
        "warnings": [], "errors": [],
    }

    if args.mode == "auto":
        sub = run_auto(
            master, out_dir,
            aro_run=args.aro_run, scenario=args.scenario,
            countries=countries, years=years,
            all_scenarios=args.all_scenarios,
            run_standalone=run_standalone,
            standalone_keys=standalone_keys,
        )
        report.update(sub)

    elif args.mode == "aro":
        sub = run_aro(
            master, out_dir,
            aro_run=args.aro_run,
            countries=countries,
            all_scenarios=args.all_scenarios,
            run_standalone=run_standalone,
            standalone_keys=standalone_keys,
        )
        report["aro"] = sub
        if not sub.get("ok", True): report["ok"] = False
        report["warnings"].extend(sub.get("warnings", []))
        report["errors"].extend(sub.get("errors", []))

    elif args.mode in ("normal", "myopic"):
        sub = run_normal(
            master, out_dir,
            scenario=args.scenario,
            countries=countries, years=years,
            run_standalone=run_standalone,
            standalone_keys=standalone_keys,
        )
        report["normal"] = sub
        if not sub.get("ok", True): report["ok"] = False
        report["warnings"].extend(sub.get("warnings", []))
        report["errors"].extend(sub.get("errors", []))

    elif args.mode in ("all", "all-scenarios"):
        sub = run_all_scenarios(
            master, out_dir,
            countries=countries, years=years,
            all_scenarios=(args.mode == "all-scenarios" or args.all_scenarios),
            run_standalone=run_standalone,
            standalone_keys=standalone_keys,
        )
        report["all"] = sub
        if not sub.get("ok", True): report["ok"] = False
        report["warnings"].extend(sub.get("warnings", []))
        report["errors"].extend(sub.get("errors", []))

    write_report(out_dir, report, name="report")
    print(f"\n[OK={report['ok']}] Bericht gespeichert: {out_dir}")


if __name__ == "__main__":
    main()
