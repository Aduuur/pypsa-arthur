#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified runner for:
- ARO analysis (aro_analysis.py)
- deterministic/myopic scenario plots (country_analysis.py)

Uses: master_config.py (single source of truth)

Usage examples:
  python plots_all/run_analysis.py --mode all
  python plots_all/run_analysis.py --mode aro --aro-run compare-robust-vol2
  python plots_all/run_analysis.py --mode myopic --scenario compare-robust-vol2 --countries ALL,DE
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

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


def _parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _year_from_path(path: str) -> Optional[int]:
    m = re.search(r"___(\d{4})\.nc$", path)  # matches base_s_24___2050.nc
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d{4})\.nc$", path)
    return int(m.group(1)) if m else None


def run_aro(master: MasterConfig, out_dir: Path, aro_run: Optional[str] = None, countries: Optional[List[str]] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {"mode": "aro", "ok": True, "steps": [], "warnings": [], "errors": []}

    aro_cfg = AROPlottingConfig(master=master)
    if aro_run is not None:
        aro_cfg.SELECTED_RUN = aro_run

    run_key = aro_cfg.SELECTED_RUN
    run_conf = aro_cfg.get_current_run_config()

    aro_out = out_dir / "aro" / run_key
    aro_out.mkdir(parents=True, exist_ok=True)

    # ensure plots saved into this run folder
    aro_cfg.PLOT_OUTPUT_PATH = str(aro_out)

    report["steps"].append({"load_run_config": run_conf})

    try:
        analyzer = AROAnalyzer(config=aro_cfg, auto_find_dispatch=True)
        report["steps"].append({"AROAnalyzer": "initialized"})
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

    if toggles.get("convergence", True):
        try:
            analyzer.plot_aro_convergence()
            report["steps"].append({"plot_aro_convergence": "ok"})
        except Exception as e:
            report["warnings"].append(f"plot_aro_convergence failed: {e}")

    if toggles.get("scenario_comparison", True):
        try:
            analyzer.plot_scenario_cost_comparison()
            report["steps"].append({"plot_scenario_cost_comparison": "ok"})
        except Exception as e:
            report["warnings"].append(f"plot_scenario_cost_comparison failed: {e}")

    if toggles.get("capacity_comparison", True):
        try:
            analyzer.plot_robust_capacity_breakdown()
            report["steps"].append({"plot_robust_capacity_breakdown": "ok"})
        except Exception as e:
            report["warnings"].append(f"plot_robust_capacity_breakdown failed: {e}")

    if toggles.get("worst_case_analysis", False):
        c_list = countries or aro_cfg.COUNTRIES_TO_ANALYZE
        for c in c_list:
            try:
                analyzer.analyze_worst_case_dispatch(country=c)
                report["steps"].append({"analyze_worst_case_dispatch": c})
            except Exception as e:
                report["warnings"].append(f"worst_case_analysis failed for {c}: {e}")

    if toggles.get("robustness_metrics", False):
        report["warnings"].append("robustness_metrics toggle enabled, but no implementation hooked yet.")

    return report


def run_myopic(master: MasterConfig,
               out_dir: Path,
               scenario: Optional[str] = None,
               countries: Optional[List[str]] = None,
               years: Optional[List[int]] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {"mode": "myopic", "ok": True, "steps": [], "warnings": [], "errors": []}

    if scenario is not None:
        master.raw["scenarios"]["selection"] = scenario

    plot_cfg = PlottingConfig(master=master)

    networks = plot_cfg.get_networks()
    if networks is None:
        report["ok"] = False
        report["errors"].append(f"No networks found for scenario selection '{plot_cfg.SCENARIO_SELECTION}'.")
        return report

    c_list = countries or plot_cfg.get_countries()
    myopic_out = out_dir / "myopic" / plot_cfg.SCENARIO_SELECTION
    myopic_out.mkdir(parents=True, exist_ok=True)

    def analyze_one_network(net_path: str):
        y = _year_from_path(net_path)
        if years is not None and y is not None and y not in years:
            return
        if not Path(net_path).is_file():
            report["warnings"].append(f"Network file missing: {net_path}")
            return

        tag = str(y) if y is not None else Path(net_path).stem
        per_net_out = myopic_out / tag
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
            analyze_one_network(p)
    elif isinstance(networks, dict):
        for branch, paths in networks.items():
            branch_out = myopic_out / branch
            branch_out.mkdir(parents=True, exist_ok=True)
            orig = myopic_out
            myopic_out = branch_out  # noqa
            for p in paths:
                analyze_one_network(p)
            myopic_out = orig

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "aro", "myopic", "validate"], default="all")

    ap.add_argument("--aro-run", default=None, help="Override ARO run key (e.g. compare-robust-vol2)")
    ap.add_argument("--scenario", default=None, help="Override scenario selection key")
    ap.add_argument("--countries", default=None, help="Comma-separated countries (e.g. ALL,DE,FR)")
    ap.add_argument("--years", default=None, help="Comma-separated years filter (e.g. 2050)")
    ap.add_argument("--strict", action="store_true", help="Strict validation (errors -> exit)")

    ap.add_argument("--run-name", default="analysis", help="Prefix for output folder naming")
    args = ap.parse_args()

    master = MasterConfig()

    v = validate_config(master, strict=args.strict)
    out_dir = make_run_output_dir(master, run_name=args.run_name)

    write_report(out_dir, v, name="validation")

    if args.mode == "validate":
        print(f"[OK={v['ok']}] validation report written to: {out_dir}")
        return

    countries = _parse_csv_list(args.countries)
    years = _parse_int_list(args.years)

    report: Dict[str, Any] = {
        "ok": True,
        "validation_ok": v["ok"],
        "out_dir": str(out_dir),
        "aro": None,
        "myopic": None,
        "warnings": [],
        "errors": [],
    }

    if args.mode in ("aro", "all"):
        aro_rep = run_aro(master, out_dir, aro_run=args.aro_run, countries=countries)
        report["aro"] = aro_rep
        if not aro_rep.get("ok", True):
            report["ok"] = False
        report["warnings"].extend(aro_rep.get("warnings", []))
        report["errors"].extend(aro_rep.get("errors", []))

    if args.mode in ("myopic", "all"):
        my_rep = run_myopic(master, out_dir, scenario=args.scenario, countries=countries, years=years)
        report["myopic"] = my_rep
        if not my_rep.get("ok", True):
            report["ok"] = False
        report["warnings"].extend(my_rep.get("warnings", []))
        report["errors"].extend(my_rep.get("errors", []))

    write_report(out_dir, report, name="report")
    print(f"[OK={report['ok']}] report written to: {out_dir}")


if __name__ == "__main__":
    main()