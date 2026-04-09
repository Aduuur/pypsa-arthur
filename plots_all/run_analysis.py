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

Output-Struktur (unified per-run):
  <plots_base>/<run_key>/
    country/<land>/        <- CountryAnalyzer (normal + ARO worst-case)
    aro_metrics/           <- ARO Konvergenz, Gap, ...
    capacity/              <- robuste Kapazitäten
    scenario_comparison/   <- Kapazitäts-Vergleich Robust vs. Basisjahr
    worst_case/            <- Worst-Case Dispatch-Analyse
    annual_dispatch/       <- Jährlicher Dispatch je Szenario
    capacity_by_country/   <- Kapazitäten nach Land (ARO)
    validation_report.json
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
    "map_nutzbare_uebertrag":  "plot_map_nutzbare_\u00fcbertragung.py",
    "co2_emissionen":          "plot_jaehrliche_co2_emissionen.py",
    "co2_emissionen_analyse":  "plot_jaehrliche_co2_emissionen_analyse.py",
    "check_waermepumpen":      "plot_check_w\u00e4rmepumpen_df.py",
    # ARO-spezifische Plots
    "aro_capacity_by_country": "plot_aro_capacity_by_country.py",
    "aro_annual_dispatch":     "plot_aro_annual_dispatch.py",
}

# Skripte die sinnvoll auf einem einzelnen Netz (ARO Worst-Case) laufen.
# HINWEIS: installed_cap_vgl ist hier NICHT enthalten — im ARO-Modus wird
# der Scenario-Vergleich ausschliesslich ueber
# analyzer.plot_scenario_capacity_comparison() gehandhabt, das intern
# plot_robust_vs_basis() aufruft. So werden n_robust und n_basis direkt
# als geladene Netzwerke uebergeben (keine Pfad-Aufloesung noetig).
ARO_APPLICABLE_SCRIPTS: set = {
    "dispatch_timeline",
    "balance_timeline",
    "energy_gen_no_storage",
    "generation_timeline_res",
    "generation_timeline_gen",
    "storage_v2",
    "storage_kombi_es_de",
    "storage_kombi_fr_de",
    "marginal_prices",
    "marginal_prices_check",
    "dec_price_normal",
    "dec_price_new_jan",
    "dec_price_split",
    "delta_prices_map",
    "dispatch_gas_h2",
    "consumption_timeline",
    "map_leitung_export",
    "map_nutzbare_uebertrag",
    "co2_emissionen",
    "co2_emissionen_analyse",
    "check_waermepumpen",
    "aro_capacity_by_country",
    "aro_annual_dispatch",
}

# Skripte die im ARO-Modus direkt über main(aro_network=...) aufgerufen werden,
# statt MASTER_CONFIG zu patchen.  Das Skript muss main(aro_network=None)
# als Signatur haben und den ARO-Pfad selbst auswerten.
# HINWEIS: installed_cap_vgl wurde entfernt — wird jetzt über den AROAnalyzer
# direkt aufgerufen (plot_scenario_capacity_comparison).
ARO_DIRECT_CALL_SCRIPTS: set = {
    "delta_prices_map",
}

# Skripte, die im ARO-Modus explizit den Vergleich
# Basisjahr (average) vs. Worst-Case-Dispatch (dunkelflaute) nutzen sollen.
# Dafür wird SCENARIO_SELECTION temporär auf "both" gesetzt.
ARO_BASIS_WC_COMPARISON_SCRIPTS: set = {
    "co2_emissionen",
    "co2_emissionen_analyse",
    "dec_delta_dispatch",
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
    """Jahr aus Dateipfad extrahieren.

    Strategie:
      1. Netzwerk laden und Jahr aus n.snapshots lesen (sicher, auch für
         ARO-Dispatch-Dateien ohne Jahr im Namen).
      2. Fallback: Regex auf den Dateinamen (klassische Planung ___YYYY.nc).
    """
    try:
        import pypsa
        n = pypsa.Network(path)
        return int(n.snapshots[0].year)
    except Exception:
        pass
    m = re.search(r"___(\d{4})\.nc$", path) or re.search(r"_(\d{4})\.nc$", path)
    return int(m.group(1)) if m else None


# -----------------------------------------------------------------------
# BUG B FIX: _inject_aro_dispatch_network
# -----------------------------------------------------------------------

def _inject_aro_dispatch_network(
    aro_dispatch_path: str,
    aro_basis_path: Optional[str] = None,
    selected_run_key: Optional[str] = None,
) -> dict:
    """
    Temporarily patch MASTER_CONFIG for ARO standalone scripts.

    - selected run -> Worst-Case-Dispatch
    - optional synthetic "both"-Vergleich:
      average     -> Basisjahr-Referenznetz
      dunkelflaute-> Worst-Case-Dispatch
    """
    from master_config import MASTER_CONFIG

    saved: dict = {}
    p_wc = str(aro_dispatch_path)
    p_basis = str(aro_basis_path) if aro_basis_path else None

    registry = MASTER_CONFIG.get("scenarios", {}).get("registry", {})
    if isinstance(registry, dict):
        for run_key, run_cfg in registry.items():
            if isinstance(run_cfg, dict) and "networks" in run_cfg:
                saved[("scenarios", "registry", run_key, "networks")] = list(run_cfg["networks"])
                # Nur den aktiven ARO-Run auf WC umbiegen; andere Einträge bleiben.
                if selected_run_key and run_key == selected_run_key:
                    run_cfg["networks"] = [p_wc]

            # synthetische Vergleichs-Szenarien für "both"
            cmp_basis_key = "__aro_basis_reference__"
            cmp_wc_key = "__aro_worst_case_dispatch__"

            saved[("scenarios", "registry", cmp_basis_key, "__exists__")] = cmp_basis_key in registry
            saved[("scenarios", "registry", cmp_wc_key, "__exists__")] = cmp_wc_key in registry
            if cmp_basis_key in registry:
                saved[("scenarios", "registry", cmp_basis_key, "__entry__")] = dict(registry[cmp_basis_key])
            if cmp_wc_key in registry:
                saved[("scenarios", "registry", cmp_wc_key, "__entry__")] = dict(registry[cmp_wc_key])

            if p_basis:
                registry[cmp_basis_key] = {
                    "run_type": "normal",
                    "description": "ARO Basisjahr-Referenz (temporär)",
                    "networks": [p_basis],
                }
            registry[cmp_wc_key] = {
                "run_type": "normal",
                "description": "ARO Worst-Case-Dispatch (temporär)",
                "networks": [p_wc],
            }

            both_mapping = MASTER_CONFIG.get("scenarios", {}).get("both_mapping", {})
            saved[("scenarios", "both_mapping", "average")] = both_mapping.get("average")
            saved[("scenarios", "both_mapping", "dunkelflaute")] = both_mapping.get("dunkelflaute")
            both_mapping["average"] = cmp_basis_key if p_basis else cmp_wc_key
            both_mapping["dunkelflaute"] = cmp_wc_key

    if "network_path" in MASTER_CONFIG:
        saved[("network_path",)] = MASTER_CONFIG["network_path"]
        MASTER_CONFIG["network_path"] = p_wc

    if "network_2050" in MASTER_CONFIG:
        saved[("network_2050",)] = MASTER_CONFIG["network_2050"]
        MASTER_CONFIG["network_2050"] = p_wc

    return saved


def _restore_network_config(saved: dict) -> None:
    """Restore MASTER_CONFIG keys previously saved by _inject_aro_dispatch_network."""
    from master_config import MASTER_CONFIG

    for key_tuple, original in saved.items():
        if len(key_tuple) == 4 and key_tuple[:2] == ("scenarios", "registry") and key_tuple[3] == "networks":
            run_key = key_tuple[2]
            registry = MASTER_CONFIG.get("scenarios", {}).get("registry", {})
            if isinstance(registry, dict) and run_key in registry:
                if isinstance(registry[run_key], dict):
                    registry[run_key]["networks"] = original
        elif len(key_tuple) == 4 and key_tuple[:2] == ("scenarios", "registry") and key_tuple[3] == "__exists__":
            run_key = key_tuple[2]
            exists_before = bool(original)
            registry = MASTER_CONFIG.get("scenarios", {}).get("registry", {})
            if isinstance(registry, dict) and not exists_before and run_key in registry:
                registry.pop(run_key, None)
        elif len(key_tuple) == 4 and key_tuple[:2] == ("scenarios", "registry") and key_tuple[3] == "__entry__":
            run_key = key_tuple[2]
            registry = MASTER_CONFIG.get("scenarios", {}).get("registry", {})
            if isinstance(registry, dict):
                registry[run_key] = original
        elif len(key_tuple) == 3 and key_tuple[:2] == ("scenarios", "both_mapping"):
            mapping_key = key_tuple[2]
            both_mapping = MASTER_CONFIG.get("scenarios", {}).get("both_mapping", {})
            if isinstance(both_mapping, dict):
                if original is None:
                    both_mapping.pop(mapping_key, None)
                else:
                    both_mapping[mapping_key] = original

        elif len(key_tuple) == 1:
            MASTER_CONFIG[key_tuple[0]] = original

            # Zusätzliche Absicherung: temporäre ARO-Vergleichskeys entfernen,
            # falls sie vor dem Patchen nicht existierten.
        registry = MASTER_CONFIG.get("scenarios", {}).get("registry", {})
        if isinstance(registry, dict):
            for tmp_key in ("__aro_basis_reference__", "__aro_worst_case_dispatch__"):
                marker = ("scenarios", "registry", tmp_key, "__exists__")
                if marker in saved and not bool(saved[marker]):
                    registry.pop(tmp_key, None)


# -----------------------------------------------------------------------
# BUG A FIX: _resolve_wc_dispatch_path
# -----------------------------------------------------------------------

def _resolve_wc_dispatch_path(analyzer: "AROAnalyzer") -> Optional[str]:
    """
    Return the filesystem path of the worst-case dispatch network.
    """
    if analyzer.n_worst_case is not None:
        if hasattr(analyzer.n_worst_case, "_source_path"):
            p = str(analyzer.n_worst_case._source_path)
            if Path(p).is_file():
                return p

    worst_cutout: Optional[str] = (
        analyzer.aro_summary
        .get("aro_final_evaluation", {})
        .get("worst_case_cutout")
    )
    dispatch_dir: Optional[Path] = analyzer._dispatch_dir()

    if dispatch_dir is not None and worst_cutout:
        safe = analyzer._safe_name(str(worst_cutout))
        for candidate_name in (
            f"dispatch_{safe}_worst_case_std.nc",
            f"dispatch_{safe}_worst_case_flat.nc",
            f"dispatch_{safe}_std.nc",
            f"dispatch_{safe}_flat.nc",
        ):
            p = dispatch_dir / candidate_name
            if p.is_file():
                return str(p)
        for pattern in (
            f"dispatch_*{safe}*_worst_case_std.nc",
            f"dispatch_*{safe}*_worst_case*.nc",
            f"dispatch_*{safe}*_std.nc",
        ):
            hits = sorted(dispatch_dir.glob(pattern))
            if hits:
                return str(hits[0])

    if worst_cutout and worst_cutout in analyzer.scenario_networks:
        n = analyzer.scenario_networks[worst_cutout]
        if n is not None and hasattr(n, "_source_path"):
            p = str(n._source_path)
            if Path(p).is_file():
                return p

    if dispatch_dir is not None:
        for pattern in ("dispatch_*_worst_case_std.nc", "dispatch_*_worst_case*.nc"):
            hits = sorted(dispatch_dir.glob(pattern))
            if hits:
                return str(hits[0])
        hits = sorted(dispatch_dir.glob("dispatch_*_std.nc"))
        if hits:
            return str(hits[-1])

    return None

def _resolve_basis_network_path(analyzer: "AROAnalyzer") -> Optional[str]:
    """
    Return the filesystem path of the configured basis/reference planning network.
    """
    n_basis = getattr(analyzer, "n_basis", None)
    if n_basis is not None and hasattr(n_basis, "_source_path"):
        p = str(n_basis._source_path)
        if Path(p).is_file():
            return p

    run_conf = analyzer.config.get_current_run_config()
    for key in ("basis_network", "reference_network", "base_network"):
        p = run_conf.get(key)
        if p and Path(p).is_file():
            return str(p)

    try:
        ref_path = getattr(analyzer.config, "REFERENCE_NETWORK_PATH", None)
        if ref_path and Path(ref_path).is_file():
            return str(ref_path)
    except Exception:
        pass

    return None


# -----------------------------------------------------------------------
# Standalone Dispatcher
# -----------------------------------------------------------------------

def _load_and_run_script(
    script_key: str,
    override_scenario: Optional[str] = None,
    aro_dispatch_path: Optional[str] = None,
    aro_basis_path: Optional[str] = None,
    aro_selected_run: Optional[str] = None,
    aro_robust_path: Optional[str] = None,
    analyzer: Optional["AROAnalyzer"] = None,
) -> Dict[str, Any]:
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

    # ARO-native Skripte: direkt über analyzer laufen
    if script_key == "aro_capacity_by_country" and analyzer is not None:
        try:
            from plot_aro_capacity_by_country import run_capacity_by_country
            out_dir = analyzer.config.get_plot_output_dir("capacity_by_country")
            run_capacity_by_country(
                n_robust=analyzer.n_robust,
                scenario_networks=analyzer.scenario_networks,
                output_dir=out_dir,
                run_name=analyzer.config.SELECTED_RUN,
                user_colors=analyzer.config.CARRIER_COLORS,
                plot_diff=True,
                save=True,
            )
            result["ok"] = True
        except Exception as e:
            import traceback
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
        return result

    if script_key == "aro_annual_dispatch" and analyzer is not None:
        try:
            from plot_aro_annual_dispatch import run_annual_dispatch
            out_dir = analyzer.config.get_plot_output_dir("annual_dispatch")
            cost_dict = (
                analyzer.aro_summary
                .get("aro_final_evaluation", {})
                .get("all_costs", {}) or {}
            )
            run_annual_dispatch(
                n_robust=analyzer.n_robust,
                scenario_networks=analyzer.scenario_networks,
                output_dir=out_dir,
                run_name=analyzer.config.SELECTED_RUN,
                user_colors=analyzer.config.CARRIER_COLORS,
                cost_dict=cost_dict,
                save=True,
            )
            result["ok"] = True
        except Exception as e:
            import traceback
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
        return result

    # installed_cap_vgl wird im ARO-Modus NICHT mehr als Standalone aufgerufen.
    # Der Vergleich läuft über analyzer.plot_scenario_capacity_comparison()
    # direkt in run_aro(). Falls das Skript dennoch explizit per --standalone
    # installed_cap_vgl aufgerufen wird, leiten wir es an den Analyzer weiter
    # (falls vorhanden) oder weisen den Nutzer hin.
    if script_key == "installed_cap_vgl":
        if analyzer is not None:
            try:
                print(
                    "  [installed_cap_vgl] Leite an analyzer.plot_scenario_capacity_comparison() weiter."
                )
                analyzer.plot_scenario_capacity_comparison(save=True)
                result["ok"] = True
            except Exception as e:
                import traceback
                result["error"] = str(e)
                result["traceback"] = traceback.format_exc()
        else:
            # Normalmodus (kein ARO): klassischer Standalone-Aufruf via MASTER_CONFIG
            try:
                spec = importlib.util.spec_from_file_location(
                    "_standalone_installed_cap_vgl", script_path
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if not hasattr(mod, "main"):
                    result["error"] = "Kein main() gefunden"
                    return result
                mod.main()
                result["ok"] = True
            except Exception as e:
                import traceback
                result["error"] = str(e)
                result["traceback"] = traceback.format_exc()
        return result

    # delta_prices_map: nur aro_network (Dispatch-Pfad)
    if script_key == "delta_prices_map" and aro_dispatch_path is not None:
        if not Path(aro_dispatch_path).is_file():
            result["error"] = f"ARO-Dispatch nicht gefunden: {aro_dispatch_path}"
            return result
        try:
            spec = importlib.util.spec_from_file_location("_standalone_delta_prices_map", script_path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, "main"):
                result["error"] = "Kein main() gefunden"
                return result
            mod.main(aro_network=aro_dispatch_path)
            result["ok"] = True
        except Exception as e:
            import traceback
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
        return result

    # Standard: MASTER_CONFIG patchen und Skript ausführen
    from master_config import MASTER_CONFIG
    _orig_sel = None
    _network_backup: dict = {}

    effective_override = override_scenario
    if script_key in ARO_BASIS_WC_COMPARISON_SCRIPTS and aro_basis_path is not None:
        effective_override = "both"

    if effective_override is not None:
        _orig_sel = MASTER_CONFIG["scenarios"]["selection"]
        MASTER_CONFIG["scenarios"]["selection"] = effective_override

    if aro_dispatch_path is not None:
        if Path(aro_dispatch_path).is_file():
            _network_backup = _inject_aro_dispatch_network(
                aro_dispatch_path=aro_dispatch_path,
                aro_basis_path=aro_basis_path,
                selected_run_key=aro_selected_run,
            )
            print(f"     [ARO] Netzwerk → {Path(aro_dispatch_path).name}")
        else:
            print(f"     [ARO] Warnung: Worst-Case-Dispatch nicht gefunden: {aro_dispatch_path}")

    try:
        spec = importlib.util.spec_from_file_location(f"_standalone_{script_key}", script_path)
        mod  = importlib.util.module_from_spec(spec)
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
    aro_basis_path: Optional[str] = None,
    aro_selected_run: Optional[str] = None,
    aro_robust_path: Optional[str] = None,
    analyzer: Optional["AROAnalyzer"] = None,
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
            aro_robust_path=aro_robust_path,
            aro_basis_path=aro_basis_path,
            aro_selected_run=aro_selected_run,
            analyzer=analyzer,
        )
        report["scripts"][key] = res
        if res["ok"]:
            print(f"  ✓ {key}")
        else:
            print(f"  ✗ {key}: {res['error']}")

    return report


# -----------------------------------------------------------------------
# Hilfsfunktion: Basisjahr-Referenznetz ermitteln
# Fuer installed_cap_vgl: rechte Seite = Basisjahr-Planungsnetz
# (nicht das robuste Portfolio, nicht der Worst-Case Dispatch)
# -----------------------------------------------------------------------

def _resolve_robust_path(analyzer: "AROAnalyzer") -> Optional[str]:
    """
    Ermittelt den Dateipfad des Basisjahr-Planungsnetzes fuer installed_cap_vgl.
    Dieses Netz bildet die RECHTE Seite im Vergleich (= Basisjahr-Referenz).

    Das robuste Portfolio (n_robust) wird separat direkt aus analyzer.n_robust
    in _load_and_run_script() gezogen.

    Priorität:
      1. run_config["basis_network"] / run_config["reference_network"] /
         run_config["base_network"] — direkt in master_config.py setzen
      2. AROPlottingConfig.REFERENCE_NETWORK_PATH
      3. config.get_reference_networks()[0]
      4. Glob: BASE_RESULTS_PATH/<run_name>/networks/base_s_*.nc
         (ohne 'dispatch' im Namen)
    """
    run_conf = analyzer.config.get_current_run_config()

    # 1. Direkt aus run_config
    for key in ("basis_network", "reference_network", "base_network"):
        p = run_conf.get(key)
        if p and Path(p).is_file():
            print(f"  [basis_path] Aus run_config['{key}']: {Path(p).name}")
            return str(p)

    # 2. REFERENCE_NETWORK_PATH aus Config
    try:
        ref_path = getattr(analyzer.config, "REFERENCE_NETWORK_PATH", None)
        if ref_path and Path(ref_path).is_file():
            print(f"  [basis_path] REFERENCE_NETWORK_PATH: {Path(ref_path).name}")
            return str(ref_path)
    except Exception:
        pass

    # 3. Referenznetzwerke aus Config-Methode
    try:
        ref_networks = analyzer.config.get_reference_networks()
        if ref_networks:
            first = ref_networks[0] if isinstance(ref_networks, list) else next(iter(ref_networks.values()))[0]
            if Path(first).is_file():
                print(f"  [basis_path] get_reference_networks(): {Path(first).name}")
                return str(first)
    except Exception:
        pass

    # 4. Glob-Suche: base_s_*.nc ohne 'dispatch' im Namen
    try:
        base = Path(analyzer.config.BASE_RESULTS_PATH)
        run_name = analyzer.run_config.get("name", "")
        if run_name:
            run_dir = base / run_name / "networks"
            hits = [
                p for p in sorted(run_dir.glob("base_s_*.nc"))
                if "dispatch" not in p.name
            ]
            if hits:
                print(f"  [basis_path] Glob base_s_*.nc: {hits[0].name}")
                return str(hits[0])
    except Exception:
        pass

    print(
        "  [basis_path] WARNUNG: Kein Basisjahr-Planungsnetz gefunden.\n"
        "  Tipp: Setze run_config['basis_network'] = '<Pfad>' in master_config.py"
    )
    return None


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

    run_key = aro_cfg.SELECTED_RUN
    aro_cfg.PLOT_OUTPUT_PATH = str(out_dir)

    run_conf = aro_cfg.get_current_run_config()
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

    # Hauptplot: Robust vs. Basisjahr (Kapazitätsvergleich)
    # Wird direkt über den Analyzer aufgerufen — n_robust und n_basis sind
    # bereits als Netzwerk-Objekte geladen (keine externe Pfad-Auflösung nötig).
    # Ruft intern plot_robust_vs_basis() + Dispatch-Szenariovergleich auf.
    try:
        analyzer.plot_scenario_capacity_comparison()
        report["steps"].append({"plot_scenario_capacity_comparison": "ok"})
    except Exception as e:
        import traceback
        report["warnings"].append(f"plot_scenario_capacity_comparison failed: {e}")
        report["warnings"].append(traceback.format_exc())

    # Kapazitäten nach Land
    try:
        from plot_aro_capacity_by_country import run_capacity_by_country
        run_capacity_by_country(
            n_robust=analyzer.n_robust,
            scenario_networks=analyzer.scenario_networks,
            output_dir=aro_cfg.get_plot_output_dir("capacity_by_country"),
            run_name=run_key,
            user_colors=aro_cfg.CARRIER_COLORS,
            plot_diff=True,
            save=True,
        )
        report["steps"].append({"capacity_by_country": "ok"})
    except Exception as e:
        import traceback
        report["warnings"].append(f"capacity_by_country failed: {e}")
        report["warnings"].append(traceback.format_exc())

    # Jährlicher Dispatch
    try:
        from plot_aro_annual_dispatch import run_annual_dispatch
        cost_dict = (
            analyzer.aro_summary
            .get("aro_final_evaluation", {})
            .get("all_costs", {}) or {}
        )
        run_annual_dispatch(
            n_robust=analyzer.n_robust,
            scenario_networks=analyzer.scenario_networks,
            output_dir=aro_cfg.get_plot_output_dir("annual_dispatch"),
            run_name=run_key,
            user_colors=aro_cfg.CARRIER_COLORS,
            cost_dict=cost_dict,
            save=True,
        )
        report["steps"].append({"annual_dispatch": "ok"})
    except Exception as e:
        import traceback
        report["warnings"].append(f"annual_dispatch failed: {e}")
        report["warnings"].append(traceback.format_exc())

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

    # Standalone Skripte (installed_cap_vgl ist in ARO_APPLICABLE_SCRIPTS
    # nicht mehr enthalten — läuft bereits oben über den Analyzer)
    if run_standalone:
        wc_dispatch_path: Optional[str] = _resolve_wc_dispatch_path(analyzer)
        basis_network_path: Optional[str] = _resolve_basis_network_path(analyzer)


        if wc_dispatch_path:
            print(f"\n  [ARO] Worst-Case-Dispatch für Standalone-Skripte: "
                  f"{Path(wc_dispatch_path).name}")
        else:
            print("\n  [ARO] Warnung: Kein Worst-Case-Dispatch gefunden – "
                  "Standalone-Skripte laufen ohne Dispatch-Netz (Ergebnisse werden leer sein)")
        if basis_network_path:
            print(f"  [ARO] Basisjahr-Referenz für Vergleichsplots: {Path(basis_network_path).name}")

        print("\n=== Standalone plot_*.py Skripte (ARO-kompatibel) ===")
        standalone_report = run_standalone_scripts(
            script_keys=standalone_keys,
            aro_only=(standalone_keys is None),
            override_scenario=run_key,
            aro_dispatch_path=wc_dispatch_path,
            aro_basis_path=basis_network_path,
            aro_selected_run=run_key,
            aro_robust_path=None,   # nicht mehr benötigt (installed_cap_vgl läuft über Analyzer)
            analyzer=analyzer,
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
    networks  = plot_cfg.get_networks()

    if networks is None:
        report["ok"] = False
        report["errors"].append(f"Keine Netzwerke für Szenario '{plot_cfg.SCENARIO_SELECTION}'.")
        return report

    c_list = countries or plot_cfg.get_countries()

    def analyze_one(net_path: str, base_out: Path):
        y = _year_from_path(net_path)
        if years is not None and y is not None and y not in years:
            return
        if not Path(net_path).is_file():
            report["warnings"].append(f"Network file missing: {net_path}")
            return

        if y is not None and isinstance(networks, list) and len(networks) > 1:
            per_net_out = base_out / str(y)
        else:
            per_net_out = base_out

        for c in c_list:
            try:
                ca = CountryAnalyzer(
                    network_path=net_path,
                    country=c,
                    output_dir=str(per_net_out / "country" / c),
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
            analyze_one(p, out_dir)
    elif isinstance(networks, dict):
        for branch, paths in networks.items():
            branch_out = out_dir / branch
            branch_out.mkdir(parents=True, exist_ok=True)
            for p in paths:
                analyze_one(p, branch_out)

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
        key_out = master.get_run_output_dir(key)
        try:
            if run_type == "aro":
                sub = run_aro(master, key_out, aro_run=key, countries=countries,
                              all_scenarios=all_scenarios,
                              run_standalone=run_standalone, standalone_keys=standalone_keys)
            else:
                sub = run_normal(master, key_out, scenario=key, countries=countries,
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
    ap.add_argument("--run-name",      default=None,
                    help="Run-Key überschreiben (bestimmt den Output-Unterordner)")
    args = ap.parse_args()

    master = MasterConfig()
    v      = validate_config(master, strict=args.strict)

    run_key = (
        args.run_name
        or args.aro_run
        or args.scenario
        or master.aro_selected_run
        if master.get_run_type() == "aro"
        else master.scenario_selection
    )
    out_dir = master.get_run_output_dir(run_key)
    write_report(out_dir, v, name="validation")

    print(f"[Output] Alle Ergebnisse in: {out_dir}")

    if args.mode == "validate":
        print(f"[OK={v['ok']}] Validierungsbericht: {out_dir}")
        return

    countries       = _parse_csv(args.countries)
    years           = _parse_int_list(args.years)
    run_standalone  = not args.no_standalone
    standalone_keys = args.standalone if args.standalone is not None else None

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
