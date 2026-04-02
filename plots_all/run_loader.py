#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_loader.py
=============
Generischer Netzwerk-Loader für beliebige Runs (normal ODER aro).

Liefert eine einheitliche Abstraktion `RunSpec` mit:
  - label    : Anzeigename im Plot
  - run_key  : Schlüssel im master_config.MASTER_CONFIG
  - run_type : "normal" | "aro"
  - networks : {year_or_tag: pypsa.Network}  (lazy-loaded)
  - network_paths : {year_or_tag: str}

Verwendung:
  from run_loader import RunLoader
  specs = RunLoader.from_keys(["my-normal-run", "my-aro-run"], labels=["Basis", "ARO"])
  for spec in specs:
      n = spec.get_network()   # erstes / einziges Netz
      n = spec.get_network(2050)

Das Modul importiert KEINE Matplotlib-Funktionen, ist also
für Berechnungen rein in sich geschlossen.

BUGFIXES (2026-04):
  - _resolve_aro_scenario_paths: Dispatch-Netze werden jetzt korrekt aus
    results/<run>/networks/dispatch/ geladen (dispatch_<safe_name>_std.nc)
    statt des alten Pfads results/<run>/networks/<sc>.nc  [BUG-1]
  - _safe_name-Logik kongruent mit solve_aro.py (replace '/', ' ') [BUG-2]
  - Glob-Fallback für Worst-Case (_worst_case_std.nc vor _std.nc) [BUG-3]
  - RunSpec.load() gibt nun auch bei leerem scenario-Set das robuste Netz zurück [BUG-4]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pypsa

from master_config import MasterConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _year_from_path(p: str) -> Optional[int]:
    """Extrahiert Jahreszahl aus Netzwerkpfad (z.B. base_s_24___2050.nc -> 2050)."""
    m = re.search(r"_(\d{4})\.nc$", p) or re.search(r"__(\d{4})\.nc$", p)
    return int(m.group(1)) if m else None


def _tag_from_path(p: str) -> str:
    """Fallback-Tag wenn kein Jahr erkennbar."""
    return Path(p).stem


def _safe_name(scenario_name: str) -> str:
    """
    Repliziert die safe_name-Logik aus solve_aro.py:
      cutout_name.replace("/", "_").replace(" ", "_")
    """
    return scenario_name.replace("/", "_").replace(" ", "_")


def _find_dispatch_for_scenario(
    dispatch_dir: Optional[Path],
    dispatch_tmp_dir: Optional[Path],
    scenario_name: str,
) -> Optional[Path]:
    """
    Sucht das Dispatch-Netzwerk für ein bestimmtes Szenario.

    Suchpfade (in dieser Reihenfolge):
    1. dispatch_dir / dispatch_{safe_name}_std.nc             (bevorzugt, persistentes dir)
    2. dispatch_dir / dispatch_{safe_name}_worst_case_std.nc
    3. dispatch_dir / dispatch_{safe_name}_flat.nc
    4. dispatch_dir / dispatch_{safe_name}*.nc                (glob-Fallback)
    5. dispatch_tmp_dir / dispatch_{safe_name}*.nc            (Rückwärtskompatibilität)

    Dateibenennungsschema aus solve_aro.py (Zeile 1025-1035):
      safe_name = cutout_name.replace("/", "_").replace(" ", "_")
      suffix = "_worst_case" if cutout_name == worst_final else ""
      dst = out_d / f"dispatch_{safe_name}{suffix}_std.nc"
    -> d.h. normales Szenario: dispatch_{safe_name}_std.nc
    -> worst case:             dispatch_{safe_name}_worst_case_std.nc
    """
    safe = _safe_name(scenario_name)

    # 1-4: persistentes dispatch/-Verzeichnis (Snakemake output.dispatch_dir)
    if dispatch_dir is not None and dispatch_dir.is_dir():
        # Direkte Treffer bevorzugen (ohne und mit _worst_case)
        for candidate_name in (
            f"dispatch_{safe}_std.nc",
            f"dispatch_{safe}_worst_case_std.nc",
            f"dispatch_{safe}_flat.nc",
            f"dispatch_{safe}_worst_case_flat.nc",
        ):
            p = dispatch_dir / candidate_name
            if p.is_file():
                return p
        # Glob-Fallback: _std.nc vor _flat.nc
        std_cands = sorted(dispatch_dir.glob(f"dispatch_*{safe}*_std.nc"))
        if std_cands:
            return std_cands[0]
        all_cands = sorted(dispatch_dir.glob(f"dispatch_*{safe}*.nc"))
        if all_cands:
            return all_cands[0]

    # 5: temporäres Verzeichnis (Rückwärtskompatibilität: _dispatch_tmp/final/)
    if dispatch_tmp_dir is not None and dispatch_tmp_dir.is_dir():
        for pattern in (f"dispatch_*{safe}*_std.nc", f"dispatch_*{safe}*.nc"):
            candidates = sorted(dispatch_tmp_dir.glob(pattern))
            if candidates:
                return candidates[0]

    return None


def _find_any_worst_case_dispatch(
    dispatch_dir: Optional[Path],
    dispatch_tmp_dir: Optional[Path],
    worst_cutout: Optional[str] = None,
) -> Optional[Path]:
    """
    Fallback: findet irgendeinen Worst-Case-Dispatch.
    Bevorzugt _worst_case_std.nc, dann _std.nc, dann alles.
    """
    if dispatch_dir is not None and dispatch_dir.is_dir():
        if worst_cutout:
            p = _find_dispatch_for_scenario(dispatch_dir, None, worst_cutout)
            if p:
                return p

        # Explizit _worst_case_std.nc suchen
        wc_std = sorted(dispatch_dir.glob("dispatch_*_worst_case_std.nc"))
        if wc_std:
            return wc_std[0]
        wc_flat = sorted(dispatch_dir.glob("dispatch_*_worst_case_flat.nc"))
        if wc_flat:
            return wc_flat[0]
        # Letzter Ausweg: irgendeinen _std.nc
        all_std = sorted(dispatch_dir.glob("dispatch_*_std.nc"))
        if all_std:
            return all_std[-1]
        all_nc = sorted(dispatch_dir.glob("dispatch_*.nc"))
        if all_nc:
            return all_nc[-1]

    if dispatch_tmp_dir is not None and dispatch_tmp_dir.is_dir():
        candidates = sorted(dispatch_tmp_dir.glob("dispatch_*.nc"))
        if worst_cutout:
            safe = _safe_name(worst_cutout)
            for p in candidates:
                if safe in p.name:
                    return p
        if candidates:
            return candidates[-1]

    return None


def _resolve_dispatch_dirs(
    master: MasterConfig,
    run_key: str,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Gibt (dispatch_dir, dispatch_tmp_dir) zurück.
    dispatch_dir      = results/<run>/networks/dispatch/
    dispatch_tmp_dir  = results/<run>/networks/_dispatch_tmp/final/
    """
    base = Path(master.aro_results_base)
    dispatch_dir = base / run_key / "networks" / "dispatch"
    dispatch_tmp = base / run_key / "networks" / "_dispatch_tmp" / "final"
    return (
        dispatch_dir if dispatch_dir.is_dir() else None,
        dispatch_tmp if dispatch_tmp.is_dir() else None,
    )


def _resolve_aro_scenario_paths(
    master: MasterConfig,
    run_key: str,
    include_robust: bool = True,
    include_scenarios: bool = True,
) -> Dict[str, str]:
    """
    Gibt Pfade für einen ARO-Run zurück.
    Keys: "robust" + aro_scenario-Namen (z.B. "cutout_rcp45_2050")

    BUG-FIX: Szenario-Dispatch-Pfade zeigen jetzt auf
      results/<run>/networks/dispatch/dispatch_<safe_name>_std.nc
    statt auf den falschen Pfad
      results/<run>/networks/<sc>.nc    <- das war der Bug
    """
    paths: Dict[str, str] = {}
    run_conf = master.aro_runs.get(run_key, {})

    # ---- Robustes Portfolio ----
    robust_raw = run_conf.get("robust_network")
    if include_robust and robust_raw:
        robust_path = master.resolve_template(robust_raw)
        if robust_path:
            paths["robust"] = robust_path

    # ---- Szenario-Dispatch-Netzwerke ----
    if include_scenarios:
        scenarios = master.get_aro_scenarios_for_run(run_key)
        dispatch_dir, dispatch_tmp_dir = _resolve_dispatch_dirs(master, run_key)

        for sc in scenarios:
            # Explizit konfigurierte Dispatch-Pfade (aus run_config["dispatch_paths"])
            explicit_paths = run_conf.get("dispatch_paths", {})
            explicit = explicit_paths.get(sc)
            if explicit and Path(explicit).is_file():
                paths[sc] = explicit
                continue

            # Auto-Suche im dispatch/-Verzeichnis (BUG-1 Fix)
            found = _find_dispatch_for_scenario(dispatch_dir, dispatch_tmp_dir, sc)
            if found:
                paths[sc] = str(found)
            else:
                # Pfad eintragen damit RunSpec das Szenario kennt;
                # "Datei fehlt"-Warnung kommt beim Laden via get_network()
                safe = _safe_name(sc)
                fallback = str(
                    Path(master.aro_results_base)
                    / run_key / "networks" / "dispatch"
                    / f"dispatch_{safe}_std.nc"
                )
                paths[sc] = fallback
                print(f"  [ARO] Kein Dispatch gefunden fuer '{sc}' - "
                      f"erwarteter Pfad: {fallback}")

    return paths


# ---------------------------------------------------------------------------
# RunSpec
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """
    Beschreibt einen einzelnen Run (normal oder aro) vollständig.

    Attributes
    ----------
    label        : Anzeigename (für Plot-Achsen/Legenden)
    run_key      : Schlüssel in master_config
    run_type     : "normal" | "aro"
    network_paths: Mapping {tag -> path}  (tag = Jahr (int) oder String)
    _cache       : interne Netzwerk-Cache
    """

    label: str
    run_key: str
    run_type: str  # "normal" | "aro"
    network_paths: Dict[Union[int, str], str] = field(default_factory=dict)
    _cache: Dict[Union[int, str], pypsa.Network] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Network access
    # ------------------------------------------------------------------

    def get_network(self, tag: Optional[Union[int, str]] = None) -> Optional[pypsa.Network]:
        """
        Lädt und cached das Netzwerk für `tag`.
        Falls tag None: erstes verfügbares Netzwerk.
        """
        if not self.network_paths:
            print(f"  [RunSpec '{self.label}'] Keine Netzwerkpfade verfügbar.")
            return None

        key = tag if tag is not None else next(iter(self.network_paths))
        if key not in self.network_paths:
            # Fuzzy-Fallback: Jahr als int/str
            for k in self.network_paths:
                if str(k) == str(key):
                    key = k
                    break
            else:
                print(f"  [RunSpec '{self.label}'] Tag '{tag}' nicht gefunden. "
                      f"Verfügbar: {list(self.network_paths.keys())}")
                return None

        if key not in self._cache:
            path = self.network_paths[key]
            if not Path(path).is_file():
                print(f"  [RunSpec '{self.label}'] Datei fehlt: {path}")
                return None
            print(f"  [RunSpec '{self.label}'] Lade Netzwerk [{key}]: {path}")
            self._cache[key] = pypsa.Network(path)

        return self._cache[key]

    def get_all_networks(self) -> Dict[Union[int, str], pypsa.Network]:
        """Lädt ALLE Netzwerke und gibt sie als Dict zurück."""
        result = {}
        for tag in self.network_paths:
            n = self.get_network(tag)
            if n is not None:
                result[tag] = n
        return result

    @property
    def tags(self) -> List[Union[int, str]]:
        return list(self.network_paths.keys())

    @property
    def years(self) -> List[int]:
        return sorted([t for t in self.tags if isinstance(t, int)])

    def __repr__(self) -> str:
        return (f"RunSpec(label='{self.label}', run_type='{self.run_type}', "
                f"tags={self.tags})")


# ---------------------------------------------------------------------------
# RunLoader
# ---------------------------------------------------------------------------

class RunLoader:
    """
    Fabrik-Klasse zum Erstellen von RunSpec-Objekten aus master_config.

    Unterstützt:
      - normale Runs (eine oder mehrere .nc-Dateien in scenarios.registry)
      - ARO-Runs    (robust + alle Worst-Case-Szenario-Netze)

    Beispiele
    ---------
    # Automatische Erkennung anhand run_type:
    specs = RunLoader.from_keys(
        keys=["basis-run", "aro-run-v2"],
        labels=["Basis", "ARO v2"],
    )

    # Explizit nur das robuste ARO-Netz (für Kapazitätsvergleich):
    specs = RunLoader.from_keys(
        keys=["aro-run-v2"],
        labels=["ARO (robust)"],
        aro_include_scenarios=False,
    )

    # Vollständig manuell:
    spec = RunLoader.from_paths(
        label="Custom",
        paths=["/path/to/network_2050.nc"],
    )
    """

    def __init__(self, master: Optional[MasterConfig] = None):
        self.master = master or MasterConfig()

    # ------------------------------------------------------------------
    # Public factory methods
    # ------------------------------------------------------------------

    def load(
        self,
        run_key: str,
        label: Optional[str] = None,
        aro_include_robust: bool = True,
        aro_include_scenarios: bool = True,
    ) -> RunSpec:
        """
        Erstellt einen RunSpec für run_key aus master_config.
        Erkennt run_type automatisch.
        """
        run_type = self.master.get_run_type(run_key)
        lbl = label or run_key

        if run_type == "aro":
            paths = _resolve_aro_scenario_paths(
                self.master, run_key,
                include_robust=aro_include_robust,
                include_scenarios=aro_include_scenarios,
            )
            # BUG-4 Fix: Wenn keine Szenario-Pfade gefunden aber robust vorhanden ->
            # trotzdem ein sinnvolles RunSpec liefern mit explizitem Hinweis
            if not paths and aro_include_robust:
                run_conf = self.master.aro_runs.get(run_key, {})
                robust_raw = run_conf.get("robust_network")
                if robust_raw:
                    robust_path = self.master.resolve_template(robust_raw)
                    if robust_path:
                        paths["robust"] = robust_path
                        print(f"  [RunLoader '{lbl}'] Nur robustes Netz verfügbar "
                              f"(keine Dispatch-Szenarien gefunden).")
        else:
            raw_paths = self.master.get_networks(run_key=run_key)
            raw_list = raw_paths if isinstance(raw_paths, list) else []
            paths = {}
            for p in raw_list:
                year = _year_from_path(p)
                tag = year if year is not None else _tag_from_path(p)
                paths[tag] = p

        return RunSpec(
            label=lbl,
            run_key=run_key,
            run_type=run_type,
            network_paths=paths,
        )

    def load_aro_robust_only(self, run_key: str, label: Optional[str] = None) -> RunSpec:
        """Kurzform: ARO-Run, nur robustes Netz (kein Worst-Case)."""
        return self.load(
            run_key, label=label or f"{run_key} (robust)",
            aro_include_robust=True,
            aro_include_scenarios=False,
        )

    @classmethod
    def from_keys(
        cls,
        keys: List[str],
        labels: Optional[List[str]] = None,
        master: Optional[MasterConfig] = None,
        aro_include_scenarios: bool = False,
    ) -> List[RunSpec]:
        """
        Bequeme Klassenmethode: erstellt RunSpec-Liste für mehrere Keys.

        Parameters
        ----------
        keys               : Run-Keys aus master_config
        labels             : Anzeigenamen (optional, default=key)
        aro_include_scenarios: Für ARO-Runs alle Worst-Case-Netze laden?
                              Default=False (nur robust) für Vergleichsplots.
        """
        loader = cls(master)
        lbl_list = labels or keys
        specs = []
        for key, lbl in zip(keys, lbl_list):
            spec = loader.load(
                key, label=lbl,
                aro_include_scenarios=aro_include_scenarios,
            )
            print(f"RunLoader: '{lbl}' ({spec.run_type}) — {len(spec.network_paths)} Netz(e)")
            specs.append(spec)
        return specs

    @staticmethod
    def from_paths(
        label: str,
        paths: List[str],
        run_type: str = "normal",
    ) -> RunSpec:
        """
        Erstellt RunSpec direkt aus Pfadliste (unabhängig von master_config).
        Nützlich für ad-hoc-Vergleiche ohne Config-Eintrag.
        """
        path_dict: Dict[Union[int, str], str] = {}
        for p in paths:
            year = _year_from_path(p)
            tag = year if year is not None else _tag_from_path(p)
            path_dict[tag] = p
        return RunSpec(label=label, run_key="_custom", run_type=run_type, network_paths=path_dict)

    @staticmethod
    def from_spec_dicts(
        spec_dicts: List[Dict],
        master: Optional[MasterConfig] = None,
    ) -> List[RunSpec]:
        """
        Erstellt RunSpec-Liste aus einer Liste von Dicts:
        [
          {"key": "my-normal", "label": "Basis"},
          {"key": "my-aro",    "label": "ARO",    "aro_include_scenarios": True},
          {"label": "Custom", "paths": ["/path/n.nc"]},
        ]
        """
        loader = RunLoader(master)
        specs = []
        for d in spec_dicts:
            if "paths" in d:
                specs.append(RunLoader.from_paths(
                    label=d.get("label", "custom"),
                    paths=d["paths"],
                    run_type=d.get("run_type", "normal"),
                ))
            else:
                specs.append(loader.load(
                    run_key=d["key"],
                    label=d.get("label"),
                    aro_include_scenarios=d.get("aro_include_scenarios", False),
                ))
        return specs
