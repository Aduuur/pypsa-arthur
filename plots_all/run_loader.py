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


def _resolve_aro_scenario_paths(
    master: MasterConfig,
    run_key: str,
    include_robust: bool = True,
    include_scenarios: bool = True,
) -> Dict[str, str]:
    """
    Gibt Pfade für einen ARO-Run zurück.
    Keys: "robust" + aro_scenario-Namen (z.B. "cutout_rcp45_2050")
    """
    paths: Dict[str, str] = {}
    run_conf = master.aro_runs.get(run_key, {})

    robust_raw = run_conf.get("robust_network")
    if include_robust and robust_raw:
        robust_path = master.resolve_template(robust_raw)
        if robust_path:
            paths["robust"] = robust_path

    if include_scenarios:
        scenarios = master.get_aro_scenarios_for_run(run_key)
        results_base = Path(master.aro_results_base)
        for sc in scenarios:
            sc_path = str(results_base / run_key / "networks" / f"{sc}.nc")
            paths[sc] = sc_path

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
