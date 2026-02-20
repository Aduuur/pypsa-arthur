#!/usr/bin/env python3

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Sequence, Optional

import pypsa

try:

    import scripts.solve_robust as solve_robust  # type: ignore
except ImportError:
    # Fallback: solve_robust.py liegt im Stammverzeichnis
    import solve_robust as solve_robust  # type: ignore  # noqa:

logger = logging.getLogger(__name__)


def run_subprocess(cmd: List[str]) -> None:
    """Hilfsfunktion zum Ausführen eines Unterprozesses mit Fehlerbehandlung."""
    logger.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Command failed with return code %s", result.returncode)
        logger.error("Stdout: %s", result.stdout)
        logger.error("Stderr: %s", result.stderr)
        raise RuntimeError(f"Command '{cmd}' failed with code {result.returncode}")
    logger.debug("Stdout: %s", result.stdout)
    logger.debug("Stderr: %s", result.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iterativer ARO‑Workflow über Cutouts")
    p.add_argument(
        "--cutouts",
        nargs="+",
        required=True,
        help="Liste aller verfügbaren Cutouts (z. B. 2013 2014 2015)",
    )
    p.add_argument(
        "--scenario-network-template",
        required=True,
        help="Pfadvorlage für vorbereitete Netzwerke, z. B. networks/prepared_{cutout}.nc",
    )
    p.add_argument(
        "--initial-scenarios",
        nargs="+",
        required=True,
        help="Liste der Anfangsszenarien für den ARO‑Loop",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximale Anzahl an Iterationen im ARO‑Loop",
    )
    p.add_argument(
        "--out-network",
        required=True,
        help="Datei für das finale robuste Netz (.nc)",
    )
    p.add_argument(
        "--out-summary-json",
        required=True,
        help="Datei für die Zusammenfassung des ARO‑Workflows (.json)",
    )
    p.add_argument(
        "--solver-name",
        default="gurobi",
        help="Name des LP/MILP‑Solvers (z. B. gurobi, highs, cbc)",
    )
    p.add_argument(
        "--solver-options-json",
        default=None,
        help="JSON‑String mit Solver‑Optionen für PyPSA",
    )
    p.add_argument(
        "--eps-ls",
        type=float,
        default=1e-6,
        help="Toleranz für z_ls in solve_robust (Stage 2)",
    )
    return p.parse_args()


def load_network(network_path: str) -> pypsa.Network:
    """Lädt ein PyPSA‑Netzwerk von einer NetCDF‑Datei."""
    p = Path(network_path)
    if not p.exists():
        raise FileNotFoundError(f"Network file not found: {network_path}")
    return pypsa.Network(str(p))


def evaluate_all_cutouts(
    portfolio_path: str,
    cutouts: Sequence[str],
    scenario_template: str,
    solver_name: str,
    solver_options: Optional[Dict],
    eps_ls: float,
) -> Dict[str, float]:
    """Bewertet alle Cutouts mit festem Portfolio und gibt Kosten pro Cutout zurück.

    Für jedes Cutout wird ein Multi‑Cutout‑Netzwerk mit nur diesem einen Szenario
    erstellt, die Investment‑Kapazitäten aus dem Portfolio übertragen und das
    dispatch‑Problem gelöst. Die Kosten werden mit evaluate_scenario_costs
    berechnet.
    """
    costs: Dict[str, float] = {}
    for c in cutouts:
        # Pfad für vorbereitete Netzwerkdatei bestimmen
        scen_net_path = scenario_template.format(cutout=c)
        # Multi‑Szenario‑Netzwerk mit nur einem Cutout erstellen
        n, _ = solve_robust.stack_scenarios_to_multisnapshot_network(
            [scen_net_path], [c]
        )
        # Übertrage Investitionskapazitäten aus Portfolio auf dieses Netz
        port_net = load_network(portfolio_path)
        # Fixiere Kapazitäten für Generatoren, Links, StorageUnits und Stores
        for comp, attr_opt, attr_cap in [
            ("generators", "p_nom_opt", "p_nom"),
            ("links", "p_nom_opt", "p_nom"),
            ("storage_units", "p_nom_opt", "p_nom"),
            ("stores", "e_nom_opt", "e_nom"),
        ]:
            if hasattr(n, comp) and hasattr(port_net, comp):
                df_new = getattr(n, comp)
                df_old = getattr(port_net, comp)
                if attr_opt in df_old.columns:
                    for idx, cap in df_old[attr_opt].dropna().items():
                        if idx in df_new.index:
                            df_new.loc[idx, attr_cap] = cap
                            df_new.loc[idx, f"{attr_cap}_extendable"] = False
        # Löse robustes Problem für dieses einzelne Szenario (Dispatch mit fixen Investments)
        # Wir nutzen solve_robust.solve_robust_lexicographic, geben eps_ls als
        # absolute Toleranz an (eps_ls_abs). Die relative Toleranz belassen
        # wir beim Standardwert. co2_cost_mode bleibt im Standard.
        diag = solve_robust.solve_robust_lexicographic(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            eps_ls_abs=eps_ls,
        )
        # Die Stage‑2‑Kosten (z_cost_star) entsprechen den Gesamtkosten für dieses Szenario
        cost = diag.get("z_cost_star", float("nan"))
        costs[c] = cost
    return costs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    cutouts = list(args.cutouts)
    scenario_template = args.scenario_network_template
    solver_name = args.solver_name
    solver_options = None
    if args.solver_options_json:
        solver_options = json.loads(args.solver_options_json)

    initial = list(args.initial_scenarios)
    if any(x not in cutouts for x in initial):
        missing = [x for x in initial if x not in cutouts]
        raise ValueError(f"Initial scenarios not among cutouts: {missing}")

    remaining = [c for c in cutouts if c not in initial]

    current_set = initial.copy()
    iteration = 0
    history = []

    while iteration < args.max_iter and remaining:
        iteration += 1
        logger.info("=== ARO Iteration %d ===", iteration)
        logger.info("Current scenario set: %s", current_set)
        # Dateipfade für Zwischenergebnisse der robusten Optimierung
        tmp_net = Path(args.out_network).with_suffix("").as_posix() + f"_iter{iteration}.nc"
        tmp_sum = Path(args.out_summary_json).with_suffix("").as_posix() + f"_iter{iteration}.json"
        # Robust optimieren für die aktuelle Szenariomenge
        cmd = [
            "python",
            # solve_robust.py befindet sich im Stammverzeichnis des Projekts.
            # Sollte es in scripts liegen, wird der Importmechanismus oben
            # entsprechend angepasst. Der Aufruf per Subprozess muss hier den
            # tatsächlichen Pfad kennen. Wir nutzen daher den Dateinamen
            # solve_robust.py und verlassen uns darauf, dass die aktuelle
            # Arbeitsumgebung im Projekt-Stammverzeichnis liegt.
            "solve_robust.py",
            "--cutouts",
            *current_set,
            "--scenario-network-template",
            scenario_template,
            "--out-network",
            tmp_net,
            "--out-summary-json",
            tmp_sum,
            "--solver-name",
            solver_name,
            "--solver-options-json",
            json.dumps(solver_options) if solver_options else "null",
            "--eps-ls",
            str(args.eps_ls),
        ]
        run_subprocess(cmd)
        # Lade robustes Netz und berechne Kosten je Cutout
        all_costs = evaluate_all_cutouts(
            portfolio_path=tmp_net,
            cutouts=cutouts,
            scenario_template=scenario_template,
            solver_name=solver_name,
            solver_options=solver_options,
            eps_ls=args.eps_ls,
        )
        logger.info("Scenario costs: %s", all_costs)
        # Finde Worst‑Case unter den verbleibenden Cutouts
        worst_cutout = max(all_costs, key=all_costs.get)
        history.append(
            {
                "iteration": iteration,
                "current_set": current_set.copy(),
                "all_costs": all_costs,
                "worst_cutout": worst_cutout,
            }
        )
        if worst_cutout in current_set:
            logger.info(
                "Worst‑case cutout '%s' already in scenario set; stopping iteration.", worst_cutout
            )
            break
        # Andernfalls hinzufügen und weiter iterieren
        logger.info("Adding worst‑case cutout '%s' to scenario set.", worst_cutout)
        current_set.append(worst_cutout)
        remaining.remove(worst_cutout)

    # letzte robuste Optimierung mit endgültigem Szenariomenge
    logger.info("=== Final robust optimisation with scenarios: %s ===", current_set)
    final_cmd = [
        "python",
        "solve_robust.py",
        "--cutouts",
        *current_set,
        "--scenario-network-template",
        scenario_template,
        "--out-network",
        args.out_network,
        "--out-summary-json",
        args.out_summary_json,
        "--solver-name",
        solver_name,
        "--solver-options-json",
        json.dumps(solver_options) if solver_options else "null",
        "--eps-ls",
        str(args.eps_ls),
    ]
    run_subprocess(final_cmd)
    # Schreibe ARO‑History in die Summary
    try:
        with open(args.out_summary_json, "r") as f:
            summary = json.load(f)
    except Exception:
        summary = {}
    summary["aro_history"] = history
    with open(args.out_summary_json, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()