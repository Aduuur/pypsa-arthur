#!/usr/bin/env python3
# solve_aro.py

#!/usr/bin/env python3
"""
solve_aro.py — Iterativer ARO-Workflow (Adaptive Robust Optimization via Szenario-Generierung)

Ziel
----
Dieses Skript implementiert einen iterativen ARO-Loop über eine endliche Menge von
Wetter-/Demand-Szenarien ("cutouts"). In jeder Iteration:

  (1) Löse ein robustes Portfolio-Problem (Investitionsentscheidung) auf dem aktuellen Szenarioset
      mittels scripts/solve_robust.py (lexikographisch: erst Worst-Case-LS, dann Worst-Case-Kosten).
  (2) Fixiere die resultierenden Investitionskapazitäten und evaluiere dieses Portfolio via
      Dispatch-only auf ALLEN Cutouts (jeweils getrennt, single-scenario).
      Die Dispatch-Kosten werden dabei mit der PyPSA-internen Zielfunktion berechnet.
  (3) Füge das Worst-Case-Cutout (höchste Gesamtkosten) zum Szenarioset hinzu.
  (4) Wiederhole bis Konvergenz oder max-iter.

Wichtig: Für die Dispatch-Evaluation nutzen wir explizit PyPSA's interne Kosten/Objective,
und addieren eine scenario-unabhängige Investitionskosten-Komponente aus dem Portfolio.

----------------------------------------------------------------------
Methodische Annahmen (explizit dokumentiert)
----------------------------------------------------------------------

A1) Two-stage Setup / "ARO via Szenario-Generierung"
    - Investitionsentscheidungen werden robust über ein Szenarioset bestimmt (solve_robust).
    - Robustheit gegen alle Cutouts wird iterativ durch Hinzufügen des Worst-Case-Szenarios erzwungen.

A2) "Fixed portfolio" Dispatch Evaluation
    - Die Investitionsentscheidung wird aus dem Portfolio-Netz (solve_robust Output) übernommen
      und im Dispatch-Netz fixiert:
        * Alle extendable-Flags werden auf False gesetzt.
        * Nennkapazitäten werden auf die optimierten Werte p_nom_opt / s_nom_opt / e_nom_opt gesetzt.
        * Falls p_nom_opt NaN oder Asset nicht in Portfolio: Kapazität = 0 (nicht gebaut).
      Dadurch gibt es im Dispatch keine Investitionsvariablen.

A3) Objective für Dispatch-only Evaluation
    - Es wird die PyPSA-interne Objective verwendet (n.optimize() ohne objective override).
    - Voraussetzung dafür: KEINE extendable Assets (A2) und damit keine Investitionsvariablen.
      Dann reduziert sich die Objective auf (gewichtete) variable Kosten + ggf. last shedding.
    - Zusätzlich wird die annualisierte Investitionskosten-Summe aus dem Portfolio
      (capital_cost * p_nom_opt / e_nom_opt / s_nom_opt) als konstante Komponente addiert.
      Diese Investitionskosten sind scenario-unabhängig und gehören zur Gesamtbewertung,
      wenn man Worst-Case Gesamtkosten über Cutouts vergleichen will.

A4) Multi-snapshot / Stacked Snapshots
    - Für einzelne Cutouts verwenden wir dennoch die solve_robust.stack_scenarios_to_multisnapshot_network
      Infrastruktur, um exakt dieselben Hilfsfunktionen für:
        * scenario boundary constraints (cyclic SOC je Szenario)
        * ramp constraints "within scenario"
      zu nutzen und Cross-Scenario-Artefakte zu vermeiden.

A5) Storage cyclic constraints
    - PyPSA's globale cyclic SOC wird deaktiviert, da sie bei stacked snapshots
      physikalisch falsche Kopplungen zwischen Szenarien erzeugt.
    - Stattdessen fügen wir per-scenario boundary constraints (solve_robust helper) hinzu.

A6) Feasibility / Load shedding
    - Für Dispatch-Evaluation fügen wir load-shedding Generatoren hinzu (idempotent),
      um Infeasibility zu vermeiden. Das stellt sicher, dass wir immer eine Lösung bekommen,
      und die Kosten (inkl. LS-Penalty) sind dann aussagekräftig.

A7) Kostenkonsistenz zwischen solve_robust und Dispatch
    - Wir verwenden in Dispatch die PyPSA-interne Objective.
    - Wenn solve_robust zusätzliche Kostenmodi (z.B. CO2) ein-/ausschaltet, müssen die
      zugrundeliegenden Netzwerkspezifikationen konsistent sein (carrier costs, marginal_cost etc.).
      Dieses Skript verändert keine Kostendaten, es übernimmt die Netzwerke "as is".

A8) Solver Objective Value Extraction
    - Je nach PyPSA/Linopy-Version ist die Objective-Zahl an unterschiedlichen Attributen verfügbar.
      Wir extrahieren robust aus:
        * model.objective_value
        * model.objective.value
        * model.solver_model.ObjVal (Gurobi)
      Falls nichts verfügbar: Fehler.

----------------------------------------------------------------------
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Import solve_robust helpers
# ---------------------------------------------------------------------
try:
    import scripts.solve_robust as solve_robust  # type: ignore
except ImportError:
    try:
        import solve_robust as solve_robust  # type: ignore  # noqa
    except ImportError as e:
        raise ImportError(
            "Could not import solve_robust. Make sure solve_robust.py is either in "
            "the 'scripts/' subdirectory or in the same directory as solve_aro.py."
        ) from e

# =============================================================================
# Subprocess helper
# =============================================================================

def run_subprocess(cmd: List[str]) -> None:
    """
    Run a subprocess, stream output for transparency, and error out on non-zero return code.
    """
    logger.info("Running command: %s", " ".join(cmd))
    # Stream stdout/stderr to console. This is much better for debugging solver logs.
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iterativer ARO-Workflow über Cutouts")
    p.add_argument(
        "--cutouts",
        nargs="+",
        required=True,
        help="Liste aller verfügbaren Cutouts (z.B. cutout_X cutout_Y ...)",
    )
    p.add_argument(
        "--scenario-network-template",
        required=True,
        help="Pfadvorlage für vorbereitete Netzwerke, z.B. networks/prepared_{cutout}.nc",
    )
    p.add_argument(
        "--initial-scenarios",
        nargs="+",
        required=True,
        help="Liste der Anfangsszenarien für den ARO-Loop (Subset von --cutouts).",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximale Anzahl an Iterationen im ARO-Loop",
    )
    p.add_argument(
        "--out-network",
        required=True,
        help="Datei für das finale robuste Netz (.nc)",
    )
    p.add_argument(
        "--out-summary-json",
        required=True,
        help="Datei für die Zusammenfassung des ARO-Workflows (.json)",
    )
    p.add_argument(
        "--solver-name",
        default="gurobi",
        help="Name des LP/MILP-Solvers (z. B. gurobi, highs, cbc)",
    )
    p.add_argument(
        "--solver-options-json",
        default=None,
        help="JSON-String mit Solver-Optionen für PyPSA/Linopy (z.B. '{\"threads\":8}')",
    )
    p.add_argument(
        "--eps-ls",
        type=float,
        default=1e-6,
        help="Toleranz für z_ls in solve_robust (Stage 2), wird als --eps-ls-abs weitergereicht",
    )
    return p.parse_args()


def load_network(network_path: str) -> pypsa.Network:
    p = Path(network_path)
    if not p.exists():
        raise FileNotFoundError(f"Network file not found: {network_path}")
    return pypsa.Network(str(p))


# =============================================================================
# Dispatch-only evaluation helpers
# =============================================================================

# Mapping: (component, opt capacity column, nominal capacity column, extendable flag column)
_CAPACITY_MAP: List[Tuple[str, str, str, str]] = [
    ("generators",    "p_nom_opt", "p_nom",  "p_nom_extendable"),
    ("links",         "p_nom_opt", "p_nom",  "p_nom_extendable"),
    ("storage_units", "p_nom_opt", "p_nom",  "p_nom_extendable"),
    ("stores",        "e_nom_opt", "e_nom",  "e_nom_extendable"),
    ("lines",         "s_nom_opt", "s_nom",  "s_nom_extendable"),
    ("transformers",  "s_nom_opt", "s_nom",  "s_nom_extendable"),
]


def _fix_portfolio_capacities(n: pypsa.Network, port_net: pypsa.Network) -> None:
    """
    Transfer optimized capacities from the portfolio network to a dispatch network and
    force all assets to non-extendable.

    - Assets present in both: set capacity = opt (NaN -> 0)
    - Assets only in dispatch network: capacity = 0
    - If portfolio lacks this component / opt column: capacity = 0 for all
    - Force extendable flags to False for all covered components
    """
    for comp, attr_opt, attr_cap, attr_ext in _CAPACITY_MAP:
        df_new = getattr(n, comp, None)
        df_old = getattr(port_net, comp, None)
        if df_new is None or len(df_new) == 0:
            continue

        # Force non-extendable if possible
        if attr_ext in df_new.columns:
            df_new[attr_ext] = False

        if df_old is None or attr_opt not in getattr(df_old, "columns", []):
            # No portfolio info -> zero out all
            if attr_cap in df_new.columns:
                df_new[attr_cap] = 0.0
            continue

        common = df_new.index.intersection(df_old.index)
        only_in_new = df_new.index.difference(df_old.index)

        if attr_cap in df_new.columns and len(only_in_new) > 0:
            df_new.loc[only_in_new, attr_cap] = 0.0

        if attr_cap in df_new.columns and len(common) > 0:
            opt_vals = df_old.loc[common, attr_opt].fillna(0.0).astype(float)
            df_new.loc[common, attr_cap] = opt_vals


def _assert_no_extendables(n: pypsa.Network) -> None:
    """
    Defensive check: ensure there are no extendable flags left True in the components we manage.
    If this fails, PyPSA may add investment variables/costs, and dispatch objective isn't "pure dispatch".
    """
    for comp, _attr_opt, _attr_cap, attr_ext in _CAPACITY_MAP:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0:
            continue
        if attr_ext not in df.columns:
            continue
        mask = df[attr_ext].fillna(False).astype(bool)
        if mask.any():
            bad = df.index[mask].tolist()[:10]
            raise RuntimeError(f"{comp}: still extendable assets detected (e.g. {bad}).")


def _compute_portfolio_investment_cost(port_net: pypsa.Network) -> float:
    """
    Compute scenario-independent annualized investment cost of the optimized portfolio:
    sum(capital_cost * capacity_opt) over all assets with capacity_opt > 0.
    """
    total = 0.0
    for comp, attr_opt, _attr_cap, _attr_ext in _CAPACITY_MAP:
        df = getattr(port_net, comp, None)
        if df is None or len(df) == 0:
            continue
        if "capital_cost" not in df.columns or attr_opt not in df.columns:
            continue

        cap = df[attr_opt].fillna(0.0).astype(float)
        invested = cap > 0.0
        if not invested.any():
            continue

        cc = df.loc[invested, "capital_cost"].fillna(0.0).astype(float)
        total += float((cc * cap[invested]).sum())

    return float(total)


def _extract_objective_value(n: pypsa.Network) -> float:
    """
    Robustly extract objective value from PyPSA/Linopy model after n.optimize().
    Supports multiple versions/backends.
    """
    model = getattr(n, "model", None)
    if model is None:
        raise RuntimeError("n.model is None after optimize().")

    # 1) linopy model: objective_value (some versions)
    obj = getattr(model, "objective_value", None)
    if obj is not None:
        return float(obj)

    # 2) linopy model: objective.value (other versions)
    try:
        return float(model.objective.value)  # type: ignore[attr-defined]
    except Exception:
        pass

    # 3) solver backend (e.g. gurobi)
    sm = getattr(model, "solver_model", None)
    if sm is not None:
        try:
            return float(sm.ObjVal)
        except Exception:
            pass

    raise RuntimeError("Could not extract objective value from model (unknown PyPSA/Linopy API).")


def _dispatch_solve(
    n: pypsa.Network,
    solver_name: str,
    solver_options: Optional[Dict],
    ramp_data: pd.DataFrame,
    scenarios: List[str],
    masks: Dict,
) -> float:
    """
    Dispatch-only solve with:
    - no investment variables (all assets fixed non-extendable)
    - per-scenario cyclic SOC constraints
    - within-scenario ramp constraints
    - load-shedding for feasibility

    Returns: PyPSA internal objective value (dispatch cost incl. LS penalties).
    """
    # (A5) Disable global cyclic constraints if present
    for comp in ("storage_units", "stores"):
        df = getattr(n, comp, None)
        if df is not None and len(df) > 0:
            for col in ("cyclic_state_of_charge", "cyclic_state_of_charge_per_period"):
                if col in df.columns:
                    df[col] = False

    # (A6) Ensure load shedding generators
    solve_robust._ensure_load_shedding_generators(n)

    # extra_functionality: scenario boundary + within-scenario ramps
    def extra_dispatch(network: pypsa.Network, snapshots: pd.Index) -> None:
        solve_robust._add_scenario_boundary_constraints(network, scenarios, masks)
        solve_robust._add_within_scenario_ramp_constraints(
            network,
            scenarios,
            masks,
            ramp_data,
            fail_on_extendable=False,  # dispatch has no extendables (asserted)
        )
        # IMPORTANT: No objective override -> PyPSA internal objective is used.

    n.optimize(
        solver_name=solver_name,
        solver_options=solver_options if solver_options is not None else {},
        extra_functionality=extra_dispatch,
    )

    # Status check (soft: rely on objective extraction; hard fail if clearly bad)
    model = getattr(n, "model", None)
    if model is None:
        raise RuntimeError("n.model is None after optimize().")

    status = str(getattr(model, "status", "unknown")).lower()
    if any(t in status for t in ("infeasible", "unbounded", "error", "failed")):
        raise RuntimeError(f"Dispatch solve failed with status '{status}'.")

    return _extract_objective_value(n)


def evaluate_all_cutouts(
    portfolio_path: str,
    cutouts: Sequence[str],
    scenario_template: str,
    solver_name: str,
    solver_options: Optional[Dict],
) -> Dict[str, float]:
    """
    Evaluate a fixed portfolio (from portfolio_path) on each cutout:
      total_cost(c) = investment_cost(portfolio) + dispatch_cost(c | fixed portfolio)

    Dispatch cost uses PyPSA internal objective (A3).
    """
    if not cutouts:
        raise ValueError("evaluate_all_cutouts received an empty cutouts list.")

    port_net = load_network(portfolio_path)
    investment_cost = _compute_portfolio_investment_cost(port_net)
    logger.info("Fixed portfolio investment cost: %.6g €/a", investment_cost)

    costs: Dict[str, float] = {}

    for c in cutouts:
        logger.info("--- Evaluating cutout '%s' under fixed portfolio ---", c)
        scen_net_path = scenario_template.format(cutout=c)
        if not Path(scen_net_path).exists():
            raise FileNotFoundError(f"Scenario network missing for cutout '{c}': {scen_net_path}")

        # Build stacked network (single scenario) for consistent helpers (A4)
        n, _ = solve_robust.stack_scenarios_to_multisnapshot_network([scen_net_path], [c])

        # Fix capacities and ensure no extendables (A2, A3)
        _fix_portfolio_capacities(n, port_net)
        _assert_no_extendables(n)

        # Prepare ramp data and scenario masks (A4)
        ramp_data = solve_robust._save_and_clear_ramp_limits(n)
        masks = solve_robust._scenario_masks_from_snapshots(n.snapshots)
        scenarios = list(masks.keys())

        op_cost = _dispatch_solve(n, solver_name, solver_options, ramp_data, scenarios, masks)
        total_cost = float(investment_cost) + float(op_cost)

        logger.info(
            "Cutout '%s': op_cost=%.6g  inv_cost=%.6g  total=%.6g",
            c, op_cost, investment_cost, total_cost,
        )
        costs[c] = float(total_cost)

    return costs


def _solver_options_arg(solver_options: Optional[Dict]) -> str:
    """
    Serialize solver options for passing to solve_robust.py via CLI JSON string.
    - If solver_options is None -> "null" (so downstream can parse JSON)
    - If solver_options is {} -> "{}" (forward empty dict explicitly)
    """
    return json.dumps(solver_options) if solver_options is not None else "null"


# =============================================================================
# Main ARO loop
# =============================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    cutouts = list(args.cutouts)
    scenario_template = args.scenario_network_template

    robust_solver_script = Path(__file__).with_name("solve_robust.py")
    if not robust_solver_script.exists():
        # also allow scripts/solve_robust.py next to repo root
        alt = Path(__file__).resolve().parent / "solve_robust.py"
        if alt.exists():
            robust_solver_script = alt
        else:
            raise FileNotFoundError(f"Could not locate robust solver script at {robust_solver_script}")

    solver_name = args.solver_name
    solver_options: Optional[Dict] = None
    if args.solver_options_json:
        solver_options = json.loads(args.solver_options_json)

    initial = list(args.initial_scenarios)
    if any(x not in cutouts for x in initial):
        missing = [x for x in initial if x not in cutouts]
        raise ValueError(f"Initial scenarios not among cutouts: {missing}")

    # Deduplicate while preserving order
    initial_set = list(dict.fromkeys(initial))
    current_set = initial_set.copy()

    # remaining = all cutouts not in current_set (dedup preserve)
    remaining = [c for c in dict.fromkeys(cutouts) if c not in set(current_set)]

    iteration = 0
    history: List[Dict] = []

    while iteration < args.max_iter and remaining:
        iteration += 1
        logger.info("=== ARO Iteration %d ===", iteration)
        logger.info("Current scenario set: %s", current_set)

        tmp_net = str(Path(args.out_network).with_suffix("")) + f"_iter{iteration}.nc"
        tmp_sum = str(Path(args.out_summary_json).with_suffix("")) + f"_iter{iteration}.json"

        # (1) Robust portfolio solve for current_set
        cmd = [
            sys.executable,
            str(robust_solver_script),
            "--cutouts", *current_set,
            "--scenario-network-template", scenario_template,
            "--out-network", tmp_net,
            "--out-summary-json", tmp_sum,
            "--solver-name", solver_name,
            "--solver-options-json", _solver_options_arg(solver_options),
            "--eps-ls-abs", str(args.eps_ls),
        ]
        run_subprocess(cmd)

        if not Path(tmp_net).exists():
            raise RuntimeError(
                f"Robust solver subprocess exited successfully but did not produce network file: {tmp_net}"
            )

        # (2) Evaluate fixed portfolio on all cutouts
        all_costs = evaluate_all_cutouts(
            portfolio_path=tmp_net,
            cutouts=cutouts,
            scenario_template=scenario_template,
            solver_name=solver_name,
            solver_options=solver_options,
        )

        if not all_costs:
            raise RuntimeError("evaluate_all_cutouts returned empty costs dict unexpectedly.")

        worst_cutout = max(all_costs, key=all_costs.get)
        history.append(
            {
                "iteration": int(iteration),
                "current_set": list(current_set),
                "all_costs": {k: float(v) for k, v in all_costs.items()},
                "worst_cutout": str(worst_cutout),
            }
        )

        if worst_cutout in current_set:
            logger.info("Worst-case cutout '%s' already in scenario set; converged.", worst_cutout)
            break

        logger.info("Adding worst-case cutout '%s' to scenario set.", worst_cutout)
        current_set.append(worst_cutout)
        if worst_cutout in remaining:
            remaining.remove(worst_cutout)

    # Final robust solve on converged set
    logger.info("=== Final robust optimisation with scenarios: %s ===", current_set)
    final_cmd = [
        sys.executable,
        str(robust_solver_script),
        "--cutouts", *current_set,
        "--scenario-network-template", scenario_template,
        "--out-network", args.out_network,
        "--out-summary-json", args.out_summary_json,
        "--solver-name", solver_name,
        "--solver-options-json", _solver_options_arg(solver_options),
        "--eps-ls-abs", str(args.eps_ls),
    ]
    run_subprocess(final_cmd)

    # Append ARO history to summary JSON (non-destructive)
    try:
        with open(args.out_summary_json, "r") as f:
            summary = json.load(f)
    except Exception:
        summary = {}

    summary["aro_history"] = history
    summary["aro_final_scenarios"] = current_set

    with open(args.out_summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("ARO completed. Final scenarios: %s", current_set)


if __name__ == "__main__":
    main()
