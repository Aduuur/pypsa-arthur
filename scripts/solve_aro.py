#!/usr/bin/env python3
# solve_aro.py
"""
solve_aro.py — Iterativer ARO-Workflow (Adaptive Robust Optimization via Szenario-Generierung)

Überblick
---------
Iterativer ARO-Loop über eine endliche Menge von Wetter-/Demand-Szenarien ("cutouts").
In jeder Iteration:

  (1) ROBUST-SOLVE:      Robuste Investitionsentscheidung über das aktuelle Szenarioset
                         via solve_robust.py (C&CG Master).
  (2) DISPATCH-EVAL:     Dispatch-only-Solve auf ALLEN Cutouts unter fixiertem Portfolio
                         (parallelisiert via ProcessPoolExecutor).
  (3) SZENARIO-AUSWAHL:  Worst-Case-Cutout wird anhand zweier Kriterien gewählt:
                         (a) "worst in set" (klassisches ARO-Kriterium, nur ohne Rotation)
                         (b) Gap-basierte Terminierung
  (4) SZENARIO-ROTATION: Masterproblem wird auf MAX_MASTER_SIZE Szenarien begrenzt.
                         FIFO-Eviction: ältestes Szenario wird verdrängt wenn das
                         Fenster voll ist. Verdrängte Szenarien kommen zurück in
                         'remaining' damit sie wieder evaluiert werden können.
                         RAM bleibt konstant bei k × n_T.
  (5) KONVERGENZ:        Stop wenn Gap < --convergence-tol (mit Rotation) ODER
                         Worst-Case bereits im Set (ohne Rotation).
  (6) FINAL-SOLVE:       Letzter Robust-Solve + finale Evaluation → Worst-Case-Dispatch.

Outputs
-------
  --out-network           Finales robustes Portfolio-Netz (flat-snapshot NetCDF).
  --out-dispatch-network  Worst-Case-Dispatch-Netz (flat-snapshot NetCDF).
  --out-dispatch-std      Worst-Case-Dispatch-Netz (DatetimeIndex NetCDF).
  --out-std-network       Standard-Adapter des Portfolio-Netzes (DatetimeIndex).
  --out-summary-json      ARO-History + Summary als JSON.

Changelog
---------
[FIX-1]   Worst-Case-Dispatch wird gespeichert.
[FIX-2]   Kostenkonsistenz Robust-Solve vs Dispatch-Evaluation.
[FIX-3]   --out-std-network nutzt den von solve_robust auto-erzeugten Adapter.
[FIX-4]   Finale Evaluation nach dem letzten Robust-Solve.
[FIX-CO2] CO2-Kostenmodus vollständig durchgereicht.

[NEU-1]  Parallelisierung der Dispatch-Evaluation via ProcessPoolExecutor.
[NEU-2]  Gap-basierte Konvergenzterminierung.
[NEU-3]  --ls-penalty weitergereicht an solve_robust und Dispatch-Evaluation.

[OPT-1]  Finale Evaluation cacht Ergebnis der letzten Iteration wenn Portfolio
         unverändert ist (MD5-Hash-Vergleich).
[OPT-2]  Temporäre Netzwerk-Dateien werden nach jeder Iteration aufgeräumt.
[OPT-3]  Python-Interpreter gecacht, PYTHONDONTWRITEBYTECODE gesetzt.

[ROT-1]  Szenario-Rotation mit festem Master-Fenster (--max-master-size, Default: 3).

         FIFO-Eviction: das älteste Szenario im Master wird verdrängt (nicht das
         billigste). Dies verhindert Zyklen, die beim Min-Cost-Ansatz entstehen können.

         Verdrängte Szenarien werden zurück in 'remaining' eingefügt, damit der
         Loop nicht zu früh terminiert wenn alle Cutouts mindestens einmal gesehen
         wurden aber noch nicht alle im finalen Master vertreten sind.

         Methodische Konsequenz:
           - Formale ARO-Garantie ist mit Rotation nicht mehr gegeben.
           - In der Praxis dominieren die k schlechtesten Szenarien die Investment-
             entscheidung. Für k=3 empirisch ausreichend.
           - Gap-Berechnung und Konvergenzcheck laufen über ALLE Cutouts
             (nicht nur das Master-Fenster) — Terminierung bleibt konservativ.
           - Konvergenzkriterium A ("worst in set") ist mit Rotation deaktiviert
             da das Szenario zwar im Master sein kann ohne dass das Portfolio
             optimal dagegen ist. Nur Kriterium B (Gap) terminiert mit Rotation.
           - History protokolliert master_set (aktiv im Solve) und all_seen
             (alle je hinzugefügten Szenarien) getrennt.

         Wann greift Rotation?
           - N_cutouts > max_master_size. Bei 2 Cutouts keine Wirkung.
           - Für 10-Cutout-Runs ist es der entscheidende RAM-Hebel.

Rolling-Window Bug Fixes (April 2026)
--------------------------------------
[ROT-FIX-1]  Evicted scenarios zurück in 'remaining'
    Vorher: verdrängte Szenarien wurden aus remaining entfernt und nie zurückgelegt.
    Loop terminierte zu früh wenn alle Cutouts einmal gesehen aber rotiert wurden.
    Fix: evicted scenario wird ans Ende von remaining appended.

[ROT-FIX-2]  FIFO statt Min-Cost Eviction
    Min-Cost-Eviction erzeugte Zyklen: unter Portfolio P1 (optimiert für {B,C,D})
    ist A teuer → wird added, B evicted. Unter P2 (für {C,D,A}) ist B teuer →
    wird added, C evicted. Kann zu master={A,B,C} → master={B,C,D} Zyklus führen.
    FIFO verhindert das, da jedes Szenario deterministisch nach max_master_size
    Iterationen rausfliegt.

[ROT-FIX-3]  Konvergenzkriterium A deaktiviert bei aktiver Rotation
    "worst_cutout in master_set" bedeutet mit Rotation nur, dass das Szenario noch
    nicht rausrotiert wurde — nicht dass das Portfolio dagegen optimal ist.
    Mit Rotation wird nur noch Kriterium B (Gap) für Terminierung verwendet.

[ROT-FIX-4]  while-Schleife hängt nicht mehr von 'remaining' ab
    Vorher: `while iteration < max_iter and remaining` terminierte zu früh wenn
    remaining nach Sichtung aller Cutouts leer war.
    Fix: Loop läuft bis max_iter, Terminierung nur durch Konvergenzcheck.

[OPT-FIX-1]  MD5-Hash statt Dateigröße für Cache-Invalidierung
    Zwei verschiedene Portfolios können identische Dateigrößen haben.
    MD5 ist zuverlässig.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pypsa

logger = logging.getLogger(__name__)

def _debug_cost_ranking(costs: Dict[str, float], stage: str) -> None:
    if not costs:
        logger.warning("[%s] No costs available.", stage)
        return
    ranking = sorted(costs.items(), key=lambda kv: kv[1], reverse=True)
    logger.info("[%s] Cost ranking (worst→best): %s",
                stage,
                " | ".join(f"{k}={v:.3e}" for k, v in ranking))


def _debug_master_membership(
        all_costs: Dict[str, float],
        master_set: List[str],
        stage: str,
) -> None:
    if not all_costs:
        return
    logger.info("[%s] Master set: %s", stage, master_set)
    for c in master_set:
        if c in all_costs:
            logger.info("[%s]   master cost: %s = %.6e", stage, c, all_costs[c])
        else:
            logger.warning("[%s]   master scenario '%s' missing in cost table.", stage, c)


def _debug_gap_details(
        all_costs: Dict[str, float],
        master_set: List[str],
        gap: float,
        worst_total: float,
        worst_in_master: float,
        stage: str,
) -> None:
    logger.info(
        "[%s] GAP details: gap=%.6f  worst_total=%.6e  worst_in_master=%.6e  diff=%.6e",
        stage, gap, worst_total, worst_in_master, worst_total - worst_in_master,
    )
    non_master = {k: v for k, v in all_costs.items() if k not in set(master_set)}
    if non_master:
        ranking = sorted(non_master.items(), key=lambda kv: kv[1], reverse=True)[:10]
        logger.info(
            "[%s] Worst outside master: %s",
            stage,
            " | ".join(f"{k}={v:.3e}" for k, v in ranking),
        )


def _debug_json_summary(tmp_sum: str, stage: str) -> None:
    p = Path(tmp_sum)
    if not p.exists():
        logger.warning("[%s] Summary JSON not found: %s", stage, tmp_sum)
        return
    try:
        with open(p, "r") as f:
            data = json.load(f)
        diag = data.get("diagnostics", {})
        rv = diag.get("robustness_verify", {})
        logger.info(
            "[%s] Robust summary: z_theta*=%.6e  n_scen=%s  annual_scale=%s  "
            "ls_penalty=%s  robustness_ok=%s  rel_gap=%s",
            stage,
            float(diag.get("z_theta_star", float("nan"))),
            diag.get("n_scenarios"),
            diag.get("annual_scale"),
            diag.get("ls_penalty"),
            rv.get("ok"),
            rv.get("rel_gap"),
        )
        caps = data.get("capacities", {})
        for comp in ["generators", "links", "storage_units", "stores"]:
            if comp in caps and caps[comp]:
                items = list(caps[comp].items())[:10]
                logger.info("[%s] capacities[%s] sample: %s", stage, comp, items)
    except Exception as exc:
        logger.warning("[%s] Could not parse summary JSON '%s': %s", stage, tmp_sum, exc)


def _debug_cache_decision(
        final_path: str,
        cached_path: str,
        use_cache: bool,
        stage: str,
) -> None:
    try:
        final_md5 = _file_md5(final_path) if Path(final_path).exists() else "missing"
        cache_md5 = _file_md5(cached_path) if Path(cached_path).exists() else "missing"
        logger.info(
            "[%s] Cache check: use_cache=%s  final=%s  cached=%s",
            stage, use_cache, final_md5, cache_md5,
        )
    except Exception as exc:
        logger.warning("[%s] Cache debug failed: %s", stage, exc)


# [OPT-3] Interpreter und Env einmal cachen
_PYTHON = sys.executable
_SUBPROCESS_ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


# =============================================================================
# Import solve_robust
# =============================================================================

try:
    import scripts.solve_robust as solve_robust  # type: ignore
except ImportError:
    try:
        import solve_robust as solve_robust  # type: ignore  # noqa
    except ImportError as exc:
        raise ImportError(
            "Could not import solve_robust. Ensure solve_robust.py is in 'scripts/' "
            "or the same directory as solve_aro.py."
        ) from exc


# =============================================================================
# I/O helpers
# =============================================================================

def _atomic_copy(src: str, dst: str) -> None:
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_p.with_name(dst_p.name + f".tmp.{uuid.uuid4().hex}")
    shutil.copyfile(str(src), str(tmp))
    os.replace(str(tmp), str(dst_p))


def _run_subprocess(cmd: List[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=_SUBPROCESS_ENV)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (code {result.returncode}): {' '.join(cmd)}"
        )


def _solver_options_arg(solver_options: Optional[Dict]) -> str:
    return json.dumps(solver_options) if solver_options is not None else "null"


def _resolve_std_network_path(robust_out_network: str) -> str:
    return str(Path(robust_out_network).with_suffix("")) + "__std.nc"


# =============================================================================
# [OPT-FIX-1] MD5-based cache invalidation
# =============================================================================

def _file_md5(path: str) -> str:
    """
    Compute MD5 hash of a file for reliable cache invalidation.
    More robust than file-size comparison: two different portfolios can have
    the same size when only float values differ.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# [OPT-2] Cleanup helper
# =============================================================================

def _cleanup_iter_files(*paths: str) -> None:
    for p in paths:
        try:
            path = Path(p)
            if path.is_file():
                path.unlink()
                logger.debug("Cleaned up file: %s", p)
            elif path.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                logger.debug("Cleaned up dir: %s", p)
        except Exception as exc:
            logger.warning("Could not clean up '%s': %s", p, exc)


# =============================================================================
# [ROT-1] Szenario-Rotation — FIFO Eviction
# =============================================================================

def _rotate_master_set(
        master_set: List[str],
        new_cutout: str,
        max_master_size: int,
) -> Tuple[List[str], Optional[str]]:
    """
    Add new_cutout to master_set using FIFO eviction.

    [ROT-FIX-2] FIFO replaces Min-Cost eviction to prevent cycling.
    Min-Cost eviction caused oscillation: under portfolio P1 (optimised for
    {B,C,D}), A is expensive → added, B evicted. Under P2 (for {C,D,A}),
    B is expensive → added, C evicted. This can lead to infinite cycles.

    FIFO is deterministic: the oldest scenario (index 0) is always evicted,
    so every scenario reappears in the master after at most max_master_size
    iterations regardless of cost values.

    new_cutout is never evicted in the same step it is added.

    Returns
    -------
    (new master_set, evicted scenario or None)
    """
    if new_cutout in master_set:
        return list(master_set), None

    new_set = master_set + [new_cutout]

    if len(new_set) <= max_master_size:
        return new_set, None

    # FIFO: evict oldest entry (index 0), never new_cutout itself
    for candidate in new_set:
        if candidate != new_cutout:
            evicted = candidate
            rotated = [c for c in new_set if c != evicted]
            logger.info(
                "[ROT-1] Master window full (%d/%d): FIFO evicting '%s' "
                "to make room for '%s'.",
                len(new_set), max_master_size, evicted, new_cutout,
            )
            return rotated, evicted

    # Fallback: all entries are new_cutout (shouldn't happen)
    return new_set[-max_master_size:], None


# =============================================================================
# [NEU-1] Dispatch worker — top-level, picklable
# =============================================================================

def _dispatch_worker_fn(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # solve_robust wird als Modul importiert → __name__="solve_robust" → explizit aktivieren
    logging.getLogger("solve_robust").setLevel(logging.INFO)
    cutout = kwargs.get("cutout", "?")
    logger.info("Worker starting for cutout '%s'.", cutout)
    try:
        result = solve_robust.evaluate_single_cutout_dispatch(**kwargs)
        logger.info(
            "Worker finished for cutout '%s' (total_cost=%.6g).",
            cutout, result.get("total_cost", float("nan")),
        )
        return result
    except Exception as exc:
        logger.error("Worker FAILED for cutout '%s': %s", cutout, exc, exc_info=True)
        raise


# =============================================================================
# [NEU-1] Parallelisierte Dispatch-Evaluation
# =============================================================================

def evaluate_all_cutouts(
        portfolio_path: str,
        cutouts: Sequence[str],
        scenario_template: str,
        solver_name: str,
        solver_options: Optional[Dict],
        dispatch_tmp_dir: str,
        *,
        cost_consistency_tol: float = 0.01,
        co2_cost_mode: str = "off",
        ls_penalty: float = 1e4,
        workers: int = 0,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, str]]:
    if not cutouts:
        raise ValueError("evaluate_all_cutouts: empty cutouts list.")

    Path(dispatch_tmp_dir).mkdir(parents=True, exist_ok=True)

    port_net = pypsa.Network(str(Path(portfolio_path)))
    investment_cost = solve_robust._compute_portfolio_investment_cost(port_net)
    logger.info("Fixed portfolio investment cost: %.6g €/a", investment_cost)
    del port_net

    n_workers = workers if workers > 0 else min(len(cutouts), os.cpu_count() or 1)
    logger.info(
        "Dispatch evaluation: %d cutout(s) with %d worker(s).",
        len(cutouts), n_workers,
    )

    job_kwargs = [
        {
            "cutout":               c,
            "portfolio_path":       portfolio_path,
            "scen_net_path":        scenario_template.format(cutout=c),
            "solver_name":          solver_name,
            "solver_options":       solver_options,
            "dispatch_tmp_dir":     dispatch_tmp_dir,
            "investment_cost":      investment_cost,
            "cost_consistency_tol": cost_consistency_tol,
            "co2_cost_mode":        co2_cost_mode,
            "ls_penalty":           ls_penalty,
        }
        for c in cutouts
    ]

    results: List[Dict[str, Any]] = []

    if n_workers == 1:
        logger.info("Running dispatch evaluation sequentially (workers=1).")
        for kw in job_kwargs:
            results.append(_dispatch_worker_fn(kw))
    else:
        futures_map: Dict[concurrent.futures.Future, str] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            for kw in job_kwargs:
                fut = pool.submit(_dispatch_worker_fn, kw)
                futures_map[fut] = kw["cutout"]
            for fut in concurrent.futures.as_completed(futures_map):
                c = futures_map[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    raise RuntimeError(
                        f"Dispatch evaluation FAILED for cutout '{c}'."
                    ) from exc

    costs:              Dict[str, float] = {}
    dispatch_paths:     Dict[str, str]   = {}
    dispatch_std_paths: Dict[str, str]   = {}

    for r in results:
        c = r["cutout"]
        costs[c]              = r["total_cost"]
        dispatch_paths[c]     = r["flat_path"]
        dispatch_std_paths[c] = r["std_path"]

        if not r["consistency"].get("consistent"):
            logger.warning(
                "Cutout '%s': cost-consistency FAILED "
                "(dispatch=%.6g manual=%.6g rel_diff=%.3f%%).",
                c,
                r["consistency"].get("pypsa_objective", float("nan")),
                r["consistency"].get("manual_cost", float("nan")),
                (r["consistency"].get("rel_diff") or 0.0) * 100,
            )

    logger.info(
        "[dispatch_eval_summary] completed %d cutouts. worst=%s  best=%s",
        len(costs),
        max(costs, key=costs.get) if costs else None,
        min(costs, key=costs.get) if costs else None,
    )
    _debug_cost_ranking(costs, "dispatch_eval_summary")
    return costs, dispatch_paths, dispatch_std_paths


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iterativer ARO-Workflow über Cutouts mit Szenario-Rotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cutouts", nargs="+", required=True,
                   help="Liste aller verfügbaren Cutouts.")
    p.add_argument("--scenario-network-template", required=True,
                   help="Pfadvorlage, z.B. networks/prepared_{cutout}.nc")
    p.add_argument("--initial-scenarios", nargs="+", required=True,
                   help="Anfangsszenarien (Subset von --cutouts).")
    p.add_argument("--max-iter", type=int, default=10,
                   help="Maximale Anzahl ARO-Iterationen.")
    p.add_argument("--out-network", required=True,
                   help="Finales robustes Portfolio-Netz (.nc).")
    p.add_argument("--out-summary-json", required=True,
                   help="ARO-Zusammenfassung (.json).")
    p.add_argument("--solver-name", default="gurobi")
    p.add_argument("--solver-options-json", default=None,
                   help="JSON-String mit Solver-Optionen.")

    # Outputs
    p.add_argument("--out-std-network", default=None,
                   help="[FIX-3] Standard-Adapter Portfolio (DatetimeIndex NetCDF).")
    p.add_argument("--out-dispatch-network", default=None,
                   help="[FIX-1] Worst-Case-Dispatch (flat-snapshot NetCDF).")
    p.add_argument("--out-dispatch-std", default=None,
                   help="[FIX-1] Worst-Case-Dispatch (DatetimeIndex NetCDF).")
    p.add_argument("--dispatch-tmp-dir", default=None,
                   help="Verzeichnis für temporäre Dispatch-Netze.")

    # Cost consistency
    p.add_argument("--cost-consistency-tol", type=float, default=0.01,
                   help="[FIX-2] Relative Toleranz Kostenkonsistenz (Default: 1%%).")

    # CO2
    p.add_argument(
        "--co2-cost-mode",
        choices=["off", "global_constraint_constant_cost"],
        default="off",
        help="[FIX-CO2] CO2-Kostenmodus, identisch zu solve_robust.",
    )

    # LS penalty
    p.add_argument(
        "--ls-penalty", type=float, default=1e4,
        help="[NEU-3] Load-shedding Strafkosten in EUR/MWh (Default: 1e4).",
    )

    # Parallelisation
    p.add_argument(
        "--dispatch-workers", type=int, default=0,
        help="[NEU-1] Parallele Worker. 0=auto. 1=sequenziell.",
    )

    # Gap convergence
    p.add_argument(
        "--convergence-tol", type=float, default=1e-4,
        help=(
            "[NEU-2] Relative Gap-Toleranz. ARO konvergiert wenn "
            "(worst_all - worst_in_master) / |worst_all| < tol. "
            "Default: 1e-4."
        ),
    )

    # [ROT-1] Master window
    p.add_argument(
        "--max-master-size", type=int, default=3,
        help=(
            "[ROT-1] Maximale Szenarien im C&CG-Master (Szenario-Rotation). "
            "FIFO-Eviction: ältestes Szenario wird verdrängt. "
            "0 = unbeschränkt (klassisches C&CG)."
        ),
    )

    return p.parse_args()


# =============================================================================
# [NEU-2] Gap computation
# =============================================================================

def _compute_aro_gap(
        all_costs: Dict[str, float],
        current_set: List[str],
) -> Tuple[float, float, float]:
    """
    Relative optimality gap:
        gap = (worst_total - worst_in_set) / max(1, |worst_total|)

    Returns (gap, worst_total, worst_in_set)
    """
    if not all_costs:
        return float("nan"), float("nan"), float("nan")

    worst_total  = max(all_costs.values())
    in_set_costs = {c: v for c, v in all_costs.items() if c in set(current_set)}
    worst_in_set = max(in_set_costs.values()) if in_set_costs else float("-inf")
    gap = (worst_total - worst_in_set) / max(1.0, abs(worst_total))
    return float(gap), float(worst_total), float(worst_in_set)


# =============================================================================
# Main ARO loop
# =============================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    cutouts           = list(dict.fromkeys(args.cutouts))
    scenario_template = args.scenario_network_template
    co2_cost_mode     = args.co2_cost_mode
    ls_penalty        = args.ls_penalty
    max_master_size   = args.max_master_size  # 0 = deaktiviert
    rotation_active   = max_master_size > 0

    robust_script = Path(__file__).with_name("solve_robust.py")
    if not robust_script.exists():
        alt = Path(__file__).resolve().parent / "solve_robust.py"
        if alt.exists():
            robust_script = alt
        else:
            raise FileNotFoundError(
                f"Could not locate solve_robust.py near {__file__}."
            )

    solver_name:    str            = args.solver_name
    solver_options: Optional[Dict] = None
    if args.solver_options_json:
        solver_options = json.loads(args.solver_options_json)

    initial = list(dict.fromkeys(args.initial_scenarios))
    unknown = [x for x in initial if x not in set(cutouts)]
    if unknown:
        raise ValueError(f"Initial scenarios not in --cutouts: {unknown}")

    master_set: List[str] = initial.copy()
    all_seen:   List[str] = initial.copy()
    remaining = [c for c in cutouts if c not in set(master_set)]

    # Initiales Master-Set auf max_master_size begrenzen falls nötig
    if max_master_size > 0 and len(master_set) > max_master_size:
        logger.warning(
            "[ROT-1] Initial set (%d) > --max-master-size (%d). Truncating.",
            len(master_set), max_master_size,
        )
        master_set = master_set[:max_master_size]

    dispatch_tmp_base = (
        args.dispatch_tmp_dir
        or str(Path(args.out_network).parent / "_dispatch_tmp")
    )

    iteration           = 0
    history: List[Dict] = []
    convergence_reason: Optional[str] = None

    # [OPT-1] Cache der letzten Dispatch-Evaluation
    last_iter_portfolio:          Optional[str]              = None
    last_iter_costs:              Optional[Dict[str, float]] = None
    last_iter_dispatch_paths:     Optional[Dict[str, str]]   = None
    last_iter_dispatch_std_paths: Optional[Dict[str, str]]   = None

    # =========================================================================
    # ARO Hauptloop
    #
    # [ROT-FIX-4] Loop-Bedingung hängt nicht mehr von 'remaining' ab.
    # Terminierung ausschließlich über Konvergenzcheck oder max_iter.
    # =========================================================================

    while iteration < args.max_iter:
        iteration += 1
        logger.info(
            "=== ARO Iteration %d/%d  |  master=%d(%s)  seen=%d  remaining=%d  "
            "max_master=%d  rotation=%s  ls=%.4g ===",
            iteration, args.max_iter,
            len(master_set), master_set,
            len(all_seen), len(remaining),
            max_master_size,
            "FIFO" if rotation_active else "off",
            ls_penalty,
        )

        tmp_net           = str(Path(args.out_network).with_suffix("")) + f"_iter{iteration}.nc"
        tmp_sum           = str(Path(args.out_summary_json).with_suffix("")) + f"_iter{iteration}.json"
        dispatch_tmp_iter = str(Path(dispatch_tmp_base) / f"iter{iteration}")

        # ------------------------------------------------------------------
        # Step 1: Robust-Solve über aktuelles master_set
        # ------------------------------------------------------------------
        logger.info("[Iter %d] Step 1: Robust-Solve over %s", iteration, master_set)
        cmd = [
            _PYTHON, str(robust_script),
            "--cutouts", *master_set,
            "--scenario-network-template", scenario_template,
            "--out-network", tmp_net,
            "--out-summary-json", tmp_sum,
            "--solver-name", solver_name,
            "--solver-options-json", _solver_options_arg(solver_options),
            "--co2-cost-mode", co2_cost_mode,
            "--ls-penalty", str(ls_penalty),
        ]
        _run_subprocess(cmd)

        if not Path(tmp_net).exists():
            raise RuntimeError(f"Robust solver did not produce: {tmp_net}")

        _debug_json_summary(tmp_sum, f"iter{iteration}_after_robust_solve")

        # ------------------------------------------------------------------
        # Step 2: Dispatch-Evaluation über ALLE Cutouts
        # ------------------------------------------------------------------
        logger.info(
            "[Iter %d] Step 2: Dispatch-Evaluation over ALL %d cutouts (workers=%d)",
            iteration, len(cutouts), args.dispatch_workers,
        )
        all_costs, iter_dispatch_paths, iter_dispatch_std_paths = evaluate_all_cutouts(
            portfolio_path=tmp_net,
            cutouts=cutouts,
            scenario_template=scenario_template,
            solver_name=solver_name,
            solver_options=solver_options,
            dispatch_tmp_dir=dispatch_tmp_iter,
            cost_consistency_tol=args.cost_consistency_tol,
            co2_cost_mode=co2_cost_mode,
            ls_penalty=ls_penalty,
            workers=args.dispatch_workers,
        )
        _debug_cost_ranking(all_costs, f"iter{iteration}_dispatch_eval")
        _debug_master_membership(all_costs, master_set, f"iter{iteration}_dispatch_eval")

        if not all_costs:
            raise RuntimeError("evaluate_all_cutouts returned empty costs dict.")

        worst_cutout = max(all_costs, key=all_costs.__getitem__)
        worst_cost   = all_costs[worst_cutout]

        gap, worst_total, worst_in_master = _compute_aro_gap(all_costs, master_set)

        _debug_gap_details(
            all_costs, master_set, gap, worst_total, worst_in_master,
            f"iter{iteration}_gap"
        )

        logger.info(
            "[Iter %d] Costs: %s",
            iteration,
            "  ".join(f"{c}={v:.3e}" for c, v in sorted(all_costs.items())),
        )
        logger.info(
            "[Iter %d] worst='%s' (%.6g)  worst_in_master=%.6g  gap=%.4f%%",
            iteration, worst_cutout, worst_total, worst_in_master, gap * 100,
        )

        history.append({
            "iteration":       int(iteration),
            "master_set":      list(master_set),
            "all_seen":        list(all_seen),
            "all_costs":       {k: float(v) for k, v in all_costs.items()},
            "worst_cutout":    str(worst_cutout),
            "worst_cost":      float(worst_cost),
            "worst_in_master": float(worst_in_master),
            "aro_gap":         float(gap),
            "convergence_tol": float(args.convergence_tol),
            "max_master_size": int(max_master_size),
            "rotation_active": rotation_active,
            "robust_network":  str(tmp_net),
            "co2_cost_mode":   co2_cost_mode,
            "ls_penalty":      ls_penalty,
        })

        # [OPT-1] Cache aktualisieren
        last_iter_portfolio          = tmp_net
        last_iter_costs              = all_costs
        last_iter_dispatch_paths     = iter_dispatch_paths
        last_iter_dispatch_std_paths = iter_dispatch_std_paths

        # ------------------------------------------------------------------
        # Step 3: Konvergenzcheck
        # ------------------------------------------------------------------

        # Kriterium A: Worst-Case bereits im aktiven Master
        # [ROT-FIX-3] Nur ohne Rotation verwenden — mit Rotation bedeutet
        # "worst in master" nicht, dass das Portfolio dagegen optimal ist.
        if not rotation_active and worst_cutout in master_set:
            convergence_reason = (
                f"worst_cutout='{worst_cutout}' already in master_set (no rotation)"
            )
            logger.info("[Iter %d] CONVERGED (A): %s.", iteration, convergence_reason)
            _cleanup_iter_files(tmp_sum)
            break

        if rotation_active and worst_cutout in master_set:
            logger.info(
                "[Iter %d] worst_cutout='%s' in master_set but rotation active "
                "— skipping criterion A, checking gap only.",
                iteration, worst_cutout,
            )

        # Info: schon gesehen aber rausrotiert
        if worst_cutout in all_seen and worst_cutout not in master_set:
            logger.info(
                "[Iter %d] Note: worst_cutout='%s' was seen before "
                "but rotated out of master. Re-adding.",
                iteration, worst_cutout,
            )

        # Kriterium B: Gap unter Toleranz
        if args.convergence_tol > 0 and gap < args.convergence_tol:
            convergence_reason = (
                f"gap={gap:.6f} < convergence_tol={args.convergence_tol} "
                f"(worst='{worst_cutout}' cost={worst_total:.6g}, "
                f"worst_in_master={worst_in_master:.6g})"
            )
            logger.info("[Iter %d] CONVERGED (B): %s.", iteration, convergence_reason)
            _cleanup_iter_files(tmp_sum)
            break

        # ------------------------------------------------------------------
        # Step 4: [ROT-1] Worst-Case-Szenario zum Master hinzufügen
        # ------------------------------------------------------------------

        # Aus remaining entfernen (falls noch drin)
        if worst_cutout in remaining:
            remaining.remove(worst_cutout)

        if max_master_size > 0:
            # [ROT-FIX-2] FIFO-Rotation
            master_set, evicted = _rotate_master_set(
                master_set, worst_cutout, max_master_size,
            )
            if evicted is not None:
                logger.info(
                    "[ROT-DEBUG] added='%s' cost=%.6e  evicted='%s' cost=%.6e  new_master=%s",
                    worst_cutout, all_costs.get(worst_cutout, float("nan")),
                    evicted, all_costs.get(evicted, float("nan")),
                    master_set,
                )
            else:
                logger.info(
                    "[ROT-DEBUG] added='%s' cost=%.6e  no eviction  new_master=%s",
                    worst_cutout, all_costs.get(worst_cutout, float("nan")),
                    master_set,
                )

            # [ROT-FIX-1] Evicted scenario zurück in remaining
            if evicted and evicted not in remaining:
                remaining.append(evicted)
                logger.info(
                    "[ROT-1] Evicted '%s' returned to remaining "
                    "(will be re-evaluated in future iterations).",
                    evicted,
                )
        else:
            # Rotation deaktiviert — klassisches C&CG
            if worst_cutout not in master_set:
                master_set.append(worst_cutout)
            evicted = None

        if worst_cutout not in all_seen:
            all_seen.append(worst_cutout)

        if evicted:
            logger.info(
                "[Iter %d] master_set after FIFO rotation: %s  (evicted: '%s')",
                iteration, master_set, evicted,
            )
        else:
            logger.info(
                "[Iter %d] master_set after update: %s", iteration, master_set,
            )

        # [OPT-2] Temp-Dateien dieser Iteration aufräumen
        # tmp_net bleibt (noch im Cache)
        _cleanup_iter_files(tmp_sum, dispatch_tmp_iter)

    else:
        convergence_reason = (
            f"max_iter={args.max_iter} reached without convergence"
        )
        logger.warning(
            "ARO reached max_iter=%d without convergence.", args.max_iter,
        )

    # =========================================================================
    # Finaler Robust-Solve
    # =========================================================================

    logger.info("=== Final robust solve (master_set: %s) ===", master_set)
    final_cmd = [
        _PYTHON, str(robust_script),
        "--cutouts", *master_set,
        "--scenario-network-template", scenario_template,
        "--out-network", args.out_network,
        "--out-summary-json", args.out_summary_json,
        "--solver-name", solver_name,
        "--solver-options-json", _solver_options_arg(solver_options),
        "--co2-cost-mode", co2_cost_mode,
        "--ls-penalty", str(ls_penalty),
    ]
    _run_subprocess(final_cmd)

    if not Path(args.out_network).exists():
        raise RuntimeError(
            f"Final robust solver did not produce: {args.out_network}"
        )

    # [OPT-2] Letztes tmp_net aufräumen
    if last_iter_portfolio and last_iter_portfolio != args.out_network:
        _cleanup_iter_files(last_iter_portfolio)

    # =========================================================================
    # [FIX-4] + [OPT-FIX-1] Finale Evaluation
    #
    # [OPT-FIX-1] Cache-Check via MD5-Hash statt Dateigröße (zuverlässiger).
    # =========================================================================

    _use_cache          = False
    dispatch_tmp_final: Optional[str] = None

    if (last_iter_costs is not None
            and last_iter_dispatch_paths is not None
            and last_iter_dispatch_std_paths is not None
            and last_iter_portfolio is not None
            and Path(last_iter_portfolio).exists()):

        try:
            final_md5 = _file_md5(args.out_network)
            cache_md5 = _file_md5(last_iter_portfolio)
            if final_md5 == cache_md5:
                _use_cache = True
                _debug_cache_decision(
                    args.out_network,
                    last_iter_portfolio,
                    _use_cache,
                    "final_cache_check",
                )
                logger.info(
                    "[OPT-1] Reusing cached dispatch results from last iteration "
                    "(MD5 match — portfolio unchanged, skipping redundant dispatch solve).",
                )
            else:
                logger.info(
                    "[OPT-1] Portfolio changed (MD5 mismatch) — "
                    "running fresh dispatch evaluation.",
                    _debug_cache_decision(
                        args.out_network,
                        last_iter_portfolio,
                        _use_cache,
                        "final_cache_check",
                    )
                )
        except Exception as exc:
            logger.warning("[OPT-1] MD5 check failed (%s) — running fresh dispatch.", exc)

    if _use_cache:
        final_costs              = last_iter_costs
        final_dispatch_paths     = last_iter_dispatch_paths
        final_dispatch_std_paths = last_iter_dispatch_std_paths
    else:
        logger.info(
            "=== Final evaluation: %d cutouts under final portfolio ===",
            len(cutouts),
        )
        dispatch_tmp_final = str(Path(dispatch_tmp_base) / "final")
        final_costs, final_dispatch_paths, final_dispatch_std_paths = evaluate_all_cutouts(
            portfolio_path=args.out_network,
            cutouts=cutouts,
            scenario_template=scenario_template,
            solver_name=solver_name,
            solver_options=solver_options,
            dispatch_tmp_dir=dispatch_tmp_final,
            cost_consistency_tol=args.cost_consistency_tol,
            co2_cost_mode=co2_cost_mode,
            ls_penalty=ls_penalty,
            workers=args.dispatch_workers,
        )

    worst_final      = max(final_costs, key=final_costs.__getitem__)
    worst_final_cost = final_costs[worst_final]
    final_gap, final_worst_total, final_worst_in_master = _compute_aro_gap(
        final_costs, master_set,
    )

    _debug_cost_ranking(final_costs, "final_evaluation")
    _debug_master_membership(final_costs, master_set, "final_evaluation")
    _debug_gap_details(
        final_costs, master_set, final_gap, final_worst_total, final_worst_in_master,
        "final_evaluation"
    )
    logger.info(
        "Final worst-case: '%s' (total_cost=%.6g)  gap=%.4f%%",
        worst_final, worst_final_cost, final_gap * 100,
    )

    # =========================================================================
    # [FIX-1] Export Worst-Case Dispatch
    # =========================================================================

    if args.out_dispatch_network:
        src = final_dispatch_paths.get(worst_final)
        if src and Path(src).exists():
            _atomic_copy(src, args.out_dispatch_network)
            logger.info("Worst-case dispatch (flat): -> %s", args.out_dispatch_network)
        else:
            logger.error(
                "Flat dispatch for '%s' not found at '%s'.", worst_final, src,
            )

    if args.out_dispatch_std:
        src = final_dispatch_std_paths.get(worst_final)
        if src and Path(src).exists():
            _atomic_copy(src, args.out_dispatch_std)
            logger.info("Worst-case dispatch (std): -> %s", args.out_dispatch_std)
        else:
            logger.error(
                "Std dispatch for '%s' not found at '%s'.", worst_final, src,
            )

    # =========================================================================
    # [FIX-3] Standard-Adapter Portfolio
    # =========================================================================

    if args.out_std_network:
        auto_std = _resolve_std_network_path(args.out_network)
        if Path(auto_std).exists():
            _atomic_copy(auto_std, args.out_std_network)
            logger.info("Portfolio std adapter: -> %s", args.out_std_network)
        else:
            logger.error(
                "[FIX-3] Expected std adapter not found: %s. "
                "Check that solve_robust.py is the patched version.", auto_std,
            )

    # =========================================================================
    # [OPT-2] Finale Cleanup
    # =========================================================================
    if dispatch_tmp_final:
        _cleanup_iter_files(dispatch_tmp_final)
    _cleanup_iter_files(dispatch_tmp_base)

    # =========================================================================
    # Summary JSON
    # =========================================================================

    try:
        with open(args.out_summary_json, "r") as f:
            summary = json.load(f)
    except Exception:
        summary = {}

    summary["aro_history"]          = history
    summary["aro_final_master_set"] = master_set
    summary["aro_all_seen_cutouts"] = all_seen
    summary["aro_convergence"] = {
        "reason":          convergence_reason,
        "iterations_run":  iteration,
        "max_iter":        args.max_iter,
        "convergence_tol": args.convergence_tol,
    }
    summary["aro_config"] = {
        "co2_cost_mode":    co2_cost_mode,
        "ls_penalty":       ls_penalty,
        "dispatch_workers": args.dispatch_workers,
        "max_master_size":  max_master_size,
        "rotation_active":  rotation_active,
        "eviction_strategy": "FIFO" if rotation_active else "disabled",
    }
    summary["aro_final_evaluation"] = {
        "all_costs":             {k: float(v) for k, v in final_costs.items()},
        "worst_case_cutout":     str(worst_final),
        "worst_case_total_cost": float(worst_final_cost),
        "aro_gap":               float(final_gap),
        "worst_in_master":       float(final_worst_in_master),
        "co2_cost_mode":         co2_cost_mode,
        "used_cache":            _use_cache,
    }
    summary["aro_outputs"] = {
        "robust_portfolio_network":        str(args.out_network),
        "robust_portfolio_std_network":    str(args.out_std_network) if args.out_std_network else None,
        "worst_case_dispatch_network":     str(args.out_dispatch_network) if args.out_dispatch_network else None,
        "worst_case_dispatch_std_network": str(args.out_dispatch_std) if args.out_dispatch_std else None,
    }

    with open(args.out_summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== ARO completed ===")
    logger.info("  Final master set:     %s", master_set)
    logger.info("  All seen cutouts:     %s", all_seen)
    logger.info("  Convergence:          %s", convergence_reason)
    logger.info("  Iterations run:       %d / %d", iteration, args.max_iter)
    logger.info("  Max master size:      %d (%s)",
                max_master_size,
                "FIFO rotation" if rotation_active else "disabled (classic C&CG)")
    logger.info("  Final gap:            %.4f%%", final_gap * 100)
    logger.info("  Worst-case cutout:    %s (%.6g EUR/a)", worst_final, worst_final_cost)
    logger.info("  CO2 cost mode:        %s", co2_cost_mode)
    logger.info("  LS penalty:           %.4g EUR/MWh", ls_penalty)
    logger.info("  Final eval cached:    %s", _use_cache)
    logger.info("  Portfolio network:    %s", args.out_network)
    if args.out_dispatch_network:
        logger.info("  Dispatch network:     %s", args.out_dispatch_network)
    logger.info("  Summary JSON:         %s", args.out_summary_json)


# =============================================================================
# Entry point guard — required for ProcessPoolExecutor on Windows/macOS (spawn)
# =============================================================================

if __name__ == "__main__":
    main()