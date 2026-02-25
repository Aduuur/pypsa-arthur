#!/usr/bin/env python3
# solve_robust.py
"""
Patched version of the robust solver from ``Aduuur/pypsa-arthur``.

Changelog vs. upstream (February 2026)
=======================================

[PATCH-1]  ``_ensure_standard_pypsa_result_frames``
    Do NOT create an empty ``buses_t.marginal_price`` DataFrame when duals are
    absent (e.g. MILP).  Left unset so downstream code can detect missing duals.

[PATCH-2]  ``export_network_stacked``
    Only reindex ``buses_t.marginal_price`` if it already exists as a DataFrame.

[PATCH-3]  ``export_network_standard_single_scenario``
    Same guard as [PATCH-2].

[PATCH-4]  ``_validate_dispatch_cost_consistency``  (NEW)
    Compare the operational cost reported by ``_dispatch_solve`` against
    ``_build_operational_cost_expression`` post-solve.  WARNING at > 1 % deviation.

[PATCH-5]  ``ls_penalty`` parameter  (NEW)
    ``solve_robust_lexicographic`` and ``run_robust`` now accept a configurable
    load-shedding penalty (default 1e4 €/MWh).  Exposed as ``--ls-penalty`` in
    the CLI.  Previously hard-coded.

[PATCH-6]  Dispatch helpers moved here from solve_aro.py  (NEW)
    ``_CAPACITY_MAP``, ``_fix_portfolio_capacities``, ``_assert_no_extendables``,
    ``_compute_portfolio_investment_cost``, ``_extract_objective_value``,
    ``_dispatch_solve``, and the high-level ``evaluate_single_cutout_dispatch``
    are now part of this module.  This makes them importable by
    ``solve_aro._dispatch_worker_fn`` in a spawned subprocess without having to
    import ``solve_aro`` itself (which would re-run its ``if __name__ == …``
    guard in unexpected ways on some platforms).

Design notes on duals / marginal prices
-----------------------------------------
``assign_all_duals=True`` is passed to both stages of
``solve_robust_lexicographic``.  The resulting ``buses_t.marginal_price``
contains nodal duals of the Minimax epigraph LP — NOT standard LMPs.  Use
dispatch-only solves (``solve_aro.py``) for market-grade marginal prices.

CO2 cost mode
--------------
``co2_cost_mode="global_constraint_constant_cost"`` adds a CO2 term via
``GlobalConstraint.constant_cost``.  This covers Generator-based emissions only.
If fossil conversion runs via Links (e.g. gas-bus → Link → power-bus), the CO2
term will be ZERO (not captured).  In that case, either keep ``co2_cost_mode="off"``
or extend ``_build_co2_cost_expression`` for your specific Link topology.

The double-counting heuristic (warn when fossil carrier + marginal_cost > 0) has
three known blind spots documented in ``_detect_possible_co2_double_counting``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

# =============================================================================
# Constants: explicit *_t attribute allow-list
# =============================================================================

_T_ATTRS: Dict[str, List[str]] = {
    "buses_t": ["p", "v_mag_pu", "v_ang", "marginal_price"],
    "generators_t": [
        "p", "p_max_pu", "p_min_pu", "marginal_cost",
        "marginal_cost_quadratic", "efficiency",
    ],
    "loads_t": ["p_set", "q_set"],
    "links_t": [
        "p0", "p1", "p2", "p3", "p4",
        "efficiency", "efficiency2", "efficiency3", "efficiency4",
        "marginal_cost", "p_min_pu", "p_max_pu",
    ],
    "lines_t": ["p0", "p1", "s_max_pu"],
    "transformers_t": ["p0", "p1", "s_max_pu"],
    "storage_units_t": [
        "p", "p_dispatch", "p_store", "state_of_charge",
        "inflow", "p_min_pu", "p_max_pu", "marginal_cost",
    ],
    "stores_t": ["p", "e", "e_min_pu", "e_max_pu", "marginal_cost"],
}

# =============================================================================
# [PATCH-6] Capacity map (moved from solve_aro, used by dispatch helpers)
# =============================================================================

# (component, opt capacity column, nominal capacity column, extendable flag column)
_CAPACITY_MAP: List[Tuple[str, str, str, str]] = [
    ("generators",    "p_nom_opt", "p_nom",  "p_nom_extendable"),
    ("links",         "p_nom_opt", "p_nom",  "p_nom_extendable"),
    ("storage_units", "p_nom_opt", "p_nom",  "p_nom_extendable"),
    ("stores",        "e_nom_opt", "e_nom",  "e_nom_extendable"),
    ("lines",         "s_nom_opt", "s_nom",  "s_nom_extendable"),
    ("transformers",  "s_nom_opt", "s_nom",  "s_nom_extendable"),
]


# =============================================================================
# General helpers
# =============================================================================

def _resolve_scenario_networks_from_cutouts(
        cutouts: Sequence[str], template: str
) -> List[str]:
    return [template.format(cutout=c) for c in cutouts]


def _ensure_datetime_snapshots(idx: pd.Index) -> pd.DatetimeIndex:
    out = idx if isinstance(idx, pd.DatetimeIndex) else pd.DatetimeIndex(idx)
    return out.rename("snapshot") if out.name != "snapshot" else out


def _weights_objective_series(n: pypsa.Network) -> pd.Series:
    sw = getattr(n, "snapshot_weightings", None)
    if sw is not None and "objective" in sw.columns:
        w = sw["objective"].copy()
        w.index = n.snapshots
        return w
    return pd.Series(1.0, index=n.snapshots)


def _mask_to_isel_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask.astype(bool))


# =============================================================================
# Time-dependent stacking helpers
# =============================================================================

def _list_time_dependent_frames(
        obj,
        container_name: str,
        *,
        base_index: pd.Index,
        warn_unknown: bool = True,
        strict_unknown: bool = False,
) -> Dict[str, pd.DataFrame]:
    allowed = set(_T_ATTRS.get(container_name, []))
    out: Dict[str, pd.DataFrame] = {}

    for attr in allowed:
        try:
            val = getattr(obj, attr)
        except AttributeError:
            continue
        if val is None:
            continue
        if isinstance(val, pd.Series):
            val = val.to_frame()
        if isinstance(val, pd.DataFrame) and not val.empty:
            out[attr] = val

    if warn_unknown or strict_unknown:
        unknown_hits: List[str] = []
        for attr in dir(obj):
            if attr.startswith("_") or attr in allowed:
                continue
            try:
                val = getattr(obj, attr)
            except Exception:
                continue
            if not isinstance(val, pd.DataFrame) or val.empty:
                continue
            try:
                if len(val.index.intersection(base_index)) == 0:
                    continue
            except Exception:
                pass
            unknown_hits.append(attr)

        if unknown_hits:
            msg = (
                f"Container '{container_name}' has DataFrame attribute(s) not in _T_ATTRS: "
                f"{unknown_hits}. They will NOT be stacked."
            )
            if strict_unknown:
                raise ValueError(msg)
            if warn_unknown:
                logger.warning(msg)

    return out


# =============================================================================
# Ramp limit handling
# =============================================================================

_RAMP_COLS = [
    "ramp_limit_up",
    "ramp_limit_down",
    "ramp_limit_start_up",
    "ramp_limit_shut_down",
]


def _save_and_clear_ramp_limits(n: pypsa.Network) -> pd.DataFrame:
    if not hasattr(n, "generators") or len(n.generators) == 0:
        return pd.DataFrame()

    saved = pd.DataFrame(index=n.generators.index)
    for col in _RAMP_COLS:
        if col in n.generators.columns:
            saved[col] = n.generators[col].copy()
            if n.generators[col].notna().any():
                logger.info("Saving and clearing '%s' from %d generators.", col,
                            int(n.generators[col].notna().sum()))
            n.generators[col] = np.nan
        else:
            saved[col] = np.nan

    if "committable" in n.generators.columns:
        saved["committable"] = n.generators["committable"].copy()
        if n.generators["committable"].fillna(False).any():
            logger.info("Saving and clearing 'committable' flags.")
        n.generators["committable"] = False
    else:
        saved["committable"] = False

    return saved


# =============================================================================
# Consistency checks
# =============================================================================

def _assert_same_static_assets_and_snapshots(networks: Sequence[pypsa.Network]) -> None:
    if len(networks) < 2:
        return
    ref = networks[0]

    def _check_component(name: str) -> None:
        if not hasattr(ref, name):
            return
        ref_df = getattr(ref, name)
        for i, net in enumerate(networks[1:], start=1):
            df = getattr(net, name)
            missing = ref_df.index.difference(df.index)
            extra = df.index.difference(ref_df.index)
            if len(missing) or len(extra):
                msg = f"Static asset mismatch in '{name}' (scenario 0 vs {i}):\n"
                if len(missing):
                    msg += f"  Missing: {list(missing)[:10]}\n"
                if len(extra):
                    msg += f"  Extra:   {list(extra)[:10]}\n"
                raise ValueError(msg)
            df_aligned = df.reindex(ref_df.index)
            for col in ["bus", "carrier", "p_nom_extendable", "capital_cost", "efficiency"]:
                if col in ref_df.columns and col in df_aligned.columns:
                    if not ref_df[col].equals(df_aligned[col]):
                        logger.warning("Static attribute differs: %s.%s scenario %d.", name, col, i)

    for comp in ["buses", "generators", "loads", "links", "lines",
                 "transformers", "storage_units", "stores"]:
        if hasattr(ref, comp):
            _check_component(comp)

    ref_snaps = _ensure_datetime_snapshots(ref.snapshots)
    for i, net in enumerate(networks[1:], start=1):
        snaps_i = _ensure_datetime_snapshots(net.snapshots)
        if not ref_snaps.equals(snaps_i):
            raise ValueError(
                f"Snapshot mismatch: scenario 0 vs {i}\n"
                f"  Scenario[0]: n={len(ref_snaps)}, {ref_snaps[0]} … {ref_snaps[-1]}\n"
                f"  Scenario[{i}]: n={len(snaps_i)}, {snaps_i[0]} … {snaps_i[-1]}\n"
            )


# =============================================================================
# Scenario stacking
# =============================================================================

def stack_scenarios_to_multisnapshot_network(
        scenario_files: Sequence[str],
        scenario_names: Sequence[str],
        *,
        warn_unknown_t: bool = True,
        strict_unknown_t: bool = False,
) -> Tuple[pypsa.Network, List[str]]:
    if len(scenario_files) != len(scenario_names):
        raise ValueError("scenario_files and scenario_names must have equal length.")
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError(f"Duplicate scenario names: {scenario_names}")

    nets: List[pypsa.Network] = []
    for s, f in zip(scenario_names, scenario_files):
        logger.info("Loading scenario=%s  network=%s", s, f)
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"Scenario network not found: {f}")
        net = pypsa.Network(str(p))
        if len(net.snapshots) == 0:
            raise ValueError(f"Scenario {s}: network has no snapshots.")
        if len(net.buses) == 0:
            raise ValueError(f"Scenario {s}: network has no buses.")
        nets.append(net)

    _assert_same_static_assets_and_snapshots(nets)

    ref = nets[0]
    base_snaps = _ensure_datetime_snapshots(ref.snapshots)
    n = ref.copy()

    period_value = 0
    if (hasattr(n, "investment_periods") and n.investment_periods is not None
            and len(n.investment_periods) > 0):
        period_value = int(list(n.investment_periods)[0])

    timestep_labels = [(scen, ts) for scen in scenario_names for ts in base_snaps]
    stacked_snaps = pd.MultiIndex.from_tuples(
        [(period_value, label) for label in timestep_labels],
        names=["period", "timestep"],
    )
    n.set_snapshots(stacked_snaps)

    if getattr(ref, "snapshot_weightings", None) is not None:
        w = ref.snapshot_weightings.copy().reindex(base_snaps)
        w_stacked = pd.concat([w] * len(scenario_names))
        w_stacked.index = stacked_snaps
        n.snapshot_weightings = w_stacked
    else:
        n.snapshot_weightings = pd.DataFrame(
            {"objective": 1.0, "generators": 1.0, "stores": 1.0},
            index=stacked_snaps,
        )

    for t_container_name in _T_ATTRS:
        if not hasattr(ref, t_container_name):
            continue
        ref_t = getattr(ref, t_container_name, None)
        if ref_t is None:
            continue

        frames = _list_time_dependent_frames(
            ref_t, t_container_name, base_index=base_snaps,
            warn_unknown=warn_unknown_t, strict_unknown=strict_unknown_t,
        )
        if not frames:
            continue

        n_t = getattr(n, t_container_name)
        for attr in frames:
            dfs: List[pd.DataFrame] = []
            for net in nets:
                net_t = getattr(net, t_container_name)
                df = getattr(net_t, attr, None)
                if df is None:
                    continue
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                dfs.append(df.reindex(base_snaps))
            if not dfs:
                continue
            df_stacked = pd.concat(dfs, axis=0)
            df_stacked.index = n.snapshots
            try:
                setattr(n_t, attr, df_stacked)
            except Exception as exc:
                logger.warning("Could not set %s.%s: %s", t_container_name, attr, exc)

    logger.info(
        "Stacked: %d scenarios × %d timesteps = %d snapshots | "
        "buses=%d gens=%d loads=%d links=%d lines=%d su=%d stores=%d",
        len(scenario_names), len(base_snaps), len(n.snapshots),
        len(n.buses), len(n.generators), len(n.loads),
        len(n.links), len(n.lines), len(n.storage_units), len(n.stores),
    )
    return n, list(scenario_names)


# =============================================================================
# Snapshot masks
# =============================================================================

def _scenario_masks_from_snapshots(snapshots: pd.Index) -> Dict[str, np.ndarray]:
    if not isinstance(snapshots, pd.MultiIndex):
        raise ValueError(f"Expected pd.MultiIndex snapshots, got {type(snapshots)}.")
    if list(snapshots.names) != ["period", "timestep"]:
        raise ValueError(f"Expected MultiIndex names ['period','timestep'], got {snapshots.names}.")
    ts = snapshots.get_level_values("timestep")
    bad = [x for x in ts if not (isinstance(x, tuple) and len(x) >= 2)]
    if bad:
        raise ValueError(
            "Expected 'timestep' to contain (scenario, timestamp) tuples. "
            f"Bad entries (first 10): {bad[:10]}"
        )
    scen_arr = np.asarray([x[0] for x in ts], dtype=object)
    return {str(s): (scen_arr == s) for s in pd.unique(scen_arr)}


# =============================================================================
# Load shedding
# =============================================================================

def _ensure_load_shedding_generators(
        n: pypsa.Network,
        *,
        carrier: str = "load_shedding",
        marginal_cost: float = 1e4,
        p_nom: float = None,  # ← None als Sentinel
) -> List[str]:
    # Berechne p_nom dynamisch falls nicht angegeben
    if p_nom is None:
        try:
            peak_load = n.loads_t.p_set.sum(axis=1).max()
            p_nom = max(peak_load * 2, 1e6)
        except Exception:
            p_nom = 1e6  # Fallback falls loads_t leer
    """Ensure one non-extendable load-shedding Generator per bus (idempotent)."""
    existing: List[str] = []
    if hasattr(n, "generators") and "carrier" in n.generators.columns:
        existing = n.generators.index[n.generators.carrier.astype(str) == carrier].tolist()

    if existing:
        logger.info("Found %d existing load-shedding generators.", len(existing))
        _ensure_ls_pmax_timeseries(n, existing)
        return existing

    if "load_shedding" not in n.carriers.index:
        n.add("Carrier", "load_shedding")

    logger.info("Adding load-shedding generators (one per bus, marginal_cost=%.4g).", marginal_cost)
    ls_names: List[str] = []
    for bus in n.buses.index:
        name = f"LS::{bus}"
        ls_names.append(name)
        n.add(
            "Generator", name, bus=bus, carrier=carrier,
            p_nom=p_nom, p_nom_extendable=False,
            marginal_cost=marginal_cost, efficiency=1.0, p_min_pu=0.0, p_max_pu=1.0,
        )

    _ensure_ls_pmax_timeseries(n, ls_names)
    return ls_names


def _ensure_ls_pmax_timeseries(n: pypsa.Network, ls_names: List[str]) -> None:
    try:
        pmax = getattr(n.generators_t, "p_max_pu", None)
    except Exception:
        pmax = None
    if pmax is None or not isinstance(pmax, pd.DataFrame):
        pmax = pd.DataFrame(index=n.snapshots)
    pmax = pmax.reindex(index=n.snapshots)
    for g in ls_names:
        if g not in pmax.columns:
            pmax[g] = 1.0
    n.generators_t.p_max_pu = pmax.fillna(1.0)


# =============================================================================
# Linopy helpers
# =============================================================================

def _get_linopy_var(model, key_candidates: Sequence[str], *, strict: bool = False):
    for k in key_candidates:
        if k in model.variables:
            return model.variables[k], k
    if not strict:
        return None, None
    available = list(model.variables.keys())
    suggestions: List[str] = []
    for c in key_candidates:
        suggestions.extend(difflib.get_close_matches(c, available, n=3, cutoff=0.6))
    hint = f" Did you mean: {sorted(set(suggestions))}?" if suggestions else ""
    raise KeyError(
        f"None of {list(key_candidates)} found in model.variables.{hint} "
        f"Available (head): {available[:50]}"
    )


def _get_time_dimension(var) -> str:
    for name in ("timestep", "snapshot", "snapshots", "time"):
        if name in getattr(var, "dims", ()):
            return name
    raise ValueError(f"No standard time dimension in dims {getattr(var, 'dims', None)}.")


def _get_gen_dimension(var) -> str:
    for name in ("Generator", "generator", "name"):
        if name in getattr(var, "dims", ()):
            return name
    td = _get_time_dimension(var)
    others = [d for d in var.dims if d != td]
    if len(others) == 1:
        return others[0]
    raise ValueError(f"Cannot infer generator dimension. dims={var.dims}")


# =============================================================================
# Solver status check
# =============================================================================

def _check_solver_status(n: pypsa.Network, stage: str, *, hard_fail_suboptimal: bool = True) -> bool:
    m = getattr(n, "model", None)
    if m is None:
        raise RuntimeError(f"[{stage}] n.model is None after optimize().")

    # --- Status und Termination ermitteln ---
    status_str: Optional[str] = None
    for attr in ("status", "termination_condition", "termination"):
        val = getattr(m, attr, None)
        if val is not None:
            status_str = str(val).lower().strip()
            break

    termination: Optional[str] = None
    for attr in ("termination_condition", "termination"):
        val = getattr(m, attr, None)
        if val is not None:
            termination = str(val).lower().strip()
            break

    sol = getattr(m, "solution", None)

    logger.info("[%s] Solver status: '%s'  termination: '%s'", stage, status_str, termination)

    if status_str is None:
        if sol is None:
            raise RuntimeError(f"[{stage}] No solution and no solver status available.")
        logger.warning("[%s] Solver status unavailable — solution exists, proceeding.", stage)
        return True

    # --- Infeasibility erkennen ---
    fatal_terms = ("infeasible", "unbounded", "error", "failed", "invalid")

    is_infeasible = (
        any(t in status_str for t in fatal_terms)
        or (termination is not None and any(t in termination for t in fatal_terms))
    )

    if is_infeasible:

        # --- IIS über Gurobi berechnen ---
        try:
            gurobi_model = m.solver_model
            if gurobi_model is not None:
                logger.error("[%s] Computing IIS (Irreducible Infeasible Subsystem)...", stage)
                gurobi_model.computeIIS()

                import numpy as np
                logger.error("[%s] Mapping IIS constraints to linopy names...", stage)
                for target_idx in [2191024, 54283919]:
                    found = False
                    for cname, constr in m.constraints.items():
                        try:
                            flat = constr.labels.values.flatten()
                            if target_idx in flat:
                                pos = np.where(flat == target_idx)
                                logger.error("  c%d → linopy constraint '%s' at pos %s",
                                             target_idx, cname, pos)
                                found = True
                                break
                        except Exception:
                            continue
                    if not found:
                        logger.error("  c%d → NOT FOUND in linopy constraints", target_idx)

                # Variable x2924968 mappen
                for vname, var in m.variables.items():
                    try:
                        flat = var.labels.values.flatten()
                        if 2924968 in flat:
                            pos = np.where(flat == 2924968)
                            logger.error("  x2924968 → linopy variable '%s' at pos %s", vname, pos)
                            break
                    except Exception:
                        continue

                ilp_path = f"/tmp/infeasible_{stage.replace(' ', '_').lower()}.ilp"
                gurobi_model.write(ilp_path)
                logger.error("[%s] IIS written to %s", stage, ilp_path)

                iis_constrs = [c.ConstrName for c in gurobi_model.getConstrs() if c.IISConstr]
                iis_lb      = [v.VarName for v in gurobi_model.getVars() if v.IISLB]
                iis_ub      = [v.VarName for v in gurobi_model.getVars() if v.IISUB]

                logger.error("[%s] IIS: %d conflicting constraints (first 30):", stage, len(iis_constrs))
                for c in iis_constrs[:30]:
                    logger.error("  CONSTR: %s", c)

                if iis_lb:
                    logger.error("[%s] IIS lower-bound violations (first 10):", stage)
                    for v in iis_lb[:10]:
                        logger.error("  LB: %s", v)

                if iis_ub:
                    logger.error("[%s] IIS upper-bound violations (first 10):", stage)
                    for v in iis_ub[:10]:
                        logger.error("  UB: %s", v)

                if not iis_constrs and not iis_lb and not iis_ub:
                    logger.error("[%s] IIS computed but empty — check %s", stage, ilp_path)
            else:
                logger.error("[%s] solver_model is None — cannot compute IIS.", stage)

        except AttributeError:
            logger.error("[%s] IIS not available (not Gurobi or model inaccessible).", stage)
        except Exception as exc:
            logger.error("[%s] IIS computation failed: %s", stage, exc, exc_info=True)

        # --- Netzwerk-Diagnose ---
        try:
            logger.error("[%s] Network diagnostics:", stage)
            logger.error("  Buses:         %d", len(n.buses))
            logger.error("  Generators:    %d  (extendable: %d)",
                         len(n.generators),
                         int(n.generators.p_nom_extendable.sum()))
            peak = n.loads_t.p_set.sum(axis=1).max() if len(n.loads_t.p_set) > 0 else float("nan")
            logger.error("  Loads peak:    %.1f MW", peak)
            ls = n.generators[n.generators.carrier == "load_shedding"]
            logger.error("  LS generators: %d  total p_nom: %.1f MW", len(ls), float(ls.p_nom.sum()))
            logger.error("  GlobalConstraints:")
            for gc_name, row in n.global_constraints.iterrows():
                logger.error("    %s  %s  %.3e", gc_name, row["sense"], row["constant"])
        except Exception as exc:
            logger.error("[%s] Network diagnostics failed: %s", stage, exc)

        raise RuntimeError(
            f"[{stage}] Infeasible (status='{status_str}', termination='{termination}'). "
            f"See IIS above or /tmp/infeasible_{stage.replace(' ', '_').lower()}.ilp"
        )

    # --- Optimal ---
    if status_str in ("ok", "optimal") or status_str.startswith("optimal"):
        return True

    # --- Suboptimal / Warning ---
    logger.warning("[%s] Non-optimal status: '%s'  termination: '%s'.", stage, status_str, termination)
    if hard_fail_suboptimal:
        raise RuntimeError(
            f"[{stage}] Aborting: non-optimal status '{status_str}'. "
            "Use --allow-suboptimal to proceed."
        )
    return False


# =============================================================================
# Boundary constraints
# =============================================================================

def _add_scenario_boundary_constraints(
        network: pypsa.Network,
        scenarios: Sequence[str],
        masks: Dict[str, np.ndarray],
) -> None:
    m = network.model

    # --- StorageUnits: nur cyclic_state_of_charge=True ---
    su_soc, su_name = _get_linopy_var(
        m, ["StorageUnit-state_of_charge", "StorageUnit-soc", "StorageUnit-energy"], strict=False,
    )
    if su_soc is not None:
        td = _get_time_dimension(su_soc)
        cyclic_sus = network.storage_units.index[
            network.storage_units.get("cyclic_state_of_charge", pd.Series(True, index=network.storage_units.index))
        ].tolist()
        # asset_dim ist der zweite Dim-Name (nicht Zeit)
        asset_dim = [d for d in su_soc.dims if d != td][0]
        for s in scenarios:
            idx = _mask_to_isel_indices(masks[s])
            if len(idx) >= 2:
                su_soc_cyclic = su_soc.sel({asset_dim: cyclic_sus})
                m.add_constraints(
                    su_soc_cyclic.isel({td: int(idx[0])}) == su_soc_cyclic.isel({td: int(idx[-1])}),
                    name=f"boundary::StorageUnit::cyclic::{s}",
                )
        logger.info(
            "StorageUnit cyclic SOC constraints added (var=%s, %d/%d cyclic).",
            su_name, len(cyclic_sus), len(network.storage_units),
        )

    # --- Stores: nur e_cyclic=True ---
    st_e, st_name = _get_linopy_var(
        m, ["Store-e", "Store-energy", "Store-state_of_charge"], strict=False,
    )
    if st_e is not None:
        td = _get_time_dimension(st_e)
        cyclic_stores = network.stores.index[
            network.stores.get("e_cyclic", pd.Series(True, index=network.stores.index))
        ].tolist()
        asset_dim = [d for d in st_e.dims if d != td][0]

        skipped = [s_name for s_name in network.stores.index if s_name not in cyclic_stores]
        if skipped:
            logger.info(
                "Store cyclic boundary skipped for %d non-cyclic stores: %s",
                len(skipped), skipped[:5],
            )

        for s in scenarios:
            idx = _mask_to_isel_indices(masks[s])
            if len(idx) >= 2:
                st_e_cyclic = st_e.sel({asset_dim: cyclic_stores})
                m.add_constraints(
                    st_e_cyclic.isel({td: int(idx[0])}) == st_e_cyclic.isel({td: int(idx[-1])}),
                    name=f"boundary::Store::cyclic::{s}",
                )
        logger.info(
            "Store cyclic energy constraints added (var=%s, %d/%d cyclic).",
            st_name, len(cyclic_stores), len(network.stores),
        )


# =============================================================================
# Ramp constraints
# =============================================================================

def _add_within_scenario_ramp_constraints(
        network: pypsa.Network,
        scenarios: Sequence[str],
        masks: Dict[str, np.ndarray],
        ramp_data: pd.DataFrame,
        *,
        fail_on_extendable: bool = True,
) -> None:
    if ramp_data is None or ramp_data.empty:
        return

    m = network.model
    gen_p, _ = _get_linopy_var(m, ["Generator-p"], strict=False)
    if gen_p is None:
        logger.info("No Generator-p variable; skipping ramp constraint rebuild.")
        return

    td = _get_time_dimension(gen_p)
    gen_dim = _get_gen_dimension(gen_p)
    gens_with_ramp = ramp_data.index[
        ramp_data[["ramp_limit_up", "ramp_limit_down"]].notna().any(axis=1)
    ]
    if len(gens_with_ramp) == 0:
        return

    ext_mask = (
        network.generators
        .get("p_nom_extendable", pd.Series(False, index=network.generators.index))
        .fillna(False).astype(bool)
    )
    bad_ext = network.generators.index[ext_mask].intersection(gens_with_ramp)
    if len(bad_ext) > 0:
        msg = f"Extendable generators with ramp limits: {list(bad_ext)}."
        if fail_on_extendable:
            raise RuntimeError(msg)
        logger.warning(msg + " Skipping (not recommended).")

    gens_to_constrain = network.generators.index[~ext_mask].intersection(gens_with_ramp)
    if len(gens_to_constrain) == 0:
        return

    n_added = 0
    for s in scenarios:
        idx = _mask_to_isel_indices(masks[s])
        if len(idx) < 2:
            continue
        prev, nxt = idx[:-1], idx[1:]
        pair_coord = np.arange(len(prev))

        for g in gens_to_constrain:
            p_nom_g = float(network.generators.loc[g, "p_nom"])
            rup = ramp_data.loc[g, "ramp_limit_up"]
            rdn = ramp_data.loc[g, "ramp_limit_down"]
            p_g = gen_p.sel({gen_dim: g})
            p_prev = p_g.isel({td: prev.tolist()}).assign_coords({td: pair_coord})
            p_next = p_g.isel({td: nxt.tolist()}).assign_coords({td: pair_coord})
            if not pd.isna(rup):
                m.add_constraints(p_next - p_prev <= float(rup) * p_nom_g, name=f"ramp_up::{g}::{s}")
                n_added += len(prev)
            if not pd.isna(rdn):
                m.add_constraints(p_prev - p_next <= float(rdn) * p_nom_g, name=f"ramp_down::{g}::{s}")
                n_added += len(prev)

    if n_added > 0:
        logger.info("Added %d within-scenario ramp constraints.", n_added)


# =============================================================================
# Cost expressions
# =============================================================================

def _build_ls_energy_expression(n: pypsa.Network, ls_generators: List[str], mask: np.ndarray):
    import xarray as xr
    m = n.model
    gen_p, gen_p_name = _get_linopy_var(m, ["Generator-p"], strict=True)
    gen_dim = _get_gen_dimension(gen_p)
    td = _get_time_dimension(gen_p)

    try:
        available = set(gen_p.coords[gen_dim].values.tolist())
    except Exception as exc:
        raise RuntimeError(f"Cannot access '{gen_dim}' coordinate on {gen_p_name}: {exc}") from exc

    missing = set(ls_generators) - available
    if missing:
        raise ValueError(f"Load-shedding generators not in model: {sorted(missing)}")

    idx = _mask_to_isel_indices(mask)
    gen_p_ls_s = gen_p.sel({gen_dim: ls_generators}).isel({td: idx})
    w = _weights_objective_series(n).to_numpy()[idx]
    w_da = xr.DataArray(w, dims=[td], coords={td: gen_p_ls_s.coords[td]})
    return (gen_p_ls_s * w_da).sum()


def _build_investment_cost_expression(n: pypsa.Network):
    import xarray as xr
    m = n.model
    expr = 0

    def _add(comp_df, var_names, dim):
        nonlocal expr
        if comp_df is None or len(comp_df) == 0 or "capital_cost" not in comp_df.columns:
            return
        var, _ = _get_linopy_var(m, var_names, strict=False)
        if var is None:
            return
        cc = comp_df["capital_cost"].reindex(comp_df.index).fillna(0.0)
        cc_da = xr.DataArray(cc.to_numpy(), dims=[dim], coords={dim: (dim, comp_df.index.to_numpy())})
        expr = expr + (var * cc_da).sum()

    _add(n.generators, ["Generator-p_nom"], "Generator")
    _add(n.links, ["Link-p_nom"], "Link")
    _add(n.storage_units, ["StorageUnit-p_nom"], "StorageUnit")
    _add(n.stores, ["Store-e_nom"], "Store")
    _add(n.lines, ["Line-s_nom"], "Line")
    _add(n.transformers, ["Transformer-s_nom"], "Transformer")
    return expr


def _detect_possible_co2_double_counting(n: pypsa.Network) -> None:
    """
    Heuristic warning for CO2 double-counting risk.

    Known blind spots (documented, not fixable without model-specific knowledge):
      (1) marginal_cost > 0 may represent O&M cost, not CO2.
      (2) Time-varying marginal_cost (*_t.marginal_cost) is not checked.
      (3) Carrier names are user-defined; the hard-coded set may miss variants.
    """
    if not hasattr(n, "generators") or "carrier" not in n.generators.columns:
        return
    if "marginal_cost" not in n.generators.columns:
        return

    fossil_like = {"coal", "lignite", "gas", "oil", "ccgt", "ocgt"}
    carriers = n.generators["carrier"].astype(str).str.lower()
    mc = n.generators["marginal_cost"].fillna(0.0).astype(float)

    mask = carriers.isin(fossil_like) & (mc > 0)
    if mask.any():
        top = n.generators.loc[mask, ["carrier", "marginal_cost"]].head(10)
        logger.warning(
            "CO2 cost mode enabled. Fossil carriers with positive marginal_cost detected — "
            "potential double counting.  Note: (1) marginal_cost may represent O&M, not CO2; "
            "(2) time-varying marginal_cost is not checked; (3) carrier names may vary. "
            "Example rows:\n%s", top.to_string(),
        )


def _build_co2_cost_expression(n: pypsa.Network, mask: np.ndarray) -> Any:
    """
    CO2 cost term via GlobalConstraint.constant_cost  (OPT-IN).

    LIMITATION: Only covers Generator-based emissions (Generator-p × carrier_emissions).
    If fossil conversion uses Links (e.g. gas-bus → Link → power-bus), emissions at those
    Links are NOT captured here.  In that case either keep co2_cost_mode="off" or extend
    this function for your specific Link topology.
    """
    import xarray as xr

    if not hasattr(n, "global_constraints") or len(n.global_constraints) == 0:
        return 0
    gc = n.global_constraints
    if "type" not in gc.columns or "constant_cost" not in gc.columns:
        return 0

    co2_rows = gc[
        gc["type"].astype(str).str.lower().isin(["primary_energy", "co2"])
        & gc["constant_cost"].fillna(0.0).gt(0)
    ]
    if co2_rows.empty:
        logger.warning("CO2 mode active but no matching GlobalConstraint rows. CO2 term = 0.")
        return 0

    m = n.model
    gen_p, _ = _get_linopy_var(m, ["Generator-p"], strict=False)
    if gen_p is None:
        return 0
    if "carrier" not in n.generators.columns:
        logger.warning("CO2: n.generators has no 'carrier' column. Skipping CO2 term.")
        return 0

    td = _get_time_dimension(gen_p)
    gen_dim = _get_gen_dimension(gen_p)
    idx = _mask_to_isel_indices(mask)
    w = _weights_objective_series(n).to_numpy()[idx]

    total_co2 = 0
    for _, row in co2_rows.iterrows():
        co2_price = float(row["constant_cost"])
        carrier_attr = row.get("carrier_attribute", "co2_emissions")
        if carrier_attr not in n.carriers.columns:
            logger.warning("CO2: carrier attribute '%s' not in n.carriers. Skipping.", carrier_attr)
            continue
        ef = n.generators["carrier"].map(n.carriers[carrier_attr]).fillna(0.0).astype(float)
        if (ef == 0).all():
            continue
        ef_da = xr.DataArray(ef.to_numpy(), dims=[gen_dim],
                             coords={gen_dim: (gen_dim, n.generators.index.to_numpy())})
        var_s = gen_p.isel({td: idx})
        w_da = xr.DataArray(w, dims=[td], coords={td: var_s.coords[td]})
        total_co2 = total_co2 + co2_price * (var_s * ef_da * w_da).sum()

    return total_co2


def _build_operational_cost_expression(
        n: pypsa.Network,
        mask: np.ndarray,
        *,
        co2_cost_mode: str = "off",
):
    """
    Canonical operational-cost expression — the single source of truth.

    Used identically in:
      - Stage 2 robust solve epigraph constraints
      - Dispatch-only objective override (when co2_cost_mode != "off")
      - Cost-consistency validation

    Covers: marginal_cost (static + time-varying), Link p0 convention,
    StorageUnit p_dispatch, UC costs, quadratic generator costs, CO2 (OPT-IN).
    """
    import xarray as xr

    m = n.model
    w = _weights_objective_series(n)
    idx = _mask_to_isel_indices(mask)
    total = 0

    def _var_isel(var, td_name):
        return var.isel({td_name: idx.tolist()}), idx.tolist()

    def _w_da(var_s, td_name, idx_list):
        return xr.DataArray(w.to_numpy()[idx_list], dims=[td_name],
                            coords={td_name: var_s.coords[td_name]})

    def _add_mc(comp_df, mc_t, var_names, dim):
        nonlocal total
        if comp_df is None or len(comp_df) == 0 or "marginal_cost" not in comp_df.columns:
            return
        var, _ = _get_linopy_var(m, var_names, strict=False)
        if var is None:
            return
        td = _get_time_dimension(var)
        var_s, idx_list = _var_isel(var, td)
        wd = _w_da(var_s, td, idx_list)
        mc_s = comp_df["marginal_cost"].reindex(comp_df.index).fillna(0.0).astype(float)
        if mc_t is not None and not mc_t.empty:
            mc_t_s = mc_t.reindex(index=n.snapshots[idx_list], columns=comp_df.index).astype(float)
            mc_t_s = mc_t_s.fillna(mc_s)
            mc_da = xr.DataArray(mc_t_s.to_numpy(), dims=[td, dim],
                                 coords={td: var_s.coords[td], dim: (dim, comp_df.index.to_numpy())})
        else:
            mc_da = xr.DataArray(mc_s.to_numpy(), dims=[dim],
                                 coords={dim: (dim, comp_df.index.to_numpy())})
        total = total + (var_s * mc_da * wd).sum()

    def _add_uc_costs():
        nonlocal total
        if not hasattr(n, "generators") or len(n.generators) == 0:
            return
        v_start, _ = _get_linopy_var(m, ["Generator-start_up", "Generator-startup", "Generator-start"], strict=False)
        v_shut, _ = _get_linopy_var(m, ["Generator-shut_down", "Generator-shutdown", "Generator-shut"], strict=False)
        v_status, _ = _get_linopy_var(m, ["Generator-status", "Generator-committable", "Generator-u"], strict=False)
        su_c = n.generators.get("start_up_cost", pd.Series(0.0, index=n.generators.index)).fillna(0.0).astype(float)
        sd_c = n.generators.get("shut_down_cost", pd.Series(0.0, index=n.generators.index)).fillna(0.0).astype(float)
        nl_c = n.generators.get("no_load_cost", pd.Series(0.0, index=n.generators.index)).fillna(0.0).astype(float)
        sb_c = n.generators.get("stand_by_cost", pd.Series(0.0, index=n.generators.index)).fillna(0.0).astype(float)
        if (su_c == 0).all() and (sd_c == 0).all() and (nl_c == 0).all() and (sb_c == 0).all():
            return

        def _add_uc(var, cs):
            nonlocal total
            if var is None:
                return
            td = _get_time_dimension(var)
            gd = _get_gen_dimension(var)
            var_s, il = _var_isel(var, td)
            wd = _w_da(var_s, td, il)
            c = cs.reindex(n.generators.index).fillna(0.0).astype(float)
            c_da = xr.DataArray(c.to_numpy(), dims=[gd], coords={gd: (gd, n.generators.index.to_numpy())})
            total = total + (var_s * c_da * wd).sum()

        _add_uc(v_start, su_c)
        _add_uc(v_shut, sd_c)
        _add_uc(v_status, nl_c)
        _add_uc(v_status, sb_c)

    def _add_quadratic():
        nonlocal total
        if not hasattr(n, "generators") or len(n.generators) == 0:
            return
        gen_p, _ = _get_linopy_var(m, ["Generator-p"], strict=False)
        if gen_p is None:
            return
        q_s = None
        if "marginal_cost_quadratic" in n.generators.columns:
            q_s = n.generators["marginal_cost_quadratic"].fillna(0.0).astype(float)
        q_t = getattr(getattr(n, "generators_t", None), "marginal_cost_quadratic", None)
        has_q = (q_s is not None and (q_s != 0).any()) or (
            isinstance(q_t, pd.DataFrame) and not q_t.empty
            and (q_t.fillna(0.0).to_numpy() != 0.0).any()
        )
        if not has_q:
            return
        td = _get_time_dimension(gen_p)
        gd = _get_gen_dimension(gen_p)
        var_s, il = _var_isel(gen_p, td)
        wd = _w_da(var_s, td, il)
        base = (q_s if q_s is not None else pd.Series(0.0, index=n.generators.index))
        base = base.reindex(n.generators.index).fillna(0.0)
        if isinstance(q_t, pd.DataFrame) and not q_t.empty:
            q_ts = q_t.reindex(index=n.snapshots[il], columns=n.generators.index).astype(float).fillna(base)
        else:
            q_ts = pd.DataFrame(np.tile(base.to_numpy(), (len(il), 1)),
                                index=n.snapshots[il], columns=n.generators.index)
        q_da = xr.DataArray(q_ts.to_numpy(), dims=[td, gd],
                            coords={td: var_s.coords[td], gd: (gd, n.generators.index.to_numpy())})
        total = total + ((var_s ** 2) * q_da * wd).sum()

    _add_mc(n.generators, getattr(getattr(n, "generators_t", None), "marginal_cost", None), ["Generator-p"], "Generator")
    _add_mc(n.storage_units, getattr(getattr(n, "storage_units_t", None), "marginal_cost", None), ["StorageUnit-p_dispatch", "StorageUnit-p"], "StorageUnit")
    _add_mc(n.stores, getattr(getattr(n, "stores_t", None), "marginal_cost", None), ["Store-p"], "Store")
    _add_mc(n.links, getattr(getattr(n, "links_t", None), "marginal_cost", None), ["Link-p0"], "Link")
    _add_uc_costs()
    _add_quadratic()

    if co2_cost_mode == "global_constraint_constant_cost":
        total = total + _build_co2_cost_expression(n, mask)
    elif co2_cost_mode != "off":
        raise ValueError(f"Unknown co2_cost_mode='{co2_cost_mode}'.")

    return total


def _evaluate_scenario_costs(
        n: pypsa.Network, masks: Dict[str, np.ndarray], co2_cost_mode: str,
) -> Dict[str, float]:
    inv_cost = float(_build_investment_cost_expression(n).evaluate())
    return {
        s: inv_cost + float(_build_operational_cost_expression(n, mask, co2_cost_mode=co2_cost_mode).evaluate())
        for s, mask in masks.items()
    }


# =============================================================================
# [PATCH-4] Cost-consistency validation
# =============================================================================

def _validate_dispatch_cost_consistency(
        n: pypsa.Network,
        pypsa_objective: float,
        mask: np.ndarray,
        scenario: str,
        *,
        rel_tol: float = 0.01,
        co2_cost_mode: str = "off",
) -> Dict[str, Any]:
    """
    Compare the cost reported by ``_dispatch_solve`` against a fresh evaluation of
    ``_build_operational_cost_expression``.

    When co2_cost_mode != "off", pypsa_objective is already the evaluated manual
    expression (not PyPSA's native ObjVal), so this check validates numerical
    stability of the linopy expression rather than formula identity.

    Returns dict: pypsa_objective, manual_cost, rel_diff, consistent.
    """
    result: Dict[str, Any] = {
        "pypsa_objective": float(pypsa_objective),
        "manual_cost": None,
        "rel_diff": None,
        "consistent": None,
    }
    try:
        manual = float(
            _build_operational_cost_expression(n, mask, co2_cost_mode=co2_cost_mode).evaluate()
        )
        result["manual_cost"] = manual
        rel_diff = abs(pypsa_objective - manual) / max(1.0, abs(pypsa_objective))
        result["rel_diff"] = rel_diff
        result["consistent"] = rel_diff < rel_tol

        if rel_diff >= rel_tol:
            logger.warning(
                "[CostConsistency] MISMATCH scenario='%s': "
                "dispatch=%.6g manual=%.6g rel_diff=%.3f%% (tol=%.1f%%). "
                "Likely causes: link direction convention, quadratic costs, UC costs.",
                scenario, pypsa_objective, manual, rel_diff * 100, rel_tol * 100,
            )
        else:
            logger.info("[CostConsistency] OK scenario='%s': rel_diff=%.4f%%.", scenario, rel_diff * 100)
    except Exception as exc:
        logger.warning("[CostConsistency] Validation failed for '%s' (non-fatal): %s", scenario, exc)

    return result


# =============================================================================
# Solution helpers
# =============================================================================

def _get_solution_dict(n: pypsa.Network) -> Dict:
    m = getattr(n, "model", None)
    if m is None:
        raise RuntimeError("n.model is None.")
    sol = getattr(m, "solution", None)
    if sol is None:
        raise RuntimeError("n.model.solution is None.")
    return sol


def _extract_scalar_solution(sol: Dict, key: str) -> float:
    if key not in sol:
        raise KeyError(f"Solution missing '{key}'. Available (head): {list(sol.keys())[:50]}")
    v = sol[key]
    try:
        return float(getattr(v, "item", lambda: v)())
    except Exception:
        return float(np.asarray(v).item())


# =============================================================================
# [PATCH-1] Standard result frames
# =============================================================================

def _ensure_standard_pypsa_result_frames(n: pypsa.Network) -> None:
    """[PATCH-1] Only enforce marginal_price shape if it already exists."""
    if not hasattr(n, "buses_t"):
        return
    try:
        mp = getattr(n.buses_t, "marginal_price", None)
    except Exception:
        mp = None
    if mp is None or not isinstance(mp, pd.DataFrame):
        logger.debug("marginal_price absent; leaving unset.")
        return
    n.buses_t.marginal_price = mp.reindex(index=n.snapshots, columns=n.buses.index)


# =============================================================================
# [PATCH-6] Capacity helpers (moved from solve_aro — needed by dispatch worker)
# =============================================================================

def _fix_portfolio_capacities(n: pypsa.Network, port_net: pypsa.Network) -> None:
    """
    Transfer optimized capacities from portfolio network to dispatch network
    and force all components to non-extendable.

    Rules (per component):
      - Present in both: capacity = opt_value (NaN → 0).
      - Only in dispatch:  capacity = 0 (asset not in portfolio).
      - Portfolio has no opt column: capacity = 0 for all.
      - Extendable flag: always False.
    """
    for comp, attr_opt, attr_cap, attr_ext in _CAPACITY_MAP:
        df_new = getattr(n, comp, None)
        df_old = getattr(port_net, comp, None)
        if df_new is None or len(df_new) == 0:
            continue
        if attr_ext in df_new.columns:
            df_new[attr_ext] = False
        if df_old is None or attr_opt not in getattr(df_old, "columns", []):
            if attr_cap in df_new.columns:
                df_new[attr_cap] = 0.0
            continue
        common = df_new.index.intersection(df_old.index)
        only_new = df_new.index.difference(df_old.index)
        if attr_cap in df_new.columns and len(only_new) > 0:
            df_new.loc[only_new, attr_cap] = 0.0
        if attr_cap in df_new.columns and len(common) > 0:
            df_new.loc[common, attr_cap] = df_old.loc[common, attr_opt].fillna(0.0).astype(float)


def _assert_no_extendables(n: pypsa.Network) -> None:
    """Raise RuntimeError if any extendable flags remain True after capacity fixing."""
    for comp, _, __, attr_ext in _CAPACITY_MAP:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0 or attr_ext not in df.columns:
            continue
        mask = df[attr_ext].fillna(False).astype(bool)
        if mask.any():
            bad = df.index[mask].tolist()[:10]
            raise RuntimeError(
                f"{comp}: still extendable after _fix_portfolio_capacities. "
                f"Examples: {bad}. PyPSA would add investment variables, "
                "violating the pure-dispatch assumption."
            )


def _compute_portfolio_investment_cost(port_net: pypsa.Network) -> float:
    """
    Annualized investment cost of the portfolio: Σ capital_cost × capacity_opt.
    Scenario-independent; added as a constant to all dispatch cost estimates so
    worst-case comparisons across cutouts are on a consistent Capex+Opex basis.
    """
    total = 0.0
    for comp, attr_opt, _, __ in _CAPACITY_MAP:
        df = getattr(port_net, comp, None)
        if df is None or len(df) == 0:
            continue
        if "capital_cost" not in df.columns or attr_opt not in df.columns:
            continue
        cap = df[attr_opt].fillna(0.0).astype(float)
        invested = cap > 0.0
        if invested.any():
            cc = df.loc[invested, "capital_cost"].fillna(0.0).astype(float)
            total += float((cc * cap[invested]).sum())
    return float(total)


def _extract_objective_value(n: pypsa.Network) -> float:
    """
    Extract PyPSA's native objective value.
    Used only when co2_cost_mode == "off".
    Tries model.objective_value → model.objective.value → solver_model.ObjVal.
    """
    model = getattr(n, "model", None)
    if model is None:
        raise RuntimeError("n.model is None after optimize().")
    obj = getattr(model, "objective_value", None)
    if obj is not None:
        return float(obj)
    try:
        return float(model.objective.value)  # type: ignore[attr-defined]
    except Exception:
        pass
    sm = getattr(model, "solver_model", None)
    if sm is not None:
        try:
            return float(sm.ObjVal)
        except Exception:
            pass
    raise RuntimeError(
        "Could not extract objective value: none of (model.objective_value, "
        "model.objective.value, model.solver_model.ObjVal) is available."
    )


# =============================================================================
# [PATCH-6] Dispatch-only solve (moved from solve_aro + ls_penalty)
# =============================================================================

def _dispatch_solve(
        n: pypsa.Network,
        solver_name: str,
        solver_options: Optional[Dict],
        ramp_data: pd.DataFrame,
        scenarios: List[str],
        masks: Dict,
        *,
        co2_cost_mode: str = "off",
        ls_penalty: float = 1e4,
) -> float:
    """
    Dispatch-only solve (no investment variables).

    CO2 mode handling  [FIX-CO2]:
      co2_cost_mode == "off":
        Uses PyPSA's native objective (pure operational cost).
        Return = PyPSA ObjVal.

      co2_cost_mode != "off":
        PyPSA's native objective does NOT include GlobalConstraint.constant_cost.
        The objective is overridden with Σ _build_operational_cost_expression (CO2).
        Return = evaluated value of that manual expression, NOT PyPSA's ObjVal.
        This guarantees cost consistency with Stage 2 of the robust solve.

    [PATCH-5] ls_penalty: configurable load-shedding marginal cost (default 1e4).

    Returns
    -------
    float  Operational cost (no investment component).
    """
    for comp in ("storage_units", "stores"):
        df = getattr(n, comp, None)
        if df is not None and len(df) > 0:
            for col in ("cyclic_state_of_charge", "cyclic_state_of_charge_per_period"):
                if col in df.columns:
                    df[col] = False

    # [PATCH-5] use configurable ls_penalty
    _ensure_load_shedding_generators(n, marginal_cost=ls_penalty)

    _manual_expr: Dict[str, Any] = {"expr": None}

    def extra_dispatch(network: pypsa.Network, snapshots: pd.Index) -> None:
        _add_scenario_boundary_constraints(network, scenarios, masks)
        _add_within_scenario_ramp_constraints(
            network, scenarios, masks, ramp_data, fail_on_extendable=False,
        )
        if co2_cost_mode != "off":
            total_cost = 0
            for s, mask in masks.items():
                total_cost = total_cost + _build_operational_cost_expression(
                    network, mask, co2_cost_mode=co2_cost_mode
                )
            network.model.objective = total_cost
            _manual_expr["expr"] = total_cost
            logger.info("[CO2-Dispatch] Objective overridden (co2_cost_mode=%s).", co2_cost_mode)

    n.optimize(
        solver_name=solver_name,
        solver_options=solver_options if solver_options is not None else {},
        extra_functionality=extra_dispatch,
        io_api="mps"

    )

    model = getattr(n, "model", None)
    if model is None:
        raise RuntimeError("n.model is None after optimize().")
    status = str(getattr(model, "status", "unknown")).lower()
    if any(t in status for t in ("infeasible", "unbounded", "error", "failed")):
        raise RuntimeError(f"Dispatch solve failed: status='{status}'.")

    if co2_cost_mode != "off":
        expr = _manual_expr["expr"]
        if expr is None:
            raise RuntimeError("CO2 mode active but manual expression was not built.")
        try:
            return float(expr.evaluate())
        except Exception as exc:
            raise RuntimeError(f"Failed to evaluate CO2-extended objective: {exc}") from exc
    else:
        return _extract_objective_value(n)


# =============================================================================
# [PATCH-6] High-level single-cutout dispatch evaluation (public API for worker)
# =============================================================================

def evaluate_single_cutout_dispatch(
        cutout: str,
        portfolio_path: str,
        scen_net_path: str,
        solver_name: str,
        solver_options: Optional[Dict],
        dispatch_tmp_dir: str,
        investment_cost: float,
        *,
        cost_consistency_tol: float = 0.01,
        co2_cost_mode: str = "off",
        ls_penalty: float = 1e4,
) -> Dict[str, Any]:
    """
    Evaluate a fixed portfolio on a single cutout: dispatch solve + export.

    This function is the primary unit of work in the ARO dispatch evaluation.
    It is designed to be safely callable from a spawned subprocess worker
    (``solve_aro._dispatch_worker_fn``) without importing ``solve_aro`` itself.

    Steps
    -----
    1. Load portfolio network from disk.
    2. Stack cutout as single-scenario MultiIndex network.
    3. Fix capacities (non-extendable), assert no extendables remain.
    4. Dispatch-only solve via ``_dispatch_solve``.
    5. Cost-consistency validation.
    6. Export: flat-snapshot NetCDF + standard DatetimeIndex NetCDF.

    Parameters
    ----------
    cutout            : Cutout identifier string.
    portfolio_path    : Path to the robust portfolio network (flat-snapshot NetCDF).
    scen_net_path     : Path to the prepared scenario network for this cutout.
    solver_name       : Solver name (e.g. "gurobi", "highs").
    solver_options    : Optional solver options dict.
    dispatch_tmp_dir  : Directory for output dispatch networks.
    investment_cost   : Pre-computed portfolio investment cost (Capex, scenario-independent).
    cost_consistency_tol : Relative tolerance for cost-consistency check.
    co2_cost_mode     : Must match --co2-cost-mode used in solve_robust.
    ls_penalty        : Load-shedding marginal cost in €/MWh.  [PATCH-5]

    Returns
    -------
    dict with keys:
        cutout, op_cost, investment_cost, total_cost,
        flat_path, std_path, consistency
    """
    if not Path(scen_net_path).exists():
        raise FileNotFoundError(f"Scenario network missing: {scen_net_path}")

    port_net = pypsa.Network(str(Path(portfolio_path)))
    n, _ = stack_scenarios_to_multisnapshot_network([scen_net_path], [cutout])

    _fix_portfolio_capacities(n, port_net)
    _assert_no_extendables(n)

    ramp_data = _save_and_clear_ramp_limits(n)
    masks = _scenario_masks_from_snapshots(n.snapshots)
    scenarios = list(masks.keys())

    op_cost = _dispatch_solve(
        n, solver_name, solver_options, ramp_data, scenarios, masks,
        co2_cost_mode=co2_cost_mode, ls_penalty=ls_penalty,
    )
    total_cost = float(investment_cost) + float(op_cost)

    mask_arr = masks[cutout]
    consistency = _validate_dispatch_cost_consistency(
        n, pypsa_objective=op_cost, mask=mask_arr, scenario=cutout,
        rel_tol=cost_consistency_tol, co2_cost_mode=co2_cost_mode,
    )

    Path(dispatch_tmp_dir).mkdir(parents=True, exist_ok=True)
    flat_out = str(Path(dispatch_tmp_dir) / f"dispatch_{cutout}.nc")
    export_network_flat_snapshots(n, flat_out)

    std_out = str(Path(dispatch_tmp_dir) / f"dispatch_{cutout}__std.nc")
    export_network_standard_single_scenario(
        n, scenario=cutout, scenario_network_file=scen_net_path, out_network=std_out,
    )

    logger.info(
        "Cutout '%s': op_cost=%.6g  inv_cost=%.6g  total=%.6g",
        cutout, op_cost, investment_cost, total_cost,
    )
    return {
        "cutout": cutout,
        "op_cost": float(op_cost),
        "investment_cost": float(investment_cost),
        "total_cost": float(total_cost),
        "flat_path": flat_out,
        "std_path": std_out,
        "consistency": consistency,
    }


# =============================================================================
# Robust solve (lexicographic)
# =============================================================================

def solve_robust_lexicographic(
        n: pypsa.Network,
        *,
        solver_name: str,
        solver_options: Optional[Dict] = None,
        eps_ls_abs: float = 1e-3,
        eps_ls_rel: float = 1e-6,
        load_shedding_carrier: str = "load_shedding",
        ls_penalty: float = 1e4,
        hard_fail_suboptimal: bool = True,
        fail_on_extendable_ramps: bool = True,
        co2_cost_mode: str = "off",
) -> Dict[str, Any]:
    """
    Two-stage lexicographic robust optimisation over stacked scenario network.

    Stage 1: min z_ls    s.t. z_ls >= LS_s  ∀s
             CO2/budget GlobalConstraints are RELAXED in Stage 1 because load
             shedding itself produces no CO2. Relaxing prevents infeasibility
             when the CO2 budget is tight.

    Stage 2: min z_cost  s.t. z_cost >= Inv + Op_s  ∀s,  z_ls <= z_ls* + eps
             CO2/budget GlobalConstraints are RESTORED for Stage 2 to guarantee
             climate neutrality in the optimal investment portfolio.

    [PATCH-5] ls_penalty: configurable load-shedding marginal cost (default 1e4 €/MWh).

    Note on duals:
        buses_t.marginal_price contains nodal duals of the Minimax epigraph LP —
        NOT standard LMPs.  Use dispatch-only solves for market-grade marginal prices.
    """
    if solver_options is None:
        solver_options = {}

    ramp_data = _save_and_clear_ramp_limits(n)
    # [PATCH-5] pass ls_penalty to LS generator creation
    ls_generators = _ensure_load_shedding_generators(
        n, carrier=load_shedding_carrier, marginal_cost=ls_penalty
    )
    masks = _scenario_masks_from_snapshots(n.snapshots)
    scenarios = list(masks.keys())
    logger.info("Robust optimisation over %d scenarios: %s  ls_penalty=%.4g",
                len(scenarios), scenarios, ls_penalty)

    if co2_cost_mode != "off":
        _detect_possible_co2_double_counting(n)



    for comp in ("storage_units", "stores"):
        df = getattr(n, comp, None)
        if df is not None and len(df) > 0:
            for col in ("cyclic_state_of_charge", "cyclic_state_of_charge_per_period", "e_cyclic"):
                if col in df.columns:
                    df[col] = False
                    logger.info("Disabled '%s' for %s (%d components)", col, comp, len(df))

    # ------------------------------------------------------------------
    # CO2 / budget GlobalConstraints: relax for Stage 1, restore Stage 2
    # Stage 1 only minimises load shedding — CO2 limits are irrelevant
    # there and can make the problem infeasible when the budget is tight.
    # ------------------------------------------------------------------
    _CO2_KEYWORDS = ("co2", "CO2", "carbon", "Carbon", "emission", "Emission")

    _gc_backup: Dict[str, float] = {}
    for gc_name in list(n.global_constraints.index):
        original = float(n.global_constraints.at[gc_name, "constant"])
        sense = n.global_constraints.at[gc_name, "sense"]
        relaxed = 1e12 if sense == "<=" else -1e12
        _gc_backup[gc_name] = original
        n.global_constraints.at[gc_name, "constant"] = relaxed
        logger.info(
            "Stage 1: relaxing GlobalConstraint '%s' (%s %.3e) → %.3e",
            gc_name, sense, original, relaxed,
        )

    def extra_stage1(network: pypsa.Network, snapshots: pd.Index) -> None:
        m = network.model
        _add_scenario_boundary_constraints(network, scenarios, masks)
        _add_within_scenario_ramp_constraints(network, scenarios, masks, ramp_data,
                                             fail_on_extendable=fail_on_extendable_ramps)
        z_ls = m.add_variables(lower=0, name="z_ls")
        for s in scenarios:
            ls_e = _build_ls_energy_expression(network, ls_generators, masks[s])
            m.add_constraints(1.0 * z_ls >= ls_e, name=f"robust_ls_epigraph::{s}")
        m.objective = 1.0 * z_ls

    logger.info("Stage 1: minimising worst-case load shedding ...")
    n.optimize(solver_name=solver_name, solver_options=solver_options,
               extra_functionality=extra_stage1, assign_all_duals=True,io_api="mps")

    _check_solver_status(n, "Stage 1", hard_fail_suboptimal=hard_fail_suboptimal)
    sol1 = _get_solution_dict(n)
    z_ls_star = _extract_scalar_solution(sol1, "z_ls")

    if z_ls_star < -1e-6:
        raise ValueError(f"Invalid negative load shedding: {z_ls_star}")
    if z_ls_star < 0:
        logger.warning("Clamping z_ls* = %.2e to 0 (numerical noise).", z_ls_star)
        z_ls_star = 0.0
    logger.info("Stage 1: z_ls* = %.6g", z_ls_star)

    # ------------------------------------------------------------------
    # Restore CO2 / budget GlobalConstraints for Stage 2
    # ------------------------------------------------------------------
    for gc_name, original in _gc_backup.items():
        n.global_constraints.at[gc_name, "constant"] = original
        logger.info(
            "Stage 2: restoring GlobalConstraint '%s' → %.3e",
            gc_name, original,
        )

    try:
        if getattr(n, "model", None) is not None:
            n.model = None
    except Exception:
        if hasattr(n, "_model"):
            n._model = None

    eps = max(float(eps_ls_abs), float(eps_ls_rel) * max(1.0, float(z_ls_star)))
    logger.info("Stage 2 LS tolerance: eps = %.6g", eps)

    def extra_stage2(network: pypsa.Network, snapshots: pd.Index) -> None:
        m = network.model
        _add_scenario_boundary_constraints(network, scenarios, masks)
        _add_within_scenario_ramp_constraints(network, scenarios, masks, ramp_data,
                                             fail_on_extendable=fail_on_extendable_ramps)
        z_ls = m.add_variables(lower=0, name="z_ls")
        z_cost = m.add_variables(lower=0, name="z_cost")
        for s in scenarios:
            ls_e = _build_ls_energy_expression(network, ls_generators, masks[s])
            m.add_constraints(1.0 * z_ls >= ls_e, name=f"robust_ls_epigraph::{s}")
        m.add_constraints(1.0 * z_ls <= (float(z_ls_star) + eps), name="robust_ls_fix")
        inv_cost = _build_investment_cost_expression(network)
        for s in scenarios:
            op_cost_s = _build_operational_cost_expression(network, masks[s], co2_cost_mode=co2_cost_mode)
            m.add_constraints(1.0 * z_cost >= inv_cost + op_cost_s, name=f"robust_cost_epigraph::{s}")
        m.objective = 1.0 * z_cost

    logger.info("Stage 2: minimising worst-case total cost ...")
    n.optimize(solver_name=solver_name, solver_options=solver_options,
               extra_functionality=extra_stage2, assign_all_duals=True,io_api="mps")

    _check_solver_status(n, "Stage 2", hard_fail_suboptimal=hard_fail_suboptimal)
    sol2 = _get_solution_dict(n)
    z_cost_star = _extract_scalar_solution(sol2, "z_cost")
    z_ls_final = _extract_scalar_solution(sol2, "z_ls")
    slack = float(z_ls_final) - float(z_ls_star)

    logger.info("Stage 2: z_cost* = %.6g", z_cost_star)
    if slack > 10 * eps:
        logger.warning("LS constraint slack large: %.2e > 10×eps=%.2e.", slack, eps)

    diag: Dict[str, Any] = {
        "z_ls_star": float(z_ls_star),
        "z_ls_final": float(z_ls_final),
        "z_cost_star": float(z_cost_star),
        "n_scenarios": len(scenarios),
        "ls_constraint_slack": float(slack),
        "eps_ls_abs": float(eps_ls_abs),
        "eps_ls_rel": float(eps_ls_rel),
        "eps_ls_used": float(eps),
        "ls_penalty": float(ls_penalty),
        "fail_on_extendable_ramps": bool(fail_on_extendable_ramps),
        "co2_cost_mode": str(co2_cost_mode),
        "relaxed_global_constraints": {
            gc: {"original": orig}
            for gc, orig in _gc_backup.items()
        },
    }

    m2 = getattr(n, "model", None)
    if m2 is not None:
        for attr in ("status", "termination_condition", "termination"):
            val = getattr(m2, attr, None)
            if val is not None:
                diag["termination_status"] = str(val)
                break
        try:
            scenario_costs = _evaluate_scenario_costs(n, masks, co2_cost_mode)
            worst_scen, worst_cost = max(scenario_costs.items(), key=lambda kv: kv[1])
            ls_per_scen = {
                s: float(_build_ls_energy_expression(n, ls_generators, masks[s]).evaluate())
                for s in masks
            }
            diag.update({
                "scenario_costs": scenario_costs,
                "worst_case_scenario": worst_scen,
                "worst_case_cost": float(worst_cost),
                "load_shedding_per_scenario": ls_per_scen,
            })
        except Exception as exc:
            logger.warning("Scenario-wise diagnostics failed (non-fatal): %s", exc)

    return diag

# =============================================================================
# Export helpers
# =============================================================================

def extract_capacities(n: pypsa.Network) -> Dict[str, Dict[str, float]]:
    def _ext(comp_df, opt_col, ext_col):
        if comp_df is None or len(comp_df) == 0:
            return {}
        if opt_col not in comp_df.columns or ext_col not in comp_df.columns:
            return {}
        ext = comp_df.index[comp_df[ext_col].fillna(False).astype(bool)]
        return comp_df.loc[ext, opt_col].dropna().to_dict()

    out = {
        "generators": _ext(n.generators, "p_nom_opt", "p_nom_extendable"),
        "links": _ext(n.links, "p_nom_opt", "p_nom_extendable"),
        "storage_units": _ext(n.storage_units, "p_nom_opt", "p_nom_extendable"),
        "stores": _ext(n.stores, "e_nom_opt", "e_nom_extendable"),
        "lines": _ext(n.lines, "s_nom_opt", "s_nom_extendable"),
        "transformers": _ext(n.transformers, "s_nom_opt", "s_nom_extendable"),
    }
    return {k: v for k, v in out.items() if v}


def export_network_stacked(n: pypsa.Network, out_network: str) -> None:
    """[PATCH-2] Export stacked network; only reindex marginal_price if it exists."""
    if getattr(n, "model", None) is not None:
        try:
            n.model.solver_model = None
        except Exception:
            pass
    if hasattr(n, "buses_t"):
        try:
            mp = getattr(n.buses_t, "marginal_price", None)
        except Exception:
            mp = None
        if mp is not None and isinstance(mp, pd.DataFrame):
            n.buses_t.marginal_price = mp.reindex(index=n.snapshots, columns=n.buses.index)

    if isinstance(n.snapshots, pd.MultiIndex):
        export_network_flat_snapshots(n, out_network)
        return
    Path(out_network).parent.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(out_network)


def export_network_standard_single_scenario(
        n_stacked: pypsa.Network,
        *,
        scenario: str,
        scenario_network_file: str,
        out_network: str,
) -> None:
    """[PATCH-3] Export DatetimeIndex single-scenario network; guard on marginal_price."""
    if not isinstance(n_stacked.snapshots, pd.MultiIndex):
        raise ValueError("Expected stacked MultiIndex snapshots.")

    masks = _scenario_masks_from_snapshots(n_stacked.snapshots)
    if scenario not in masks:
        raise KeyError(f"Scenario '{scenario}' not found. Available: {list(masks.keys())}")

    idx = np.flatnonzero(masks[scenario].astype(bool))
    ts = n_stacked.snapshots.get_level_values("timestep")
    times = pd.DatetimeIndex([pd.Timestamp(ts[i][1]) for i in idx], name="snapshot")

    n_std = pypsa.Network(str(Path(scenario_network_file)))
    n_std.set_snapshots(times)

    for comp, opt_col, nom_col, ext_col in [
        ("generators", "p_nom_opt", "p_nom", "p_nom_extendable"),
        ("links", "p_nom_opt", "p_nom", "p_nom_extendable"),
        ("storage_units", "p_nom_opt", "p_nom", "p_nom_extendable"),
        ("stores", "e_nom_opt", "e_nom", "e_nom_extendable"),
        ("lines", "s_nom_opt", "s_nom", "s_nom_extendable"),
        ("transformers", "s_nom_opt", "s_nom", "s_nom_extendable"),
    ]:
        df_sol = getattr(n_stacked, comp, None)
        df_std = getattr(n_std, comp, None)
        if df_sol is None or df_std is None or len(df_std) == 0:
            continue
        if opt_col not in df_sol.columns or ext_col not in df_std.columns:
            continue
        common = df_std.index.intersection(df_sol.index)
        if len(common) == 0:
            continue
        df_std.loc[common, opt_col] = df_sol.loc[common, opt_col]
        ext_common = common.intersection(df_std.index[df_std[ext_col].fillna(False).astype(bool)])
        if len(ext_common) > 0 and nom_col in df_std.columns:
            df_std.loc[ext_common, nom_col] = df_sol.loc[ext_common, opt_col].fillna(0.0).astype(float)

    comp_cols = {
        "buses_t": n_std.buses.index, "generators_t": n_std.generators.index,
        "links_t": n_std.links.index, "storage_units_t": n_std.storage_units.index,
        "stores_t": n_std.stores.index, "lines_t": n_std.lines.index,
        "transformers_t": n_std.transformers.index, "loads_t": n_std.loads.index,
    }
    for t_name, attrs in _T_ATTRS.items():
        if not hasattr(n_stacked, t_name) or not hasattr(n_std, t_name):
            continue
        src = getattr(n_stacked, t_name)
        dst = getattr(n_std, t_name)
        for attr in attrs:
            df = getattr(src, attr, None)
            if df is None or not isinstance(df, (pd.DataFrame, pd.Series)) or len(df) == 0:
                continue
            if isinstance(df, pd.Series):
                df = df.to_frame()
            df_s = df.iloc[idx].copy()
            df_s.index = times
            target = comp_cols.get(t_name)
            if target is not None:
                df_s = df_s.reindex(columns=target)
            setattr(dst, attr, df_s)

    # [PATCH-3] guard
    try:
        mp = getattr(n_std.buses_t, "marginal_price", None)
    except Exception:
        mp = None
    if mp is not None and isinstance(mp, pd.DataFrame):
        n_std.buses_t.marginal_price = mp.reindex(index=n_std.snapshots, columns=n_std.buses.index)

    Path(out_network).parent.mkdir(parents=True, exist_ok=True)
    n_std.export_to_netcdf(out_network)


def export_network_flat_snapshots(n: pypsa.Network, out_network: str) -> None:
    """Export with flattened snapshot strings (copy, no mutation)."""
    if getattr(n, "model", None) is not None:
        try:
            n.model.solver_model = None
        except Exception:
            pass
    n_out = n.copy()
    if isinstance(n_out.snapshots, pd.MultiIndex):
        flat = []
        for p, ts in zip(n_out.snapshots.get_level_values("period"),
                         n_out.snapshots.get_level_values("timestep")):
            scen, t = ts[0], ts[1]
            try:
                t_str = pd.Timestamp(t).isoformat()
            except Exception:
                t_str = str(t)
            flat.append(f"{p}::{scen}::{t_str}")
        n_out.set_snapshots(pd.Index(flat, name="snapshot"))
    Path(out_network).parent.mkdir(parents=True, exist_ok=True)
    n_out.export_to_netcdf(out_network)


# =============================================================================
# Orchestration
# =============================================================================

def run_robust(
        *,
        cutouts: Sequence[str],
        scenario_network_template: str,
        out_network: str,
        out_summary_json: str,
        solver_name: str,
        solver_options: Optional[Dict],
        eps_ls_abs: float,
        eps_ls_rel: float,
        allow_suboptimal: bool,
        allow_extendable_ramps: bool,
        co2_cost_mode: str,
        warn_unknown_t: bool,
        strict_unknown_t: bool,
        ls_penalty: float = 1e4,
) -> None:
    scenario_files = _resolve_scenario_networks_from_cutouts(cutouts, scenario_network_template)
    logger.info("=== Robust solve: co2_cost_mode=%s  ls_penalty=%.4g ===", co2_cost_mode, ls_penalty)
    for c, f in zip(cutouts, scenario_files):
        logger.info("  %-35s -> %s", c, f)

    n, scen_names = stack_scenarios_to_multisnapshot_network(
        scenario_files=scenario_files, scenario_names=list(cutouts),
        warn_unknown_t=warn_unknown_t, strict_unknown_t=strict_unknown_t,
    )
    diag = solve_robust_lexicographic(
        n, solver_name=solver_name, solver_options=solver_options,
        eps_ls_abs=eps_ls_abs, eps_ls_rel=eps_ls_rel,
        hard_fail_suboptimal=not allow_suboptimal,
        fail_on_extendable_ramps=not allow_extendable_ramps,
        co2_cost_mode=co2_cost_mode,
        ls_penalty=ls_penalty,
    )
    capacities = extract_capacities(n)

    Path(out_network).parent.mkdir(parents=True, exist_ok=True)
    Path(out_summary_json).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting STACKED robust network to %s", out_network)
    _ensure_standard_pypsa_result_frames(n)
    export_network_stacked(n, out_network)

    out_network_std = str(Path(out_network).with_suffix("")) + "__std.nc"
    export_scenario = str(list(cutouts)[0])
    export_file = str(scenario_files[0])
    logger.info("Exporting STANDARD adapter (scenario=%s) to %s", export_scenario, out_network_std)
    export_network_standard_single_scenario(
        n, scenario=export_scenario, scenario_network_file=export_file, out_network=out_network_std,
    )

    payload: Dict[str, Any] = {
        "scenario_names": list(scen_names),
        "scenario_networks": list(map(str, scenario_files)),
        "diagnostics": diag,
        "capacities": capacities,
        "outputs": {
            "robust_stacked_network": str(out_network),
            "postprocess_adapter_network": str(out_network_std),
            "postprocess_adapter_scenario": export_scenario,
        },
        "assumptions": {
            "co2_cost_mode": co2_cost_mode,
            "ls_penalty": ls_penalty,
            "duals_note": (
                "buses_t.marginal_price = nodal duals of Minimax epigraph LP "
                "(NOT standard LMPs). Use dispatch-only solve for market-grade prices."
            ),
            "co2_limitation": (
                "CO2 term covers Generator-based emissions only. "
                "Link-based fossil conversion is NOT captured."
            ),
        },
    }
    logger.info("Writing summary JSON to %s", out_summary_json)
    with open(out_summary_json, "w") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shared-investment robust optimisation over cutout scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cutouts", nargs="+")
    p.add_argument("--scenario-network-template")
    p.add_argument("--out-network")
    p.add_argument("--out-summary-json")
    p.add_argument("--solver-name", default="gurobi")
    p.add_argument("--solver-options-json", default=None)
    p.add_argument("--eps-ls-abs", type=float, default=1e-3)
    p.add_argument("--eps-ls-rel", type=float, default=1e-6)
    p.add_argument("--allow-suboptimal", action="store_true")
    p.add_argument("--allow-extendable-ramps", action="store_true")
    p.add_argument(
        "--co2-cost-mode",
        choices=["off", "global_constraint_constant_cost"],
        default="off",
        help=(
            "CO2 cost term in Stage 2 (OPT-IN). "
            "'off' = no CO2 (default). "
            "'global_constraint_constant_cost' = use GlobalConstraint.constant_cost. "
            "Covers Generator emissions only. WARNING: double-counting risk if CO2 is in marginal_cost."
        ),
    )
    # [PATCH-5] configurable LS penalty
    p.add_argument(
        "--ls-penalty", type=float, default=1e4,
        help=(
            "[PATCH-5] Load-shedding marginal cost in €/MWh (default: 1e4). "
            "Must be strictly higher than any real generator marginal cost to avoid "
            "the solver preferring LS over dispatch. "
            "Increase if your model has very high-cost peakers."
        ),
    )
    p.add_argument("--warn-unknown-t", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict-unknown-t", action="store_true")
    p.add_argument("--test", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    if args.test:
        logger.info("No unit tests in this version.")
        return

    required = ["cutouts", "scenario_network_template", "out_network", "out_summary_json"]
    missing = [r for r in required if getattr(args, r.replace("-", "_"), None) is None]
    if missing:
        raise SystemExit(f"Missing required arguments: {missing}\nUse --help.")

    solver_options = json.loads(args.solver_options_json) if args.solver_options_json else None

    run_robust(
        cutouts=args.cutouts,
        scenario_network_template=args.scenario_network_template,
        out_network=args.out_network,
        out_summary_json=args.out_summary_json,
        solver_name=args.solver_name,
        solver_options=solver_options,
        eps_ls_abs=args.eps_ls_abs,
        eps_ls_rel=args.eps_ls_rel,
        allow_suboptimal=args.allow_suboptimal,
        allow_extendable_ramps=args.allow_extendable_ramps,
        co2_cost_mode=args.co2_cost_mode,
        warn_unknown_t=args.warn_unknown_t,
        strict_unknown_t=args.strict_unknown_t,
        ls_penalty=args.ls_penalty,
    )


if __name__ == "__main__":
    main()