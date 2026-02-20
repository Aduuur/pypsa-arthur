#!/usr/bin/env python3
"""
solve_robust.py  (cutout-native inputs, shared-investment robust 2050 design)

Goal
----
ONE single 2050 investment design (portfolio x) that is feasible/optimal across ALL
weather cutouts (scenario set S).

Lexicographic robust model:
  Stage 1: minimise worst-case load shedding energy
           min z_ls   s.t.  z_ls >= LS_s  for all s in S
  Stage 2: minimise worst-case total cost given LS optimality
           min z_cost  s.t.  z_cost >= Cost_s  for all s in S
                             z_ls <= z_ls* + eps

Key engineering ideas
---------------------
* Stack per-cutout networks into ONE PyPSA Network with MultiIndex snapshots
  names=["period","timestep"], timestep=(scenario_name, datetime).
* Shared static assets; only time series vary by scenario.
* Per-scenario cyclic boundary constraints on SOC / Store-e prevent
  artificial cross-scenario state coupling from stacking.
* Ramp constraints:
    - we SAVE generator ramp_limit_* + committable, then CLEAR them from the network
      so PyPSA does not add cross-scenario ramp/UC constraints.
    - we RE-ADD ramp constraints manually, strictly within each scenario segment.
    - extendable generators with ramp limits would yield bilinear constraints ->
      by default we HARD-FAIL. Can be overridden with --allow-extendable-ramps (not recommended).
    - the difference expression p_next - p_prev is built with matching integer "pair" coordinates
  to avoid xarray alignment errors on mismatched time indices.

Design assumptions & known limitations
---------------------------------------
* Stage-2 cost is an approximation covering:
    - investment (capital_cost * optimized capacity)
    - operational (marginal_cost * dispatch)
  plus optional CO2 via GlobalConstraint constant_cost (OPT-IN; off by default).
* Complex PyPSA-Eur sector-coupled custom objective terms may still be missing.
* DSM / delay-based custom components are not boundary-constrained.
* CO2 double-counting warning heuristic detects obvious cases but is not foolproof.

Usage
-----
Run optimisation:
    python solve_robust.py \\
        --cutouts 2013 2014 2015 \\
        --scenario-network-template networks/prepared_{cutout}.nc \\
        --out-network results/robust.nc \\
        --out-summary-json results/robust_summary.json \\
        --solver-name highs

Run unit tests (no solver required):
    python solve_robust.py --test
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
# General helpers
# =============================================================================

def _resolve_scenario_networks_from_cutouts(
        cutouts: Sequence[str], template: str
) -> List[str]:
    """Resolve scenario network file paths from cutout ids via string template."""
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
    """
    Return non-empty time-dependent DataFrame attributes for a PyPSA *_t container.

    Uses an explicit allow-list (_T_ATTRS) instead of dir() introspection.

    Unknown DataFrames (not in allow-list) are detected conservatively:
    we only warn/raise if the DataFrame's index overlaps base_index (i.e. it looks
    time-dependent). This avoids False-Positives from metadata DataFrames.

    If strict_unknown=True, raises ValueError instead of warning.
    """
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
            # Only flag if it looks time-dependent (index overlaps base_index)
            try:
                overlap = val.index.intersection(base_index)
                if len(overlap) == 0:
                    continue
            except Exception:
                pass
            unknown_hits.append(attr)

        if unknown_hits:
            msg = (
                f"Container '{container_name}' has DataFrame attribute(s) not in _T_ATTRS: "
                f"{unknown_hits}. They will NOT be stacked across scenarios. "
                "Add them to _T_ATTRS if they carry scenario-dependent time series."
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
    """
    Save generator ramp limits and committable flags, then clear them from the network.

    Clearing is necessary so PyPSA does not generate its own ramp/UC constraints
    that would couple consecutive snapshots across scenario boundaries.

    Returns a DataFrame with the saved values (indexed by generator).
    Returns an empty DataFrame if n has no generators.
    """
    if not hasattr(n, "generators") or len(n.generators) == 0:
        return pd.DataFrame()

    saved = pd.DataFrame(index=n.generators.index)

    for col in _RAMP_COLS:
        if col in n.generators.columns:
            saved[col] = n.generators[col].copy()
            if n.generators[col].notna().any():
                logger.info(
                    "Saving and clearing '%s' from %d generators "
                    "(will be re-added as within-scenario constraints).",
                    col,
                    int(n.generators[col].notna().sum()),
                )
            n.generators[col] = np.nan
        else:
            saved[col] = np.nan

    if "committable" in n.generators.columns:
        saved["committable"] = n.generators["committable"].copy()
        if n.generators["committable"].fillna(False).any():
            logger.info(
                "Saving and clearing 'committable' flags "
                "(UC constraints would couple scenario boundaries)."
            )
        n.generators["committable"] = False
    else:
        saved["committable"] = False

    return saved


# =============================================================================
# Consistency checks
# =============================================================================

def _assert_same_static_assets_and_snapshots(
        networks: Sequence[pypsa.Network],
) -> None:
    """Enforce that all scenarios represent the same system (same assets, same time grid)."""
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
                    msg += f"  Missing in scenario {i}: {list(missing)[:10]}\n"
                if len(extra):
                    msg += f"  Extra in scenario {i}: {list(extra)[:10]}\n"
                raise ValueError(msg)
            df_aligned = df.reindex(ref_df.index)
            for col in ["bus", "carrier", "p_nom_extendable", "capital_cost", "efficiency"]:
                if col in ref_df.columns and col in df_aligned.columns:
                    if not ref_df[col].equals(df_aligned[col]):
                        logger.warning(
                            "Static attribute differs: component=%s col=%s scenario=%d.",
                            name, col, i,
                        )

    for comp in [
        "buses", "generators", "loads", "links", "lines",
        "transformers", "storage_units", "stores",
    ]:
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
    """
    Load prepared per-scenario networks and stack into one MultiIndex-snapshot network.

    Snapshot MultiIndex: names=["period","timestep"], timestep=(scenario_name, datetime).
    All static assets must be identical across scenarios; only time series differ.
    """
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
    if (
            hasattr(n, "investment_periods")
            and n.investment_periods is not None
            and len(n.investment_periods) > 0
    ):
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
            ref_t,
            t_container_name,
            base_index=base_snaps,
            warn_unknown=warn_unknown_t,
            strict_unknown=strict_unknown_t,
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
# Snapshot masks (hard fail on unexpected structure)
# =============================================================================

def _scenario_masks_from_snapshots(snapshots: pd.Index) -> Dict[str, np.ndarray]:
    """
    Extract per-scenario boolean masks from stacked MultiIndex.
    Hard fails on any unexpected structure to prevent silent mis-grouping.
    """
    if not isinstance(snapshots, pd.MultiIndex):
        raise ValueError(
            f"Expected pd.MultiIndex snapshots, got {type(snapshots)}. "
            "Use stack_scenarios_to_multisnapshot_network() first."
        )
    if list(snapshots.names) != ["period", "timestep"]:
        raise ValueError(
            f"Expected MultiIndex names ['period','timestep'], got {snapshots.names}."
        )

    ts = snapshots.get_level_values("timestep")
    bad = [x for x in ts if not (isinstance(x, tuple) and len(x) >= 2)]
    if bad:
        raise ValueError(
            "Expected 'timestep' level to contain (scenario, timestamp) tuples. "
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
        p_nom: float = 1e9,
) -> List[str]:
    """Ensure one non-extendable load-shedding Generator per bus."""
    existing: List[str] = []
    if hasattr(n, "generators") and "carrier" in n.generators.columns:
        existing = n.generators.index[
            n.generators.carrier.astype(str) == carrier
            ].tolist()

    if existing:
        logger.info("Found %d existing load-shedding generators.", len(existing))
        _ensure_ls_pmax_timeseries(n, existing)
        return existing

    if "load_shedding" not in n.carriers.index:
        n.add("Carrier", "load_shedding")

    logger.info("Adding load-shedding generators (one per bus).")
    ls_names: List[str] = []
    for bus in n.buses.index:
        name = f"LS::{bus}"
        ls_names.append(name)
        n.add(
            "Generator", name,
            bus=bus, carrier=carrier,
            p_nom=p_nom, p_nom_extendable=False,
            marginal_cost=marginal_cost,
            efficiency=1.0, p_min_pu=0.0, p_max_pu=1.0,
        )

    _ensure_ls_pmax_timeseries(n, ls_names)
    return ls_names


def _ensure_ls_pmax_timeseries(n: pypsa.Network, ls_names: List[str]) -> None:
    """
    Ensure generators_t.p_max_pu covers all LS generators with 1.0
    across ALL stacked snapshots.
    """
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
    raise ValueError(
        f"No standard time dimension in variable dims {getattr(var, 'dims', None)}."
    )


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

def _check_solver_status(
        n: pypsa.Network,
        stage: str,
        *,
        hard_fail_suboptimal: bool = True,
) -> bool:
    """
    Verify solver status after optimize().

    Status string handling is intentionally permissive on exact wording to cope
    with differences across PyPSA / Linopy / solver versions:
      - Infeasible / unbounded / error keyword in status → always raises RuntimeError.
      - "ok" or anything starting with "optimal" → accepted (with a warning if not
        exactly "ok" or "optimal", e.g. "optimal_inaccurate").
      - Everything else (time_limit, suboptimal, warning, ...) → logs warning;
        raises RuntimeError if hard_fail_suboptimal=True (default).
      - No status attribute but solution dict exists → logs warning, returns True.
      - No status and no solution → always raises RuntimeError.
    """
    m = getattr(n, "model", None)
    if m is None:
        raise RuntimeError(f"[{stage}] n.model is None after optimize().")

    status_str: Optional[str] = None
    for attr in ("status", "termination_condition", "termination"):
        val = getattr(m, attr, None)
        if val is not None:
            status_str = str(val).lower().strip()
            break

    sol = getattr(m, "solution", None)

    if status_str is None:
        if sol is None:
            raise RuntimeError(
                f"[{stage}] No solution and no solver status available. "
                "Optimisation likely failed silently."
            )
        logger.warning(
            "[%s] Solver status attribute unavailable — solution dict exists, proceeding.", stage
        )
        return True

    fatal_terms = ("infeasible", "unbounded", "error", "failed", "invalid")
    if any(t in status_str for t in fatal_terms):
        obj_val = getattr(m, "objective_value", "unknown")
        raise RuntimeError(
            f"[{stage}] Solver reports fatal status '{status_str}' (objective={obj_val}). "
            "Check model and solver logs."
        )

    if status_str == "ok" or status_str.startswith("optimal"):
        if status_str not in ("ok", "optimal"):
            obj_val = getattr(m, "objective_value", "unknown")
            logger.warning(
                "[%s] Solver status '%s' (objective=%s) — solution may be slightly suboptimal.",
                stage, status_str, obj_val,
            )
        return True

    # Suboptimal / time_limit / warning / other non-fatal
    obj_val = getattr(m, "objective_value", "unknown")
    logger.warning(
        "[%s] Non-optimal solver status: '%s' (objective=%s). Solution may be suboptimal.",
        stage, status_str, obj_val,
    )
    if hard_fail_suboptimal:
        raise RuntimeError(
            f"[{stage}] Aborting due to non-optimal solver status: '{status_str}'. "
            "Use --allow-suboptimal to proceed anyway (not recommended)."
        )
    return False


# =============================================================================
# Boundary constraints (per-scenario cyclic for true state variables)
# =============================================================================

def _add_scenario_boundary_constraints(
        network: pypsa.Network,
        scenarios: Sequence[str],
        masks: Dict[str, np.ndarray],
) -> None:
    """
    Enforce per-scenario cyclic boundary conditions on intertemporal state variables:
        SOC(s, t_first) == SOC(s, t_last)
        StoreE(s, t_first) == StoreE(s, t_last)

    Prevents artificial cross-scenario state coupling from the stacked layout.
    Ramp constraints are handled separately via _add_within_scenario_ramp_constraints.

    Known limitation: DSM / delay-based custom components (rare in standard PyPSA-Eur)
    are not covered here. Add them if your model uses such components.
    """
    m = network.model

    su_soc, su_name = _get_linopy_var(
        m,
        ["StorageUnit-state_of_charge", "StorageUnit-soc", "StorageUnit-energy"],
        strict=False,
    )
    if su_soc is not None:
        td = _get_time_dimension(su_soc)
        for s in scenarios:
            idx = _mask_to_isel_indices(masks[s])
            if len(idx) >= 2:
                m.add_constraints(
                    su_soc.isel({td: int(idx[0])}) == su_soc.isel({td: int(idx[-1])}),
                    name=f"boundary::StorageUnit::cyclic::{s}",
                )
        logger.info("StorageUnit cyclic SOC boundary constraints added (var=%s).", su_name)

    st_e, st_name = _get_linopy_var(
        m,
        ["Store-e", "Store-energy", "Store-state_of_charge"],
        strict=False,
    )
    if st_e is not None:
        td = _get_time_dimension(st_e)
        for s in scenarios:
            idx = _mask_to_isel_indices(masks[s])
            if len(idx) >= 2:
                m.add_constraints(
                    st_e.isel({td: int(idx[0])}) == st_e.isel({td: int(idx[-1])}),
                    name=f"boundary::Store::cyclic::{s}",
                )
        logger.info("Store cyclic energy boundary constraints added (var=%s).", st_name)


# =============================================================================
# Ramp constraints (rebuild per-scenario, fail-fast on extendable ramps)
# =============================================================================

def _add_within_scenario_ramp_constraints(
        network: pypsa.Network,
        scenarios: Sequence[str],
        masks: Dict[str, np.ndarray],
        ramp_data: pd.DataFrame,
        *,
        fail_on_extendable: bool = True,
) -> None:
    """
    Re-add ramp constraints as explicit Linopy constraints, strictly within each scenario.

    For each non-extendable generator g with ramp_limit_up / ramp_limit_down, and for
    each consecutive pair (t, t+1) WITHIN a scenario:
        p[g, t+1] - p[g, t] <= ramp_limit_up   * p_nom
        p[g, t]   - p[g, t+1] <= ramp_limit_down * p_nom

    Coordinate alignment fix
    -------------------------
    p_prev and p_next are sliced from the stacked time axis and therefore have
    DIFFERENT time coordinate values (t vs t+1). Direct arithmetic between them in
    xarray/Linopy would fail or produce NaNs due to label-based alignment.

    We fix this by re-assigning both slices to a shared integer "pair" coordinate
    [0, 1, ..., n_pairs-1] before subtraction. This makes the operation element-wise
    as intended, without losing the structural relationship between consecutive timesteps.

    Extendable generators with ramp limits would yield bilinear constraints (p * p_nom
    where both are variables) and are therefore not supported. By default we HARD-FAIL;
    use fail_on_extendable=False to skip them with a warning (not recommended).
    """
    if ramp_data is None or ramp_data.empty:
        return

    m = network.model
    gen_p, _ = _get_linopy_var(m, ["Generator-p"], strict=False)
    if gen_p is None:
        logger.info("No Generator-p variable found; skipping ramp constraint rebuild.")
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
        .fillna(False)
        .astype(bool)
    )
    ext_idx = network.generators.index[ext_mask]
    non_ext_idx = network.generators.index[~ext_mask]

    bad_ext = ext_idx.intersection(gens_with_ramp)
    if len(bad_ext) > 0:
        msg = (
            f"Extendable generators with ramp limits detected (bilinear if enforced): {list(bad_ext)}. "
            "Fix p_nom (make non-extendable) or remove ramp limits before robust solve."
        )
        if fail_on_extendable:
            raise RuntimeError(msg)
        logger.warning(msg + " Proceeding by SKIPPING these ramp constraints (not recommended).")

    gens_to_constrain = non_ext_idx.intersection(gens_with_ramp)
    if len(gens_to_constrain) == 0:
        logger.info("No non-extendable generators with ramp limits — nothing to add.")
        return

    n_added = 0
    for s in scenarios:
        idx = _mask_to_isel_indices(masks[s])
        if len(idx) < 2:
            continue

        prev = idx[:-1]
        nxt = idx[1:]
        n_pairs = len(prev)
        # Integer "pair" coordinate shared by p_prev and p_next so xarray arithmetic
        # operates element-wise instead of trying to align on mismatched time labels.
        pair_coord = np.arange(n_pairs)

        for g in gens_to_constrain:
            p_nom_g = float(network.generators.loc[g, "p_nom"])
            rup = ramp_data.loc[g, "ramp_limit_up"]
            rdn = ramp_data.loc[g, "ramp_limit_down"]

            p_g = gen_p.sel({gen_dim: g})

            # Slice and reassign to shared pair coordinate to avoid label-alignment errors
            p_prev = p_g.isel({td: prev.tolist()}).assign_coords({td: pair_coord})
            p_next = p_g.isel({td: nxt.tolist()}).assign_coords({td: pair_coord})

            if not pd.isna(rup):
                m.add_constraints(
                    p_next - p_prev <= float(rup) * p_nom_g,
                    name=f"ramp_up::{g}::{s}",
                )
                n_added += n_pairs
            if not pd.isna(rdn):
                m.add_constraints(
                    p_prev - p_next <= float(rdn) * p_nom_g,
                    name=f"ramp_down::{g}::{s}",
                )
                n_added += n_pairs

    if n_added > 0:
        logger.info(
            "Added %d within-scenario ramp constraints for %d non-extendable generators.",
            n_added, len(gens_to_constrain),
        )


# =============================================================================
# Cost expressions
# =============================================================================

def _build_ls_energy_expression(
        n: pypsa.Network, ls_generators: List[str], mask: np.ndarray
):
    """
    Weighted load-shedding energy for one scenario:
        LS_s = sum_{t in s} w_t * sum_{g in LS} p_{g,t}
    """
    import xarray as xr

    m = n.model
    gen_p, gen_p_name = _get_linopy_var(m, ["Generator-p"], strict=True)
    gen_dim = _get_gen_dimension(gen_p)
    td = _get_time_dimension(gen_p)

    try:
        available = set(gen_p.coords[gen_dim].values.tolist())
    except Exception as exc:
        raise RuntimeError(
            f"Cannot access '{gen_dim}' coordinate on {gen_p_name}: {exc}"
        ) from exc

    missing = set(ls_generators) - available
    if missing:
        raise ValueError(f"Load-shedding generators not found in model: {sorted(missing)}")

    idx = _mask_to_isel_indices(mask)
    gen_p_ls_s = gen_p.sel({gen_dim: ls_generators}).isel({td: idx})
    w = _weights_objective_series(n).to_numpy()[idx]
    w_da = xr.DataArray(w, dims=[td], coords={td: gen_p_ls_s.coords[td]})

    return (gen_p_ls_s * w_da).sum()


def _build_investment_cost_expression(n: pypsa.Network):
    """
    Investment cost expression for ALL extendable assets.
    Shared across scenarios (investment decisions are scenario-independent).

    Covers:
        Generators:   capital_cost * p_nom
        Links:        capital_cost * p_nom
        StorageUnits: capital_cost * p_nom
        Stores:       capital_cost * e_nom
        Lines:        capital_cost * s_nom
        Transformers: capital_cost * s_nom
    """
    import xarray as xr

    m = n.model
    expr = 0

    def _add(comp_df: pd.DataFrame, var_names: List[str], dim: str) -> None:
        nonlocal expr
        if comp_df is None or len(comp_df) == 0:
            return
        if "capital_cost" not in comp_df.columns:
            return
        var, _ = _get_linopy_var(m, var_names, strict=False)
        if var is None:
            return
        cc = comp_df["capital_cost"].reindex(comp_df.index).fillna(0.0)
        cc_da = xr.DataArray(
            cc.to_numpy(), dims=[dim],
            coords={dim: (dim, comp_df.index.to_numpy())},
        )
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
    Heuristic check: warn if fossil generators have non-zero marginal_cost when
    CO2 cost mode is enabled, since CO2 might already be embedded in marginal_cost.

    This is not foolproof but catches the most common double-counting pattern
    in PyPSA-Eur setups that fold CO2 price into generator marginal costs.
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
            "CO2 cost mode enabled. Detected generators with fossil-like carriers and "
            "positive marginal_cost — CO2 might already be embedded in marginal_cost. "
            "This can cause double counting. Example rows:\n%s",
            top.to_string(),
        )


def _build_co2_cost_expression(
        n: pypsa.Network,
        mask: np.ndarray,
) -> Any:
    """
    OPT-IN approximation of CO2 cost via GlobalConstraint with constant_cost.

    IMPORTANT: This is workflow-dependent and can cause double counting if CO2 is
    already embedded in marginal_cost. Enable only if your networks represent CO2
    cost via GlobalConstraint constant_cost and NOT via marginal_cost.

    Looks for GlobalConstraint rows with:
        type in {"primary_energy", "co2"}
        constant_cost > 0

    For each such row, adds:
        co2_price * sum_{t in s} w_t * sum_g ef_g * p_g,t
    where ef_g is the emission factor from n.carriers[carrier_attribute].
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
        logger.warning(
            "CO2 cost mode enabled, but no GlobalConstraint rows with type in "
            "{primary_energy, co2} and constant_cost > 0 found. CO2 term = 0."
        )
        return 0

    m = n.model
    gen_p, _ = _get_linopy_var(m, ["Generator-p"], strict=False)
    if gen_p is None:
        return 0

    if "carrier" not in n.generators.columns:
        logger.warning("CO2 cost: n.generators has no 'carrier' column → skipping CO2 term.")
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
            logger.warning(
                "CO2 cost: carrier attribute '%s' not in n.carriers → skipping this row.",
                carrier_attr,
            )
            continue

        ef = n.generators["carrier"].map(n.carriers[carrier_attr]).fillna(0.0).astype(float)
        if (ef == 0).all():
            continue

        ef_da = xr.DataArray(
            ef.to_numpy(), dims=[gen_dim],
            coords={gen_dim: (gen_dim, n.generators.index.to_numpy())},
        )
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
    Scenario-specific operational cost approximation:
        OpCost_s = sum_{t in s} w_t * (dispatch_vars * marginal_costs)

    Covers:
      - Generator-p * marginal_cost
      - Link-p0 * marginal_cost
      - StorageUnit-p_dispatch * marginal_cost
      - Store-p * marginal_cost
      - Optional CO2 term (OPT-IN): GlobalConstraint constant_cost mode

    Known remaining gaps:
      - Multi-port Link costs beyond p0 (model-specific; add marginal_cost_port1/2/... if needed)
      - Custom PyPSA-Eur objective terms outside the marginal_cost framework
    """
    import xarray as xr

    m = n.model
    w = _weights_objective_series(n)
    idx = _mask_to_isel_indices(mask)
    total = 0

    def _add_mc(comp_df: pd.DataFrame, var_names: List[str], dim: str) -> None:
        nonlocal total
        if comp_df is None or len(comp_df) == 0:
            return
        if "marginal_cost" not in comp_df.columns:
            return
        var, _ = _get_linopy_var(m, var_names, strict=False)
        if var is None:
            return
        mc = comp_df["marginal_cost"].reindex(comp_df.index).fillna(0.0)
        mc_da = xr.DataArray(
            mc.to_numpy(), dims=[dim],
            coords={dim: (dim, comp_df.index.to_numpy())},
        )
        td = _get_time_dimension(var)
        var_s = var.isel({td: idx})
        w_s = w.to_numpy()[idx]
        w_da = xr.DataArray(w_s, dims=[td], coords={td: var_s.coords[td]})
        total = total + (var_s * mc_da * w_da).sum()

        # Generator-Dispatch (unverändert)

    _add_mc(n.generators, ["Generator-p"], "Generator")
    # Storage-Units: Entladung, Nettoeinspeisung und neu auch Ladeleistung p_store
    _add_mc(n.storage_units, ["StorageUnit-p_dispatch", "StorageUnit-p", "StorageUnit-p_store"], "StorageUnit")
    # Stores: Leistung (unverändert)
    _add_mc(n.stores, ["Store-p"], "Store")
    # Links: alle definierten Ports p0–p4 berücksichtigen, da PyPSA Kosten auf mehreren Ports erlaubt
    _add_mc(n.links, ["Link-p0"], "Link")
    _add_mc(n.links, ["Link-p1"], "Link")
    _add_mc(n.links, ["Link-p2"], "Link")
    _add_mc(n.links, ["Link-p3"], "Link")
    _add_mc(n.links, ["Link-p4"], "Link")


if co2_cost_mode == "global_constraint_constant_cost":
    total = total + _build_co2_cost_expression(n, mask)
elif co2_cost_mode != "off":
    raise ValueError(
        f"Unknown co2_cost_mode='{co2_cost_mode}'. "
        "Use 'off' or 'global_constraint_constant_cost'."
    )

return total


def _evaluate_scenario_costs(n: pypsa.Network, masks: Dict[str, np.ndarray], co2_cost_mode: str) -> Dict[str, float]:
    # gemeinsamer Investitionskosten­anteil
    inv_expr = _build_investment_cost_expression(n)
    inv_cost = float(inv_expr.evaluate())

    costs = {}
    for scen, mask in masks.items():
        op_expr = _build_operational_cost_expression(n, mask, co2_cost_mode=co2_cost_mode)
        op_cost = float(op_expr.evaluate())
        costs[scen] = inv_cost + op_cost
    return costs


# =============================================================================
# Solution helpers
# =============================================================================

def _get_solution_dict(n: pypsa.Network) -> Dict:
    m = getattr(n, "model", None)
    if m is None:
        raise RuntimeError("n.model is None.")
    sol = getattr(m, "solution", None)
    if sol is None:
        raise RuntimeError("n.model.solution is None. Optimisation likely failed.")
    return sol


def _extract_scalar_solution(sol: Dict, key: str) -> float:
    if key not in sol:
        raise KeyError(
            f"Solution missing '{key}'. Available (head): {list(sol.keys())[:50]}"
        )
    v = sol[key]
    try:
        return float(getattr(v, "item", lambda: v)())
    except Exception:
        return float(np.asarray(v).item())


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
        hard_fail_suboptimal: bool = True,
        fail_on_extendable_ramps: bool = True,
        co2_cost_mode: str = "off",
) -> Dict[str, Any]:
    """
    Two-stage lexicographic robust optimisation over stacked scenario network.

    Stage 1: min z_ls    s.t. z_ls >= LS_s  ∀s
    Stage 2: min z_cost  s.t. z_cost >= Cost_s  ∀s
                              z_ls <= z_ls* + eps
             eps = max(eps_ls_abs, eps_ls_rel * max(1, z_ls*))

    Ramp constraints are saved, cleared (so PyPSA does not add cross-scenario ones),
    and re-added strictly within each scenario in both extra_functionality calls.

    n.model is explicitly set to None between stages to guarantee a clean rebuild.
    CO2 cost is OPT-IN to avoid double counting risk.
    """
    if solver_options is None:
        solver_options = {}

    ramp_data = _save_and_clear_ramp_limits(n)
    ls_generators = _ensure_load_shedding_generators(n, carrier=load_shedding_carrier)
    masks = _scenario_masks_from_snapshots(n.snapshots)
    scenarios = list(masks.keys())
    logger.info("Robust optimisation over %d scenarios: %s", len(scenarios), scenarios)

    if co2_cost_mode != "off":
        _detect_possible_co2_double_counting(n)

    # ------------------------------------------------------------------
    # Stage 1: minimise worst-case load shedding
    # ------------------------------------------------------------------
    def extra_stage1(network: pypsa.Network, snapshots: pd.Index) -> None:
        m = network.model
        _add_scenario_boundary_constraints(network, scenarios, masks)
        _add_within_scenario_ramp_constraints(
            network, scenarios, masks, ramp_data,
            fail_on_extendable=fail_on_extendable_ramps,
        )
        z_ls = m.add_variables(lower=0, name="z_ls")
        for s in scenarios:
            ls_e = _build_ls_energy_expression(network, ls_generators, masks[s])
            m.add_constraints(1.0 * z_ls >= ls_e, name=f"robust_ls_epigraph::{s}")
        m.objective = 1.0 * z_ls

    logger.info("Stage 1: minimising worst-case load shedding energy ...")
    n.optimize(
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=extra_stage1,
    )

    _check_solver_status(n, "Stage 1", hard_fail_suboptimal=hard_fail_suboptimal)
    sol1 = _get_solution_dict(n)
    z_ls_star = _extract_scalar_solution(sol1, "z_ls")

    if z_ls_star < -1e-6:
        raise ValueError(f"Invalid negative load shedding after Stage 1: {z_ls_star}")
    if z_ls_star < 0:
        logger.warning("Clamping small negative z_ls* = %.2e to 0 (numerical noise).", z_ls_star)
        z_ls_star = 0.0

    logger.info("Stage 1 optimum: z_ls* = %.6g", z_ls_star)

    # Drop model so Stage 2 starts with a fresh Linopy model
    if getattr(n, "model", None) is not None:
        n.model = None

    eps = max(float(eps_ls_abs), float(eps_ls_rel) * max(1.0, float(z_ls_star)))
    logger.info(
        "Stage 2 LS tolerance: eps = max(abs=%.3e, rel=%.3e × z_ls*=%.3g) = %.6g",
        eps_ls_abs, eps_ls_rel, z_ls_star, eps,
    )

    # ------------------------------------------------------------------
    # Stage 2: minimise worst-case total cost given LS optimality
    # ------------------------------------------------------------------
    def extra_stage2(network: pypsa.Network, snapshots: pd.Index) -> None:
        m = network.model
        _add_scenario_boundary_constraints(network, scenarios, masks)
        _add_within_scenario_ramp_constraints(
            network, scenarios, masks, ramp_data,
            fail_on_extendable=fail_on_extendable_ramps,
        )

        z_ls = m.add_variables(lower=0, name="z_ls")
        z_cost = m.add_variables(lower=0, name="z_cost")

        for s in scenarios:
            ls_e = _build_ls_energy_expression(network, ls_generators, masks[s])
            m.add_constraints(1.0 * z_ls >= ls_e, name=f"robust_ls_epigraph::{s}")

        m.add_constraints(1.0 * z_ls <= (float(z_ls_star) + eps), name="robust_ls_fix")

        inv_cost = _build_investment_cost_expression(network)
        for s in scenarios:
            op_cost_s = _build_operational_cost_expression(
                network, masks[s], co2_cost_mode=co2_cost_mode
            )
            m.add_constraints(
                1.0 * z_cost >= inv_cost + op_cost_s,
                name=f"robust_cost_epigraph::{s}",
            )
        m.objective = 1.0 * z_cost

    logger.info("Stage 2: minimising worst-case total cost ...")
    n.optimize(
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=extra_stage2,
    )

    _check_solver_status(n, "Stage 2", hard_fail_suboptimal=hard_fail_suboptimal)
    sol2 = _get_solution_dict(n)
    z_cost_star = _extract_scalar_solution(sol2, "z_cost")
    z_ls_final = _extract_scalar_solution(sol2, "z_ls")

    logger.info("Stage 2 optimum: z_cost* = %.6g", z_cost_star)
    logger.info(
        "Stage 2 achieved: z_ls = %.6g  (bound = z_ls* + eps = %.6g)",
        z_ls_final, z_ls_star + eps,
    )

    slack = float(z_ls_final) - float(z_ls_star)
    if slack > 10 * eps:
        logger.warning(
            "LS constraint slack is large: slack=%.2e > 10×eps=%.2e. "
            "Consider increasing eps_ls_abs.",
            slack, eps,
        )

    diag: Dict[str, Any] = {
        "z_ls_star": float(z_ls_star),
        "z_ls_final": float(z_ls_final),
        "z_cost_star": float(z_cost_star),
        "n_scenarios": len(scenarios),
        "ls_constraint_slack": float(slack),
        "eps_ls_abs": float(eps_ls_abs),
        "eps_ls_rel": float(eps_ls_rel),
        "eps_ls_used": float(eps),
        "fail_on_extendable_ramps": bool(fail_on_extendable_ramps),
        "co2_cost_mode": str(co2_cost_mode),
    }

    m2 = getattr(n, "model", None)
    if m2 is not None:
        for attr in ("status", "termination_condition", "termination"):
            val = getattr(m2, attr, None)
            if val is not None:
                diag["termination_status"] = str(val)
                break

    return diag

    scenario_costs = _evaluate_scenario_costs(n, masks, co2_cost_mode)
    worst_scen, worst_cost = max(scenario_costs.items(), key=lambda kv: kv[1])

    ls_per_scen = {s: float(_build_ls_energy_expression(n, ls_generators, masks[s]).evaluate()) for s in masks}

    diag.update({
        "scenario_costs": scenario_costs,
        "worst_case_scenario": worst_scen,
        "worst_case_cost": worst_cost,
        "load_shedding_per_scenario": ls_per_scen,
    })


# =============================================================================
# Output extraction / export
# =============================================================================

def extract_capacities(n: pypsa.Network) -> Dict[str, Dict[str, float]]:
    """Extract optimised capacities for all extendable component types."""

    def _ext(comp_df: pd.DataFrame, opt_col: str, ext_col: str) -> Dict[str, float]:
        if comp_df is None or len(comp_df) == 0:
            return {}
        if opt_col not in comp_df.columns or ext_col not in comp_df.columns:
            return {}
        ext = comp_df.index[comp_df[ext_col].fillna(False).astype(bool)]
        return comp_df.loc[ext, opt_col].dropna().to_dict()

    out: Dict[str, Dict[str, float]] = {
        "generators": _ext(n.generators, "p_nom_opt", "p_nom_extendable"),
        "links": _ext(n.links, "p_nom_opt", "p_nom_extendable"),
        "storage_units": _ext(n.storage_units, "p_nom_opt", "p_nom_extendable"),
        "stores": _ext(n.stores, "e_nom_opt", "e_nom_extendable"),
        "lines": _ext(n.lines, "s_nom_opt", "s_nom_extendable"),
        "transformers": _ext(n.transformers, "s_nom_opt", "s_nom_extendable"),
    }
    return {k: v for k, v in out.items() if v}


def export_network_flat_snapshots(n: pypsa.Network, out_network: str) -> None:
    """
    Export network to NetCDF with flattened snapshot strings.
    Operates on a copy — does not mutate the solved network.
    """
    if getattr(n, "model", None) is not None:
        try:
            n.model.solver_model = None
        except Exception:
            pass

    n_out = n.copy()

    if isinstance(n_out.snapshots, pd.MultiIndex):
        flat = []
        for p, ts in zip(
                n_out.snapshots.get_level_values("period"),
                n_out.snapshots.get_level_values("timestep"),
        ):
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
) -> None:
    scenario_files = _resolve_scenario_networks_from_cutouts(cutouts, scenario_network_template)

    logger.info("=== Robust solve configuration ===")
    for c, f in zip(cutouts, scenario_files):
        logger.info("  %-35s -> %s", c, f)

    n, scen_names = stack_scenarios_to_multisnapshot_network(
        scenario_files=scenario_files,
        scenario_names=list(cutouts),
        warn_unknown_t=warn_unknown_t,
        strict_unknown_t=strict_unknown_t,
    )

    diag = solve_robust_lexicographic(
        n,
        solver_name=solver_name,
        solver_options=solver_options,
        eps_ls_abs=eps_ls_abs,
        eps_ls_rel=eps_ls_rel,
        hard_fail_suboptimal=not allow_suboptimal,
        fail_on_extendable_ramps=not allow_extendable_ramps,
        co2_cost_mode=co2_cost_mode,
    )

    capacities = extract_capacities(n)
    Path(out_network).parent.mkdir(parents=True, exist_ok=True)
    Path(out_summary_json).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting network to %s", out_network)
    export_network_flat_snapshots(n, out_network)

    payload: Dict[str, Any] = {
        "scenario_names": list(scen_names),
        "scenario_networks": list(map(str, scenario_files)),
        "diagnostics": diag,
        "capacities": capacities,
        "assumptions": {
            "design_target": "single shared-investment 2050 portfolio robust across all cutouts",
            "static_assets_identical": True,
            "snapshots_identical": True,
            "snapshot_index_structure": (
                "MultiIndex(['period','timestep']) with timestep=(scenario, datetime)"
            ),
            "scenario_boundary_constraints": (
                "cyclic StorageUnit SOC + cyclic Store energy per scenario"
            ),
            "ramp_constraints": (
                "saved+cleared from network; rebuilt per scenario (element-wise via shared pair "
                "coordinate) for non-extendable generators; extendable generators with ramp limits "
                "fail-fast unless --allow-extendable-ramps is set (not recommended)"
            ),
            "load_shedding": "one high-cost Generator per bus, carrier=load_shedding",
            "stage2_cost_note": (
                "Approximation: capital_cost + marginal_cost dispatch. "
                "Optional CO2 term via GlobalConstraint constant_cost is OPT-IN to avoid "
                "double counting. See _build_operational_cost_expression for extension points."
            ),
            "co2_cost_mode": co2_cost_mode,
            "unknown_time_series_handling": {
                "warn_unknown_t": bool(warn_unknown_t),
                "strict_unknown_t": bool(strict_unknown_t),
            },
        },
    }

    logger.info("Writing summary JSON to %s", out_summary_json)
    with open(out_summary_json, "w") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# Unit tests (run with: python solve_robust.py --test)
# No solver required. No NetCDF I/O (avoids version-dependent flakiness).
# =============================================================================

def _run_unit_tests() -> None:
    import traceback

    passed = 0
    failed = 0

    def ok(name: str) -> None:
        nonlocal passed
        print(f"  PASS  {name}")
        passed += 1

    def fail(name: str, exc: Exception) -> None:
        nonlocal failed
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()
        failed += 1

    print("\n=== Running unit tests ===\n")

    # ------------------------------------------------------------------
    # 1. scenario_masks: valid MultiIndex
    # ------------------------------------------------------------------
    try:
        base = pd.date_range("2030-01-01", periods=3, freq="h")
        mi = pd.MultiIndex.from_tuples(
            [(0, ("S1", t)) for t in base] + [(0, ("S2", t)) for t in base],
            names=["period", "timestep"],
        )
        masks = _scenario_masks_from_snapshots(mi)
        assert set(masks.keys()) == {"S1", "S2"}
        assert masks["S1"].sum() == 3 and masks["S2"].sum() == 3
        assert not np.any(masks["S1"] & masks["S2"])
        ok("scenario_masks_valid")
    except Exception as e:
        fail("scenario_masks_valid", e)

    # ------------------------------------------------------------------
    # 2. scenario_masks: flat Index → ValueError
    # ------------------------------------------------------------------
    try:
        try:
            _scenario_masks_from_snapshots(pd.date_range("2030", periods=4, freq="h"))
            fail("scenario_masks_rejects_flat", AssertionError("Should raise"))
        except ValueError:
            ok("scenario_masks_rejects_flat")
    except Exception as e:
        fail("scenario_masks_rejects_flat", e)

    # ------------------------------------------------------------------
    # 3. scenario_masks: wrong MultiIndex names → ValueError
    # ------------------------------------------------------------------
    try:
        bad_mi = pd.MultiIndex.from_arrays(
            [pd.date_range("2030", periods=2, freq="h")] * 2,
            names=["time", "other"],
        )
        try:
            _scenario_masks_from_snapshots(bad_mi)
            fail("scenario_masks_rejects_wrong_names", AssertionError("Should raise"))
        except ValueError:
            ok("scenario_masks_rejects_wrong_names")
    except Exception as e:
        fail("scenario_masks_rejects_wrong_names", e)

    # ------------------------------------------------------------------
    # 4. scenario_masks: non-tuple timestep → ValueError
    # ------------------------------------------------------------------
    try:
        bad_mi2 = pd.MultiIndex.from_tuples(
            [(0, "not_a_tuple"), (0, "also_not")],
            names=["period", "timestep"],
        )
        try:
            _scenario_masks_from_snapshots(bad_mi2)
            fail("scenario_masks_rejects_non_tuple", AssertionError("Should raise"))
        except ValueError:
            ok("scenario_masks_rejects_non_tuple")
    except Exception as e:
        fail("scenario_masks_rejects_non_tuple", e)

    # ------------------------------------------------------------------
    # 5. _mask_to_isel_indices: correctness
    # ------------------------------------------------------------------
    try:
        idx = _mask_to_isel_indices(np.array([True, False, True, False, True]))
        assert list(idx) == [0, 2, 4]
        ok("mask_to_isel_indices")
    except Exception as e:
        fail("mask_to_isel_indices", e)

    # ------------------------------------------------------------------
    # 6. _ensure_datetime_snapshots: string coercion + name
    # ------------------------------------------------------------------
    try:
        result = _ensure_datetime_snapshots(pd.Index(["2030-01-01", "2030-01-02"]))
        assert isinstance(result, pd.DatetimeIndex) and result.name == "snapshot"
        ok("ensure_datetime_snapshots")
    except Exception as e:
        fail("ensure_datetime_snapshots", e)

    # ------------------------------------------------------------------
    # 7. _assert_same_static_assets: identical → passes
    # ------------------------------------------------------------------
    try:
        n1 = pypsa.Network()
        n1.set_snapshots(pd.date_range("2030", periods=2, freq="h"))
        n1.add("Bus", "A")
        n1.add("Generator", "G1", bus="A", p_nom=100)
        n2 = n1.copy()
        _assert_same_static_assets_and_snapshots([n1, n2])
        ok("assert_same_assets_pass")
    except Exception as e:
        fail("assert_same_assets_pass", e)

    # ------------------------------------------------------------------
    # 8. _assert_same_static_assets: extra generator → ValueError
    # ------------------------------------------------------------------
    try:
        n1 = pypsa.Network()
        n1.set_snapshots(pd.date_range("2030", periods=2, freq="h"))
        n1.add("Bus", "A")
        n1.add("Generator", "G1", bus="A", p_nom=100)
        n2 = n1.copy()
        n2.add("Generator", "G2", bus="A", p_nom=50)
        try:
            _assert_same_static_assets_and_snapshots([n1, n2])
            fail("assert_same_assets_fail_extra", AssertionError("Should raise"))
        except ValueError:
            ok("assert_same_assets_fail_extra")
    except Exception as e:
        fail("assert_same_assets_fail_extra", e)

    # ------------------------------------------------------------------
    # 9. _assert_same_static_assets: snapshot mismatch → ValueError
    # ------------------------------------------------------------------
    try:
        n1 = pypsa.Network()
        n1.set_snapshots(pd.date_range("2030", periods=3, freq="h"))
        n1.add("Bus", "A")
        n2 = n1.copy()
        n2.set_snapshots(pd.date_range("2030", periods=4, freq="h"))
        try:
            _assert_same_static_assets_and_snapshots([n1, n2])
            fail("assert_same_snapshots_fail", AssertionError("Should raise"))
        except ValueError:
            ok("assert_same_snapshots_fail")
    except Exception as e:
        fail("assert_same_snapshots_fail", e)

    # ------------------------------------------------------------------
    # 10. _save_and_clear_ramp_limits: values saved, network cleared
    # ------------------------------------------------------------------
    try:
        n = pypsa.Network()
        n.add("Bus", "A")
        n.add(
            "Generator", "G1", bus="A", p_nom=100,
            ramp_limit_up=0.3, ramp_limit_down=0.2, committable=True,
        )
        saved = _save_and_clear_ramp_limits(n)
        assert float(saved.loc["G1", "ramp_limit_up"]) == 0.3
        assert float(saved.loc["G1", "ramp_limit_down"]) == 0.2
        assert bool(saved.loc["G1", "committable"]) is True
        assert pd.isna(n.generators.loc["G1", "ramp_limit_up"])
        assert n.generators.loc["G1", "committable"] is False
        ok("save_and_clear_ramp_limits")
    except Exception as e:
        fail("save_and_clear_ramp_limits", e)

    # ------------------------------------------------------------------
    # 11. _save_and_clear_ramp_limits: network without ramp columns → no crash
    # ------------------------------------------------------------------
    try:
        n = pypsa.Network()
        n.add("Bus", "A")
        n.add("Generator", "G1", bus="A", p_nom=100)
        saved = _save_and_clear_ramp_limits(n)
        assert pd.isna(saved.loc["G1", "ramp_limit_up"])
        ok("save_and_clear_ramp_no_columns")
    except Exception as e:
        fail("save_and_clear_ramp_no_columns", e)

    # ------------------------------------------------------------------
    # 12. _save_and_clear_ramp_limits: no generators → empty DataFrame
    # ------------------------------------------------------------------
    try:
        n = pypsa.Network()
        n.add("Bus", "A")
        saved = _save_and_clear_ramp_limits(n)
        assert saved.empty
        ok("save_and_clear_ramp_no_generators")
    except Exception as e:
        fail("save_and_clear_ramp_no_generators", e)

    # ------------------------------------------------------------------
    # 13. _add_within_scenario_ramp_constraints: fail on extendable
    # ------------------------------------------------------------------
    try:
        n = pypsa.Network()
        snaps = pd.date_range("2030", periods=4, freq="h")
        n.set_snapshots(snaps)
        n.add("Bus", "A")
        n.add("Generator", "G_ext", bus="A", p_nom_extendable=True, p_nom=0)

        ramp_data = pd.DataFrame(
            {"ramp_limit_up": [0.3], "ramp_limit_down": [0.2]},
            index=["G_ext"],
        )
        # Build a minimal stacked MultiIndex to test the function signature
        base = pd.date_range("2030", periods=4, freq="h")
        mi = pd.MultiIndex.from_tuples(
            [(0, ("S1", t)) for t in base],
            names=["period", "timestep"],
        )
        masks_t = _scenario_masks_from_snapshots(mi)

        class FakeVar:
            dims = ("timestep", "Generator")
            coords = {"Generator": np.array(["G_ext"]), "timestep": np.arange(4)}

            def sel(self, d):
                return self

            def isel(self, d):
                return self

            def assign_coords(self, d):
                return self

        class FakeModel:
            variables = {"Generator-p": FakeVar()}

        n.model = FakeModel()

        try:
            _add_within_scenario_ramp_constraints(
                n, ["S1"], masks_t, ramp_data, fail_on_extendable=True
            )
            fail("ramp_fail_on_extendable", AssertionError("Should raise"))
        except RuntimeError:
            ok("ramp_fail_on_extendable")
    except Exception as e:
        fail("ramp_fail_on_extendable", e)

    # ------------------------------------------------------------------
    # 14. _list_time_dependent_frames: known attrs returned
    # ------------------------------------------------------------------
    try:
        n = pypsa.Network()
        snaps = pd.date_range("2030", periods=3, freq="h")
        n.set_snapshots(snaps)
        n.add("Bus", "A")
        n.add("Generator", "G1", bus="A", p_nom_extendable=True)
        n.generators_t.p_max_pu = pd.DataFrame({"G1": [0.8, 0.9, 0.7]}, index=snaps)
        result = _list_time_dependent_frames(
            n.generators_t, "generators_t",
            base_index=snaps, warn_unknown=False, strict_unknown=False,
        )
        assert "p_max_pu" in result
        assert "buses" not in result
        ok("list_time_dependent_frames_known_attrs")
    except Exception as e:
        fail("list_time_dependent_frames_known_attrs", e)

    # ------------------------------------------------------------------
    # 15. _list_time_dependent_frames: strict_unknown raises on unknown time-like DF
    # ------------------------------------------------------------------
    try:
        n = pypsa.Network()
        snaps = pd.date_range("2030", periods=3, freq="h")
        n.set_snapshots(snaps)
        n.add("Bus", "A")
        n.add("Generator", "G1", bus="A", p_nom=1.0)
        n.generators_t.some_custom_df = pd.DataFrame({"G1": [1, 2, 3]}, index=snaps)
        try:
            _list_time_dependent_frames(
                n.generators_t, "generators_t",
                base_index=snaps, warn_unknown=False, strict_unknown=True,
            )
            fail("list_time_frames_strict_raises", AssertionError("Should raise"))
        except ValueError:
            ok("list_time_frames_strict_raises")
    except Exception as e:
        fail("list_time_frames_strict_raises", e)

    # ------------------------------------------------------------------
    # 16. _weights_objective_series: fallback to 1.0
    # ------------------------------------------------------------------
    try:
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2030", periods=4, freq="h"))
        w = _weights_objective_series(n)
        assert len(w) == 4 and (w == 1.0).all()
        ok("weights_objective_fallback")
    except Exception as e:
        fail("weights_objective_fallback", e)

    # ------------------------------------------------------------------
    # 17. _resolve_scenario_networks_from_cutouts: template substitution
    # ------------------------------------------------------------------
    try:
        result = _resolve_scenario_networks_from_cutouts(["2013", "2014"], "nets/{cutout}.nc")
        assert result == ["nets/2013.nc", "nets/2014.nc"]
        ok("resolve_scenario_networks_template")
    except Exception as e:
        fail("resolve_scenario_networks_template", e)

    # ------------------------------------------------------------------
    # 18. extract_capacities: graceful when *_opt columns absent (pre-solve)
    # ------------------------------------------------------------------
    try:
        n = pypsa.Network()
        n.add("Bus", "A")
        n.add("Generator", "G1", bus="A", p_nom_extendable=True)
        caps = extract_capacities(n)
        assert "generators" not in caps or caps.get("generators", {}) == {}
        ok("extract_capacities_no_opt_columns")
    except Exception as e:
        fail("extract_capacities_no_opt_columns", e)

    # ------------------------------------------------------------------
    # 19. _check_solver_status: no status attr but solution exists → True
    # ------------------------------------------------------------------
    try:
        class FakeModelNoStatus:
            solution = {"z_ls": 0.0}

        n_f = pypsa.Network()
        n_f.model = FakeModelNoStatus()
        result = _check_solver_status(n_f, "Test", hard_fail_suboptimal=False)
        assert result is True
        ok("check_solver_status_no_attr")
    except Exception as e:
        fail("check_solver_status_no_attr", e)

    # ------------------------------------------------------------------
    # 20. _check_solver_status: "infeasible" → RuntimeError
    # ------------------------------------------------------------------
    try:
        class FakeModelInfeasible:
            status = "infeasible"
            solution = None
            objective_value = None

        n_f = pypsa.Network()
        n_f.model = FakeModelInfeasible()
        try:
            _check_solver_status(n_f, "Test")
            fail("check_solver_status_infeasible_raises", AssertionError("Should raise"))
        except RuntimeError:
            ok("check_solver_status_infeasible_raises")
    except Exception as e:
        fail("check_solver_status_infeasible_raises", e)

    # ------------------------------------------------------------------
    # 21. _check_solver_status: "optimal_inaccurate" → True (substring match)
    # ------------------------------------------------------------------
    try:
        class FakeModelSuboptimal:
            status = "optimal_inaccurate"
            solution = {"z_ls": 1.0}
            objective_value = 1.0

        n_f = pypsa.Network()
        n_f.model = FakeModelSuboptimal()
        result = _check_solver_status(n_f, "Test", hard_fail_suboptimal=False)
        assert result is True
        ok("check_solver_status_optimal_inaccurate")
    except Exception as e:
        fail("check_solver_status_optimal_inaccurate", e)

    # ------------------------------------------------------------------
    # 22. _check_solver_status: "time_limit" + hard_fail=True → RuntimeError
    # ------------------------------------------------------------------
    try:
        class FakeModelTimeout:
            status = "time_limit"
            solution = {"z_ls": 1.0}
            objective_value = 1.0

        n_f = pypsa.Network()
        n_f.model = FakeModelTimeout()
        try:
            _check_solver_status(n_f, "Test", hard_fail_suboptimal=True)
            fail("check_solver_status_timeout_hard_fail", AssertionError("Should raise"))
        except RuntimeError:
            ok("check_solver_status_timeout_hard_fail")
    except Exception as e:
        fail("check_solver_status_timeout_hard_fail", e)

    # ------------------------------------------------------------------
    # 23. stacked MultiIndex structure (in-memory, no file I/O)
    # ------------------------------------------------------------------
    try:
        base = pd.date_range("2030-01-01", periods=4, freq="h")

        def _make_net(seed: int) -> pypsa.Network:
            rng = np.random.default_rng(seed)
            net = pypsa.Network()
            net.set_snapshots(base)
            net.add("Bus", "A")
            net.add("Generator", "G1", bus="A", p_nom_extendable=True, capital_cost=1e5)
            net.add("Load", "L1", bus="A", p_set=100.0)
            net.generators_t.p_max_pu = pd.DataFrame(
                {"G1": rng.uniform(0.1, 0.9, len(base))}, index=base
            )
            return net

        nets_tmp = [_make_net(0), _make_net(1)]
        _assert_same_static_assets_and_snapshots(nets_tmp)

        ref = nets_tmp[0]
        base_snaps = _ensure_datetime_snapshots(ref.snapshots)
        n_stacked = ref.copy()
        timestep_labels = [("S1", ts) for ts in base_snaps] + [("S2", ts) for ts in base_snaps]
        stacked_snaps = pd.MultiIndex.from_tuples(
            [(0, lbl) for lbl in timestep_labels], names=["period", "timestep"]
        )
        n_stacked.set_snapshots(stacked_snaps)

        assert len(n_stacked.snapshots) == 8
        assert isinstance(n_stacked.snapshots, pd.MultiIndex)
        assert list(n_stacked.snapshots.names) == ["period", "timestep"]

        masks_t = _scenario_masks_from_snapshots(n_stacked.snapshots)
        assert set(masks_t.keys()) == {"S1", "S2"}
        assert masks_t["S1"].sum() == 4 and masks_t["S2"].sum() == 4
        ok("stack_structure_in_memory")
    except Exception as e:
        fail("stack_structure_in_memory", e)

    # ------------------------------------------------------------------
    # 24. ramp pair coordinate fix: no alignment error
    # ------------------------------------------------------------------
    try:
        # Verify that assign_coords on isel slices with different source indices
        # produces arrays that can be subtracted element-wise.
        import xarray as xr

        time_idx = np.arange(6)
        data = xr.DataArray(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            dims=["t"],
            coords={"t": time_idx},
        )

        prev_idx = np.array([0, 1, 2])
        nxt_idx = np.array([1, 2, 3])
        pair_coord = np.arange(len(prev_idx))

        p_prev = data.isel(t=prev_idx.tolist()).assign_coords({"t": pair_coord})
        p_next = data.isel(t=nxt_idx.tolist()).assign_coords({"t": pair_coord})

        diff = (p_next - p_prev).values
        # Each consecutive difference should be 1.0
        assert np.allclose(diff, [1.0, 1.0, 1.0]), f"Unexpected diff: {diff}"
        ok("ramp_pair_coordinate_alignment")
    except Exception as e:
        fail("ramp_pair_coordinate_alignment", e)

    # ------------------------------------------------------------------
    print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
    if failed:
        raise SystemExit(1)


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shared-investment robust optimisation over cutout scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cutouts", nargs="+", help="Cutout scenario ids.")
    p.add_argument(
        "--scenario-network-template",
        help='Path template, e.g. "networks/prepared_{cutout}.nc".',
    )
    p.add_argument("--out-network", help="Output robust network (.nc).")
    p.add_argument("--out-summary-json", help="Output robust summary (.json).")
    p.add_argument("--solver-name", default="gurobi", help="Solver name.")
    p.add_argument("--solver-options-json", default=None, help="JSON solver options.")
    p.add_argument(
        "--eps-ls-abs", type=float, default=1e-3,
        help="Absolute LS tolerance for Stage 2.",
    )
    p.add_argument(
        "--eps-ls-rel", type=float, default=1e-6,
        help="Relative LS tolerance for Stage 2.",
    )
    p.add_argument(
        "--allow-suboptimal", action="store_true",
        help="Do not raise on non-optimal solver status (time_limit, suboptimal, ...).",
    )
    p.add_argument(
        "--allow-extendable-ramps", action="store_true",
        help="Skip (rather than fail) ramp constraints for extendable generators. Not recommended.",
    )
    p.add_argument(
        "--co2-cost-mode",
        choices=["off", "global_constraint_constant_cost"],
        default="off",
        help=(
            "Optional CO2 cost term in Stage 2 objective (OPT-IN). "
            "'off' = no CO2 term (default). "
            "'global_constraint_constant_cost' = use GlobalConstraint constant_cost. "
            "WARNING: enable only if CO2 is NOT already in marginal_cost (risk of double counting)."
        ),
    )
    # --warn-unknown-t is the default; --no-warn-unknown-t disables it.
    # Using BooleanOptionalAction avoids the store_true/default=True contradiction.
    p.add_argument(
        "--warn-unknown-t",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Warn if *_t containers have unknown time-dependent DataFrames not in _T_ATTRS. "
            "Use --no-warn-unknown-t to silence. Default: warn."
        ),
    )
    p.add_argument(
        "--strict-unknown-t", action="store_true",
        help="Raise error (instead of warning) on unknown *_t DataFrames.",
    )
    p.add_argument("--test", action="store_true", help="Run unit tests and exit.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()

    if args.test:
        _run_unit_tests()
        return

    required = ["cutouts", "scenario_network_template", "out_network", "out_summary_json"]
    missing = [r for r in required if getattr(args, r.replace("-", "_"), None) is None]
    if missing:
        raise SystemExit(
            f"Missing required arguments: {missing}\n"
            "Use --help for usage, or --test to run unit tests."
        )

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
    )


if __name__ == "__main__":
    main()