#!/usr/bin/env python3
# solve_robust.py
"""
Patched version of the robust solver from ``Aduuur/pypsa-arthur``.

Changelog vs. upstream (February 2026)
=======================================

[PATCH-1..6]  Various patches (see previous versions)

================================================================================
MEM-PATCHES (March 2026)
================================================================================

[MEM-PATCH-A]  stack_scenarios_to_multisnapshot_network
[MEM-PATCH-B]  solve_aro_master / _build_operational_cost_expression
[MEM-PATCH-C]  snapshot_weightings stacking float32

================================================================================
BUG FIXES (April 2026)
================================================================================

[BUG-FIX-1]  cyclic_overrides=e_cyclic_backup in extra_master
[BUG-FIX-2]  _add_scenario_boundary_constraints is not None guard
[BUG-FIX-3]  GlobalConstraint constants restored after solve
[BUG-FIX-4]  Ramp limits and committable flags restored after solve
[BUG-FIX-5]  e_cyclic_backup restored after solve
[BUG-FIX-6]  IIS debug code behind ARO_IIS_DEBUG env flag
[BUG-FIX-7]  annual_scale applied in _dispatch_solve
[BUG-FIX-8]  annual_scale passed to cost consistency validator

================================================================================
DISPATCH FIXES (May 2026)
================================================================================

[DISPATCH-FIX-1]  _fix_negative_loads  — methodically correct treatment
    Negative loads_t.p_set = net-export nodes (e.g. GB1 during high wind).
    WRONG approach: clip to 0 — removes real energy flows from nodal balance,
    can cause artificial infeasibility or bias investment decisions.
    CORRECT approach: convert negative portion into a zero-marginal-cost
    Generator with a p_max_pu timeseries. Clip original load to >= 0.
    This preserves the full nodal energy balance.
    Applied in both master (solve_aro_master) and dispatch (_dispatch_solve).

[DISPATCH-FIX-2]  _fix_inf_store_initial  — extracted as reusable helper
    e_initial=-1e6 for unbounded stores (e.g. co2 atmosphere) is methodically
    correct: an atmospheric CO2 store has no meaningful initial state.
    Previously inline in master only — now also applied in dispatch.

[DISPATCH-FIX-3]  _prune_zero_capacity_assets in dispatch
    Dispatch model was 2.7x larger than master because zero-capacity assets
    were not pruned. Now pruned consistently in _dispatch_solve.

================================================================================
ARO METHODOLOGY FIXES (June 2026)
================================================================================

[ARO-FIX-1]  _fix_cross_scenario_storage_constraints  — break cross-scenario SOC coupling

    PROBLEM (critical, was present since initial implementation):
    PyPSA auto-generates a SOC balance constraint for every timestep t:
        soc[t] = (1 - standing_loss) * soc[t-1] + eff_store*p_store[t] - p_dispatch[t]/eff + inflow[t]
    In the stacked multi-scenario network, at the first timestep of scenario i (i > 0),
    `t-1` is the LAST timestep of scenario i-1. This creates:
        soc[first_t_scen_i] = f(soc[last_t_scen_{i-1}])
    — a direct coupling between independent ARO scenarios. This violates the
    fundamental ARO independence assumption: each scenario must be solved
    with its own self-consistent storage equilibrium.

    CONSEQUENCE without fix:
    - The solver must simultaneously satisfy intra-scenario cyclic constraints
      AND cross-scenario dynamic coupling → the scenarios are NOT independent.
    - Storage dispatch decisions in one scenario directly constrain another.
    - The ARO bound (worst-case operational cost) is computed over a coupled
      set of scenarios, not the correct independent scenario set.
    - This can make the master problem artificially infeasible or produce
      a sub-optimal portfolio that is not actually robust.

    FIX:
    After PyPSA builds the linopy model (inside extra_functionality callback),
    identify the SOC balance constraint rows at each scenario boundary
    (first timestep of each scenario except the first) and remove them.
    The intra-scenario cyclic constraints added by _add_scenario_boundary_constraints
    (soc[first_t_scen_i] == soc[last_t_scen_i]) then provide the only
    inter-timestep SOC connection at those positions, correctly enforcing
    scenario-independent cyclic equilibrium.

    ASSUMPTION: ARO scenarios are fully independent. Storage cannot carry
    energy between scenarios. Each scenario reaches its own cyclic steady
    state independently of all other scenarios.

    IMPORTANT: _add_scenario_boundary_constraints MUST be called before this
    function, so the cyclic closure constraints are in place before the
    cross-coupling balance rows are removed.

[ARO-FIX-2]  z_theta lower bound: lower=0 → lower=-np.inf

    PROBLEM:
    `z_theta = m.add_variables(lower=0, name="z_theta")` imposes an
    artificial non-negativity constraint on the worst-case operational cost
    variable. This is methodologically wrong.

    In the C&CG master formulation:
        min_{x} Inv(x) + z_theta
        s.t.    z_theta >= Op(x, s)   for all s in S_k
    the epigraph constraints themselves enforce that z_theta >= max_s Op(x,s).
    There is no theoretical reason to additionally require z_theta >= 0.

    If operational costs can be negative (e.g. negative marginal prices,
    export revenues, negative CO2 costs) in any scenario, lower=0 forces
    the solver to a strictly sub-optimal solution by preventing the correct
    worst-case cost from being negative.

    Even in standard energy systems where Op > 0, the lower=0 bound adds
    a redundant constraint that increases the problem size and can interfere
    with numerical methods (Barrier, Crossover).

    FIX: Set lower=-np.inf. The epigraph constraints guarantee correctness.
    The investment cost Inv(x) >= 0 is enforced by the capacity variable
    bounds, so the overall objective remains bounded.
"""

from __future__ import annotations

import argparse
import difflib
import gc
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
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

_T_ATTRS_INPUT_ONLY: Dict[str, List[str]] = {
    "buses_t":         [],
    "generators_t":    ["p_max_pu", "p_min_pu", "marginal_cost", "efficiency"],
    "loads_t":         ["p_set"],
    "links_t":         ["efficiency", "efficiency2", "efficiency3",
                        "marginal_cost", "p_min_pu", "p_max_pu"],
    "lines_t":         ["s_max_pu"],
    "transformers_t":  ["s_max_pu"],
    "storage_units_t": ["inflow", "p_min_pu", "p_max_pu", "marginal_cost"],
    "stores_t":        ["e_min_pu", "e_max_pu", "marginal_cost"],
}

_CAPACITY_MAP: List[Tuple[str, str, str, str]] = [
    ("generators",    "p_nom_opt", "p_nom",  "p_nom_extendable"),
    ("links",         "p_nom_opt", "p_nom",  "p_nom_extendable"),
    ("storage_units", "p_nom_opt", "p_nom",  "p_nom_extendable"),
    ("stores",        "e_nom_opt", "e_nom",  "e_nom_extendable"),
    ("lines",         "s_nom_opt", "s_nom",  "s_nom_extendable"),
    ("transformers",  "s_nom_opt", "s_nom",  "s_nom_extendable"),
]

_STATIC_COMPONENTS = [
    "buses", "generators", "loads", "links", "lines",
    "transformers", "storage_units", "stores", "carriers",
    "global_constraints", "investment_periods", "investment_period_weightings",
]

_T_CONTAINER_TO_STATIC_COMP: Dict[str, str] = {
    "generators_t":    "generators",
    "loads_t":         "loads",
    "links_t":         "links",
    "lines_t":         "lines",
    "transformers_t":  "transformers",
    "storage_units_t": "storage_units",
    "stores_t":        "stores",
    "buses_t":         "buses",
}

_RAMP_COLS = [
    "ramp_limit_up", "ramp_limit_down",
    "ramp_limit_start_up", "ramp_limit_shut_down",
]

# =============================================================================
# General helpers
# =============================================================================

def _resolve_scenario_networks_from_cutouts(
        cutouts: Sequence[str], template: str,
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
# [MEM-PATCH-A] Push constant *_t columns into static component tables
# =============================================================================

def _apply_constant_cols_to_static(
        n: pypsa.Network,
        *,
        t_container_name: str,
        attr: str,
        constant_cols: pd.Index,
        constant_vals: np.ndarray,
) -> None:
    if constant_cols is None or len(constant_cols) == 0:
        return
    static_comp = _T_CONTAINER_TO_STATIC_COMP.get(t_container_name)
    if static_comp is None or not hasattr(n, static_comp):
        return
    df_static = getattr(n, static_comp)
    if df_static is None or len(df_static) == 0:
        return
    if attr not in df_static.columns:
        df_static[attr] = np.nan
    common = df_static.index.intersection(constant_cols)
    if len(common) == 0:
        return
    s = pd.Series(constant_vals, index=constant_cols, dtype=np.float32)
    df_static.loc[common, attr] = s.loc[common].to_numpy(dtype=np.float32, copy=False)


# =============================================================================
# [DISPATCH-FIX-1] Negative load → Generator conversion (methodically correct)
# =============================================================================

def _fix_negative_loads(n: pypsa.Network) -> None:
    """
    [DISPATCH-FIX-1] Methodically correct treatment of negative loads_t.p_set.

    Negative load = net-export node: a bus where local generation exceeds
    consumption (e.g. GB1 during high-wind hours). This is a real physical
    flow that must be preserved in the nodal energy balance.

    WHY CLIPPING IS WRONG:
      Clipping negative p_set to 0 silently removes ~595 MW of generation
      from the system balance. This can cause:
        - Artificial infeasibility in the dispatch (supply > demand cannot
          be expressed at the affected bus)
        - Investment bias: storage/transmission capacity for export hours
          is underestimated
        - Masking of an upstream preprocessing bug in prepare_network.py

    CORRECT APPROACH:
      For each load with any negative p_set values:
        1. Create a zero-marginal-cost Generator at the same bus with
           a p_max_pu timeseries reflecting only the negative (export) portion.
        2. Clip the original load's p_set to >= 0 (only the demand side).
      The full nodal balance is preserved: export energy is supplied by the
      NegLoad:: generator instead of being silently removed.

    Applied in both solve_aro_master and _dispatch_solve to ensure
    consistent feasibility across master and dispatch solves.
    """
    if not hasattr(n, "loads_t") or getattr(n.loads_t, "p_set", None) is None:
        return
    if not isinstance(n.loads_t.p_set, pd.DataFrame) or n.loads_t.p_set.empty:
        return

    p_set = n.loads_t.p_set

    # Find loads that have at least one negative value
    neg_mask = (p_set < 0).any(axis=0)
    neg_loads = p_set.columns[neg_mask].tolist()

    if not neg_loads:
        return

    logger.info(
        "[DISPATCH-FIX-1] %d load(s) with negative p_set detected — "
        "converting to zero-cost generators to preserve nodal energy balance: %s",
        len(neg_loads), neg_loads,
    )

    # Ensure net_export carrier exists
    if hasattr(n, "carriers") and "net_export" not in n.carriers.index:
        try:
            n.add("Carrier", "net_export")
        except Exception:
            pass

    # Build or extend the p_max_pu DataFrame for generators
    pmax_df = getattr(n.generators_t, "p_max_pu", None)
    if pmax_df is None or not isinstance(pmax_df, pd.DataFrame):
        pmax_df = pd.DataFrame(index=n.snapshots)
    pmax_df = pmax_df.reindex(index=n.snapshots)

    for load_name in neg_loads:
        if load_name not in n.loads.index:
            continue

        load_row = n.loads.loc[load_name]
        bus = load_row["bus"]

        # Export timeseries: only negative p_set values, sign-flipped to positive
        neg_ts = (-p_set[load_name]).clip(lower=0.0)
        p_nom_val = float(neg_ts.max())

        if p_nom_val <= 0.0:
            # Pure numerical noise — just clip
            n.loads_t.p_set[load_name] = p_set[load_name].clip(lower=0.0)
            continue

        gen_name = f"NegLoad::{load_name}"

        if gen_name not in n.generators.index:
            n.add(
                "Generator", gen_name,
                bus=bus,
                carrier="net_export",
                p_nom=p_nom_val,
                p_nom_extendable=False,
                marginal_cost=0.0,
                efficiency=1.0,
                p_min_pu=0.0,
                p_max_pu=1.0,
            )
            logger.info(
                "  NegLoad::%s → Generator at bus '%s' (p_nom=%.2f MW).",
                load_name, bus, p_nom_val,
            )
        else:
            # Already exists (e.g. re-applied after reload) — update p_nom
            n.generators.at[gen_name, "p_nom"] = max(
                p_nom_val, float(n.generators.at[gen_name, "p_nom"]),
            )

        # Normalised p_max_pu timeseries
        p_max_pu_ts = (neg_ts / p_nom_val).reindex(n.snapshots).fillna(0.0)
        pmax_df[gen_name] = p_max_pu_ts

        # Clip original load to non-negative (demand side only)
        n_neg_snaps = int((p_set[load_name] < 0).sum())
        n.loads_t.p_set[load_name] = p_set[load_name].clip(lower=0.0)
        logger.info(
            "  Load '%s': %d snapshot(s) converted (max export=%.2f MW).",
            load_name, n_neg_snaps, p_nom_val,
        )

    n.generators_t.p_max_pu = pmax_df


# =============================================================================
# [DISPATCH-FIX-2] Unbounded store e_initial fix (extracted as reusable helper)
# =============================================================================

def _fix_inf_store_initial(n: pypsa.Network) -> None:
    """
    [DISPATCH-FIX-2] For stores with e_nom=inf and e_min_pu<0 (e.g. co2
    atmosphere), PyPSA adds an explicit constraint e[t=0] >= e_initial (=0)
    which conflicts with negative energy balance flows.

    Setting e_initial to a large negative value relaxes this constraint.
    This is methodically correct: an atmospheric CO2 store has no meaningful
    initial state — it is effectively unbounded in both directions.

    Applied in both master and dispatch.
    """
    if not hasattr(n, "stores") or len(n.stores) == 0:
        return
    for sname, srow in n.stores.iterrows():
        if np.isinf(float(srow.get("e_nom", 0))) and float(srow.get("e_min_pu", 0)) < 0:
            n.stores.at[sname, "e_initial"] = -1e6
            logger.info("Set e_initial=-1e6 for unbounded store: %s", sname)


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
    Stack N scenario networks into a single MultiIndex-snapshot network.
    Memory-optimized v2 + MEM-PATCH-A.
    """
    if len(scenario_files) != len(scenario_names):
        raise ValueError("scenario_files and scenario_names must have equal length.")
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError(f"Duplicate scenario names: {scenario_names}")

    n_scenarios = len(scenario_names)

    logger.info("Loading reference scenario=%s  network=%s", scenario_names[0], scenario_files[0])
    p0 = Path(scenario_files[0])
    if not p0.exists():
        raise FileNotFoundError(f"Scenario network not found: {scenario_files[0]}")
    ref = pypsa.Network(str(p0))

    if len(ref.snapshots) == 0:
        raise ValueError(f"Scenario {scenario_names[0]}: network has no snapshots.")
    if len(ref.buses) == 0:
        raise ValueError(f"Scenario {scenario_names[0]}: network has no buses.")

    base_snaps = _ensure_datetime_snapshots(ref.snapshots)
    n_snaps = len(base_snaps)

    n = pypsa.Network()
    for comp in _STATIC_COMPONENTS:
        if hasattr(ref, comp):
            try:
                val = getattr(ref, comp)
                if isinstance(val, pd.DataFrame):
                    setattr(n, comp, val.copy())
                elif val is not None:
                    setattr(n, comp, val)
            except Exception as exc:
                logger.warning("Could not copy static component '%s': %s", comp, exc)

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
        w_base = ref.snapshot_weightings.copy().reindex(base_snaps)
    else:
        w_base = pd.DataFrame(
            {"objective": 1.0, "generators": 1.0, "stores": 1.0},
            index=base_snaps,
        )
    w_np = np.tile(w_base.to_numpy(dtype=np.float32, copy=False), (n_scenarios, 1))
    n.snapshot_weightings = pd.DataFrame(
        w_np, index=stacked_snaps, columns=w_base.columns,
    )

    needed: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for t_container_name, allowed_attrs in _T_ATTRS_INPUT_ONLY.items():
        if not allowed_attrs or not hasattr(ref, t_container_name):
            continue
        ref_t = getattr(ref, t_container_name, None)
        if ref_t is None:
            continue

        found_attrs: Dict[str, pd.DataFrame] = {}
        for attr in allowed_attrs:
            try:
                val = getattr(ref_t, attr, None)
            except Exception:
                continue
            if val is None:
                continue
            if isinstance(val, pd.Series):
                val = val.to_frame()
            if isinstance(val, pd.DataFrame) and not val.empty:
                found_attrs[attr] = val

        if not found_attrs:
            continue

        needed[t_container_name] = {}
        for attr, ref_df in found_attrs.items():
            ref_aligned = ref_df.reindex(base_snaps).astype(np.float32)
            ref_arr = ref_aligned.to_numpy()
            all_cols = ref_aligned.columns

            col_min = np.nanmin(ref_arr, axis=0)
            col_max = np.nanmax(ref_arr, axis=0)
            is_constant = (col_max - col_min) < 1e-6

            variable_mask = ~is_constant
            variable_cols  = all_cols[variable_mask]
            constant_cols  = all_cols[is_constant]
            constant_vals  = ref_arr[0, is_constant].astype(np.float32, copy=False)
            n_variable = int(variable_mask.sum())

            needed[t_container_name][attr] = {
                "all_cols":      all_cols,
                "variable_cols": variable_cols,
                "constant_cols": constant_cols,
                "constant_vals": constant_vals,
                "has_variable":  n_variable > 0,
                "ref_arr":       ref_arr[:, variable_mask] if n_variable > 0 else None,
            }

    arrays: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for t_container_name, attrs in needed.items():
        arrays[t_container_name] = {}
        for attr, meta in attrs.items():
            if not meta["has_variable"]:
                arrays[t_container_name][attr] = None
                continue
            arr = np.empty(
                (n_scenarios * n_snaps, len(meta["variable_cols"])), dtype=np.float32,
            )
            arr[:n_snaps, :] = meta["ref_arr"]
            arrays[t_container_name][attr] = arr

    for i_scen in range(1, n_scenarios):
        scen_name = scenario_names[i_scen]
        scen_file = scenario_files[i_scen]
        logger.info("Loading scenario %d/%d: %s  file=%s",
                    i_scen + 1, n_scenarios, scen_name, scen_file)

        p_i = Path(scen_file)
        if not p_i.exists():
            raise FileNotFoundError(f"Scenario network not found: {scen_file}")

        net_i = pypsa.Network(str(p_i))
        if len(net_i.snapshots) == 0:
            raise ValueError(f"Scenario {scen_name}: network has no snapshots.")

        missing_buses = n.buses.index.difference(net_i.buses.index)
        extra_buses   = net_i.buses.index.difference(n.buses.index)
        if len(missing_buses) or len(extra_buses):
            msg = f"Bus mismatch in scenario '{scen_name}':\n"
            if len(missing_buses):
                msg += f"  Missing: {list(missing_buses)[:10]}\n"
            if len(extra_buses):
                msg += f"  Extra:   {list(extra_buses)[:10]}\n"
            raise ValueError(msg)

        snaps_i = _ensure_datetime_snapshots(net_i.snapshots)
        if len(snaps_i) != len(base_snaps) or not base_snaps.equals(snaps_i):
            raise ValueError(
                f"Snapshot mismatch: scenario[0] vs '{scen_name}'\n"
                f"  [0]: n={len(base_snaps)}, {base_snaps[0]} ... {base_snaps[-1]}\n"
                f"  [{i_scen}]: n={len(snaps_i)}, {snaps_i[0]} ... {snaps_i[-1]}\n"
            )

        start = i_scen * n_snaps
        end   = start + n_snaps

        for t_container_name, attrs in needed.items():
            if not hasattr(net_i, t_container_name):
                for attr, meta in attrs.items():
                    if meta["has_variable"] and arrays[t_container_name][attr] is not None:
                        arrays[t_container_name][attr][start:end, :] = np.nan
                continue

            net_t_i = getattr(net_i, t_container_name)
            for attr, meta in attrs.items():
                if not meta["has_variable"]:
                    continue
                arr  = arrays[t_container_name][attr]
                df_i = getattr(net_t_i, attr, None)
                if df_i is None:
                    arr[start:end, :] = np.nan
                    continue
                if isinstance(df_i, pd.Series):
                    df_i = df_i.to_frame()
                arr[start:end, :] = (
                    df_i
                    .reindex(index=base_snaps, columns=meta["variable_cols"])
                    .to_numpy(dtype=np.float32)
                )

        del net_i
        gc.collect()

    for t_container_name, attrs in needed.items():
        if not hasattr(n, t_container_name):
            continue
        n_t = getattr(n, t_container_name)

        for attr, meta in attrs.items():
            _apply_constant_cols_to_static(
                n,
                t_container_name=t_container_name,
                attr=attr,
                constant_cols=meta["constant_cols"],
                constant_vals=meta["constant_vals"],
            )

            if (meta["has_variable"]
                    and arrays[t_container_name][attr] is not None
                    and len(meta["variable_cols"]) > 0):
                df_var = pd.DataFrame(
                    arrays[t_container_name][attr],
                    index=stacked_snaps,
                    columns=meta["variable_cols"],
                    copy=False,
                ).astype(np.float32, copy=False)
                setattr(n_t, attr, df_var)
                del arrays[t_container_name][attr]

    del ref
    gc.collect()

    total_bytes = 0
    for tc in _T_ATTRS_INPUT_ONLY:
        if not hasattr(n, tc):
            continue
        n_t = getattr(n, tc)
        for attr in _T_ATTRS_INPUT_ONLY.get(tc, []):
            df = getattr(n_t, attr, None)
            if isinstance(df, pd.DataFrame):
                total_bytes += df.memory_usage(deep=False).sum()

    logger.info(
        "Stacked (MEM-v2+PATCH-A): %d scen × %d ts = %d snaps | "
        "buses=%d gens=%d loads=%d links=%d lines=%d su=%d stores=%d | "
        "*_t frames: %.1f MB",
        n_scenarios, n_snaps, len(n.snapshots),
        len(n.buses), len(n.generators), len(n.loads),
        len(n.links), len(n.lines), len(n.storage_units), len(n.stores),
        total_bytes / 1e6,
    )
    return n, list(scenario_names)


# =============================================================================
# Scenario masks
# =============================================================================

def _scenario_masks_from_snapshots(snapshots: pd.Index) -> Dict[str, np.ndarray]:
    if not isinstance(snapshots, pd.MultiIndex):
        raise ValueError(f"Expected pd.MultiIndex snapshots, got {type(snapshots)}.")
    if list(snapshots.names) != ["period", "timestep"]:
        raise ValueError(f"Expected names ['period','timestep'], got {snapshots.names}.")
    ts = snapshots.get_level_values("timestep")
    bad = [x for x in ts if not (isinstance(x, tuple) and len(x) >= 2)]
    if bad:
        raise ValueError(f"Expected (scenario, timestamp) tuples. Bad: {bad[:10]}")
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
        p_nom: float = 1e8,
) -> List[str]:
    if hasattr(n, "generators") and "carrier" in n.generators.columns:
        load_gens = n.generators.index[n.generators.carrier.astype(str) == "load"].tolist()
        if load_gens:
            n.generators.loc[load_gens, "marginal_cost"] = marginal_cost
            logger.info("Reset marginal_cost of %d 'load' generators to %.4g EUR/MWh.",
                        len(load_gens), marginal_cost)

    existing: List[str] = []
    if hasattr(n, "generators") and "carrier" in n.generators.columns:
        existing = n.generators.index[
            n.generators.carrier.astype(str).isin([carrier, "load"])
        ].tolist()
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
    missing_cols = [g for g in ls_names if g not in pmax.columns]
    if missing_cols:
        pmax = pd.concat(
            [pmax, pd.DataFrame(1.0, index=pmax.index, columns=missing_cols)],
            axis=1,
        )
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
# Ramp limits
# =============================================================================

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
        n.generators["committable"] = False
    else:
        saved["committable"] = False
    return saved


def _restore_ramp_limits(n: pypsa.Network, ramp_data: pd.DataFrame) -> None:
    """[BUG-FIX-4] Restore ramp limits and committable flags after solve."""
    if ramp_data is None or ramp_data.empty:
        return
    if not hasattr(n, "generators") or len(n.generators) == 0:
        return
    for col in _RAMP_COLS:
        if col in ramp_data.columns and col in n.generators.columns:
            n.generators[col] = ramp_data[col].reindex(n.generators.index)
    if "committable" in ramp_data.columns and "committable" in n.generators.columns:
        n.generators["committable"] = ramp_data["committable"].reindex(n.generators.index)
    logger.info("Ramp limits and committable flags restored.")


# =============================================================================
# Solver status
# =============================================================================

def _check_solver_status(
        n: pypsa.Network, stage: str, *, hard_fail_suboptimal: bool = True,
) -> bool:
    m = getattr(n, "model", None)
    if m is None:
        raise RuntimeError(f"[{stage}] n.model is None after optimize().")

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

    logger.info("[%s] status='%s'  termination='%s'", stage, status_str, termination)

    if status_str is None:
        sol = getattr(m, "solution", None)
        if sol is None:
            raise RuntimeError(f"[{stage}] No solution and no solver status available.")
        logger.warning("[%s] Status unavailable — solution exists, proceeding.", stage)
        return True

    fatal = ("infeasible", "unbounded", "error", "failed", "invalid")
    if any(t in (status_str or "") for t in fatal) or any(t in (termination or "") for t in fatal):
        raise RuntimeError(
            f"[{stage}] Infeasible (status='{status_str}', termination='{termination}')."
        )

    if status_str in ("ok", "optimal") or (status_str or "").startswith("optimal"):
        return True

    logger.warning("[%s] Non-optimal status: '%s'  termination: '%s'.",
                   stage, status_str, termination)
    if hard_fail_suboptimal:
        raise RuntimeError(f"[{stage}] Aborting: non-optimal status '{status_str}'.")
    return False


# =============================================================================
# Boundary constraints
# =============================================================================

def _add_scenario_boundary_constraints(
        network: pypsa.Network,
        scenarios: Sequence[str],
        masks: Dict[str, np.ndarray],
        cyclic_overrides: Optional[Dict[tuple, pd.Series]] = None,
) -> None:
    """
    [BUG-FIX-2] Guard changed from `if cyclic_overrides and ...` to
    `if cyclic_overrides is not None and ...` so an empty dict (falsy)
    is handled correctly.
    """
    m = network.model

    su_soc, su_name = _get_linopy_var(
        m, ["StorageUnit-state_of_charge", "StorageUnit-soc", "StorageUnit-energy"],
        strict=False,
    )
    if su_soc is not None:
        td = _get_time_dimension(su_soc)
        if cyclic_overrides is not None and ("storage_units", "cyclic_state_of_charge") in cyclic_overrides:
            su_cyclic_mask = cyclic_overrides[("storage_units", "cyclic_state_of_charge")]
        else:
            su_cyclic_mask = network.storage_units.get(
                "cyclic_state_of_charge",
                pd.Series(True, index=network.storage_units.index),
            )
        cyclic_sus = network.storage_units.index[su_cyclic_mask].tolist()
        asset_dim = [d for d in su_soc.dims if d != td][0]
        for s in scenarios:
            idx = _mask_to_isel_indices(masks[s])
            if len(idx) >= 2 and cyclic_sus:
                soc_c = su_soc.sel({asset_dim: cyclic_sus})
                m.add_constraints(
                    soc_c.isel({td: int(idx[0])}) == soc_c.isel({td: int(idx[-1])}),
                    name=f"boundary::StorageUnit::cyclic::{s}",
                )
        logger.info("StorageUnit cyclic SOC constraints added (var=%s, %d cyclic).",
                    su_name, len(cyclic_sus))

    st_e, st_name = _get_linopy_var(
        m, ["Store-e", "Store-energy", "Store-state_of_charge"], strict=False,
    )
    if st_e is not None:
        td = _get_time_dimension(st_e)
        if cyclic_overrides is not None and ("stores", "e_cyclic") in cyclic_overrides:
            st_cyclic_mask = cyclic_overrides[("stores", "e_cyclic")]
        else:
            st_cyclic_mask = network.stores.get(
                "e_cyclic", pd.Series(True, index=network.stores.index),
            )
        cyclic_stores = network.stores.index[st_cyclic_mask].tolist()
        asset_dim = [d for d in st_e.dims if d != td][0]
        for s in scenarios:
            idx = _mask_to_isel_indices(masks[s])
            if len(idx) >= 2 and cyclic_stores:
                st_ec = st_e.sel({asset_dim: cyclic_stores})
                m.add_constraints(
                    st_ec.isel({td: int(idx[0])}) == st_ec.isel({td: int(idx[-1])}),
                    name=f"boundary::Store::cyclic::{s}",
                )
        logger.info("Store cyclic energy constraints added (var=%s, %d cyclic).",
                    st_name, len(cyclic_stores))


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
    import xarray as xr

    if ramp_data is None or ramp_data.empty:
        return

    m = network.model
    gen_p, _ = _get_linopy_var(m, ["Generator-p"], strict=False)
    if gen_p is None:
        logger.info("No Generator-p variable; skipping ramp constraints.")
        return

    td      = _get_time_dimension(gen_p)
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
        logger.warning(msg + " Skipping.")

    gens_to_constrain = list(
        network.generators.index[~ext_mask].intersection(gens_with_ramp)
    )
    if not gens_to_constrain:
        return

    rup_vals  = ramp_data.loc[gens_to_constrain, "ramp_limit_up"]
    rdn_vals  = ramp_data.loc[gens_to_constrain, "ramp_limit_down"]
    p_nom_v   = network.generators.loc[gens_to_constrain, "p_nom"].astype(float)

    n_added = 0
    for s in scenarios:
        idx  = _mask_to_isel_indices(masks[s])
        if len(idx) < 2:
            continue
        prev = idx[:-1].tolist()
        nxt  = idx[1:].tolist()
        pair_coord = np.arange(len(prev))

        p_batch = gen_p.sel({gen_dim: gens_to_constrain})
        p_prev  = p_batch.isel({td: prev}).assign_coords({td: pair_coord})
        p_next  = p_batch.isel({td: nxt}).assign_coords({td: pair_coord})

        if rup_vals.notna().any():
            rup_lim = (rup_vals.fillna(np.inf) * p_nom_v).to_numpy()
            rup_da  = xr.DataArray(rup_lim, dims=[gen_dim],
                                   coords={gen_dim: gens_to_constrain})
            m.add_constraints(p_next - p_prev <= rup_da, name=f"ramp_up::{s}")
            n_added += len(prev) * len(gens_to_constrain)

        if rdn_vals.notna().any():
            rdn_lim = (rdn_vals.fillna(np.inf) * p_nom_v).to_numpy()
            rdn_da  = xr.DataArray(rdn_lim, dims=[gen_dim],
                                   coords={gen_dim: gens_to_constrain})
            m.add_constraints(p_prev - p_next <= rdn_da, name=f"ramp_down::{s}")
            n_added += len(prev) * len(gens_to_constrain)

    if n_added > 0:
        logger.info("Added %d within-scenario ramp constraints (vectorised).", n_added)


# =============================================================================
# Cost expressions
# =============================================================================

def _build_investment_cost_expression(n: pypsa.Network):
    import xarray as xr
    m = n.model
    expr = 0

    def _add(comp_df, var_names, dim, ext_col="p_nom_extendable"):
        nonlocal expr
        if comp_df is None or len(comp_df) == 0 or "capital_cost" not in comp_df.columns:
            return
        if ext_col in comp_df.columns:
            comp_df = comp_df[comp_df[ext_col].fillna(False).astype(bool)]
        if len(comp_df) == 0:
            return
        var, _ = _get_linopy_var(m, var_names, strict=False)
        if var is None:
            return
        cc    = comp_df["capital_cost"].reindex(comp_df.index).fillna(0.0)
        cc_da = xr.DataArray(cc.to_numpy(), dims=[dim],
                             coords={dim: (dim, comp_df.index.to_numpy())})
        expr = expr + (var * cc_da).sum()

    _add(n.generators,    ["Generator-p_nom"],   "Generator")
    _add(n.links,         ["Link-p_nom"],         "Link")
    _add(n.storage_units, ["StorageUnit-p_nom"],  "StorageUnit")
    _add(n.stores,        ["Store-e_nom"],         "Store",       ext_col="e_nom_extendable")
    _add(n.lines,         ["Line-s_nom"],          "Line",        ext_col="s_nom_extendable")
    _add(n.transformers,  ["Transformer-s_nom"],   "Transformer", ext_col="s_nom_extendable")
    return expr


def _build_co2_cost_expression(n: pypsa.Network, mask: np.ndarray) -> Any:
    import xarray as xr

    if not hasattr(n, "global_constraints") or len(n.global_constraints) == 0:
        return 0
    gc_df = n.global_constraints
    if "type" not in gc_df.columns or "constant_cost" not in gc_df.columns:
        return 0

    co2_rows = gc_df[
        gc_df["type"].astype(str).str.lower().isin(["primary_energy", "co2"])
        & gc_df["constant_cost"].fillna(0.0).gt(0)
    ]
    if co2_rows.empty:
        logger.warning("CO2 mode active but no matching GlobalConstraint rows.")
        return 0

    m = n.model
    gen_p, _ = _get_linopy_var(m, ["Generator-p"], strict=False)
    if gen_p is None:
        return 0
    if "carrier" not in n.generators.columns:
        return 0

    td      = _get_time_dimension(gen_p)
    gen_dim = _get_gen_dimension(gen_p)
    idx     = _mask_to_isel_indices(mask)
    w       = _weights_objective_series(n).to_numpy()[idx].astype(np.float32, copy=False)

    total_co2 = 0
    for _, row in co2_rows.iterrows():
        co2_price    = float(row["constant_cost"])
        carrier_attr = row.get("carrier_attribute", "co2_emissions")
        if carrier_attr not in n.carriers.columns:
            logger.warning("CO2: carrier attribute '%s' not in n.carriers.", carrier_attr)
            continue
        ef = n.generators["carrier"].map(n.carriers[carrier_attr]).fillna(0.0).astype(float)
        if (ef == 0).all():
            continue
        ef_da  = xr.DataArray(ef.to_numpy(), dims=[gen_dim],
                              coords={gen_dim: (gen_dim, n.generators.index.to_numpy())})
        var_s  = gen_p.isel({td: idx.tolist()})
        w_da   = xr.DataArray(w, dims=[td], coords={td: var_s.coords[td]})
        total_co2 = total_co2 + co2_price * (var_s * ef_da * w_da).sum()

    return total_co2


def _build_operational_cost_expression(
        n: pypsa.Network,
        mask: np.ndarray,
        *,
        co2_cost_mode: str = "off",
        w_override_np: Optional[np.ndarray] = None,
        annual_scale: float = 1.0,
):
    """Canonical operational-cost expression. [MEM-PATCH-B]"""
    import xarray as xr

    m = n.model
    idx      = _mask_to_isel_indices(mask)
    idx_list = idx.tolist()

    if w_override_np is not None:
        w_np = np.asarray(w_override_np, dtype=np.float32)
        if w_np.shape[0] != len(n.snapshots):
            raise ValueError(
                f"w_override_np length mismatch: {w_np.shape[0]} vs {len(n.snapshots)}"
            )
    else:
        w_np = _weights_objective_series(n).to_numpy(dtype=np.float32, copy=False)

    if abs(annual_scale - 1.0) > 1e-9:
        w_np = w_np * np.float32(annual_scale)

    total = 0

    def _var_isel(var, td_name):
        return var.isel({td_name: idx_list})

    def _w_da(var_s, td_name):
        return xr.DataArray(w_np[idx_list], dims=[td_name],
                            coords={td_name: var_s.coords[td_name]})

    def _add_mc(comp_df, mc_t, var_names, dim):
        nonlocal total
        if comp_df is None or len(comp_df) == 0 or "marginal_cost" not in comp_df.columns:
            return
        var, _ = _get_linopy_var(m, var_names, strict=False)
        if var is None:
            return
        td    = _get_time_dimension(var)
        var_s = _var_isel(var, td)
        wd    = _w_da(var_s, td)
        mc_s  = comp_df["marginal_cost"].reindex(comp_df.index).fillna(0.0).astype(float)

        if mc_t is not None and not mc_t.empty:
            mc_t_s = (mc_t.reindex(columns=comp_df.index)
                      .reindex(index=n.snapshots[idx_list]).astype(float).fillna(mc_s))
            mc_da = xr.DataArray(mc_t_s.to_numpy(), dims=[td, dim],
                                 coords={td: var_s.coords[td],
                                         dim: (dim, comp_df.index.to_numpy())})
        else:
            mc_da = xr.DataArray(mc_s.to_numpy(), dims=[dim],
                                 coords={dim: (dim, comp_df.index.to_numpy())})

        total = total + (var_s * mc_da * wd).sum()

    def _add_uc_costs():
        nonlocal total
        if not hasattr(n, "generators") or len(n.generators) == 0:
            return
        v_start,  _ = _get_linopy_var(m, ["Generator-start_up",  "Generator-startup",  "Generator-start"],  strict=False)
        v_shut,   _ = _get_linopy_var(m, ["Generator-shut_down",  "Generator-shutdown", "Generator-shut"],   strict=False)
        v_status, _ = _get_linopy_var(m, ["Generator-status",     "Generator-committable", "Generator-u"],   strict=False)
        su_c = n.generators.get("start_up_cost",  pd.Series(0.0, index=n.generators.index)).fillna(0.0).astype(float)
        sd_c = n.generators.get("shut_down_cost", pd.Series(0.0, index=n.generators.index)).fillna(0.0).astype(float)
        nl_c = n.generators.get("no_load_cost",   pd.Series(0.0, index=n.generators.index)).fillna(0.0).astype(float)
        if (su_c == 0).all() and (sd_c == 0).all() and (nl_c == 0).all():
            return

        def _add_uc(var, cs):
            nonlocal total
            if var is None:
                return
            td  = _get_time_dimension(var)
            gd  = _get_gen_dimension(var)
            vs  = _var_isel(var, td)
            wd  = _w_da(vs, td)
            c   = cs.reindex(n.generators.index).fillna(0.0).astype(float)
            c_da = xr.DataArray(c.to_numpy(), dims=[gd],
                                coords={gd: (gd, n.generators.index.to_numpy())})
            total = total + (vs * c_da * wd).sum()

        _add_uc(v_start,  su_c)
        _add_uc(v_shut,   sd_c)
        _add_uc(v_status, nl_c)

    _add_mc(n.generators,    getattr(getattr(n, "generators_t",    None), "marginal_cost", None), ["Generator-p"],                            "Generator")
    _add_mc(n.storage_units, getattr(getattr(n, "storage_units_t", None), "marginal_cost", None), ["StorageUnit-p_dispatch", "StorageUnit-p"], "StorageUnit")
    _add_mc(n.stores,        getattr(getattr(n, "stores_t",        None), "marginal_cost", None), ["Store-p"],                                "Store")
    _add_mc(n.links,         getattr(getattr(n, "links_t",         None), "marginal_cost", None), ["Link-p0"],                                "Link")
    _add_uc_costs()

    if co2_cost_mode == "global_constraint_constant_cost":
        total = total + _build_co2_cost_expression(n, mask)
    elif co2_cost_mode != "off":
        raise ValueError(f"Unknown co2_cost_mode='{co2_cost_mode}'.")

    return total


# =============================================================================
# Cost-consistency validation
# =============================================================================

def _validate_dispatch_cost_consistency(
        n: pypsa.Network,
        pypsa_objective: float,
        mask: np.ndarray,
        scenario: str,
        *,
        rel_tol: float = 0.01,
        co2_cost_mode: str = "off",
        annual_scale: float = 1.0,
) -> Dict[str, Any]:
    """[BUG-FIX-8] annual_scale added for consistent manual cost recomputation."""
    result: Dict[str, Any] = {
        "pypsa_objective": float(pypsa_objective),
        "manual_cost": None,
        "rel_diff": None,
        "consistent": None,
    }
    try:
        manual = float(
            _build_operational_cost_expression(
                n, mask,
                co2_cost_mode=co2_cost_mode,
                annual_scale=annual_scale,
            ).evaluate()
        )
        result["manual_cost"] = manual
        rel_diff = abs(pypsa_objective - manual) / max(1.0, abs(pypsa_objective))
        result["rel_diff"] = rel_diff
        result["consistent"] = rel_diff < rel_tol
        if rel_diff >= rel_tol:
            logger.warning(
                "[CostConsistency] MISMATCH '%s': dispatch=%.6g manual=%.6g rel_diff=%.3f%%.",
                scenario, pypsa_objective, manual, rel_diff * 100,
            )
        else:
            logger.info("[CostConsistency] OK '%s': rel_diff=%.4f%%.", scenario, rel_diff * 100)
    except Exception as exc:
        logger.warning("[CostConsistency] Validation failed for '%s' (non-fatal): %s", scenario, exc)
    return result


# =============================================================================
# Capacity helpers
# =============================================================================

def _fix_portfolio_capacities(n: pypsa.Network, port_net: pypsa.Network) -> None:
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
        common   = df_new.index.intersection(df_old.index)
        only_new = df_new.index.difference(df_old.index)
        if attr_cap in df_new.columns and len(only_new) > 0:
            df_new.loc[only_new, attr_cap] = 0.0
        if attr_cap in df_new.columns and len(common) > 0:
            df_new.loc[common, attr_cap] = (
                df_old.loc[common, attr_opt].fillna(0.0).astype(float)
            )


def _assert_no_extendables(n: pypsa.Network) -> None:
    for comp, _, __, attr_ext in _CAPACITY_MAP:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0 or attr_ext not in df.columns:
            continue
        mask = df[attr_ext].fillna(False).astype(bool)
        if mask.any():
            bad = df.index[mask].tolist()[:10]
            raise RuntimeError(
                f"{comp}: still extendable after _fix_portfolio_capacities. "
                f"Examples: {bad}."
            )


def _compute_portfolio_investment_cost(port_net: pypsa.Network) -> float:
    total = 0.0
    for comp, attr_opt, _, attr_ext in _CAPACITY_MAP:
        df = getattr(port_net, comp, None)
        if df is None or len(df) == 0:
            continue
        if attr_opt not in df.columns or "capital_cost" not in df.columns:
            continue
        if attr_ext in df.columns:
            ext = df[df[attr_ext].fillna(False).astype(bool)]
        else:
            ext = df
        if len(ext) == 0:
            continue
        cost = (
            ext[attr_opt].fillna(0.0).astype(float)
            * ext["capital_cost"].fillna(0.0).astype(float)
        ).sum()
        total += float(cost)
    logger.debug("Portfolio investment cost: %.6g EUR/a", total)
    return total


def _extract_objective_value(n: pypsa.Network) -> float:
    model = getattr(n, "model", None)
    if model is None:
        raise RuntimeError("n.model is None after optimize().")
    obj = getattr(model, "objective_value", None)
    if obj is not None:
        return float(obj)
    try:
        return float(model.objective.value)
    except Exception:
        pass
    sm = getattr(model, "solver_model", None)
    if sm is not None:
        try:
            return float(sm.ObjVal)
        except Exception:
            pass
    raise RuntimeError("Could not extract objective value from model.")


def _get_solution_dict(n: pypsa.Network) -> Dict:
    m = getattr(n, "model", None)
    if m is None:
        raise RuntimeError("n.model is None.")
    sol = getattr(m, "solution", None)
    if sol is None:
        raise RuntimeError("n.model.solution is None after optimize().")
    return sol


def _extract_scalar_solution(sol: Dict, key: str) -> float:
    """[FIX-3] Multi-fallback scalar extraction for different Linopy versions."""
    if key not in sol:
        raise RuntimeError(f"Key '{key}' not found in solution. Available: {list(sol.keys())}")
    val = sol[key]
    item_fn = getattr(val, "item", None)
    if item_fn is not None:
        try:
            return float(item_fn())
        except Exception:
            pass
    values = getattr(val, "values", None)
    if values is not None:
        try:
            return float(np.squeeze(values))
        except Exception:
            pass
    try:
        return float(val)
    except Exception:
        pass
    raise RuntimeError(f"Cannot extract scalar from solution['{key}']: {type(val)} = {val}")


# =============================================================================
# Dispatch helpers
# =============================================================================

def _compute_dispatch_annual_scale(n: pypsa.Network) -> float:
    """[BUG-FIX-7] Compute annual_scale from actual snapshot weightings."""
    sw = getattr(n, "snapshot_weightings", None)
    if sw is not None and "objective" in sw.columns:
        total_hours = float(sw["objective"].sum())
    else:
        total_hours = float(len(n.snapshots))
    return 8760.0 / max(1.0, total_hours)


def _dispatch_solve(
        n: pypsa.Network,
        *,
        solver_name: str,
        solver_options: Optional[Dict],
        co2_cost_mode: str = "off",
        ls_penalty: float = 1e4,
) -> float:
    """
    Dispatch-only solve with fixed capacities.

    [DISPATCH-FIX-1] _fix_negative_loads — converts net-export nodes to
        zero-cost generators instead of clipping to 0. This is the
        methodically correct treatment that preserves nodal energy balance
        and prevents artificial infeasibility.
    [DISPATCH-FIX-2] _fix_inf_store_initial — relaxes e_initial for
        unbounded stores (e.g. co2 atmosphere).
    [DISPATCH-FIX-3] _prune_zero_capacity_assets — reduces model size to
        match master (dispatch was 2.7x larger without this).
    [BUG-FIX-7]      annual_scale computed from actual snapshot weightings.
    """
    if solver_options is None:
        solver_options = {}

    # [DISPATCH-FIX-1] Convert negative loads to generators (methodically correct)
    _fix_negative_loads(n)

    # [DISPATCH-FIX-2] Relax e_initial for unbounded stores
    _fix_inf_store_initial(n)

    # [DISPATCH-FIX-3] Prune zero-capacity assets to match master model size
    _prune_zero_capacity_assets(n)

    # [BUG-FIX-7] Consistent annual_scale and LS penalty
    annual_scale   = _compute_dispatch_annual_scale(n)
    ls_mc_adjusted = ls_penalty / annual_scale

    _ensure_load_shedding_generators(
        n, carrier="load_shedding", marginal_cost=ls_mc_adjusted,
    )

    def extra_dispatch(network: pypsa.Network, snapshots: pd.Index) -> None:
        m = network.model
        mask_all = np.ones(len(network.snapshots), dtype=bool)
        op_cost = _build_operational_cost_expression(
            network, mask_all,
            co2_cost_mode=co2_cost_mode,
            annual_scale=annual_scale,
        )
        m.objective = op_cost

    n.optimize(
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=extra_dispatch,
        assign_all_duals=True,
        io_api="direct",
    )

    _check_solver_status(n, "Dispatch", hard_fail_suboptimal=True)
    return _extract_objective_value(n)


def evaluate_single_cutout_dispatch(
        *,
        cutout: str,
        portfolio_path: str,
        scen_net_path: str,
        solver_name: str,
        solver_options: Optional[Dict],
        dispatch_tmp_dir: str,
        investment_cost: float,
        cost_consistency_tol: float = 0.01,
        co2_cost_mode: str = "off",
        ls_penalty: float = 1e4,
) -> Dict[str, Any]:
    """
    Evaluate a fixed portfolio on a single scenario (cutout).
    All preprocessing fixes are applied inside _dispatch_solve.
    """
    logger.info("Dispatch eval: cutout='%s'  portfolio='%s'", cutout, portfolio_path)

    p_scen = Path(scen_net_path)
    if not p_scen.exists():
        raise FileNotFoundError(f"Scenario network not found: {scen_net_path}")
    n = pypsa.Network(str(p_scen))

    port_net = pypsa.Network(str(portfolio_path))
    _fix_portfolio_capacities(n, port_net)
    del port_net
    gc.collect()

    _assert_no_extendables(n)

    dispatch_op_cost = _dispatch_solve(
        n,
        solver_name=solver_name,
        solver_options=solver_options,
        co2_cost_mode=co2_cost_mode,
        ls_penalty=ls_penalty,
    )
    total_cost = investment_cost + dispatch_op_cost
    logger.info("Dispatch '%s': op_cost=%.6g  inv_cost=%.6g  total=%.6g",
                cutout, dispatch_op_cost, investment_cost, total_cost)

    annual_scale = _compute_dispatch_annual_scale(n)
    mask_all    = np.ones(len(n.snapshots), dtype=bool)
    consistency = _validate_dispatch_cost_consistency(
        n, dispatch_op_cost, mask_all, cutout,
        rel_tol=cost_consistency_tol,
        co2_cost_mode=co2_cost_mode,
        annual_scale=annual_scale,
    )

    Path(dispatch_tmp_dir).mkdir(parents=True, exist_ok=True)
    flat_path = str(Path(dispatch_tmp_dir) / f"dispatch_{cutout}_flat.nc")
    std_path  = str(Path(dispatch_tmp_dir) / f"dispatch_{cutout}_std.nc")

    _ensure_standard_pypsa_result_frames(n)
    export_network_flat_snapshots(n, flat_path)

    n_std = pypsa.Network(str(p_scen))
    n_std.set_snapshots(_ensure_datetime_snapshots(n.snapshots))
    for t_name, attrs in _T_ATTRS.items():
        if not hasattr(n, t_name) or not hasattr(n_std, t_name):
            continue
        src = getattr(n, t_name)
        dst = getattr(n_std, t_name)
        for attr in attrs:
            df = getattr(src, attr, None)
            if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
                continue
            df_std = df.copy()
            df_std.index = n_std.snapshots
            setattr(dst, attr, df_std)

    for comp, attr_opt, _, __ in _CAPACITY_MAP:
        df_src = getattr(n, comp, None)
        df_dst = getattr(n_std, comp, None)
        if df_src is None or df_dst is None:
            continue
        if attr_opt in df_src.columns:
            common = df_dst.index.intersection(df_src.index)
            if len(common) > 0:
                df_dst.loc[common, attr_opt] = df_src.loc[common, attr_opt]

    n_std.export_to_netcdf(std_path)
    del n_std
    gc.collect()

    return {
        "cutout":      cutout,
        "total_cost":  float(total_cost),
        "flat_path":   flat_path,
        "std_path":    std_path,
        "consistency": consistency,
    }


# =============================================================================
# Prune zero-capacity assets
# =============================================================================

def _prune_zero_capacity_assets(n: pypsa.Network) -> Dict[str, List[str]]:
    _COMP_MAP = [
        ("generators",    "p_nom", "p_nom_extendable", "generators_t"),
        ("links",         "p_nom", "p_nom_extendable", "links_t"),
        ("storage_units", "p_nom", "p_nom_extendable", "storage_units_t"),
        ("stores",        "e_nom", "e_nom_extendable", "stores_t"),
        ("lines",         "s_nom", "s_nom_extendable", "lines_t"),
    ]
    pruned: Dict[str, List[str]] = {}
    for comp, cap_col, ext_col, t_container in _COMP_MAP:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0:
            continue
        if cap_col not in df.columns or ext_col not in df.columns:
            continue
        is_zero  = df[cap_col].fillna(0.0).astype(float) == 0.0
        is_fixed = ~df[ext_col].fillna(False).astype(bool)
        dead     = df.index[is_zero & is_fixed].tolist()
        if not dead:
            continue

        pruned[comp] = dead
        setattr(n, comp, df.drop(index=dead))

        t_obj = getattr(n, t_container, None)
        if t_obj is not None:
            for attr in list(vars(t_obj)):
                val = getattr(t_obj, attr, None)
                if isinstance(val, pd.DataFrame) and len(val.columns) > 0:
                    cols_to_drop = val.columns.intersection(dead)
                    if len(cols_to_drop) > 0:
                        setattr(t_obj, attr, val.drop(columns=cols_to_drop))

        logger.info("Pruned %d zero-capacity %s (e.g. %s)", len(dead), comp, dead[:3])

    total = sum(len(v) for v in pruned.values())
    if total:
        logger.info("Total pruned: %d across %d component types.", total, len(pruned))
    return pruned


def _fix_zero_capital_cost_extendables(n: pypsa.Network) -> int:
    fixed = 0
    for comp, ext_col in [
        ("generators",    "p_nom_extendable"),
        ("links",         "p_nom_extendable"),
        ("storage_units", "p_nom_extendable"),
        ("stores",        "e_nom_extendable"),
        ("lines",         "s_nom_extendable"),
    ]:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0:
            continue
        if ext_col not in df.columns or "capital_cost" not in df.columns:
            continue
        is_ext  = df[ext_col].fillna(False).astype(bool)
        is_zero = df.loc[is_ext, "capital_cost"].fillna(0.0).astype(float) == 0.0
        bad     = df.index[is_ext][is_zero].tolist()
        if bad:
            df.loc[bad, "capital_cost"] = 1.0
            logger.warning(
                "_fix_zero_capital_cost_extendables: %d extendable %s had "
                "capital_cost=0 → set to 1 EUR/MW. Examples: %s",
                len(bad), comp, bad[:3],
            )
            fixed += len(bad)
    return fixed


# =============================================================================
# [ARO-FIX-1]  Break cross-scenario SOC/energy balance coupling
# =============================================================================

def _fix_cross_scenario_storage_constraints(
        network: pypsa.Network,
        scenarios: Sequence[str],
        masks: Dict[str, np.ndarray],
) -> int:
    """
    [ARO-FIX-1] Remove PyPSA's auto-generated SOC/energy balance constraints
    at scenario boundaries so that ARO scenarios are fully independent.

    BACKGROUND
    ----------
    PyPSA's linopy backend generates a balance constraint for every timestep t:

      StorageUnit:  soc[t] - (1-sl)*soc[t-1] - eff_s*p_store[t]*w
                            + p_disp[t]/eff_d*w - inflow[t]*w = 0
      Store:        e[t]   - (1-sl)*e[t-1]   - (-p[t]*w)              = 0

    In the stacked multi-scenario network the snapshots are ordered:
      [scen0_t0, ..., scen0_tN, scen1_t0, ..., scen1_tN, scen2_t0, ...]

    At position scen_i_t0 (first timestep of scenario i > 0), t-1 is
    scen_{i-1}_tN — the last step of the PREVIOUS scenario. This creates
    a cross-scenario coupling that violates the ARO independence assumption.

    FIX
    ---
    After PyPSA builds the linopy model:
    1. Identify the balance constraint row at each boundary position.
    2. Remove the full constraint, rebuild it for all non-boundary timesteps.
    3. The intra-scenario cyclic constraints (soc[first] == soc[last]),
       already added by _add_scenario_boundary_constraints, provide the
       correct closure at the removed positions.

    PRECONDITION
    ------------
    _add_scenario_boundary_constraints MUST have been called before this
    function so the cyclic closure is in place before we remove the rows.

    ASSUMPTION
    ----------
    Each ARO scenario achieves its own independent cyclic storage equilibrium.
    Storage energy cannot be transferred between scenarios.

    RETURN
    ------
    Number of constraint rows successfully removed (0 if nothing to fix or
    if the fix failed non-fatally).
    """
    import xarray as xr

    m = network.model

    if len(scenarios) <= 1:
        logger.debug("[ARO-FIX-1] Single scenario — no cross-scenario boundary to fix.")
        return 0

    # Global (0-indexed) snapshot positions that open a new scenario (skip scen 0)
    boundary_positions: List[int] = []
    for s in scenarios[1:]:
        idx = _mask_to_isel_indices(masks[s])
        if len(idx) > 0:
            boundary_positions.append(int(idx[0]))

    if not boundary_positions:
        return 0

    n_snaps = len(network.snapshots)
    n_fixed = 0

    # Candidate constraint names produced by different PyPSA / linopy versions
    candidates: List[Tuple[str, List[str]]] = [
        ("StorageUnit", [
            "StorageUnit-soc_balance",
            "StorageUnit-state_of_charge_balance",
            "StorageUnit-storage_balance",
        ]),
        ("Store", [
            "Store-e_balance",
            "Store-energy_balance",
            "Store-storage_balance",
        ]),
    ]

    for comp_label, con_name_list in candidates:
        con_key = next((k for k in con_name_list if k in m.constraints), None)
        if con_key is None:
            logger.debug(
                "[ARO-FIX-1] No balance constraint found for %s "
                "(tried: %s). Skipping.",
                comp_label, con_name_list,
            )
            continue

        con = m.constraints[con_key]

        # Identify the time dimension
        td = next(
            (d for d in ("timestep", "snapshot", "snapshots", "time")
             if d in getattr(con, "dims", ())),
            None,
        )
        if td is None:
            logger.warning(
                "[ARO-FIX-1] Cannot identify time dimension in '%s' "
                "(dims=%s). Cross-scenario coupling NOT removed for %s.",
                con_key, getattr(con, "dims", "?"), comp_label,
            )
            continue

        all_time_vals = list(con.coords[td].values)
        n_con = len(all_time_vals)

        # PyPSA may produce the balance for ALL snapshots (offset=0) if
        # e_cyclic=False and e_initial is set, or for snapshots 1..T
        # (offset=1) if it uses soc[t-1] for t>0 only.
        # We detect this by comparing constraint length to total snapshot count.
        if n_con == n_snaps:
            offset = 0
        elif n_con == n_snaps - 1:
            offset = 1   # constraint starts at global index 1
        else:
            logger.warning(
                "[ARO-FIX-1] Unexpected constraint '%s' length %d "
                "(expected %d or %d). Cross-scenario fix SKIPPED for %s.",
                con_key, n_con, n_snaps, n_snaps - 1, comp_label,
            )
            continue

        # Map global boundary positions to constraint-local row indices
        remove_rows: List[int] = []
        for bp in boundary_positions:
            con_row = bp - offset
            if 0 <= con_row < n_con:
                remove_rows.append(con_row)

        if not remove_rows:
            logger.info(
                "[ARO-FIX-1] '%s': boundary positions %s all outside "
                "constraint row range [%d, %d]. Nothing removed.",
                con_key, boundary_positions, offset, offset + n_con - 1,
            )
            continue

        keep_rows = [i for i in range(n_con) if i not in set(remove_rows)]

        # ------------------------------------------------------------------
        # Rebuild constraint: remove → re-add without boundary rows
        # ------------------------------------------------------------------
        try:
            lhs = con.lhs                            # LinearExpression
            rhs = con.rhs                            # xr.DataArray or scalar

            lhs_filtered = lhs.isel({td: keep_rows})
            rhs_filtered = (
                rhs.isel({td: keep_rows})
                if isinstance(rhs, xr.DataArray)
                else rhs
            )

            m.remove_constraints(con_key)

            if keep_rows:
                m.add_constraints(
                    lhs_filtered, "=", rhs_filtered, name=con_key,
                )

            n_fixed += len(remove_rows)
            logger.info(
                "[ARO-FIX-1] '%s': removed %d cross-scenario boundary rows "
                "at global positions %s (scenarios: %s). "
                "Intra-scenario cyclic constraints provide SOC closure.",
                con_key,
                len(remove_rows),
                [bp for bp in boundary_positions if 0 <= bp - offset < n_con],
                scenarios[1:],
            )

        except Exception as exc:
            logger.error(
                "[ARO-FIX-1] Failed to rebuild '%s' after removing boundary "
                "rows: %s.\n"
                "  *** Cross-scenario SOC coupling is NOT removed. ***\n"
                "  ARO independence assumption is VIOLATED for %s.\n"
                "  Inspect linopy version compatibility.",
                con_key, exc, comp_label,
            )
            # Attempt to restore the original constraint to avoid a broken model
            try:
                if con_key not in m.constraints:
                    m.add_constraints(con.lhs, "=", con.rhs, name=con_key)
            except Exception:
                pass

    if n_fixed > 0:
        logger.info(
            "[ARO-FIX-1] Total cross-scenario balance rows removed: %d. "
            "Scenarios are now dynamically independent.",
            n_fixed,
        )
    else:
        logger.warning(
            "[ARO-FIX-1] No cross-scenario balance rows were removed. "
            "Check constraint names in the linopy model — "
            "cross-scenario SOC coupling may still be present.",
        )

    return n_fixed


# =============================================================================
# C&CG Master Problem
# =============================================================================

def solve_aro_master(
        n: pypsa.Network,
        *,
        solver_name: str,
        solver_options: Optional[Dict] = None,
        load_shedding_carrier: str = "load_shedding",
        ls_penalty: float = 1e4,
        hard_fail_suboptimal: bool = True,
        fail_on_extendable_ramps: bool = True,
        co2_cost_mode: str = "off",
) -> Dict[str, Any]:
    """
    C&CG Master Problem for ARO over stacked scenario network.

    Mathematical formulation
    ─────────────────────────
    min_{x, z_theta}   Inv(x) + z_theta
    s.t.               z_theta >= Op(x, s)   for all s in S_k
                       x >= 0

    z_theta is an epigraph variable representing the worst-case operational
    cost across all scenarios in the current master set S_k.

    Fixes applied
    ─────────────
    [MEM-PATCH-B]   No snapshot_weightings mutation; w_override_np + annual_scale.
    [FIX-2]         LS generators: marginal_cost = ls_penalty / annual_scale.
    [FIX-3]         z_theta extracted via multi-fallback _extract_scalar_solution.
    [BUG-FIX-1]     cyclic_overrides = e_cyclic_backup (not {}) in extra_master.
    [BUG-FIX-2]     is not None guard in _add_scenario_boundary_constraints.
    [BUG-FIX-3]     GlobalConstraint constants saved and restored after solve.
    [BUG-FIX-4]     Ramp limits restored via _restore_ramp_limits after solve.
    [BUG-FIX-5]     e_cyclic_backup restored after solve.
    [BUG-FIX-6]     IIS debug gated behind ARO_IIS_DEBUG env variable.
    [DISPATCH-FIX-1] _fix_negative_loads: net-export → zero-cost Generator.
    [DISPATCH-FIX-2] _fix_inf_store_initial: e_initial=-1e6 for CO2 stores.
    [ARO-FIX-1]     _fix_cross_scenario_storage_constraints: remove PyPSA's
                    auto-generated SOC balance rows at scenario boundaries so
                    that each scenario is dynamically independent.
    [ARO-FIX-2]     z_theta lower bound = -inf (not 0). The epigraph
                    constraints enforce z_theta >= Op(x,s). Imposing lower=0
                    is methodically wrong and can yield sub-optimal solutions
                    when any operational cost is negative.
    """
    if solver_options is None:
        solver_options = {}

    ramp_data = _save_and_clear_ramp_limits(n)
    pruned    = _prune_zero_capacity_assets(n)
    _fix_zero_capital_cost_extendables(n)
    masks     = _scenario_masks_from_snapshots(n.snapshots)
    scenarios = list(masks.keys())

    hours_per_scenario = (
        float(n.snapshot_weightings["objective"].sum()) / max(1, len(scenarios))
    )
    annual_scale = 8760.0 / float(hours_per_scenario)

    # [FIX-2] LS penalty divided by annual_scale
    ls_mc_adjusted = ls_penalty / annual_scale
    _ensure_load_shedding_generators(
        n, carrier=load_shedding_carrier, marginal_cost=ls_mc_adjusted,
    )

    logger.info("C&CG Master: %d scenarios: %s  ls_penalty=%.4g  annual_scale=%.3f",
                len(scenarios), scenarios, ls_penalty, annual_scale)

    if co2_cost_mode != "off":
        logger.info("CO2 cost mode: %s", co2_cost_mode)

    # Save and disable cyclic flags
    e_cyclic_backup = {}
    for comp in ("storage_units", "stores"):
        df = getattr(n, comp, None)
        if df is not None and len(df) > 0:
            for col in ("cyclic_state_of_charge", "cyclic_state_of_charge_per_period", "e_cyclic"):
                if col in df.columns:
                    e_cyclic_backup[(comp, col)] = df[col].copy()
                    df[col] = False

    # [BUG-FIX-3] Save GlobalConstraint constants BEFORE scaling
    _original_gc_constants: Dict[str, float] = {
        gc_name: float(n.global_constraints.at[gc_name, "constant"])
        for gc_name in n.global_constraints.index
    }

    for gc_name in list(n.global_constraints.index):
        original = float(n.global_constraints.at[gc_name, "constant"])
        sense    = n.global_constraints.at[gc_name, "sense"]
        scaled   = original / annual_scale
        n.global_constraints.at[gc_name, "constant"] = scaled
        logger.info("Master: GlobalConstraint '%s' (%s %.3e) → %.3e (÷%.2fx)",
                    gc_name, sense, original, scaled, annual_scale)

    # [MEM-PATCH-B] Build weight array once
    w_base_np = n.snapshot_weightings["objective"].to_numpy(dtype=np.float32, copy=False)

    # [DISPATCH-FIX-1] Convert negative loads to generators (methodically correct)
    # Replaces the previous inline clip which incorrectly removed energy from balance
    _fix_negative_loads(n)

    # [DISPATCH-FIX-2] Relax e_initial for unbounded stores
    _fix_inf_store_initial(n)

    def extra_master(network: pypsa.Network, snapshots: pd.Index) -> None:
        m = network.model
        # [BUG-FIX-1] Pass e_cyclic_backup (not {}) so cyclic constraints use
        # the original flags, not the disabled-for-solve False values.
        _add_scenario_boundary_constraints(
            network, scenarios, masks, cyclic_overrides=e_cyclic_backup,
        )

        # [ARO-FIX-1] Remove PyPSA's auto-generated SOC/energy balance rows
        # at scenario boundaries. These couple consecutive scenarios through
        # storage dynamics (soc[scen_i, t=0] = f(soc[scen_{i-1}, t=T])),
        # which violates the ARO independence assumption.
        # PRECONDITION: _add_scenario_boundary_constraints has just been
        # called, so each scenario's own cyclic closure (soc[t=0]==soc[t=T])
        # is already in place before we remove the cross-boundary rows.
        _fix_cross_scenario_storage_constraints(network, scenarios, masks)

        _add_within_scenario_ramp_constraints(
            network, scenarios, masks, ramp_data,
            fail_on_extendable=fail_on_extendable_ramps,
        )

        # [ARO-FIX-2] lower=-np.inf  (was lower=0).
        # Rationale: in the C&CG epigraph formulation
        #   min Inv(x) + z_theta   s.t.  z_theta >= Op(x, s)  ∀ s ∈ S_k
        # the epigraph constraints alone ensure z_theta equals the worst-case
        # operational cost. Imposing lower=0 is an extraneous constraint that
        # is methodically wrong: if any scenario has Op < 0 (e.g. due to
        # negative marginal prices or export revenues) the true optimum is
        # blocked and the master returns a sub-optimal, non-robust portfolio.
        # Even when Op > 0 in practice, lower=0 adds a redundant constraint
        # that increases problem size and can worsen Barrier/Crossover numerics.
        z_theta  = m.add_variables(lower=-np.inf, name="z_theta")
        inv_cost = _build_investment_cost_expression(network)

        for s in scenarios:
            op_cost_s = _build_operational_cost_expression(
                network,
                masks[s],
                co2_cost_mode=co2_cost_mode,
                w_override_np=w_base_np,
                annual_scale=annual_scale,
            )
            m.add_constraints(
                1.0 * z_theta >= op_cost_s,
                name=f"robust_op_epigraph::{s}",
            )
            del op_cost_s

        m.objective = inv_cost + 1.0 * z_theta

    logger.info("C&CG Master: solving min Inv + worst-case Op ...")
    n.optimize(
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=extra_master,
        assign_all_duals=True,
        io_api="direct",
    )

    # [BUG-FIX-6] IIS debug behind env flag — does NOT run by default
    if os.environ.get("ARO_IIS_DEBUG", "0") == "1":
        try:
            import numpy as _np2
            _gm2 = n.model.solver_model if n.model is not None else None
            if _gm2 is not None:
                _gm2.computeIIS()
                _gm2.write("/tmp/aro_master_iis.ilp")
                _iis_var_ids = set()
                for _c in _gm2.getConstrs():
                    if _c.IISConstr:
                        _row = _gm2.getRow(_c)
                        _vars = [(_row.getVar(i).VarName, _row.getCoeff(i)) for i in range(min(_row.size(), 3))]
                        logger.info("IIS row %s (rhs=%.4f): %s", _c.ConstrName, _c.RHS, _vars)
                        for _i in range(_row.size()):
                            _iis_var_ids.add(_row.getVar(_i).VarName)
                for _vname in _iis_var_ids:
                    _vid = int(_vname.replace("x", ""))
                    for _lname, _lvar in n.model.variables.items():
                        _flat = _lvar.labels.values.flatten()
                        _hits = _np2.where(_flat == _vid)[0]
                        if len(_hits):
                            _coords = _np2.unravel_index(_hits[0], _lvar.labels.shape)
                            _dim_vals = {d: _lvar.labels[d].values.flat[_coords[i]] for i, d in enumerate(_lvar.dims)}
                            logger.info("  %s → linopy var '%s' coords=%s", _vname, _lname, _dim_vals)
                            break
        except Exception as _iis_e:
            logger.warning("IIS failed: %s", _iis_e)

    _check_solver_status(n, "Master", hard_fail_suboptimal=hard_fail_suboptimal)

    sol          = _get_solution_dict(n)
    z_theta_star = _extract_scalar_solution(sol, "z_theta")

    # [BUG-FIX-3] Restore GlobalConstraint constants
    for gc_name, orig_val in _original_gc_constants.items():
        if gc_name in n.global_constraints.index:
            n.global_constraints.at[gc_name, "constant"] = orig_val
    logger.info("GlobalConstraint constants restored.")

    # [BUG-FIX-4] Restore ramp limits and committable flags
    _restore_ramp_limits(n, ramp_data)

    # [BUG-FIX-5] Restore cyclic storage flags
    for (comp, col), series in e_cyclic_backup.items():
        df = getattr(n, comp, None)
        if df is not None and col in df.columns:
            df[col] = series
    if e_cyclic_backup:
        logger.info("Cyclic storage flags restored.")

    try:
        if n.model is not None:
            n.model.solver_model = None
    except Exception:
        pass
    gc.collect()

    logger.info("Master: z_theta* = %.6g", z_theta_star)

    return {
        "z_theta_star":       float(z_theta_star),
        "n_scenarios":        len(scenarios),
        "annual_scale":       float(annual_scale),
        "hours_per_scenario": int(hours_per_scenario),
        "co2_cost_mode":      co2_cost_mode,
        "ls_penalty":         float(ls_penalty),
        "pruned_components":  {k: len(v) for k, v in pruned.items()},
    }


# =============================================================================
# Export helpers
# =============================================================================

def _ensure_standard_pypsa_result_frames(n: pypsa.Network) -> None:
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


def export_network_flat_snapshots(n: pypsa.Network, out_network: str) -> None:
    if getattr(n, "model", None) is not None:
        try:
            n.model.solver_model = None
        except Exception:
            pass

    Path(out_network).parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(n.snapshots, pd.MultiIndex):
        n.export_to_netcdf(out_network)
        return

    flat = [
        f"{p}::{ts[0]}::{pd.Timestamp(ts[1]).isoformat()}"
        for p, ts in zip(
            n.snapshots.get_level_values("period"),
            n.snapshots.get_level_values("timestep"),
        )
    ]
    orig_snaps = n.snapshots
    n.set_snapshots(pd.Index(flat, name="snapshot"))
    try:
        n.export_to_netcdf(out_network)
    finally:
        n.set_snapshots(orig_snaps)


def export_network_stacked(n: pypsa.Network, out_network: str) -> None:
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
            n.buses_t.marginal_price = mp.reindex(
                index=n.snapshots, columns=n.buses.index,
            )
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
    if not isinstance(n_stacked.snapshots, pd.MultiIndex):
        raise ValueError("Expected stacked MultiIndex snapshots.")

    masks = _scenario_masks_from_snapshots(n_stacked.snapshots)
    if scenario not in masks:
        raise KeyError(f"Scenario '{scenario}' not found. Available: {list(masks.keys())}")

    idx  = np.flatnonzero(masks[scenario].astype(bool))
    ts   = n_stacked.snapshots.get_level_values("timestep")
    times = pd.DatetimeIndex(
        [pd.Timestamp(ts[i][1]) for i in idx], name="snapshot",
    )

    n_std = pypsa.Network(str(Path(scenario_network_file)))
    n_std.set_snapshots(times)

    for comp, opt_col, nom_col, ext_col in [
        ("generators",    "p_nom_opt", "p_nom", "p_nom_extendable"),
        ("links",         "p_nom_opt", "p_nom", "p_nom_extendable"),
        ("storage_units", "p_nom_opt", "p_nom", "p_nom_extendable"),
        ("stores",        "e_nom_opt", "e_nom", "e_nom_extendable"),
        ("lines",         "s_nom_opt", "s_nom", "s_nom_extendable"),
        ("transformers",  "s_nom_opt", "s_nom", "s_nom_extendable"),
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
        ext_common = common.intersection(
            df_std.index[df_std[ext_col].fillna(False).astype(bool)]
        )
        if len(ext_common) > 0 and nom_col in df_std.columns:
            df_std.loc[ext_common, nom_col] = (
                df_sol.loc[ext_common, opt_col].fillna(0.0).astype(float)
            )

    comp_cols = {
        "buses_t":         n_std.buses.index,
        "generators_t":    n_std.generators.index,
        "links_t":         n_std.links.index,
        "storage_units_t": n_std.storage_units.index,
        "stores_t":        n_std.stores.index,
        "lines_t":         n_std.lines.index,
        "transformers_t":  n_std.transformers.index,
        "loads_t":         n_std.loads.index,
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

    try:
        mp = getattr(n_std.buses_t, "marginal_price", None)
    except Exception:
        mp = None
    if mp is not None and isinstance(mp, pd.DataFrame):
        n_std.buses_t.marginal_price = mp.reindex(
            index=n_std.snapshots, columns=n_std.buses.index,
        )

    Path(out_network).parent.mkdir(parents=True, exist_ok=True)
    n_std.export_to_netcdf(out_network)


def extract_capacities(n: pypsa.Network) -> Dict[str, Dict[str, float]]:
    def _ext(comp_df, opt_col, ext_col):
        if comp_df is None or len(comp_df) == 0:
            return {}
        if opt_col not in comp_df.columns or ext_col not in comp_df.columns:
            return {}
        ext = comp_df.index[comp_df[ext_col].fillna(False).astype(bool)]
        return comp_df.loc[ext, opt_col].dropna().to_dict()

    out = {
        "generators":    _ext(n.generators,    "p_nom_opt", "p_nom_extendable"),
        "links":         _ext(n.links,          "p_nom_opt", "p_nom_extendable"),
        "storage_units": _ext(n.storage_units,  "p_nom_opt", "p_nom_extendable"),
        "stores":        _ext(n.stores,          "e_nom_opt", "e_nom_extendable"),
        "lines":         _ext(n.lines,           "s_nom_opt", "s_nom_extendable"),
        "transformers":  _ext(n.transformers,    "s_nom_opt", "s_nom_extendable"),
    }
    return {k: v for k, v in out.items() if v}


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
        allow_suboptimal: bool,
        allow_extendable_ramps: bool,
        co2_cost_mode: str,
        warn_unknown_t: bool,
        strict_unknown_t: bool,
        ls_penalty: float = 1e4,
) -> None:
    scenario_files = _resolve_scenario_networks_from_cutouts(
        cutouts, scenario_network_template,
    )
    logger.info("=== Robust solve: co2_cost_mode=%s  ls_penalty=%.4g ===",
                co2_cost_mode, ls_penalty)
    for c, f in zip(cutouts, scenario_files):
        logger.info("  %-35s -> %s", c, f)

    n, scen_names = stack_scenarios_to_multisnapshot_network(
        scenario_files=scenario_files,
        scenario_names=list(cutouts),
        warn_unknown_t=warn_unknown_t,
        strict_unknown_t=strict_unknown_t,
    )
    diag = solve_aro_master(
        n,
        solver_name=solver_name,
        solver_options=solver_options,
        hard_fail_suboptimal=not allow_suboptimal,
        fail_on_extendable_ramps=not allow_extendable_ramps,
        co2_cost_mode=co2_cost_mode,
        ls_penalty=ls_penalty,
    )
    capacities = extract_capacities(n)

    Path(out_network).parent.mkdir(parents=True, exist_ok=True)
    Path(out_summary_json).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting stacked robust network → %s", out_network)
    _ensure_standard_pypsa_result_frames(n)
    export_network_stacked(n, out_network)

    out_network_std = str(Path(out_network).with_suffix("")) + "__std.nc"
    export_scenario  = str(list(cutouts)[0])
    export_file      = str(scenario_files[0])
    logger.info("Exporting standard adapter (scenario=%s) → %s",
                export_scenario, out_network_std)
    export_network_standard_single_scenario(
        n,
        scenario=export_scenario,
        scenario_network_file=export_file,
        out_network=out_network_std,
    )

    try:
        if getattr(n, "model", None) is not None:
            n.model = None
    except Exception:
        pass
    gc.collect()

    payload: Dict[str, Any] = {
        "scenario_names":    list(scen_names),
        "scenario_networks": list(map(str, scenario_files)),
        "diagnostics":       diag,
        "capacities":        capacities,
        "outputs": {
            "robust_stacked_network":       str(out_network),
            "postprocess_adapter_network":  str(out_network_std),
            "postprocess_adapter_scenario": export_scenario,
        },
        "assumptions": {
            "co2_cost_mode": co2_cost_mode,
            "ls_penalty":    ls_penalty,
        },
    }
    logger.info("Writing summary JSON → %s", out_summary_json)
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
    p.add_argument("--allow-suboptimal", action="store_true")
    p.add_argument("--allow-extendable-ramps", action="store_true")
    p.add_argument(
        "--co2-cost-mode",
        choices=["off", "global_constraint_constant_cost"],
        default="off",
    )
    p.add_argument("--ls-penalty", type=float, default=1e4)
    p.add_argument("--warn-unknown-t", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict-unknown-t", action="store_true")
    p.add_argument("--test", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()

    if args.test:
        logger.info("No unit tests in this version.")
        return

    required = ["cutouts", "scenario_network_template", "out_network", "out_summary_json"]
    missing  = [r for r in required if getattr(args, r.replace("-", "_"), None) is None]
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
        allow_suboptimal=args.allow_suboptimal,
        allow_extendable_ramps=args.allow_extendable_ramps,
        co2_cost_mode=args.co2_cost_mode,
        warn_unknown_t=args.warn_unknown_t,
        strict_unknown_t=args.strict_unknown_t,
        ls_penalty=args.ls_penalty,
    )


if __name__ == "__main__":
    main()