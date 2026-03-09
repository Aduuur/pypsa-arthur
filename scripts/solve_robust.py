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

[ARO-FIX-3]  _add_per_scenario_global_constraints  — correct CO2/energy budget handling

    PROBLEM (critical):
    PyPSA's GlobalConstraint mechanism sums over ALL snapshots in the model:
        ∑_{t ∈ ALL_SNAPS} w_t · ∑_g ef_g · p_{g,t}  <=  constant
    In a stacked N-scenario network this means the constraint aggregates over
    N × T timesteps. With N=2 scenarios the solver sees twice the emission
    budget and the CO2 cap is twice as permissive as intended.
    The previous fix (constant / annual_scale) addressed the per-scenario
    weighting but not the N-fold summing.

    FIX:
    1. Before n.optimize(): pre-scan each time-aggregated GC via
       _resolve_gc_emission_factors(). If factors are resolvable (column
       lookup or carrier-name fallback), set constant = abs(original)*1000
       — non-binding but numerically harmless (avoids the Barrier failure
       caused by 1e+15 RHS values). Unresolvable GCs are NOT neutralized
       and fall back to annual_scale scaling with a warning.
    2. Inside extra_master: add one explicit linopy constraint per (gc, scenario)
       with the correct per-scenario budget:
           ∑_{t ∈ scen_s} w_t · ∑_g ef_g · p_{g,t}  <=  original_constant / annual_scale
    This guarantees each scenario is independently constrained to its own
    budget, independent of all other scenarios.

    ASSUMPTION: Global constraints without carrier_attribute (e.g. pure
    investment-side constraints) are not time-aggregated and are left to
    PyPSA's native handling unchanged.

[LS-FIX-1]  Per-bus load-shedding p_nom  — eliminate source of Markowitz warnings

    PROBLEM:
    LS generators were added with a global p_nom=1e8 MW, creating matrix
    coefficients 4–5 orders of magnitude larger than physical capacity
    variables (~1e3–1e4 MW). This caused the Markowitz tolerance warnings
    observed in the solver log and contributed to Crossover instability.

    FIX:
    Compute p_nom per bus as peak_load_at_bus × safety_factor (default 2.0),
    with a minimum floor of 100 MW. This keeps LS variables in the same
    numerical range as physical generation variables while still ensuring
    the load-shedding constraint is never binding at feasible solutions.

    ASSUMPTION: Peak load is taken from loads_t.p_set (or static p_set if
    no time series present). Buses with no attached loads get the floor value.

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
import contextlib
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

try:
    from scripts.cost_validation import validate_extendable_capital_costs
except ModuleNotFoundError:
    from cost_validation import validate_extendable_capital_costs


logger = logging.getLogger(__name__)

# =============================================================================
# Debug / filtering helpers
# =============================================================================

_SYNTHETIC_GENERATOR_CARRIERS = {"load", "load_shedding", "net_export"}
_SYNTHETIC_GENERATOR_PREFIXES = ("LS::", "NegLoad::")

# Alles, was wir im Portfolio-Debug als "verdächtig" ansehen
_SUSPICIOUS_CARRIERS = {
    "load", "load_shedding", "net_export",
    "co2", "co2 stored", "co2 sequestered",
    "uranium", "coal", "lignite", "geothermal_heat",
    "oil refining", "electricity distribution grid",
    "biomass", "solid biomass", "biogas",
    "urban central water pits charger",
    "urban central water pits discharger",
    "H2 Store",
}

def _is_synthetic_generator_row(name: str, row: pd.Series) -> bool:
    carrier = str(row.get("carrier", ""))
    if carrier in _SYNTHETIC_GENERATOR_CARRIERS:
        return True
    if any(str(name).startswith(p) for p in _SYNTHETIC_GENERATOR_PREFIXES):
        return True
    return False


def _drop_orphan_timeseries_columns(n: pypsa.Network) -> None:
    """
    Remove *_t columns whose assets no longer exist in the corresponding static table.
    This fixes the repeated PyPSA warning about p_max_pu columns for missing generators.
    """
    checks = [
        ("generators", "generators_t"),
        ("links", "links_t"),
        ("storage_units", "storage_units_t"),
        ("stores", "stores_t"),
        ("loads", "loads_t"),
        ("lines", "lines_t"),
        ("transformers", "transformers_t"),
        ("buses", "buses_t"),
    ]

    total_dropped = 0

    for static_name, t_name in checks:
        static_df = getattr(n, static_name, None)
        t_obj = getattr(n, t_name, None)
        if static_df is None or t_obj is None:
            continue

        valid = set(static_df.index)

        for attr in list(vars(t_obj)):
            df = getattr(t_obj, attr, None)
            if isinstance(df, pd.DataFrame) and len(df.columns) > 0:
                keep = [c for c in df.columns if c in valid]
                dropped = len(df.columns) - len(keep)
                if dropped > 0:
                    setattr(t_obj, attr, df.loc[:, keep])
                    total_dropped += dropped
                    logger.warning(
                        "[ORPHAN-FIX] %s.%s: dropped %d orphan columns.",
                        t_name, attr, dropped,
                    )

    if total_dropped > 0:
        logger.warning("[ORPHAN-FIX] Total orphan *_t columns dropped: %d", total_dropped)
    else:
        logger.info("[ORPHAN-FIX] No orphan *_t columns found.")


def _debug_suspicious_assets(n: pypsa.Network, stage: str) -> None:
    logger.info("=== DEBUG SUSPICIOUS ASSETS: %s ===", stage)

    for comp in ["generators", "links", "stores", "storage_units"]:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0 or "carrier" not in df.columns:
            continue

        suspicious = df[df["carrier"].astype(str).isin(_SUSPICIOUS_CARRIERS)].copy()
        if suspicious.empty:
            continue

        cols_show = [c for c in [
            "carrier",
            "p_nom", "p_nom_opt", "p_nom_max",
            "e_nom", "e_nom_opt", "e_nom_max",
            "capital_cost", "marginal_cost",
            "p_nom_extendable", "e_nom_extendable",
        ] if c in suspicious.columns]

        logger.warning(
            "[%s] suspicious %s: n=%d carriers=%s",
            stage, comp, len(suspicious),
            suspicious["carrier"].value_counts().to_dict(),
        )
        if cols_show:
            logger.warning("[%s] suspicious %s sample:\n%s",
                           stage, comp, suspicious[cols_show].head(20).to_string())

def _debug_component_overview(n: pypsa.Network, stage: str) -> None:
    logger.info("=== DEBUG OVERVIEW: %s ===", stage)
    logger.info(
        "components: buses=%d gens=%d loads=%d links=%d lines=%d su=%d stores=%d snaps=%d",
        len(getattr(n, "buses", [])),
        len(getattr(n, "generators", [])),
        len(getattr(n, "loads", [])),
        len(getattr(n, "links", [])),
        len(getattr(n, "lines", [])),
        len(getattr(n, "storage_units", [])),
        len(getattr(n, "stores", [])),
        len(getattr(n, "snapshots", [])),
    )


def _debug_extendables(n: pypsa.Network, stage: str) -> None:
    logger.info("=== DEBUG EXTENDABLES: %s ===", stage)

    for comp, ext_col, cap_col, max_col in [
        ("generators", "p_nom_extendable", "p_nom", "p_nom_max"),
        ("links", "p_nom_extendable", "p_nom", "p_nom_max"),
        ("storage_units", "p_nom_extendable", "p_nom", "p_nom_max"),
        ("stores", "e_nom_extendable", "e_nom", "e_nom_max"),
        ("lines", "s_nom_extendable", "s_nom", "s_nom_max"),
        ("transformers", "s_nom_extendable", "s_nom", "s_nom_max"),
    ]:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0 or ext_col not in df.columns:
            continue

        ext = df[df[ext_col].fillna(False).astype(bool)].copy()
        if len(ext) == 0:
            logger.info("[%s] %s: no extendables", stage, comp)
            continue

        carrier_info = {}
        if "carrier" in ext.columns:
            try:
                carrier_info = ext["carrier"].value_counts().head(15).to_dict()
            except Exception:
                pass

        zero_cap_cost = 0
        nan_cap_cost = 0
        if "capital_cost" in ext.columns:
            cc = pd.to_numeric(ext["capital_cost"], errors="coerce")
            zero_cap_cost = int((cc.fillna(0.0) == 0.0).sum())
            nan_cap_cost = int(cc.isna().sum())

        inf_max = 0
        if max_col in ext.columns:
            try:
                inf_max = int(np.isinf(pd.to_numeric(ext[max_col], errors="coerce")).sum())
            except Exception:
                pass

        logger.info(
            "[%s] %s: n_ext=%d  carriers=%s  zero_cap_cost=%d  nan_cap_cost=%d  inf_%s=%d",
            stage, comp, len(ext), carrier_info, zero_cap_cost, nan_cap_cost, max_col, inf_max,
        )

        cols_show = [c for c in ["carrier", cap_col, max_col, "capital_cost", "marginal_cost"] if c in ext.columns]
        if cols_show:
            logger.info("[%s] %s sample extendables:\n%s", stage, comp, ext[cols_show].head(10).to_string())


def _debug_nonextendable_infinite_assets(n: pypsa.Network, stage: str) -> None:
    logger.info("=== DEBUG NON-EXTENDABLE INFINITE ASSETS: %s ===", stage)

    for comp, ext_col, max_col in [
        ("generators", "p_nom_extendable", "p_nom_max"),
        ("links", "p_nom_extendable", "p_nom_max"),
        ("storage_units", "p_nom_extendable", "p_nom_max"),
        ("stores", "e_nom_extendable", "e_nom_max"),
    ]:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0 or ext_col not in df.columns or max_col not in df.columns:
            continue

        is_ext = df[ext_col].fillna(False).astype(bool)
        max_vals = pd.to_numeric(df[max_col], errors="coerce")
        bad = df[(~is_ext) & np.isinf(max_vals)]

        if len(bad) > 0:
            cols_show = [c for c in ["carrier", max_col, "capital_cost", "marginal_cost"] if c in bad.columns]
            logger.warning(
                "[%s] %s: %d non-extendable assets still have %s=inf. carriers=%s",
                stage, comp, len(bad), max_col,
                bad["carrier"].value_counts().head(15).to_dict() if "carrier" in bad.columns else {},
            )
            if cols_show:
                logger.warning("[%s] %s sample non-extendable inf assets:\n%s",
                               stage, comp, bad[cols_show].head(10).to_string())


def _debug_timeseries_orphans(n: pypsa.Network, stage: str) -> None:
    logger.info("=== DEBUG TIMESERIES ORPHANS: %s ===", stage)

    checks = [
        ("generators", "generators_t", ["p_max_pu", "p_min_pu", "marginal_cost", "p"]),
        ("links", "links_t", ["p_max_pu", "p_min_pu", "marginal_cost", "p0"]),
        ("storage_units", "storage_units_t", ["p_max_pu", "p_min_pu", "marginal_cost", "p"]),
        ("stores", "stores_t", ["marginal_cost", "p", "e"]),
        ("loads", "loads_t", ["p_set"]),
    ]

    for static_name, t_name, attrs in checks:
        static_df = getattr(n, static_name, None)
        t_obj = getattr(n, t_name, None)
        if static_df is None or t_obj is None:
            continue

        valid = set(static_df.index)
        for attr in attrs:
            df = getattr(t_obj, attr, None)
            if isinstance(df, pd.DataFrame) and len(df.columns) > 0:
                orphans = [c for c in df.columns if c not in valid]
                if orphans:
                    logger.warning(
                        "[%s] %s.%s has %d orphan columns. examples=%s",
                        stage, t_name, attr, len(orphans), orphans[:10],
                    )


def _debug_load_shedding_breakdown(n: pypsa.Network, stage: str) -> None:
    if not hasattr(n, "generators") or not hasattr(n, "generators_t"):
        return
    p = getattr(n.generators_t, "p", None)
    if not isinstance(p, pd.DataFrame) or p.empty:
        return

    ls_mask = n.generators["carrier"].isin(["load_shedding", "load"])
    ls_names = n.generators.index[ls_mask]
    if len(ls_names) == 0:
        logger.info("[%s] No LS generators found.", stage)
        return

    cols = p.columns.intersection(ls_names)
    if len(cols) == 0:
        logger.info("[%s] LS generators exist, but no dispatch columns found.", stage)
        return

    total = float(p[cols].sum().sum())
    peak = float(p[cols].max().max())
    by_carrier = (
        n.generators.loc[cols, "carrier"]
        .value_counts()
        .to_dict()
        if "carrier" in n.generators.columns else {}
    )
    by_bus = (
        p[cols].sum(axis=0)
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )

    logger.warning(
        "[%s] LS breakdown: total=%.3e MWh  peak=%.3e MW  carriers=%s  top_assets=%s",
        stage, total, peak, by_carrier, by_bus,
    )

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

_COMP_FOR_T_CONTAINER: Dict[str, str] = {
    "generators_t":    "generators",
    "links_t":         "links",
    "storage_units_t": "storage_units",
    "stores_t":        "stores",
    "loads_t":         "loads",
    "lines_t":         "lines",
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
# [HEAT-VENT-FIX] Heat vents are not extendable — cap to existing p_nom
# =============================================================================
def _fix_heat_vents(n: pypsa.Network) -> None:
    """
    Heat vent generators (urban/rural heat vent) must NOT be extendable.
    In the prepared network they are incorrectly set to p_nom_extendable=True
    with p_nom_max=inf and capital_cost=1.0. This causes the solver to build
    absurd capacities (>9 GW/bus) because mc=-0.01 creates revenue from dispatch.
    Fix: set p_nom_extendable=False and cap p_nom_max=p_nom.
    """
    mask = n.generators.carrier.str.contains("heat vent", na=False) | (n.generators.carrier == "oil")
    n_vents = mask.sum()
    if n_vents == 0:
        return
    n.generators.loc[mask, "p_nom_extendable"] = False
    # p_nom_max = p_nom (existing capacity only, no new build)
    n.generators.loc[mask, "p_nom_max"] = n.generators.loc[mask, "p_nom"]
    logger.info(
        "[HEAT-VENT-FIX] Capped %d heat vent / oil generators to non-extendable "
        "(p_nom_max=p_nom). Was: p_nom_extendable=True, p_nom_max=inf.",
        n_vents,
    )

# =============================================================================
# [DIST-GRID-FIX] Distribution grid p_nom = existing peak LV demand
# =============================================================================
def _fix_distribution_grid(n: pypsa.Network) -> None:
    """
    [DIST-GRID-FIX] Distribution grid links haben p_nom=0 im prepared network,
    obwohl das Grid in der Realität bereits existiert und die aktuelle Last trägt.
    
    Fix: p_nom = peak LV load per bus (existing capacity).
    p_nom_extendable=True bleibt erhalten → Ausbau über Bestand hinaus möglich.
    p_nom_max=inf bleibt → kein Cap nach oben.
    
    Ohne diesen Fix hat der Master p_nom_opt=0 weil LS-Gens direkt auf LV-Bussen
    sitzen und das Grid aus Solver-Sicht wertlos ist.
    """
    dg_mask = (
        n.links.carrier.str.contains("electricity distribution grid", na=False)
        & ~n.links.index.str.contains("reversed", na=False)
    )
    n_dg = dg_mask.sum()
    if n_dg == 0:
        return

    # Peak LV load pro Bus aus loads_t.p_set
    lv_buses = n.buses[n.buses.carrier.str.contains("low voltage", case=False, na=False)].index
    lv_loads = n.loads[n.loads.bus.isin(lv_buses)]

    bus_peak: Dict[str, float] = {}
    if len(lv_loads) > 0:
        if hasattr(n, "loads_t") and hasattr(n.loads_t, "p_set") and len(n.loads_t.p_set.columns) > 0:
            p_set_t = n.loads_t.p_set
            for load_name, load_row in lv_loads.iterrows():
                bus = str(load_row["bus"])
                if load_name in p_set_t.columns:
                    peak = float(p_set_t[load_name].abs().max())
                else:
                    peak = float(abs(load_row.get("p_set", 0.0) or 0.0))
                bus_peak[bus] = bus_peak.get(bus, 0.0) + peak
        else:
            for load_name, load_row in lv_loads.iterrows():
                bus = str(load_row["bus"])
                peak = float(abs(load_row.get("p_set", 0.0) or 0.0))
                bus_peak[bus] = bus_peak.get(bus, 0.0) + peak

    # p_nom setzen: peak LV load des Zielbusses (bus1 = LV bus)
    fixed = 0
    for link_name in n.links.index[dg_mask]:
        bus1 = str(n.links.at[link_name, "bus1"])
        peak = bus_peak.get(bus1, 0.0)
        if peak > 0:
            n.links.at[link_name, "p_nom"] = peak
            fixed += 1

    logger.info(
        "[DIST-GRID-FIX] Set p_nom=peak_LV_load for %d/%d distribution grid links. "
        "p_nom range: [%.0f, %.0f] MW. p_nom_extendable remains True.",
        fixed, n_dg,
        n.links.loc[dg_mask, "p_nom"].min(),
        n.links.loc[dg_mask, "p_nom"].max(),
    )

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
# =============================================================================
# Stacking debug helpers
# =============================================================================

def _find_component_columns_by_keywords(
        cols: pd.Index,
        keywords: Sequence[str],
) -> List[str]:
    """
    Return columns whose names contain at least one keyword (case-insensitive).
    """
    if cols is None or len(cols) == 0:
        return []
    kws = [str(k).lower() for k in keywords]
    out = []
    for c in cols:
        s = str(c).lower()
        if any(kw in s for kw in kws):
            out.append(str(c))
    return out


def _debug_compare_stacked_timeseries(
        n: pypsa.Network,
        *,
        scenario_names: Sequence[str],
        t_container_name: str,
        attr: str,
        keywords: Sequence[str],
        top_n: int = 8,
        atol_identical: float = 1e-7,
) -> None:
    """
    Debug helper:
    For selected columns in a stacked *_t DataFrame, compare scenario blocks and log whether
    they are identical or different across scenarios.

    This is useful to detect stacking bugs where scenario-dependent time series were
    accidentally flattened to one common profile.
    """
    if not hasattr(n, t_container_name):
        logger.info("[STACK-DEBUG] %s.%s missing on stacked network.", t_container_name, attr)
        return

    t_obj = getattr(n, t_container_name)
    df = getattr(t_obj, attr, None)

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.info("[STACK-DEBUG] %s.%s empty or absent.", t_container_name, attr)
        return

    if not isinstance(n.snapshots, pd.MultiIndex):
        logger.warning("[STACK-DEBUG] Expected MultiIndex snapshots, got %s.", type(n.snapshots))
        return

    masks = _scenario_masks_from_snapshots(n.snapshots)
    available_scenarios = [s for s in scenario_names if s in masks]

    if len(available_scenarios) < 2:
        logger.info("[STACK-DEBUG] Need at least 2 scenarios for comparison in %s.%s.", t_container_name, attr)
        return

    candidate_cols = _find_component_columns_by_keywords(df.columns, keywords)
    if not candidate_cols:
        logger.info(
            "[STACK-DEBUG] No columns matched keywords=%s in %s.%s.",
            list(keywords), t_container_name, attr
        )
        return

    candidate_cols = candidate_cols[:top_n]

    logger.info(
        "[STACK-DEBUG] Comparing %s.%s for keywords=%s | matched=%d | showing=%d",
        t_container_name, attr, list(keywords), len(_find_component_columns_by_keywords(df.columns, keywords)), len(candidate_cols)
    )

    for col in candidate_cols:
        series_per_scenario: Dict[str, np.ndarray] = {}
        stats_per_scenario: Dict[str, Tuple[float, float, float]] = {}

        for scen in available_scenarios:
            block = df.loc[masks[scen], col]
            arr = block.to_numpy(dtype=np.float64, copy=False)
            series_per_scenario[scen] = arr

            if arr.size == 0 or np.all(np.isnan(arr)):
                stats_per_scenario[scen] = (np.nan, np.nan, np.nan)
            else:
                stats_per_scenario[scen] = (
                    float(np.nanmin(arr)),
                    float(np.nanmax(arr)),
                    float(np.nanmean(arr)),
                )

        reference_scen = available_scenarios[0]
        ref_arr = series_per_scenario[reference_scen]

        all_identical = True
        pair_summaries = []

        for scen in available_scenarios[1:]:
            arr = series_per_scenario[scen]

            same_shape = ref_arr.shape == arr.shape
            if not same_shape:
                all_identical = False
                pair_summaries.append(f"{reference_scen} vs {scen}: shape {ref_arr.shape} != {arr.shape}")
                continue

            equal_mask = np.isclose(ref_arr, arr, atol=atol_identical, rtol=0.0, equal_nan=True)
            identical = bool(np.all(equal_mask))

            if not identical:
                all_identical = False
                max_abs_diff = float(np.nanmax(np.abs(ref_arr - arr))) if ref_arr.size else np.nan
                pair_summaries.append(f"{reference_scen} vs {scen}: DIFFERENT (max_abs_diff={max_abs_diff:.3e})")
            else:
                pair_summaries.append(f"{reference_scen} vs {scen}: identical")

        logger.info(
            "[STACK-DEBUG] %-60s | %s",
            col,
            "ALL IDENTICAL" if all_identical else "SCENARIO-DEPENDENT"
        )

        for scen in available_scenarios:
            mn, mx, av = stats_per_scenario[scen]
            logger.info(
                "[STACK-DEBUG]    %-20s min=%12.5g max=%12.5g mean=%12.5g",
                scen, mn, mx, av
            )

        for msg in pair_summaries:
            logger.info("[STACK-DEBUG]    %s", msg)


def _debug_log_stacking_overview(
        n: pypsa.Network,
        *,
        scenario_names: Sequence[str],
) -> None:
    """
    High-level debug overview after stacking.
    Focuses on the most relevant operational drivers:
      - generators_t.p_max_pu  (VRE availability)
      - loads_t.p_set          (demand differences)
      - generators_t.marginal_cost
      - links_t.efficiency / p_max_pu (if relevant)
    """
    logger.info("=" * 79)
    logger.info("[STACK-DEBUG] START stacked scenario diagnostics")
    logger.info("=" * 79)

    try:
        _debug_compare_stacked_timeseries(
            n,
            scenario_names=scenario_names,
            t_container_name="generators_t",
            attr="p_max_pu",
            keywords=["solar", "onwind", "offwind", "wind", "biomass", "ror"],
            top_n=12,
        )
    except Exception as exc:
        logger.warning("[STACK-DEBUG] generators_t.p_max_pu debug failed: %s", exc)

    try:
        _debug_compare_stacked_timeseries(
            n,
            scenario_names=scenario_names,
            t_container_name="loads_t",
            attr="p_set",
            keywords=["load", "demand", "DE", "FR", "GB"],
            top_n=12,
        )
    except Exception as exc:
        logger.warning("[STACK-DEBUG] loads_t.p_set debug failed: %s", exc)

    try:
        _debug_compare_stacked_timeseries(
            n,
            scenario_names=scenario_names,
            t_container_name="generators_t",
            attr="marginal_cost",
            keywords=["biomass", "gas", "OCGT", "CCGT", "coal", "lignite", "oil"],
            top_n=12,
        )
    except Exception as exc:
        logger.warning("[STACK-DEBUG] generators_t.marginal_cost debug failed: %s", exc)

    try:
        _debug_compare_stacked_timeseries(
            n,
            scenario_names=scenario_names,
            t_container_name="links_t",
            attr="p_max_pu",
            keywords=["battery", "H2", "electrolysis", "fuel cell", "heat pump"],
            top_n=12,
        )
    except Exception as exc:
        logger.warning("[STACK-DEBUG] links_t.p_max_pu debug failed: %s", exc)

    logger.info("=" * 79)
    logger.info("[STACK-DEBUG] END stacked scenario diagnostics")
    logger.info("=" * 79)

def stack_scenarios_to_multisnapshot_network(
        scenario_files: Sequence[str],
        scenario_names: Sequence[str],
        *,
        warn_unknown_t: bool = True,
        strict_unknown_t: bool = False,
        name_normalization_mode: str = "exact",
) -> Tuple[pypsa.Network, List[str]]:
    """
    Stack N scenario networks into a single MultiIndex-snapshot network.

    Robust version:
    - decides constant vs variable across ALL scenarios, not only ref
    - handles year suffix mismatches robustly (e.g. '-2050')
    - validates static component index consistency
    - keeps a column static ONLY if:
        (a) present in all scenarios
        (b) constant over time in every scenario
        (c) same constant value in every scenario
    - everything else is stacked as scenario-dependent *_t data
    """
    logger.info(
        "Scenario stacking: %d scenarios | name_normalization_mode=%s",
        len(scenario_names),
        name_normalization_mode,
    )
    CONST_TOL = 1e-6

    if len(scenario_files) != len(scenario_names):
        raise ValueError("scenario_files and scenario_names must have equal length.")
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError(f"Duplicate scenario names: {scenario_names}")
    if len(scenario_files) == 0:
        raise ValueError("No scenario files provided.")

    # -------------------------------------------------------------------------
    # local helpers
    # -------------------------------------------------------------------------
    import re as _re
    from collections import defaultdict


    def _safe_get_t_attrs(t_obj: Any) -> List[str]:
        """
        Try to list dynamic attributes of a PyPSA *_t container.
        Only keep DataFrame/Series-like attributes.
        """
        attrs = []
        for name in dir(t_obj):
            if name.startswith("_"):
                continue
            try:
                val = getattr(t_obj, name)
            except Exception:
                continue
            if isinstance(val, (pd.DataFrame, pd.Series)):
                attrs.append(name)
        return sorted(set(attrs))

    def _get_static_comp_name(t_container_name: str) -> Optional[str]:
        return _T_CONTAINER_TO_STATIC_COMP.get(t_container_name)

    # =============================================================================
    # Stacking name helpers
    # =============================================================================

    def _identity_name(x: Any) -> str:
        return str(x)

    def _strip_trailing_year(x: Any) -> str:
        import re
        return re.sub(r"-\d{4}$", "", str(x))

    def _make_name_normalizer(mode: str):
        """
        mode:
          - 'exact'      -> keep names exactly as they are
          - 'strip_year' -> remove trailing -YYYY
        """
        if mode == "exact":
            return _identity_name
        if mode == "strip_year":
            return _strip_trailing_year
        raise ValueError(f"Unknown name_normalization_mode='{mode}'")

    def _build_static_name_lookup(
            n_ref: pypsa.Network,
            *,
            name_normalization_mode: str = "exact",
    ) -> Dict[str, Dict[str, str]]:
        """
        Build normalized_name -> actual_name lookup for each static component table.

        In exact mode this is just identity mapping.
        In strip_year mode collisions are checked explicitly and raise an error.
        """
        norm = _make_name_normalizer(name_normalization_mode)

        lookup: Dict[str, Dict[str, str]] = {}

        for t_container_name, static_comp in _T_CONTAINER_TO_STATIC_COMP.items():
            if not hasattr(n_ref, static_comp):
                continue

            df = getattr(n_ref, static_comp, None)
            if df is None or len(df) == 0:
                lookup[t_container_name] = {}
                continue

            norm_to_actual: Dict[str, str] = {}
            collisions: Dict[str, List[str]] = {}

            for nm in df.index:
                key = norm(nm)
                if key in norm_to_actual and norm_to_actual[key] != str(nm):
                    collisions.setdefault(key, [norm_to_actual[key]])
                    if str(nm) not in collisions[key]:
                        collisions[key].append(str(nm))
                else:
                    norm_to_actual[key] = str(nm)

            if collisions:
                examples = {k: v for k, v in list(collisions.items())[:10]}
                raise ValueError(
                    f"Normalized-name collision in static component '{static_comp}' "
                    f"under mode='{name_normalization_mode}'. Examples: {examples}"
                )

            lookup[t_container_name] = norm_to_actual

        return lookup

    def _normalize_df_columns(
            df: pd.DataFrame,
            *,
            t_container_name: str,
            static_lookup: Dict[str, Dict[str, str]],
            scenario_name: str,
            attr: str,
            name_normalization_mode: str = "exact",
            drop_unknown_columns: bool = True,
    ) -> pd.DataFrame:
        norm = _make_name_normalizer(name_normalization_mode)

        if df is None or df.empty:
            return pd.DataFrame(index=base_snaps)

        df = df.copy()
        df = df.reindex(base_snaps)

        norm_cols = pd.Index([norm(c) for c in df.columns])

        dup_mask = norm_cols.duplicated(keep=False)
        if dup_mask.any():
            dup_vals = norm_cols[dup_mask].unique().tolist()
            raise ValueError(
                f"After normalization, duplicate columns detected in scenario='{scenario_name}', "
                f"{t_container_name}.{attr}, mode='{name_normalization_mode}'. "
                f"Examples: {dup_vals[:10]}"
            )

        df.columns = norm_cols

        lookup = static_lookup.get(t_container_name, {})
        if lookup:
            known_cols = [c for c in df.columns if c in lookup]
            unknown_cols = [c for c in df.columns if c not in lookup]

            if unknown_cols:
                msg = (
                    f"Scenario '{scenario_name}': unknown columns in {t_container_name}.{attr} "
                    f"under mode='{name_normalization_mode}' not found in reference static index. "
                    f"Examples: {unknown_cols[:10]}"
                )
                if strict_unknown_t:
                    raise ValueError(msg)
                if warn_unknown_t:
                    logger.warning(msg)

            if drop_unknown_columns:
                df = df.reindex(columns=known_cols)

        return df.astype(np.float32)

    def _validate_static_index_compatibility(
            net_ref: pypsa.Network,
            net_i: pypsa.Network,
            *,
            scenario_name: str,
            name_normalization_mode: str = "exact",
    ) -> None:
        """
        Validate that all relevant static component indices match the reference
        after optional normalization.
        """
        norm = _make_name_normalizer(name_normalization_mode)

        for t_container_name in _T_ATTRS_INPUT_ONLY.keys():
            static_comp = _T_CONTAINER_TO_STATIC_COMP.get(t_container_name)
            if static_comp is None:
                continue
            if not hasattr(net_ref, static_comp) or not hasattr(net_i, static_comp):
                continue

            df_ref = getattr(net_ref, static_comp, None)
            df_i = getattr(net_i, static_comp, None)
            if df_ref is None or df_i is None:
                continue

            idx_ref = pd.Index([norm(x) for x in df_ref.index])
            idx_i = pd.Index([norm(x) for x in df_i.index])

            missing = idx_ref.difference(idx_i)
            extra = idx_i.difference(idx_ref)

            if len(missing) or len(extra):
                msg = (
                    f"Static index mismatch in scenario '{scenario_name}' for component "
                    f"'{static_comp}' under mode='{name_normalization_mode}':\n"
                )
                if len(missing):
                    msg += f"  Missing: {list(missing)[:10]}\n"
                if len(extra):
                    msg += f"  Extra:   {list(extra)[:10]}\n"
                raise ValueError(msg)

    # -------------------------------------------------------------------------
    # load reference network
    # -------------------------------------------------------------------------
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
    n_scenarios = len(scenario_names)

    # -------------------------------------------------------------------------
    # create stacked network skeleton from reference
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # discover static-name lookup from reference
    # -------------------------------------------------------------------------
    static_lookup = _build_static_name_lookup(
        ref,
        name_normalization_mode=name_normalization_mode,
    )
    # -------------------------------------------------------------------------
    # optional: warn/error on unknown *_t attrs in each scenario
    # -------------------------------------------------------------------------
    if warn_unknown_t or strict_unknown_t:
        allowed_map = {k: set(v) for k, v in _T_ATTRS_INPUT_ONLY.items()}
        for t_container_name in allowed_map:
            if hasattr(ref, t_container_name):
                t_obj = getattr(ref, t_container_name)
                found = set(_safe_get_t_attrs(t_obj))
                unknown = sorted(found.difference(allowed_map[t_container_name]))
                if unknown:
                    msg = (
                        f"Reference network has time-dependent attrs in {t_container_name} "
                        f"that are not included in stacking allowlist: {unknown}"
                    )
                    if strict_unknown_t:
                        raise ValueError(msg)
                    logger.warning(msg)

    # -------------------------------------------------------------------------
    # PASS 1: scan all scenarios, determine which columns are globally constant
    #         vs scenario/time-dependent
    # -------------------------------------------------------------------------
    scan_meta: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for i_scen, (scen_name, scen_file) in enumerate(zip(scenario_names, scenario_files), start=1):
        logger.info("Scanning scenario %d/%d: %s  file=%s", i_scen, n_scenarios, scen_name, scen_file)

        p_i = Path(scen_file)
        if not p_i.exists():
            raise FileNotFoundError(f"Scenario network not found: {scen_file}")

        net_i = ref if i_scen == 1 else pypsa.Network(str(p_i))

        if len(net_i.snapshots) == 0:
            raise ValueError(f"Scenario {scen_name}: network has no snapshots.")

        snaps_i = _ensure_datetime_snapshots(net_i.snapshots)
        if len(snaps_i) != len(base_snaps) or not base_snaps.equals(snaps_i):
            raise ValueError(
                f"Snapshot mismatch: scenario[0] vs '{scen_name}'\n"
                f"  [0]: n={len(base_snaps)}, {base_snaps[0]} ... {base_snaps[-1]}\n"
                f"  [{i_scen-1}]: n={len(snaps_i)}, {snaps_i[0]} ... {snaps_i[-1]}\n"
            )

        _validate_static_index_compatibility(
            ref,
            net_i,
            scenario_name=scen_name,
            name_normalization_mode=name_normalization_mode,
        )
        for t_container_name, allowed_attrs in _T_ATTRS_INPUT_ONLY.items():
            if not allowed_attrs:
                continue

            if not hasattr(net_i, t_container_name):
                continue

            net_t_i = getattr(net_i, t_container_name, None)
            if net_t_i is None:
                continue

            if warn_unknown_t or strict_unknown_t:
                found_attrs = set(_safe_get_t_attrs(net_t_i))
                unknown = sorted(found_attrs.difference(set(allowed_attrs)))
                if unknown:
                    msg = (
                        f"Scenario '{scen_name}' has attrs in {t_container_name} not included "
                        f"in stacking allowlist: {unknown}"
                    )
                    if strict_unknown_t:
                        raise ValueError(msg)
                    logger.warning(msg)

            for attr in allowed_attrs:
                try:
                    val = getattr(net_t_i, attr, None)
                except Exception:
                    val = None

                if val is None:
                    continue
                if isinstance(val, pd.Series):
                    val = val.to_frame()
                if not isinstance(val, pd.DataFrame) or val.empty:
                    continue

                df_norm = _normalize_df_columns(
                    val,
                    t_container_name=t_container_name,
                    static_lookup=static_lookup,
                    scenario_name=scen_name,
                    attr=attr,
                    name_normalization_mode=name_normalization_mode,
                )

                if df_norm.empty:
                    continue

                arr = df_norm.to_numpy(dtype=np.float32, copy=False)

                col_min = np.nanmin(arr, axis=0)
                col_max = np.nanmax(arr, axis=0)
                has_nan = np.isnan(arr).any(axis=0)
                is_const_this_scen = (~has_nan) & ((col_max - col_min) < CONST_TOL)
                const_vals_this_scen = arr[0, :]

                for j, canon_col in enumerate(df_norm.columns):
                    meta = scan_meta[t_container_name][attr].get(canon_col)
                    if meta is None:
                        scan_meta[t_container_name][attr][canon_col] = {
                            "present_count": 1,
                            "const_all": bool(is_const_this_scen[j]),
                            "const_value": float(const_vals_this_scen[j]) if bool(is_const_this_scen[j]) else np.nan,
                        }
                    else:
                        meta["present_count"] += 1
                        same_const = (
                            meta["const_all"]
                            and bool(is_const_this_scen[j])
                            and np.isfinite(meta["const_value"])
                            and np.isfinite(const_vals_this_scen[j])
                            and abs(float(meta["const_value"]) - float(const_vals_this_scen[j])) < CONST_TOL
                        )
                        meta["const_all"] = bool(same_const)
                        if not same_const:
                            meta["const_value"] = np.nan

        if i_scen > 1:
            del net_i
            gc.collect()

    # -------------------------------------------------------------------------
    # build final metadata: variable cols / constant cols per t_container.attr
    # -------------------------------------------------------------------------
    needed: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for t_container_name, attrs_meta in scan_meta.items():
        needed[t_container_name] = {}

        static_comp = _get_static_comp_name(t_container_name)
        static_df = getattr(n, static_comp, None) if static_comp and hasattr(n, static_comp) else None

        norm = _make_name_normalizer(name_normalization_mode)

        static_norm_order: List[str] = []
        if static_df is not None and len(static_df) > 0:
            static_norm_order = [norm(x) for x in static_df.index]

        for attr, col_meta in attrs_meta.items():
            if not col_meta:
                continue

            all_canon_cols = list(col_meta.keys())

            constant_canon_cols = []
            constant_vals = []

            variable_canon_cols = []

            for c in all_canon_cols:
                meta = col_meta[c]
                globally_constant = (
                    meta["present_count"] == n_scenarios
                    and bool(meta["const_all"])
                    and np.isfinite(meta["const_value"])
                )
                if globally_constant:
                    constant_canon_cols.append(c)
                    constant_vals.append(np.float32(meta["const_value"]))
                else:
                    variable_canon_cols.append(c)

            # stable output order:
            # first follow static component order, then append any extras
            variable_canon_cols = sorted(
                variable_canon_cols,
                key=lambda x: (
                    static_norm_order.index(x) if x in static_norm_order else 10**9,
                    x,
                ),
            )
            constant_canon_cols = sorted(
                constant_canon_cols,
                key=lambda x: (
                    static_norm_order.index(x) if x in static_norm_order else 10**9,
                    x,
                ),
            )

            # map canonical -> actual reference names wherever possible
            lookup = static_lookup.get(t_container_name, {})
            variable_actual_cols = pd.Index([lookup.get(c, c) for c in variable_canon_cols])
            constant_actual_cols = pd.Index([lookup.get(c, c) for c in constant_canon_cols])

            needed[t_container_name][attr] = {
                "variable_canon_cols": pd.Index(variable_canon_cols),
                "variable_actual_cols": variable_actual_cols,
                "constant_cols": constant_actual_cols,
                "constant_vals": np.asarray(constant_vals, dtype=np.float32),
                "has_variable": len(variable_canon_cols) > 0,
            }

    # -------------------------------------------------------------------------
    # allocate arrays for variable columns
    # -------------------------------------------------------------------------
    arrays: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for t_container_name, attrs in needed.items():
        arrays[t_container_name] = {}
        for attr, meta in attrs.items():
            if not meta["has_variable"]:
                arrays[t_container_name][attr] = None
                continue
            arrays[t_container_name][attr] = np.full(
                (n_scenarios * n_snaps, len(meta["variable_canon_cols"])),
                np.nan,
                dtype=np.float32,
            )

    # -------------------------------------------------------------------------
    # PASS 2: fill variable arrays for every scenario
    # -------------------------------------------------------------------------
    for i_scen, (scen_name, scen_file) in enumerate(zip(scenario_names, scenario_files)):
        logger.info(
            "Filling stacked arrays for scenario %d/%d: %s  file=%s",
            i_scen + 1, n_scenarios, scen_name, scen_file,
        )

        net_i = ref if i_scen == 0 else pypsa.Network(str(Path(scen_file)))
        start = i_scen * n_snaps
        end = start + n_snaps

        for t_container_name, attrs in needed.items():
            if not hasattr(net_i, t_container_name):
                continue

            net_t_i = getattr(net_i, t_container_name, None)
            if net_t_i is None:
                continue

            for attr, meta in attrs.items():
                if not meta["has_variable"]:
                    continue

                arr = arrays[t_container_name][attr]
                if arr is None:
                    continue

                df_i = getattr(net_t_i, attr, None)
                if df_i is None:
                    continue
                if isinstance(df_i, pd.Series):
                    df_i = df_i.to_frame()
                if not isinstance(df_i, pd.DataFrame) or df_i.empty:
                    continue

                df_norm = _normalize_df_columns(
                    df_i,
                    t_container_name=t_container_name,
                    static_lookup=static_lookup,
                    scenario_name=scen_name,
                    attr=attr,
                    name_normalization_mode=name_normalization_mode,
                )

                if df_norm.empty:
                    continue

                df_block = df_norm.reindex(
                    index=base_snaps,
                    columns=meta["variable_canon_cols"],
                )

                arr[start:end, :] = df_block.to_numpy(dtype=np.float32, copy=False)

        if i_scen > 0:
            del net_i
            gc.collect()

    # -------------------------------------------------------------------------
    # write static constants + stacked variable *_t frames back to network
    # -------------------------------------------------------------------------
    for t_container_name, attrs in needed.items():
        if not hasattr(n, t_container_name):
            continue

        n_t = getattr(n, t_container_name)

        for attr, meta in attrs.items():
            # push globally constant cols into static component table
            _apply_constant_cols_to_static(
                n,
                t_container_name=t_container_name,
                attr=attr,
                constant_cols=meta["constant_cols"],
                constant_vals=meta["constant_vals"],
            )

            # attach variable stacked dataframe
            if (
                meta["has_variable"]
                and arrays[t_container_name][attr] is not None
                and len(meta["variable_actual_cols"]) > 0
            ):
                df_var = pd.DataFrame(
                    arrays[t_container_name][attr],
                    index=stacked_snaps,
                    columns=meta["variable_actual_cols"],
                    copy=False,
                ).astype(np.float32, copy=False)

                setattr(n_t, attr, df_var)
                del arrays[t_container_name][attr]

    # -------------------------------------------------------------------------
    # cleanup and diagnostics
    # -------------------------------------------------------------------------
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
        "Stacked (robust): %d scen × %d ts = %d snaps | "
        "buses=%d gens=%d loads=%d links=%d lines=%d su=%d stores=%d | "
        "*_t frames: %.1f MB",
        n_scenarios, n_snaps, len(n.snapshots),
        len(n.buses), len(n.generators), len(n.loads),
        len(n.links), len(n.lines), len(n.storage_units), len(n.stores),
        total_bytes / 1e6,
    )

    # log short summary of constant vs variable classification
    for t_container_name, attrs in needed.items():
        for attr, meta in attrs.items():
            logger.info(
                "Stack meta %-16s %-20s | constant=%4d | variable=%4d",
                t_container_name,
                attr,
                len(meta["constant_cols"]),
                len(meta["variable_actual_cols"]),
            )


    # -------------------------------------------------------------------------
    # post-stacking diagnostics
    # -------------------------------------------------------------------------
    try:
        _debug_log_stacking_overview(
            n,
            scenario_names=list(scenario_names),
        )
    except Exception as exc:
        logger.warning("[STACK-DEBUG] post-stacking diagnostics failed: %s", exc)
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

def _compute_ls_p_nom_per_bus(
        n: pypsa.Network,
        safety_factor: float = 1.0,
        floor_mw: float = 100.0,
) -> Dict[str, float]:
    """
    [LS-FIX-1] Compute per-bus load-shedding p_nom from peak load.

    p_nom_bus = max(peak_load_at_bus × safety_factor, floor_mw)

    Using a global 1e8 MW caused Markowitz tolerance warnings because the
    resulting matrix coefficient span was ~5 orders of magnitude wider than
    physical capacity variables. Per-bus scaling keeps LS variables
    numerically comparable to physical generation.

    ASSUMPTION: peak load determined from loads_t.p_set timeseries.
    Falls back to static p_set, then to floor_mw if no data available.
    """
    bus_peak: Dict[str, float] = {}

    if hasattr(n, "loads") and len(n.loads) > 0 and "bus" in n.loads.columns:
        p_set_t = getattr(getattr(n, "loads_t", None), "p_set", None)
        for load_name, load_row in n.loads.iterrows():
            bus = str(load_row["bus"])
            if p_set_t is not None and isinstance(p_set_t, pd.DataFrame) and load_name in p_set_t.columns:
                peak = float(p_set_t[load_name].abs().max())
            elif "p_set" in load_row.index:
                peak = float(abs(load_row["p_set"] or 0.0))
            else:
                peak = 0.0
            bus_peak[bus] = bus_peak.get(bus, 0.0) + peak

    result: Dict[str, float] = {}
    for bus in n.buses.index:
        raw = bus_peak.get(str(bus), 0.0)
        result[str(bus)] = max(raw * safety_factor, floor_mw)
    return result


def _ensure_load_shedding_generators(
        n: pypsa.Network,
        *,
        carrier: str = "load_shedding",
        marginal_cost: float = 1e4,
        p_nom: float = 0.0,          # 0 = auto (per-bus from peak load × 2)
) -> List[str]:
    if hasattr(n, "generators") and "carrier" in n.generators.columns:
        load_gens = n.generators.index[n.generators.carrier.astype(str) == "load"].tolist()
        if load_gens:
            # FIX: Setze p_nom und p_nom_max basierend auf Peak-Load
            p_nom_map = _compute_ls_p_nom_per_bus(n)

            for gen in load_gens:
                bus = n.generators.loc[gen, 'bus']
                p_nom_bus = p_nom_map.get(str(bus), 100.0)
                n.generators.loc[gen, 'p_nom'] = p_nom_bus
                n.generators.loc[gen, 'p_nom_max'] = p_nom_bus
                n.generators.loc[gen, 'p_nom_extendable'] = False
                n.generators.loc[gen, 'marginal_cost'] = marginal_cost

            logger.info("Fixed %d existing 'load' generators: set p_nom from peak load.", len(load_gens))
            _ensure_ls_pmax_timeseries(n, load_gens)
            return load_gens



    if "load_shedding" not in n.carriers.index:
        n.add("Carrier", "load_shedding")

    # [LS-FIX-1] Per-bus p_nom: peak_load × 2.0, floor 100 MW.
    # If caller passes explicit p_nom > 0 that value is used for all buses
    # (backwards-compatible override). Otherwise compute per bus.
    if p_nom > 0:
        p_nom_map: Dict[str, float] = {str(bus): p_nom for bus in n.buses.index}
    else:
        p_nom_map = _compute_ls_p_nom_per_bus(n)

    logger.info(
        "Adding load-shedding generators (one per bus, marginal_cost=%.4g). "
        "p_nom range: [%.1f, %.1f] MW  (per-bus peak-load scaling).",
        marginal_cost,
        min(p_nom_map.values()) if p_nom_map else 0.0,
        max(p_nom_map.values()) if p_nom_map else 0.0,
    )
    ls_names: List[str] = []
    for bus in n.buses.index:
        name  = f"LS::{bus}"
        p_nom_bus = p_nom_map.get(str(bus), 100.0)
        ls_names.append(name)
        n.add(
            "Generator", name, bus=bus, carrier=carrier,
            p_nom=p_nom_bus, p_nom_extendable=False,
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
        # [BUG-FIX-11b] Use actual var dim to avoid xarray cross-product.
        actual_dim = var.dims[-1]
        cc_da = xr.DataArray(cc.to_numpy(), dims=[actual_dim],
                             coords={actual_dim: (actual_dim, comp_df.index.to_numpy())})
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


def _finite_stores(n: pypsa.Network) -> pd.DataFrame:
    s = n.stores
    is_ext = s.get("e_nom_extendable", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    e_nom_vals = s["e_nom"].fillna(0).astype(float)

    # Für nicht-extendable Stores: e_nom ist die fixe Kapazität.
    # Wenn e_nom nicht endlich-positiv → Store-p unbegrenzt → ausschließen.
    # KEIN is_inf_max Check — e_nom_max ist für nicht-extendable irrelevant
    # und wird von _fix_unbounded_infrastructure geändert (was den alten Check bricht).
    is_bad_nom = ~(np.isfinite(e_nom_vals) & (e_nom_vals > 0))
    exclude = (~is_ext) & is_bad_nom

    if exclude.any():
        logger.info(
            "_finite_stores: excluding %d unbounded non-extendable stores: %s  carriers=%s",
            int(exclude.sum()),
            s.index[exclude][:5].tolist(),
            s.loc[exclude, "carrier"].value_counts().to_dict(),
        )

        # IMMER loggen, nicht nur bei exclude.any()
        logger.info(
            "_finite_stores: total=%d  excluded=%d  included=%d  excluded_carriers=%s  "
            "co2_atm_e_nom=%s  co2_atm_ext=%s",
            len(s), int(exclude.sum()), int((~exclude).sum()),
            s.loc[exclude, "carrier"].value_counts().to_dict() if exclude.any() else {},
            str(s.loc["co2 atmosphere", "e_nom"]) if "co2 atmosphere" in s.index else "N/A",
            str(s.loc["co2 atmosphere", "e_nom_extendable"]) if "co2 atmosphere" in s.index else "N/A",
        )

    if "co2 atmosphere" in s.index:
        logger.warning(
            "_finite_stores: co2 atmosphere present with e_nom=%s e_nom_max=%s ext=%s capital_cost=%s marginal_cost=%s",
            s.at["co2 atmosphere", "e_nom"] if "e_nom" in s.columns else "NA",
            s.at["co2 atmosphere", "e_nom_max"] if "e_nom_max" in s.columns else "NA",
            s.at["co2 atmosphere", "e_nom_extendable"] if "e_nom_extendable" in s.columns else "NA",
            s.at["co2 atmosphere", "capital_cost"] if "capital_cost" in s.columns else "NA",
            s.at["co2 atmosphere", "marginal_cost"] if "marginal_cost" in s.columns else "NA",
        )
    return s[~exclude].copy()

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
        td = _get_time_dimension(var)
        var_s = _var_isel(var, td)
        wd = _w_da(var_s, td)
        # [BUG-FIX-11] Use actual variable dim name to avoid xarray cross-product.
        actual_dim = [d for d in var_s.dims if d != td][0]
        mc_s = comp_df["marginal_cost"].reindex(comp_df.index).fillna(0.0).astype(float)

        if mc_t is not None and not mc_t.empty:
            mc_t_s = (mc_t.reindex(columns=comp_df.index)
                      .reindex(index=n.snapshots[idx_list]).astype(float).fillna(mc_s))
            mc_da = xr.DataArray(mc_t_s.to_numpy(), dims=[td, actual_dim],
                                 coords={td: var_s.coords[td],
                                         actual_dim: (actual_dim, comp_df.index.to_numpy())})
        else:
            mc_da = xr.DataArray(mc_s.to_numpy(), dims=[actual_dim],
                                 coords={actual_dim: (actual_dim, comp_df.index.to_numpy())})

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
    _add_mc(_finite_stores(n), getattr(getattr(n, "stores_t", None), "marginal_cost", None), ["Store-p"], "Store")
    _add_mc(n.links,         getattr(getattr(n, "links_t",         None), "marginal_cost", None), ["Link-p0"],                                "Link")
    _add_uc_costs()

    if co2_cost_mode == "global_constraint_constant_cost":
        total = total + _build_co2_cost_expression(n, mask)
    elif co2_cost_mode != "off":
        raise ValueError(f"Unknown co2_cost_mode='{co2_cost_mode}'.")

    # === DIAGNOSE: Op-Cost Aufbau ===
    try:
        n_terms = len(getattr(total, 'terms', [])) if hasattr(total, 'terms') else -1
        logger.debug(
            "[DIAG] Op-cost expression built: scenario_mask sum=%d  annual_scale=%.4f  "
            "n_terms=%d  co2_mode=%s",
            int(mask.sum()), annual_scale, n_terms, co2_cost_mode,
        )
    except Exception:
        pass
    # === ENDE ===

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
        expr = _build_operational_cost_expression(
            n, mask,
            co2_cost_mode=co2_cost_mode,
            annual_scale=annual_scale,
        )
        _val = None
        for _attr in ("solution", "value"):
            try:
                _raw = getattr(expr, _attr, None)
                if _raw is not None:
                    _val = float(np.squeeze(
                        _raw.values if hasattr(_raw, "values") else _raw
                    ))
                    break
            except Exception:
                pass
        if _val is None:
            raise RuntimeError("Cannot evaluate cost expression (linopy version mismatch).")
        manual = _val
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
    # [HEAT-VENT-FIX] Heat vents must not be extendable
    _fix_heat_vents(n)
    # [DIST-GRID-FIX] Distribution grid p_nom = existing peak LV demand
    _fix_distribution_grid(n)

    _fix_negative_loads(n)

    # [DISPATCH-FIX-2] Relax e_initial for unbounded stores
    _fix_inf_store_initial(n)
    _fix_unbounded_infrastructure(n)

    # [DISPATCH-FIX-3] Prune zero-capacity assets to match master model size
    _prune_zero_capacity_assets(n)

    _drop_orphan_timeseries_columns(n)
    _debug_suspicious_assets(n, "dispatch_after_prune_before_ls")

    # === DIAGNOSE: Dispatch-Netzwerk nach Prune ===
    logger.info(
        "[DIAG] Dispatch network post-prune: gens=%d  links=%d  stores=%d  su=%d",
        len(n.generators), len(n.links), len(n.stores), len(n.storage_units),
    )
    # Gibt es noch LS-Gens?
    ls = n.generators[n.generators.carrier.isin(["load_shedding", "load"])]
    logger.info("[DIAG] LS generators pre-dispatch: %d  p_nom_total=%.1f GW  mc=%.0f EUR/MWh",
                len(ls), ls.p_nom.sum() / 1e3 if len(ls) else 0,
                ls.marginal_cost.mean() if len(ls) else 0)
    # === ENDE ===

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
    w = n.snapshot_weightings["objective"]
    logger.info("DISPATCH WEIGHT CHECK: min=%.3f max=%.3f sum=%.1f n=%d annual_scale=%.4f",
    w.min(), w.max(), w.sum(), len(w), annual_scale)

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

    # Harte Plausibilitätsprüfung: synthetische Generatoren dürfen keine portfolio-
    # relevanten optimalen Kapazitäten tragen.
    if hasattr(n, "generators") and "p_nom_opt" in n.generators.columns:
        syn = n.generators[
            n.generators.apply(lambda r: _is_synthetic_generator_row(r.name, r), axis=1)
        ].copy()
        if len(syn) > 0:
            bad = syn[syn["p_nom_opt"].fillna(0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0) > 1.0]
            if len(bad) > 0:
                cols_show = [c for c in ["carrier", "p_nom", "p_nom_opt", "capital_cost", "marginal_cost"] if c in bad.columns]
                logger.error(
                    "[PORTFOLIO-CHECK] Synthetic generators carry nontrivial p_nom_opt after portfolio transfer!"
                )
                logger.error("[PORTFOLIO-CHECK] Offending rows:\n%s", bad[cols_show].head(20).to_string())

    _debug_component_overview(n, f"dispatch_{cutout}_after_fix_portfolio")
    _debug_extendables(n, f"dispatch_{cutout}_after_fix_portfolio")
    _debug_nonextendable_infinite_assets(n, f"dispatch_{cutout}_after_fix_portfolio")
    _debug_timeseries_orphans(n, f"dispatch_{cutout}_after_fix_portfolio")

    # === DIAGNOSE: Portfolio-Kapazitäten ===
    for comp, opt_col, _, ext_col in [
        ("generators", "p_nom_opt", "p_nom", "p_nom_extendable"),
        ("links", "p_nom_opt", "p_nom", "p_nom_extendable"),
        ("stores", "e_nom_opt", "e_nom", "e_nom_extendable"),
    ]:
        df = getattr(n, comp)
        if opt_col not in df.columns:
            continue

        built = df[df[opt_col].fillna(0) > 1].copy()

        # Für Generatoren synthetische Carrier rausfiltern
        if comp == "generators" and len(built) > 0:
            synthetic_mask = built.apply(lambda r: _is_synthetic_generator_row(r.name, r), axis=1)
            synthetic = built[synthetic_mask]
            built = built[~synthetic_mask]

            if len(synthetic) > 0:
                cols_show = [c for c in ["carrier", opt_col, "capital_cost", "marginal_cost"] if c in synthetic.columns]
                logger.warning(
                    "[DIAG] Portfolio generators include %d synthetic assets (excluded from build summary).",
                    len(synthetic),
                )
                if cols_show:
                    logger.warning("[DIAG] Synthetic generator sample:\n%s",
                                   synthetic[cols_show].head(20).to_string())

        if len(built):
            logger.info(
                "[DIAG] Portfolio %s built (filtered): %s",
                comp,
                built.groupby("carrier")[opt_col]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .to_dict(),
            )
        else:
            logger.warning(
                "[DIAG] Portfolio %s: no non-synthetic built assets with %s > 1.",
                comp, opt_col,
            )
    # === ENDE ===

    _debug_suspicious_assets(n, f"dispatch_{cutout}_after_fix_portfolio")

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
    _debug_load_shedding_breakdown(n, f"dispatch_{cutout}_post_solve")
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

    _cast_bool_cols_for_netcdf(n_std)
    with _patch_xarray_nc4_bool():
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


def _apply_carrier_max_growth_limits(n: pypsa.Network) -> Dict[str, int]:
    """
    [BIOMASS-FIX] Carrier max_growth → p_nom_max auf extendable Assets.

    PyPSA-EE speichert Biomasse-Potenziale in n.carriers['max_growth'] und
    n.carriers['max_relative_growth'], aber der Optimizer liest diese NICHT.
    Ohne diesen Schritt explodiert extendable Biomasse mit capital_cost~0
    auf unrealistische Werte (beobachtet: 155 TW für DE, 2050).

    effective_limit = min(abs_limit, rel_limit)
        abs_limit  = carriers.max_growth [MW]
        rel_limit  = Σ p_nom_existing × max_relative_growth

    Jedes extendable Asset bekommt einen proportionalen Anteil von
    effective_limit (gewichtet nach p_nom). Bestehende engere p_nom_max
    werden nie gelockert.
    """
    if not hasattr(n, "carriers") or len(n.carriers) == 0:
        return {}
    carriers = n.carriers
    has_abs = "max_growth" in carriers.columns
    has_rel = "max_relative_growth" in carriers.columns
    if not has_abs and not has_rel:
        return {}

    updated: Dict[str, int] = {}

    for comp, nom_col, ext_col, max_col in [
        ("generators", "p_nom", "p_nom_extendable", "p_nom_max"),
        ("links", "p_nom", "p_nom_extendable", "p_nom_max"),
        ("storage_units", "p_nom", "p_nom_extendable", "p_nom_max"),
    ]:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0:
            continue
        if "carrier" not in df.columns or ext_col not in df.columns:
            continue
        if nom_col not in df.columns:
            continue
        is_ext = df[ext_col].fillna(False).astype(bool)
        if not is_ext.any():
            continue
        if max_col not in df.columns:
            df[max_col] = np.inf

        for carrier_name, grp in df[is_ext].groupby("carrier"):
            if carrier_name not in carriers.index:
                continue

            abs_limit = np.inf
            if has_abs:
                try:
                    v = float(carriers.at[carrier_name, "max_growth"])
                    if np.isfinite(v) and v > 0:
                        abs_limit = v
                except Exception:
                    pass

            rel_limit = np.inf
            if has_rel:
                try:
                    rv = float(carriers.at[carrier_name, "max_relative_growth"])
                    if np.isfinite(rv) and rv > 0:
                        all_idx = df.index[df["carrier"] == carrier_name]
                        existing = df.loc[all_idx, nom_col].fillna(0.0).sum()
                        rel_limit = float(existing) * rv
                except Exception:
                    pass

            effective = min(abs_limit, rel_limit)
            if not np.isfinite(effective):
                continue  # Kein Limit definiert → unverändert lassen

            grp_noms = grp[nom_col].fillna(0.0).astype(float)
            total = grp_noms.sum()
            n_gens = len(grp)
            count = 0

            for asset_idx in grp.index:
                cur_max = df.at[asset_idx, max_col]
                try:
                    cur_max = float(cur_max)
                except Exception:
                    cur_max = np.inf

                share = (grp_noms[asset_idx] / total) if total > 0 else (1.0 / n_gens)
                gen_limit = max(float(grp_noms[asset_idx]), effective * share)
                new_max = min(cur_max, gen_limit) if np.isfinite(cur_max) else gen_limit

                if new_max < cur_max - 1e-6:
                    df.at[asset_idx, max_col] = new_max
                    count += 1

            if count > 0:
                key = f"{comp}::{carrier_name}"
                updated[key] = count
                logger.info(
                    "[BIOMASS-FIX] carrier=%r (%s): p_nom_max=%.3e MW auf %d Asset(s) "
                    "(abs=%.3e, rel=%.3e).",
                    carrier_name, comp, effective, count, abs_limit, rel_limit,
                )

    if not updated:
        logger.debug("[BIOMASS-FIX] Keine Carrier-Limits angewendet.")
    return updated

def _fix_zero_capital_cost_extendables(n: pypsa.Network) -> int:
    """Backward-compatible wrapper around generic capital-cost validation."""
    fixed_by_comp = validate_extendable_capital_costs(
        n,
        stage="robust_master_pre_solve",
        strict=False,
        auto_fix=True,
        floor_cost=1.0,
        min_positive=0.0,
    )
    return int(sum(fixed_by_comp.values()))


def _fix_unbounded_infrastructure(n: pypsa.Network) -> Dict[str, int]:
    """
    [INFRA-FIX] Fix infinite capacity bounds to finite realistic values.

    PROBLEM:
    PyPSA-EE networks contain components with p_nom_max=inf or e_nom_max=inf,
    which leads to unrealistic expansion (observed: 155 TW biomass in DE0).
    Even with capital_cost > 0, the solver may expand to absurd levels
    if no physical constraint exists.

    This function caps:
    - Biomass generators: p_nom_max inf → 1e6 MW (1 TW per generator)
    - Gas stores: e_nom_max inf → 1e9 MWh (1 TWh per store)
    - Load-shedding generators: handled separately in _ensure_load_shedding_generators

    These caps are HIGH enough to not artificially constrain realistic scenarios,
    but LOW enough to prevent solver explosions.
    """
    fixed: Dict[str, int] = {}

    # Fix biomass generators
    # In _fix_unbounded_infrastructure, ersetze die bisherigen co2/gas Store Blöcke
    # durch diesen generellen Block:


    if hasattr(n, "generators") and len(n.generators) > 0:
        bio_mask = n.generators.carrier.str.contains("biomass", case=False, na=False)
        bio_inf = bio_mask & (n.generators.p_nom_max == np.inf)
        if bio_inf.any():
            n.generators.loc[bio_inf, "p_nom_max"] = 1e6
            count = int(bio_inf.sum())
            fixed["generators::biomass"] = count
            examples = n.generators.index[bio_inf][:3].tolist()
            logger.info(
                "[INFRA-FIX] Fixed %d biomass generators: p_nom_max inf→1e6 MW. "
                "Examples: %s", count, examples
            )
        # [BIOMASS-FIX] ENSPRESO speichert Jahrespotenzial in MWh als p_nom (MW)
        # z.B. DE0: 155e6 MWh -> faelschlich als 155 TW interpretiert
        # Korrektur: p_nom_MW = p_nom_MWh / 8760
        solid_mask = n.generators.carrier.str.lower() == "solid biomass"
        solid_big  = solid_mask & (n.generators.p_nom > 1e3)  # ENSPRESO: auch 10k-100k MW sind falsch
        if solid_big.any():
            n.generators.loc[solid_big, "p_nom"]     = (n.generators.loc[solid_big, "p_nom"] / 8760.0).clip(upper=1e6)
            n.generators.loc[solid_big, "p_nom_max"] = n.generators.loc[solid_big, "p_nom"]
            count_sb = int(solid_big.sum())
            fixed["generators::solid_biomass_pnom"] = count_sb
            logger.info("[INFRA-FIX] Converted %d solid biomass p_nom MWh→MW (÷8760). ", count_sb)
        # [BIOGAS-FIX] Biogas p_nom und p_nom_max cappen
        # ENSPRESO speichert Energiepotenziale (MWh) faelschlich als p_nom (MW)
        # z.B. FR0: 7.1e7 MWh -> wird als 71 TW Kapazitaet interpretiert
        bio_gas_mask = n.generators.carrier.str.lower() == 'biogas'
        bio_gas_big  = bio_gas_mask & (n.generators.p_nom > 1e3)  # ENSPRESO: Threshold gesenkt
        if bio_gas_big.any():
            n.generators.loc[bio_gas_big, 'p_nom']     = (n.generators.loc[bio_gas_big, 'p_nom'] / 8760.0).clip(upper=1e6)
            n.generators.loc[bio_gas_big, 'p_nom_max'] = n.generators.loc[bio_gas_big, 'p_nom']
            count_bg = int(bio_gas_big.sum())
            fixed['generators::biogas_cap'] = count_bg
            logger.info('[INFRA-FIX] Converted %d biogas generators p_nom MWh->MW (÷8760), capped at 1e6 MW', count_bg)
        # Auch kleinere biogas mit p_nom_max > p_nom_max-Limit cappen
        bio_gas_pmax_big = bio_gas_mask & (~bio_gas_big) & (n.generators.p_nom_max > 1e6)
        if bio_gas_pmax_big.any():
            n.generators.loc[bio_gas_pmax_big, 'p_nom_max'] = n.generators.loc[bio_gas_pmax_big, 'p_nom'].clip(upper=1e6)
    # Links mit p_nom_max=inf, mc>0, capital_cost≈0
    if hasattr(n, "links") and len(n.links) > 0:
        lk = n.links
        is_ext = lk.get("p_nom_extendable", False).fillna(False).astype(bool)
        inf_max = lk.get("p_nom_max", np.inf).fillna(np.inf).apply(np.isinf)
        mc_pos = lk.get("marginal_cost", 0).fillna(0).astype(float) > 0
        low_cc = lk.get("capital_cost", 0).fillna(0).astype(float) < 1.0

        to_fix = lk.index[is_ext & inf_max & mc_pos & low_cc]
        if len(to_fix) > 0:
            n.links.loc[to_fix, "capital_cost"] = 1.0  # 1 EUR/MW verhindert inf-Expansion
            fixed["links::zero_cc"] = len(to_fix)
            logger.info(
                "[INFRA-FIX] Set capital_cost=1 EUR/MW for %d links "
                "(p_nom_max=inf, mc>0, cc≈0). Carriers: %s",
                len(to_fix),
                lk.loc[to_fix, "carrier"].value_counts().to_dict(),
            )

    if hasattr(n, "stores") and len(n.stores) > 0:
        # Carrier-spezifische Obergrenzen (MWh)
        carrier_caps = {
            "co2 stored": 1e9,  # CCS-Speicher: 1 TWh
            "co2": 1e12,  # Atmosphäre: praktisch unbegrenzt
            "gas": 1e9,  # Gas-Speicher
            "H2 Store": 1e9,  # Wasserstoff
            "ammonia store": 1e9,
            "battery": 1e8,
            "home battery": 1e7,
            "biomass": 1e9,
            "coal": 1e9,
            "lignite": 1e9,
            "uranium": 1e12,  # Kernbrennstoff: sehr groß
            "oil": 1e9,
            "methanol": 1e9,
            "geothermal_heat": 1e8,
            "rural water tanks": 1e8,
            "urban central water pits": 1e8,
            "urban central water tanks": 1e8,
            "urban decentral water tanks": 1e8,
        }
        default_cap = 1e9  # für alle nicht gelisteten Carrier

        inf_mask = n.stores.e_nom_max.fillna(np.inf).apply(np.isinf)
        # Cap ALL extendable stores with inf e_nom_max, not just mc>0
        ext_mask = n.stores.e_nom_extendable.fillna(False).astype(bool)
        to_fix = n.stores.index[inf_mask & ext_mask]

        if len(to_fix) > 0:
            for store_name in to_fix:
                carrier = str(n.stores.at[store_name, "carrier"])
                cap = carrier_caps.get(carrier, default_cap)
                n.stores.at[store_name, "e_nom_max"] = cap

            count = len(to_fix)
            fixed["stores::all_unbounded"] = count
            logger.info(
                "[INFRA-FIX] Capped %d unbounded stores (e_nom_max inf→carrier-specific). "
                "Carriers: %s",
                count,
                n.stores.loc[to_fix, "carrier"].value_counts().to_dict(),
            )

    if not fixed:
        logger.debug("[INFRA-FIX] No unbounded infrastructure found.")

    return fixed


# =============================================================================
# [ARO-FIX-3]  Per-scenario GlobalConstraint equivalents
# =============================================================================

def _resolve_gc_emission_factors(
        network: pypsa.Network,
        carrier_attr: str,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Resolve generator and link emission/attribute factors for a GlobalConstraint.

    PyPSA's carrier_attribute field is used in two distinct ways:
      (a) As a column name in n.carriers  (e.g. 'co2_emissions')
          → ef_g = n.carriers.loc[gen_carrier, carrier_attr]
      (b) As a carrier NAME filter        (e.g. 'solid biomass', 'co2 sequestered')
          → ef_g = 1.0 if gen.carrier == carrier_attr else 0.0

    This function tries (a) first, then falls back to (b).

    Returns
    -------
    (gen_ef, link_ef): pd.Series indexed on generators/links, or (None, None)
    if neither mode yields non-zero factors.
    """
    if not hasattr(network, "generators") or len(network.generators) == 0:
        return None, None
    if "carrier" not in network.generators.columns:
        return None, None

    # Mode (a): carrier_attr is a column in n.carriers
    if (hasattr(network, "carriers")
            and carrier_attr in getattr(network.carriers, "columns", [])):
        gen_ef = (
            network.generators["carrier"]
            .map(network.carriers[carrier_attr])
            .fillna(0.0)
            .astype(float)
        )
        link_ef = None
        if (hasattr(network, "links") and len(network.links) > 0
                and "carrier" in network.links.columns):
            link_ef = (
                network.links["carrier"]
                .map(network.carriers[carrier_attr])
                .fillna(0.0)
                .astype(float)
            )
        if (gen_ef != 0).any() or (link_ef is not None and (link_ef != 0).any()):
            return gen_ef, link_ef
        return None, None

    # Mode (b): carrier_attr is a carrier name — indicator variable
    if (hasattr(network, "carriers")
            and carrier_attr in network.carriers.index):
        gen_ef = (network.generators["carrier"] == carrier_attr).astype(float)
        link_ef = None
        if (hasattr(network, "links") and len(network.links) > 0
                and "carrier" in network.links.columns):
            link_ef = (network.links["carrier"] == carrier_attr).astype(float)
        if (gen_ef != 0).any() or (link_ef is not None and (link_ef != 0).any()):
            logger.debug(
                "[ARO-FIX-3] carrier_attribute='%s' resolved as carrier NAME "
                "(not a column). Using indicator factors.", carrier_attr,
            )
            return gen_ef, link_ef
        return None, None

    return None, None



def _add_per_scenario_global_constraints(
        network: pypsa.Network,
        scenarios: Sequence[str],
        masks: Dict[str, np.ndarray],
        original_gc_constants: Dict[str, float],
        annual_scale: float,
) -> int:
    """
    [ARO-FIX-3] Add per-scenario equivalents of PyPSA's time-aggregated
    GlobalConstraints so each ARO scenario is independently constrained.

    See module docstring [ARO-FIX-3] for full rationale.

    Only GCs for which _resolve_gc_emission_factors succeeded are processed
    here — those are the ones that were neutralized before model build.
    GCs whose carrier_attribute could not be resolved were NOT neutralized
    and are handled natively by PyPSA (scaled by annual_scale).

    RETURN
    ------
    Number of per-scenario constraint rows added.
    """
    import xarray as xr

    m = network.model

    if not hasattr(network, "global_constraints") or len(network.global_constraints) == 0:
        return 0

    gc_df   = network.global_constraints
    n_added = 0

    gen_p,  _ = _get_linopy_var(m, ["Generator-p"],  strict=False)
    link_p, _ = _get_linopy_var(m, ["Link-p0"],       strict=False)

    if gen_p is None:
        logger.warning("[ARO-FIX-3] Generator-p not in model — no per-scenario GCs added.")
        return 0

    td      = _get_time_dimension(gen_p)
    gen_dim = _get_gen_dimension(gen_p)
    w_np    = network.snapshot_weightings["objective"].to_numpy(dtype=np.float32)

    for gc_name, original_constant in original_gc_constants.items():
        if gc_name not in gc_df.index:
            continue

        row          = gc_df.loc[gc_name]
        carrier_attr = str(row.get("carrier_attribute", "") or "")
        sense        = str(row.get("sense", "<=") or "<=").strip()

        if not carrier_attr:
            continue  # non-time-aggregated; handled natively by PyPSA

        gen_ef, link_ef = _resolve_gc_emission_factors(network, carrier_attr)
        if gen_ef is None:
            # Could not resolve — constraint was NOT neutralized; skip
            continue

        per_scenario_constant = original_constant / max(annual_scale, 1e-9)

        ef_da = xr.DataArray(
            gen_ef.to_numpy(), dims=[gen_dim],
            coords={gen_dim: (gen_dim, network.generators.index.to_numpy())},
        )

        # Link emission factors if resolvable and non-zero
        link_ef_da = None
        if link_ef is not None and link_p is not None and (link_ef != 0).any():
            link_dim = next(
                (d for d in link_p.dims
                 if d not in (td, "snapshot", "timestep", "time")),
                None,
            )
            if link_dim is not None:
                link_ef_da = xr.DataArray(
                    link_ef.to_numpy(), dims=[link_dim],
                    coords={link_dim: (link_dim, network.links.index.to_numpy())},
                )

        for s in scenarios:
            idx      = _mask_to_isel_indices(masks[s])
            idx_list = idx.tolist()

            gen_p_s = gen_p.isel({td: idx_list})
            w_s     = xr.DataArray(
                w_np[idx_list], dims=[td],
                coords={td: gen_p_s.coords[td]},
            )
            lhs_expr = (gen_p_s * ef_da * w_s).sum()

            if link_ef_da is not None and link_p is not None:
                link_td  = _get_time_dimension(link_p)
                link_p_s = link_p.isel({link_td: idx_list})
                link_w_s = xr.DataArray(
                    w_np[idx_list], dims=[link_td],
                    coords={link_td: link_p_s.coords[link_td]},
                )
                lhs_expr = lhs_expr + (link_p_s * link_ef_da * link_w_s).sum()

            con_name = f"gc_{gc_name}_per_scenario::{s}"
            try:
                if sense in ("<=", "le", "leq", "<"):
                    m.add_constraints(lhs_expr <= per_scenario_constant, name=con_name)
                elif sense in (">=", "ge", "geq", ">"):
                    m.add_constraints(lhs_expr >= per_scenario_constant, name=con_name)
                else:
                    m.add_constraints(lhs_expr == per_scenario_constant, name=con_name)
                n_added += 1
            except Exception as exc:
                logger.error(
                    "[ARO-FIX-3] Failed to add per-scenario constraint '%s': %s",
                    con_name, exc,
                )

        logger.info(
            "[ARO-FIX-3] GC '%s' (%s): added %d per-scenario constraints "
            "(budget=%.3e/scenario, carrier_attr='%s').",
            gc_name, sense, len(scenarios), per_scenario_constant, carrier_attr,
        )

    return n_added

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


    if solver_options is None:
        solver_options = {}

    ramp_data = _save_and_clear_ramp_limits(n)
    pruned = _prune_zero_capacity_assets(n)
    infra_fixed = _fix_unbounded_infrastructure(n)
    if infra_fixed:
        logger.info("[INFRA-FIX] Unbounded infrastructure capped: %s", infra_fixed)
    growth_limits = _apply_carrier_max_growth_limits(n)
    if growth_limits:
        logger.info("[BIOMASS-FIX] Carrier growth limits applied: %s",
                    list(growth_limits.keys()))
    _fix_zero_capital_cost_extendables(n)

    _debug_suspicious_assets(n, "master_after_fixes")

    # ------------------------------------------------------------------
    # HARD GUARDRAIL: carriers that must never be investment drivers
    # ------------------------------------------------------------------
    for comp, ext_col in [
        ("generators", "p_nom_extendable"),
        ("links", "p_nom_extendable"),
        ("storage_units", "p_nom_extendable"),
        ("stores", "e_nom_extendable"),
    ]:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0 or "carrier" not in df.columns or ext_col not in df.columns:
            continue

        forbid = df["carrier"].astype(str).isin({
            "load", "load_shedding", "net_export",
            "co2", "co2 stored", "co2 sequestered",
            "uranium", "coal", "lignite", "geothermal_heat",
            "oil refining",
        })

        n_forbid = int(forbid.sum())
        if n_forbid > 0:
            df.loc[forbid, ext_col] = False
            logger.warning(
                "[GUARDRAIL] %s: forced %d assets to %s=False for forbidden carriers=%s",
                comp, n_forbid, ext_col,
                sorted(df.loc[forbid, "carrier"].astype(str).unique().tolist()),
            )

    _debug_component_overview(n, "master_after_fixes")
    _debug_extendables(n, "master_after_fixes")
    _debug_nonextendable_infinite_assets(n, "master_after_fixes")
    _debug_timeseries_orphans(n, "master_after_fixes")

    # === DIAGNOSE: Investment-Potenzial ===
    for comp, nom_col, ext_col, cap_col in [
        ("generators", "p_nom", "p_nom_extendable", "capital_cost"),
        ("links", "p_nom", "p_nom_extendable", "capital_cost"),
        ("stores", "e_nom", "e_nom_extendable", "capital_cost"),
    ]:
        df = getattr(n, comp)
        if ext_col not in df.columns: continue
        ext = df[df[ext_col].fillna(False).astype(bool)]
        if len(ext) == 0: continue
        logger.info(
            "[DIAG] Extendable %s: n=%d  carriers=%s  cap_cost=[%.0f, %.0f] EUR/MW",
            comp, len(ext),
            ext.get("carrier", pd.Series(dtype=str)).value_counts().head(5).to_dict(),
            ext[cap_col].fillna(0).min(), ext[cap_col].fillna(0).max(),
        )
    # === ENDE ===

    for comp in ["generators", "links", "stores", "storage_units"]:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0 or "carrier" not in df.columns:
            continue

        suspicious = df[df["carrier"].astype(str).isin([
            "biomass", "solid biomass", "biogas", "coal", "lignite",
            "uranium", "geothermal_heat", "co2", "co2 stored",
            "oil refining", "electricity distribution grid",
            "urban central water pits charger", "urban central water pits discharger",
            "H2 Store"
        ])]

        if len(suspicious) > 0:
            cols_show = [c for c in [
                "carrier", "capital_cost", "marginal_cost",
                "p_nom", "p_nom_max", "e_nom", "e_nom_max",
                "p_nom_extendable", "e_nom_extendable"
            ] if c in suspicious.columns]
            logger.warning(
                "[master_pre_solve] suspicious %s assets found: n=%d carriers=%s",
                comp, len(suspicious),
                suspicious["carrier"].value_counts().to_dict()
            )
            logger.warning(
                "[master_pre_solve] suspicious %s sample:\n%s",
                comp, suspicious[cols_show].head(20).to_string()
            )

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

    logger.info("ARO Master (full-stack): %d scenarios: %s  ls_penalty=%.4g  annual_scale=%.3f",
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
        row          = n.global_constraints.loc[gc_name]
        carrier_attr = str(row.get("carrier_attribute", "") or "")
        sense        = str(row.get("sense", "<=") or "<=")
        original     = _original_gc_constants[gc_name]

        if carrier_attr:
            # [ARO-FIX-3] Time-aggregated constraint: neutralize PyPSA's native
            # N×T-summing version ONLY if we can successfully build a per-scenario
            # replacement (i.e. emission factors are resolvable).
            #
            # NEUTRALIZATION VALUE: abs(original) × 1000
            # — safely non-binding (1000× the actual budget)
            # — avoids extreme RHS values (1e+15 caused Barrier numerical failure)
            # — scales with the actual constraint magnitude
            #
            # If resolution fails, fall back to annual_scale approach so PyPSA's
            # native constraint remains active (possibly N× too permissive, but
            # better than silently dropping the constraint).
            gen_ef, _ = _resolve_gc_emission_factors(n, carrier_attr)
            if gen_ef is not None:
                # Sense-aware neutralization so the native PyPSA GC becomes non-binding
                # before we add per-scenario replacements in extra_master.
                #
                # <= : use very large positive RHS
                # >= : use very large negative RHS
                # == : cannot be neutralized with a single RHS shift; keep scaled native GC
                #      and log a warning.
                sense_norm = sense.replace(" ", "")
                if sense_norm in ("<=", "<", "le", "leq"):
                    # Remove GC entirely — per-scenario replacement added in extra_master
                    n.global_constraints.drop(gc_name, inplace=True)
                    logger.info("Master: GC '%s' dropped (per-scenario replacement active).", gc_name)
                    continue
                    logger.info(
                        "Master: GC '%s' (%s %.3e) → neutralized to %.3e (sense-aware, <=). "
                        "Per-scenario constraints added in extra_master. "
                        "carrier_attr='%s'.",
                        gc_name, sense, original, neutralized, carrier_attr,
                    )
                elif sense_norm in (">=", ">", "ge", "geq"):
                    # Remove GC entirely — per-scenario replacement added in extra_master
                    n.global_constraints.drop(gc_name, inplace=True)
                    logger.info("Master: GC '%s' dropped (per-scenario replacement active).", gc_name)
                    continue
                    logger.info(
                        "Master: GC '%s' (%s %.3e) → neutralized to %.3e (sense-aware, >=). "
                        "Per-scenario constraints added in extra_master. "
                        "carrier_attr='%s'.",
                        gc_name, sense, original, neutralized, carrier_attr,
                    )
                else:
                    scaled = original / annual_scale
                    n.global_constraints.at[gc_name, "constant"] = scaled
                    logger.warning(
                        "Master: GC '%s' has equality/unknown sense '%s'; "
                        "cannot neutralize safely. Keeping scaled native GC %.3e and "
                        "also adding per-scenario constraints.",
                        gc_name, sense, scaled,
                    )
            else:
                scaled = original / annual_scale
                n.global_constraints.at[gc_name, "constant"] = scaled
                logger.warning(
                    "Master: GC '%s' (%s %.3e) → scaled to %.3e (÷%.2fx). "
                    "carrier_attribute='%s' could not be resolved — "
                    "per-scenario replacement SKIPPED. "
                    "Constraint sums over ALL N×T snapshots (may be N× too permissive).",
                    gc_name, sense, original, scaled, annual_scale, carrier_attr,
                )
        else:
            # Non-time-aggregated constraint (e.g. investment limits): scale
            # by annual_scale so PyPSA handles it natively.
            scaled = original / annual_scale
            n.global_constraints.at[gc_name, "constant"] = scaled
            logger.info(
                "Master: GC '%s' (%s %.3e) → %.3e (÷%.2fx, non-time-aggregated).",
                gc_name, sense, original, scaled, annual_scale,
            )

    # [MEM-PATCH-B] Build weight array once
    w_base_np = n.snapshot_weightings["objective"].to_numpy(dtype=np.float32, copy=False)

    # [DISPATCH-FIX-1] Convert negative loads to generators (methodically correct)
    # Replaces the previous inline clip which incorrectly removed energy from balance
    # [HEAT-VENT-FIX] Heat vents must not be extendable
    _fix_heat_vents(n)
    # [DIST-GRID-FIX] Distribution grid p_nom = existing peak LV demand
    _fix_distribution_grid(n)
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
        # at scenario boundaries...
        _fix_cross_scenario_storage_constraints(network, scenarios, masks)

        # [ARO-FIX-3] Add per-scenario GlobalConstraint equivalents...
        _add_per_scenario_global_constraints(
            network, scenarios, masks, _original_gc_constants, annual_scale,
        )

        # === DIAGNOSE: Modellgröße ===
        logger.info(
            "[DIAG] Variables in model: %s",
            list(m.variables)[:20],
        )
        logger.info(
            "[DIAG] Constraints in model: %d  names: %s",
            len(m.constraints),
            list(m.constraints)[:10],
        )
        # === ENDE ===

        _add_within_scenario_ramp_constraints(
            network, scenarios, masks, ramp_data,
            fail_on_extendable=fail_on_extendable_ramps,
        )

        # [ARO-FIX-2] lower=-np.inf  (was lower=0).
        z_theta = m.add_variables(lower=-np.inf, name="z_theta")
        inv_cost = _build_investment_cost_expression(network)

        for s in scenarios:
            op_cost_s = _build_operational_cost_expression(
                network,
                masks[s],
                co2_cost_mode=co2_cost_mode,
                w_override_np=w_base_np,
                annual_scale=annual_scale,
            )
            # op_cost_s enthält LS-Kosten (Penalty-Mechanismus für ARO)
            # [DIAG] Obere Schranke für op_cost_s berechnen
            try:
                print('ALL CARRIERS:', network.generators.carrier.value_counts().to_dict())
                ls_gens = network.generators[network.generators.carrier.isin(['load','load_shedding'])]
                ls_mc = ls_gens['marginal_cost'].mean() if len(ls_gens) > 0 else 0
                ls_pnom = ls_gens['p_nom'].sum() if len(ls_gens) > 0 else 0
                w_sum = float(w_base_np[masks[s]].sum())
                max_ls = ls_mc * ls_pnom * w_sum
                logger.info("[DIAG-OPCOST] s=%s  ls_mc=%.2f  ls_pnom=%.3e MW  w_sum=%.1f  max_ls=%.3e",
                            s, ls_mc, ls_pnom, w_sum, max_ls)
                # Biomasse als weiterer Treiber
                bio = network.generators[network.generators.carrier.isin(['solid biomass','biogas'])]
                bio_mc = bio['marginal_cost'].mean() if len(bio) > 0 else 0
                bio_pnom = bio['p_nom'].sum() if len(bio) > 0 else 0
                max_bio = bio_mc * bio_pnom * w_sum
                logger.info("[DIAG-OPCOST] s=%s  bio_mc=%.2f  bio_pnom=%.3e MW  max_bio=%.3e",
                            s, bio_mc, bio_pnom, max_bio)
            except Exception as _e:
                logger.warning("[DIAG-OPCOST] Fehler: %s", _e)
            m.add_constraints(
                1.0 * z_theta >= op_cost_s,
                name=f"robust_op_epigraph::{s}",
            )
            # DIAG: Evaluiere op_cost_s Terme
            try:
                import xarray as _xr
                _terms = list(getattr(op_cost_s, 'data', {}).items()) if hasattr(op_cost_s, 'data') else []
                logger.info("[DIAG-OPCOST] op_cost_s terms: %d  type=%s", len(_terms), type(op_cost_s).__name__)
                # Zähle Koeffizienten
                if hasattr(op_cost_s, 'coeffs'):
                    c = op_cost_s.coeffs.values.ravel()
                    c = c[~np.isnan(c) & (c != 0)]
                    logger.info("[DIAG-OPCOST] coeffs: n=%d  min=%.3e  max=%.3e  sum=%.3e", len(c), float(c.min()), float(c.max()), float(c.sum()))
            except Exception as _de:
                logger.warning("[DIAG-OPCOST] eval failed: %s", _de)
            del op_cost_s
        m.objective = inv_cost + 1.0 * z_theta

        # === DIAGNOSE: Zielfunktion ===
        logger.info("[DIAG] Objective set: inv_cost + z_theta. Epigraph constraints: %d",
                    sum(1 for k in m.constraints if k.startswith("robust_op_epigraph")))
        # Prüfe ob inv_cost > 0 (sonst keine Investment-Variablen)
        try:
            inv_terms = sum(1 for k in m.variables if "p_nom" in k or "e_nom" in k or "s_nom" in k)
            logger.info("[DIAG] Investment variables in model: %d (p_nom/e_nom/s_nom)", inv_terms)
        except Exception:
            pass
        # === ENDE ===


    logger.info("ARO Master: solving min Inv + worst-case Op (full-stack, %d scenarios) ...",
                len(scenarios))

    # === BILLIONEN-DIAGNOSE ===
    w = n.snapshot_weightings["objective"]
    logger.info("WEIGHTS: min=%.3f  max=%.3f  sum=%.1f  n=%d  n_scen=%d",
                w.min(), w.max(), w.sum(), len(w), len(scenarios))
    logger.info("SCALE:   hours_per_scenario=%.1f  annual_scale=%.4f",
                hours_per_scenario, annual_scale)
    logger.info("LS_MC:   ls_mc_adjusted=%.4f EUR/MWh  (ls_penalty=%.4g / annual_scale=%.4f)",
                ls_mc_adjusted, ls_penalty, annual_scale)

    # Maximale theoretische LS-Kosten
    ls_gens_check = n.generators[n.generators.carrier.isin(["load_shedding", "load"])]
    max_ls_mwh = float(ls_gens_check.p_nom.sum()) * float(w.sum())
    max_ls_cost = max_ls_mwh * ls_mc_adjusted
    logger.info("MAX_THEORETICAL_LS: p_nom_total=%.1f GW  max_mwh=%.3e  max_cost=%.3e EUR",
                ls_gens_check.p_nom.sum() / 1e3, max_ls_mwh, max_ls_cost)

    # Extendables nach Prune
    for comp in ["generators", "links", "storage_units"]:
        df = getattr(n, comp)
        n_ext = df.get("p_nom_extendable", pd.Series(False)).fillna(False).sum()
        logger.info("EXTENDABLE_%s: %d assets", comp.upper(), n_ext)
    # === ENDE DIAGNOSE ===

    for comp, ext_col in [
        ("generators", "p_nom_extendable"),
        ("links", "p_nom_extendable"),
        ("storage_units", "p_nom_extendable"),
        ("stores", "e_nom_extendable"),
    ]:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0 or ext_col not in df.columns or "capital_cost" not in df.columns:
            continue
        ext = df[df[ext_col].fillna(False).astype(bool)].copy()
        if len(ext) == 0:
            continue

        cols_show = [c for c in ["carrier", "capital_cost", "marginal_cost", "p_nom_max", "e_nom_max"] if c in ext.columns]
        if cols_show:
            logger.info(
                "[master_pre_optimize] %s extendables sorted by capital_cost:\n%s",
                comp,
                ext[cols_show].sort_values("capital_cost").head(25).to_string()
            )

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

    # ⚠️ DIAGNOSE: Wieviel Load Shedding wurde genutzt? ⚠️
    if hasattr(n, 'generators_t') and hasattr(n.generators_t, 'p'):
        gen_p_df = n.generators_t.p
        ls_gens = n.generators.index[
            n.generators["carrier"].isin(["load_shedding", "load"])
        ]
        if len(ls_gens) > 0 and isinstance(gen_p_df, pd.DataFrame) and not gen_p_df.empty:
            ls_cols = gen_p_df.columns.intersection(ls_gens)
            if len(ls_cols) > 0:
                ls_usage_total = float(gen_p_df[ls_cols].sum().sum())  # Total MWh
                ls_usage_max = float(gen_p_df[ls_cols].max().max())  # Peak MW

                if ls_usage_total > 1.0:
                    logger.error(
                        "⚠️  EXCESSIVE LOAD SHEDDING: %.2f MWh total, %.2f MW peak! "
                        "Check network feasibility or increase penalty.",
                        ls_usage_total, ls_usage_max
                    )
                elif ls_usage_total > 0.01:
                    logger.warning(
                        "⚠️  Minor load shedding: %.4f MWh total, %.2f MW peak.",
                        ls_usage_total, ls_usage_max
                    )
                else:
                    logger.info("✓ No significant load shedding detected (%.2e MWh).", ls_usage_total)

    # [BUG-FIX-3] Restore GlobalConstraint constants
    for gc_name, orig_val in _original_gc_constants.items():
        if gc_name in n.global_constraints.index:
            n.global_constraints.at[gc_name, "constant"] = orig_val
    logger.info("GlobalConstraint constants restored.")

    # [BUG-FIX-4] Restore ramp limits and committable flags
    _restore_ramp_limits(n, ramp_data)

    # [BUG-FIX-5] Restore cyclic storage flags
    # Cast to int8 (not bool) — netCDF4 rejects bool dtype.
    for (comp, col), series in e_cyclic_backup.items():
        df = getattr(n, comp, None)
        if df is not None and col in df.columns:
            df[col] = series.astype("int8")
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
# Synthetic generator cleanup (NegLoad::, LS::)
# =============================================================================

def _remove_synthetic_generators(n: pypsa.Network) -> int:
    """
    Remove NegLoad:: and LS:: generators inserted by preprocessing before
    the network is exported.

    WHY THIS IS NECESSARY
    ---------------------
    _fix_negative_loads() and _ensure_load_shedding_generators() mutate ``n``
    in-place by adding synthetic generators. Without cleanup these appear in
    the exported NetCDF as installed capacity entries (p_nom > 0, carrier
    "net_export" / "load_shedding"), causing downstream postprocessing tools
    that iterate over all generators to miscount capacity or dispatch.

    extract_capacities() already filters them out (not p_nom_extendable), but
    any tool that accesses n.generators directly will see them.

    WHAT IS REMOVED
    ---------------
    - Generators whose name starts with "NegLoad::"
    - Generators whose name starts with "LS::"
    - Corresponding columns in generators_t time-series frames

    RETURN
    ------
    Number of generators removed.
    """
    if not hasattr(n, "generators") or len(n.generators) == 0:
        return 0

    prefixes = ("NegLoad::", "LS::")
    to_remove = [
        g for g in n.generators.index
        if any(str(g).startswith(p) for p in prefixes)
    ]
    if not to_remove:
        return 0

    n.generators = n.generators.drop(index=to_remove)

    if hasattr(n, "generators_t"):
        for attr in vars(n.generators_t):
            df = getattr(n.generators_t, attr, None)
            if isinstance(df, pd.DataFrame) and len(df.columns) > 0:
                cols_drop = df.columns.intersection(to_remove)
                if len(cols_drop) > 0:
                    setattr(n.generators_t, attr, df.drop(columns=cols_drop))

    logger.info(
        "_remove_synthetic_generators: removed %d synthetic generators %s.",
        len(to_remove), to_remove[:6],
    )
    return len(to_remove)




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


def _cast_bool_cols_for_netcdf(n: pypsa.Network) -> None:
    """
    Cast bool-dtype columns to int8 before netCDF4 export.

    Root cause: restoring e_cyclic_backup sets bool dtype on columns like
    cyclic_state_of_charge. NetCDF4 rejects bool; int8 is fine.

    Access path: in PyPSA 1.x, component DataFrames live in
    n.components[comp_name].static and n.components[comp_name].dynamic.
    Accessing via getattr(n, list_name) goes through a property that returns
    the same object, but computing list_name from comp_name is error-prone
    (e.g. 'StorageUnit' → fallback 'storageunits', not 'storage_units').
    We access .static/.dynamic directly to avoid that.
    """
    def _cast_df(df: pd.DataFrame) -> None:
        for c in list(df.columns):
            try:
                if df[c].dtype == bool or str(df[c].dtype) == "bool":
                    df[c] = df[c].astype("int8")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # PyPSA 1.x path: n.components[comp].static / .dynamic
    # ------------------------------------------------------------------
    comps = getattr(n, "components", None)
    all_comps = getattr(n, "all_components", None)

    if comps is not None and all_comps is not None:
        for comp_name in all_comps:
            try:
                comp_obj = comps[comp_name]
                # Static DataFrame
                static = getattr(comp_obj, "static", None)
                if isinstance(static, pd.DataFrame) and not static.empty:
                    _cast_df(static)
                # Dynamic DataFrames (time-series)
                dynamic = getattr(comp_obj, "dynamic", None)
                if dynamic is not None:
                    _iter = (dynamic.values()
                             if hasattr(dynamic, "values")
                             else dynamic.__dict__.values()
                             if hasattr(dynamic, "__dict__") else [])
                    for df in _iter:
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            _cast_df(df)
            except Exception:
                pass
    else:
        # ------------------------------------------------------------------
        # Fallback: older PyPSA — use explicit component list
        # ------------------------------------------------------------------
        _COMP_ATTRS = [
            "buses", "generators", "loads", "links", "lines",
            "transformers", "storage_units", "stores", "carriers",
            "global_constraints", "sub_networks",
            "investment_periods", "investment_period_weightings",
        ]
        for attr in _COMP_ATTRS:
            try:
                df = getattr(n, attr, None)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    _cast_df(df)
            except Exception:
                pass
        for t_name in list(_T_ATTRS.keys()):
            try:
                t_obj = getattr(n, t_name, None)
                if t_obj is None:
                    continue
                for sub in list(vars(t_obj)):
                    try:
                        df = getattr(t_obj, sub, None)
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            _cast_df(df)
                    except Exception:
                        pass
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Always cast these regardless of version (not in component registry)
    # ------------------------------------------------------------------
    for attr in ("snapshot_weightings", "investment_period_weightings"):
        try:
            df = getattr(n, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                _cast_df(df)
        except Exception:
            pass


@contextlib.contextmanager
def _patch_xarray_nc4_bool():
    """
    Temporarily patch xarray's netCDF4 backend to silently convert bool
    variables to int8.

    ROOT CAUSE: PyPSA 1.x builds an xarray Dataset internally during
    export_to_netcdf. Some variables (e.g. cyclic_state_of_charge in
    StorageUnit, p_nom_extendable in generators) retain numpy bool dtype
    through xarray serialisation. NetCDF4's _nc4_dtype raises ValueError
    for bool — it has no native bool type. Casting the source DataFrames
    is unreliable because PyPSA's exporter may access internal structures
    not reachable from Python-side DataFrame mutations.

    This context manager patches _nc4_dtype at the xarray backend level,
    guaranteeing the fix regardless of PyPSA version or internal data path.
    """
    import numpy as np
    try:
        import xarray.backends.netCDF4_ as _xr_nc4
        _orig_fn = _xr_nc4._nc4_dtype

        def _safe_nc4_dtype(var):
            if getattr(var, 'dtype', None) == np.dtype('bool'):
                var = var.astype(np.int8)
            return _orig_fn(var)

        _xr_nc4._nc4_dtype = _safe_nc4_dtype
        try:
            yield
        finally:
            _xr_nc4._nc4_dtype = _orig_fn
    except Exception:
        # If patching fails for any reason, proceed unpatched
        yield


def export_network_flat_snapshots(n: pypsa.Network, out_network: str) -> None:
    if getattr(n, "model", None) is not None:
        try:
            n.model.solver_model = None
        except Exception:
            pass

    Path(out_network).parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(n.snapshots, pd.MultiIndex):
        _cast_bool_cols_for_netcdf(n)
        with _patch_xarray_nc4_bool():
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
        _cast_bool_cols_for_netcdf(n)
        with _patch_xarray_nc4_bool():
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
    _cast_bool_cols_for_netcdf(n)
    with _patch_xarray_nc4_bool():
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
    _cast_bool_cols_for_netcdf(n_std)
    with _patch_xarray_nc4_bool():
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

    n, scen = stack_scenarios_to_multisnapshot_network(
        scenario_files=scenario_files,
        scenario_names=list(cutouts),
        warn_unknown_t=warn_unknown_t,
        strict_unknown_t=strict_unknown_t,
        name_normalization_mode="exact",
    )

    _drop_orphan_timeseries_columns(n)
    _debug_timeseries_orphans(n, "after_orphan_drop")

    _debug_component_overview(n, "after_stacking")
    _debug_extendables(n, "after_stacking")
    _debug_nonextendable_infinite_assets(n, "after_stacking")
    _debug_timeseries_orphans(n, "after_stacking")

    logger.info("Applying infrastructure fixes to stacked master network ...")
    _fix_unbounded_infrastructure(n)
    _fix_zero_capital_cost_extendables(n)

    diag = solve_aro_master(
        n,
        solver_name=solver_name,
        solver_options=solver_options,
        hard_fail_suboptimal=not allow_suboptimal,
        fail_on_extendable_ramps=not allow_extendable_ramps,
        co2_cost_mode=co2_cost_mode,
        ls_penalty=ls_penalty,
    )

    try:
        masks_verify    = _scenario_masks_from_snapshots(n.snapshots)
        scenarios_verify = list(masks_verify.keys())
        annual_scale_v  = 8760.0 / max(
            1.0, float(n.snapshot_weightings["objective"].sum()) / max(1, len(scenarios_verify))
        )
        w_base_v = n.snapshot_weightings["objective"].to_numpy(dtype=np.float32)
        op_costs_verify: Dict[str, float] = {}
        for sv in scenarios_verify:
            expr = _build_operational_cost_expression(
                n, masks_verify[sv],
                co2_cost_mode=co2_cost_mode,
                w_override_np=w_base_v,
                annual_scale=annual_scale_v,
            )
            _val = None
            for _attr in ("solution", "value"):
                try:
                    _raw = getattr(expr, _attr, None)
                    if _raw is not None:
                        _val = float(np.squeeze(
                            _raw.values if hasattr(_raw, "values") else _raw
                        ))
                        break
                except Exception:
                    pass
            if _val is None:
                try:
                    import xarray as _xr
                    _acc = 0.0
                    for _term in getattr(expr, "terms", []):
                        _coef = float(np.squeeze(_term.coeffs.values))
                        _var_val = float(np.squeeze(_term.vars.solution.values))
                        _acc += _coef * _var_val
                    _val = _acc
                except Exception as _e2:
                    raise RuntimeError(
                        f"Cannot evaluate LinearExpression (linopy version mismatch): {_e2}"
                    ) from _e2
            op_costs_verify[sv] = _val
        worst_case_op = max(op_costs_verify.values())
        z_star        = diag.get("z_theta_star", float("nan"))
        rel_gap       = abs(z_star - worst_case_op) / max(1.0, abs(z_star))
        diag["robustness_verify"] = {
            "z_theta_star":   z_star,
            "worst_case_op":  worst_case_op,
            "per_scenario":   op_costs_verify,
            "rel_gap":        rel_gap,
            "ok":             rel_gap < 0.01,
        }
        if rel_gap >= 0.01:
            logger.warning(
                "Robustness verification MISMATCH: z_theta*=%.6g  "
                "max_s Op(x*,s)=%.6g  rel_gap=%.2f%%  "
                "— check annual_scale, co2_cost_mode, and component coverage.",
                z_star, worst_case_op, rel_gap * 100,
            )
        else:
            logger.info(
                "Robustness verification OK: z_theta*=%.6g  "
                "max_s Op(x*,s)=%.6g  rel_gap=%.4f%%.",
                z_star, worst_case_op, rel_gap * 100,
            )
    except Exception as _ve:
        logger.warning("Robustness verification failed (non-fatal): %s", _ve)
        diag["robustness_verify"] = {"ok": None, "error": str(_ve)}

    capacities = extract_capacities(n)

    Path(out_network).parent.mkdir(parents=True, exist_ok=True)
    Path(out_summary_json).parent.mkdir(parents=True, exist_ok=True)

    z = diag.get("z_theta_star", float("nan"))
    ls_pen = diag.get("ls_penalty", float("nan"))
    n_scen = diag.get("n_scenarios", 1)
    logger.info(
        "[DIAG] Master result: z_theta*=%.3e  ls_penalty=%.4g  "
        "max_realistic_z=%.3e (= ls_penalty × 8760h × peak_load_approx)",
        z, ls_pen, ls_pen * 8760 * 800e3
    )
    if z > ls_pen * 8760 * 1e6:
        logger.error("[DIAG] z_theta* implausibly large — likely still driven by LS or unbounded stores!")
    elif z < 1e9:
        logger.warning("[DIAG] z_theta* very small (%.3e) — check if op-cost expression is empty", z)
    else:
        logger.info("[DIAG] z_theta* in plausible range ✓")

    logger.info("Exporting stacked robust network → %s", out_network)
    _ensure_standard_pypsa_result_frames(n)
    _remove_synthetic_generators(n)
    _cast_bool_cols_for_netcdf(n)
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
        "scenario_names":    list(scen),
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