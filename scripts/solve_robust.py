#!/usr/bin/env python3
"""
solve_robust.py  (cutout-native inputs, shared-investment robust 2050 design)

Target
------
You want ONE single 2050 investment design (one portfolio x) that is feasible/optimal
across ALL manipulated weather cutouts (discrete scenario set S).

This script therefore implements a *shared-investment* scenario-robust model:

    min_x  max_{s in S}  min_{dispatch u_s}  Cost(x, u_s; weather_s)
    with lexicographic priority:
      Stage 1: minimise worst-case load shedding energy
      Stage 2: given optimal worst-case LS, minimise worst-case total cost

Key engineering idea
--------------------
We do NOT build "one network per cutout" conceptually. We only *use* per-cutout
prepared networks as intermediate artefacts of the standard PyPSA-Eur workflow.

We then stack them into ONE PyPSA Network with MultiIndex snapshots:
    snapshots = MultiIndex(["scenario","time"])
Static assets are shared; time series vary by scenario; investment variables are shared.

IMPORTANT: scenario boundary coupling
-------------------------------------
Stacking scenarios back-to-back introduces artificial adjacency between scenarios,
which can incorrectly couple intertemporal states (e.g., storage SOC, store energy).
We therefore add explicit per-scenario boundary constraints:
    SOC(s, first_t) == SOC(s, last_t)
and likewise for Store energy (if present).

Inputs
------
This script accepts ONLY cutouts as "semantic input" (scenario ids), but it does not
run Snakemake itself. It assumes the PyPSA-Eur pipeline has produced per-cutout
scenario networks and resolves them via a template.

Required CLI:
  --cutouts scenA scenB ...
  --scenario-network-template "networks/prepared_{cutout}.nc"
  --out-network ...
  --out-summary-json ...

Hard invariants (enforced)
--------------------------
- All scenario networks MUST have identical snapshots (same time index).
- All scenario networks MUST have identical static assets (component indices).
If not, the robust methodology is invalid and we abort.

Outputs
-------
- robust network: .nc
- robust summary: .json (diagnostics + capacities + scenario list)

"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


# =============================================================================
# Small utilities
# =============================================================================
def _is_dataframe_like(x) -> bool:
    return isinstance(x, (pd.DataFrame, pd.Series))


def _list_time_dependent_frames(obj) -> Dict[str, pd.DataFrame]:
    """
    Return all public attributes of a PyPSA *_t container (e.g. network.generators_t)
    that look like pandas DataFrames/Series.
    """
    out: Dict[str, pd.DataFrame] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if _is_dataframe_like(val):
            if isinstance(val, pd.Series):
                val = val.to_frame()
            out[name] = val
    return out


def _resolve_scenario_networks_from_cutouts(cutouts: Sequence[str], template: str) -> List[str]:
    """
    Resolve scenario network file paths from cutout ids.
    Example template: "networks/prepared_{cutout}.nc"
    """
    return [template.format(cutout=c) for c in cutouts]


# =============================================================================
# Formal consistency checks for shared-investment scenario stacking
# =============================================================================
def _assert_same_static_assets_and_snapshots(networks: Sequence[pypsa.Network]) -> None:
    """
    Enforce that all scenarios represent THE SAME SYSTEM (same assets, same time grid).

    - Static asset indices must match for key components.
    - Snapshot index must match EXACTLY (no reindexing; hiding mismatch breaks meaning).
    """
    if len(networks) < 2:
        return

    ref = networks[0]

    def _check_component(name: str) -> None:
        if not hasattr(ref, name):
            return
        ref_df = getattr(ref, name)

        for i, n in enumerate(networks[1:], start=1):
            df = getattr(n, name)
            if not ref_df.index.equals(df.index):
                missing = ref_df.index.difference(df.index)
                extra = df.index.difference(ref_df.index)
                msg = f"Static asset mismatch in '{name}' (scenario 0 vs {i}):\n"
                if len(missing) > 0:
                    msg += f"  Missing in scenario {i}: {list(missing)[:10]}\n"
                if len(extra) > 0:
                    msg += f"  Extra in scenario {i}: {list(extra)[:10]}\n"
                raise ValueError(msg)

            # Warn on critical column differences (structure should not depend on cutout)
            critical_cols = ["bus", "carrier", "p_nom_extendable", "capital_cost", "efficiency"]
            for col in critical_cols:
                if col in ref_df.columns and col in df.columns and not ref_df[col].equals(df[col]):
                    logger.warning(
                        "Static attribute differs: component=%s col=%s scenario=%d. "
                        "This is unusual for cutout-only scenario differences.",
                        name, col, i
                    )

    for comp in ["buses", "generators", "loads", "links", "lines", "transformers", "storage_units", "stores"]:
        if hasattr(ref, comp):
            _check_component(comp)

    # Snapshots MUST match exactly (your cutouts have identical time ranges)
    ref_snaps = ref.snapshots
    for i, n in enumerate(networks[1:], start=1):
        if not ref_snaps.equals(n.snapshots):
            raise ValueError(
                "Snapshot index mismatch across scenarios. This MUST NOT happen.\n"
                f"Scenario[0]: n={len(ref_snaps)}, start={ref_snaps[0]}, end={ref_snaps[-1]}\n"
                f"Scenario[{i}]: n={len(n.snapshots)}, start={n.snapshots[0]}, end={n.snapshots[-1]}\n"
            )


# =============================================================================
# Load shedding representation (linear, robustly measurable)
# =============================================================================
def _ensure_load_shedding_generators(
    n: pypsa.Network,
    *,
    carrier: str = "load_shedding",
    marginal_cost: float = 1e4,
    p_nom: float = 1e9,
) -> List[str]:
    """
    Ensure a controllable positive dispatch option representing unmet demand.
    Implemented as one non-extendable Generator per bus.

    Assumptions:
    - Any dispatch from these units corresponds to load shedding.
    - Keeps the model linear and allows robust LS energy measurement.
    """
    existing: List[str] = []
    if hasattr(n, "generators") and "carrier" in n.generators.columns:
        existing = n.generators.index[n.generators.carrier.astype(str) == carrier].tolist()
    if existing:
        logger.info("Found %d existing load-shedding generators.", len(existing))
        return existing

    logger.info("Adding load-shedding generators (one per bus).")
    ls_names: List[str] = []
    for bus in n.buses.index:
        name = f"LS::{bus}"
        ls_names.append(name)
        n.add(
            "Generator",
            name,
            bus=bus,
            carrier=carrier,
            p_nom=p_nom,
            p_nom_extendable=False,
            marginal_cost=marginal_cost,
            efficiency=1.0,
            p_min_pu=0.0,
            p_max_pu=1.0,
        )

    # Make sure p_max_pu has these columns (if already present)
    try:
        pmax = n.generators_t.p_max_pu
        if isinstance(pmax, pd.DataFrame):
            missing = [g for g in ls_names if g not in pmax.columns]
            if missing:
                n.generators_t.p_max_pu = pmax.reindex(columns=list(pmax.columns) + missing, fill_value=1.0)
    except Exception:
        pass

    return ls_names


# =============================================================================
# Scenario stacking
# =============================================================================
def stack_scenarios_to_multisnapshot_network(
    scenario_files: Sequence[str],
    scenario_names: Sequence[str],
) -> Tuple[pypsa.Network, List[str]]:
    """
    Load prepared scenario networks and stack into one MultiIndex-snapshot network.

    Strict:
    - static assets identical
    - snapshots identical
    """
    if len(scenario_files) != len(scenario_names):
        raise ValueError("scenario_files and scenario_names must have equal length.")
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError(f"Duplicate scenario names: {scenario_names}")

    nets: List[pypsa.Network] = []
    for s, f in zip(scenario_names, scenario_files):
        logger.info("Loading scenario=%s network=%s", s, f)
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"Scenario network not found: {f}")
        n = pypsa.Network(str(p))
        if len(n.snapshots) == 0:
            raise ValueError(f"Scenario {s}: network has no snapshots.")
        if len(n.buses) == 0:
            raise ValueError(f"Scenario {s}: network has no buses.")
        nets.append(n)

    _assert_same_static_assets_and_snapshots(nets)

    ref = nets[0]
    base_snaps = ref.snapshots

    # Start with reference network as base container
    n = ref.copy()

    stacked_snaps = pd.MultiIndex.from_product(
        [list(scenario_names), list(base_snaps)],
        names=["scenario", "time"],
    )
    n.set_snapshots(stacked_snaps)

    # Snapshot weightings: repeat per scenario (objective weight typically hours)
    if hasattr(ref, "snapshot_weightings") and ref.snapshot_weightings is not None:
        w = ref.snapshot_weightings.copy().reindex(base_snaps)
        w_stacked = pd.concat([w] * len(scenario_names), keys=scenario_names, names=["scenario", "time"])
        n.snapshot_weightings = w_stacked
    else:
        n.snapshot_weightings = pd.DataFrame(
            index=stacked_snaps,
            data={"objective": 1.0, "generators": 1.0, "stores": 1.0},
        )

    # Stack *_t tables
    for t_container_name in [
        "buses_t",
        "generators_t",
        "loads_t",
        "links_t",
        "lines_t",
        "transformers_t",
        "storage_units_t",
        "stores_t",
    ]:
        if not hasattr(ref, t_container_name):
            continue
        ref_t = getattr(ref, t_container_name, None)
        if ref_t is None:
            continue

        frames = _list_time_dependent_frames(ref_t)
        if not frames:
            continue

        n_t = getattr(n, t_container_name)
        for attr in frames.keys():
            dfs = []
            for scen, net in zip(scenario_names, nets):
                net_t = getattr(net, t_container_name)
                df = getattr(net_t, attr)
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                # strict: snapshots equal, but keep explicit alignment
                df = df.reindex(base_snaps)
                dfs.append(df)

            df_stacked = pd.concat(dfs, keys=scenario_names, names=["scenario", "time"])
            try:
                setattr(n_t, attr, df_stacked)
            except Exception as e:
                logger.warning("Could not set %s.%s (skipping): %s", t_container_name, attr, e)

    # Debug outputs
    logger.info(
        "Stacked network: %d scenarios × %d timesteps = %d snapshots",
        len(scenario_names), len(base_snaps), len(n.snapshots)
    )
    logger.info("Static sizes: buses=%d gens=%d loads=%d links=%d lines=%d su=%d stores=%d",
                len(getattr(n, "buses", [])),
                len(getattr(n, "generators", [])),
                len(getattr(n, "loads", [])),
                len(getattr(n, "links", [])),
                len(getattr(n, "lines", [])),
                len(getattr(n, "storage_units", [])),
                len(getattr(n, "stores", [])))

    return n, list(scenario_names)


# =============================================================================
# Linopy / robust expressions
# =============================================================================
def _get_linopy_var(model, key_candidates: Sequence[str], *, strict: bool = False):
    for k in key_candidates:
        if k in model.variables:
            return model.variables[k], k
    if not strict:
        return None, None

    available = list(model.variables.keys())
    suggestions = []
    for candidate in key_candidates:
        suggestions.extend(difflib.get_close_matches(candidate, available, n=3, cutoff=0.6))
    if suggestions:
        raise KeyError(
            f"None of {list(key_candidates)} found in model.variables. "
            f"Did you mean: {sorted(set(suggestions))}? Available: {available}"
        )
    raise KeyError(f"None of {list(key_candidates)} found in model.variables. Available: {available}")


def _get_time_dimension(var) -> str:
    for name in ("snapshot", "snapshots", "time"):
        if name in getattr(var, "dims", ()):
            return name
    raise ValueError(f"No standard time dimension found in variable dims {getattr(var, 'dims', None)}.")


def _scenario_masks_from_snapshots(snapshots: pd.Index) -> Dict[str, np.ndarray]:
    if not isinstance(snapshots, pd.MultiIndex) or "scenario" not in snapshots.names:
        raise ValueError("Snapshots must be a MultiIndex with level 'scenario'.")
    scen = snapshots.get_level_values("scenario").astype(str)
    unique = scen.unique().tolist()
    return {s: (scen == s).to_numpy() for s in unique}


def _weights_objective_series(n: pypsa.Network) -> pd.Series:
    if hasattr(n, "snapshot_weightings") and n.snapshot_weightings is not None:
        if "objective" in n.snapshot_weightings.columns:
            w = n.snapshot_weightings["objective"].copy()
            w.index = n.snapshots
            return w
    return pd.Series(1.0, index=n.snapshots)


def _mask_to_isel_indices(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return np.flatnonzero(mask)


def _add_scenario_boundary_constraints(network: pypsa.Network, scenarios: Sequence[str], masks: Dict[str, np.ndarray]) -> None:
    """
    Prevent artificial cross-scenario coupling of intertemporal states (SOC, store energy)
    induced by stacking scenarios in one snapshot index.

    Default: cyclic boundary within each scenario
        state(s, first) == state(s, last)

    This is a modelling choice to make each scenario internally self-contained.
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
        logger.info("Boundary constraints added for StorageUnit SOC (var=%s).", su_name)

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
        logger.info("Boundary constraints added for Store energy (var=%s).", st_name)


def _build_ls_energy_expression(n: pypsa.Network, ls_generators: List[str], mask: np.ndarray):
    """
    LS energy (weighted) per scenario:
      sum_{t in scen} w_t * sum_{g in LS} p_g,t
    """
    m = n.model
    gen_p, gen_p_name = _get_linopy_var(m, ["Generator-p"], strict=True)

    try:
        available_gens = set(gen_p.coords["Generator"].values.tolist())
    except Exception as e:
        raise RuntimeError(f"Cannot access 'Generator' coordinate on {gen_p_name}: {e}") from e

    missing = set(ls_generators) - available_gens
    if missing:
        raise ValueError(f"Load shedding generators not found in model: {sorted(missing)}")

    gen_p_ls = gen_p.sel(Generator=ls_generators)
    w = _weights_objective_series(n)

    td = _get_time_dimension(gen_p_ls)
    idx = _mask_to_isel_indices(mask)

    gen_p_ls_s = gen_p_ls.isel({td: idx})
    w_s = w.to_numpy()[idx]

    import xarray as xr
    w_da = xr.DataArray(w_s, dims=[td], coords={td: gen_p_ls_s.coords[td]})
    return (gen_p_ls_s * w_da).sum()


def _build_investment_cost_expression(n: pypsa.Network):
    """
    Investment cost expression for shared extendable assets.
    (No scenario-specific recourse variables in this version.)
    """
    m = n.model
    expr = 0

    import xarray as xr

    # Generators
    if hasattr(n, "generators") and "capital_cost" in n.generators.columns:
        var, _ = _get_linopy_var(m, ["Generator-p_nom"], strict=False)
        if var is not None:
            cap_cost = n.generators["capital_cost"].reindex(n.generators.index).fillna(0.0)
            cc = xr.DataArray(cap_cost.to_numpy(), dims=["Generator"], coords={"Generator": n.generators.index})
            expr = expr + (var * cc).sum()

    # Links
    if hasattr(n, "links") and "capital_cost" in n.links.columns:
        var, _ = _get_linopy_var(m, ["Link-p_nom"], strict=False)
        if var is not None:
            cap_cost = n.links["capital_cost"].reindex(n.links.index).fillna(0.0)
            cc = xr.DataArray(cap_cost.to_numpy(), dims=["Link"], coords={"Link": n.links.index})
            expr = expr + (var * cc).sum()

    # StorageUnits
    if hasattr(n, "storage_units") and "capital_cost" in n.storage_units.columns:
        var, _ = _get_linopy_var(m, ["StorageUnit-p_nom"], strict=False)
        if var is not None:
            cap_cost = n.storage_units["capital_cost"].reindex(n.storage_units.index).fillna(0.0)
            cc = xr.DataArray(cap_cost.to_numpy(), dims=["StorageUnit"], coords={"StorageUnit": n.storage_units.index})
            expr = expr + (var * cc).sum()

    # Stores
    if hasattr(n, "stores") and "capital_cost" in n.stores.columns:
        var, _ = _get_linopy_var(m, ["Store-e_nom"], strict=False)
        if var is not None:
            cap_cost = n.stores["capital_cost"].reindex(n.stores.index).fillna(0.0)
            cc = xr.DataArray(cap_cost.to_numpy(), dims=["Store"], coords={"Store": n.stores.index})
            expr = expr + (var * cc).sum()

    return expr


def _build_operational_cost_expression(n: pypsa.Network, mask: np.ndarray):
    """
    Scenario-specific operational cost:
      sum_{t in scen} w_t * (dispatch vars * marginal costs)
    """
    m = n.model
    w = _weights_objective_series(n)

    import xarray as xr

    idx = _mask_to_isel_indices(mask)
    total = 0

    # Generators
    if hasattr(n, "generators") and "marginal_cost" in n.generators.columns:
        var, _ = _get_linopy_var(m, ["Generator-p"], strict=False)
        if var is not None:
            mc = n.generators["marginal_cost"].reindex(n.generators.index).fillna(0.0)
            mc_da = xr.DataArray(mc.to_numpy(), dims=["Generator"], coords={"Generator": n.generators.index})
            td = _get_time_dimension(var)
            var_s = var.isel({td: idx})
            w_s = w.to_numpy()[idx]
            w_da = xr.DataArray(w_s, dims=[td], coords={td: var_s.coords[td]})
            total = total + (var_s * mc_da * w_da).sum()

    # Links
    if hasattr(n, "links") and "marginal_cost" in n.links.columns:
        var, _ = _get_linopy_var(m, ["Link-p0", "Link-p"], strict=False)
        if var is not None:
            mc = n.links["marginal_cost"].reindex(n.links.index).fillna(0.0)
            mc_da = xr.DataArray(mc.to_numpy(), dims=["Link"], coords={"Link": n.links.index})
            td = _get_time_dimension(var)
            var_s = var.isel({td: idx})
            w_s = w.to_numpy()[idx]
            w_da = xr.DataArray(w_s, dims=[td], coords={td: var_s.coords[td]})
            total = total + (var_s * mc_da * w_da).sum()

    # StorageUnits
    if hasattr(n, "storage_units") and "marginal_cost" in n.storage_units.columns:
        var, _ = _get_linopy_var(m, ["StorageUnit-p_dispatch", "StorageUnit-p"], strict=False)
        if var is not None:
            mc = n.storage_units["marginal_cost"].reindex(n.storage_units.index).fillna(0.0)
            mc_da = xr.DataArray(mc.to_numpy(), dims=["StorageUnit"], coords={"StorageUnit": n.storage_units.index})
            td = _get_time_dimension(var)
            var_s = var.isel({td: idx})
            w_s = w.to_numpy()[idx]
            w_da = xr.DataArray(w_s, dims=[td], coords={td: var_s.coords[td]})
            total = total + (var_s * mc_da * w_da).sum()

    # Stores
    if hasattr(n, "stores") and "marginal_cost" in n.stores.columns:
        var, _ = _get_linopy_var(m, ["Store-p"], strict=False)
        if var is not None:
            mc = n.stores["marginal_cost"].reindex(n.stores.index).fillna(0.0)
            mc_da = xr.DataArray(mc.to_numpy(), dims=["Store"], coords={"Store": n.stores.index})
            td = _get_time_dimension(var)
            var_s = var.isel({td: idx})
            w_s = w.to_numpy()[idx]
            w_da = xr.DataArray(w_s, dims=[td], coords={td: var_s.coords[td]})
            total = total + (var_s * mc_da * w_da).sum()

    return total


def _get_solution_dict(n: pypsa.Network) -> Dict:
    m = getattr(n, "model", None)
    if m is None:
        raise RuntimeError("Network has no built model (n.model is None).")
    sol = getattr(m, "solution", None)
    if sol is None:
        raise RuntimeError("Model has no solution (n.model.solution is None). Optimisation likely failed.")
    return sol


def _extract_scalar_solution(sol: Dict, key: str) -> float:
    if key not in sol:
        raise KeyError(f"Solution missing variable '{key}'. Available keys (head): {list(sol.keys())[:50]}")
    v = sol[key]
    try:
        return float(getattr(v, "item", lambda: v)())
    except Exception:
        return float(np.asarray(v).item())


def _get_termination_status(n: pypsa.Network):
    m = getattr(n, "model", None)
    if m is None:
        return None
    for attr in ("status", "termination_condition", "termination", "result"):
        if hasattr(m, attr):
            return getattr(m, attr)
    return None


# =============================================================================
# Robust solve (lexicographic)
# =============================================================================
def solve_robust_lexicographic(
    n: pypsa.Network,
    *,
    solver_name: str,
    solver_options: Optional[Dict] = None,
    eps_ls: float = 1e-6,
    load_shedding_carrier: str = "load_shedding",
) -> Dict[str, float]:
    """
    Two-stage lexicographic solve:
      Stage 1: minimise worst-case LS energy
      Stage 2: minimise worst-case total cost subject to LS optimality
    """
    if solver_options is None:
        solver_options = {}

    ls_generators = _ensure_load_shedding_generators(n, carrier=load_shedding_carrier)

    masks = _scenario_masks_from_snapshots(n.snapshots)
    scenarios = list(masks.keys())
    logger.info("Robust optimisation over %d scenarios: %s", len(scenarios), scenarios)

    # Stage 1
    def extra_stage1(network: pypsa.Network, snapshots: pd.Index) -> None:
        m = network.model

        # formal fix: prevent cross-scenario intertemporal coupling
        _add_scenario_boundary_constraints(network, scenarios, masks)

        z_ls = m.add_variables(lower=0, name="z_ls")
        for s in scenarios:
            ls_energy = _build_ls_energy_expression(network, ls_generators, masks[s])
            m.add_constraints(z_ls >= ls_energy, name=f"robust_ls_epigraph::{s}")
        m.objective = z_ls

    logger.info("Stage 1: minimising worst-case load shedding energy ...")
    _ = n.optimize(
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=extra_stage1,
    )

    sol1 = _get_solution_dict(n)
    z_ls_star = _extract_scalar_solution(sol1, "z_ls")
    if z_ls_star < -1e-6:
        raise ValueError(f"Invalid negative load shedding: {z_ls_star}")
    if z_ls_star < 0:
        logger.warning("Negative z_ls* = %.2e (numerical); clamping to 0", z_ls_star)
        z_ls_star = 0.0
    logger.info("Stage 1 optimum worst-case LS energy: z_ls* = %.6g", z_ls_star)

    # Stage 2
    def extra_stage2(network: pypsa.Network, snapshots: pd.Index) -> None:
        m = network.model

        _add_scenario_boundary_constraints(network, scenarios, masks)

        z_ls = m.add_variables(lower=0, name="z_ls")
        z_cost = m.add_variables(lower=0, name="z_cost")

        for s in scenarios:
            ls_energy = _build_ls_energy_expression(network, ls_generators, masks[s])
            m.add_constraints(z_ls >= ls_energy, name=f"robust_ls_epigraph::{s}")

        m.add_constraints(z_ls <= (z_ls_star + eps_ls), name="robust_ls_fix")

        inv_cost = _build_investment_cost_expression(network)

        for s in scenarios:
            op_cost_s = _build_operational_cost_expression(network, masks[s])
            total_cost_s = inv_cost + op_cost_s
            m.add_constraints(z_cost >= total_cost_s, name=f"robust_cost_epigraph::{s}")

        m.objective = z_cost

    logger.info("Stage 2: minimising worst-case total cost given optimal LS ...")
    _ = n.optimize(
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=extra_stage2,
    )

    sol2 = _get_solution_dict(n)
    z_cost_star = _extract_scalar_solution(sol2, "z_cost")
    z_ls_final = _extract_scalar_solution(sol2, "z_ls")

    logger.info("Stage 2 optimum worst-case cost: z_cost* = %.6g", z_cost_star)
    logger.info("Stage 2 achieved worst-case LS: z_ls = %.6g (<= z_ls*+eps expected)", z_ls_final)

    slack = z_ls_final - z_ls_star
    if slack > 10 * eps_ls:
        logger.warning("Large slack in LS constraint: %.2e (eps=%.2e).", slack, eps_ls)

    diagnostics: Dict[str, float] = {
        "z_ls_star": float(z_ls_star),
        "z_ls_final": float(z_ls_final),
        "z_cost_star": float(z_cost_star),
        "n_scenarios": float(len(scenarios)),
        "ls_constraint_slack": float(slack),
    }

    term = _get_termination_status(n)
    if term is not None:
        diagnostics["termination_status"] = str(term)  # type: ignore[assignment]

    return diagnostics


def extract_capacities(n: pypsa.Network) -> Dict[str, Dict[str, float]]:
    """
    Extract extendable capacities after optimisation.
    Uses *_opt columns produced by PyPSA.
    """
    out: Dict[str, Dict[str, float]] = {}

    if hasattr(n, "generators") and "p_nom_opt" in n.generators.columns:
        ext = n.generators.index[n.generators.p_nom_extendable]
        out["generators"] = n.generators.loc[ext, "p_nom_opt"].dropna().to_dict()

    if hasattr(n, "links") and "p_nom_opt" in n.links.columns:
        ext = n.links.index[n.links.p_nom_extendable]
        out["links"] = n.links.loc[ext, "p_nom_opt"].dropna().to_dict()

    if hasattr(n, "storage_units") and "p_nom_opt" in n.storage_units.columns:
        ext = n.storage_units.index[n.storage_units.p_nom_extendable]
        out["storage_units"] = n.storage_units.loc[ext, "p_nom_opt"].dropna().to_dict()

    if hasattr(n, "stores") and "e_nom_opt" in n.stores.columns:
        ext = n.stores.index[n.stores.e_nom_extendable]
        out["stores"] = n.stores.loc[ext, "e_nom_opt"].dropna().to_dict()

    return out


def run_robust(
    *,
    cutouts: Sequence[str],
    scenario_network_template: str,
    out_network: str,
    out_summary_json: str,
    solver_name: str,
    solver_options: Optional[Dict],
    eps_ls: float,
) -> None:
    scenario_files = _resolve_scenario_networks_from_cutouts(cutouts, scenario_network_template)

    logger.info("=== Robust solve configuration ===")
    logger.info("Scenarios (cutouts): %d", len(cutouts))
    for c, f in zip(cutouts, scenario_files):
        logger.info("  - %-20s -> %s", c, f)

    # 1) stack scenario networks
    n, scen_names = stack_scenarios_to_multisnapshot_network(
        scenario_files=scenario_files,
        scenario_names=list(cutouts),
    )

    # 2) robust solve (shared-investment design)
    diag = solve_robust_lexicographic(
        n,
        solver_name=solver_name,
        solver_options=solver_options,
        eps_ls=eps_ls,
    )

    # 3) export
    capacities = extract_capacities(n)

    Path(out_network).parent.mkdir(parents=True, exist_ok=True)
    Path(out_summary_json).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting robust network to %s", out_network)
    n.export_to_netcdf(out_network)

    payload = {
        "scenario_names": list(scen_names),
        "scenario_networks": list(map(str, scenario_files)),
        "diagnostics": diag,
        "capacities": capacities,
        "assumptions": {
            "design_target": "single shared-investment 2050 portfolio robust across all cutouts",
            "static_assets_identical": True,
            "snapshots_identical": True,
            "scenario_boundary_constraints": "cyclic SOC and cyclic Store energy per scenario (if variables exist)",
            "load_shedding": "one high-cost Generator per bus, carrier=load_shedding",
        },
    }
    logger.info("Writing robust summary JSON to %s", out_summary_json)
    with open(out_summary_json, "w") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# CLI
# =============================================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shared-investment robust optimisation over cutout scenarios.")
    p.add_argument("--cutouts", nargs="+", required=True, help="Cutout scenario ids (discrete scenarios).")
    p.add_argument(
        "--scenario-network-template",
        required=True,
        help='Template to locate prepared scenario networks, e.g. "networks/prepared_{cutout}.nc".',
    )
    p.add_argument("--out-network", required=True, help="Output robust network .nc")
    p.add_argument("--out-summary-json", required=True, help="Output robust summary .json")
    p.add_argument("--solver-name", default="gurobi", help="Solver name (e.g. gurobi, highs, cbc)")
    p.add_argument("--solver-options-json", default=None, help="JSON string with solver options")
    p.add_argument("--eps-ls", type=float, default=1e-6, help="Tolerance to fix z_ls in stage 2")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    solver_options = None
    if args.solver_options_json:
        solver_options = json.loads(args.solver_options_json)

    run_robust(
        cutouts=args.cutouts,
        scenario_network_template=args.scenario_network_template,
        out_network=args.out_network,
        out_summary_json=args.out_summary_json,
        solver_name=args.solver_name,
        solver_options=solver_options,
        eps_ls=args.eps_ls,
    )


if __name__ == "__main__":
    main()
