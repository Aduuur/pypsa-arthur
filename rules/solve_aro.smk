# rules/solve_aro.smk
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts._helpers import get_rdir

# =============================================================================
# ARO config (paths MUST match Snakefile logic)
# =============================================================================
run = config["run"]
RDIR = str(get_rdir(run)).strip("/")  # IMPORTANT: same as Snakefile uses (and avoid trailing '/')

RESULTS_DIR = Path("results") / RDIR

ARO = config.get("aro", config.get("robust", {}).get("aro", {}))
ROB = config.get("robust", {})  # fallback: reuse robust block

CUTOUTS = list(ARO.get("cutouts", ROB.get("cutouts", [])))
if not CUTOUTS:
    raise ValueError(
        "ARO requires a non-empty cutout list in config['aro']['cutouts'] or config['robust']['cutouts']."
    )

PREPARED_TEMPLATE = ARO.get(
    "prepared_template",
    ROB.get("prepared_template", "networks/prepared_{cutout}.nc"),
)

# Where ARO writes its final artefacts (inside classic run folder)
OUT_NETWORK = Path(ARO.get("out_network", str(RESULTS_DIR / "networks" / "aro_robust.nc")))

# IMPORTANT: keep the exact filename used in rules/postprocess.smk
# (there, _select_postprocess_network prefers RESULTS + "networks/aro_robust__std.nc")
OUT_NETWORK_STD = Path(ARO.get("out_network_std", str(RESULTS_DIR / "networks" / "aro_robust__std.nc")))

# FIX: summary goes directly into RESULTS_DIR, not into RESULTS_DIR/results/
# (the old default caused a double-nesting: results/<RDIR>/results/aro_summary.json)
OUT_SUMMARY = Path(ARO.get("out_summary", str(RESULTS_DIR / "aro_summary.json")))

# ARO loop controls
INITIAL = list(ARO.get("initial_cutouts", ARO.get("initial_scenarios", [CUTOUTS[0]])))
MAX_ITER = int(ARO.get("max_iter", 5))
MAX_MASTER_SIZE = int(ARO.get("max_master_size", 3))


ARO_CO2_COST_MODE = str(ARO.get("co2_cost_mode", "off"))
ARO_LS_PENALTY = float(ARO.get("ls_penalty", 1e4))
ARO_CONVERGENCE_TOL = float(ARO.get("convergence_tol", 1e-4))
ARO_DISPATCH_WORKERS = int(ARO.get("dispatch_workers", 0))


# Solver passthrough (use same block as robust)
SOLVING = config.get("solving", {})
SOLVER_BLOCK = SOLVING.get("solver", {})
SOLVER_NAME = SOLVER_BLOCK.get("name", "gurobi")
SOLVER_OPT_KEY = SOLVER_BLOCK.get("options", None)
ALL_SOLVER_OPTIONS = SOLVING.get("solver_options", {}) or {}
if SOLVER_OPT_KEY is not None:
    SOLVER_OPTIONS = ALL_SOLVER_OPTIONS.get(SOLVER_OPT_KEY, {})
else:
    SOLVER_OPTIONS = SOLVER_BLOCK.get("solver_options", {}) or {}
SOLVER_OPTIONS_JSON = json.dumps(SOLVER_OPTIONS)

# ARO-spezifische solver_options überschreiben den globalen Block wenn vorhanden.
# Konfiguriert unter config["robust"]["aro"]["solver_options"] — ermöglicht
# separate Gurobi-Optionen für den ARO-Master ohne den normalen Solve zu beeinflussen.
ARO_SOLVER_OPTIONS = ARO.get("solver_options", None)
if ARO_SOLVER_OPTIONS:
    SOLVER_OPTIONS = dict(ARO_SOLVER_OPTIONS)
    SOLVER_OPTIONS_JSON = json.dumps(SOLVER_OPTIONS)

# =============================================================================
# Scenario singleton contract (bind wildcards to concrete filenames)
# =============================================================================
SC = config.get("scenario", {})


def _require_singleton(name: str, xs):
    if not isinstance(xs, list) or len(xs) != 1:
        raise ValueError(
            f"ARO workflow requires scenario.{name} to be a singleton list, got: {xs}. "
            "Reason: we must bind postprocess filenames deterministically."
        )
    return xs[0]


def _tok(x) -> str:
    """Filename token normalization: map None/'none' -> ''."""
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() == "none" else s


CLUSTERS = _require_singleton("clusters", SC.get("clusters", []))
OPTS = _require_singleton("opts", SC.get("opts", []))
SECTOR_OPTS = _require_singleton("sector_opts", SC.get("sector_opts", []))
PLANNING_HORIZON = _require_singleton("planning_horizons", SC.get("planning_horizons", []))

OPTS_TOKEN = _tok(OPTS)
SECTOR_OPTS_TOKEN = _tok(SECTOR_OPTS)
PH_TOKEN = _tok(PLANNING_HORIZON)

# -----------------------------------------------------------------------------
# Canonical solved network naming (for compatibility with existing postprocess rules)
# -----------------------------------------------------------------------------
# You can override the canonical solved network path via:
#   aro:
#     canonical_solved: "results/<run>/networks/....nc"
#
CANONICAL_SOLVED_OVERRIDE = ARO.get("canonical_solved", None)
if CANONICAL_SOLVED_OVERRIDE is not None:
    CANONICAL_SOLVED = Path(str(CANONICAL_SOLVED_OVERRIDE))
else:
    IS_SECTOR_RUN = bool(SECTOR_OPTS_TOKEN)
    if IS_SECTOR_RUN:
        CANONICAL_SOLVED = (
                RESULTS_DIR
                / "networks"
                / f"base_s_{CLUSTERS}_{OPTS_TOKEN}_{SECTOR_OPTS_TOKEN}_{PH_TOKEN}.nc"
        )
    else:
        # Myopic run: kein sector_opts aber planning_horizon vorhanden
        if PH_TOKEN:
            CANONICAL_SOLVED = (
                    RESULTS_DIR
                    / "networks"
                    / f"base_s_{CLUSTERS}_{OPTS_TOKEN}__{PH_TOKEN}.nc"
            )
        elif OPTS_TOKEN:
            CANONICAL_SOLVED = RESULTS_DIR / "networks" / f"base_s_{CLUSTERS}_elec_{OPTS_TOKEN}.nc"
        else:
            CANONICAL_SOLVED = RESULTS_DIR / "networks" / f"base_s_{CLUSTERS}_elec.nc"


# =============================================================================
# Helper: remove broken symlink (prevents Snakemake mtime errors during DAG build)
# =============================================================================
def _unlink_if_broken_symlink(p: Path) -> None:
    try:
        if p.is_symlink() and not p.exists():
            p.unlink()
    except Exception:
        pass


_unlink_if_broken_symlink(CANONICAL_SOLVED)

# =============================================================================
# Targets
# =============================================================================
rule aro:
    input:
        expand(PREPARED_TEMPLATE, cutout=CUTOUTS),
        str(OUT_NETWORK),
        str(OUT_NETWORK_STD),
        str(OUT_SUMMARY),
        str(CANONICAL_SOLVED)


# =============================================================================
# ARO solve rule
# =============================================================================
rule solve_aro:
    """
    scripts/solve_aro.py must:
      - write OUT_NETWORK (final robust network),
      - write OUT_NETWORK_STD (single-network artefact suitable for postprocess),
      - write OUT_SUMMARY.
    """
    input:
        scenario_networks=expand(PREPARED_TEMPLATE, cutout=CUTOUTS),
        config="config/config.yaml",
    output:
        network=str(OUT_NETWORK),
        summary=str(OUT_SUMMARY),
        std_network=str(OUT_NETWORK_STD),
        dispatch_dir=directory(str(RESULTS_DIR / "networks" / "dispatch")),
    log:
        str(RESULTS_DIR / "logs" / "solve_aro.log"),
    params:
        cutouts=lambda wc: " ".join(CUTOUTS),
        prepared_template=lambda wc: PREPARED_TEMPLATE,
        initial=lambda wc: " ".join(INITIAL),
        max_iter=lambda wc: str(MAX_ITER),
        solver=lambda wc: SOLVER_NAME,
        solver_opts=lambda wc: SOLVER_OPTIONS_JSON,
        co2_cost_mode= lambda wc: ARO_CO2_COST_MODE,
        ls_penalty=lambda wc: ARO_LS_PENALTY,
        convergence_tol=lambda wc: ARO_CONVERGENCE_TOL,
        dispatch_workers=lambda wc: ARO_DISPATCH_WORKERS,
        dispatch_solver_opts=lambda wc: json.dumps(ARO.get('dispatch_solver_options', {})) if ARO.get('dispatch_solver_options') else SOLVER_OPTIONS_JSON,
        max_master_size= lambda wc: MAX_MASTER_SIZE,

    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.network})"
        mkdir -p "$(dirname {output.std_network})"
        mkdir -p "$(dirname {output.summary})"

        {sys.executable} scripts/solve_aro.py \\
          --cutouts {params.cutouts} \\
          --scenario-network-template "{params.prepared_template}" \\
          --initial-scenarios {params.initial} \\
          --max-iter {params.max_iter} \\
          --out-network {output.network} \\
          --out-std-network {output.std_network} \\
          --out-summary-json {output.summary} \\
          --solver-name {params.solver} \\
          --solver-options-json '{params.solver_opts}' \\
          --co2-cost-mode {params.co2_cost_mode} \\
          --ls-penalty {params.ls_penalty} \\
          --convergence-tol {params.convergence_tol} \\
          --dispatch-workers {params.dispatch_workers} \\
          --dispatch-solver-options-json '{params.dispatch_solver_opts}' \\
          --max-master-size {params.max_master_size} \\
          --out-dispatch-dir {output.dispatch_dir} \\
           2>&1 | tee {log}  
        """


# =============================================================================
# ARO-only: Adapter rule for canonical postprocess filename
# =============================================================================
MODE = "plain" if _IS_NESTED_SUBRUN else (config.get("workflow", {}) or {}).get("mode", "plain").lower()
if MODE not in {"plain", "robust", "aro"}:
    raise ValueError(f"Invalid workflow.mode={MODE!r} (expected plain|robust|aro)")

if MODE == "aro":

    rule aro_as_canonical_solved_network:
        """
        Expose OUT_NETWORK_STD under the canonical solved-network path.

        Do NOT symlink: broken symlinks are the root cause of Snakemake mtime errors on NFS-like FS.
        We copy to guarantee a real file exists at the canonical path.
        """
        input:
            aro=str(OUT_NETWORK_STD)
        output:
            canonical=str(CANONICAL_SOLVED)
        shell:
            r"""
            set -euo pipefail
            mkdir -p "$(dirname {output.canonical})"

            # remove broken symlink if any (extra safety at runtime)
            if [ -L "{output.canonical}" ] && [ ! -e "{output.canonical}" ]; then
              rm -f "{output.canonical}"
            fi

            cp -f "{input.aro}" "{output.canonical}"
            """
