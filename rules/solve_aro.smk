# SPDX-FileCopyrightText: Contributors to PyPSA-Eur
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

# =============================================================================
# ARO config
# =============================================================================
RUN_NAME = config["run"]["name"]
RESULTS_DIR = f"results/{RUN_NAME}/"

ARO = config.get("aro", {})  # optional: allow dedicated aro block
ROB = config.get("robust", {})  # fallback: reuse robust block

CUTOUTS = list(ARO.get("cutouts", ROB.get("cutouts", [])))
if not CUTOUTS:
    raise ValueError("ARO requires a non-empty cutout list in config['aro']['cutouts'] or config['robust']['cutouts'].")

PREPARED_TEMPLATE = ARO.get("prepared_template", ROB.get("prepared_template", "networks/prepared_{cutout}.nc"))

# Where ARO writes its final artefacts (inside classic run folder)
OUT_NETWORK = ARO.get("out_network", RESULTS_DIR + "networks/aro_robust.nc")
OUT_SUMMARY = ARO.get("out_summary", RESULTS_DIR + "results/aro_summary.json")

# ARO loop controls
INITIAL = list(ARO.get("initial_scenarios", [CUTOUTS[0]]))
MAX_ITER = int(ARO.get("max_iter", 5))
EPS_LS = float(ARO.get("eps_ls", 1e-6))

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


# =============================================================================
# Targets
# =============================================================================
rule aro:
    input:
        # ensure the scenario networks exist for all cutouts
        expand(PREPARED_TEMPLATE, cutout=CUTOUTS),
        OUT_NETWORK,
        OUT_SUMMARY


# =============================================================================
# ARO solve rule
# =============================================================================
rule solve_aro:
    """
    Methodical assumptions:
    - scripts/solve_aro.py runs an ARO loop that repeatedly calls scripts/solve_robust.py
      and evaluates candidate cutouts by dispatch-only solves using PyPSA's internal objective.
    - The produced OUT_NETWORK is a "solved" PyPSA network suitable for standard postprocess.
    """
    input:
        scenario_networks=expand(PREPARED_TEMPLATE, cutout=CUTOUTS),
    output:
        network=OUT_NETWORK,
        summary=OUT_SUMMARY,
    params:
        # IMPORTANT: params must be fully determined without wildcards -> use lambdas
        cutouts=lambda wc: " ".join(CUTOUTS),
        prepared_template=lambda wc: PREPARED_TEMPLATE,
        initial=lambda wc: " ".join(INITIAL),
        max_iter=lambda wc: str(MAX_ITER),
        solver=lambda wc: SOLVER_NAME,
        solver_opts=lambda wc: SOLVER_OPTIONS_JSON,
        eps_ls=lambda wc: str(EPS_LS),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.network})"
        mkdir -p "$(dirname {output.summary})"

        python scripts/solve_aro.py \
          --cutouts {params.cutouts} \
          --scenario-network-template "{params.prepared_template}" \
          --initial-scenarios {params.initial} \
          --max-iter {params.max_iter} \
          --out-network {output.network} \
          --out-summary-json {output.summary} \
          --solver-name {params.solver} \
          --solver-options-json '{params.solver_opts}' \
          --eps-ls {params.eps_ls}
        """


# =============================================================================
# Adapter: expose ARO result under canonical filename expected by postprocess
# =============================================================================
rule aro_as_canonical_solved_network:
    """
    Expose the ARO robust solution under the canonical solved-network path so that
    the standard PyPSA-Eur postprocessing rules are triggered unchanged.
    """
    input:
        aro=OUT_NETWORK
    output:
        canonical=RESULTS_DIR + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.canonical})"
        ln -sf "$(realpath {input.aro})" "{output.canonical}"
        """


# =============================================================================
# Target: run ARO + classical postprocess outputs
# =============================================================================
rule aro_postprocess:
    input:
        RESULTS_DIR + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        RESULTS_DIR + "csvs/individual/metrics_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        RESULTS_DIR + "csvs/costs.csv",
        RESULTS_DIR + "graphs/costs.svg",
        RESULTS_DIR + "graphs/energy.svg",
        RESULTS_DIR + "graphs/balances-energy.svg",