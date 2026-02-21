# rules/solve_robust.smk
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import re
import json
import yaml
import sys
import shutil
import hashlib
import subprocess
from pathlib import Path


# =============================================================================
# Robust config
# =============================================================================
ROB = config.get("robust", {})
CUTOUTS = list(ROB.get("cutouts", []))

if not CUTOUTS:
    raise ValueError(
        "config['robust']['cutouts'] is empty. Provide at least one cutout for robust optimisation."
    )

PREPARED_TEMPLATE = ROB.get("prepared_template", "networks/prepared_{cutout}.nc")
EPS_LS = float(ROB.get("eps_ls", 1e-6))

RUN_NAME = config["run"]["name"]
FORESIGHT = config.get("foresight")
if FORESIGHT not in {"overnight", "myopic", "perfect"}:
    raise ValueError(
        f"Invalid config['foresight'] value {FORESIGHT!r}. Expected: overnight, myopic, perfect."
    )

# IMPORTANT: We write robust artefacts into the standard run folder to match the
# classical PyPSA-Eur results layout.
RESULTS_DIR = f"results/{RUN_NAME}/"
OUT_NETWORK = ROB.get("out_network", RESULTS_DIR + "networks/robust.nc")
OUT_SUMMARY = ROB.get("out_summary", RESULTS_DIR + "results/robust_summary.json")

# Scenario singleton contract (robust pipeline expects one "scenario" combination)
SC = config.get("scenario", {})

def _require_singleton(name: str, xs):
    if not isinstance(xs, list) or len(xs) != 1:
        raise ValueError(
            f"Robust workflow requires scenario.{name} to be a singleton list, got: {xs}. "
            f"Reason: we build one network per cutout and then stack them."
        )
    return xs[0]

CLUSTERS = _require_singleton("clusters", SC.get("clusters", []))
OPTS = _require_singleton("opts", SC.get("opts", []))
SECTOR_OPTS = _require_singleton("sector_opts", SC.get("sector_opts", []))
PLANNING_HORIZON = _require_singleton("planning_horizons", SC.get("planning_horizons", []))

def _normalise_wildcard_token(x) -> str:
    """Map config sentinel values to filename wildcard tokens."""
    if x is None:
        return ""
    token = str(x).strip()
    if token.lower() == "none":
        return ""
    return token

OPTS_TOKEN = _normalise_wildcard_token(OPTS)
SECTOR_OPTS_TOKEN = _normalise_wildcard_token(SECTOR_OPTS)

# =============================================================================
# Solver settings (match config: solving.solver.name + solving.solver.options)
# =============================================================================
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
# Helpers
# =============================================================================
def _sanitize_for_run_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    return s.strip("-")

def _parent_run_name() -> str:
    run = config.get("run", {})
    name = run.get("name", "run")
    prefix = run.get("prefix", "")
    return f"{prefix}{name}" if prefix else name

PARENT_RUN = _parent_run_name()

def _subrun_name_for_cutout(parent_run: str, cutout: str) -> str:
    h = hashlib.sha1(cutout.encode("utf-8")).hexdigest()[:8]
    return f"{parent_run}__cutout__{_sanitize_for_run_name(cutout)}__{h}"

def _base_network_candidates_for_cutout(cutout: str) -> list[str]:
    """
    Candidate file names to locate the produced base network after nested run.
    We prefer resources/<subrun>/networks/ (unsolved artefacts),
    and fall back to results/<subrun>/networks/ if needed.
    """
    subrun = _subrun_name_for_cutout(PARENT_RUN, cutout)
    opts_tok = OPTS_TOKEN or str(OPTS).strip()
    sect_tok = SECTOR_OPTS_TOKEN
    ph = str(PLANNING_HORIZON).strip()

    cands: list[str] = []
    cands.append(f"resources/{subrun}/networks/base_s_{CLUSTERS}_elec.nc")
    cands.append(f"resources/{subrun}/networks/base_s_{CLUSTERS}_elec_{opts_tok}.nc")
    if sect_tok:
        cands.append(f"resources/{subrun}/networks/base_s_{CLUSTERS}_elec_{opts_tok}_{sect_tok}.nc")
    cands.append(f"results/{subrun}/networks/base_s_{CLUSTERS}_{opts_tok}_{ph}.nc")
    if sect_tok:
        cands.append(f"results/{subrun}/networks/base_s_{CLUSTERS}_{opts_tok}_{sect_tok}_{ph}.nc")

    return list(dict.fromkeys(cands))


# =============================================================================
# Meta-targets
# =============================================================================
rule robust:
    input:
        expand(RESULTS_DIR + "robust_scenarios/{cutout}/base.nc", cutout=CUTOUTS),
        expand(PREPARED_TEMPLATE, cutout=CUTOUTS),
        OUT_NETWORK,
        OUT_SUMMARY


# =============================================================================
# Rule 1: build per-cutout base network (nested snakemake)
# =============================================================================
rule build_base_network_per_cutout:
    output:
        base=RESULTS_DIR + "robust_scenarios/{cutout}/base.nc",
    params:
        cutout=lambda wc: wc.cutout,
        subrun=lambda wc: _subrun_name_for_cutout(PARENT_RUN, wc.cutout),
        nested_target_rule="prepare_elec_networks",
    threads: 1
    resources:
        mem_mb=2000
    run:
        cutout = params.cutout
        subrun = params.subrun
        nested_target_rule = params.nested_target_rule

        overlay = {
            "run": {"name": subrun},
            "atlite": {"default_cutout": cutout},
        }

        overlay_dir = Path("resources") / "robust_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"overlay__{subrun}.yaml"
        with open(overlay_path, "w") as f:
            yaml.safe_dump(overlay, f, sort_keys=False)

        cmd = [
            sys.executable, "-m", "snakemake",
            "-s", "Snakefile",
            "--cores", "8",
            "--scheduler", "greedy",
            "--nolock",
            "--rerun-incomplete",
            "--keep-going",
            "--configfile", "config/config.yaml",
            "--configfile", str(overlay_path),
            "--until", nested_target_rule,
        ]

        print("\n[robust] Building per-cutout base network via nested snakemake:")
        print("         cutout    :", cutout)
        print("         subrun    :", subrun)
        print("         target    :", nested_target_rule)
        print("         cmd       :", " ".join(cmd))

        subprocess.run(cmd, check=True)

        candidate_paths = [Path(p) for p in _base_network_candidates_for_cutout(cutout)]
        existing = [p for p in candidate_paths if p.exists()]

        if not existing:
            cand_dirs = [
                Path("resources") / subrun / "networks",
                Path("results") / subrun / "networks",
            ]
            globbed = []
            for d in cand_dirs:
                if d.exists():
                    globbed += sorted(d.glob("base_s_*.nc"))
                    globbed += sorted(d.glob("base*_elec*.nc"))
                    globbed += sorted(d.glob("base*.nc"))
            existing = globbed

        if not existing:
            looked = "\n  - " + "\n  - ".join(_base_network_candidates_for_cutout(cutout))
            raise FileNotFoundError(
                f"[robust] Nested run produced no base network artefact.\n"
                f"Looked for candidates:{looked}\n"
                f"Also tried globbing in:\n"
                f"  - resources/{subrun}/networks/\n"
                f"  - results/{subrun}/networks/\n"
            )

        final_base = max(existing, key=lambda p: p.stat().st_mtime)

        Path(os.path.dirname(output.base)).mkdir(parents=True, exist_ok=True)

        print("\n[robust] Staging base network for robust DAG:")
        print("         src :", str(final_base))
        print("         dst :", output.base)

        shutil.copyfile(final_base, output.base)


# =============================================================================
# Rule 2: prepare scenario network for robust solver
# =============================================================================
rule prepared_network_for_robust:
    input:
        staged=RESULTS_DIR + "robust_scenarios/{cutout}/base.nc",
    output:
        prepared=PREPARED_TEMPLATE,
    run:
        Path(os.path.dirname(output.prepared)).mkdir(parents=True, exist_ok=True)

        print("\n[robust] Preparing scenario network for robust optimisation:")
        print("         cutout :", wildcards.cutout)
        print("         src    :", input.staged)
        print("         dst    :", output.prepared)

        shutil.copyfile(input.staged, output.prepared)


# =============================================================================
# Rule 3: robust solve using prepared scenario networks
# =============================================================================
rule solve_robust:
    input:
        staged_bases=expand(RESULTS_DIR + "robust_scenarios/{cutout}/base.nc", cutout=CUTOUTS),
        scenario_networks=expand(PREPARED_TEMPLATE, cutout=CUTOUTS),
    output:
        network=OUT_NETWORK,
        summary=OUT_SUMMARY,
    params:
        solver=SOLVER_NAME,
        solver_opts=SOLVER_OPTIONS_JSON,
        eps_ls=EPS_LS,
        cutouts=" ".join(CUTOUTS),
        template=lambda wc: PREPARED_TEMPLATE
    shell:
        r"""
        set -euo pipefail

        mkdir -p "$(dirname {output.network})"
        mkdir -p "$(dirname {output.summary})"

        echo "[robust] Staged base networks:"
        for f in {input.staged_bases}; do echo "  - $f"; done

        echo "[robust] Scenario networks:"
        for f in {input.scenario_networks}; do echo "  - $f"; done

        {sys.executable} scripts/solve_robust.py \
          --cutouts {params.cutouts} \
          --scenario-network-template "{params.template}" \
          --out-network {output.network} \
          --out-summary-json {output.summary} \
          --solver-name {params.solver} \
          --solver-options-json '{params.solver_opts}' \
          --eps-ls-abs {params.eps_ls}
        """


# =============================================================================
# Adapter: expose robust result under canonical filename expected by postprocess
# =============================================================================
rule robust_as_canonical_solved_network:
    """
    Methodical note:
    - Postprocessing is unchanged and reads a "solved network" from the canonical path.
    - We expose the robust solution under that name (symlink preferred).
    """
    input:
        robust=OUT_NETWORK
    output:
        canonical=RESULTS_DIR + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.canonical})"
        # Symlink (preferred). If your FS disallows it, replace with: cp -f
        ln -sf "$(realpath {input.robust})" "{output.canonical}"
        """


# =============================================================================
# Target: run robust + classical postprocess outputs
# =============================================================================
rule robust_postprocess:
    input:
        # canonical solved network
        RESULTS_DIR + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        # key classical artifacts to force the standard pipeline
        RESULTS_DIR + "csvs/individual/metrics_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        RESULTS_DIR + "csvs/costs.csv",
        RESULTS_DIR + "graphs/costs.svg",
        RESULTS_DIR + "graphs/energy.svg",
        RESULTS_DIR + "graphs/balances-energy.svg",
