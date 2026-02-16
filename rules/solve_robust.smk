# rules/solve_robust.smk
from __future__ import annotations

import os
import re
import json
import yaml
import shutil
import hashlib
import subprocess
from pathlib import Path


# -----------------------------------------------------------------------------
# Robust config
# -----------------------------------------------------------------------------
ROB = config.get("robust", {})
CUTOUTS = list(ROB.get("cutouts", []))

if not CUTOUTS:
    raise ValueError(
        "config['robust']['cutouts'] is empty. Provide at least one cutout for robust optimisation."
    )

PREPARED_TEMPLATE = ROB.get("prepared_template", "networks/prepared_{cutout}.nc")
OUT_NETWORK = ROB.get("out_network", "networks/robust.nc")
OUT_SUMMARY = ROB.get("out_summary", "results/robust_summary.json")
EPS_LS = float(ROB.get("eps_ls", 1e-6))

RUN_NAME = config["run"]["name"]



# -----------------------------------------------------------------------------
# Scenario singleton contract
# -----------------------------------------------------------------------------
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




# -----------------------------------------------------------------------------
# Solver settings (match your config: solving.solver.name + solving.solver.options)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _sanitize_for_run_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    return s.strip("-")


def _parent_run_name() -> str:
    # Assumption: results dir uses run.name (prefix not used in your observed paths)
    run = config.get("run", {})
    name = run.get("name", "run")
    prefix = run.get("prefix", "")
    return f"{prefix}{name}" if prefix else name


PARENT_RUN = _parent_run_name()


def _subrun_name_for_cutout(parent_run: str, cutout: str) -> str:
    # Deterministic and stable across machines:
    h = hashlib.sha1(cutout.encode("utf-8")).hexdigest()[:8]
    return f"{parent_run}__cutout__{_sanitize_for_run_name(cutout)}__{h}"


def _base_network_path_for_cutout(cutout: str) -> str:
    # PyPSA-Eur output convention:
    # results/<run.name>/networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc
    subrun = _subrun_name_for_cutout(PARENT_RUN, cutout)
    return f"results/{subrun}/networks/base_s_{CLUSTERS}_{OPTS}_{SECTOR_OPTS}_{PLANNING_HORIZON}.nc"


# -----------------------------------------------------------------------------
# Convenience meta-target
# -----------------------------------------------------------------------------
rule robust:
    input:
        OUT_NETWORK,
        OUT_SUMMARY


# -----------------------------------------------------------------------------
# Rule 1: build per-cutout base network (nested snakemake)
# IMPORTANT: output contains {cutout} so Snakemake can bind it.
# -----------------------------------------------------------------------------
rule build_base_network_per_cutout:
    output:
        base="results/robust_scenarios/{cutout}/base.nc",
    params:
        cutout=lambda wc: wc.cutout,
        subrun=lambda wc: _subrun_name_for_cutout(PARENT_RUN, wc.cutout),
        final_base=lambda wc: _base_network_path_for_cutout(wc.cutout),
    threads: 1
    resources:
        mem_mb=2000
    run:
        cutout = params.cutout
        subrun = params.subrun
        final_base = params.final_base

        # 1) write overlay config
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
            "snakemake",
            "-s", "Snakefile",
            "--cores", "8",  # <-- NEU (oder "all")
            "--scheduler", "greedy",  # optional, aber konsistent
            "--nolock",
            "--rerun-incomplete",
            "--keep-going",
            "--configfile", "config/config.yaml",
            "--configfile", str(overlay_path),
            "--",
            final_base,
        ]

        print("\n[robust] Building per-cutout base network via nested snakemake:")
        print("         cutout    :", cutout)
        print("         subrun    :", subrun)
        print("         target    :", final_base)
        print("         cmd       :", " ".join(cmd))

        subprocess.run(cmd, check=True)

        # 3) copy to a stable location that carries {cutout} for DAG stability
        Path(os.path.dirname(output.base)).mkdir(parents=True, exist_ok=True)

        print("\n[robust] Staging base network for robust DAG:")
        print("         src :", final_base)
        print("         dst :", output.base)

        shutil.copyfile(final_base, output.base)


# -----------------------------------------------------------------------------
# Rule 2: prepare scenario network for robust solver
# (In this minimal version: it's just the staged base network.)
# -----------------------------------------------------------------------------
rule prepared_network_for_robust:
    input:
        staged="results/robust_scenarios/{cutout}/base.nc",
    output:
        prepared=PREPARED_TEMPLATE,  # e.g. networks/prepared_{cutout}.nc
    run:
        Path(os.path.dirname(output.prepared)).mkdir(parents=True, exist_ok=True)

        print("\n[robust] Preparing scenario network for robust optimisation:")
        print("         cutout :", wildcards.cutout)
        print("         src    :", input.staged)
        print("         dst    :", output.prepared)

        shutil.copyfile(input.staged, output.prepared)


rule all_plus_robust:
    input:
        expand(
            "results/{run}/csvs/individual/metrics_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            run=RUN_NAME,
            clusters=CLUSTERS,
            opts=OPTS,
            sector_opts=SECTOR_OPTS,
            planning_horizons=PLANNING_HORIZON,
        ),
        OUT_NETWORK,
        OUT_SUMMARY

# -----------------------------------------------------------------------------
# Rule 3: robust solve using prepared scenario networks
# -----------------------------------------------------------------------------
rule solve_robust:
    input:
        scenario_networks=expand(PREPARED_TEMPLATE, cutout=CUTOUTS),
    output:
        network=OUT_NETWORK,
        summary=OUT_SUMMARY,
    params:
        solver=SOLVER_NAME,
        solver_opts=SOLVER_OPTIONS_JSON,
        eps_ls=EPS_LS,
        cutouts=" ".join(CUTOUTS),
        template=lambda wc: PREPARED_TEMPLATE.replace("{", "{{").replace("}", "}}"),
    shell:
        r"""
        set -euo pipefail

        echo "[robust] Scenario networks:"
        for f in {input.scenario_networks}; do
          echo "  - $f"
        done

        mkdir -p "$(dirname {output.network})"
        mkdir -p "$(dirname {output.summary})"

        python scripts/solve_robust.py \
          --cutouts {params.cutouts} \
          --scenario-network-template "{params.template}" \
          --out-network {output.network} \
          --out-summary-json {output.summary} \
          --solver-name {params.solver} \
          --solver-options-json '{params.solver_opts}' \
          --eps-ls {params.eps_ls}
        """

