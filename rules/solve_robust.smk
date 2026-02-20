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
BASE_TARGET = ROB.get("base_target", "networks/base.nc")

FORESIGHT = config.get("foresight")
if FORESIGHT not in {"overnight", "myopic", "perfect"}:
    raise ValueError(
        "Invalid config['foresight'] value "
        f"{FORESIGHT!r}. Expected one of: overnight, myopic, perfect. "
        "A misspelling here prevents nested robust builds from finding the base-network rule."
    )

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
    run = config.get("run", {})
    name = run.get("name", "run")
    prefix = run.get("prefix", "")
    return f"{prefix}{name}" if prefix else name


PARENT_RUN = _parent_run_name()


def _subrun_name_for_cutout(parent_run: str, cutout: str) -> str:
    # Deterministic and stable across machines:
    h = hashlib.sha1(cutout.encode("utf-8")).hexdigest()[:8]
    return f"{parent_run}__cutout__{_sanitize_for_run_name(cutout)}__{h}"


def _nested_target_for_cutout(cutout: str) -> str:
    """
    Return the file path expected from the nested run for the base network artefact.

    NOTE: we DO NOT call Snakemake with this file path as a target!
          We call the RULE name (prepare_elec_networks) and then *locate* the produced file.
    """
    subrun = _subrun_name_for_cutout(PARENT_RUN, cutout)

    opts_tok = OPTS_TOKEN or str(OPTS).strip()
    sect_tok = SECTOR_OPTS_TOKEN  # '' if none

    # prepare_elec_networks output (from your repo): resources("networks/base_s_{clusters}_elec.nc")
    # Some variants might include opts; we cover in the candidates finder below.
    return f"resources/{subrun}/networks/base_s_{CLUSTERS}_elec.nc"


def _base_network_candidates_for_cutout(cutout: str) -> list[str]:
    """
    Candidate file names to locate the produced base network after nested run.

    We prefer resources/<subrun>/networks/ (unsolved build artefacts),
    and fall back to results/<subrun>/networks/ if needed.
    """
    subrun = _subrun_name_for_cutout(PARENT_RUN, cutout)

    opts_tok = OPTS_TOKEN or str(OPTS).strip()
    sect_tok = SECTOR_OPTS_TOKEN
    ph = str(PLANNING_HORIZON).strip()  # should be "2050"

    cands: list[str] = []

    # --- (1) resources/ (preferred) ---
    # from prepare_elec_networks:
    cands.append(f"resources/{subrun}/networks/base_s_{CLUSTERS}_elec.nc")

    # from prepare_network variants (repo-dependent):
    cands.append(f"resources/{subrun}/networks/base_s_{CLUSTERS}_elec_{opts_tok}.nc")
    if sect_tok:
        cands.append(f"resources/{subrun}/networks/base_s_{CLUSTERS}_elec_{opts_tok}_{sect_tok}.nc")

    # --- (2) results/ (fallback) ---
    # common solved base naming patterns:
    cands.append(f"results/{subrun}/networks/base_s_{CLUSTERS}_{opts_tok}_{ph}.nc")
    if sect_tok:
        cands.append(f"results/{subrun}/networks/base_s_{CLUSTERS}_{opts_tok}_{sect_tok}_{ph}.nc")

    # unique, preserve order
    return list(dict.fromkeys(cands))


# -----------------------------------------------------------------------------
# Convenience meta-target
# -----------------------------------------------------------------------------
rule robust:
    input:
        expand("results/robust_scenarios/{cutout}/base.nc", cutout=CUTOUTS),
        expand(PREPARED_TEMPLATE, cutout=CUTOUTS),
        OUT_NETWORK,
        OUT_SUMMARY


# -----------------------------------------------------------------------------
# Rule 1: build per-cutout base network (nested snakemake)
# IMPORTANT CHANGE:
# - We call the RULE name `prepare_elec_networks` as nested target (NOT a file target),
#   to avoid MissingRuleException and the snakemake fmt_iofile/is_storage crash you hit.
# - Afterwards, we locate the produced .nc in resources/<subrun>/networks (preferred)
#   and copy it into results/robust_scenarios/<cutout>/base.nc for the robust DAG.
# -----------------------------------------------------------------------------
rule build_base_network_per_cutout:
    output:
        base="results/robust_scenarios/{cutout}/base.nc",
    params:
        cutout=lambda wc: wc.cutout,
        subrun=lambda wc: _subrun_name_for_cutout(PARENT_RUN, wc.cutout),
        nested_target_rule="prepare_networks",
    threads: 1
    resources:
        mem_mb=2000
    run:
        cutout = params.cutout
        subrun = params.subrun
        nested_target_rule = params.nested_target_rule

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
            "/home/endata/PycharmProjects/pypsa-ee/.pixi/envs/default/bin/python3.12",
            "-m", "snakemake",
            "-s", "Snakefile",
            "--cores", "8",
            "--scheduler", "greedy",
            "--nolock",
            "--rerun-incomplete",
            "--keep-going",
            "--configfile", "config/config.yaml",
            "--configfile", str(overlay_path),
            "--until", "prepare_elec_networks",
        ]

        print("\n[robust] Building per-cutout base network via nested snakemake:")
        print("         cutout    :", cutout)
        print("         subrun    :", subrun)
        print("         target    :", nested_target_rule)
        print("         cmd       :", " ".join(cmd))

        subprocess.run(cmd, check=True)

        # 2) locate produced base network file
        # Prefer explicit candidates first (fast), then glob fallback.
        candidate_paths = [Path(p) for p in _base_network_candidates_for_cutout(cutout)]
        existing = [p for p in candidate_paths if p.exists()]

        if not existing:
            # glob fallback in both dirs
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

        # Take newest by mtime (defensive)
        final_base = max(existing, key=lambda p: p.stat().st_mtime)

        # 3) stage it for robust DAG
        Path(os.path.dirname(output.base)).mkdir(parents=True, exist_ok=True)

        print("\n[robust] Staging base network for robust DAG:")
        print("         src :", str(final_base))
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


# -----------------------------------------------------------------------------
# Optional: a combined target that keeps your "normal" pipeline outputs + robust
# -----------------------------------------------------------------------------
rule all_plus_robust:
    input:
        expand(
            "results/{run}/csvs/individual/metrics_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            run=RUN_NAME,
            clusters=CLUSTERS,
            opts=OPTS_TOKEN,
            sector_opts=SECTOR_OPTS_TOKEN,
            planning_horizons=PLANNING_HORIZON,
        ),
        OUT_NETWORK,
        OUT_SUMMARY


# -----------------------------------------------------------------------------
# Rule 3: robust solve using prepared scenario networks
# -----------------------------------------------------------------------------
rule solve_robust:
    input:
        # enforce Build/Staging of the base nets per cutout
        staged_bases=expand("results/robust_scenarios/{cutout}/base.nc", cutout=CUTOUTS),
        # and then the prepared inputs
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

        echo "[robust] Staged base networks:"
        for f in {input.staged_bases}; do
          echo "  - $f"
        done

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

