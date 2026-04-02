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


def _is_sector_run() -> bool:
    """
    Decide whether we are in a sector-coupled run based on the *actual config value*.
    Treat "", None, "none" (case-insensitive) as "no sector coupling".
    """
    v = str(SECTOR_OPTS).strip()
    return bool(v) and v.lower() != "none"


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


def _canonical_solved_network_path(wc) -> str:
    """
    Canonical solved-network path used by standard postprocess rules.

    - sector run: base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc
    - elec only : base_s_{clusters}_elec_{opts}.nc  (and if opts empty: base_s_{clusters}_elec.nc)
    """
    clusters = wc.clusters
    opts = wc.opts
    sector_opts = wc.sector_opts
    ph = wc.planning_horizons

    # sector-coupled
    if sector_opts and str(sector_opts).strip() and str(sector_opts).strip().lower() != "none":
        return RESULTS_DIR + f"networks/base_s_{clusters}_{opts}_{sector_opts}_{ph}.nc"

    # electricity-only
    if opts and str(opts).strip() and str(opts).strip().lower() != "none":
        return RESULTS_DIR + f"networks/base_s_{clusters}_elec_{opts}.nc"
    return RESULTS_DIR + f"networks/base_s_{clusters}_elec.nc"


def _nested_anchor_path(subrun: str) -> Path:
    """
    The wildcard-free anchor produced by the nested run.

    This file must be produced by a rule in the *root* Snakefile, e.g.
        rule robust_nested_anchor:
            output: results/<run>/networks/__robust_nested_anchor__.nc
    """
    return Path(f"results/{subrun}/networks/__robust_nested_anchor__.nc")


def _validate_network_is_nonempty(nc_path: Path) -> None:
    """Fail early if the produced/staged network is empty or invalid."""
    import pypsa  # type: ignore

    n = pypsa.Network(str(nc_path))
    if len(n.buses) == 0 or len(n.snapshots) == 0:
        raise ValueError(
            f"staged network looks empty: buses={len(n.buses)}, snapshots={len(n.snapshots)}"
        )


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
    input:
        config="config/config.yaml",
    output:
        base=RESULTS_DIR + "robust_scenarios/{cutout}/base.nc",
    params:
        cutout=lambda wc: wc.cutout,
        subrun=lambda wc: _subrun_name_for_cutout(PARENT_RUN, wc.cutout),
    threads: 1
    resources:
        mem_mb=2000
    run:
        cutout = params.cutout
        subrun = params.subrun

        # Construct overlay for nested run: override cutout + techfiles
        overlay = {
            "run": {"name": subrun},
            "atlite": {"default_cutout": cutout},
            "workflow": {"mode": "plain"},
            "scenario": config.get("scenario",{}),
            "renewable": {...},
        }

        # ------------------------------------------------------------------
        # Dynamic renewable techfile override (cutout-specific techfiles)
        # ------------------------------------------------------------------
        # Extract stress token and model prefix from cutout name
        # Cutout:   cutout_<model>_stress_NN_from_<date>.nc
        # Techfile: <prefix>_<model>_notAgg_pypsa__stress_NN.nc
        import re as _re
        _stress_match = _re.search(r"(stress_[0-9]+)", cutout)
        _model_match  = _re.search(r"cutout_(.+?)(?:[.]nc)?_stress_[0-9]+", cutout)
        if _stress_match and _model_match:
            _stress_token     = _stress_match.group(1)
            _model_token      = _model_match.group(1)
            _use_stress_naming = True
        else:
            _stress_token     = None
            _model_token      = cutout[7:] if cutout.startswith("cutout_") else cutout
            _use_stress_naming = False
        token = _model_token  # fallback compat

        techs = {
            "onwind": ("turbine", "wind"),
            "offwind-ac": ("turbine", "wind_offshore"),
            "offwind-dc": ("turbine", "wind_offshore"),
            "offwind-float": ("turbine", "wind_offshore"),
            "solar": ("panel", "pv"),
            "solar-hsat": ("panel", "pv"),
        }

        renewable_overlay = {}
        missing = []
        for tech, (res_key, prefix) in techs.items():
            try:
                orig_path = config["renewable"][tech]["resource"][res_key]
            except Exception:
                continue

            base_dir = os.path.dirname(orig_path)
            if _use_stress_naming:
                new_file = f"{prefix}_{_model_token}_notAgg_pypsa__{_stress_token}.nc"
                new_path = os.path.join("/home/endata/techfiles_manipulated/d", new_file)
            else:
                new_file = f"{prefix}_{token}_notAgg_pypsa.nc"
                new_path = os.path.join(base_dir, new_file)

            if not os.path.exists(new_path):
                missing.append((tech, res_key, orig_path, new_path))
                continue

            renewable_overlay.setdefault(tech, {}).setdefault("resource", {})[res_key] = new_path

        if missing:
            print("\n[robust] WARNING: Missing per-cutout techfiles; keeping original paths for those:")
            for tech, res_key, orig_path, new_path in missing:
                print(f"         - {tech}.{res_key}: expected {new_path}  (kept {orig_path})")

        if renewable_overlay:
            overlay["renewable"] = renewable_overlay

        # Temporal resolution: nested run muss konsistent zum Netzwerktyp sein
        is_temporal_sector = (
                _is_sector_run()
                or (PLANNING_HORIZON is not None and str(PLANNING_HORIZON).strip() not in ("", "None"))
        )

        if is_temporal_sector:
            overlay.setdefault("clustering",{}).setdefault("temporal",{})["resolution_elec"] = False
            if "resolution_sector" in config.get("clustering",{}).get("temporal",{}):
                overlay.setdefault("clustering",{}).setdefault("temporal",{})["resolution_sector"] = \
                    config["clustering"]["temporal"]["resolution_sector"]
        else:
            overlay.setdefault("clustering",{}).setdefault("temporal",{})["resolution_elec"] = \
                config.get("clustering",{}).get("temporal",{}).get("resolution_elec",False)

        overlay_dir = Path("resources") / "robust_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"overlay__{subrun}.yaml"
        with open(overlay_path, "w") as f:
            yaml.safe_dump(overlay, f, sort_keys=False)

        # ------------------------------------------------------------------
        # Nested snakemake invocation
        # ------------------------------------------------------------------
        # We target a wildcard-free anchor file to avoid:
        # - "Target rules may not contain wildcards"
        # - guessing filenames
        # - mtime/globbing roulette
        nested_target_file = Path(f"results/{subrun}/networks/base_s_{CLUSTERS}_{OPTS_TOKEN}_{SECTOR_OPTS_TOKEN}_{PLANNING_HORIZON}.nc")



        cmd = [
            sys.executable, "-m", "snakemake",
            "-s", "Snakefile",
            "--directory", ".",  # Expliziter Working Directory
            "--cores", "8",
            "--scheduler", "greedy",
            "--nolock",
            "--rerun-incomplete",
            "--keep-going",
            "--configfile", "config/config.yaml",
            "--configfile", str(overlay_path),
            "--config", "workflow={mode: plain} pypsa_nested=1",
            "--",
            str(nested_target_file),
        ]

        print("\n[robust] Building per-cutout base network via nested snakemake:")
        print("         cutout      :", cutout)
        print("         subrun      :", subrun)
        print("         sector_run  :", _is_sector_run())
        print("         target_file :", str(nested_target_file))
        print("         cmd         :", " ".join(cmd))

        nested_env = os.environ.copy()
        nested_env["PYPSA_ROBUST_NESTED"] = "1"

        for key in list(nested_env.keys()):
            if any(key.startswith(prefix) for prefix in [
                "SNAKEMAKE_", "__PYVENV_", "CONDA_", "MAMBA_"
            ]):
                del nested_env[key]

        subprocess.run(cmd, check=True, env=nested_env)

        if not nested_target_file.exists():
            raise FileNotFoundError(
                f"[robust] Nested run did not produce network:\n"
                f"  expected: {nested_target_file}\n"
            )

        _validate_network_is_nonempty(nested_target_file)

        Path(os.path.dirname(output.base)).mkdir(parents=True,exist_ok=True)
        shutil.copyfile(nested_target_file,output.base)

        # Validate (hard fail) to avoid silent "empty" networks
        try:
            _validate_network_is_nonempty(nested_target_file)
        except Exception as e:
            raise RuntimeError(
                f"[robust] Produced nested anchor network is invalid/empty.\n"
                f"  subrun: {subrun}\n"
                f"  file : {nested_target_file}\n"
                f"  error: {e}\n"
            )


# =============================================================================
# Rule 2: prepare scenario network for robust solver
# =============================================================================
rule prepared_network_for_robust:
    input:
        staged=RESULTS_DIR + "robust_scenarios/{cutout}/base.nc",
        config="config/config.yaml",
    output:
        prepared=PREPARED_TEMPLATE,
    run:
        # Snapshot-Validierung
        import pypsa

        n_check = pypsa.Network(input.staged)
        expected_start = config.get("snapshots",{}).get("start")
        if expected_start:
            actual_start = str(n_check.snapshots[0])[:10]
            if actual_start != expected_start:
                raise RuntimeError(
                    f"[robust] Snapshot mismatch!\n"
                    f"  config expects: {expected_start}\n"
                    f"  network has:    {actual_start}\n"
                    f"  Fix: rm -rf results/{RESULTS_DIR}robust_scenarios/ networks/prepared_*.nc"
                )

        Path(os.path.dirname(output.prepared)).mkdir(parents=True,exist_ok=True)
        print("\n[robust] Preparing scenario network for robust optimisation:")
        print("         cutout :",wildcards.cutout)
        print("         src    :",input.staged)
        print("         dst    :",output.prepared)
        shutil.copyfile(input.staged,output.prepared)




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
        template=lambda wc: PREPARED_TEMPLATE,
        co2_cost_mode= lambda wc: config.get("aro",{}).get("co2_cost_mode","off"),
        ls_penalty=lambda wc: config.get("aro",{}).get("ls_penalty",1e4),
        convergence_tol=lambda wc: config.get("aro",{}).get("convergence_tol",1e-4),
        dispatch_workers=lambda wc: config.get("aro",{}).get("dispatch_workers",0),
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
          --eps-ls {params.eps_ls} \
          --co2-cost-mode {params.co2_cost_mode} \
          --ls-penalty {params.ls_penalty} \
          --convergence-tol {params.convergence_tol} \
          --dispatch-workers {params.dispatch_workers}
        """


# =============================================================================
# Then wrap your existing robust adapter + robust_postprocess block like this:
# (i.e. indent the whole block under if MODE == "robust":)
# =============================================================================

if MODE == "robust":

    # =============================================================================
    # Adapter: expose robust result under canonical filenames expected by postprocess
    # =============================================================================

    rule robust_as_canonical_elec_noopts:
        input:
            robust=OUT_NETWORK
        output:
            canonical=RESULTS_DIR + "networks/base_s_{clusters}_elec.nc"
        shell:
            r"""
            set -euo pipefail
            mkdir -p "$(dirname {output.canonical})"
            cp -f "$(realpath {input.robust})" "{output.canonical}"
            """

    # Gleiche Anpassungen für robust_as_canonical_elec_withopts und robust_as_canonical_sector:
    rule robust_as_canonical_elec_withopts:
        input:
            robust=OUT_NETWORK
        output:
            canonical=RESULTS_DIR + "networks/base_s_{clusters}_elec_{opts}.nc"
        shell:
            r"""
            set -euo pipefail
            mkdir -p "$(dirname {output.canonical})"
            cp -f "$(realpath {input.robust})" "{output.canonical}"
            """

    rule robust_as_canonical_sector:
        input:
            robust=OUT_NETWORK
        output:
            canonical=RESULTS_DIR + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
        shell:
            r"""
            set -euo pipefail
            mkdir -p "$(dirname {output.canonical})"
            cp -f "$(realpath {input.robust})" "{output.canonical}"
            """

    # =============================================================================
    # Target: robust + classical postprocess outputs (as pinned artefacts)
    # =============================================================================
    def _canonical_inputs_for_postprocess(wc):
        if _is_sector_run():
            return [
                RESULTS_DIR
                + f"networks/base_s_{wc.clusters}_{wc.opts}_{wc.sector_opts}_{wc.planning_horizons}.nc"
            ]
        if wc.opts and str(wc.opts).strip() and str(wc.opts).strip().lower() != "none":
            return [RESULTS_DIR + f"networks/base_s_{wc.clusters}_elec_{wc.opts}.nc"]
        return [RESULTS_DIR + f"networks/base_s_{wc.clusters}_elec.nc"]

    rule robust_postprocess:
        input:
            _canonical_inputs_for_postprocess
        output:
            done=RESULTS_DIR + "postprocess/robust_postprocess.done"
        run:
            Path(os.path.dirname(output.done)).mkdir(parents=True, exist_ok=True)
            Path(output.done).write_text("ok\n")