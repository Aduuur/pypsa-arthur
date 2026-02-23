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
RESOURCES_DIR = Path("resources") / RDIR

ARO = config.get("aro", {})  # optional: dedicated aro block
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
OUT_NETWORK_STD = Path(
    ARO.get("out_network_std", str(OUT_NETWORK).replace(".nc", "__std.nc"))
)
OUT_SUMMARY = Path(ARO.get("out_summary", str(RESULTS_DIR / "results" / "aro_summary.json")))

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
# Canonical solved network naming (GENERIC: supports elec-only and sector-coupled)
# -----------------------------------------------------------------------------
# You can override the canonical solved network path via:
#   aro:
#     canonical_solved: "results/<run>/networks/....nc"
#
CANONICAL_SOLVED_OVERRIDE = ARO.get("canonical_solved", None)
if CANONICAL_SOLVED_OVERRIDE is not None:
    CANONICAL_SOLVED = Path(str(CANONICAL_SOLVED_OVERRIDE))
else:
    # Heuristic: if sector_opts token is non-empty -> sector-coupled naming
    IS_SECTOR_RUN = bool(SECTOR_OPTS_TOKEN)

    if IS_SECTOR_RUN:
        # Sector-coupled canonical solved network (PyPSA-Eur style: allow empty tokens -> double underscores)
        CANONICAL_SOLVED = (
            RESULTS_DIR
            / "networks"
            / f"base_s_{CLUSTERS}_{OPTS_TOKEN}_{SECTOR_OPTS_TOKEN}_{PH_TOKEN}.nc"
        )
    else:
        # Elec-only canonical solved network (upstream style)
        if OPTS_TOKEN:
            CANONICAL_SOLVED = RESULTS_DIR / "networks" / f"base_s_{CLUSTERS}_elec_{OPTS_TOKEN}.nc"
        else:
            CANONICAL_SOLVED = RESULTS_DIR / "networks" / f"base_s_{CLUSTERS}_elec.nc"

ARO_POSTPROCESS_DONE = RESULTS_DIR / "postprocess" / "aro_postprocess.done"

# =============================================================================
# Postprocess targets (NO nested snakemake; pure file dependencies)
# =============================================================================
def _default_postprocess_targets() -> list[Path]:
    """
    Minimal, robust set of concrete outputs to force standard postprocess/plots.
    Keep it generic (works for elec-only and sector-coupled), and avoid targets that
    may not exist in some configs/forks.

    Extend via config['aro']['postprocess_targets'] (list[str]).
    """
    targets: list[Path] = []

    # Common, stable artefacts
    targets.append(RESOURCES_DIR / "maps" / "power-network.pdf")
    targets.append(RESOURCES_DIR / "maps" / f"power-network-s-{CLUSTERS}.pdf")

    # Metrics are usually produced by postprocess
    targets.append(
        RESULTS_DIR
        / "csvs"
        / "individual"
        / f"metrics_s_{CLUSTERS}_{OPTS_TOKEN}_{SECTOR_OPTS_TOKEN}_{PH_TOKEN}.csv"
    )

    return targets


_EXTRA = ARO.get("postprocess_targets", None)
if _EXTRA is not None:
    if not isinstance(_EXTRA, list) or not all(isinstance(x, str) for x in _EXTRA):
        raise ValueError("config['aro']['postprocess_targets'] must be a list of strings.")
    # Interpret as repo-root-relative paths (same as CLI targets)
    ARO_POSTPROCESS_TARGETS = [Path(x) for x in _EXTRA]
else:
    ARO_POSTPROCESS_TARGETS = _default_postprocess_targets()

# =============================================================================
# Targets
# =============================================================================
rule aro:
    input:
        expand(PREPARED_TEMPLATE, cutout=CUTOUTS),
        str(OUT_NETWORK),
        str(OUT_NETWORK_STD),
        str(OUT_SUMMARY),
        str(ARO_POSTPROCESS_DONE)

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
    output:
        network=str(OUT_NETWORK),
        summary=str(OUT_SUMMARY),
        std_network=str(OUT_NETWORK_STD),
    params:
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

        {sys.executable} scripts/solve_aro.py \
          --cutouts {params.cutouts} \
          --scenario-network-template "{params.prepared_template}" \
          --initial-scenarios {params.initial} \
          --max-iter {params.max_iter} \
          --out-network {output.network} \
          --out-std-network {output.std_network} \
          --out-summary-json {output.summary} \
          --solver-name {params.solver} \
          --solver-options-json '{params.solver_opts}' \
          --eps-ls {params.eps_ls}
        """

# =============================================================================
# ARO-only: Adapter + full postprocess targets
# =============================================================================
MODE = (config.get("workflow", {}) or {}).get("mode", "plain").lower()
if MODE not in {"plain", "robust", "aro"}:
    raise ValueError(f"Invalid workflow.mode={MODE!r} (expected plain|robust|aro)")

if MODE == "aro":

    rule aro_as_canonical_solved_network:
        """
        Expose OUT_NETWORK_STD under the canonical solved-network path so that
        the existing postprocess rules can be triggered unchanged.

         IMPORTANT (NFS/autofs + Snakemake mtime race fix):
         Do NOT use symlinks here. On some shared filesystems Snakemake may fail to
         stat() a symlink target during concurrent updates ("Unable to obtain modification time ...").
         Instead create a real file at the canonical path via hardlink/copy.
        """
        input:
            aro=str(OUT_NETWORK_STD)
        output:
            canonical=str(CANONICAL_SOLVED)
        shell:
            r"""
                    set -euo pipefail
                    mkdir -p "$(dirname {output.canonical})"

                    # If an old broken symlink exists, remove it (extra safety at runtime).
                    if [ -L "{output.canonical}" ] && [ ! -e "{output.canonical}" ]; then
                      rm -f "{output.canonical}"
                    fi

                    tmp="{output.canonical}.tmp.$$"
                    cp -f "{input.aro}" "$tmp"
                    mv -f "$tmp" "{output.canonical}"
                    """


    rule aro_full_postprocess:
        input:
            canonical=str(CANONICAL_SOLVED),
            targets=[str(p) for p in ARO_POSTPROCESS_TARGETS],
        output:
            done=str(ARO_POSTPROCESS_DONE)
        run:
            Path(output.done).parent.mkdir(parents=True, exist_ok=True)
            Path(output.done).write_text("ok\n")