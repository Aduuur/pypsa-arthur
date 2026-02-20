from __future__ import annotations

import json

ROB = config.get("robust", {})
CUTOUTS = list(ROB.get("cutouts", []))
if not CUTOUTS:
    raise ValueError("config['robust']['cutouts'] muss mindestens ein Cutout enthalten.")

PREPARED_TEMPLATE = ROB.get("prepared_template", "networks/prepared_{cutout}.nc")

# Zieldateien für das ARO-Endergebnis
ARO_OUT_NETWORK = ROB.get("aro_out_network", "networks/aro_robust.nc")
ARO_OUT_SUMMARY = ROB.get("aro_out_summary", "results/aro_summary.json")

# Solver-Konfiguration
SOLVING = config.get("solving", {})
SOLVER_BLOCK = SOLVING.get("solver", {})
SOLVER_NAME = SOLVER_BLOCK.get("name", "gurobi")
SOLVER_OPT_KEY = SOLVER_BLOCK.get("options")
ALL_SOLVER_OPTIONS = SOLVING.get("solver_options", {}) or {}
SOLVER_OPTIONS = (
    ALL_SOLVER_OPTIONS.get(SOLVER_OPT_KEY, {}) if SOLVER_OPT_KEY is not None else SOLVER_BLOCK.get("solver_options", {}) or {}
)
SOLVER_OPTIONS_JSON = json.dumps(SOLVER_OPTIONS)

ARO_CFG = ROB.get("aro", {})
INITIAL = ARO_CFG.get("initial_cutouts", CUTOUTS[:1])
MAX_ITER = int(ARO_CFG.get("max_iter", 5))


rule aro:
    """
    Iterative ARO-Optimierung über mehrere Cutouts.

    Diese Regel ruft ein separates Skript solve_aro.py auf, das die
    iterativen Schritte steuert: Robust-Lösung mit einem Anfangs-
    Szenariomenge, Bewertung der Kosten aller Cutouts, Identifikation des
    Worst‑Case‑Cutouts und Erweiterung des Szenariomenge. Der Prozess endet,
    wenn kein neues Worst‑Case hinzugefügt wird oder die maximale
    Iterationsanzahl erreicht ist.
    """
    input:
        # Basisnetze und vorbereitete Netzwerke für alle Cutouts müssen
        # bereits existieren. Diese Targets werden von solve_robust.smk
        # bereitgestellt.
        expand(PREPARED_TEMPLATE, cutout=CUTOUTS),
    output:
        network=ARO_OUT_NETWORK,
        summary=ARO_OUT_SUMMARY,
    params:
        solver=SOLVER_NAME,
        solver_opts=SOLVER_OPTIONS_JSON,
        prepared_template=PREPARED_TEMPLATE,
        cutouts=" ".join(CUTOUTS),
        initial=" ".join(INITIAL),
        max_iter=str(MAX_ITER),
    shell:
        r"""
        set -euo pipefail
        python scripts/solve_aro.py \
          --cutouts {params.cutouts} \
          --scenario-network-template "{params.prepared_template}" \
          --initial-scenarios {params.initial} \
          --max-iter {params.max_iter} \
          --out-network {output.network} \
          --out-summary-json {output.summary} \
          --solver-name {params.solver} \
          --solver-options-json '{params.solver_opts}'
        """