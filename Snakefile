# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

from pathlib import Path
import yaml
from os.path import normpath, exists, join
from shutil import copyfile, move, rmtree
from snakemake.utils import min_version
from itertools import product

min_version("8.11")

from scripts._helpers import (
    path_provider,
    get_scenarios,
    get_rdir,
    get_shadow,
)


configfile: "config/config.yaml"
configfile: "config/plotting.default.yaml"


if Path("config/config.yaml").exists():

    configfile: "config/config.yaml"

# Auto-load overlay for nested subruns — ensures ALL sub-job levels get it via configfile: propagation
_current_nested_file = Path("resources/robust_overlays/.current_nested_run")
if _current_nested_file.exists():
    try:
        _nested_run_name = _current_nested_file.read_text().strip()
        _nested_overlay_path = (Path("resources/robust_overlays") / f"overlay__{_nested_run_name}.yaml").resolve()
        if _nested_overlay_path.exists():
            configfile: str(_nested_overlay_path)
    except Exception:
        pass

import sys
import os
_IS_NESTED_SUBRUN = "__cutout__" in config.get("run", {}).get("name", "")
MODE = "plain" if _IS_NESTED_SUBRUN else (config.get("workflow", {}) or {}).get("mode", "plain").lower()

if MODE not in {"plain", "robust", "aro"}:
    raise ValueError(f"Invalid workflow.mode={MODE!r} (expected plain|robust|aro)")





run = config["run"]
scenarios = get_scenarios(run)
RDIR = get_rdir(run)
shadow_config = get_shadow(run)

RESULTS = f"results/{RDIR}/"
# (optional) normalize double slashes
RESULTS = RESULTS.replace("//", "/")


shared_resources = run["shared_resources"]["policy"]
exclude_from_shared = run["shared_resources"]["exclude"]
logs = path_provider("logs/", RDIR, shared_resources, exclude_from_shared)
benchmarks = path_provider("benchmarks/", RDIR, shared_resources, exclude_from_shared)
resources = path_provider("resources/", RDIR, shared_resources, exclude_from_shared)


PLAIN_DONE = RESULTS + "postprocess/__plain_workflow.done"
ROBUST_DONE = RESULTS + "postprocess/__robust_workflow.done"
ARO_DONE  = RESULTS + "postprocess/__aro_workflow.done"



localrules:
    purge,


wildcard_constraints:
    clusters="[0-9]+(m|c)?|all|adm",
    opts=r"[-+a-zA-Z0-9\.]*",
    sector_opts=r"[-+a-zA-Z0-9\.\s]*",
    planning_horizons=r"[0-9]{4}",


include: "rules/common.smk"
include: "rules/collect.smk"
include: "rules/retrieve.smk"
include: "rules/build_electricity.smk"
include: "rules/build_sector.smk"
include: "rules/solve_electricity.smk"
include: "rules/postprocess.smk"
include: "rules/development.smk"
include: "rules/solve_robust.smk"
include: "rules/solve_aro.smk"


if config["foresight"] == "overnight":

    include: "rules/solve_overnight.smk"


if config["foresight"] == "myopic":

    include: "rules/solve_myopic.smk"


if config["foresight"] == "perfect":

    include: "rules/solve_perfect.smk"

# Disambiguate canonical solved-network outputs in robust/aro mode.
# Otherwise, classical solve_* rules (myopic/perfect) compete with adapter rules for the same filename.
if MODE == "aro":
    ruleorder:
        aro_as_canonical_solved_network > solve_sector_network_myopic

if MODE == "robust":
    ruleorder:
        robust_as_canonical_sector > solve_sector_network_myopic

# -----------------------------------------------------------------------------
# Robust nested anchor (wildcard-free target file)
# -----------------------------------------------------------------------------
from pathlib import Path

def _norm(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() == "none" else s

def _get_singleton_or_none(name: str):
    sc = config.get("scenario", {})
    xs = sc.get(name, None)
    if xs is None:
        return None
    if not isinstance(xs, list):
        return xs
    if len(xs) == 1:
        return xs[0]
    # multi-scenario → cannot derive deterministic wildcard-free paths
    return None

def _results_dir() -> Path:
    # Use RDIR as computed by get_rdir(run). Protect against trailing slashes.
    return Path("results") / Path(RDIR)

def _anchor_path() -> str:
    return str(_results_dir() / "networks" / "__robust_nested_anchor__.nc")

def _anchor_path_input() -> str:
    """
    Return the first candidate source path — used as input dependency
    so Snakemake waits for the solved network before running robust_nested_anchor.
    """
    clusters = _get_singleton_or_none("clusters")
    opts = _norm(_get_singleton_or_none("opts"))
    sector_opts = _norm(_get_singleton_or_none("sector_opts"))
    ph = _norm(_get_singleton_or_none("planning_horizons"))
    # Primary candidate — matches your repo's actual output name
    return str(_results_dir() / "networks" / f"base_s_{clusters}_{opts}_{sector_opts}_{ph}.nc")

def _candidate_sources() -> list[Path]:
    clusters = _get_singleton_or_none("clusters")
    opts = _norm(_get_singleton_or_none("opts"))
    sector_opts = _norm(_get_singleton_or_none("sector_opts"))
    ph = _norm(_get_singleton_or_none("planning_horizons"))

    if clusters is None or ph == "":
        return []

    base = _results_dir() / "networks"
    cands: list[Path] = []

    # --- Your repo seems to produce this even in electricity-only:
    # base_s_<clusters>_<opts>_<sector_opts>_<planning_horizons>.nc
    # When opts="" and sector_opts="" → base_s_24___2050.nc
    cands.append(base / f"base_s_{clusters}_{opts}_{sector_opts}_{ph}.nc")

    # --- Other common variants (keep as fallback)
    cands.append(base / "prepared.nc")
    cands.append(base / f"prepared_{run['name']}.nc")  # harmless fallback

    # Electricity naming variants (some forks)
    if opts:
        cands.append(base / f"base_s_{clusters}_elec_{opts}.nc")
        cands.append(base / f"elec_s_{clusters}_{opts}.nc")
    cands.append(base / f"base_s_{clusters}_elec.nc")
    cands.append(base / f"elec_s_{clusters}.nc")

    # Sector naming variants (some forks)
    if sector_opts:
        cands.append(base / f"sector_s_{clusters}_{opts}_{sector_opts}_{ph}.nc")


    # de-dup preserve order
    seen = set()
    uniq = []
    for p in cands:
        ps = str(p)
        if ps in seen:
            continue
        seen.add(ps)
        uniq.append(p)
    return uniq

rule robust_nested_anchor:
    """
    Wildcard-free target for nested snakemake calls.
    Creates: results/<RDIR>/networks/__robust_nested_anchor__.nc
    """

    input:
        source=_anchor_path_input()
    output:
        anchor=_anchor_path()
    run:
        from pathlib import Path

        anchor = Path(output.anchor)
        src = Path(input.source)

        if not src.exists():
            raise FileNotFoundError(f"Source network not found: {src}")

        anchor.parent.mkdir(parents=True,exist_ok=True)
        if anchor.exists() or anchor.is_symlink():
            anchor.unlink()

        # Symlink oder copy
        try:
            anchor.symlink_to(src.resolve())
        except Exception:
            import shutil

            shutil.copyfile(src,anchor)


rule plain_all:
    input:
        expand(RESULTS + "graphs/costs.svg", run=config["run"]["name"]),
        expand(resources("maps/power-network.pdf"), run=config["run"]["name"]),
        expand(
            resources("maps/power-network-s-{clusters}.pdf"),
            run=config["run"]["name"],
            **config["scenario"],
        ),
        expand(
            RESULTS
            + "maps/base_s_{clusters}_{opts}_{sector_opts}-costs-all_{planning_horizons}.pdf",
            run=config["run"]["name"],
            **config["scenario"],
        ),
        # COP profiles plots
        expand(
            RESULTS + "graphs/cop_profiles_s_{clusters}_{planning_horizons}.html",
            run=config["run"]["name"],
            **config["scenario"],
        ),
        lambda w: expand(
            (
                RESULTS
                + "maps/base_s_{clusters}_{opts}_{sector_opts}-h2_network_{planning_horizons}.pdf"
                if config_provider("sector", "H2_network")(w)
                else []
            ),
            run=config["run"]["name"],
            **config["scenario"],
        ),
        lambda w: expand(
            (
                RESULTS
                + "maps/base_s_{clusters}_{opts}_{sector_opts}-ch4_network_{planning_horizons}.pdf"
                if config_provider("sector", "gas_network")(w)
                else []
            ),
            run=config["run"]["name"],
            **config["scenario"],
        ),
        lambda w: expand(
            (
                RESULTS + "csvs/cumulative_costs.csv"
                if config_provider("foresight")(w) == "myopic"
                else []
            ),
            run=config["run"]["name"],
        ),
        lambda w: expand(
            (
                RESULTS
                + "maps/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}-balance_map_{carrier}.pdf"
            ),
            **config["scenario"],
            run=config["run"]["name"],
            carrier=config_provider("plotting", "balance_map", "bus_carriers")(w),
        ),
        expand(
            RESULTS
            + "graphics/balance_timeseries/s_{clusters}_{opts}_{sector_opts}_{planning_horizons}",
            run=config["run"]["name"],
            **config["scenario"],
        ),
        expand(
            RESULTS
            + "graphics/heatmap_timeseries/s_{clusters}_{opts}_{sector_opts}_{planning_horizons}",
            run=config["run"]["name"],
            **config["scenario"],
        ),
        # Explicitly list heat source types for temperature maps
        lambda w: expand(
            (
                RESULTS
                + "maps/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}-heat_source_temperature_map_river_water.html"
                if config_provider("plotting", "enable_heat_source_maps")(w)
                and "river_water"
                in config_provider("sector", "heat_pump_sources", "urban central")(w)
                else []
            ),
            **config["scenario"],
            run=config["run"]["name"],
        ),
        lambda w: expand(
            (
                RESULTS
                + "maps/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}-heat_source_temperature_map_sea_water.html"
                if config_provider("plotting", "enable_heat_source_maps")(w)
                and "sea_water"
                in config_provider("sector", "heat_pump_sources", "urban central")(w)
                else []
            ),
            **config["scenario"],
            run=config["run"]["name"],
        ),
        lambda w: expand(
            (
                RESULTS
                + "maps/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}-heat_source_temperature_map_ambient_air.html"
                if config_provider("plotting", "enable_heat_source_maps")(w)
                and "air"
                in config_provider("sector", "heat_pump_sources", "urban central")(w)
                else []
            ),
            **config["scenario"],
            run=config["run"]["name"],
        ),
        # Only river_water has energy maps
        lambda w: expand(
            (
                RESULTS
                + "maps/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}-heat_source_energy_map_river_water.html"
                if config_provider("plotting", "enable_heat_source_maps")(w)
                and "river_water"
                in config_provider("sector", "heat_pump_sources", "urban central")(w)
                else []
            ),
            **config["scenario"],
            run=config["run"]["name"],
        ),
        expand(
            RESULTS
            + "graphics/balance_timeseries/s_{clusters}_{opts}_{sector_opts}_{planning_horizons}",
            run=config["run"]["name"],
            **config["scenario"],
        ),
        expand(
            RESULTS
            + "graphics/heatmap_timeseries/s_{clusters}_{opts}_{sector_opts}_{planning_horizons}",
            run=config["run"]["name"],
            **config["scenario"],
        ),
        expand(
            RESULTS
            + "graphics/interactive_bus_balance/s_{clusters}_{opts}_{sector_opts}_{planning_horizons}",
            run=config["run"]["name"],
            **config["scenario"],
        ),
    default_target: True


rule create_scenarios:
    output:
        config["run"]["scenarios"]["file"],
    script:
        "config/create_scenarios.py"


rule purge:
    run:
        import builtins

        do_purge = builtins.input(
            "Do you really want to delete all generated files?\n"
            "\t* resources\n"
            "\t* results\n"
            "\t* docs\n"
            "Downloaded files are kept.\n"
            "Delete all files in the folders above? [y/N] "
        )
        if do_purge == "y":

            # Remove the directories and recreate them with .gitkeep
            for dir_path in ["resources/", "results/"]:
                rmtree(dir_path, ignore_errors=True)
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                (Path(dir_path) / ".gitkeep").touch()

            rmtree("doc/_build", ignore_errors=True)
            print(
                "Purging all generated resources, results and docs. Downloads are kept."
            )
        else:
            raise Exception(f"Input {do_purge}. Aborting purge.")


rule dump_graph_config:
    """Dump the current Snakemake configuration to a YAML file for graph generation."""
    output:
        config_file=temp(resources("dag_final_config.yaml")),
    run:
        import yaml

        with open(output.config_file, "w") as f:
            yaml.dump(config, f)


rule rulegraph:
    """Generates Rule DAG in DOT, PDF, PNG, and SVG formats using the final configuration."""
    message:
        "Creating RULEGRAPH dag in multiple formats using the final configuration."
    input:
        config_file=rules.dump_graph_config.output.config_file,
    output:
        dot=resources("dag_rulegraph.dot"),
        pdf=resources("dag_rulegraph.pdf"),
        png=resources("dag_rulegraph.png"),
        svg=resources("dag_rulegraph.svg"),
    shell:
        r"""
        # Generate DOT file using nested snakemake with the dumped final config
        echo "[Rule rulegraph] Using final config file: {input.config_file}"
        snakemake --rulegraph --configfile {input.config_file} --quiet | sed -n "/digraph/,\$p" > {output.dot}

        # Generate visualizations from the DOT file
        if [ -s {output.dot} ]; then
            dot -c

            echo "[Rule rulegraph] Generating PDF from DOT"
            dot -Tpdf -o {output.pdf} {output.dot} || {{ echo "Error: Failed to generate PDF. Is graphviz installed?" >&2; exit 1; }}

            echo "[Rule rulegraph] Generating PNG from DOT"
            dot -Tpng -o {output.png} {output.dot} || {{ echo "Error: Failed to generate PNG. Is graphviz installed?" >&2; exit 1; }}

            echo "[Rule rulegraph] Generating SVG from DOT"
            dot -Tsvg -o {output.svg} {output.dot} || {{ echo "Error: Failed to generate SVG. Is graphviz installed?" >&2; exit 1; }}

            echo "[Rule rulegraph] Successfully generated all formats."
        else
            echo "[Rule rulegraph] Error: Failed to generate valid DOT content." >&2
            exit 1
        fi
        """


rule filegraph:
    """Generates File DAG in DOT, PDF, PNG, and SVG formats using the final configuration."""
    message:
        "Creating FILEGRAPH dag in multiple formats using the final configuration."
    input:
        config_file=rules.dump_graph_config.output.config_file,
    output:
        dot=resources("dag_filegraph.dot"),
        pdf=resources("dag_filegraph.pdf"),
        png=resources("dag_filegraph.png"),
        svg=resources("dag_filegraph.svg"),
    shell:
        r"""
        # Generate DOT file using nested snakemake with the dumped final config
        echo "[Rule filegraph] Using final config file: {input.config_file}"
        snakemake --filegraph all --configfile {input.config_file} --quiet | sed -n "/digraph/,\$p" > {output.dot}

        # Generate visualizations from the DOT file
        if [ -s {output.dot} ]; then
            echo "[Rule filegraph] Generating PDF from DOT"
            dot -Tpdf -o {output.pdf} {output.dot} || {{ echo "Error: Failed to generate PDF. Is graphviz installed?" >&2; exit 1; }}

            echo "[Rule filegraph] Generating PNG from DOT"
            dot -Tpng -o {output.png} {output.dot} || {{ echo "Error: Failed to generate PNG. Is graphviz installed?" >&2; exit 1; }}

            echo "[Rule filegraph] Generating SVG from DOT"
            dot -Tsvg -o {output.svg} {output.dot} || {{ echo "Error: Failed to generate SVG. Is graphviz installed?" >&2; exit 1; }}

            echo "[Rule filegraph] Successfully generated all formats."
        else
            echo "[Rule filegraph] Error: Failed to generate valid DOT content." >&2
            exit 1
        fi
        """


rule doc:
    message:
        "Build documentation."
    output:
        directory("doc/_build"),
    shell:
        "pixi run build-docs {output} html"


rule sync:
    params:
        cluster=f"{config['remote']['ssh']}:{config['remote']['path']}",
    shell:
        """
        rsync -uvarh --ignore-missing-args --files-from=.sync-send . {params.cluster}
        rsync -uvarh --no-g {params.cluster}/resources . || echo "No resources directory, skipping rsync"
        rsync -uvarh --no-g {params.cluster}/results . || echo "No results directory, skipping rsync"
        rsync -uvarh --no-g {params.cluster}/logs . || echo "No logs directory, skipping rsync"
        """


rule sync_dry:
    params:
        cluster=f"{config['remote']['ssh']}:{config['remote']['path']}",
    shell:
        """
        rsync -uvarh --ignore-missing-args --files-from=.sync-send . {params.cluster} -n
        rsync -uvarh --no-g {params.cluster}/resources . -n || echo "No resources directory, skipping rsync"
        rsync -uvarh --no-g {params.cluster}/results . -n || echo "No results directory, skipping rsync"
        rsync -uvarh --no-g {params.cluster}/logs . -n || echo "No logs directory, skipping rsync"
        """


# =============================================================================
# default target
# =============================================================================
rule all:
    input:
        PLAIN_DONE if MODE == "plain" else ROBUST_DONE if MODE == "robust" else ARO_DONE
    default_target: True


if MODE == "plain":

    rule plain_done:
        input:
            rules.plain_all.input
        output:
            PLAIN_DONE
        run:
            Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
            Path(output[0]).write_text("ok\n")


elif MODE == "robust":

    rule robust_done:
        input:
            rules.robust_postprocess.output.done
        output:
            ROBUST_DONE
        run:
            Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
            Path(output[0]).write_text("ok\n")




elif MODE == "aro":

    rule aro_full_postprocess:
        """
        Trigger the standard postprocess stack for the ARO result.

        NOTE:
        - We cannot reuse rules.plain_all.input directly because it contains input functions (lambda w: ...).
        - So we materialize a minimal set of standard postprocess targets here (strings only).
        """
        input:
            aro_std=RESULTS + "networks/aro_robust__std.nc",
            canonical=rules.aro_as_canonical_solved_network.output.canonical,
            costs_svg=expand(RESULTS + "graphs/costs.svg", run=config["run"]["name"]),
            power_network=expand(resources("maps/power-network.pdf"), run=config["run"]["name"]),
            power_network_clustered=expand(
                resources("maps/power-network-s-{clusters}.pdf"),
                run=config["run"]["name"],
                **config["scenario"],
            ),
            costs_all_map=expand(
                RESULTS + "maps/base_s_{clusters}_{opts}_{sector_opts}-costs-all_{planning_horizons}.pdf",
                run=config["run"]["name"],
                **config["scenario"],
            ),
            cop_profiles=expand(
                RESULTS + "graphs/cop_profiles_s_{clusters}_{planning_horizons}.html",
                run=config["run"]["name"],
                **config["scenario"],
            ),
            balance_timeseries=expand(
                RESULTS + "graphics/balance_timeseries/s_{clusters}_{opts}_{sector_opts}_{planning_horizons}",
                run=config["run"]["name"],
                **config["scenario"],
            ),
            heatmap_timeseries=expand(
                RESULTS + "graphics/heatmap_timeseries/s_{clusters}_{opts}_{sector_opts}_{planning_horizons}",
                run=config["run"]["name"],
                **config["scenario"],
            ),
            interactive_bus_balance=expand(
                RESULTS + "graphics/interactive_bus_balance/s_{clusters}_{opts}_{sector_opts}_{planning_horizons}",
                run=config["run"]["name"],
                **config["scenario"],
            ),
        output:
            done=RESULTS + "postprocess/aro_postprocess.done"
        run:
            Path(output.done).parent.mkdir(parents=True, exist_ok=True)
            Path(output.done).write_text("ok\n")

    rule aro_done:
        input:
            rules.aro_full_postprocess.output.done
        output:
            ARO_DONE
        run:
            Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
            Path(output[0]).write_text("ok\n")


rule analyze_aro_results:
    input:
        summary = "results/{run}/aro_summary.json",
        network = "results/{run}/robust_portfolio.nc"
    output:
        convergence = "results/{run}/plots/aro_metrics/aro_convergence.png",
        capacity = "results/{run}/plots/capacity/robust_capacity_breakdown.png"
    script:
        "plots_all/aro_analysis.py"
