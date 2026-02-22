# SPDX-FileCopyrightText: : 2025 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Create global summary CSV files by concatenating individual summary CSVs.

This version is:
- robust against missing/empty individual files (e.g. partial/ARO workflows)
- tolerant to slight format differences when reading
- BUT it keeps the *PyPSA-Eur downstream expectations* (plot_summary etc.):

  => Each individual CSV is reduced to exactly ONE data column (prefer column "0",
     otherwise the first column). The global summary then has a 3-level MultiIndex
     on columns: (cluster, opt, planning_horizon).

This avoids breaking plot_summary with unexpected extra column levels and avoids
errors like KeyError: 'carrier' due to changed CSV layouts.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

INDEX_COLS = {
    "nodal_costs": 4,
    "nodal_capacities": 3,
    "nodal_capacity_factors": 3,
    "capacity_factors": 2,
    "costs": 3,
    "capacities": 2,
    "curtailment": 1,
    "energy": 2,
    "energy_balance": 3,
    "nodal_energy_balance": 4,
    "prices": 1,
    "weighted_prices": 1,
    "market_values": 1,
    "metrics": 1,
}


def _read_individual_csv(path: str, kind: str) -> pd.DataFrame | None:
    """
    Read one individual summary CSV defensively.
    Returns a DataFrame (possibly empty) or None if file missing/unreadable.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Missing input for %s: %s", kind, path)
        return None

    n_index_cols = INDEX_COLS.get(kind, 1)

    # Try expected index layout first
    try:
        df = pd.read_csv(p, index_col=list(range(n_index_cols)))
    except Exception as e:
        logger.warning(
            "Failed to read %s with index_col=0..%d (%s). Falling back to index_col=0. File=%s",
            kind,
            n_index_cols - 1,
            repr(e),
            path,
        )
        try:
            df = pd.read_csv(p, index_col=0)
        except Exception as e2:
            logger.error(
                "Failed to read %s even with fallback (%s). Skipping file=%s",
                kind,
                repr(e2),
                path,
            )
            return None

    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Some CSVs might be "index-only" => no data columns
    if df.shape[1] == 0:
        logger.warning("Input %s is empty (no data columns): %s", kind, path)

    return df


def _squeeze_to_single_column(df: pd.DataFrame, kind: str, path: str) -> pd.DataFrame | None:
    """
    Normalize an individual summary dataframe to exactly ONE data column
    for downstream compatibility (plot_summary expects this).

    Preference order:
      1) column named "0"
      2) first column
    Returns None if there is no data column.
    """
    if df is None:
        return None

    if df.shape[1] == 0:
        # no data columns
        logger.warning("Skipping %s (no columns) for %s: %s", kind, kind, path)
        return None

    if "0" in df.columns:
        out = df[["0"]].copy()
        out.columns = ["0"]
        return out

    # take first column
    out = df.iloc[:, [0]].copy()
    out.columns = ["0"]
    return out


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("make_global_summary")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    base_dir = Path("results") / snakemake.params.RDIR / "csvs" / "individual"

    for kind in snakemake.output.keys():
        logger.info("Creating global summary for %s", kind)

        # Build mapping run_id -> file
        summaries_dict: dict[tuple, str] = {}
        for cluster in snakemake.params.scenario["clusters"]:
            for opt in snakemake.params.scenario["opts"]:
                for sector_opt in snakemake.params.scenario["sector_opts"]:
                    for planning_horizon in snakemake.params.scenario["planning_horizons"]:
                        run_key = (cluster, opt + sector_opt, planning_horizon)
                        fname = f"{kind}_s_{cluster}_{opt}_{sector_opt}_{planning_horizon}.csv"
                        summaries_dict[run_key] = str(base_dir / fname)

        read_frames: list[pd.DataFrame] = []
        read_keys: list[tuple] = []

        for key, filename in summaries_dict.items():
            df = _read_individual_csv(filename, kind)
            df = _squeeze_to_single_column(df, kind=kind, path=filename)
            if df is None:
                continue

            read_frames.append(df)
            read_keys.append(key)

        out_path = Path(snakemake.output[kind])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # If nothing could be read, write empty CSV (so Snakemake can continue)
        if len(read_frames) == 0:
            logger.warning(
                "No readable/non-empty inputs found for %s. Writing empty CSV to %s",
                kind,
                out_path,
            )
            pd.DataFrame().to_csv(out_path)
            continue

        # Concatenate along columns: each run contributes exactly one column named "0"
        summaries = pd.concat(read_frames, axis=1)

        # Set expected 3-level MultiIndex columns (PyPSA-Eur compatible)
        cols = pd.MultiIndex.from_tuples(
            read_keys,
            names=["cluster", "opt", "planning_horizon"],
        )

        # In the normalized setup, number of columns must match number of keys
        if summaries.shape[1] != len(cols):
            logger.warning(
                "Column mismatch for %s: summaries has %d cols, expected %d. "
                "Will truncate to the minimum to avoid crashing.",
                kind,
                summaries.shape[1],
                len(cols),
            )
            m = min(summaries.shape[1], len(cols))
            summaries = summaries.iloc[:, :m]
            cols = cols[:m]

        summaries.columns = cols
        summaries.sort_index().to_csv(out_path)

        del summaries