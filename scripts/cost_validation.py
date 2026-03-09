# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Utilities to validate/fix suspicious capital_cost values on extendable assets."""

from __future__ import annotations

import logging
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def validate_extendable_capital_costs(
    n: pypsa.Network,
    *,
    stage: str,
    strict: bool = False,
    auto_fix: bool = False,
    floor_cost: float = 1.0,
    min_positive: float = 0.0,
    ignore_carriers: Iterable[str] | None = None,
) -> Dict[str, int]:
    """Validate extendable assets and optionally fix non-positive capital_cost values.

    Returns a dict with per-component number of fixed/flagged assets.
    """
    ignore = set(ignore_carriers or [])
    out: Dict[str, int] = {}

    comp_map = [
        ("generators", "p_nom_extendable", "capital_cost"),
        ("links", "p_nom_extendable", "capital_cost"),
        ("storage_units", "p_nom_extendable", "capital_cost"),
        ("stores", "e_nom_extendable", "capital_cost"),
        ("lines", "s_nom_extendable", "capital_cost"),
        ("transformers", "s_nom_extendable", "capital_cost"),
    ]

    total_bad = 0

    for comp, ext_col, cap_col in comp_map:
        df = getattr(n, comp, None)
        if df is None or len(df) == 0 or ext_col not in df.columns or cap_col not in df.columns:
            continue

        ext = df[df[ext_col].fillna(False).astype(bool)].copy()
        if ext.empty:
            continue

        if "carrier" in ext.columns and ignore:
            ext = ext[~ext["carrier"].astype(str).isin(ignore)]
            if ext.empty:
                continue

        cc = pd.to_numeric(ext[cap_col], errors="coerce")
        bad_mask = cc.isna() | ~np.isfinite(cc) | (cc <= min_positive)
        if not bad_mask.any():
            continue

        bad_idx = ext.index[bad_mask]
        total_bad += len(bad_idx)

        if auto_fix:
            df.loc[bad_idx, cap_col] = float(floor_cost)
            out[comp] = int(len(bad_idx))
            sample_cols = [c for c in ["carrier", cap_col] if c in df.columns]
            logger.warning(
                "[%s] auto-fix: %d extendable %s had non-positive/invalid %s -> %.3f. Examples: %s",
                stage,
                len(bad_idx),
                comp,
                cap_col,
                float(floor_cost),
                df.loc[bad_idx, sample_cols].head(5).to_dict("index") if sample_cols else list(bad_idx[:5]),
            )
        else:
            out[comp] = int(len(bad_idx))
            sample_cols = [c for c in ["carrier", cap_col] if c in df.columns]
            logger.error(
                "[%s] invalid extendable %s capital_cost on %d asset(s). Examples: %s",
                stage,
                comp,
                len(bad_idx),
                df.loc[bad_idx, sample_cols].head(5).to_dict("index") if sample_cols else list(bad_idx[:5]),
            )

    if total_bad and strict and not auto_fix:
        raise ValueError(
            f"[{stage}] Found {total_bad} extendable assets with invalid/non-positive capital_cost. "
            "Enable auto-fix or fix upstream cost data."
        )

    if total_bad == 0:
        logger.info("[%s] capital_cost validation OK: no invalid extendable assets found.", stage)

    return out