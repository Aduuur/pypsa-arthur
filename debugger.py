#!/usr/bin/env python3
"""Inspect and compare availability/profile NetCDF files across run folders.

Default focus is on the three test-normal-run folders:
- resources/test-normal-run-26
- resources/test-normal-run-45
- resources/test-normal-run

The script is robust to missing files and prints:
1) a compact per-file summary
2) a technology-level attractiveness summary (potentials/CF/energy-index)
3) run-to-run deltas for key metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

TARGET_FILES = [
    "availability_matrix_24_offwind-ac.nc",
    "availability_matrix_24_offwind-dc.nc",
    "availability_matrix_24_offwind-float.nc",
    "profile_24_offwind-ac.nc",
    "profile_24_offwind-dc.nc",
    "profile_24_offwind-float.nc",
    "profile_24_onwind.nc",
    "profile_24_solar-hsat.nc",
    "profile_24_solar.nc",
    "profile_hydro.nc",
]


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _stats_da(da: xr.DataArray) -> dict[str, float]:
    arr = np.asarray(da.values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return {
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "nonzero_share": np.nan,
        }

    vec = arr[finite]
    return {
        "mean": float(np.mean(vec)),
        "min": float(np.min(vec)),
        "max": float(np.max(vec)),
        "p50": float(np.percentile(vec, 50)),
        "p95": float(np.percentile(vec, 95)),
        "nonzero_share": float(np.mean(vec != 0.0)),
    }


def inspect_nc(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "kind": "unknown",
        "dims": {},
        "variables": [],
    }
    if not path.exists():
        out["error"] = "missing"
        return out

    # Some files are DataArrays (availability/hydro)
    try:
        obj = xr.open_dataarray(path)
        out["kind"] = "dataarray"
        out["dims"] = dict(obj.sizes)
        out["variables"] = [obj.name or "__dataarray__"]
        s = _stats_da(obj)
        out.update({f"value_{k}": v for k, v in s.items()})
        obj.close()
        return out
    except Exception:
        pass

    # Most profile files are Datasets
    try:
        ds = xr.open_dataset(path)
        out["kind"] = "dataset"
        out["dims"] = dict(ds.sizes)
        out["variables"] = list(ds.data_vars)

        for var in ["p_nom_max", "profile", "inflow", "average_distance"]:
            if var in ds:
                s = _stats_da(ds[var])
                out.update({f"{var}_{k}": v for k, v in s.items()})
                if var == "p_nom_max":
                    out["p_nom_max_sum"] = _safe_float(ds[var].sum().values)

        # fallback for arbitrary numeric variable
        if all(k not in out for k in ("profile_mean", "p_nom_max_mean", "inflow_mean")):
            numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
            if numeric_vars:
                v0 = numeric_vars[0]
                s = _stats_da(ds[v0])
                out.update({f"{v0}_{k}": v2 for k, v2 in s.items()})

        ds.close()
        return out
    except Exception as exc:
        out["error"] = f"read_failed: {exc}"
        return out


def _tech_from_file(fn: str) -> str | None:
    if not fn.startswith("profile_"):
        return None
    if fn == "profile_hydro.nc":
        return "hydro"
    # profile_24_<tech>.nc
    stem = fn.replace("profile_24_", "").replace(".nc", "")
    return stem


def _print_tech_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["file"].str.startswith("profile_")].copy()
    sub["tech"] = sub["file"].map(_tech_from_file)
    sub = sub[sub["tech"].notna()]

    # Energy index = potential * mean profile * 8760 (only for profiles with p_nom_max)
    if "p_nom_max_sum" not in sub.columns:
        sub["p_nom_max_sum"] = np.nan
    if "profile_mean" not in sub.columns:
        sub["profile_mean"] = np.nan
    sub["energy_index_mwh"] = sub["p_nom_max_sum"] * sub["profile_mean"] * 8760.0

    cols = [
        "run",
        "tech",
        "p_nom_max_sum",
        "profile_mean",
        "profile_p95",
        "profile_nonzero_share",
        "energy_index_mwh",
    ]
    cols = [c for c in cols if c in sub.columns]
    tech_df = sub[cols].sort_values(["run", "tech"]).reset_index(drop=True)

    print("\n=== Technology summary (profiles) ===")
    if len(tech_df) == 0:
        print("No profile rows available.")
        return tech_df

    print(tech_df.to_string(index=False))

    # Offshore aggregation + simple ratios
    off = sub[sub["tech"].isin(["offwind-ac", "offwind-dc", "offwind-float"])]
    grp = off.groupby("run", dropna=False).agg(
        offshore_p_nom_max_sum=("p_nom_max_sum", "sum"),
        offshore_energy_index_mwh=("energy_index_mwh", "sum"),
    )
    onw = sub[sub["tech"] == "onwind"].set_index("run")
    sol = sub[sub["tech"] == "solar"].set_index("run")
    if not grp.empty:
        grp["ratio_offshore_to_onwind_potential"] = grp["offshore_p_nom_max_sum"] / onw["p_nom_max_sum"]
        grp["ratio_offshore_to_solar_potential"] = grp["offshore_p_nom_max_sum"] / sol["p_nom_max_sum"]
        print("\n=== Offshore aggregate ratios ===")
        print(grp.reset_index().to_string(index=False))

    return tech_df


def _print_run_deltas(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "p_nom_max_sum",
        "profile_mean",
        "profile_p95",
        "profile_nonzero_share",
        "value_mean",
    ]
    have = [m for m in metrics if m in df.columns]
    if not have:
        print("\n=== Run deltas ===\nNo comparable metrics available.")
        return pd.DataFrame()

    piv = df.pivot_table(index="file", columns="run", values=have, aggfunc="first")
    if piv.empty or len(piv.columns.levels[1]) < 2:
        print("\n=== Run deltas ===\nNeed at least 2 runs for deltas.")
        return pd.DataFrame()

    run_order = sorted(df["run"].dropna().unique().tolist())
    base = run_order[0]
    out_rows: list[dict[str, Any]] = []
    for file_name in piv.index:
        for m in have:
            v_base = _safe_float(piv.loc[file_name, (m, base)]) if (m, base) in piv.columns else np.nan
            for r in run_order[1:]:
                key = (m, r)
                if key not in piv.columns:
                    continue
                v = _safe_float(piv.loc[file_name, key])
                delta = v - v_base
                rel = delta / abs(v_base) if np.isfinite(v_base) and abs(v_base) > 0 else np.nan
                out_rows.append(
                    {
                        "file": file_name,
                        "metric": m,
                        "base_run": base,
                        "run": r,
                        "base": v_base,
                        "value": v,
                        "delta": delta,
                        "rel_delta": rel,
                    }
                )

    out = pd.DataFrame(out_rows)
    print("\n=== Run deltas (vs first run alphabetically) ===")
    if out.empty:
        print("No deltas computed.")
    else:
        print(out.to_string(index=False))
    return out


def _emit_warnings(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []

    p = df[df["file"].str.startswith("profile_")].copy()
    if "p_nom_max_sum" in p.columns:
        small = p[(p["file"].str.contains("offwind-dc")) & (p["p_nom_max_sum"] < 1e5)]
        for _, r in small.iterrows():
            warnings.append(
                f"{r['run']} {r['file']}: very low offshore-dc potential (p_nom_max_sum={r['p_nom_max_sum']:.3g} MW)."
            )

    if "profile_mean" in p.columns:
        poor = p[(p["file"].str.contains("offwind")) & (p["profile_mean"] < 0.2)]
        for _, r in poor.iterrows():
            warnings.append(
                f"{r['run']} {r['file']}: low offshore mean profile ({r['profile_mean']:.3f})."
            )

    if "profile_nonzero_share" in p.columns:
        sparse = p[(p["file"].str.contains("offwind")) & (p["profile_nonzero_share"] < 0.8)]
        for _, r in sparse.iterrows():
            warnings.append(
                f"{r['run']} {r['file']}: sparse offshore profile ({r['profile_nonzero_share']:.3f} nonzero share)."
            )

    print("\n=== Heuristic warnings ===")
    if warnings:
        for w in warnings:
            print("-", w)
    else:
        print("None")
    return warnings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect availability/profile NetCDFs for test-normal runs.")
    p.add_argument("--base-dir", default="resources", help="Base directory where run folders live.")
    p.add_argument(
        "--run-dirs",
        nargs="+",
        default=["test-normal-run-26", "test-normal-run-45", "test-normal-run"],
        help="Run folder names under base-dir.",
    )
    p.add_argument("--output-csv", default="results/profile_inspection_summary.csv", help="CSV output path.")
    p.add_argument("--output-json", default="results/profile_inspection_summary.json", help="JSON output path.")
    p.add_argument(
        "--output-delta-csv",
        default="results/profile_inspection_deltas.csv",
        help="CSV output path for run-to-run deltas.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    base = Path(args.base_dir)

    rows: list[dict[str, Any]] = []
    for run in args.run_dirs:
        for fn in TARGET_FILES:
            p = base / run / fn
            rec = inspect_nc(p)
            rec["run"] = run
            rec["file"] = fn
            rows.append(rec)

    df = pd.DataFrame(rows)

    show_cols = [
        "run",
        "file",
        "exists",
        "kind",
        "p_nom_max_sum",
        "profile_mean",
        "profile_p95",
        "profile_nonzero_share",
        "value_mean",
        "error",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print("\n=== Profile inspection summary (compact) ===")
    print(df[show_cols].to_string(index=False))

    miss = df[~df["exists"]]
    print(f"\nMissing files: {len(miss)}/{len(df)}")
    if not miss.empty:
        print(miss[["run", "file"]].to_string(index=False))

    tech_df = _print_tech_summary(df)
    delta_df = _print_run_deltas(df)
    warnings = _emit_warnings(df)

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_delta = Path(args.output_delta_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rows": rows,
                "technology_summary": tech_df.to_dict(orient="records") if len(tech_df) else [],
                "run_deltas": delta_df.to_dict(orient="records") if len(delta_df) else [],
                "warnings": warnings,
            },
            f,
            indent=2,
        )
    if len(delta_df):
        delta_df.to_csv(out_delta, index=False)

    print(f"\nWrote CSV:       {out_csv}")
    print(f"Wrote JSON:      {out_json}")
    print(f"Wrote delta CSV: {out_delta if len(delta_df) else 'n/a (no deltas)'}")


if __name__ == "__main__":
    main()