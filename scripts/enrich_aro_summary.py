#!/usr/bin/env python3
"""
Post-processing script: enriches aro_summary.json with
capacities, costs, and LS data from the solved network.

Usage:
  python scripts/enrich_aro_summary.py \
    --network results/baserun-rcp85-test/networks/aro_robust.nc \
    --summary results/baserun-rcp85-test/aro_summary.json \
    --dispatch-log results/baserun-rcp85-test/logs/solve_aro.log
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pypsa


def _total(n, comp, pnom, ext_col):
    df = getattr(n, comp)
    if ext_col not in df.columns:
        return {}
    ext = df[df[ext_col].fillna(False).astype(bool)]
    fix = df[~df[ext_col].fillna(False).astype(bool)]
    result = {}
    for carrier in df["carrier"].unique():
        e = ext[ext["carrier"] == carrier]
        f = fix[fix["carrier"] == carrier]
        total_eu = e[pnom].sum() + f[pnom.replace("_opt", "")].sum() if pnom.endswith("_opt") else e[pnom].sum() + f[pnom].sum()
        new_eu   = e[pnom].sum()
        result[carrier] = {"total_eu": float(total_eu), "new_eu": float(new_eu)}
    return result


def extract_capacities(n):
    caps = {}
    # Generators
    for carrier in ["solar", "solar rooftop", "onwind", "offwind-ac", "offwind-dc", "nuclear", "ror"]:
        gens = n.generators[n.generators.carrier == carrier]
        ext  = gens[gens.p_nom_extendable.fillna(False)]
        fix  = gens[~gens.p_nom_extendable.fillna(False)]
        for region, mask_fn in [("EU", lambda x: x), ("DE", lambda x: x[x.index.str.startswith("DE0")])]:
            e = mask_fn(ext); f = mask_fn(fix)
            caps[f"gen_{carrier.replace(' ','_').replace('-','_')}_{region}_GW"] = round(
                (e.p_nom_opt.sum() + f.p_nom.sum()) / 1e3, 2)
            caps[f"gen_{carrier.replace(' ','_').replace('-','_')}_{region}_new_GW"] = round(
                e.p_nom_opt.sum() / 1e3, 2)

    # Links
    for carrier in ["urban central air heat pump", "urban central resistive heater",
                    "urban decentral air heat pump", "urban decentral resistive heater",
                    "rural air heat pump", "OCGT methanol", "battery charger",
                    "H2 Electrolysis", "urban central water pits charger"]:
        lk  = n.links[n.links.carrier == carrier]
        ext = lk[lk.p_nom_extendable.fillna(False)]
        fix = lk[~lk.p_nom_extendable.fillna(False)]
        key = carrier.replace(' ','_').replace('-','_')
        for region, mask_fn in [("EU", lambda x: x), ("DE", lambda x: x[x.index.str.startswith("DE0")])]:
            e = mask_fn(ext); f = mask_fn(fix)
            caps[f"link_{key}_{region}_GW"] = round(
                (e.p_nom_opt.sum() + f.p_nom.sum()) / 1e3, 2)

    # Stores
    for carrier in ["battery", "H2 Store", "urban central water pits"]:
        st  = n.stores[n.stores.carrier == carrier]
        ext = st[st.e_nom_extendable.fillna(False)]
        fix = st[~st.e_nom_extendable.fillna(False)]
        key = carrier.replace(' ','_').replace('-','_')
        for region, mask_fn in [("EU", lambda x: x), ("DE", lambda x: x[x.index.str.startswith("DE0")])]:
            e = mask_fn(ext); f = mask_fn(fix)
            caps[f"store_{key}_{region}_GWh"] = round(
                (e.e_nom_opt.sum() + f.e_nom.sum()) / 1e3, 2)
    return caps


def extract_costs(n):
    inv = 0.0
    for comp, pnom, cc, ext in [
        ("generators", "p_nom_opt", "capital_cost", "p_nom_extendable"),
        ("links",      "p_nom_opt", "capital_cost", "p_nom_extendable"),
        ("stores",     "e_nom_opt", "capital_cost", "e_nom_extendable"),
        ("lines",      "s_nom_opt", "capital_cost", "s_nom_extendable"),
    ]:
        df = getattr(n, comp)
        if ext in df.columns and cc in df.columns and pnom in df.columns:
            e = df[df[ext].fillna(False).astype(bool)]
            inv += (e[cc] * e[pnom]).sum()
    return {"investment_cost_EUR_per_a": round(inv, 2),
            "investment_cost_Mrd_EUR_per_a": round(inv / 1e9, 3)}


def extract_ls_from_log(log_path):
    """Parse LS breakdown from solve_aro.log."""
    if not Path(log_path).exists():
        return {}
    txt = Path(log_path).read_text()
    # Find last LS breakdown
    matches = re.findall(
        r"LS breakdown: total=([\d.e+\-]+) MWh.*?peak=([\d.e+\-]+) MW.*?top_assets=(\{[^}]+\})",
        txt
    )
    if not matches:
        return {}
    total, peak, top_str = matches[-1]
    # Parse top assets
    top = {}
    for m in re.finditer(r"'([^']+)':\s*([\d.e+\-]+)", top_str):
        top[m.group(1)] = float(m.group(2))
    return {
        "ls_total_MWh": float(total),
        "ls_peak_MW":   float(peak),
        "ls_top_assets": top,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",      required=True)
    parser.add_argument("--summary",      required=True)
    parser.add_argument("--dispatch-log", default="")
    args = parser.parse_args()

    print(f"Loading network: {args.network}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = pypsa.Network(args.network)

    with open(args.summary) as f:
        summary = json.load(f)

    summary["capacities_GW"]  = extract_capacities(n)
    summary["costs"]          = extract_costs(n)
    if args.dispatch_log:
        summary["dispatch_ls"] = extract_ls_from_log(args.dispatch_log)

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary enriched: {args.summary}")
    print(f"  Investment: {summary['costs']['investment_cost_Mrd_EUR_per_a']:.1f} Mrd EUR/a")
    if "dispatch_ls" in summary:
        print(f"  LS total:   {summary['dispatch_ls'].get('ls_total_MWh', 0):.1f} MWh")


if __name__ == "__main__":
    main()
