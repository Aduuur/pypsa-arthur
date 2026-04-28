#!/usr/bin/env python3
"""
ARO Erzeugungsvergleich: Robustes Portfolio (Worst-Case-Dispatch) vs. Basisjahr
- Gestapeltes Balkendiagramm je Land
- Differenzplot (Rob - Basis)
- Anteil erneuerbarer Energien
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa
from pathlib import Path
from master_config import PlottingConfig, get_planning_year

plt.switch_backend("Agg")

# Carrier die als Stromerzeugung zählen
ELEC_CARRIERS = [
    "onwind", "offwind-ac", "offwind-dc", "offwind-float",
    "solar", "solar rooftop", "solar-hsat",
    "ror", "hydro", "PHS",
    "nuclear", "coal", "lignite", "gas", "oil", "OCGT", "CCGT",
    "biogas", "solid biomass",
    "H2 turbine", "H2 Fuel Cell",
    "OCGT methanol", "CCGT methanol",
]

RES_CARRIERS = {"onwind", "offwind-ac", "offwind-dc", "offwind-float",
                "solar", "solar rooftop", "solar-hsat", "ror", "hydro"}

COUNTRIES = ["DE", "FR", "ES", "GB", "PL", "IT", "NL", "BE",
             "AT", "CH", "CZ", "DK", "NO", "SE", "FI", "PT"]


def get_elec_generation_twh(n: pypsa.Network, country: str) -> pd.Series:
    """Gibt Erzeugung je Carrier in TWh zurück."""
    hps = float(n.snapshot_weightings.generators.mean())
    result = {}

    if country == "EU":
        bus_mask = pd.Series(True, index=n.buses.index)
    else:
        bus_mask = n.buses.index.str.startswith(country)
        bus_mask = pd.Series(bus_mask, index=n.buses.index)

    buses = n.buses[bus_mask].index

    # Generatoren
    gens = n.generators[n.generators.bus.isin(buses)]
    gens = gens[gens.carrier.isin(ELEC_CARRIERS)]
    for carrier, grp in gens.groupby("carrier"):
        twh = 0.0
        for g in grp.index:
            if g in n.generators_t.p.columns:
                twh += (n.generators_t.p[g] * hps).sum() / 1e6
            else:
                # Kein Dispatch — p_nom_opt * capacity_factor schätzen
                twh += grp.at[g, "p_nom_opt"] * 0.0  # nur echte Dispatch-Daten
        if twh > 0:
            result[carrier] = result.get(carrier, 0) + twh

    # Links (elektrische Ausgabe via bus1)
    # Strategie: bus1 muss ein elektrischer Bus im gesuchten Land sein.
    # Gilt unabhängig von bus0 (z.B. bus0=EU methanol ist OK).
    elec_carriers_links = {
        "H2 turbine", "H2 Fuel Cell",
        "OCGT methanol", "CCGT methanol",
        "OCGT", "CCGT", "gas",
        "lignite", "coal", "oil",
        "nuclear",
        "urban central solid biomass CHP",
        "urban central solid biomass CHP CC",
        "urban central gas CHP", "urban central gas CHP CC",
        "solid biomass",
    }
    # Elektrische Busse im Land: carrier AC, DC oder electricity
    elec_bus_carriers = {"AC", "DC", "electricity", "low voltage"}
    elec_buses_country = n.buses[
        n.buses.carrier.isin(elec_bus_carriers) & bus_mask
    ].index

    for lk in n.links.index:
        link = n.links.loc[lk]
        carrier = link.carrier
        if carrier not in elec_carriers_links:
            continue
        # bus1 muss elektrischer Bus im Land sein
        if link.bus1 not in elec_buses_country:
            continue
        if lk in n.links_t.p1.columns:
            # p1 ist negativ wenn Strom aus bus1 fliesst (Erzeugung)
            twh = (-n.links_t.p1[lk]).clip(lower=0).sum() * hps / 1e6
            if twh > 0.001:
                result[carrier] = result.get(carrier, 0) + twh

    return pd.Series(result).sort_values(ascending=False)


def plot_comparison_bar(gen_rob: pd.Series, gen_bas: pd.Series,
                         country: str, out_dir: str, run_name: str,
                         config: PlottingConfig, year_rob: int, year_bas: int):
    """Gestapeltes Balkendiagramm Robust vs. Basis."""
    all_carriers = sorted(set(gen_rob.index) | set(gen_bas.index))
    colors = [config.CARRIER_COLORS.get(c, config.DEFAULT_COLOR) for c in all_carriers]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.array([0, 1])
    bottom_r = np.zeros(1)
    bottom_b = np.zeros(1)
    handles = []

    for carrier, color in zip(all_carriers, colors):
        val_r = gen_rob.get(carrier, 0.0)
        val_b = gen_bas.get(carrier, 0.0)
        if val_r == 0 and val_b == 0:
            continue
        bar_r = ax.bar(0, val_r, bottom=bottom_r, color=color, edgecolor="white", lw=0.3)
        ax.bar(1, val_b, bottom=bottom_b, color=color, edgecolor="white", lw=0.3)
        bottom_r += val_r
        bottom_b += val_b
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=carrier))

    total_r = gen_rob.sum()
    total_b = gen_bas.sum()
    ax.text(0, total_r + total_r * 0.01, f"{total_r:.0f} TWh",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(1, total_b + total_b * 0.01, f"{total_b:.0f} TWh",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    res_r = sum(gen_rob.get(c, 0) for c in RES_CARRIERS)
    res_b = sum(gen_bas.get(c, 0) for c in RES_CARRIERS)
    pct_r = res_r / total_r * 100 if total_r > 0 else 0
    pct_b = res_b / total_b * 100 if total_b > 0 else 0
    ax.text(0, -total_r * 0.06, f"EE: {pct_r:.1f}%",
            ha="center", va="top", fontsize=9, color="#2ca02c", fontweight="bold")
    ax.text(1, -total_b * 0.06, f"EE: {pct_b:.1f}%",
            ha="center", va="top", fontsize=9, color="#2ca02c", fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Robust ARO\n(WC-Dispatch {year_rob})",
                         f"Basisjahr\n{year_bas}"], fontsize=11)
    ax.set_ylabel("Stromerzeugung [TWh/a]")
    ax.set_title(f"Stromerzeugung: Robust vs. Basis — {country} [{run_name}]",
                 fontsize=13, fontweight="bold")
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              ncol=2, bbox_to_anchor=(1.18, 1))
    ax.set_xlim(-0.6, 1.6)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"energy_comparison_{country}_{run_name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {p}")


def plot_diff_heatmap(diffs: pd.DataFrame, out_dir: str, run_name: str,
                       config: PlottingConfig):
    """Differenz-Heatmap: (Robust - Basis) je Land und Carrier."""
    if diffs.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    data = diffs.fillna(0)
    vmax = max(abs(data.values.max()), abs(data.values.min()), 1)

    im = ax.imshow(data.T, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(data.index)))
    ax.set_xticklabels(data.index, fontsize=9)
    ax.set_yticks(range(len(data.columns)))
    ax.set_yticklabels(data.columns, fontsize=9)
    plt.colorbar(im, ax=ax, label="ΔErzeugung [TWh] (Robust − Basis)")
    ax.set_title(f"Δ Stromerzeugung Robust − Basis [{run_name}]",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    p = os.path.join(out_dir, f"energy_diff_heatmap_{run_name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {p}")


def plot_res_share_comparison(res_data: dict, out_dir: str, run_name: str,
                               config: PlottingConfig):
    """EE-Anteil Vergleich je Land."""
    countries = list(res_data.keys())
    rob_pct = [res_data[c]["rob_pct"] for c in countries]
    bas_pct = [res_data[c]["bas_pct"] for c in countries]

    x = np.arange(len(countries))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, rob_pct, w, label="Robust ARO (WC)", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, bas_pct, w, label="Basisjahr", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=9)
    ax.set_ylabel("EE-Anteil [%]")
    ax.set_ylim(0, 110)
    ax.axhline(100, color="green", lw=1, linestyle="--", alpha=0.5)
    ax.set_title(f"Erneuerbare-Energien-Anteil: Robust vs. Basis [{run_name}]",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for i, (r, b) in enumerate(zip(rob_pct, bas_pct)):
        ax.text(i - w/2, r + 1, f"{r:.0f}%", ha="center", va="bottom", fontsize=7)
        ax.text(i + w/2, b + 1, f"{b:.0f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    p = os.path.join(out_dir, f"res_share_comparison_{run_name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {p}")


def main():
    config = PlottingConfig()
    out_dir = os.path.join(config.PLOT_OUTPUT_PATH, "energy_comparison")
    run_name = config.SCENARIO_SELECTION

    # Pfade aus Config
    from master_config import MASTER_CONFIG
    results_base = MASTER_CONFIG.get("aro", {}).get("results_base",
                   MASTER_CONFIG.get("results", {}).get("base_path", "results"))

    # Worst-Case Dispatch finden
    dispatch_dir = Path(results_base) / run_name / "networks" / "dispatch"
    if not dispatch_dir.is_dir():
        for sub in ("final", "iter1"):
            fb = Path(results_base) / run_name / "networks" / "_dispatch_tmp" / sub
            if fb.is_dir() and any(fb.glob("dispatch_*_std.nc")):
                dispatch_dir = fb
                break

    # Worst-Case bevorzugen, sonst erstes Szenario
    wc_files = sorted(dispatch_dir.glob("dispatch_*worst_case*_std.nc"))
    all_files = sorted(dispatch_dir.glob("dispatch_*_std.nc"))
    nc_rob = wc_files[0] if wc_files else (all_files[0] if all_files else None)

    if nc_rob is None:
        print("  ⚠️ Kein Dispatch-Netz gefunden")
        return

    # Basis-Netz
    runs = MASTER_CONFIG.get("runs", {})
    ref_run = None
    for rk, rv in runs.items():
        if rv.get("run_type") == "normal":
            ref_run = rk
            break
    if ref_run is None:
        ref_run = MASTER_CONFIG.get("reference_run", "")

    bas_candidates = list(Path(results_base).glob(f"{ref_run}/networks/base_s_*___*.nc")) if ref_run else []
    if not bas_candidates:
        bas_candidates = list(Path(results_base).glob("*/networks/base_s_*___2050.nc"))

    if not bas_candidates:
        print("  ⚠️ Kein Basisnetz gefunden")
        return

    nc_bas = bas_candidates[0]
    print(f"  Robust (WC): {nc_rob.name}")
    print(f"  Basis:       {nc_bas.name}")

    n_rob = pypsa.Network(str(nc_rob))
    n_bas = pypsa.Network(str(nc_bas))
    year_rob = get_planning_year(n_rob, str(nc_rob))
    year_bas = get_planning_year(n_bas, str(nc_bas))

    print(f"  Jahre: Robust={year_rob}, Basis={year_bas}")

    # Daten sammeln
    diff_data = {}
    res_data = {}

    for country in COUNTRIES:
        gen_r = get_elec_generation_twh(n_rob, country)
        gen_b = get_elec_generation_twh(n_bas, country)

        if gen_r.sum() == 0 and gen_b.sum() == 0:
            continue

        # Vergleichs-Balken je Land
        plot_comparison_bar(gen_r, gen_b, country, out_dir, run_name,
                            config, year_rob, year_bas)

        # Diff für Heatmap
        all_c = set(gen_r.index) | set(gen_b.index)
        diff_data[country] = {c: gen_r.get(c, 0) - gen_b.get(c, 0) for c in all_c}

        # EE-Anteil
        tot_r = gen_r.sum()
        tot_b = gen_b.sum()
        res_r = sum(gen_r.get(c, 0) for c in RES_CARRIERS)
        res_b = sum(gen_b.get(c, 0) for c in RES_CARRIERS)
        res_data[country] = {
            "rob_pct": res_r / tot_r * 100 if tot_r > 0 else 0,
            "bas_pct": res_b / tot_b * 100 if tot_b > 0 else 0,
        }

    # Heatmap
    if diff_data:
        df_diff = pd.DataFrame(diff_data).T.fillna(0)
        df_diff = df_diff.loc[:, (df_diff.abs() > 0.1).any()]
        plot_diff_heatmap(df_diff, out_dir, run_name, config)

    # EE-Anteil Vergleich
    if res_data:
        plot_res_share_comparison(res_data, out_dir, run_name, config)

    print(f"\n✅ Erzeugungsvergleich fertig: {out_dir}")


if __name__ == "__main__":
    main()
