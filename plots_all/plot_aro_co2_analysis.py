#!/usr/bin/env python3
"""
ARO CO2-Analyse: 4 Plots
1. CO2-Emissionen je Szenario (Balken)
2. CO2-Intensität nach Technologie (gestapelt)
3. CO2-Budget-Check (Emissionen vs. Capture vs. Limit)
4. Robustheit der Emissionen (Boxplot über alle Szenarien)
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pypsa
from pathlib import Path
from master_config import PlottingConfig, get_planning_year

plt.switch_backend("Agg")

# ─────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────

def calc_co2_emissions(n: pypsa.Network) -> dict:
    """Berechnet Brutto-Emissionen, DAC-Capture und Netto je Land."""
    results = {}
    hps = float(n.snapshot_weightings.generators.mean())

    # Brutto-Emissionen aus Generatoren
    for country in ["DE", "FR", "ES", "GB", "PL", "IT", "NL", "BE",
                     "AT", "CH", "CZ", "DK", "EE", "FI", "LT", "LV",
                     "NO", "PT", "SE", "EU"]:
        if country == "EU":
            buses = n.buses
        else:
            buses = n.buses[n.buses.index.str.startswith(country)]

        gross = 0.0
        # Generatoren
        for g in n.generators.index:
            gen = n.generators.loc[g]
            if gen.bus not in buses.index:
                continue
            carrier = gen.carrier
            ef = n.carriers.at[carrier, "co2_emissions"] if carrier in n.carriers.index else 0.0
            if ef == 0.0:
                continue
            if g in n.generators_t.p.columns:
                gross += (n.generators_t.p[g] * hps * ef).sum() / 1e6  # Mt
            else:
                gross += gen.p_nom_opt * gen.capital_cost * 0  # nur Dispatch

        # Links (CCGT, OCGT, coal, lignite)
        # Elektrische Busse im Land fuer Link-Filterung
        elec_bus_carriers = {"AC", "DC", "electricity", "low voltage"}
        if country == "EU":
            elec_buses_c = n.buses[n.buses.carrier.isin(elec_bus_carriers)].index
        else:
            elec_buses_c = n.buses[
                n.buses.carrier.isin(elec_bus_carriers) &
                n.buses.index.str.startswith(country)
            ].index

        for lk in n.links.index:
            link = n.links.loc[lk]
            carrier = link.carrier
            ef = n.carriers.at[carrier, "co2_emissions"] if carrier in n.carriers.index else 0.0
            if ef == 0.0:
                continue
            # bus1 muss elektrischer Bus im Land sein (unabhaengig von bus0)
            if link.bus1 not in elec_buses_c:
                continue
            if lk in n.links_t.p0.columns:
                gross += (n.links_t.p0[lk].clip(lower=0) * hps * ef).sum() / 1e6

        # DAC Capture (negativ)
        capture = 0.0
        dac_links = n.links[
            n.links.carrier.str.contains("DAC|dac", case=False, na=False) &
            (n.links.bus0.str.startswith(country) if country != "EU"
             else pd.Series(True, index=n.links.index))
        ]
        for lk in dac_links.index:
            if lk in n.links_t.p0.columns:
                capture += (n.links_t.p0[lk].abs() * hps).sum() / 1e6

        # CO2-Sequestration
        seq_links = n.links[
            n.links.carrier.str.contains("sequester|sequestered", case=False, na=False) &
            (n.links.bus0.str.startswith(country) if country != "EU"
             else pd.Series(True, index=n.links.index))
        ]
        for lk in seq_links.index:
            if lk in n.links_t.p0.columns:
                capture += (n.links_t.p0[lk].abs() * hps).sum() / 1e6

        results[country] = {
            "gross": gross,
            "capture": capture,
            "net": gross - capture,
        }
    return results


def short_name(nc_path: str) -> str:
    m = re.search(r"stress_(\d+)_from_(\d{4})_(\d{2})_(\d{2})", Path(nc_path).name)
    if m:
        s, y, mo, d = m.groups()
        wc = "_WC" if "worst_case" in Path(nc_path).name else ""
        return f"S{s}\n{mo}/{d}{wc}"
    return Path(nc_path).stem[:15]


# ─────────────────────────────────────────────
# Plot 1: CO2 je Szenario
# ─────────────────────────────────────────────

def plot_co2_per_scenario(scenario_data: dict, out_dir: str, run_name: str,
                           config: PlottingConfig):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"CO₂-Emissionen je Szenario [{run_name}]", fontsize=14, fontweight="bold")

    for ax, country, title in zip(axes, ["DE", "EU"],
                                   ["Deutschland", "Europa (Summe)"]):
        names, gross, capture, net = [], [], [], []
        for nc, data in scenario_data.items():
            names.append(short_name(nc))
            gross.append(data.get(country, {}).get("gross", 0))
            capture.append(data.get(country, {}).get("capture", 0))
            net.append(data.get(country, {}).get("net", 0))

        x = np.arange(len(names))
        w = 0.25
        ax.bar(x - w, gross, w, label="Brutto", color="#d62728", alpha=0.8)
        ax.bar(x, capture, w, label="DAC/Capture", color="#2ca02c", alpha=0.8)
        ax.bar(x + w, net, w, label="Netto", color="#1f77b4", alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("CO₂ [Mt/a]")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"co2_per_scenario_{run_name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {p}")


# ─────────────────────────────────────────────
# Plot 2: CO2 nach Technologie
# ─────────────────────────────────────────────

def plot_co2_by_technology(n: pypsa.Network, out_dir: str, run_name: str,
                            config: PlottingConfig, scenario_label: str = ""):
    hps = float(n.snapshot_weightings.generators.mean())
    carrier_em = {}

    for g in n.generators.index:
        carrier = n.generators.at[g, "carrier"]
        ef = n.carriers.at[carrier, "co2_emissions"] if carrier in n.carriers.index else 0.0
        if ef == 0.0:
            continue
        if g in n.generators_t.p.columns:
            em = (n.generators_t.p[g] * hps * ef).sum() / 1e6
            carrier_em[carrier] = carrier_em.get(carrier, 0) + em

    for lk in n.links.index:
        carrier = n.links.at[lk, "carrier"]
        ef = n.carriers.at[carrier, "co2_emissions"] if carrier in n.carriers.index else 0.0
        if ef == 0.0:
            continue
        if lk in n.links_t.p0.columns:
            em = (n.links_t.p0[lk].clip(lower=0) * hps * ef).sum() / 1e6
            carrier_em[carrier] = carrier_em.get(carrier, 0) + em

    if not carrier_em:
        print("  ⚠️ Keine CO2-Emissionen gefunden")
        return

    df = pd.Series(carrier_em).sort_values(ascending=False)
    colors = [config.CARRIER_COLORS.get(c, config.DEFAULT_COLOR) for c in df.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot.bar(ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("CO₂ [Mt/a]")
    ax.set_title(f"CO₂-Emissionen nach Technologie [{run_name}] {scenario_label}",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(ax.patches, df.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    label = scenario_label.replace("\n", "_").replace("/", "-").replace(" ", "_")
    p = os.path.join(out_dir, f"co2_by_technology_{run_name}_{label}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {p}")


# ─────────────────────────────────────────────
# Plot 3: CO2-Budget-Check
# ─────────────────────────────────────────────

def plot_co2_budget_check(scenario_data: dict, out_dir: str, run_name: str,
                           config: PlottingConfig):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"CO₂-Budget-Check: Emissionen vs. Capture [{run_name}]",
                 fontsize=13, fontweight="bold")

    names = [short_name(nc) for nc in scenario_data]
    gross_eu = [d.get("EU", {}).get("gross", 0) for d in scenario_data.values()]
    cap_eu   = [d.get("EU", {}).get("capture", 0) for d in scenario_data.values()]
    net_eu   = [d.get("EU", {}).get("net", 0) for d in scenario_data.values()]

    x = np.arange(len(names))
    w = 0.3
    ax.bar(x - w/2, gross_eu, w, label="Brutto-Emissionen", color="#d62728", alpha=0.85)
    ax.bar(x + w/2, cap_eu,   w, label="DAC + Sequestration", color="#2ca02c", alpha=0.85)

    # Netto als Linie
    ax.plot(x, net_eu, "o-", color="#1f77b4", lw=2, markersize=6, label="Netto", zorder=5)
    ax.axhline(0, color="black", lw=1.2, linestyle="--", label="Net-Zero Ziel")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("CO₂ [Mt/a]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"co2_budget_check_{run_name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {p}")


# ─────────────────────────────────────────────
# Plot 4: Robustheit der Emissionen (Boxplot)
# ─────────────────────────────────────────────

def plot_co2_robustness(scenario_data: dict, out_dir: str, run_name: str,
                         config: PlottingConfig):
    countries = ["DE", "FR", "ES", "GB", "PL", "IT", "NL", "NO", "EU"]
    net_by_country = {c: [] for c in countries}

    for data in scenario_data.values():
        for c in countries:
            net_by_country[c].append(data.get(c, {}).get("net", 0))

    fig, ax = plt.subplots(figsize=(12, 5))
    data_list = [net_by_country[c] for c in countries]
    bp = ax.boxplot(data_list, labels=countries, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.6)

    ax.axhline(0, color="red", lw=1.2, linestyle="--", label="Net-Zero")
    ax.set_ylabel("Netto-CO₂ [Mt/a]")
    ax.set_title(f"Robustheit der CO₂-Emissionen über alle Szenarien [{run_name}]",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"co2_robustness_{run_name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {p}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    config = PlottingConfig()
    out_dir = os.path.join(config.PLOT_OUTPUT_PATH, "co2_aro")
    run_name = config.SCENARIO_SELECTION

    # Alle Dispatch-Netze finden
    from aro_analysis import AROAnalyzer
    analyzer = AROAnalyzer(config)
    dispatch_dir = Path(config.BASE_RESULTS_PATH if hasattr(config, "BASE_RESULTS_PATH")
                        else analyzer.config.BASE_RESULTS_PATH) / run_name / "networks" / "dispatch"

    if not dispatch_dir.is_dir():
        # Fallback _dispatch_tmp
        for sub in ("final", "iter1"):
            fb = dispatch_dir.parent / "_dispatch_tmp" / sub
            if fb.is_dir() and any(fb.glob("dispatch_*_std.nc")):
                dispatch_dir = fb
                break

    nc_files = sorted(dispatch_dir.glob("dispatch_*_std.nc"))
    if not nc_files:
        print("  ⚠️ Keine Dispatch-Netze gefunden")
        return

    print(f"  Lade {len(nc_files)} Dispatch-Netze für CO2-Analyse...")
    scenario_data = {}
    networks = {}
    for nc in nc_files:
        try:
            n = pypsa.Network(str(nc))
            scenario_data[str(nc)] = calc_co2_emissions(n)
            networks[str(nc)] = n
            print(f"    ✓ {nc.name}")
        except Exception as e:
            print(f"    ✗ {nc.name}: {e}")

    if not scenario_data:
        print("  ⚠️ Keine Daten verfügbar")
        return

    # Plot 1: CO2 je Szenario
    plot_co2_per_scenario(scenario_data, out_dir, run_name, config)

    # Plot 2: CO2 nach Technologie (Worst-Case)
    wc = next((k for k in networks if "worst_case" in k), list(networks.keys())[0])
    plot_co2_by_technology(networks[wc], out_dir, run_name, config,
                           scenario_label=short_name(wc))

    # Plot 3: Budget-Check
    plot_co2_budget_check(scenario_data, out_dir, run_name, config)

    # Plot 4: Robustheit
    plot_co2_robustness(scenario_data, out_dir, run_name, config)

    print(f"\n✅ CO2-Analyse fertig: {out_dir}")


if __name__ == "__main__":
    main()
