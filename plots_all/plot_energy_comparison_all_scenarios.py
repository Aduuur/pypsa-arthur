#!/usr/bin/env python3
"""
Erzeugungsvergleich (TWh) im Stil der Kapazitätsvergleichs-Plots:
  1. ARO Worst-Case vs. Basisrun  (2 Balken, Schraffur)
  2. Alle 3 ARO-Szenarien nebeneinander (3 Balken)

Erzeugt Plots für ALL und DE.
"""
import os, sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pypsa
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from master_config import PlottingConfig, get_planning_year

plt.switch_backend("Agg")

# ── Carrier-Konfiguration ──────────────────────────────────────────
ELEC_CARRIERS = [
    "onwind", "offwind-ac", "offwind-dc",
    "solar", "solar rooftop",
    "ror", "hydro", "PHS",
    "nuclear", "coal", "lignite", "oil", "OCGT", "CCGT",
    "biogas", "solid biomass",
    "urban central solid biomass CHP", "urban central solid biomass CHP CC",
    "H2 turbine", "H2 Fuel Cell",
    "OCGT methanol", "CCGT methanol",
    "battery discharger", "home battery discharger",
]

# Von unten nach oben im Stapel
STACK_ORDER = [
    "nuclear",
    "solar", "solar rooftop",
    "onwind", "offwind-ac", "offwind-dc",
    "ror", "hydro", "PHS",
    "biomass", "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
    "solid biomass", "biogas",
    "coal", "lignite", "oil",
    "CCGT", "OCGT",
    "OCGT methanol", "CCGT methanol",
    "H2 turbine", "H2 Fuel Cell",
    "battery discharger", "home battery discharger",
]

CARRIER_LABELS_DE = {
    "onwind":       "Wind Onshore",
    "offwind-ac":   "Wind Offshore (AC)",
    "offwind-dc":   "Wind Offshore (DC)",
    "solar":        "Photovoltaik",
    "solar rooftop":"Solar Aufdach",
    "ror":          "Laufwasser",
    "hydro":        "Wasserkraft",
    "PHS":          "Pumpspeicher",
    "nuclear":      "Kernkraft",
    "CCGT":         "Erdgas (GuD)",
    "OCGT":         "Erdgas (Gasturbine)",
    "coal":         "Steinkohle",
    "lignite":      "Braunkohle",
    "oil":          "Öl",
    "biomass":      "Biomasse",
    "solid biomass":"Festbiomasse",
    "biogas":       "Biogas",
    "urban central solid biomass CHP": "Biomasse (KWK)",
    "urban central solid biomass CHP CC": "Biomasse (KWK CC)",
    "H2 turbine":   "H₂-Turbine",
    "H2 Fuel Cell": "Brennstoffzelle",
    "OCGT methanol":"OCGT Methanol",
    "CCGT methanol":"CCGT Methanol",
    "battery discharger": "Batteriespeicher",
    "home battery discharger": "Heimspeicher",
}


def get_elec_generation_twh(n, country="ALL"):
    """Stromerzeugung in TWh pro Carrier."""
    result = {}
    dt = n.snapshot_weightings["objective"].iloc[0]  # Zeitschritt in Stunden

    for carrier in ELEC_CARRIERS:
        # Generatoren
        gens = n.generators[n.generators.carrier == carrier]
        if country != "ALL":
            gens = gens[gens.bus.str.startswith(country)]
        cols = [c for c in gens.index if c in n.generators_t.p.columns]
        energy = 0.0
        if cols:
            energy += (n.generators_t.p[cols].sum().sum() * dt) / 1e6

        # Storage Units (hydro, PHS)
        sus = n.storage_units[n.storage_units.carrier == carrier]
        if country != "ALL":
            sus = sus[sus.bus.str.startswith(country)]
        su_cols = [c for c in sus.index if c in n.storage_units_t.p.columns]
        if su_cols:
            # Nur Entladung (p > 0)
            energy += (n.storage_units_t.p[su_cols].clip(lower=0).sum().sum() * dt) / 1e6

        # Links (battery discharger, H2 turbine etc.)
        lks = n.links[n.links.carrier == carrier]
        if country != "ALL":
            # bus1 = Strom-Bus für Discharger
            lks = lks[lks.bus0.str.startswith(country) | lks.bus1.str.startswith(country)]
        lk_cols = [c for c in lks.index if c in n.links_t.p0.columns]
        if lk_cols:
            # Für Discharger: p0 ist positiv (aus Store), Strom geht raus über bus1
            # Effektive Einspeisung = -p1
            p1_cols = [c for c in lk_cols if c in n.links_t.p1.columns]
            if p1_cols:
                energy += (-n.links_t.p1[p1_cols].clip(upper=0).sum().sum() * dt) / 1e6

        if energy > 0.5:
            result[carrier] = energy

    return pd.Series(result)


def plot_two_bar_comparison(gen_left, gen_right, label_left, label_right,
                            country, out_dir, config, hatch_right=True):
    """Zwei gestapelte Balken nebeneinander im Stil des Kapazitätsvergleichs."""
    # Alle Carrier sammeln
    all_carriers = list(set(gen_left.index) | set(gen_right.index))
    # Sortieren
    ordered = [c for c in STACK_ORDER if c in all_carriers]
    rest = [c for c in all_carriers if c not in ordered]
    all_carriers = ordered + rest

    title_country = "Europa (Gesamt)" if country == "ALL" else country

    fig, ax = plt.subplots(figsize=(8, 10))
    bar_width = 0.38
    x_positions = [0.3, 0.7]

    for bar_idx, (gen_data, x_pos) in enumerate([(gen_left, x_positions[0]),
                                                   (gen_right, x_positions[1])]):
        bottom = 0
        for carrier in all_carriers:
            val = gen_data.get(carrier, 0)
            if val < 0.5:
                continue
            color = config.CARRIER_COLORS.get(carrier, "#999999")
            label = CARRIER_LABELS_DE.get(carrier, carrier)
            hatch = "///" if (bar_idx == 1 and hatch_right) else None

            bar = ax.bar(x_pos, val, width=bar_width, bottom=bottom,
                         color=color, edgecolor="white", linewidth=0.5,
                         hatch=hatch, alpha=0.9,
                         label=label if bar_idx == 0 else None)

            # Wert ins Segment schreiben (nur wenn >30 TWh)
            if val > 30:
                y_center = bottom + val / 2
                fontcolor = "white" if val > 80 else "#333"
                ax.text(x_pos, y_center, f"{val:.0f}",
                        ha="center", va="center", fontsize=8,
                        fontweight="bold", color=fontcolor)

            bottom += val

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label_left, label_right], fontsize=11)
    ax.set_ylabel("Stromerzeugung (TWh)", fontsize=13)
    ax.set_title(f"Stromerzeugung: {title_country}\n"
                 f"(Links: {label_left} | Rechts: {label_right})",
                 fontsize=13)
    ax.set_xlim(-0.05, 1.05)

    # Legende (nur einmal pro Carrier)
    handles, labels = ax.get_legend_handles_labels()
    # Umkehren für von-oben-nach-unten
    ax.legend(handles[::-1], labels[::-1],
              bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, title="Technologie")

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    safe_l = label_left.replace(" ", "_").replace("(", "").replace(")", "")
    safe_r = label_right.replace(" ", "_").replace("(", "").replace(")", "")
    fname = f"energy_comparison_{safe_l}_vs_{safe_r}_{country}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {fname}")


def plot_multi_bar_comparison(gen_dict, country, out_dir, config):
    """Mehrere Szenarien als gestapelte Balken nebeneinander."""
    all_carriers = set()
    for gen in gen_dict.values():
        all_carriers |= set(gen.index)
    ordered = [c for c in STACK_ORDER if c in all_carriers]
    rest = [c for c in all_carriers if c not in ordered]
    all_carriers = ordered + rest

    title_country = "Europa (Gesamt)" if country == "ALL" else country
    n_bars = len(gen_dict)

    fig, ax = plt.subplots(figsize=(3 + 2.5 * n_bars, 9))
    bar_width = 0.6
    x_positions = np.arange(n_bars)

    for bar_idx, (label, gen_data) in enumerate(gen_dict.items()):
        bottom = 0
        for carrier in all_carriers:
            val = gen_data.get(carrier, 0)
            if val < 0.5:
                continue
            color = config.CARRIER_COLORS.get(carrier, "#999999")
            clabel = CARRIER_LABELS_DE.get(carrier, carrier)

            ax.bar(x_positions[bar_idx], val, width=bar_width, bottom=bottom,
                   color=color, edgecolor="white", linewidth=0.5, alpha=0.9,
                   label=clabel if bar_idx == 0 else None)

            if val > 40:
                y_center = bottom + val / 2
                fontcolor = "white" if val > 100 else "#333"
                ax.text(x_positions[bar_idx], y_center, f"{val:.0f}",
                        ha="center", va="center", fontsize=8,
                        fontweight="bold", color=fontcolor)
            bottom += val

    ax.set_xticks(x_positions)
    ax.set_xticklabels(list(gen_dict.keys()), fontsize=10)
    ax.set_ylabel("Stromerzeugung (TWh)", fontsize=13)
    ax.set_title(f"Erzeugungsvergleich: {title_country}", fontsize=14)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, title="Technologie")

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = f"energy_comparison_all_scenarios_{country}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {fname}")


def main():
    config = PlottingConfig()
    results_base = Path(config.master.aro_results_base)
    aro_run = config.master.aro_selected_run
    dispatch_dir = results_base / aro_run / "networks" / "dispatch"

    if not dispatch_dir.is_dir():
        print(f"✗ Dispatch-Ordner nicht gefunden: {dispatch_dir}")
        return

    all_nc = sorted(dispatch_dir.glob("dispatch_*_std.nc"))
    print(f"Lade {len(all_nc)} Dispatch-Netzwerke...")

    scenarios = {}
    wc_network = None
    for nc in all_nc:
        m = re.search(r"stress_(\d+)", nc.name)
        is_wc = "worst_case" in nc.name
        if m:
            num = m.group(1)
            label = f"Stress {num}"
            if is_wc:
                label += " (WC)"
        else:
            label = nc.stem[:25]
        print(f"  {label}: {nc.name}")
        n = pypsa.Network(str(nc))
        scenarios[label] = n
        if is_wc:
            wc_network = n

    # Basisrun
    # Dispatch-Netzwerk des Basisruns (nicht das Planungsnetz!)
    bas_candidates = list(results_base.glob("Referenzrun*/networks/dispatch/dispatch_*_std.nc"))
    if not bas_candidates:
        ref_run = config.master.raw.get("scenarios", {}).get("reference_run", "")
        if ref_run:
            bas_candidates = list(results_base.glob(f"{ref_run}/networks/dispatch/dispatch_*_std.nc"))
    if not bas_candidates:
        bas_candidates = list(results_base.glob("*/networks/dispatch/dispatch_*worst_case*_std.nc"))

    n_bas = None
    if bas_candidates:
        print(f"  Basisrun: {bas_candidates[0].name}")
        n_bas = pypsa.Network(str(bas_candidates[0]))
    else:
        print("  ⚠️ Kein Basisrun gefunden")

    out_dir = os.path.join(config.PLOT_OUTPUT_PATH, "energy_comparison")

    for country in ["ALL", "DE"]:
        print(f"\n{'='*60}")
        print(f"  {country}")
        print(f"{'='*60}")

        # ── Plot 1: WC vs. Basisrun ──────────────────────────────
        if wc_network is not None and n_bas is not None:
            gen_wc = get_elec_generation_twh(wc_network, country)
            gen_bas = get_elec_generation_twh(n_bas, country)
            print(f"  WC:    {gen_wc.sum():.0f} TWh")
            print(f"  Basis: {gen_bas.sum():.0f} TWh")
            plot_two_bar_comparison(
                gen_wc, gen_bas,
                "Robustes Portfolio", "Basisjahr",
                country, out_dir, config, hatch_right=True,
            )

        # ── Plot 2: Alle Szenarien ───────────────────────────────
        gen_dict = {}
        if n_bas is not None:
            gen_dict["Basisrun"] = get_elec_generation_twh(n_bas, country)
        for label, n in scenarios.items():
            gen_dict[label] = get_elec_generation_twh(n, country)
        plot_multi_bar_comparison(gen_dict, country, out_dir, config)

    print(f"\n✅ Fertig. Plots in: {out_dir}")


if __name__ == "__main__":
    main()
