#!/usr/bin/env python3
"""
Kapazitätsvergleich: Basisrun vs. ARO (Robustes Portfolio) vs. Worstcase.
Drei gestapelte Balken nebeneinander, im Stil von plot_installed_cap_new_vgl.py.
Erzeugt Plots für ALL und DE.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pypsa
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from master_config import PlottingConfig

plt.switch_backend("Agg")

# ── Carrier-Listen (identisch zu plot_installed_cap_new_vgl.py) ──────
ELECTRICITY_CARRIERS = [
    "onwind", "offwind-ac", "offwind-dc",
    "solar", "solar rooftop", "solar-hsat",
    "ror", "hydro",
    "coal", "lignite", "oil", "nuclear",
    "OCGT", "CCGT",
    "H2 turbine", "H2 OCGT", "H2 Fuel Cell",
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
    "battery discharger", "home battery discharger",
    "PHS",
    "OCGT methanol", "biomass",
]

STORAGE_DISCHARGE_CARRIERS = [
    "battery discharger", "home battery discharger", "PHS"
]

CARRIER_TRANSLATION = {
    "solar": "Photovoltaik",
    "onwind": "Wind Onshore",
    "offwind-ac": "Wind Offshore (AC)",
    "offwind-dc": "Wind Offshore (DC)",
    "ror": "Laufwasser",
    "hydro": "Wasserkraft",
    "nuclear": "Kernkraft",
    "lignite": "Braunkohle",
    "coal": "Steinkohle",
    "oil": "Öl",
    "CCGT": "Erdgas (GuD)",
    "OCGT": "Erdgas (Gasturbine)",
    "PHS": "Pumpspeicher",
    "battery discharger": "Batteriespeicher",
    "home battery discharger": "Heimspeicher",
    "H2 turbine": "H₂-Turbine",
    "H2 OCGT": "H₂-Gasturbine",
    "H2 Fuel Cell": "Brennstoffzelle",
    "OCGT methanol": "OCGT Methanol",
}

COUNTRY_NAMES = {
    "ALL": "Europa (Gesamt)", "DE": "Deutschland", "FR": "Frankreich",
    "ES": "Spanien", "IT": "Italien", "GB": "Großbritannien",
    "PL": "Polen", "SE": "Schweden", "NO": "Norwegen",
    "NL": "Niederlande", "BE": "Belgien", "AT": "Österreich",
    "CH": "Schweiz", "DK": "Dänemark", "CZ": "Tschechien",
}

FONT_SIZES = {
    "title": 20, "subtitle": 16, "axis_label": 16,
    "ticks": 15, "bar_text": 13, "legend": 14, "legend_title": 15,
}

# Reihenfolge im Stapel (von unten nach oben)
TECH_ORDER = [
    "Kernkraft", "Photovoltaik",
    "Wind Onshore", "Wind Offshore (AC)", "Wind Offshore (DC)",
    "Laufwasser", "Wasserkraft", "Biomasse",
    "Braunkohle", "Steinkohle", "Öl",
    "Erdgas (GuD)", "Erdgas (Gasturbine)",
    "OCGT Methanol",
    "Pumpspeicher", "Batteriespeicher", "Heimspeicher",
    "H₂-Turbine", "H₂-Gasturbine", "Brennstoffzelle",
]


def get_country_codes(n):
    prefixes = set(b[:2] for b in n.buses.index)
    return sorted([p for p in prefixes if p != "EU" and p.isalpha() and p.isupper() and len(p) == 2])


def get_installed_capacity(n, country_code):
    """Installierte Kapazität in GW (deutsche Labels), identisch zu plot_installed_cap_new_vgl."""
    parts = []
    pcol = "p_nom_opt"
    valid_countries = get_country_codes(n)
    regex_countries = "^(" + "|".join(valid_countries) + ")"

    # 1. Generatoren
    gens = n.generators.copy()
    if country_code != "ALL":
        gens = gens[gens.bus.str.startswith(country_code)]
    else:
        gens = gens[gens.bus.str.contains(regex_countries)]
    gens = gens[gens.carrier.isin(ELECTRICITY_CARRIERS)]
    if not gens.empty:
        parts.append(gens.groupby("carrier")[pcol].sum())

    # 2. Links
    links = n.links.copy()
    if country_code != "ALL":
        links = links[links.bus0.str.startswith(country_code) | links.bus1.str.startswith(country_code)]
    else:
        links = links[links.bus1.str.contains(regex_countries)]
    links = links[links.carrier.isin(ELECTRICITY_CARRIERS)]
    links = links[links.carrier != "nuclear"]
    if not links.empty:
        CORRECTION_MAP = {"OCGT": 0.39, "CCGT": 0.58}
        cap_final = links[pcol].copy()
        for tech in links.carrier.unique():
            needs_correction = False
            correction_factor = 1.0
            if tech in CORRECTION_MAP:
                needs_correction = True
                correction_factor = CORRECTION_MAP[tech]
            elif any(x in tech for x in ["OCGT", "CCGT"]):
                needs_correction = True
                correction_factor = 0.58 if "CCGT" in tech else 0.39
            if needs_correction:
                mask = links.carrier == tech
                if "efficiency" in links.columns:
                    eff_vals = links.loc[mask, "efficiency"]
                    if eff_vals.mean() < 0.9:
                        cap_final.loc[mask] *= eff_vals
                    else:
                        cap_final.loc[mask] *= correction_factor
                else:
                    cap_final.loc[mask] *= correction_factor
        parts.append(cap_final.groupby(links.carrier).sum())

    # 3. Storage Units
    sus = n.storage_units
    if country_code != "ALL":
        sus = sus[sus.bus.str.startswith(country_code)]
    else:
        sus = sus[sus.bus.str.contains(regex_countries)]
    sus_elec = sus[sus.carrier.isin(STORAGE_DISCHARGE_CARRIERS)]
    if not sus_elec.empty:
        parts.append(sus_elec.groupby("carrier")[pcol].sum())
    hydro_sus = n.storage_units[~n.storage_units.bus.str.contains("EU", case=False, na=False)]
    if country_code != "ALL":
        hydro_sus = hydro_sus[hydro_sus.bus.str.startswith(country_code)]
    hydro_sus = hydro_sus[hydro_sus.carrier.isin(["hydro", "ror"])]
    if not hydro_sus.empty:
        parts.append(hydro_sus.groupby("carrier")[pcol].sum())

    if not parts:
        return pd.Series(dtype=float)

    raw = pd.concat(parts).groupby(level=0).sum() / 1000.0  # MW → GW
    clean = {}

    # Solar zusammenfassen
    solar_sum = raw.filter(regex="solar").sum()
    raw = raw.drop(raw.index[raw.index.str.contains("solar")], errors="ignore")
    if solar_sum > 0:
        clean["Photovoltaik"] = solar_sum

    # Biomasse zusammenfassen
    biomass_sum = raw.filter(regex="biomass").sum()
    raw = raw.drop(raw.index[raw.index.str.contains("biomass")], errors="ignore")
    if biomass_sum > 0:
        clean["Biomasse"] = biomass_sum

    for c, v in raw.items():
        name = CARRIER_TRANSLATION.get(c, c)
        clean[name] = clean.get(name, 0.0) + v

    s = pd.Series(clean)
    return s[s > 0.01]


def plot_three_bars(caps_dict, country, config, out_dir):
    """Drei gestapelte Balken im Stil von plot_installed_cap_new_vgl."""
    # Alle Technologien sammeln
    all_techs = set()
    for cap in caps_dict.values():
        all_techs |= set(cap.index)

    # Sortieren nach TECH_ORDER
    ordered = [t for t in TECH_ORDER if t in all_techs]
    rest = [t for t in all_techs if t not in ordered]
    techs_ordered = ordered + rest

    # Daten als DataFrame
    df = pd.DataFrame(caps_dict).reindex(techs_ordered).fillna(0)

    # Filter: nur Techs > 1% des Maximums
    total_max = df.sum().max()
    ALWAYS_KEEP = ["Braunkohle", "Steinkohle", "Biomasse", "Öl", "Kernkraft",
                   "Laufwasser", "Wasserkraft", "Erdgas (GuD)", "Erdgas (Gasturbine)",
                   "Pumpspeicher", "Batteriespeicher", "Heimspeicher"]
    keep = (df.max(axis=1) >= 0.01 * total_max) | df.index.isin(ALWAYS_KEEP)
    df = df[keep]
    techs_ordered = df.index.tolist()

    # Plot
    c_name = COUNTRY_NAMES.get(country, country)
    labels = list(caps_dict.keys())
    n_bars = len(labels)

    fig, ax = plt.subplots(figsize=(10 + 2 * n_bars, 10))
    x = np.arange(n_bars)
    width = 0.55
    TEXT_THRESHOLD = 0.025 * total_max

    # Schraffur: mittlerer Balken (ARO) ohne, links und rechts mit
    hatches = [None, None, None]
    if n_bars == 3:
        hatches = [None, None, None]

    bottoms = [np.zeros(1) for _ in range(n_bars)]

    for tech in techs_ordered:
        # Farbe bestimmen
        if tech == "Biomasse":
            color = config.CARRIER_COLORS.get("biomass", "#6a3d9a")
        else:
            k = next((k for k, v in CARRIER_TRANSLATION.items() if v == tech), tech)
            color = config.CARRIER_COLORS.get(k, config.DEFAULT_COLOR)

        for bar_idx in range(n_bars):
            val = df.loc[tech, labels[bar_idx]]
            if val < 0.01:
                continue
            hatch = hatches[bar_idx] if bar_idx < len(hatches) else None

            ax.bar(x[bar_idx], val, width, bottom=bottoms[bar_idx][0],
                   color=color, edgecolor="white", linewidth=0.5,
                   hatch=hatch, alpha=0.95,
                   label=tech if bar_idx == 0 else None)

            if val >= TEXT_THRESHOLD:
                y_center = bottoms[bar_idx][0] + val / 2
                fontcolor = "white" if val > 0.05 * total_max else "#333"
                ax.text(x[bar_idx], y_center, f"{int(round(val))}",
                        ha="center", va="center", color=fontcolor,
                        fontsize=FONT_SIZES["bar_text"], fontweight="bold")

            bottoms[bar_idx][0] += val

    ax.set_title(
        f"Installierte Kapazität: {c_name}\n"
        f"(Vergleich aller drei Optimierungsläufe)",
        fontsize=FONT_SIZES["title"], pad=20,
    )
    ax.set_ylabel("Kapazität (GW)", fontsize=FONT_SIZES["axis_label"])
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZES["ticks"])
    ax.tick_params(axis="y", labelsize=FONT_SIZES["ticks"])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    # Legende
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    legend_order = techs_ordered[::-1]
    legend_handles = [by_label[t] for t in legend_order if t in by_label]
    legend_labels = [t for t in legend_order if t in by_label]
    ax.legend(legend_handles, legend_labels,
              title="Technologie",
              fontsize=FONT_SIZES["legend"],
              title_fontsize=FONT_SIZES["legend_title"],
              loc="upper left", bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = f"capacity_comparison_3runs_{country}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {fname}")


def main():
    config = PlottingConfig()
    results_base = Path(config.master.aro_results_base)

    # ── Die drei Planungsnetzwerke laden (aro_robust__std.nc) ─────
    runs = {
        "Basisrun": list(results_base.glob("Referenzrun*/networks/aro_robust__std.nc")),
        "Robustes\nPortfolio (ARO)": [results_base / config.master.aro_selected_run / "networks" / "aro_robust__std.nc"],
        "Deterministischer\nWorstcase": list(results_base.glob("Worstcase*/networks/aro_robust__std.nc")),
    }

    networks = {}
    for label, candidates in runs.items():
        found = [p for p in candidates if p.is_file()]
        if found:
            print(f"  {label.replace(chr(10),' ')}: {found[0].parent.parent.name}/{found[0].name}")
            networks[label] = pypsa.Network(str(found[0]))
        else:
            print(f"  ✗ {label.replace(chr(10),' ')}: nicht gefunden")

    if len(networks) < 2:
        print("✗ Mindestens 2 Netzwerke nötig")
        return

    out_dir = os.path.join(config.PLOT_OUTPUT_PATH, "capacity_comparison")

    for country in ["ALL", "DE"]:
        print(f"\n{'='*50}")
        print(f"  Kapazitätsvergleich: {country}")
        print(f"{'='*50}")

        caps = {}
        for label, n in networks.items():
            cap = get_installed_capacity(n, country)
            caps[label] = cap
            print(f"  {label.replace(chr(10),' ')}: {cap.sum():.0f} GW")

        plot_three_bars(caps, country, config, out_dir)

    print(f"\n✅ Fertig. Plots in: {out_dir}")


if __name__ == "__main__":
    main()
