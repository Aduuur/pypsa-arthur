#!/usr/bin/env python3
"""
plot_supplementary.py — Ergänzende Plots für die ARO-Auswertung
================================================================

Generiert die fehlenden Plots:
  1. energy_generated pro Szenario (Balkendiagramm TWh/Carrier)
  2. energy_generated Vergleich: alle 3 Szenarien + Basisrun
  3. Dunkelflaute-Detail-Dispatch (7-Tage-Zoom, gestapelt)
  4. LS-Vergleich über die Szenarien
  5. Curtailment-Vergleich über die Szenarien

Aufruf:
  python plots_all/plot_supplementary.py

Liest die run_config.yaml aus dem ARO-Ergebnisordner.
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Pfade ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

RESULT_DIR = os.path.join(BASE_DIR, "results", "3-Szenarien-Run")
BASIS_DIR  = os.path.join(BASE_DIR, "results", "Referenzrun-rcp45")
WC_DIR     = os.path.join(BASE_DIR, "results", "Worstcase-rcp45")

DISPATCH_DIR = os.path.join(RESULT_DIR, "networks", "dispatch")
PLOT_BASE    = "/mnt/endata/MA_Arthur/PyPSA-results/plots/3-Szenarien-Run"

# ── Farben (aus master_config.py) ──────────────────────────────────────
CARRIER_COLORS = {
    "onwind":       "#235ebc",
    "offwind-ac":   "#6895dd",
    "offwind-dc":   "#74c6f2",
    "solar":        "#f9d002",
    "solar rooftop":"#ffea80",
    "ror":          "#78d4b6",
    "hydro":        "#298c81",
    "PHS":          "#3dbfb0",
    "nuclear":      "#ff8c00",
    "CCGT":         "#ee8340",
    "OCGT":         "#e07a5f",
    "coal":         "#545454",
    "lignite":      "#826a4c",
    "biomass":      "#baa741",
    "biogas":       "#e6d690",
    "oil":          "#262626",
    "H2 turbine":   "#ba91c8",
    "H2 Fuel Cell": "#c8a2d1",
    "battery discharger": "#7b2d8b",
    "home battery discharger": "#b05cc7",
    "OCGT methanol":"#d4553a",
    "urban central solid biomass CHP": "#9d7c3a",
    "urban central solid biomass CHP CC": "#8a6d32",
    "load_shedding":"#ff0000",
}

CARRIER_LABELS = {
    "onwind": "Wind Onshore",
    "offwind-ac": "Wind Offshore (AC)",
    "offwind-dc": "Wind Offshore (DC)",
    "solar": "Photovoltaik",
    "solar rooftop": "Solar Aufdach",
    "ror": "Laufwasser",
    "hydro": "Speicherwasser",
    "PHS": "Pumpspeicher",
    "nuclear": "Kernkraft",
    "CCGT": "Erdgas (GuD)",
    "OCGT": "Erdgas (OC)",
    "coal": "Steinkohle",
    "lignite": "Braunkohle",
    "biomass": "Biomasse (KWK)",
    "biogas": "Biogas",
    "H2 turbine": "H₂-Turbine",
    "H2 Fuel Cell": "Brennstoffzelle",
    "battery discharger": "Batterie",
    "OCGT methanol": "OCGT Methanol",
    "oil": "Öl",
}

COUNTRIES = ["ALL", "DE", "FR", "ES", "DK", "NO", "GB", "IT", "NL", "PL",
             "BE", "FI", "AT", "CH", "SE", "PT", "CZ", "LT", "LV", "EE"]

# ── Hilfsfunktionen ───────────────────────────────────────────────────

def load_network(path):
    """Lade ein PyPSA-Netzwerk."""
    import pypsa
    n = pypsa.Network(path)
    return n


def get_dispatch_files():
    """Alle Dispatch-Netzwerke im Dispatch-Ordner finden."""
    files = {}
    for f in sorted(os.listdir(DISPATCH_DIR)):
        if f.endswith("_std.nc") and f.startswith("dispatch_"):
            # Szenario-Name extrahieren
            name = f.replace("dispatch_cutout_mCNRM-CERFACS-CM5_rcp45_2028_", "")
            name = name.replace("_std.nc", "")
            label = name
            if "stress_01" in name:
                label = "stress_01"
            elif "stress_02" in name:
                label = "stress_02"
            elif "stress_03" in name or "worst_case" in name:
                label = "stress_03 (WC)"
            files[label] = os.path.join(DISPATCH_DIR, f)
    return files


def get_energy_by_carrier(n, country="ALL"):
    """Berechne Energieerzeugung (TWh) pro Carrier für ein Land oder EU."""
    gen_data = {}
    for carrier in n.generators.carrier.unique():
        if "load_shedding" in str(carrier) or "NegLoad" in str(carrier):
            continue
        gens = n.generators[n.generators.carrier == carrier]
        if country != "ALL":
            gens = gens[gens.bus.str.startswith(country)]
        if gens.empty:
            continue
        cols = [c for c in gens.index if c in n.generators_t.p.columns]
        if not cols:
            # Statische Generatoren
            static_p = gens["p_nom_opt"].sum()
            if static_p > 0:
                gen_data[carrier] = static_p * len(n.snapshots) * 2 / 1e6  # 2h Zeitschritt
            continue
        energy_twh = n.generators_t.p[cols].sum().sum() * 2 / 1e6  # 2h Auflösung
        if energy_twh > 0.01:
            gen_data[carrier] = energy_twh
    return gen_data


def get_df_window(n, days_before=3, days_after=4):
    """Finde das Dunkelflauten-Fenster basierend auf der Stunde mit höchster Residuallast."""
    # Residuallast = Last - RE-Erzeugung
    re_carriers = ["onwind", "offwind-ac", "offwind-dc", "solar", "solar rooftop", "ror"]
    re_gen = pd.Series(0.0, index=n.snapshots)
    for carrier in re_carriers:
        gens = n.generators[n.generators.carrier == carrier]
        cols = [c for c in gens.index if c in n.generators_t.p.columns]
        if cols:
            re_gen += n.generators_t.p[cols].sum(axis=1)

    load = n.loads_t.p_set.sum(axis=1)
    residual = load - re_gen
    peak_time = residual.idxmax()

    start = peak_time - pd.Timedelta(days=days_before)
    end = peak_time + pd.Timedelta(days=days_after)
    start = max(start, n.snapshots[0])
    end = min(end, n.snapshots[-1])

    return start, end, peak_time


def get_ls_data(n, country="ALL"):
    """Berechne Load-Shedding pro Stunde."""
    ls_cols = [c for c in n.generators_t.p.columns if "LS::" in c]
    if not ls_cols:
        return pd.Series(0.0, index=n.snapshots)
    if country != "ALL":
        ls_cols = [c for c in ls_cols if country in c]
    return n.generators_t.p[ls_cols].sum(axis=1) if ls_cols else pd.Series(0.0, index=n.snapshots)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: Energy Generated pro Szenario (Einzelbalken)
# ═══════════════════════════════════════════════════════════════════════

def plot_energy_per_scenario(networks, out_dir):
    """Erzeuge energy_generated Plots pro Szenario und Land."""
    os.makedirs(out_dir, exist_ok=True)

    for label, n in networks.items():
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        for country in ["ALL", "DE"]:
            data = get_energy_by_carrier(n, country)
            if not data:
                continue

            # Sortieren nach Größe
            s = pd.Series(data).sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = [CARRIER_COLORS.get(c, "#999999") for c in s.index]
            labels = [CARRIER_LABELS.get(c, c) for c in s.index]
            bars = ax.barh(range(len(s)), s.values, color=colors)
            ax.set_yticks(range(len(s)))
            ax.set_yticklabels(labels)
            ax.set_xlabel("Energieerzeugung (TWh)")
            ax.set_title(f"Stromerzeugung {country} — {label}")

            # Werte an die Balken
            for bar, val in zip(bars, s.values):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f"{val:.0f}", va="center", fontsize=8)

            ax.set_xlim(0, s.max() * 1.15)
            fig.tight_layout()
            fname = f"energy_generated_{country}_{safe_label}.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=150)
            plt.close(fig)
            print(f"  ✅ {fname}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: Energy Generated Vergleich (alle Szenarien + Basisrun)
# ═══════════════════════════════════════════════════════════════════════

def plot_energy_comparison(networks, basis_n, out_dir):
    """Vergleiche Energieerzeugung über alle Szenarien + Basisrun."""
    os.makedirs(out_dir, exist_ok=True)

    for country in ["ALL", "DE"]:
        all_data = {}
        # Basisrun
        all_data["Basisrun"] = get_energy_by_carrier(basis_n, country)
        # Szenarien
        for label, n in networks.items():
            all_data[label] = get_energy_by_carrier(n, country)

        # DataFrame bauen
        df = pd.DataFrame(all_data).fillna(0)
        # Top-Carrier nach Gesamterzeugung
        df["total"] = df.sum(axis=1)
        df = df.sort_values("total", ascending=False).drop(columns="total")
        df = df[df.max(axis=1) > 1]  # Mindestens 1 TWh

        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(df.columns))
        width = 0.8 / len(df)
        bottom = np.zeros(len(df.columns))

        for i, (carrier, row) in enumerate(df.iterrows()):
            color = CARRIER_COLORS.get(carrier, "#999999")
            label = CARRIER_LABELS.get(carrier, carrier)
            ax.bar(x, row.values, width=0.7, bottom=bottom,
                   color=color, label=label, edgecolor="white", linewidth=0.3)
            bottom += row.values

        ax.set_xticks(x)
        ax.set_xticklabels(df.columns, rotation=15, ha="right")
        ax.set_ylabel("Energieerzeugung (TWh)")
        ax.set_title(f"Stromerzeugung nach Szenario — {country}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1)
        fig.tight_layout()

        fname = f"energy_comparison_{country}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ {fname}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 3: Dunkelflaute-Detail-Dispatch (7-Tage-Zoom)
# ═══════════════════════════════════════════════════════════════════════

def plot_df_detail_dispatch(networks, out_dir):
    """Dispatch-Detail für die Dunkelflaute, pro Szenario."""
    os.makedirs(out_dir, exist_ok=True)

    dispatch_carriers_order = [
        "nuclear", "ror", "hydro", "PHS", "biomass", "biogas",
        "onwind", "offwind-ac", "offwind-dc", "solar", "solar rooftop",
        "battery discharger", "home battery discharger",
        "H2 turbine", "H2 Fuel Cell", "OCGT methanol",
        "CCGT", "OCGT", "coal", "lignite", "oil",
    ]

    for label, n in networks.items():
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        start, end, peak = get_df_window(n)

        for country in ["ALL", "DE"]:
            # Erzeugung pro Carrier
            gen_series = {}
            for carrier in dispatch_carriers_order:
                gens = n.generators[n.generators.carrier == carrier]
                if country != "ALL":
                    gens = gens[gens.bus.str.startswith(country)]
                cols = [c for c in gens.index if c in n.generators_t.p.columns]
                if cols:
                    s = n.generators_t.p[cols].sum(axis=1).loc[start:end] / 1e3  # GW
                    if s.abs().max() > 0.01:
                        gen_series[carrier] = s

            # Batterie Discharger (Links)
            for link_carrier in ["battery discharger", "home battery discharger",
                                 "H2 turbine", "H2 Fuel Cell", "OCGT methanol"]:
                lks = n.links[n.links.carrier == link_carrier]
                if country != "ALL":
                    # bus1 ist der AC-Bus
                    lks = lks[lks.bus1.str.startswith(country)]
                cols = [c for c in lks.index if c in n.links_t.p0.columns]
                if cols:
                    # p0 ist negativ für Entladung (bus0=store, bus1=AC)
                    # → Einspeisung ins Netz = -p0 * efficiency oder |p1|
                    p1_cols = [c for c in cols if c in n.links_t.p1.columns]
                    if p1_cols:
                        s = -n.links_t.p1[p1_cols].sum(axis=1).loc[start:end] / 1e3
                    else:
                        s = -n.links_t.p0[cols].sum(axis=1).loc[start:end] / 1e3
                    if s.abs().max() > 0.01:
                        gen_series[link_carrier] = s.clip(lower=0)

            if not gen_series:
                continue

            # Last
            loads = n.loads_t.p_set
            if country != "ALL":
                load_cols = [c for c in loads.columns
                             if c.startswith(country) and "heat" not in c.lower()]
            else:
                load_cols = [c for c in loads.columns if "heat" not in c.lower()]
            load_total = loads[load_cols].sum(axis=1).loc[start:end] / 1e3 if load_cols else None

            # Stapel-Plot
            fig, ax = plt.subplots(figsize=(14, 6))
            ordered = [c for c in dispatch_carriers_order if c in gen_series]
            vals = np.array([gen_series[c].values for c in ordered])
            colors = [CARRIER_COLORS.get(c, "#999") for c in ordered]
            labels = [CARRIER_LABELS.get(c, c) for c in ordered]
            times = gen_series[ordered[0]].index

            ax.stackplot(times, vals, colors=colors, labels=labels, alpha=0.85)
            if load_total is not None:
                ax.plot(load_total.index, load_total.values,
                        "k-", linewidth=1.5, label="Last (el.)")

            # Dunkelflaute-Peak markieren
            if start <= peak <= end:
                ax.axvline(peak, color="red", linestyle="--", alpha=0.7, linewidth=1)
                ax.text(peak, ax.get_ylim()[1]*0.95, "DF-Peak",
                        ha="center", fontsize=8, color="red")

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            plt.xticks(rotation=45)
            ax.set_ylabel("Leistung (GW)")
            ax.set_title(f"Dispatch Dunkelflaute — {country} — {label}")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, ncol=1)
            ax.set_xlim(times[0], times[-1])
            fig.tight_layout()

            fname = f"dispatch_dunkelflaute_{country}_{safe_label}.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✅ {fname}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 4: LS-Vergleich über die Szenarien
# ═══════════════════════════════════════════════════════════════════════

def plot_ls_comparison(networks, out_dir):
    """Vergleiche Load Shedding über alle Szenarien."""
    os.makedirs(out_dir, exist_ok=True)

    # a) Gesamt-LS pro Szenario als Balken
    ls_totals = {}
    ls_peaks = {}
    ls_hours = {}
    for label, n in networks.items():
        ls = get_ls_data(n, "ALL")
        ls_totals[label] = ls.sum() * 2 / 1e6  # TWh (2h Auflösung)
        ls_peaks[label] = ls.max() / 1e3  # GW
        ls_hours[label] = (ls > 1).sum()  # Stunden mit LS > 1 MW

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    labels = list(ls_totals.keys())
    x = range(len(labels))

    axes[0].bar(x, [ls_totals[l] for l in labels], color=["#4a90d9", "#d4553a", "#2ca02c"])
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel("TWh"); axes[0].set_title("LS Gesamt (TWh)")

    axes[1].bar(x, [ls_peaks[l] for l in labels], color=["#4a90d9", "#d4553a", "#2ca02c"])
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel("GW"); axes[1].set_title("LS Spitze (GW)")

    axes[2].bar(x, [ls_hours[l] for l in labels], color=["#4a90d9", "#d4553a", "#2ca02c"])
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[2].set_ylabel("Stunden"); axes[2].set_title("LS-Stunden (>1 MW)")

    fig.suptitle("Load Shedding — Szenariovergleich (EU)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ls_comparison_scenarios.png"), dpi=150)
    plt.close(fig)
    print(f"  ✅ ls_comparison_scenarios.png")

    # b) LS pro Land pro Szenario
    ls_by_country = {}
    for label, n in networks.items():
        country_ls = {}
        for c in COUNTRIES[1:]:  # ohne ALL
            ls = get_ls_data(n, c)
            total = ls.sum() * 2 / 1e6
            if total > 0.0001:
                country_ls[c] = total
        ls_by_country[label] = country_ls

    df = pd.DataFrame(ls_by_country).fillna(0)
    if not df.empty and df.sum().sum() > 0:
        df = df[df.max(axis=1) > 0]  # Nur Länder mit LS
        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(kind="bar", ax=ax)
        ax.set_ylabel("Load Shedding (TWh)")
        ax.set_title("Load Shedding pro Land und Szenario")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "ls_by_country_scenarios.png"), dpi=150)
        plt.close(fig)
        print(f"  ✅ ls_by_country_scenarios.png")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 5: Curtailment-Vergleich
# ═══════════════════════════════════════════════════════════════════════

def plot_curtailment_comparison(networks, basis_n, out_dir):
    """Vergleiche Curtailment pro Carrier über alle Szenarien + Basis."""
    os.makedirs(out_dir, exist_ok=True)

    re_carriers = ["onwind", "offwind-ac", "solar", "solar rooftop"]
    all_nets = {"Basisrun": basis_n}
    all_nets.update(networks)

    rows = []
    for run_label, n in all_nets.items():
        for carrier in re_carriers:
            gens = n.generators[n.generators.carrier == carrier]
            cols_p  = [c for c in gens.index if c in n.generators_t.p.columns]
            cols_pu = [c for c in gens.index if c in n.generators_t.p_max_pu.columns]
            if cols_p and cols_pu:
                actual    = n.generators_t.p[cols_p].sum().sum() * 2 / 1e6
                potential = (n.generators_t.p_max_pu[cols_pu] *
                             gens.loc[cols_pu, "p_nom_opt"]).sum().sum() * 2 / 1e6
                curtailed = potential - actual
                pct = curtailed / potential * 100 if potential > 0 else 0
                rows.append({"Run": run_label, "Carrier": carrier,
                             "Curtailed_TWh": curtailed, "Pct": pct})

    df = pd.DataFrame(rows)
    if df.empty:
        print("  ⚠️ Keine Curtailment-Daten")
        return

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    runs = df["Run"].unique()
    carriers = df["Carrier"].unique()
    x = np.arange(len(carriers))
    width = 0.8 / len(runs)

    for i, run in enumerate(runs):
        sub = df[df["Run"] == run]
        vals = [sub[sub["Carrier"]==c]["Pct"].values[0]
                if len(sub[sub["Carrier"]==c]) > 0 else 0 for c in carriers]
        ax.bar(x + i*width, vals, width, label=run, alpha=0.85)

    ax.set_xticks(x + width*(len(runs)-1)/2)
    ax.set_xticklabels([CARRIER_LABELS.get(c, c) for c in carriers])
    ax.set_ylabel("Curtailment (%)")
    ax.set_title("Abregelung nach Carrier und Szenario (EU)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curtailment_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  ✅ curtailment_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    import pypsa

    print("=" * 60)
    print("SUPPLEMENTARY PLOTS — ARO-Auswertung")
    print("=" * 60)

    # ── Dispatch-Netzwerke laden ────────────────────────────────────
    dispatch_files = get_dispatch_files()
    print(f"\nGefunden: {len(dispatch_files)} Dispatch-Netzwerke:")
    for label, path in dispatch_files.items():
        print(f"  {label}: {os.path.basename(path)}")

    networks = {}
    for label, path in dispatch_files.items():
        print(f"\nLade {label}...")
        networks[label] = load_network(path)

    # ── Basisrun laden ─────────────────────────────────────────────
    basis_path = os.path.join(BASIS_DIR, "networks", "base_s_24___2050.nc")
    print(f"\nLade Basisrun: {os.path.basename(basis_path)}")
    basis_n = load_network(basis_path)

    # ── Plot 1: Energy Generated pro Szenario ──────────────────────
    print("\n" + "─" * 60)
    print("1. Energy Generated pro Szenario")
    out1 = os.path.join(PLOT_BASE, "supplementary", "energy_per_scenario")
    plot_energy_per_scenario(networks, out1)

    # ── Plot 2: Energy Vergleich ───────────────────────────────────
    print("\n" + "─" * 60)
    print("2. Energy Generated Vergleich (Szenarien + Basisrun)")
    out2 = os.path.join(PLOT_BASE, "supplementary", "energy_comparison")
    plot_energy_comparison(networks, basis_n, out2)

    # ── Plot 3: DF-Detail-Dispatch ────────────────────────────────
    print("\n" + "─" * 60)
    print("3. Dunkelflaute-Detail-Dispatch (7-Tage-Zoom)")
    out3 = os.path.join(PLOT_BASE, "supplementary", "df_detail_dispatch")
    plot_df_detail_dispatch(networks, out3)

    # ── Plot 4: LS-Vergleich ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("4. Load Shedding — Szenariovergleich")
    out4 = os.path.join(PLOT_BASE, "supplementary", "ls_comparison")
    plot_ls_comparison(networks, out4)

    # ── Plot 5: Curtailment-Vergleich ─────────────────────────────
    print("\n" + "─" * 60)
    print("5. Curtailment — Szenariovergleich")
    out5 = os.path.join(PLOT_BASE, "supplementary", "curtailment_comparison")
    plot_curtailment_comparison(networks, basis_n, out5)

    print("\n" + "=" * 60)
    print(f"FERTIG. Alle Plots unter: {PLOT_BASE}/supplementary/")
    print("=" * 60)


if __name__ == "__main__":
    main()