#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot: Residuallast über das Jahr & Dunkelflautenperiode
=======================================================

Erstellt für jedes Land und jeden Planungshorizont:
1️⃣ Zeitreihe der Residuallast (12. Jan – 12. Feb, stündlich)
2️⃣ Jahresverlauf der Residuallast (Tagesmittelwerte)
3️⃣ Ausgabe der 24 Stunden mit höchster Residuallast
"""

import os
import re
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from config_final import PlottingConfig


# =====================================================================
# --- Zeitraumdefinition ---
# =====================================================================
DETAIL_START = "2005-01-12"
DETAIL_END   = "2005-02-12"
YEAR_START   = "2005-01-01"
YEAR_END     = "2005-12-31"


# =====================================================================
# --- Berechnung der Residuallast ---
# =====================================================================
def get_residual_load(n: pypsa.Network, country: str) -> pd.Series:
    """
    Berechnet Residuallast = Last - (Wind + Solar + Laufwasser + Speicheraustrag)
    Gibt eine Zeitreihe (GW) zurück.
    """
    # === Lastdaten ===
    if country == "ALL":
        load = n.loads_t.p_set.sum(axis=1)
    else:
        load_ids = n.loads.index[n.loads.bus.str.startswith(country)]
        load = n.loads_t.p_set[load_ids].sum(axis=1)

    # === Erzeugung Wind/Solar/Wasser ===
    def gen_sum(carrier_list):
        if country == "ALL":
            gens = n.generators[n.generators.carrier.isin(carrier_list)]
        else:
            gens = n.generators[
                n.generators.carrier.isin(carrier_list)
                & n.generators.bus.str.startswith(country)
            ]
        if gens.empty:
            return pd.Series(0, index=n.snapshots)
        return n.generators_t.p[gens.index].sum(axis=1)

    renew_gen = gen_sum(["solar", "solar-hsat", "onwind", "offwind-ac", "offwind-dc", "ror", "hydro"])

    # === Batterie- und Pumpspeicher-Entladung ===
    discharge = pd.Series(0.0, index=n.snapshots)
    if not n.storage_units.empty:
        dis_units = n.storage_units[n.storage_units.carrier.isin(["PHS", "battery discharger", "battery"])]
        if country != "ALL":
            dis_units = dis_units[dis_units.bus.str.startswith(country)]
        if not dis_units.empty:
            discharge += n.storage_units_t.p[dis_units.index].clip(lower=0).sum(axis=1)
    if not n.links.empty:
        link_mask = n.links.carrier.str.contains("discharger|PHS", case=False, na=False)
        dis_links = n.links[link_mask]
        if country != "ALL":
            dis_links = dis_links[
                dis_links.bus0.str.startswith(country)
                | dis_links.bus1.str.startswith(country)
            ]
        for name in dis_links.index:
            if name in n.links_t.p0:
                discharge += n.links_t.p0[name].clip(lower=0)

    # === Residuallast ===
    residual_load = load - (renew_gen + discharge)
    return residual_load / 1e3  # MW → GW



# =====================================================================
# --- Plotfunktion ---
# =====================================================================
def plot_residual_load(res_load: pd.Series, year_label: int, country: str, config: PlottingConfig, network_path: str):
    """Erstellt Jahres- und Detailplot der Residuallast."""
    title_country = "Gesamtnetz" if country == "ALL" else country
    network_name = os.path.basename(os.path.dirname(os.path.dirname(network_path)))
    save_dir = os.path.join(config.BASE_SAVE_PATH, network_name, "residual_load")
    os.makedirs(save_dir, exist_ok=True)

    # --- 1️⃣ Detailzeitraum (Jan–Feb) ---
    df_detail = res_load.loc[DETAIL_START:DETAIL_END]
    fig, ax = plt.subplots(figsize=(13, 5))
    df_detail.plot(ax=ax, color="#1f77b4", linewidth=1.5)
    ax.set_ylabel("Residuallast [GW]", fontsize=config.FONT_SIZES["label"])
    ax.set_title(f"Residuallast ({title_country}, {year_label}) – {DETAIL_START} bis {DETAIL_END}",
                 fontsize=config.FONT_SIZES["title"])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path_detail = os.path.join(save_dir, f"residual_load_detail_{country}_{year_label}.png")
    plt.savefig(save_path_detail, dpi=300)
    plt.close(fig)
    print(f"✅ Detailplot gespeichert: {save_path_detail}")

    # --- 2️⃣ Jahresverlauf (Tagesmittel) ---
    df_year = res_load.loc[YEAR_START:YEAR_END].resample("1D").mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    df_year.plot(ax=ax, color="#ff7f0e", linewidth=1.5)
    ax.set_ylabel("Residuallast [GW]", fontsize=config.FONT_SIZES["label"])
    ax.set_title(f"Jahresverlauf der Residuallast ({title_country}, {year_label})",
                 fontsize=config.FONT_SIZES["title"])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path_year = os.path.join(save_dir, f"residual_load_year_{country}_{year_label}.png")
    plt.savefig(save_path_year, dpi=300)
    plt.close(fig)
    print(f"✅ Jahresplot gespeichert: {save_path_year}")

    # --- 3️⃣ 24 Stunden mit höchster Residuallast ---
    top24 = res_load.sort_values(ascending=False).head(24)
    print(f"\n📊 Höchste 24 Stunden Residuallast – {title_country} ({year_label}):")
    print(top24.round(2).to_string())

    # --- 4️⃣ Detailtabelle: 25.–29. Januar ---
    jan_start = "2005-01-25"
    jan_end   = "2005-01-29"
    jan_window = res_load.loc[jan_start:jan_end]

    if not jan_window.empty:
        jan_sorted = jan_window.sort_values(ascending=False)
        top5_idx = set(jan_sorted.head(5).index)

        print(f"\n📆 Residuallast {jan_start}–{jan_end} – {title_country} ({year_label}):")
        print("-------------------------------------------------------------")
        print("Zeitpunkt                Residuallast [GW]")
        print("-------------------------------------------------------------")
        for ts, val in jan_window.items():
            mark = "★" if ts in top5_idx else " "
            print(f"{ts:%Y-%m-%d %H:%M:%S}   {val:8.2f} {mark}")
        print("-------------------------------------------------------------")
        print("★ = 5 höchste Werte innerhalb des Zeitraums\n")



# =====================================================================
# --- Hauptablauf ---
# =====================================================================
def main():
    config = PlottingConfig()
    networks = config.get_networks()

    for path in networks:
        if not os.path.isfile(path):
            print(f"⚠️ Datei nicht gefunden: {path}")
            continue

        m = re.search(r"_(\d{4})\.nc$", path)
        if not m:
            continue
        year = int(m.group(1))

        print(f"\n📂 Lade Netzwerk {year}: {path}")
        n = pypsa.Network(path)

        print(f"\n📊 Debug: Ladezeitreihe für DE (Last, Wind, Solar) ...")

        # === Gesamtlast (nur elektrische Busse für DE) ===
        total_load = n.loads_t.p_set.loc[:, n.loads.bus.str.startswith("DE")].sum(axis=1)

        # === Windproduktion (Onshore + Offshore) ===
        wind = n.generators_t.p.loc[
               :, n.generators.carrier.str.contains("wind", case=False) & n.generators.bus.str.startswith("DE")
               ].sum(axis=1)

        # === Solarproduktion ===
        solar = n.generators_t.p.loc[
                :, n.generators.carrier.str.contains("solar", case=False) & n.generators.bus.str.startswith("DE")
                ].sum(axis=1)

        # === Residuallast ===
        residual_load = total_load - (wind + solar)

        # === Alles zusammenfassen ===
        df = pd.DataFrame({
            "load": total_load,
            "wind": wind,
            "solar": solar,
            "residual_load": residual_load
        })

        # === Zeitraum 25.–29. Januar filtern ===
        subset = df.loc["2005-01-25":"2005-01-29"]
        print(subset.head(10))  # nur zum Testen

        # === Plot für Vergleich ===
        subset.plot(y=["load", "wind", "solar", "residual_load"], figsize=(10, 5))
        plt.title("Deutschland: Last, Wind, Solar und Residuallast (25.–29. Jan 2005)")
        plt.ylabel("Leistung [GW]")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        for country in config.get_countries():
            print(f"\n📊 Debug: Ladezeitreihe für DE (Last, Wind, Solar) ...")

            # === Gesamtlast (nur elektrische Busse für DE) ===
            total_load = n.loads_t.p_set.loc[:, n.loads.bus.str.startswith("DE")].sum(axis=1)

            # === Windproduktion (Onshore + Offshore) ===
            wind = n.generators_t.p.loc[
                   :, n.generators.carrier.str.contains("wind", case=False) & n.generators.bus.str.startswith("DE")
                   ].sum(axis=1)

            # === Solarproduktion ===
            solar = n.generators_t.p.loc[
                    :, n.generators.carrier.str.contains("solar", case=False) & n.generators.bus.str.startswith("DE")
                    ].sum(axis=1)

            # === Residuallast ===
            residual_load = total_load - (wind + solar)

            # === Alles zusammenfassen ===
            df = pd.DataFrame({
                "load": total_load,
                "wind": wind,
                "solar": solar,
                "residual_load": residual_load
            })

            # === Zeitraum 25.–29. Januar filtern ===
            subset = df.loc["2005-01-26":"2005-01-28"]
            print(subset.head(10))  # nur zum Testen

            # === Plot für Vergleich ===
            subset.plot(y=["load", "wind", "solar", "residual_load"], figsize=(10, 5))
            plt.title("Deutschland: Last, Wind, Solar und Residuallast (25.–29. Jan 2005)")
            plt.ylabel("Leistung [GW]")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            print(f"\n🔹 Berechne Residuallast für {country} ...")
            res_load = get_residual_load(n, country)
            plot_residual_load(res_load, year, country, config, path)


if __name__ == "__main__":
    main()
