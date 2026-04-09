#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_delta_prices_map.py
========================
Delta-Preiskarte: Dunkelflaute-Periode vs. Referenzszenario.

Modi
----
mode='both':  Klassisch – zwei separate Netzlisten ('average' und 'dunkelflaute')
              aus PlottingConfig.get_networks().  Wie bisher.

mode='aro':   ARO-kompatibel – ein einzelnes Dispatch-Netzwerk (Worst-Case oder
              beliebiges Szenario).  Das Skript erkennt die Dunkelflaute-Periode
              intern (START/END) und berechnet den Jahresdurchschnitt aus den
              verbleibenden Snapshots.

Aufruf im ARO-Runner
--------------------
    mod.main(aro_network="/.../dispatch_..._worst_case_std.nc")

Standalone-Aufruf
-----------------
    python plot_delta_prices_map.py                         # mode=both
    python plot_delta_prices_map.py --aro path/to/net.nc    # mode=aro
"""

import argparse
import os

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from config_final import PlottingConfig

# =====================================================================
# PARAMETER
# =====================================================================
# Dunkelflaute-Fenster (innerhalb des Netzwerk-Zeitraums)
START = "2005-01-07"
END   = "2005-01-28"

SAVE_STEM = "price_delta_map_Jan7_3weeks_final"

# Modell-Länder
MODEL_COUNTRIES = [
    "DE", "FR", "PL", "CZ", "AT", "IT", "ES", "BE", "NL",
    "DK", "SE", "PT", "FI", "LT", "LV", "EE",
    "NO", "GB", "CH",
]

NAME_TO_ISO = {
    "France":         "FR",
    "Norway":         "NO",
    "United Kingdom": "GB",
    "Germany":        "DE",
    "Spain":          "ES",
}


# =====================================================================
# HILFSFUNKTIONEN
# =====================================================================

def extract_year(path: str) -> int:
    return int(path.split("___")[-1].split(".")[0])


def normalize_country_code(code: str | None) -> str | None:
    if code is None:
        return None
    code = code.upper()
    if code in MODEL_COUNTRIES:
        return code
    for pref in MODEL_COUNTRIES:
        if code.startswith(pref):
            return pref
    return None


def get_country_iso(rec) -> str | None:
    iso_a2 = rec.attributes.get("ISO_A2", "").strip()
    name   = rec.attributes.get("NAME",   "").strip()
    if iso_a2 in MODEL_COUNTRIES:
        return iso_a2
    if name in NAME_TO_ISO:
        return NAME_TO_ISO[name]
    iso_a3 = rec.attributes.get("ISO_A3", "")
    if iso_a3 == "FRA": return "FR"
    if iso_a3 == "NOR": return "NO"
    return None


def get_total_electric_load(network: pypsa.Network, start: str, end: str) -> pd.DataFrame:
    """Gesamte elektrische Last: statische Loads + Wärmepumpen/Heizstäbe."""
    loads_t = network.loads_t.p_set.loc[start:end]
    loads_t = loads_t.reindex(columns=network.loads.index).fillna(network.loads.p_set)
    total = loads_t.groupby(network.loads.bus, axis=1).sum()

    if "carrier" in network.links.columns:
        mask = network.links.carrier.str.contains(
            "heat pump|resistive heater", case=False, regex=True
        )
        heat_links = network.links.index[mask]
        if not heat_links.empty:
            lp0 = network.links_t.p0.loc[start:end, heat_links]
            bus_map = network.links.loc[heat_links, "bus0"]
            total = total.add(lp0.groupby(bus_map, axis=1).sum(), fill_value=0)
            print(f"   -> {len(heat_links)} Wärmepumpen/Heizstäbe zur Last addiert.")
    return total


def _country_prices(network: pypsa.Network, start: str, end: str) -> tuple[pd.Series, pd.Series]:
    """
    Gibt (vwap_series, simple_series) zurück – jeweils länderbezogen.
    """
    prices = network.buses_t.marginal_price.loc[start:end]
    total_load = get_total_electric_load(network, start, end)
    common = prices.columns.intersection(total_load.columns)

    group_key = lambda x: x[:2]

    # VWAP
    p_aln = prices[common]
    l_aln = total_load[common]
    revenue      = (p_aln * l_aln).groupby(group_key, axis=1).sum().sum()
    load_sum     = l_aln.groupby(group_key, axis=1).sum().sum()
    vwap         = revenue / load_sum.replace(0, np.nan)

    # Simple average
    simple = prices.groupby(group_key, axis=1).mean().mean()

    return vwap, simple


def _infer_year(network: pypsa.Network, path: str) -> int | None:
    """Jahr aus Snapshot oder Pfad."""
    try:
        snaps = pd.to_datetime(network.snapshots)
        return int(snaps[0].year)
    except Exception:
        pass
    import re
    m = re.search(r"___(\\d{4})\\.nc$", path) or re.search(r"_(\\d{4})\\.nc$", path)
    return int(m.group(1)) if m else None


# =====================================================================
# PLOT-FUNKTION
# =====================================================================

def plot_delta_map(
    delta_series: pd.Series,
    year: int | str,
    suffix: str,
    output_dir: str,
    records,
    title_override: str | None = None,
    vmin: float = -200,
    vmax: float =  200,
    tick_step: float = 50,
) -> None:
    delta_dict = {}
    for bus_code, val in delta_series.items():
        iso = normalize_country_code(str(bus_code))
        if iso in MODEL_COUNTRIES:
            delta_dict[iso] = float(val)

    if title_override:
        main_title = title_override
    elif "weighted" in suffix:
        main_title = "Differenz der lastgewichteten Strompreise (VWAP)"
    else:
        main_title = "Differenz der mittleren Strompreise (Time-Weighted)"

    TITLE = f"{main_title}\nDunkelflaute vs. Referenzszenario {year} (07. – 28. Jan.)"

    fig = plt.figure(figsize=(10, 9))
    projection = ccrs.LambertConformal(central_longitude=12, central_latitude=54)
    ax = plt.axes(projection=projection, frameon=False)
    ax.set_extent([-12, 35, 35, 72], crs=ccrs.PlateCarree())
    ax.set_title(TITLE, fontsize=18, pad=15, linespacing=1.4)
    ax.spines["geo"].set_visible(False)

    cmap = plt.colormaps["RdYlBu_r"]

    def get_color(val):
        if val is None or np.isnan(val):
            return "#e0e0e0"
        return cmap(np.clip((val - vmin) / (vmax - vmin), 0, 1))

    colored = []
    for rec in records:
        iso = get_country_iso(rec)
        if not iso or iso not in MODEL_COUNTRIES:
            continue
        val = delta_dict.get(iso, None)
        color = get_color(val)
        if val is not None and not np.isnan(val):
            colored.append((iso, val))
        ax.add_geometries(
            [rec.geometry], ccrs.PlateCarree(),
            facecolor=color, edgecolor="white", linewidth=0.8, alpha=0.9,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02, fraction=0.046)
    cb.set_label("Preisdifferenz (EUR/MWh)", fontsize=16, labelpad=10)
    ticks = np.arange(vmin, vmax + 0.1, tick_step)
    cb.set_ticks(ticks)
    cb.set_ticklabels(
        [f"{int(t):+}" if t != 0 else "0" for t in ticks], fontsize=14
    )
    cb.outline.set_visible(False)

    plt.tight_layout(pad=0.5)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"A_{SAVE_STEM}_{year}{suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"✅ {len(colored)} Länder gespeichert: {save_path}")


# =====================================================================
# KERNLOGIK – ARO-MODUS
# =====================================================================

def _run_aro_mode(aro_network_path: str) -> None:
    """
    ARO-kompatibler Pfad: Ein einzelnes Dispatch-Netzwerk.

    Das Skript erkennt die Dunkelflaute-Periode (START/END) innerhalb der
    Snapshots des Netzwerks und berechnet den Jahresdurchschnitt aus allen
    übrigen Stunden.  Delta = DF-Periode − Jahresdurchschnitt.
    """
    cfg = PlottingConfig()
    output_dir = cfg.PLOT_OUTPUT_PATH

    print(f"[ARO] Lade Netzwerk: {aro_network_path}")
    n = pypsa.Network(aro_network_path)
    n.snapshots = pd.to_datetime(n.snapshots)

    year = _infer_year(n, aro_network_path)
    year_label = str(year) if year is not None else "ARO"

    # Zeitraum-Detektion: START/END müssen im Netzwerk vorhanden sein
    # → Falls nicht, versuche Jahr-Anpassung
    snaps = n.snapshots
    year_start = snaps[0].year
    year_end   = snaps[-1].year

    def _adjust_date(date_str: str, target_year: int) -> str:
        """Ersetzt das Jahr in einem Datumsstring durch target_year."""
        return f"{target_year}{date_str[4:]}"

    df_start = START
    df_end   = END

    # Prüfe ob START/END im Netzwerk-Zeitraum liegt
    snap_idx = pd.DatetimeIndex(snaps)
    _start_ts = pd.Timestamp(START)
    _end_ts   = pd.Timestamp(END)

    if _start_ts not in snap_idx or (_start_ts < snap_idx[0]):
        # Passe Jahr an
        df_start = _adjust_date(START, year_start)
        df_end   = _adjust_date(END,   year_start)
        print(f"[ARO] START/END auf Netzwerkjahr angepasst: {df_start} – {df_end}")

    df_mask = (snap_idx >= pd.Timestamp(df_start)) & (snap_idx <= pd.Timestamp(df_end))

    if not df_mask.any():
        print(
            f"⚠️  [ARO] Keine Snapshots im DF-Fenster {df_start}–{df_end} gefunden.\n"
            f"     Verwende stattdessen erste 21 Tage des Netzwerks als DF-Proxy."
        )
        df_mask = pd.Series(False, index=range(len(snaps)))
        df_mask.iloc[:21 * 24] = True
        df_mask = df_mask.values

    avg_mask = ~df_mask

    df_snaps  = snap_idx[df_mask]
    avg_snaps = snap_idx[avg_mask]

    if len(df_snaps) == 0 or len(avg_snaps) == 0:
        print("⚠️  [ARO] Zu wenig Snapshots für DF-Vergleich – überspringe.")
        return

    print(
        f"[ARO] DF-Periode: {df_snaps[0]} – {df_snaps[-1]} ({len(df_snaps)} h)\n"
        f"[ARO] Referenz:   {avg_snaps[0]} – {avg_snaps[-1]} ({len(avg_snaps)} h)"
    )

    # Shapefile laden
    print("[ARO] Lade Natural Earth Shapefile (10m)...")
    shp_path = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    records = list(shpreader.Reader(shp_path).records())

    # Preisberechnung
    vwap_df,   simple_df  = _country_prices(
        n,
        str(df_snaps[0]),
        str(df_snaps[-1]),
    )
    vwap_avg, simple_avg = _country_prices(
        n,
        str(avg_snaps[0]),
        str(avg_snaps[-1]),
    )

    delta_vwap   = vwap_df   - vwap_avg
    delta_simple = simple_df - simple_avg

    print(f"\n[ARO] Delta VWAP {year_label} (Auszug):")
    for c, v in delta_vwap.items():
        if normalize_country_code(str(c)) in ["DE", "FR", "ES"]:
            print(f"  {c:2s}: {v:6.1f} EUR/MWh")

    plot_delta_map(
        delta_vwap, year_label, "_aro_weighted_total", output_dir, records,
        title_override="Δ lastgewichtete Preise – DF vs. Jahresschnitt (ARO)",
        vmin=-200, vmax=200, tick_step=50,
    )
    plot_delta_map(
        delta_simple, year_label, "_aro_simple_average", output_dir, records,
        title_override="Δ mittlere Marktwerte – DF vs. Jahresschnitt (ARO)",
        vmin=-50, vmax=50, tick_step=10,
    )

    print("[ARO] Delta-Preiskarte fertig.")


# =====================================================================
# KERNLOGIK – BOTH-MODUS (klassisch)
# =====================================================================

def _run_both_mode() -> None:
    """Klassischer Modus: zwei Netzlisten aus PlottingConfig (average + dunkelflaute)."""
    cfg      = PlottingConfig()
    networks = cfg.get_networks()
    output_dir = cfg.PLOT_OUTPUT_PATH

    if not isinstance(networks, dict) or "average" not in networks or "dunkelflaute" not in networks:
        print(
            "⚠️ plot_delta_prices_map übersprungen: benötigt mode='both' "
            "mit Keys 'average' und 'dunkelflaute'."
        )
        return

    base_paths = networks["average"]
    df_paths   = networks["dunkelflaute"]

    base_years = {extract_year(p): p for p in base_paths}
    df_years   = {extract_year(p): p for p in df_paths}
    common_years = sorted(set(base_years) & set(df_years))
    print("Gemeinsame Planungsjahre:", common_years)

    print("Lade Natural Earth Shapefile (10m)...")
    shp_path = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    records = list(shpreader.Reader(shp_path).records())

    for year in common_years:
        print(f"\n{'=' * 60}\n=== Jahr {year} ===")

        n_avg = pypsa.Network(base_years[year])
        n_df  = pypsa.Network(df_years[year])
        n_avg.snapshots = pd.to_datetime(n_avg.snapshots)
        n_df.snapshots  = pd.to_datetime(n_df.snapshots)

        avg_vwap,   avg_simple  = _country_prices(n_avg, START, END)
        df_vwap,    df_simple   = _country_prices(n_df,  START, END)

        delta_vwap   = df_vwap   - avg_vwap
        delta_simple = df_simple - avg_simple

        print(f"\nWeighted Price Delta {year} (Auszug):")
        for c, v in delta_vwap.items():
            if normalize_country_code(str(c)) in ["DE", "FR", "ES"]:
                print(f"  {c:2s}: {v:6.1f}")

        plot_delta_map(
            delta_vwap,   year, "_weighted_total",  output_dir, records,
            title_override="Differenz lastgewichtete Preise (VWAP)",
            vmin=-200, vmax=200, tick_step=50,
        )
        plot_delta_map(
            delta_simple, year, "_simple_average",  output_dir, records,
            title_override="Differenz mittlere Marktwerte (Time-Weighted)",
            vmin=-50, vmax=50, tick_step=10,
        )

    print("\nALLE PLOTS FERTIG!")


# =====================================================================
# MAIN
# =====================================================================

def main(aro_network: str | None = None) -> None:
    """
    Entry point für run_analysis.py (standalone dispatcher).

    Parameters
    ----------
    aro_network : str | None
        Pfad zum ARO-Dispatch-Netzwerk.  Wenn angegeben, wird der
        ARO-Modus aktiviert (ein Netz, interner DF-Vergleich).
        Wenn None, wird der klassische 'both'-Modus verwendet.
    """
    if aro_network is not None:
        _run_aro_mode(aro_network)
    else:
        _run_both_mode()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--aro", metavar="PATH", default=None,
        help="Pfad zum ARO-Dispatch-Netzwerk (aktiviert ARO-Modus)",
    )
    args = ap.parse_args()
    main(aro_network=args.aro)
