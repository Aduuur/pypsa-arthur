#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_heat_network_analysis.py
=====================================
Vollständige Wärmenetz-Auswertung für PyPSA-Eur (Horizont 2050)

Komponenten:
  - Installierte Kapazitäten (Links + Stores) nach Wärmetechnologie
  - Jährliche Wärmebereitstellung [TWh_th] nach Träger
  - Thermische Speicher: Kapazität, Jahresenergie, Lade-/Entladeprofile
  - COP-Zeitreihen für Wärmepumpen (falls verfügbar)
  - Wärmebilanz: Erzeugung vs. Nachfrage pro Land
  - ARO-Modus: Auswertung von Robust-Netz + allen Worst-Case-Szenarien

Pipeline-Kompatibilität:
  - Importiert PlottingConfig / AROPlottingConfig aus config_final.py
  - Liest Netzwerk-Pfade aus master_config.MASTER_CONFIG
  - Erkennt run_type automatisch ("normal" vs. "aro")
  - Ausgabe in plots_base/heat_analysis/<scenario>/ bzw.
              aro_plots_base/<run>/<scenario>/heat_analysis/

Aufruf:
  python plot_heat_network_analysis.py
  RUN_NAME=my-aro-run python plot_heat_network_analysis.py
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")

from config_final import PlottingConfig, AROPlottingConfig, MasterConfig, MASTER_CONFIG

# ──────────────────────────────────────────────────────────────────────────────
# HEAT CARRIER DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
# Jede Eintrag: (label, carrier-substrings, Farbe)
# Reihenfolge bestimmt Stapelreihenfolge (Grundlast unten)
HEAT_LINK_CARRIERS: List[Tuple[str, List[str], str]] = [
    ("Groß-WP (Fernwärme)",  ["central heat pump"],            "#1f77b4"),
    ("Dezentrale WP",        ["heat pump"],                    "#00bfff"),  # fallback wenn kein 'central'
    ("Heizstab",             ["resistive heater"],              "#d62728"),
    ("Biomasse-Kessel",      ["biomass boiler", "solid biomass boiler"], "#2ca02c"),
    ("Gas-Kessel",           ["gas boiler"],                   "#ff7f0e"),
    ("Öl-Kessel",            ["oil boiler"],                   "#222222"),
    ("Abwärme / Geotherm.",  ["geothermal", "waste heat"],     "#ba91b1"),
    ("KWK (Gas)",            ["CHP", "combined heat"],         "#a85522"),
    ("Sonstige Wärme",       ["heat"],                         "#7f7f7f"),  # catch-all
]

HEAT_STORE_CARRIERS: List[Tuple[str, List[str], str]] = [
    ("Fernwärme-Speicher",   ["central water tank", "pit thermal", "central heat storage"], "#17becf"),
    ("Dezentraler Speicher", ["water tank", "decentral heat storage"],                      "#9edae5"),
]

HEAT_LOAD_CARRIERS: List[str] = ["heat", "urban heat", "rural heat", "decentral heat", "central heat"]


def _is_heat_bus(bus_name: str) -> bool:
    """Grobe Heuristik: Bus-Name enthält 'heat'."""
    return "heat" in bus_name.lower()


def _match_carrier(carrier: str, substrings: List[str]) -> bool:
    c = carrier.lower()
    return any(s.lower() in c for s in substrings)


# ──────────────────────────────────────────────────────────────────────────────
# CORE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

class HeatNetworkExtractor:
    """
    Extrahiert alle wärmerelevanten Daten aus einem pypsa.Network.
    Liefert normalisierte DataFrames unabhängig vom run_type.
    """

    def __init__(self, n: pypsa.Network, country: str = "ALL"):
        self.n = n
        self.country = country
        self._filter_prefix = None if country == "ALL" else country

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _country_links(self) -> pd.DataFrame:
        lks = self.n.links.copy()
        if self._filter_prefix:
            mask = (
                lks.bus0.str.startswith(self._filter_prefix) |
                lks.bus1.str.startswith(self._filter_prefix)
            )
            lks = lks[mask]
        return lks

    def _country_stores(self) -> pd.DataFrame:
        sts = self.n.stores.copy()
        if self._filter_prefix:
            sts = sts[sts.bus.str.startswith(self._filter_prefix)]
        return sts

    def _country_loads(self) -> pd.DataFrame:
        lds = self.n.loads.copy()
        if self._filter_prefix:
            lds = lds[lds.bus.str.startswith(self._filter_prefix)]
        return lds

    def _p_nom_opt(self, df: pd.DataFrame) -> pd.Series:
        col = "p_nom_opt" if "p_nom_opt" in df.columns else "p_nom"
        return df[col]

    def _e_nom_opt(self, df: pd.DataFrame) -> pd.Series:
        col = "e_nom_opt" if "e_nom_opt" in df.columns else "e_nom"
        return df[col]

    # ------------------------------------------------------------------
    # Installed capacities
    # ------------------------------------------------------------------

    def installed_link_capacities(self) -> pd.Series:
        """
        Installierte thermische Kapazitäten [GW_th] nach Label.
        Nutzt p_nom_opt * efficiency für Wärmeoutput (bus1).
        """
        lks = self._country_links()
        result = {}

        assigned = pd.Index([])

        for label, substrings, _ in HEAT_LINK_CARRIERS:
            # Spezialfall: dezentrale WP = heat pump OHNE 'central'
            if label == "Dezentrale WP":
                mask = (
                    lks.carrier.str.contains("heat pump", case=False) &
                    ~lks.carrier.str.contains("central", case=False)
                )
            else:
                mask = lks.carrier.apply(lambda c: _match_carrier(c, substrings))

            # catch-all nur für noch nicht zugewiesene
            if label == "Sonstige Wärme":
                mask = mask & ~lks.index.isin(assigned) & lks.bus1.apply(_is_heat_bus)

            subset = lks[mask & ~lks.index.isin(assigned)]
            if subset.empty:
                result[label] = 0.0
                continue

            assigned = assigned.union(subset.index)
            cap_mw = self._p_nom_opt(subset)
            # Wärme-Output: p_nom_opt * efficiency (COP-Näherung)
            if "efficiency" in subset.columns:
                heat_mw = (cap_mw * subset["efficiency"].fillna(1.0)).sum()
            else:
                heat_mw = cap_mw.sum()
            result[label] = heat_mw / 1000.0  # GW_th

        return pd.Series(result)

    def installed_storage_capacities(self) -> pd.Series:
        """
        Thermische Speicherkapazitäten [GWh_th] aus n.stores.
        """
        sts = self._country_stores()
        # Nur Stores mit Heat-Bus
        sts = sts[sts.bus.apply(_is_heat_bus)]
        result = {}

        assigned_mask = pd.Series(False, index=sts.index)
        for label, substrings, _ in HEAT_STORE_CARRIERS:
            mask = sts.carrier.apply(lambda c: _match_carrier(c, substrings))
            subset = sts[mask]
            result[label] = self._e_nom_opt(subset).sum() / 1000.0  # GWh_th
            assigned_mask |= mask

        rest = sts[~assigned_mask]
        result["Sonstige Wärmespeicher"] = self._e_nom_opt(rest).sum() / 1000.0

        return pd.Series(result)

    # ------------------------------------------------------------------
    # Annual heat generation [TWh_th]
    # ------------------------------------------------------------------

    def annual_heat_generation(self) -> pd.Series:
        """
        Jährliche Wärmeerzeugung [TWh_th] nach Label aus links_t.p1.
        p1 ist in PyPSA für outflow negativ → abs().
        """
        lks = self._country_links()
        result = {}
        assigned = pd.Index([])

        available = self.n.links_t.p1.columns

        for label, substrings, _ in HEAT_LINK_CARRIERS:
            if label == "Dezentrale WP":
                mask = (
                    lks.carrier.str.contains("heat pump", case=False) &
                    ~lks.carrier.str.contains("central", case=False)
                )
            else:
                mask = lks.carrier.apply(lambda c: _match_carrier(c, substrings))

            if label == "Sonstige Wärme":
                mask = mask & ~lks.index.isin(assigned) & lks.bus1.apply(_is_heat_bus)

            subset = lks[mask & ~lks.index.isin(assigned)]
            assigned = assigned.union(subset.index)

            idx = subset.index.intersection(available)
            if idx.empty:
                result[label] = 0.0
                continue

            gen_twh = self.n.links_t.p1[idx].abs().sum().sum() / 1e6
            result[label] = gen_twh

        return pd.Series(result)

    # ------------------------------------------------------------------
    # Heat demand [TWh_th]
    # ------------------------------------------------------------------

    def heat_demand(self) -> float:
        """Jährliche Wärmenachfrage [TWh_th] aus loads mit heat-Bus."""
        lds = self._country_loads()
        heat_loads = lds[lds.bus.apply(_is_heat_bus)]
        idx = heat_loads.index.intersection(self.n.loads_t.p.columns)
        if idx.empty:
            return 0.0
        return self.n.loads_t.p[idx].sum().sum() / 1e6

    # ------------------------------------------------------------------
    # Thermal storage dispatch
    # ------------------------------------------------------------------

    def storage_dispatch(self) -> pd.DataFrame:
        """
        Lade/Entladezeitreihe der thermischen Speicher [GW_th].
        """
        sts = self._country_stores()
        sts = sts[sts.bus.apply(_is_heat_bus)]

        df = pd.DataFrame(index=self.n.snapshots)
        for label, substrings, _ in HEAT_STORE_CARRIERS:
            mask = sts.carrier.apply(lambda c: _match_carrier(c, substrings))
            subset = sts[mask]
            idx = subset.index.intersection(self.n.stores_t.p.columns)
            if idx.empty:
                df[label] = 0.0
            else:
                df[label] = self.n.stores_t.p[idx].sum(axis=1) / 1000.0  # GW
        return df

    def storage_state_of_charge(self) -> pd.DataFrame:
        """State-of-Charge (e) thermischer Speicher [GWh_th]."""
        sts = self._country_stores()
        sts = sts[sts.bus.apply(_is_heat_bus)]

        df = pd.DataFrame(index=self.n.snapshots)
        for label, substrings, _ in HEAT_STORE_CARRIERS:
            mask = sts.carrier.apply(lambda c: _match_carrier(c, substrings))
            subset = sts[mask]
            idx = subset.index.intersection(self.n.stores_t.e.columns)
            if idx.empty:
                df[label] = 0.0
            else:
                df[label] = self.n.stores_t.e[idx].sum(axis=1) / 1000.0  # GWh
        return df

    # ------------------------------------------------------------------
    # Heat pump COP timeseries
    # ------------------------------------------------------------------

    def heat_pump_cop_timeseries(self) -> pd.DataFrame:
        """COP-Zeitreihen: ratio |p1|/|p0| für alle Wärmepumpen."""
        lks = self._country_links()
        hp_mask = lks.carrier.str.contains("heat pump", case=False)
        hp_links = lks[hp_mask]

        available_p0 = self.n.links_t.p0.columns
        available_p1 = self.n.links_t.p1.columns

        df = pd.DataFrame(index=self.n.snapshots)
        for name in hp_links.index:
            if name not in available_p0 or name not in available_p1:
                continue
            p0 = self.n.links_t.p0[name].abs()
            p1 = self.n.links_t.p1[name].abs()
            with np.errstate(divide="ignore", invalid="ignore"):
                cop = np.where(p0 > 0.01, p1 / p0, np.nan)
            label = hp_links.at[name, "carrier"]
            df[label + f" [{name}]"] = cop

        return df

    # ------------------------------------------------------------------
    # Full heat balance timeseries [GW_th]
    # ------------------------------------------------------------------

    def heat_balance_timeseries(self) -> pd.DataFrame:
        """
        Zeitreihe der Wärmebereitstellung [GW_th] nach Label + Speicher-Entladung.
        """
        lks = self._country_links()
        available = self.n.links_t.p1.columns
        df = pd.DataFrame(index=self.n.snapshots)

        assigned = pd.Index([])
        for label, substrings, _ in HEAT_LINK_CARRIERS:
            if label == "Dezentrale WP":
                mask = (
                    lks.carrier.str.contains("heat pump", case=False) &
                    ~lks.carrier.str.contains("central", case=False)
                )
            else:
                mask = lks.carrier.apply(lambda c: _match_carrier(c, substrings))

            if label == "Sonstige Wärme":
                mask = mask & ~lks.index.isin(assigned) & lks.bus1.apply(_is_heat_bus)

            subset = lks[mask & ~lks.index.isin(assigned)]
            assigned = assigned.union(subset.index)

            idx = subset.index.intersection(available)
            if idx.empty:
                df[label] = 0.0
            else:
                df[label] = self.n.links_t.p1[idx].abs().sum(axis=1) / 1000.0  # GW

        # Thermische Speicher-Entladung (positiv = Entladung)
        storage = self.storage_dispatch()
        for col in storage.columns:
            discharge = storage[col].clip(lower=0)
            if discharge.sum() > 0:
                df[col + " (Entladung)"] = discharge

        return df


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _carrier_color(label: str, config_colors: Dict[str, str]) -> str:
    """Sucht Farbe aus heat-carrier defs, dann config_colors, dann Fallback."""
    for lbl, _, color in HEAT_LINK_CARRIERS + HEAT_STORE_CARRIERS:
        if lbl == label:
            return color
    for key, color in config_colors.items():
        if key.lower() in label.lower():
            return color
    return "#aaaaaa"


def plot_installed_capacities(
    cap_links: pd.Series,
    cap_stores: pd.Series,
    country: str,
    scenario_label: str,
    out_dir: Path,
    config_colors: Dict[str, str],
    font_sizes: Dict[str, int],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, data, unit, title in zip(
        axes,
        [cap_links, cap_stores],
        ["GW_th", "GWh_th"],
        ["Installierte Wärmekapazität", "Thermische Speicherkapazität"],
    ):
        data_plot = data[data > 0.001].sort_values(ascending=False)
        if data_plot.empty:
            ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=font_sizes["title"])
            continue
        colors = [_carrier_color(l, config_colors) for l in data_plot.index]
        bars = ax.bar(range(len(data_plot)), data_plot.values, color=colors,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(data_plot)))
        ax.set_xticklabels(data_plot.index, rotation=40, ha="right", fontsize=font_sizes["tick"])
        ax.set_ylabel(f"Kapazität [{unit}]", fontsize=font_sizes["label"])
        ax.set_title(f"{title} — {country}\n{scenario_label}", fontsize=font_sizes["title"] - 2)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        for bar in bars:
            h = bar.get_height()
            if h > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=font_sizes["bar_label"])
        ax.grid(axis="y", alpha=0.35, linestyle="--")

    plt.tight_layout()
    _save(fig, out_dir, f"heat_installed_cap_{country}.png")


def plot_annual_heat_generation(
    gen: pd.Series,
    demand: float,
    country: str,
    scenario_label: str,
    out_dir: Path,
    config_colors: Dict[str, str],
    font_sizes: Dict[str, int],
) -> None:
    gen_plot = gen[gen > 0.001].sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # --- Balken ---
    ax = axes[0]
    if gen_plot.empty:
        ax.text(0.5, 0.5, "Keine Wärmeerzeugung", ha="center", va="center", transform=ax.transAxes)
    else:
        colors = [_carrier_color(l, config_colors) for l in gen_plot.index]
        bars = ax.bar(range(len(gen_plot)), gen_plot.values, color=colors,
                      edgecolor="white", linewidth=0.5)
        ax.axhline(demand, color="#d63031", linestyle="--", linewidth=1.8,
                   label=f"Nachfrage {demand:.1f} TWh")
        ax.set_xticks(range(len(gen_plot)))
        ax.set_xticklabels(gen_plot.index, rotation=40, ha="right", fontsize=font_sizes["tick"])
        ax.set_ylabel("Wärmeenergie [TWh_th]", fontsize=font_sizes["label"])
        ax.legend(fontsize=font_sizes["legend"])
        ax.set_title(f"Jährl. Wärmeerzeugung — {country}\n{scenario_label}",
                     fontsize=font_sizes["title"] - 2)
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01, f"{h:.1f}",
                        ha="center", va="bottom", fontsize=font_sizes["bar_label"])
        ax.grid(axis="y", alpha=0.35, linestyle="--")

    # --- Pie ---
    ax2 = axes[1]
    if not gen_plot.empty:
        colors2 = [_carrier_color(l, config_colors) for l in gen_plot.index]
        wedges, texts, autotexts = ax2.pie(
            gen_plot.values,
            labels=gen_plot.index,
            colors=colors2,
            autopct="%1.1f%%",
            pctdistance=0.82,
            startangle=90,
        )
        for t in texts:
            t.set_fontsize(font_sizes["tick"] - 1)
        for at in autotexts:
            at.set_fontsize(font_sizes["bar_label"])
        ax2.set_title("Anteilsverteilung", fontsize=font_sizes["title"] - 2)
    else:
        ax2.axis("off")

    plt.tight_layout()
    _save(fig, out_dir, f"heat_annual_generation_{country}.png")


def plot_heat_balance_timeseries(
    balance_ts: pd.DataFrame,
    demand_ts: pd.Series,
    country: str,
    scenario_label: str,
    out_dir: Path,
    config_colors: Dict[str, str],
    font_sizes: Dict[str, int],
    time_slice: Optional[slice] = None,
) -> None:
    ts = balance_ts if time_slice is None else balance_ts.loc[time_slice]
    dem = demand_ts if time_slice is None else demand_ts.loc[time_slice]

    active_cols = [c for c in ts.columns if ts[c].sum() > 0.01]
    if not active_cols:
        print(f"    [plot_heat_balance_timeseries] Keine aktiven Wärmequellen für {country}.")
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    colors = [_carrier_color(c, config_colors) for c in active_cols]
    ax.stackplot(ts.index, *[ts[c] for c in active_cols],
                 labels=active_cols, colors=colors, alpha=0.88)

    if dem.sum() > 0:
        ax.plot(dem.index, dem.values, color="#d63031", linewidth=1.5,
                linestyle="-", label="Wärmenachfrage", zorder=5)

    ax.set_ylabel("Wärmeleistung [GW_th]", fontsize=font_sizes["label"])
    title_suffix = "(Jahresübersicht)" if time_slice is None else "(Detail-Zeitraum)"
    ax.set_title(f"Wärmebereitstellung {title_suffix} — {country}\n{scenario_label}",
                 fontsize=font_sizes["title"] - 2)
    ax.legend(loc="upper right", fontsize=font_sizes["legend"], ncol=2,
              frameon=True, framealpha=0.85)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(
        mdates.MonthLocator() if time_slice is None else mdates.DayLocator(interval=3)
    )
    ax.grid(alpha=0.3, linestyle="--")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    suffix = "annual" if time_slice is None else "detail"
    _save(fig, out_dir, f"heat_balance_timeseries_{suffix}_{country}.png")


def plot_thermal_storage(
    storage_soc: pd.DataFrame,
    storage_dispatch_df: pd.DataFrame,
    country: str,
    scenario_label: str,
    out_dir: Path,
    font_sizes: Dict[str, int],
) -> None:
    active = [c for c in storage_soc.columns if storage_soc[c].abs().sum() > 0.001]
    if not active:
        print(f"    [plot_thermal_storage] Keine thermischen Speicher für {country}.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax1 = axes[0]
    for col in active:
        ax1.plot(storage_soc.index, storage_soc[col], label=col, linewidth=1.2)
    ax1.set_ylabel("State of Charge [GWh_th]", fontsize=font_sizes["label"])
    ax1.set_title(f"Thermische Speicher — {country}\n{scenario_label}",
                  fontsize=font_sizes["title"] - 2)
    ax1.legend(fontsize=font_sizes["legend"])
    ax1.grid(alpha=0.3, linestyle="--")

    ax2 = axes[1]
    for col in active:
        if col in storage_dispatch_df.columns:
            dispatch = storage_dispatch_df[col]
            ax2.fill_between(dispatch.index, 0, dispatch.clip(lower=0),
                             alpha=0.7, label=f"{col} (Entladung)")
            ax2.fill_between(dispatch.index, dispatch.clip(upper=0), 0,
                             alpha=0.5, label=f"{col} (Ladung)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Lade-/Entladeleistung [GW_th]", fontsize=font_sizes["label"])
    ax2.legend(fontsize=font_sizes["legend"], ncol=2)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, out_dir, f"heat_storage_{country}.png")


def plot_cop_timeseries(
    cop_df: pd.DataFrame,
    country: str,
    scenario_label: str,
    out_dir: Path,
    font_sizes: Dict[str, int],
) -> None:
    if cop_df.empty or cop_df.isnull().all().all():
        print(f"    [plot_cop_timeseries] Keine COP-Daten für {country}.")
        return

    fig, ax = plt.subplots(figsize=(16, 5))
    for col in cop_df.columns:
        ax.plot(cop_df.index, cop_df[col], linewidth=0.8, alpha=0.75, label=col)
    ax.set_ylabel("COP [-]", fontsize=font_sizes["label"])
    ax.set_title(f"Wärmepumpen-COP Jahresverlauf — {country}\n{scenario_label}",
                 fontsize=font_sizes["title"] - 2)
    ax.legend(fontsize=font_sizes["legend"] - 1, ncol=2)
    ax.grid(alpha=0.3, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, out_dir, f"heat_cop_timeseries_{country}.png")


def plot_heat_balance_summary(
    gen_dict: Dict[str, pd.Series],
    demand_dict: Dict[str, float],
    country: str,
    out_dir: Path,
    config_colors: Dict[str, str],
    font_sizes: Dict[str, int],
    mode: str = "normal",
) -> None:
    """
    Vergleichsplot mehrerer Szenarien (ARO-Modus: robust vs. worst-case-szenarien).
    gen_dict: {scenario_label: pd.Series (TWh_th per carrier)}
    """
    if not gen_dict:
        return

    all_carriers = sorted({c for s in gen_dict.values() for c in s.index if s[c] > 0.001})
    x = np.arange(len(gen_dict))
    width = 0.55

    fig, ax = plt.subplots(figsize=(max(10, len(gen_dict) * 1.8 + 3), 8))
    bottoms = np.zeros(len(gen_dict))

    for carrier in all_carriers:
        vals = np.array([gen_dict[s].get(carrier, 0.0) for s in gen_dict])
        color = _carrier_color(carrier, config_colors)
        ax.bar(x, vals, width, bottom=bottoms, label=carrier, color=color,
               edgecolor="white", linewidth=0.4)
        bottoms += vals

    demands = [demand_dict.get(s, 0.0) for s in gen_dict]
    ax.plot(x, demands, "D--", color="#d63031", linewidth=1.8, markersize=6,
            zorder=5, label="Wärmenachfrage")

    ax.set_xticks(x)
    ax.set_xticklabels(list(gen_dict.keys()), rotation=35, ha="right",
                       fontsize=font_sizes["tick"] - 1)
    ax.set_ylabel("Wärmeenergie [TWh_th]", fontsize=font_sizes["label"])
    title_mode = "ARO-Szenarien" if mode == "aro" else "Szenarienvergleich"
    ax.set_title(f"Wärmebilanz: {title_mode} — {country}", fontsize=font_sizes["title"] - 1)
    ax.legend(loc="upper right", fontsize=font_sizes["legend"], ncol=2,
              frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    plt.tight_layout()
    _save(fig, out_dir, f"heat_balance_summary_{mode}_{country}.png")


def _save(fig, out_dir: Path, fname: str, dpi: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Gespeichert: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE NETWORK RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def _demand_timeseries(n: pypsa.Network, country: str) -> pd.Series:
    """Aggregierte Wärmenachfrage-Zeitreihe [GW_th]."""
    lds = n.loads.copy()
    if country != "ALL":
        lds = lds[lds.bus.str.startswith(country)]
    heat_loads = lds[lds.bus.apply(_is_heat_bus)]
    idx = heat_loads.index.intersection(n.loads_t.p.columns)
    if idx.empty:
        return pd.Series(0.0, index=n.snapshots)
    return n.loads_t.p[idx].sum(axis=1) / 1000.0  # GW


def analyse_single_network(
    n: pypsa.Network,
    country: str,
    scenario_label: str,
    out_dir: Path,
    config_colors: Dict[str, str],
    font_sizes: Dict[str, int],
) -> Tuple[pd.Series, float]:
    """
    Führt alle Wärme-Analysen für ein einzelnes Netzwerk durch.
    Gibt (annual_gen [TWh_th], demand [TWh_th]) zurück (für Sammelplot).
    """
    print(f"    → Wärmeanalyse: {country} | {scenario_label}")
    extractor = HeatNetworkExtractor(n, country)

    # 1) Kapazitäten
    cap_links  = extractor.installed_link_capacities()
    cap_stores = extractor.installed_storage_capacities()
    print(f"       Kapazitäten Links [GW_th]:  {cap_links[cap_links>0].to_dict()}")
    print(f"       Kapazitäten Stores [GWh_th]: {cap_stores[cap_stores>0].to_dict()}")
    plot_installed_capacities(
        cap_links, cap_stores, country, scenario_label, out_dir, config_colors, font_sizes
    )

    # 2) Jährliche Erzeugung + Nachfrage
    gen    = extractor.annual_heat_generation()
    demand = extractor.heat_demand()
    print(f"       Erzeugung [TWh_th]: {gen[gen>0].to_dict()}")
    print(f"       Nachfrage [TWh_th]: {demand:.2f}")
    plot_annual_heat_generation(
        gen, demand, country, scenario_label, out_dir, config_colors, font_sizes
    )

    # 3) Zeitreihe Wärme-Bilanz (Jahresübersicht)
    balance_ts = extractor.heat_balance_timeseries()
    demand_ts  = _demand_timeseries(n, country)
    plot_heat_balance_timeseries(
        balance_ts, demand_ts, country, scenario_label, out_dir, config_colors, font_sizes,
        time_slice=None
    )

    # 4) Detail-Zeitraum (erste 3 Januarwochen des Referenzjahres)
    try:
        ref_year = n.snapshots[0].year
        ts_start = pd.Timestamp(f"{ref_year}-01-07")
        ts_end   = pd.Timestamp(f"{ref_year}-01-28")
        if ts_start in n.snapshots and ts_end in n.snapshots:
            plot_heat_balance_timeseries(
                balance_ts, demand_ts, country, scenario_label, out_dir,
                config_colors, font_sizes,
                time_slice=slice(ts_start, ts_end)
            )
    except Exception as e:
        print(f"       WARNUNG Detail-Zeitraum: {e}")

    # 5) Thermische Speicher
    soc_df      = extractor.storage_state_of_charge()
    dispatch_df = extractor.storage_dispatch()
    plot_thermal_storage(soc_df, dispatch_df, country, scenario_label, out_dir, font_sizes)

    # 6) WP-COP-Zeitreihen
    cop_df = extractor.heat_pump_cop_timeseries()
    plot_cop_timeseries(cop_df, country, scenario_label, out_dir, font_sizes)

    return gen, demand


# ──────────────────────────────────────────────────────────────────────────────
# NORMAL RUN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_normal(
    master: MasterConfig,
    config: PlottingConfig,
    countries: List[str],
) -> None:
    print("\n══════════════ NORMAL RUN: Wärmenetz-Analyse ══════════════")

    networks_raw = config.get_networks()
    if networks_raw is None:
        print("FEHLER: Keine Netzwerke in master_config gefunden.")
        return

    if isinstance(networks_raw, list):
        networks = {master.scenario_selection: networks_raw}
    else:
        networks = networks_raw  # type: ignore[assignment]

    plots_base = Path(master.plots_base)

    for scenario_label, paths in networks.items():
        print(f"\n── Szenario: {scenario_label} ──")
        out_base = plots_base / "heat_analysis" / scenario_label
        out_base.mkdir(parents=True, exist_ok=True)

        for path in paths:
            if not Path(path).is_file():
                print(f"  FEHLT: {path}")
                continue

            print(f"  Lade Netzwerk: {path}")
            n = pypsa.Network(path)

            gen_all: Dict[str, pd.Series] = {}
            demand_all: Dict[str, float] = {}

            for country in countries:
                c_dir = out_base / country
                c_dir.mkdir(parents=True, exist_ok=True)
                gen, demand = analyse_single_network(
                    n, country, scenario_label, c_dir,
                    config.CARRIER_COLORS, config.FONT_SIZES,
                )
                gen_all[country] = gen
                demand_all[country] = demand

            # Ländervergleich-Plot
            if len(gen_all) > 1:
                compare_dir = out_base / "_laendervergleich"
                compare_dir.mkdir(exist_ok=True)
                plot_heat_balance_summary(
                    gen_all, demand_all, "Alle Länder",
                    compare_dir, config.CARRIER_COLORS, config.FONT_SIZES,
                    mode="normal",
                )

    print("\n══════════════ NORMAL RUN abgeschlossen ══════════════")


# ──────────────────────────────────────────────────────────────────────────────
# ARO RUN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_aro(
    master: MasterConfig,
    aro_config: AROPlottingConfig,
    countries: List[str],
) -> None:
    print("\n══════════════ ARO RUN: Wärmenetz-Analyse ══════════════")

    run_cfg  = aro_config.get_current_run_config()
    run_key  = aro_config.SELECTED_RUN
    aro_plots_base = Path(master.aro_plots_base)
    results_base   = Path(master.aro_results_base)

    robust_path = run_cfg.get("robust_network")
    scenarios   = run_cfg.get("scenarios", [])

    def _run_single(net_path: str, label: str, country: str) -> Optional[Tuple[pd.Series, float]]:
        if not Path(net_path).is_file():
            print(f"  FEHLT: {net_path}")
            return None
        print(f"  Lade [{label}]: {net_path}")
        n = pypsa.Network(net_path)
        out_dir = aro_plots_base / run_key / label / "heat_analysis" / country
        out_dir.mkdir(parents=True, exist_ok=True)
        return analyse_single_network(
            n, country, label, out_dir,
            aro_config.CARRIER_COLORS, aro_config.FONT_SIZES,
        )

    for country in countries:
        print(f"\n── Land: {country} ──")
        gen_dict: Dict[str, pd.Series] = {}
        demand_dict: Dict[str, float] = {}

        # a) Robustes Netz
        if robust_path:
            res = _run_single(robust_path, "robust", country)
            if res:
                gen_dict["Robust"], demand_dict["Robust"] = res

        # b) Alle Worst-Case-Szenarien
        for sc in scenarios:
            sc_path = str(results_base / run_key / "networks" / f"{sc}.nc")
            res = _run_single(sc_path, sc, country)
            if res:
                gen_dict[sc], demand_dict[sc] = res

        # c) Sammelplot für dieses Land (Robust vs. alle Szenarien)
        if gen_dict:
            summary_dir = aro_plots_base / run_key / "_summary" / country
            summary_dir.mkdir(parents=True, exist_ok=True)
            plot_heat_balance_summary(
                gen_dict, demand_dict, country,
                summary_dir, aro_config.CARRIER_COLORS, aro_config.FONT_SIZES,
                mode="aro",
            )

    print("\n══════════════ ARO RUN abgeschlossen ══════════════")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    master   = MasterConfig()
    run_type = master.get_run_type()
    print(f"Run-Type erkannt: {run_type.upper()}")

    if run_type == "aro":
        aro_config = AROPlottingConfig(master)
        countries  = aro_config.COUNTRIES_TO_ANALYZE
        run_aro(master, aro_config, countries)
    else:
        config    = PlottingConfig(master)
        countries = config.get_countries()
        run_normal(master, config, countries)


if __name__ == "__main__":
    main()
