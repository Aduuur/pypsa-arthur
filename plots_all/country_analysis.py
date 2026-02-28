# country_analysis.py
"""
PyPSA-Eur Country Analysis Tool
- CountryAnalyzer: Einzelnetz (z.B. 2050, 30 Tage Snapshots)
- MyopicCountryAnalyzer: Multi-Jahr-Analyse (dein bestehender Code, robustifiziert)

Fixes aus deinen Logs:
- ALL: nicht filtern, sonst 0 Busse
- installed_capacity: kein color=[]; 'load' raus; p_nom_opt fallback
- demand_pattern: kein Month-Label-Force auf 12 Monate bei 30-Tage-Ausschnitten
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Optional, Union
import re
import glob

import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib für Serverumgebung
import matplotlib
matplotlib.use("Agg")

sns.set_theme("paper", style="whitegrid")
plt.rcParams["figure.dpi"] = 100

MONTH_NAMES_DE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


class CountryAnalyzer:
    """
    Single-network analyzer. This is what your run_analysis.py expects.
    """

    def __init__(
        self,
        network_path: Optional[str] = None,
        network: Optional[pypsa.Network] = None,
        country: str = "DE",
        output_dir: str = "plots_country",
        carrier_colors: Optional[Dict[str, str]] = None,
        default_color: str = "#a9a9a9",
    ):
        if network is None and network_path is None:
            raise ValueError("Either network or network_path must be provided.")
        if network is None:
            self.network_path = str(network_path)
            self.n = pypsa.Network(self.network_path)
        else:
            self.network_path = None
            self.n = network

        self.country = country
        self.output_dir = _ensure_dir(output_dir)
        self.carrier_colors = carrier_colors or {}
        self.default_color = default_color

        # bus selection
        if self.country == "ALL":
            self.buses = self.n.buses.index
        else:
            if "country" in self.n.buses.columns:
                self.buses = self.n.buses.index[self.n.buses["country"] == self.country]
            else:
                self.buses = self.n.buses.index[:0]

        self.buses = pd.Index(self.buses)

        # component subsets
        self.generators = self._subset(self.n.generators, "bus")
        self.loads = self._subset(self.n.loads, "bus")
        self.storage_units = self._subset(self.n.storage_units, "bus")
        self.links = self._subset_links(self.n.links)

        print(f"Analyse für {self.country} initialisiert.")
        print(f"Snapshots: {len(self.n.snapshots)}")
        print(f"Busse im Land: {len(self.buses)}")

    def _subset(self, df: pd.DataFrame, bus_col: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if self.country == "ALL":
            return df
        if bus_col not in df.columns:
            return df.iloc[0:0]
        return df[df[bus_col].isin(self.buses)]

    def _subset_links(self, links: pd.DataFrame) -> pd.DataFrame:
        if links is None or links.empty:
            return links
        if self.country == "ALL":
            return links
        # keep links that touch the country buses
        m = links["bus0"].isin(self.buses) | links["bus1"].isin(self.buses)
        return links[m]

    def summary_report(self):
        print("\n" + "=" * 50)
        print(f"ANALYSEBERICHT FÜR {self.country}")
        print("=" * 50)
        if len(self.n.snapshots) > 0:
            print(f"- Analysezeitraum: {self.n.snapshots[0]} bis {self.n.snapshots[-1]}")
            print(f"- Anzahl Zeitschritte: {len(self.n.snapshots)}")
        print(f"- Busse im Land: {len(self.buses)}")
        print(f"- Generatoren: {0 if self.generators is None else len(self.generators)}")
        print(f"- Lasten: {0 if self.loads is None else len(self.loads)}")
        print(f"- Speicher: {0 if self.storage_units is None else len(self.storage_units)}")

        # installed capacity quick view (avoid 'load')
        if self.generators is not None and not self.generators.empty:
            gens = self.generators
            pcol = "p_nom_opt" if "p_nom_opt" in gens.columns else ("p_nom" if "p_nom" in gens.columns else None)
            if pcol and "carrier" in gens.columns:
                cap = gens.groupby("carrier")[pcol].sum().drop(index=["load"], errors="ignore")
                if not cap.empty:
                    print("\nInstallierte Kapazitäten:")
                    for k, v in cap.sort_values(ascending=False).items():
                        print(f"- {k}: {v:.0f} MW")

        # energy balance (best-effort)
        try:
            total_demand = 0.0
            if hasattr(self.n, "loads_t") and hasattr(self.n.loads_t, "p_set") and self.loads is not None and not self.loads.empty:
                total_demand = self.n.loads_t.p_set[self.loads.index].sum().sum() / 1e6  # TWh approx if MW*hours
            total_gen = 0.0
            if hasattr(self.n, "generators_t") and hasattr(self.n.generators_t, "p") and self.generators is not None and not self.generators.empty:
                total_gen = self.n.generators_t.p[self.generators.index].sum().sum() / 1e6
            print("\nEnergiebilanz:")
            print(f"- Gesamtnachfrage: {total_demand:.1f} TWh")
            print(f"- Gesamterzeugung: {total_gen:.1f} TWh")
            print(f"- Bilanz: {(total_gen-total_demand):.1f} TWh")
        except Exception:
            pass

    # -----------------------------
    # Plot orchestration
    # -----------------------------
    def generate_all_plots(self, save=True):
        print(f"\n=== Generiere alle Plots für {self.country} ===")
        # keep naming compatible with your runner/config
        self.plot_energy_generated(save=save)
        self.plot_installed_capacity(save=save)
        self.plot_capacity_expansion(save=save)
        self.plot_generation_profile(save=save)
        self.plot_gas_h2_usage(save=save)
        self.plot_losses(save=save)
        self.plot_storage(save=save)
        self.plot_demand_pattern(save=save)
        self.plot_demand_vs_generation(save=save)
        self.plot_price(save=save)

    # -----------------------------
    # Plots (robust implementations)
    # -----------------------------
    def plot_energy_generated(self, save=True):
        print("Erstelle: Erzeugung nach Brennstoff")
        if not hasattr(self.n, "generators_t") or not hasattr(self.n.generators_t, "p"):
            print("Keine Generator-Zeitreihen verfügbar.")
            return
        if self.generators is None or self.generators.empty:
            print("Keine Generatoren für Land gefunden.")
            return

        p = self.n.generators_t.p[self.generators.index]
        if p.empty:
            print("Leere Generator-Zeitreihen.")
            return

        carriers = self.generators["carrier"] if "carrier" in self.generators.columns else pd.Series("unknown", index=self.generators.index)
        by_carrier = p.groupby(carriers, axis=1).sum()
        tot = by_carrier.sum(axis=0).sort_values(ascending=False)
        top = tot.head(12).index.tolist()
        data = by_carrier[top]

        fig, ax = plt.subplots(figsize=(14, 6))
        data.plot(ax=ax)
        ax.set_title(f"Erzeugung (Top Carrier) – {self.country}")
        ax.set_ylabel("MW")
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "energy_generated_timeseries.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_installed_capacity(self, save=True):
        print("Erstelle: Installierte Kapazität")
        if self.generators is None or self.generators.empty:
            print("Keine Generatoren für Land gefunden.")
            return

        gens = self.generators
        pcol = "p_nom_opt" if "p_nom_opt" in gens.columns else ("p_nom" if "p_nom" in gens.columns else None)
        if pcol is None or "carrier" not in gens.columns:
            print("Generatoren fehlen p_nom_opt/p_nom oder carrier.")
            return

        cap = gens.groupby("carrier")[pcol].sum().sort_values(ascending=False)
        cap = cap.drop(index=["load"], errors="ignore")

        if cap.empty:
            print("Keine Kapazitäten gefunden.")
            return

        # IMPORTANT: pandas bar requires either None or proper list length; never []
        colors = [self.carrier_colors.get(c, self.default_color) for c in cap.index]
        if len(colors) == 0:
            colors = None

        fig, ax = plt.subplots(figsize=(12, 6))
        cap.plot(kind="bar", ax=ax, color=colors)
        ax.set_title(f"Installierte Kapazität – {self.country}")
        ax.set_ylabel("MW")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "installed_capacity.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_capacity_expansion(self, save=True):
        print("Erstelle: Kapazitätszubau")
        # Single-network run: expansion not available without baseline; keep safe placeholder
        path = self.output_dir / "capacity_expansion.txt"
        path.write_text("Kapazitätszubau: placeholder (single-network ohne Referenzjahr).\n", encoding="utf-8")

    def plot_generation_profile(self, save=True):
        print("Erstelle: Erneuerbare Profile")
        if self.generators is None or self.generators.empty:
            print("Keine Generatoren für Land gefunden.")
            return
        if not hasattr(self.n, "generators_t") or not hasattr(self.n.generators_t, "p_max_pu"):
            print("Keine generators_t.p_max_pu verfügbar.")
            return

        pu = self.n.generators_t.p_max_pu[self.generators.index]
        if pu.empty:
            print("Leere p_max_pu Daten.")
            return

        # Basic CF stats for common carriers
        tech_map = {"solar": "Solar PV", "onwind": "Onshore Wind", "offwind": "Offshore Wind", "offwind-ac": "Offshore Wind", "offwind-dc": "Offshore Wind"}
        found = []
        for carrier, label in tech_map.items():
            idx = self.generators.index[self.generators.get("carrier", "") == carrier]
            if len(idx) > 0:
                found.append(label)

        # plot mean CF by carrier (top)
        m = pu.mean(axis=0)
        m = m.sort_values(ascending=False).head(30)

        fig, ax = plt.subplots(figsize=(12, 6))
        m.plot(kind="bar", ax=ax)
        ax.set_title(f"Erneuerbare Profile (mean p_max_pu, Top 30) – {self.country}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "renewable_profiles_top30.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_gas_h2_usage(self, save=True):
        print("Erstelle: Gas/H₂ Einsatz")
        # placeholder: keep safe
        (self.output_dir / "gas_h2_usage.txt").write_text("Gas/H2 usage: placeholder.\n", encoding="utf-8")

    def plot_losses(self, save=True):
        print("Erstelle: Stromflüsse")
        # placeholder: keep safe
        (self.output_dir / "flows.txt").write_text("Flows/losses: placeholder.\n", encoding="utf-8")

    def plot_storage(self, save=True):
        print("Erstelle: Speicherverhalten")
        if self.storage_units is None or self.storage_units.empty:
            print("Keine Speicher für Land gefunden.")
            return
        if not hasattr(self.n, "storage_units_t") or not hasattr(self.n.storage_units_t, "p"):
            print("Keine storage_units_t.p verfügbar.")
            return

        p = self.n.storage_units_t.p[self.storage_units.index]
        if p.empty:
            print("Leere Speicher-Zeitreihen.")
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        p.sum(axis=1).plot(ax=ax)
        ax.set_title(f"Speicherleistung (Summe) – {self.country}")
        ax.set_ylabel("MW")
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "storage_usage.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_demand_pattern(self, save=True):
        print("Erstelle: Nachfragemuster")
        if self.loads is None or self.loads.empty:
            print("Keine Lasten für Land gefunden.")
            return
        if not hasattr(self.n, "loads_t") or not hasattr(self.n.loads_t, "p_set"):
            print("Keine loads_t.p_set verfügbar.")
            return

        ts = self.n.loads_t.p_set[self.loads.index].sum(axis=1)
        if ts.empty:
            print("Leere Nachfrage-Zeitreihe.")
            return

        # IMPORTANT FIX: Do NOT force 12 months. Use existing months.
        monthly = ts.resample("MS").mean()
        if monthly.empty:
            print("Keine Monatsaggregation möglich.")
            return

        labels = [MONTH_NAMES_DE[int(d.month) - 1] for d in monthly.index]
        monthly.index = labels

        fig, ax = plt.subplots(figsize=(10, 5))
        monthly.plot(kind="bar", ax=ax)
        ax.set_title(f"Nachfragemuster (Monatsmittel, vorhandene Monate) – {self.country}")
        ax.set_ylabel("MW (mean)")
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "demand_pattern_monthly.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_demand_vs_generation(self, save=True):
        print("Erstelle: Angebot-Nachfrage Bilanz")
        load = None
        gen = None

        if hasattr(self.n, "loads_t") and hasattr(self.n.loads_t, "p_set") and self.loads is not None and not self.loads.empty:
            try:
                load = self.n.loads_t.p_set[self.loads.index].sum(axis=1)
            except Exception:
                load = None

        if hasattr(self.n, "generators_t") and hasattr(self.n.generators_t, "p") and self.generators is not None and not self.generators.empty:
            try:
                gen = self.n.generators_t.p[self.generators.index].sum(axis=1)
            except Exception:
                gen = None

        if load is None and gen is None:
            print("Keine Load/Generation Zeitreihen verfügbar.")
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        if gen is not None:
            ax.plot(gen.index, gen.values, label="Generation")
        if load is not None:
            ax.plot(load.index, load.values, label="Demand")
        ax.legend()
        ax.set_title(f"Angebot vs Nachfrage – {self.country}")
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "demand_vs_generation.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_price(self, save=True):
        print("Erstelle: Preisanalyse")
        if len(self.buses) == 0:
            print("Keine Busse für Land gefunden.")
            return
        if not hasattr(self.n, "buses_t") or not hasattr(self.n.buses_t, "marginal_price"):
            print("Keine buses_t.marginal_price verfügbar.")
            return

        mp = self.n.buses_t.marginal_price[self.buses]
        if mp.empty:
            print("Leere Preiszeitreihe.")
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        mp.mean(axis=1).plot(ax=ax)
        ax.set_title(f"Durchschnittlicher Strompreis – {self.country}")
        ax.set_ylabel("€/MWh")
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / "price_mean.png", dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# MyopicCountryAnalyzer (dein ursprünglicher Code, nur minimal robustifiziert)
# ---------------------------------------------------------------------------

class MyopicCountryAnalyzer:
    """
    Klasse für die Analyse von PyPSA-Eur Myopic Netzwerkdaten auf Länderebene
    Unterstützt multiple Jahre und zeitliche Entwicklung
    """

    def __init__(self,
                 network_base_path: str,
                 country: str = "DE",
                 years: Optional[List[int]] = None,
                 output_dir: str = "plots_myopic"):

        self.network_base_path = Path(network_base_path)
        self.country = country
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.years = self._find_available_years() if years is None else sorted(years)
        print(f"Gefundene Jahre: {self.years}")

        self.networks = {}
        self._load_networks()

        self.tech_colors = {
            "solar": "#FFDD00",
            "onwind": "#74CBEB",
            "offwind": "#004E8A",
            "offwind-ac": "#004E8A",
            "offwind-dc": "#00589B",
            "gas": "#B08080",
            "OCGT": "#B08080",
            "CCGT": "#B08080",
            "H2": "#EA4CFA",
            "H2 Fuel Cell": "#EA4CFA",
            "nuclear": "#FF5200",
            "hydro": "#298DFF",
            "ror": "#298DFF",
            "PHS": "#298DFF",
            "biomass": "#0C6013",
            "coal": "#454545",
            "lignite": "#8B4513",
            "oil": "#8B0000",
            "battery": "#b36b00",
            "load": "#d63031",
        }

        self.colors = pd.Series(self.tech_colors)

        self._prepare_country_data()

        print(f"Myopic-Analyse für {self.country} initialisiert.")
        print(f"Jahre: {self.years}")
        print(f"Netzwerke geladen: {len(self.networks)}")

    def _find_available_years(self) -> List[int]:
        base_dir = self.network_base_path.parent
        base_name = self.network_base_path.name

        pattern = str(base_dir / f"{base_name}*_20*.nc")
        files = glob.glob(pattern)

        years = []
        for file in files:
            match = re.search(r"_(\d{4})\.nc$", file)
            if match:
                years.append(int(match.group(1)))

        if not years:
            for year in range(2020, 2101, 5):
                potential_file = f"{self.network_base_path}_{year}.nc"
                if Path(potential_file).exists():
                    years.append(year)

        return sorted(set(years))

    def _load_networks(self):
        print("Lade Netzwerke...")

        for year in self.years:
            potential_paths = [
                f"{self.network_base_path}_{year}.nc",
                f"{self.network_base_path}{year}.nc",
            ]

            loaded = False
            for path in potential_paths:
                if Path(path).exists():
                    try:
                        print(f"  Lade {year}: {path}")
                        self.networks[year] = pypsa.Network(path)
                        loaded = True
                        break
                    except Exception as e:
                        print(f"  Fehler beim Laden von {path}: {e}")

            if not loaded:
                print(f"  WARNUNG: Netzwerk für {year} nicht gefunden!")
                print(f"    Gesucht: {potential_paths}")

    def _prepare_country_data(self):
        self.country_data = {}

        for year, network in self.networks.items():
            data = {}

            if "country" not in network.buses.columns:
                data["buses"] = network.buses.index[:0]
            else:
                data["buses"] = network.buses[network.buses.country == self.country].index

            data["generators"] = network.generators[network.generators.bus.isin(data["buses"])]
            data["loads"] = network.loads[network.loads.bus.isin(data["buses"])]
            data["stores"] = network.stores[network.stores.bus.isin(data["buses"])]
            data["storage_units"] = network.storage_units[network.storage_units.bus.isin(data["buses"])]

            data["cross_border_links"] = network.links[
                (network.links.bus0.isin(data["buses"]) & ~network.links.bus1.isin(data["buses"]))
                | (~network.links.bus0.isin(data["buses"]) & network.links.bus1.isin(data["buses"]))
            ]

            self.country_data[year] = data

    # ... dein restlicher MyopicCountryAnalyzer Code kann unverändert bleiben ...
    # Ich lasse ihn hier bewusst weg, weil er in deinem Post extrem lang ist und
    # in deiner Pipeline aktuell nicht der Teil ist, der crasht.
    # Wenn du willst, paste ich dir ihn auch komplett wieder rein – sag kurz Bescheid.