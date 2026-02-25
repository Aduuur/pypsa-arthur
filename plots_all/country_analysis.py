"""
PyPSA-Eur Country Analysis Tool
Hauptmodul für die Analyse und Visualisierung von Länderdaten aus PyPSA-Eur Netzwerken
"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from master_config import AROPlottingConfig


# Matplotlib Backend für Server ohne Display setzen
import matplotlib
matplotlib.use('Agg')  # Für Server ohne GUI

# Seaborn Style setzen
sns.set_theme("paper", style="whitegrid")
plt.rcParams['figure.dpi'] = 100

class CountryAnalyzer:
    """
    Klasse für die Analyse von PyPSA-Eur Netzwerkdaten auf Länderebene
    """

    def __init__(self, network_path: str, country: str = "DE", output_dir: str = "plots"):
        """
        Initialisiert den CountryAnalyzer

        Parameters:
        -----------
        network_path : str
            Pfad zur .nc Datei des PyPSA Netzwerks
        country : str
            Ländercode (z.B. "DE", "FR", "ES")
        output_dir : str
            Verzeichnis für die Ausgabe der Plots
        """
        self.network_path = Path(network_path)
        self.country = country
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Netzwerk laden
        print(f"Lade Netzwerk aus {self.network_path}...")
        self.n = pypsa.Network(str(self.network_path))

        # Tech-Farben definieren (PyPSA-Eur Standard)
        self.tech_colors = {
            'solar': '#FFDD00',
            'solar rooftop': '#FFE640',
            'solar-hsat': '#FFF080',
            'onwind': '#74CBEB',
            'offwind': '#004E8A',
            'offwind-ac': '#003366',
            'offwind-dc': '#002244',
            'offwind-float': '#001122',
            'gas': '#B08080',
            'OCGT': '#B08080',
            'CCGT': '#A07070',
            'H2': '#EA4CFA',
            'H2 Fuel Cell': '#DA3CEA',
            'nuclear': '#FF5200',
            'hydro': '#298DFF',
            'ror': '#1E7DDF',
            'PHS': '#298DFF',
            'biomass': '#0C6013',
            'biogas': '#0C8013',
            'solid biomass': '#0A5011',
            'unsustainable solid biomass': '#8B4513',
            'unsustainable biogas': '#CD853F',
            'unsustainable bioliquids': '#D2691E',
            'coal': '#454545',
            'lignite': '#8B4513',
            'oil': '#8B0000',
            'battery': '#b36b00',
            'load': '#d63031',
            'rural heat vent': '#FF6B6B',
            'urban central heat vent': '#FF8E8E',
            'urban decentral heat vent': '#FFB1B1',
            'rural solar thermal': '#FFD700',
            'urban central solar thermal': '#FFA500',
            'urban decentral solar thermal': '#FF8C00'
        }

        # Carrier-Farben setzen
        self._setup_colors()

        # Länderdaten filtern
        self._filter_country_data()

        print(f"Analyse für {self.country} initialisiert.")
        print(f"Snapshots: {len(self.n.snapshots)}")
        print(f"Busse im Land: {len(self.country_buses)}")

    def _setup_colors(self):
        """Setzt die Farben für die verschiedenen Carrier"""
        if hasattr(self.n, 'carriers') and not self.n.carriers.empty:
            for carrier, color in self.tech_colors.items():
                if carrier in self.n.carriers.index:
                    self.n.carriers.loc[carrier, 'color'] = color

        self.colors = pd.Series(self.tech_colors)

    def _filter_country_data(self):
        """Filtert alle relevanten Daten für das spezifische Land"""
        # Busse des Landes
        self.country_buses = self.n.buses[self.n.buses.country == self.country].index

        # Komponenten des Landes
        self.country_generators = self.n.generators[
            self.n.generators.bus.isin(self.country_buses)
        ]
        self.country_loads = self.n.loads[
            self.n.loads.bus.isin(self.country_buses)
        ]
        self.country_stores = self.n.stores[
            self.n.stores.bus.isin(self.country_buses)
        ]
        self.country_storage_units = self.n.storage_units[
            self.n.storage_units.bus.isin(self.country_buses)
        ]

        # Grenzüberschreitende Links
        self.cross_border_links = self.n.links[
            (self.n.links.bus0.isin(self.country_buses) &
             ~self.n.links.bus1.isin(self.country_buses)) |
            (~self.n.links.bus0.isin(self.country_buses) &
             self.n.links.bus1.isin(self.country_buses))
        ]

    def plot_generation_by_fuel(self, save=True):
        """1. Erzeugung pro Brennstofftyp"""
        if self.country_generators.empty:
            print(f"Keine Generatoren für {self.country} gefunden.")
            return

        generation_by_carrier = (
            self.n.generators_t.p.loc[:, self.country_generators.index]
            .groupby(self.country_generators.carrier, axis=1).sum()
        )

        fig, ax = plt.subplots(figsize=(15, 8))
        generation_by_carrier.plot(
            kind='area',
            stacked=True,
            color=[self.colors.get(c, 'grey') for c in generation_by_carrier.columns],
            ax=ax
        )
        ax.set_title(f"Stromerzeugung nach Brennstoff - {self.country}", fontsize=16)
        ax.set_ylabel("Erzeugung [MW]", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_generation_by_fuel.png",
                       dpi=300, bbox_inches='tight')
        plt.close()  # Wichtig: Figure schließen

    def plot_installed_capacity(self, save=True):
        """2. Installierte Leistung pro Brennstofftyp"""
        capacity_by_carrier = self.country_generators.groupby('carrier')['p_nom_opt'].sum()
        if capacity_by_carrier.empty:
            print("Keine installierten Kapazitäten gefunden.")
            return

        colors = [self.colors.get(c, self.default_color) for c in capacity_by_carrier.index]
        if len(colors) == 0:
            colors = None  # fallback: pandas default

        capacity_by_carrier.plot(kind="bar", color=colors, ax=ax)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = capacity_by_carrier.plot(
            kind='bar',
            color=[self.colors.get(c, 'grey') for c in capacity_by_carrier.index],
            ax=ax
        )
        ax.set_title(f"Installierte Leistung nach Technologie - {self.country}", fontsize=16)
        ax.set_ylabel("Kapazität [MW]", fontsize=12)
        ax.set_xlabel("Technologie", fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # Werte auf Balken anzeigen
        for i, v in enumerate(capacity_by_carrier.values):
            ax.text(i, v + capacity_by_carrier.max()*0.01, f'{v:.0f}',
                   ha='center', va='bottom')

        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / f"{self.country}_installed_capacity.png",
                       dpi=300, bbox_inches='tight')
        plt.close()

    def plot_capacity_expansion(self, save=True):
        """3. Kapazitätszubau"""
        expansion = self.country_generators['p_nom_opt'] - self.country_generators['p_nom']
        expansion_by_carrier = expansion.groupby(self.country_generators['carrier']).sum()
        expansion_positive = expansion_by_carrier[expansion_by_carrier > 0]

        if expansion_positive.empty:
            print(f"Kein Kapazitätszubau für {self.country} gefunden.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        expansion_positive.plot(
            kind='bar',
            color=[self.colors.get(c, 'grey') for c in expansion_positive.index],
            ax=ax
        )
        ax.set_title(f"Kapazitätszubau nach Technologie - {self.country}", fontsize=16)
        ax.set_ylabel("Zubau [MW]", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_capacity_expansion.png",
                       dpi=300, bbox_inches='tight')
        plt.close()

    def plot_renewables_profiles(self, resolution="monthly", save=True):
        """4. Profile PV/Wind mit verschiedenen Auflösungen - KORRIGIERT"""
        # Separate Technologien identifizieren
        solar_gens = self.country_generators[
            self.country_generators.carrier.str.contains("solar", case=False)
        ]
        onwind_gens = self.country_generators[
            self.country_generators.carrier == "onwind"
        ]
        offwind_gens = self.country_generators[
            self.country_generators.carrier.str.contains("offwind", case=False)
        ]

        # Prüfen ob Daten vorhanden
        available_techs = []
        if not solar_gens.empty:
            available_techs.append("Solar")
        if not onwind_gens.empty:
            available_techs.append("Onshore Wind")
        if not offwind_gens.empty:
            available_techs.append("Offshore Wind")

        if not available_techs:
            print(f"Keine erneuerbaren Energien für {self.country} gefunden.")
            return

        print(f"Gefundene erneuerbare Technologien: {available_techs}")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        resolutions = {
            'hourly': ('H', 'Stündlich'),
            'daily': ('D', 'Täglich'),
            'weekly': ('W', 'Wöchentlich'),
            'monthly': ('M', 'Monatlich')
        }

        for idx, (res_key, (freq, title)) in enumerate(resolutions.items()):
            ax = axes[idx//2, idx%2]

            # Solar PV
            if not solar_gens.empty:
                solar_generation = self.n.generators_t.p.loc[:, solar_gens.index].sum(axis=1)
                solar_capacity = solar_gens.p_nom_opt.sum()
                solar_cf = solar_generation / solar_capacity if solar_capacity > 0 else pd.Series(0, index=solar_generation.index)

                if res_key != 'hourly':
                    solar_cf = solar_cf.resample(freq).mean()

                solar_cf.plot(label="Solar PV", color=self.colors.get('solar', 'orange'), ax=ax)

            # Onshore Wind
            if not onwind_gens.empty:
                onwind_generation = self.n.generators_t.p.loc[:, onwind_gens.index].sum(axis=1)
                onwind_capacity = onwind_gens.p_nom_opt.sum()
                onwind_cf = onwind_generation / onwind_capacity if onwind_capacity > 0 else pd.Series(0, index=onwind_generation.index)

                if res_key != 'hourly':
                    onwind_cf = onwind_cf.resample(freq).mean()

                onwind_cf.plot(label="Onshore Wind", color=self.colors.get('onwind', 'lightblue'), ax=ax)

            # Offshore Wind
            if not offwind_gens.empty:
                offwind_generation = self.n.generators_t.p.loc[:, offwind_gens.index].sum(axis=1)
                offwind_capacity = offwind_gens.p_nom_opt.sum()
                offwind_cf = offwind_generation / offwind_capacity if offwind_capacity > 0 else pd.Series(0, index=offwind_generation.index)

                if res_key != 'hourly':
                    offwind_cf = offwind_cf.resample(freq).mean()

                offwind_cf.plot(label="Offshore Wind", color=self.colors.get('offwind', 'darkblue'), ax=ax)

            ax.set_title(f"{title} - Kapazitätsfaktoren")
            ax.set_ylabel("Kapazitätsfaktor")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Erneuerbare Energien Profile - {self.country}", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_renewables_profiles.png",
                       dpi=300, bbox_inches='tight')
        plt.close()

        # Zusätzliche Statistiken ausgeben
        print(f"\n=== Kapazitätsfaktor-Statistiken für {self.country} ===")
        if not solar_gens.empty:
            solar_gen_total = self.n.generators_t.p.loc[:, solar_gens.index].sum(axis=1)
            solar_cap_total = solar_gens.p_nom_opt.sum()
            solar_cf_annual = (solar_gen_total.sum() / (solar_cap_total * len(self.n.snapshots))) if solar_cap_total > 0 else 0
            print(f"Solar PV - Jahres-Kapazitätsfaktor: {solar_cf_annual:.2%}")

        if not onwind_gens.empty:
            onwind_gen_total = self.n.generators_t.p.loc[:, onwind_gens.index].sum(axis=1)
            onwind_cap_total = onwind_gens.p_nom_opt.sum()
            onwind_cf_annual = (onwind_gen_total.sum() / (onwind_cap_total * len(self.n.snapshots))) if onwind_cap_total > 0 else 0
            print(f"Onshore Wind - Jahres-Kapazitätsfaktor: {onwind_cf_annual:.2%}")

        if not offwind_gens.empty:
            offwind_gen_total = self.n.generators_t.p.loc[:, offwind_gens.index].sum(axis=1)
            offwind_cap_total = offwind_gens.p_nom_opt.sum()
            offwind_cf_annual = (offwind_gen_total.sum() / (offwind_cap_total * len(self.n.snapshots))) if offwind_cap_total > 0 else 0
            print(f"Offshore Wind - Jahres-Kapazitätsfaktor: {offwind_cf_annual:.2%}")

    def plot_gas_h2_dispatch(self, save=True):
        """5. Gas-/H₂-Kraftwerke Einsatz"""
        gas_gens = self.country_generators[
            self.country_generators.carrier.str.contains("gas", case=False) |
            self.country_generators.carrier.str.contains("OCGT", case=False) |
            self.country_generators.carrier.str.contains("CCGT", case=False)
        ]
        h2_gens = self.country_generators[
            self.country_generators.carrier.str.contains("H2", case=False)
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Gas-Kraftwerke
        if not gas_gens.empty:
            gas_dispatch = self.n.generators_t.p.loc[:, gas_gens.index].sum(axis=1)

            # Zeitreihe
            gas_dispatch.plot(ax=axes[0,0], color=self.colors.get('gas', 'brown'))
            axes[0,0].set_title("Gas-Kraftwerke Einsatz (Zeitreihe)")
            axes[0,0].set_ylabel("Leistung [MW]")

            # Monatliche Statistik
            monthly_gas = gas_dispatch.resample('M').agg(['mean', 'max', 'sum'])
            monthly_gas['sum'].plot(kind='bar', ax=axes[0,1],
                                   color=self.colors.get('gas', 'brown'))
            axes[0,1].set_title("Gas-Kraftwerke Monatliche Erzeugung")
            axes[0,1].set_ylabel("Energie [MWh]")
            plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45)
        else:
            axes[0,0].text(0.5, 0.5, 'Keine Gas-Kraftwerke gefunden',
                          ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,1].text(0.5, 0.5, 'Keine Gas-Kraftwerke gefunden',
                          ha='center', va='center', transform=axes[0,1].transAxes)

        # H₂-Kraftwerke
        if not h2_gens.empty:
            h2_dispatch = self.n.generators_t.p.loc[:, h2_gens.index].sum(axis=1)

            # Zeitreihe
            h2_dispatch.plot(ax=axes[1,0], color=self.colors.get('H2', 'purple'))
            axes[1,0].set_title("H₂-Kraftwerke Einsatz (Zeitreihe)")
            axes[1,0].set_ylabel("Leistung [MW]")

            # Monatliche Statistik
            monthly_h2 = h2_dispatch.resample('M').sum()
            monthly_h2.plot(kind='bar', ax=axes[1,1],
                           color=self.colors.get('H2', 'purple'))
            axes[1,1].set_title("H₂-Kraftwerke Monatliche Erzeugung")
            axes[1,1].set_ylabel("Energie [MWh]")
            plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)
        else:
            axes[1,0].text(0.5, 0.5, 'Keine H₂-Kraftwerke gefunden',
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,1].text(0.5, 0.5, 'Keine H₂-Kraftwerke gefunden',
                          ha='center', va='center', transform=axes[1,1].transAxes)

        plt.suptitle(f"Gas- und H₂-Kraftwerke Einsatz - {self.country}", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_gas_h2_dispatch.png",
                       dpi=300, bbox_inches='tight')
        plt.close()

    def plot_power_flows(self, save=True):
        """6. Stromflüsse (Export/Import)"""
        if self.cross_border_links.empty:
            print(f"Keine grenzüberschreitenden Verbindungen für {self.country} gefunden.")
            return

        # Berechnung der Stromflüsse
        flows = pd.DataFrame()
        for link_id in self.cross_border_links.index:
            link = self.cross_border_links.loc[link_id]
            flow = self.n.links_t.p0.loc[:, link_id]

            # Bestimme Richtung (positiv = Import, negativ = Export)
            if link.bus0 in self.country_buses:
                # Flow geht vom Land weg (Export)
                neighbor_country = link.bus1.split(' ')[0][:2] if ' ' in link.bus1 else link.bus1[:2]
                flows[f"zu_{neighbor_country}"] = -flow
            else:
                # Flow kommt ins Land (Import)
                neighbor_country = link.bus0.split(' ')[0][:2] if ' ' in link.bus0 else link.bus0[:2]
                flows[f"von_{neighbor_country}"] = flow

        net_import = flows.sum(axis=1)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Gesamter Netto-Import
        net_import.plot(ax=axes[0,0])
        axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0,0].set_title("Netto Stromimport/-export")
        axes[0,0].set_ylabel("Leistung [MW] (positiv = Import)")

        # Einzelne Flows
        flows.plot(ax=axes[0,1], alpha=0.7)
        axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0,1].set_title("Stromflüsse nach Ländern")
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Monatliche Bilanz
        monthly_flows = flows.resample('M').sum()
        monthly_flows.plot(kind='bar', ax=axes[1,0], stacked=True)
        axes[1,0].set_title("Monatliche Stromflüsse")
        axes[1,0].set_ylabel("Energie [MWh]")
        plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45)

        # Jahresstatistik
        annual_stats = pd.DataFrame({
            'Import': flows[flows > 0].sum(),
            'Export': flows[flows < 0].sum().abs()
        }).fillna(0)
        annual_stats.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title("Jährliche Import/Export Bilanz")
        axes[1,1].set_ylabel("Energie [MWh]")
        plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)

        plt.suptitle(f"Stromflüsse - {self.country}", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_power_flows.png",
                       dpi=300, bbox_inches='tight')
        plt.close()

    def plot_storage_behavior(self, save=True):
        """7. Speicherverhalten"""
        # Stores (z.B. H2, Gas Speicher)
        store_levels = pd.Series(0.0, index=self.n.snapshots)
        if not self.country_stores.empty:
            store_levels = self.n.stores_t.e.loc[:, self.country_stores.index].sum(axis=1)

        # Storage Units (z.B. Batterien, Pumpspeicher)
        su_levels = pd.Series(0.0, index=self.n.snapshots)
        if not self.country_storage_units.empty:
            su_levels = (self.n.storage_units_t.state_of_charge
                        .loc[:, self.country_storage_units.index].sum(axis=1))

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Gesamte Speicher
        total_storage = store_levels + su_levels
        total_storage.plot(ax=axes[0,0])
        axes[0,0].set_title("Gesamt Speicher-Füllstände")
        axes[0,0].set_ylabel("Gespeicherte Energie [MWh]")

        # Separate Speichertypen
        if not self.country_stores.empty:
            store_levels.plot(ax=axes[0,1], label="Stores", color='blue')
        if not self.country_storage_units.empty:
            su_levels.plot(ax=axes[0,1], label="Storage Units", color='red')
        axes[0,1].set_title("Speicher nach Typ")
        axes[0,1].legend()

        # Monatliche Durchschnitte
        monthly_storage = total_storage.resample('M').mean()
        monthly_storage.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title("Monatliche Durchschnittsfüllstände")
        axes[1,0].set_ylabel("Energie [MWh]")
        plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45)

        # Speicherkapazität vs. Nutzung
        if not self.country_stores.empty or not self.country_storage_units.empty:
            max_capacity = (self.country_stores.e_nom_opt.sum() +
                           self.country_storage_units.p_nom_opt.sum())
            if max_capacity > 0:
                utilization = (total_storage / max_capacity * 100)
                utilization.plot(ax=axes[1,1])
                axes[1,1].set_title("Speicherauslastung")
                axes[1,1].set_ylabel("Auslastung [%]")
            else:
                axes[1,1].text(0.5, 0.5, 'Keine Speicherkapazität verfügbar',
                              ha='center', va='center', transform=axes[1,1].transAxes)

        plt.suptitle(f"Speicherverhalten - {self.country}", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_storage_behavior.png",
                       dpi=300, bbox_inches='tight')
        plt.close()

    def plot_demand_pattern(self, save=True):
        """8. Nachfragepattern über das Jahr"""
        if self.country_loads.empty:
            print(f"Keine Lasten für {self.country} gefunden.")
            return

        demand = self.n.loads_t.p.loc[:, self.country_loads.index].sum(axis=1)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Tägliches Muster (Durchschnitt über alle Tage)
        hourly_pattern = demand.groupby(demand.index.hour).mean()
        hourly_pattern.plot(ax=axes[0,0], marker='o')
        axes[0,0].set_title("Durchschnittliches Tägliches Nachfragemuster")
        axes[0,0].set_xlabel("Stunde")
        axes[0,0].set_ylabel("Nachfrage [MW]")
        axes[0,0].grid(True, alpha=0.3)

        # Wöchentliches Muster
        weekly_pattern = demand.groupby(demand.index.dayofweek).mean()
        weekly_pattern.index = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']
        weekly_pattern.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title("Durchschnittliches Wöchentliches Nachfragemuster")
        axes[0,1].set_ylabel("Nachfrage [MW]")
        plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45)

        # Monatliches Muster
        monthly_pattern = demand.resample('M').mean()
        monthly_pattern.plot(ax=axes[1,0])
        axes[1,0].set_title("Monatliche Durchschnittsnachfrage")
        axes[1,0].set_ylabel("Nachfrage [MW]")

        # Saisonales Muster
        seasonal_pattern = demand.groupby(demand.index.month).mean()
        seasonal_pattern.index = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
        seasonal_pattern.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title("Saisonales Nachfragemuster")
        axes[1,1].set_ylabel("Nachfrage [MW]")
        plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)

        plt.suptitle(f"Nachfragepattern - {self.country}", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_demand_pattern.png",
                       dpi=300, bbox_inches='tight')
        plt.close()

    def plot_supply_demand_balance(self, save=True):
        """9. Nachfragesumme mit Abgleich der erzeugten Menge - VERBESSERT"""
        demand = self.n.loads_t.p.loc[:, self.country_loads.index].sum(axis=1)
        supply = self.n.generators_t.p.loc[:, self.country_generators.index].sum(axis=1)

        # Statistiken berechnen
        total_demand = demand.sum() / 1e6  # TWh
        total_supply = supply.sum() / 1e6  # TWh
        balance = total_supply - total_demand

        print(f"\n=== Energiebilanz {self.country} ===")
        print(f"Gesamtnachfrage: {total_demand:.1f} TWh")
        print(f"Gesamterzeugung: {total_supply:.1f} TWh")
        print(f"Bilanz: {balance:.1f} TWh ({balance / total_demand * 100:.1f}%)")

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # Größer für bessere Lesbarkeit

        # Monatliche Bilanz
        monthly_demand = demand.resample('M').sum() / 1e3  # GWh
        monthly_supply = supply.resample('M').sum() / 1e3  # GWh

        monthly_df = pd.DataFrame({
            'Nachfrage': monthly_demand,
            'Erzeugung': monthly_supply
        })
        monthly_df.plot(ax=axes[0, 0])
        axes[0, 0].set_title("Monatliche Energiebilanz")
        axes[0, 0].set_ylabel("Energie [GWh]")

        # Kumulierte Bilanz
        cumulative_balance = (supply - demand).cumsum() / 1e3
        cumulative_balance.plot(ax=axes[0, 1])
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_title("Kumulierte Energiebilanz")
        axes[0, 1].set_ylabel("Kumulierte Bilanz [GWh]")

        # Load Duration Curve
        demand_sorted = demand.sort_values(ascending=False)
        supply_sorted = supply.sort_values(ascending=False)
        hours = np.arange(1, len(demand_sorted) + 1)

        axes[1, 0].plot(hours, demand_sorted.values, label='Nachfrage', color='red')
        axes[1, 0].plot(hours, supply_sorted.values, label='Erzeugung', color='blue')
        axes[1, 0].set_title("Dauerlinie")
        axes[1, 0].set_xlabel("Stunden")
        axes[1, 0].set_ylabel("Leistung [MW]")
        axes[1, 0].legend()

        # VERBESSERTE Erzeugung nach Carrier
        generation_by_carrier = (
                self.n.generators_t.p.loc[:, self.country_generators.index]
                .groupby(self.country_generators.carrier, axis=1).sum().sum() / 1e6
        )

        # Nur positive Werte und signifikante Anteile (> 0.5% oder > 5 TWh)
        generation_positive = generation_by_carrier[generation_by_carrier > 0]
        total_generation = generation_positive.sum()
        significant_threshold = max(total_generation * 0.005, 5)  # 0.5% oder 5 TWh

        # Große und kleine Segmente trennen
        large_segments = generation_positive[generation_positive >= significant_threshold]
        small_segments = generation_positive[generation_positive < significant_threshold]

        # Kleine Segmente zusammenfassen
        if not small_segments.empty:
            other_sum = small_segments.sum()
            plot_data = large_segments.copy()
            plot_data['Sonstige'] = other_sum
        else:
            plot_data = large_segments.copy()

        if not plot_data.empty:
            # Bessere Farben zuweisen
            colors = []
            labels = []
            for carrier in plot_data.index:
                if carrier == 'Sonstige':
                    colors.append('#CCCCCC')  # Grau für Sonstige
                    labels.append(f'Sonstige\n({other_sum:.1f} TWh)')
                else:
                    colors.append(self.colors.get(carrier, '#808080'))
                    labels.append(f'{carrier}\n({plot_data[carrier]:.1f} TWh)')

            # Pie Chart mit verbesserter Lesbarkeit
            wedges, texts, autotexts = axes[1, 1].pie(
                plot_data.values,
                labels=None,  # Labels separat hinzufügen
                colors=colors,
                autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',  # Nur große Prozente anzeigen
                startangle=90,  # Beginne oben
                textprops={'fontsize': 10}
            )

            # Separate Legende außerhalb des Pie Charts
            axes[1, 1].legend(wedges, labels,
                              title="Erzeugung nach Technologie",
                              loc="center left",
                              bbox_to_anchor=(1, 0, 0.5, 1),
                              fontsize=9)

            axes[1, 1].set_title("Erzeugung nach Technologie [TWh]", pad=20)

            # Zusätzliche Statistiken als Text
            stats_text = f"""Gesamt: {total_generation:.1f} TWh
    Anzahl Technologien: {len(generation_positive)}
    Größte: {large_segments.index[0]} ({large_segments.iloc[0]:.1f} TWh)"""

            axes[1, 1].text(0.02, 0.98, stats_text,
                            transform=axes[1, 1].transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                            fontsize=9)

        else:
            axes[1, 1].text(0.5, 0.5, 'Keine positive Erzeugung gefunden',
                            ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.suptitle(f"Angebot-Nachfrage Bilanz - {self.country}", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_supply_demand_balance.png",
                        dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'total_demand_TWh': total_demand,
            'total_supply_TWh': total_supply,
            'balance_TWh': balance,
            'balance_percent': balance / total_demand * 100
        }

    def plot_marginal_units(self, save=True):
        """10. Welche erzeugende Einheit bestimmt zu welcher Stunde den Preis? - KORRIGIERT"""
        if self.country_buses.empty:
            print(f"Keine Busse für {self.country} gefunden.")
            return

        # Prüfen ob Preisdaten verfügbar sind
        if not hasattr(self.n, 'buses_t') or 'marginal_price' not in self.n.buses_t:
            print(f"Keine Preisdaten verfügbar für {self.country}.")
            return

        # Preise für das Land
        try:
            prices = self.n.buses_t.marginal_price.loc[:, self.country_buses].mean(axis=1)
        except KeyError:
            print(f"Keine Preisdaten in buses_t für {self.country} gefunden.")
            return

        # Grenzkosten der Technologien
        marginal_costs = self.country_generators.groupby('carrier')['marginal_cost'].mean()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Preisverlauf
        prices.plot(ax=axes[0,0])
        axes[0,0].set_title("Strompreisverlauf")
        axes[0,0].set_ylabel("Preis [€/MWh]")

        # Preisdauerlinie
        prices_sorted = prices.sort_values(ascending=False)
        hours = np.arange(1, len(prices_sorted) + 1)
        axes[0,1].plot(hours, prices_sorted.values)
        axes[0,1].set_title("Preisdauerlinie")
        axes[0,1].set_xlabel("Stunden")
        axes[0,1].set_ylabel("Preis [€/MWh]")

        # Grenzkosten nach Technologie
        marginal_costs.sort_values().plot(kind='bar', ax=axes[1,0],
                                         color=[self.colors.get(c, 'grey') for c in marginal_costs.sort_values().index])
        axes[1,0].set_title("Grenzkosten nach Technologie")
        axes[1,0].set_ylabel("Grenzkosten [€/MWh]")
        plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Preisstatistiken
        price_stats = pd.DataFrame({
            'Statistik': ['Mittel', 'Median', 'Min', 'Max', 'Std'],
            'Wert [€/MWh]': [
                prices.mean(),
                prices.median(),
                prices.min(),
                prices.max(),
                prices.std()
            ]
        })

        # Tabelle als Plot
        axes[1,1].axis('tight')
        axes[1,1].axis('off')
        table = axes[1,1].table(cellText=price_stats.round(2).values,
                               colLabels=price_stats.columns,
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        axes[1,1].set_title("Preisstatistiken")

        plt.suptitle(f"Preisanalyse - {self.country}", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_marginal_units.png",
                       dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n=== Grenzkosten nach Technologie ===")
        print(marginal_costs.sort_values().round(2))

    def generate_all_plots(self, save=True):
        """Generiert alle Plots auf einmal"""
        print(f"\n=== Generiere alle Plots für {self.country} ===")

        plot_functions = [
            ("Erzeugung nach Brennstoff", self.plot_generation_by_fuel),
            ("Installierte Kapazität", self.plot_installed_capacity),
            ("Kapazitätszubau", self.plot_capacity_expansion),
            ("Erneuerbare Profile", self.plot_renewables_profiles),
            ("Gas/H₂ Einsatz", self.plot_gas_h2_dispatch),
            ("Stromflüsse", self.plot_power_flows),
            ("Speicherverhalten", self.plot_storage_behavior),
            ("Nachfragemuster", self.plot_demand_pattern),
            ("Angebot-Nachfrage Bilanz", self.plot_supply_demand_balance),
            ("Preisanalyse", self.plot_marginal_units)
        ]

        for name, func in plot_functions:
            try:
                print(f"Erstelle: {name}")
                func(save=save)
            except Exception as e:
                print(f"Fehler bei {name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\nAlle Plots gespeichert in: {self.output_dir}")

    def summary_report(self):
        """Erstellt einen zusammenfassenden Bericht"""
        print(f"\n{'='*50}")
        print(f"ANALYSEBERICHT FÜR {self.country}")
        print(f"{'='*50}")

        print(f"\nNetzwerk-Übersicht:")
        print(f"- Analysezeitraum: {self.n.snapshots[0]} bis {self.n.snapshots[-1]}")
        print(f"- Anzahl Zeitschritte: {len(self.n.snapshots)}")
        print(f"- Busse im Land: {len(self.country_buses)}")
        print(f"- Generatoren: {len(self.country_generators)}")
        print(f"- Lasten: {len(self.country_loads)}")
        print(f"- Speicher: {len(self.country_stores) + len(self.country_storage_units)}")

        # Kapazitäten
        if not self.country_generators.empty:
            print(f"\nInstallierte Kapazitäten:")
            capacity_by_carrier = self.country_generators.groupby('carrier')['p_nom_opt'].sum()
            for carrier, capacity in capacity_by_carrier.sort_values(ascending=False).items():
                print(f"- {carrier}: {capacity:.0f} MW")

        # Energiebilanz
        if not self.country_loads.empty and not self.country_generators.empty:
            demand = self.n.loads_t.p.loc[:, self.country_loads.index].sum().sum() / 1e6
            supply = self.n.generators_t.p.loc[:, self.country_generators.index].sum().sum() / 1e6
            print(f"\nEnergiebilanz:")
            print(f"- Gesamtnachfrage: {demand:.1f} TWh")
            print(f"- Gesamterzeugung: {supply:.1f} TWh")
            print(f"- Bilanz: {(supply-demand):.1f} TWh")

def main():
    """Beispiel für die Verwendung"""
    # Pfad zu Ihrer .nc Datei
    network_path = "/home/endata/pypsa-eur/results/masterarbeit_myopic_heating_transport_biomass/networks/base_s_18__H-T-B_2030.nc"

    # Analyzer für Deutschland erstellen
    analyzer = CountryAnalyzer(network_path, country="DE", output_dir="/shared_endata/atlite_cutouts/results/PyPSA-eur results/Plots/testrun_1997")

    # Zusammenfassungsbericht
    analyzer.summary_report()

    # Alle Plots erstellen
    analyzer.generate_all_plots(save=True)

    # Oder einzelne Plots erstellen
    # analyzer.plot_generation_by_fuel()
    # analyzer.plot_installed_capacity()
    # etc.


if __name__ == "__main__":
    main()