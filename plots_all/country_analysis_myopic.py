"""
PyPSA-Eur Country Analysis Tool - Myopic Version
Hauptmodul für die Analyse von Länderdaten aus PyPSA-Eur Myopic Runs
"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Union
import glob
import re

warnings.filterwarnings('ignore')

# Seaborn Style setzen
sns.set_theme("paper", style="whitegrid")
plt.rcParams['figure.dpi'] = 100


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
        """
        Initialisiert den MyopicCountryAnalyzer

        Parameters:
        -----------
        network_base_path : str
            Basis-Pfad zu den .nc Dateien (ohne _20XX.nc)
            z.B. "results/networks/base_s_37_"
        country : str
            Ländercode (z.B. "DE", "FR", "ES")
        years : List[int], optional
            Liste der zu analysierenden Jahre. Falls None, werden alle gefunden.
        output_dir : str
            Verzeichnis für die Ausgabe der Plots
        """
        self.network_base_path = Path(network_base_path)
        self.country = country
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Jahre bestimmen
        self.years = self._find_available_years() if years is None else sorted(years)
        print(f"Gefundene Jahre: {self.years}")

        # Netzwerke laden
        self.networks = {}
        self._load_networks()

        # Tech-Farben definieren (PyPSA-Eur Standard)
        self.tech_colors = {
            'solar': '#FFDD00',
            'onwind': '#74CBEB',
            'offwind': '#004E8A',
            'offwind-ac': '#004E8A',
            'offwind-dc': '#00589B',
            'gas': '#B08080',
            'OCGT': '#B08080',
            'CCGT': '#B08080',
            'H2': '#EA4CFA',
            'H2 Fuel Cell': '#EA4CFA',
            'nuclear': '#FF5200',
            'hydro': '#298DFF',
            'ror': '#298DFF',
            'PHS': '#298DFF',
            'biomass': '#0C6013',
            'coal': '#454545',
            'lignite': '#8B4513',
            'oil': '#8B0000',
            'battery': '#b36b00',
            'load': '#d63031'
        }

        self.colors = pd.Series(self.tech_colors)

        # Länderdaten für alle Jahre aufbereiten
        self._prepare_country_data()

        print(f"Myopic-Analyse für {self.country} initialisiert.")
        print(f"Jahre: {self.years}")
        print(f"Netzwerke geladen: {len(self.networks)}")

    def _find_available_years(self) -> List[int]:
        """Findet alle verfügbaren Jahre basierend auf Dateinamen"""
        # Pattern für Dateien wie "base_s_37_*_2030.nc"
        base_dir = self.network_base_path.parent
        base_name = self.network_base_path.name

        # Suche nach allen .nc Dateien mit Jahreszahlen
        pattern = str(base_dir / f"{base_name}*_20*.nc")
        files = glob.glob(pattern)

        years = []
        for file in files:
            # Extrahiere Jahr aus Dateiname
            match = re.search(r'_(\d{4})\.nc$', file)
            if match:
                years.append(int(match.group(1)))

        if not years:
            # Fallback: auch 2030, 2040, 2050 etc. suchen
            for year in range(2020, 2101, 5):
                potential_file = f"{self.network_base_path}_{year}.nc"
                if Path(potential_file).exists():
                    years.append(year)

        return sorted(years)

    def _load_networks(self):
        """Lädt alle verfügbaren Netzwerke"""
        print("Lade Netzwerke...")

        for year in self.years:
            # Verschiedene Dateinamen-Patterns probieren
            potential_paths = [
                f"{self.network_base_path}_{year}.nc",
                f"{self.network_base_path}{year}.nc",
                # Weitere Patterns falls nötig
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
        """Bereitet Länderdaten für alle Jahre auf"""
        self.country_data = {}

        for year, network in self.networks.items():
            data = {}

            # Busse des Landes
            data['buses'] = network.buses[network.buses.country == self.country].index

            # Komponenten des Landes
            data['generators'] = network.generators[
                network.generators.bus.isin(data['buses'])
            ]
            data['loads'] = network.loads[
                network.loads.bus.isin(data['buses'])
            ]
            data['stores'] = network.stores[
                network.stores.bus.isin(data['buses'])
            ]
            data['storage_units'] = network.storage_units[
                network.storage_units.bus.isin(data['buses'])
            ]

            # Grenzüberschreitende Links
            data['cross_border_links'] = network.links[
                (network.links.bus0.isin(data['buses']) &
                 ~network.links.bus1.isin(data['buses'])) |
                (~network.links.bus0.isin(data['buses']) &
                 network.links.bus1.isin(data['buses']))
                ]

            self.country_data[year] = data

    def get_capacity_evolution(self) -> pd.DataFrame:
        """Berechnet die Kapazitätsentwicklung über die Jahre"""
        capacity_data = []

        for year in self.years:
            network = self.networks[year]
            generators = self.country_data[year]['generators']

            if not generators.empty:
                capacities = generators.groupby('carrier')['p_nom_opt'].sum()
                for carrier, capacity in capacities.items():
                    capacity_data.append({
                        'Year': year,
                        'Carrier': carrier,
                        'Capacity_MW': capacity
                    })

        return pd.DataFrame(capacity_data)

    def get_energy_evolution(self) -> pd.DataFrame:
        """Berechnet die Energieentwicklung über die Jahre"""
        energy_data = []

        for year in self.years:
            network = self.networks[year]
            generators = self.country_data[year]['generators']
            loads = self.country_data[year]['loads']

            if not generators.empty:
                # Erzeugung nach Carrier
                generation = network.generators_t.p.loc[:, generators.index]
                gen_by_carrier = generation.groupby(generators.carrier, axis=1).sum().sum()

                for carrier, energy in gen_by_carrier.items():
                    energy_data.append({
                        'Year': year,
                        'Type': 'Generation',
                        'Carrier': carrier,
                        'Energy_MWh': energy
                    })

            if not loads.empty:
                # Nachfrage
                demand = network.loads_t.p.loc[:, loads.index].sum().sum()
                energy_data.append({
                    'Year': year,
                    'Type': 'Demand',
                    'Carrier': 'load',
                    'Energy_MWh': demand
                })

        return pd.DataFrame(energy_data)

    def plot_capacity_evolution(self, save=True):
        """1. Kapazitätsentwicklung über die Jahre - KORRIGIERT"""
        capacity_df = self.get_capacity_evolution()

        if capacity_df.empty:
            print(f"Keine Kapazitätsdaten für {self.country} gefunden.")
            return

        # Pivot für bessere Darstellung
        capacity_pivot = capacity_df.pivot(index='Year', columns='Carrier', values='Capacity_MW').fillna(0)

        # NUR für MEHR ALS EIN JAHR
        if len(self.years) < 2:
            print("Nur ein Jahr verfügbar - verwende Einzeljahr-Analyse")
            self._plot_single_year_capacity(capacity_pivot)
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Stacked Area Chart
        capacity_pivot.plot(kind='area', stacked=True, ax=axes[0, 0],
                            color=[self.colors.get(c, self._get_unique_color(c)) for c in capacity_pivot.columns])
        axes[0, 0].set_title("Kapazitätsentwicklung (Gestapelt)")
        axes[0, 0].set_ylabel("Kapazität [MW]")
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. Einzelne Linien
        capacity_pivot.plot(ax=axes[0, 1], marker='o',
                            color=[self.colors.get(c, self._get_unique_color(c)) for c in capacity_pivot.columns])
        axes[0, 1].set_title("Kapazitätsentwicklung (Einzeln)")
        axes[0, 1].set_ylabel("Kapazität [MW]")
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 3. Nur Erneuerbare
        renewables = ['solar', 'onwind', 'offwind', 'offwind-ac', 'offwind-dc', 'hydro', 'ror']
        renewable_cols = [col for col in capacity_pivot.columns if any(ren in col.lower() for ren in renewables)]
        if renewable_cols:
            capacity_pivot[renewable_cols].plot(kind='bar', ax=axes[1, 0],
                                                color=[self.colors.get(c, self._get_unique_color(c)) for c in
                                                       renewable_cols])
            axes[1, 0].set_title("Erneuerbare Energien Entwicklung")
            axes[1, 0].set_ylabel("Kapazität [MW]")
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 4. Kapazitätszuwachs pro Periode
        if len(self.years) > 1:
            capacity_growth = capacity_pivot.diff().fillna(capacity_pivot.iloc[0])  # Erste Zeile = Startwerte
            capacity_growth[capacity_growth < 0] = 0  # Nur Zuwachs zeigen
            capacity_growth.plot(kind='bar', stacked=True, ax=axes[1, 1],
                                 color=[self.colors.get(c, self._get_unique_color(c)) for c in capacity_growth.columns])
            axes[1, 1].set_title("Kapazitätszuwachs pro Periode")
            axes[1, 1].set_ylabel("Zuwachs [MW]")
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.suptitle(f"Kapazitätsentwicklung - {self.country}", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_capacity_evolution.png",
                        dpi=300, bbox_inches='tight')
        plt.close()

        return capacity_pivot

    def _get_unique_color(self, carrier):
        """Generiert eindeutige Farben für unbekannte Carrier"""
        import hashlib
        # Hash basierend auf Carrier-Name für konsistente Farben
        hash_object = hashlib.md5(carrier.encode())
        hex_dig = hash_object.hexdigest()
        return f'#{hex_dig[:6]}'

    def _plot_single_year_capacity(self, capacity_pivot):
        """Spezielle Darstellung für nur ein Jahr"""
        year = capacity_pivot.index[0]
        capacities = capacity_pivot.iloc[0]

        # Nur signifikante Kapazitäten (> 100 MW)
        significant_capacities = capacities[capacities > 100].sort_values(ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Balkendiagramm
        colors = [self.colors.get(c, self._get_unique_color(c)) for c in significant_capacities.index]
        significant_capacities.plot(kind='bar', ax=axes[0], color=colors)
        axes[0].set_title(f"Installierte Kapazitäten {year}")
        axes[0].set_ylabel("Kapazität [MW]")
        axes[0].tick_params(axis='x', rotation=45)

        # Pie Chart
        if len(significant_capacities) > 0:
            axes[1].pie(significant_capacities.values,
                        labels=significant_capacities.index,
                        colors=colors,
                        autopct='%1.1f%%')
            axes[1].set_title(f"Kapazitätsverteilung {year}")

        plt.suptitle(f"Kapazitätsanalyse {year} - {self.country}", fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{self.country}_capacity_single_year.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_energy_evolution(self, save=True):
        """2. Energieentwicklung über die Jahre"""
        energy_df = self.get_energy_evolution()

        if energy_df.empty:
            print(f"Keine Energiedaten für {self.country} gefunden.")
            return

        # Separate DataFrames für Generation und Demand
        generation_df = energy_df[energy_df['Type'] == 'Generation']
        demand_df = energy_df[energy_df['Type'] == 'Demand']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        if not generation_df.empty:
            # 1. Erzeugung über Zeit
            gen_pivot = generation_df.pivot(index='Year', columns='Carrier', values='Energy_MWh').fillna(0)
            gen_pivot_twh = gen_pivot / 1e6  # Convert to TWh

            gen_pivot_twh.plot(kind='area', stacked=True, ax=axes[0, 0],
                               color=[self.colors.get(c, 'grey') for c in gen_pivot_twh.columns])
            axes[0, 0].set_title("Stromerzeugung Entwicklung")
            axes[0, 0].set_ylabel("Erzeugung [TWh]")
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # 2. Relativer Anteil
            gen_share = gen_pivot_twh.div(gen_pivot_twh.sum(axis=1), axis=0) * 100
            gen_share.plot(kind='area', stacked=True, ax=axes[0, 1],
                           color=[self.colors.get(c, 'grey') for c in gen_share.columns])
            axes[0, 1].set_title("Erzeugungsanteile")
            axes[0, 1].set_ylabel("Anteil [%]")
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 3. Nachfrage vs. Erzeugung
        if not demand_df.empty and not generation_df.empty:
            total_generation = generation_df.groupby('Year')['Energy_MWh'].sum() / 1e6
            total_demand = demand_df.groupby('Year')['Energy_MWh'].sum() / 1e6

            balance_df = pd.DataFrame({
                'Erzeugung': total_generation,
                'Nachfrage': total_demand
            })
            balance_df.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title("Erzeugung vs. Nachfrage")
            axes[1, 0].set_ylabel("Energie [TWh]")

            # 4. Energiebilanz
            balance = total_generation - total_demand
            balance.plot(kind='bar', ax=axes[1, 1],
                         color=['green' if x >= 0 else 'red' for x in balance])
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_title("Energiebilanz (Überschuss/Defizit)")
            axes[1, 1].set_ylabel("Bilanz [TWh]")

        plt.suptitle(f"Energieentwicklung - {self.country}", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_energy_evolution.png",
                        dpi=300, bbox_inches='tight')
        plt.show()

        return energy_df

    def plot_yearly_comparison(self, metric='capacity', carriers=None, save=True):
        """3. Jahresvergleich für spezifische Metriken"""
        if carriers is None:
            carriers = ['solar', 'onwind', 'offwind', 'gas', 'nuclear', 'hydro']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, carrier in enumerate(carriers[:6]):  # Max 6 plots
            if i >= len(axes):
                break

            data_series = []

            for year in self.years:
                generators = self.country_data[year]['generators']
                carrier_gens = generators[generators.carrier.str.contains(carrier, case=False)]

                if not carrier_gens.empty:
                    if metric == 'capacity':
                        value = carrier_gens['p_nom_opt'].sum()
                        unit = 'MW'
                    elif metric == 'energy':
                        network = self.networks[year]
                        value = network.generators_t.p.loc[:, carrier_gens.index].sum().sum() / 1e6
                        unit = 'TWh'
                    elif metric == 'capacity_factor':
                        network = self.networks[year]
                        generation = network.generators_t.p.loc[:, carrier_gens.index].sum().sum()
                        max_generation = (carrier_gens['p_nom_opt'].sum() *
                                          len(network.snapshots))
                        value = (generation / max_generation * 100) if max_generation > 0 else 0
                        unit = '%'
                    else:
                        value = 0
                        unit = ''

                    data_series.append(value)
                else:
                    data_series.append(0)

            # Plot
            ax = axes[i]
            bars = ax.bar(self.years, data_series, color=self.colors.get(carrier, 'grey'))
            ax.set_title(f"{carrier.title()} - {metric.replace('_', ' ').title()}")
            ax.set_ylabel(f"{metric.replace('_', ' ').title()} [{unit}]")

            # Werte auf Balken
            for bar, value in zip(bars, data_series):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(data_series) * 0.01,
                            f'{value:.1f}', ha='center', va='bottom')

        # Leere Subplots ausblenden
        for i in range(len(carriers), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f"{metric.replace('_', ' ').title()} Entwicklung - {self.country}", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_{metric}_comparison.png",
                        dpi=300, bbox_inches='tight')
        plt.show()

    def plot_investment_timeline(self, save=True):
        """4. Investitions-Timeline (was wird wann gebaut)"""
        investment_data = []

        # Kapazitätsdifferenzen zwischen Jahren berechnen
        for i, year in enumerate(self.years):
            generators = self.country_data[year]['generators']

            if i == 0:
                # Erstes Jahr: Gesamte optimierte Kapazität als "Investment"
                investments = generators.groupby('carrier')['p_nom_opt'].sum()
            else:
                # Folgende Jahre: Differenz zur vorherigen Periode
                prev_year = self.years[i - 1]
                prev_generators = self.country_data[prev_year]['generators']

                current_cap = generators.groupby('carrier')['p_nom_opt'].sum()
                prev_cap = prev_generators.groupby('carrier')['p_nom_opt'].sum()

                # Reindex um fehlende Carrier zu berücksichtigen
                all_carriers = set(current_cap.index) | set(prev_cap.index)
                current_cap = current_cap.reindex(all_carriers, fill_value=0)
                prev_cap = prev_cap.reindex(all_carriers, fill_value=0)

                investments = current_cap - prev_cap
                investments = investments[investments > 0]  # Nur positive Investitionen

            for carrier, investment in investments.items():
                if investment > 0:
                    investment_data.append({
                        'Year': year,
                        'Carrier': carrier,
                        'Investment_MW': investment,
                        'Period': f"{year - 5 if i > 0 else 2020}-{year}"
                    })

        if not investment_data:
            print(f"Keine Investitionsdaten für {self.country} gefunden.")
            return

        investment_df = pd.DataFrame(investment_data)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Investitionen pro Jahr (gestapelt)
        inv_pivot = investment_df.pivot(index='Year', columns='Carrier', values='Investment_MW').fillna(0)
        inv_pivot.plot(kind='bar', stacked=True, ax=axes[0, 0],
                       color=[self.colors.get(c, 'grey') for c in inv_pivot.columns])
        axes[0, 0].set_title("Investitionen pro Planungsperiode")
        axes[0, 0].set_ylabel("Neue Kapazität [MW]")
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. Kumulierte Investitionen
        inv_cumsum = inv_pivot.cumsum()
        inv_cumsum.plot(ax=axes[0, 1], marker='o',
                        color=[self.colors.get(c, 'grey') for c in inv_cumsum.columns])
        axes[0, 1].set_title("Kumulierte Investitionen")
        axes[0, 1].set_ylabel("Kumulierte Kapazität [MW]")
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 3. Top Investitionen pro Technologie
        total_inv_by_carrier = investment_df.groupby('Carrier')['Investment_MW'].sum().sort_values(ascending=False)
        total_inv_by_carrier.head(8).plot(kind='bar', ax=axes[1, 0],
                                          color=[self.colors.get(c, 'grey') for c in
                                                 total_inv_by_carrier.head(8).index])
        axes[1, 0].set_title("Gesamtinvestitionen nach Technologie")
        axes[1, 0].set_ylabel("Investition [MW]")

        # 4. Investment Rate (MW pro Jahr)
        if len(self.years) > 1:
            period_length = self.years[1] - self.years[0]  # Annahme: gleichmäßige Abstände
            investment_rate = inv_pivot / period_length
            investment_rate.plot(kind='area', stacked=True, ax=axes[1, 1],
                                 color=[self.colors.get(c, 'grey') for c in investment_rate.columns])
            axes[1, 1].set_title("Jährliche Investitionsrate")
            axes[1, 1].set_ylabel("MW pro Jahr")
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.suptitle(f"Investitions-Timeline - {self.country}", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_investment_timeline.png",
                        dpi=300, bbox_inches='tight')
        plt.show()

        return investment_df

    def plot_cross_border_evolution(self, save=True):
        """5. Entwicklung der grenzüberschreitenden Stromflüsse"""
        flow_data = []

        for year in self.years:
            network = self.networks[year]
            cross_border_links = self.country_data[year]['cross_border_links']

            if cross_border_links.empty:
                continue

            for link_id in cross_border_links.index:
                link = cross_border_links.loc[link_id]
                flows = network.links_t.p0.loc[:, link_id]

                # Bestimme Nachbarland
                country_buses = self.country_data[year]['buses']
                if link.bus0 in country_buses:
                    neighbor = link.bus1[:2]  # Erste 2 Zeichen als Ländercode
                    flow_direction = -1  # Negative für Export
                else:
                    neighbor = link.bus0[:2]
                    flow_direction = 1  # Positive für Import

                # Jahresstatistiken
                total_flow = flows.sum() * flow_direction / 1e6  # TWh
                avg_flow = flows.mean() * flow_direction
                max_flow = flows.abs().max()

                flow_data.append({
                    'Year': year,
                    'Neighbor': neighbor,
                    'Total_Flow_TWh': total_flow,
                    'Avg_Flow_MW': avg_flow,
                    'Max_Flow_MW': max_flow,
                    'Link_Capacity_MW': link.p_nom_opt
                })

        if not flow_data:
            print(f"Keine grenzüberschreitenden Flows für {self.country} gefunden.")
            return

        flow_df = pd.DataFrame(flow_data)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Jährliche Stromflüsse nach Nachbarland
        flow_pivot = flow_df.pivot(index='Year', columns='Neighbor', values='Total_Flow_TWh').fillna(0)
        flow_pivot.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 0].set_title("Jährliche Stromflüsse nach Land")
        axes[0, 0].set_ylabel("Netto-Flow [TWh] (+ Import, - Export)")
        axes[0, 0].legend(title="Nachbarland")

        # 2. Übertragungskapazität Entwicklung
        capacity_pivot = flow_df.pivot(index='Year', columns='Neighbor', values='Link_Capacity_MW').fillna(0)
        capacity_pivot.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title("Übertragungskapazität Entwicklung")
        axes[0, 1].set_ylabel("Kapazität [MW]")
        axes[0, 1].legend(title="Nachbarland")

        # 3. Netto Import/Export Bilanz
        total_flow_by_year = flow_df.groupby('Year')['Total_Flow_TWh'].sum()
        colors = ['green' if x >= 0 else 'red' for x in total_flow_by_year]
        total_flow_by_year.plot(kind='bar', ax=axes[1, 0], color=colors)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title("Netto Import/Export Bilanz")
        axes[1, 0].set_ylabel("Netto-Flow [TWh]")

        # 4. Kapazitätsauslastung
        flow_df['Utilization'] = abs(flow_df['Avg_Flow_MW']) / flow_df['Link_Capacity_MW'] * 100
        util_pivot = flow_df.pivot(index='Year', columns='Neighbor', values='Utilization').fillna(0)
        util_pivot.plot(kind='line', marker='o', ax=axes[1, 1])
        axes[1, 1].set_title("Leitungsauslastung")
        axes[1, 1].set_ylabel("Auslastung [%]")
        axes[1, 1].legend(title="Nachbarland")

        plt.suptitle(f"Grenzüberschreitende Stromflüsse - {self.country}", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / f"{self.country}_cross_border_evolution.png",
                        dpi=300, bbox_inches='tight')
        plt.show()

        return flow_df

    def analyze_single_year(self, year: int, create_detailed_plots: bool = True):
        """Detailanalyse für ein spezifisches Jahr"""
        if year not in self.years:
            print(f"Jahr {year} nicht verfügbar. Verfügbare Jahre: {self.years}")
            return None

        print(f"\n=== Detailanalyse für {self.country} - {year} ===")

        # Erstelle SingleYearAnalyzer für detaillierte Plots
        from country_analysis import CountryAnalyzer  # Import der ursprünglichen Klasse

        network_path = f"{self.network_base_path}_{year}.nc"
        if not Path(network_path).exists():
            # Alternative Pfade probieren
            alt_paths = [f"{self.network_base_path}{year}.nc"]
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    network_path = alt_path
                    break

        single_analyzer = CountryAnalyzer(
            network_path=network_path,
            country=self.country,
            output_dir=f"{self.output_dir}/{year}"
        )

        # Zusammenfassungsbericht
        single_analyzer.summary_report()

        if create_detailed_plots:
            # Erstelle alle detaillierten Plots für dieses Jahr
            single_analyzer.generate_all_plots(save=True)

        return single_analyzer

    def generate_all_myopic_plots(self, save=True):
        """Generiert alle myopic-spezifischen Plots"""
        print(f"\n=== Generiere alle Myopic-Plots für {self.country} ===")

        plot_functions = [
            ("Kapazitätsentwicklung", self.plot_capacity_evolution),
            ("Energieentwicklung", self.plot_energy_evolution),
            ("Investitions-Timeline", self.plot_investment_timeline),
            ("Grenzüberschreitende Flows", self.plot_cross_border_evolution),
        ]

        for name, func in plot_functions:
            try:
                print(f"Erstelle: {name}")
                func(save=save)
            except Exception as e:
                print(f"Fehler bei {name}: {e}")

        # Zusätzliche Vergleichsplots
        try:
            print("Erstelle: Kapazitätsvergleich")
            self.plot_yearly_comparison(metric='capacity', save=save)
            print("Erstelle: Energievergleich")
            self.plot_yearly_comparison(metric='energy', save=save)
            print("Erstelle: Kapazitätsfaktor-Vergleich")
            self.plot_yearly_comparison(metric='capacity_factor', save=save)
        except Exception as e:
            print(f"Fehler bei Vergleichsplots: {e}")

        print(f"\nAlle Myopic-Plots gespeichert in: {self.output_dir}")

    def summary_report_myopic(self):
        """Erstellt einen zusammenfassenden Bericht über alle Jahre"""
        print(f"\n{'=' * 60}")
        print(f"MYOPIC ANALYSEBERICHT FÜR {self.country}")
        print(f"{'=' * 60}")

        print(f"\nPlanungshorizont: {min(self.years)} - {max(self.years)}")
        print(f"Analysierte Jahre: {', '.join(map(str, self.years))}")
        print(f"Anzahl Planungsperioden: {len(self.years)}")

        # Kapazitätsentwicklung
        capacity_df = self.get_capacity_evolution()
        if not capacity_df.empty:
            print(f"\n{'=' * 40}")
            print("KAPAZITÄTSENTWICKLUNG")
            print(f"{'=' * 40}")

            capacity_pivot = capacity_df.pivot(index='Year', columns='Carrier', values='Capacity_MW').fillna(0)

            print(f"Installierte Kapazität {max(self.years)} [MW]:")
            final_capacities = capacity_pivot.iloc[-1].sort_values(ascending=False)
            for carrier, capacity in final_capacities.items():
                if capacity > 0:
                    print(f"  {carrier}: {capacity:.0f} MW")

            # Wachstumsraten
            if len(self.years) > 1:
                print(f"\nDurchschnittliches jährliches Wachstum:")
                total_years = max(self.years) - min(self.years)
                for carrier in capacity_pivot.columns:
                    initial = capacity_pivot.loc[min(self.years), carrier]
                    final = capacity_pivot.loc[max(self.years), carrier]
                    if initial > 0:
                        growth_rate = ((final / initial) ** (1 / total_years) - 1) * 100
                        print(f"  {carrier}: {growth_rate:.1f}% p.a.")

        # Energieentwicklung
        energy_df = self.get_energy_evolution()
        if not energy_df.empty:
            print(f"\n{'=' * 40}")
            print("ENERGIEENTWICKLUNG")
            print(f"{'=' * 40}")

            generation_df = energy_df[energy_df['Type'] == 'Generation']
            demand_df = energy_df[energy_df['Type'] == 'Demand']

            for year in self.years:
                year_gen = generation_df[generation_df['Year'] == year]['Energy_MWh'].sum() / 1e6
                year_dem = demand_df[demand_df['Year'] == year]['Energy_MWh'].sum() / 1e6
                balance = year_gen - year_dem
                print(f"{year}: Erzeugung {year_gen:.1f} TWh, Nachfrage {year_dem:.1f} TWh, Bilanz {balance:.1f} TWh")


def main():
    """Beispiel für die Verwendung der Myopic-Analyse"""
    # Basis-Pfad zu Ihren .nc Dateien (ohne _20XX.nc Endung)
    network_base_path = "/home/endata/pypsa-eur/results/masterarbeit_myopic_heating_transport_biomass/networks/base_s_18__H-T-B_"

    # Analyzer für Deutschland erstellen
    analyzer = MyopicCountryAnalyzer(
        network_base_path=network_base_path,
        country="DE",
        years=None,  # Alle verfügbaren Jahre
        output_dir="/shared_endata/atlite_cutouts/results/PyPSA-eur results/Plots/testrun_1997"
    )

    # Zusammenfassungsbericht
    analyzer.summary_report_myopic()

    # Alle Myopic-Plots erstellen
    analyzer.generate_all_myopic_plots(save=True)

    # Detailanalyse für spezifisches Jahr (optional)
    # analyzer.analyze_single_year(2030, create_detailed_plots=True)


if __name__ == "__main__":
    main()