# config.py
# Zentrale Konfigurationsdatei für die PyPSA-Plotting-Skripte.

import os


class PlottingConfig:
    """
    Eine flexible Konfigurationsklasse, die mehrere Skripte unterstützt.
    Pfade, Länderlisten, Schriftgrößen und Farbpaletten werden hier definiert.
    """

    # =========================================================================
    # --- ZENTRALE SZENARIO-AUSWAHL ---
    SCENARIO_SELECTION =  "dunkelflaute_neu_2" #dunkelflaute_neu_2" #both" #dunkelflaute_neu_2

    # =========================================================================

    def __init__(self):
        """
        Konstruktor: Definiert alle instanzspezifischen Variablen.
        """
        self.BASE_NETWORK_PATH = "/home/endata/pypsa-eur/results"
        self.BASE_SAVE_PATH = '/shared_endata/atlite_cutouts/results/PyPSA-eur results/final_plots_FINAL'

        self.PLOT_OUTPUT_PATH = os.path.join(self.BASE_SAVE_PATH, "plots_combined")

        self.COUNTRIES_TO_PLOT = ["DE", "FR", "ES", "CH", "ALL", "DK", "SE", "NO", "IT", "GB", "NL", "PL", "BE", "FI", "AT", "PT","CZ", "LT", "LV", "EE"]
        self.FONT_SIZES = {
            "title": 20, "label": 15, "tick": 13, "bar_label": 10, "legend": 11,
        }

        # HINWEIS: Das 'CARRIER_COLORS' Dictionary wurde um die fehlenden
        # gruppierten Namen und neuen Carrier erweitert.
        self.CARRIER_COLORS = {
            # Wind
            'onwind': "#235ebc",
            'onshore wind': "#235ebc",  # NEU: Hinzugefügt für die Gruppierung
            'offwind-ac': "#6895dd",
            'offwind-dc': "#74c6f2",
            'offshore wind': "#6895dd",  # NEU: Hinzugefügt für die Gruppierung
            'offwind-float': "#15a0bf",  # NEU: Für schwimmende Offshore-Anlagen

            # === Biomasse (detailliert) ===
            'biogas': '#e3d37d',  # Gelb-Grau
            'biomass': '#baa741',  # Oliv
            'solid biomass': '#baa741',  # Oliv
            'municipal solid waste': '#91ba41',  # Gelbgrün
            'solid biomass import': '#d5ca8d',  # Helleres Oliv
            'solid biomass transport': '#baa741',  # Oliv
            'solid biomass for industry': '#7a6d26',  # Dunkelbraun
            'solid biomass for industry CC': '#47411c',  # Sehr dunkles Braun
            'solid biomass for industry co2 from atmosphere': '#736412',  # Braun
            'solid biomass for industry co2 to stored': '#47411c',  # Sehr dunkles Braun
            'urban central solid biomass CHP': '#9d9042',  # Mittleres Oliv
            'urban central solid biomass CHP CC': '#6c5d28',  # Dunkles Oliv
            'biomass boiler': '#8A9A5B',  # Graugrün
            'residential rural biomass boiler': '#a1a066',  # Helles Graugrün
            'residential urban decentral biomass boiler': '#b0b87b',  # Helleres Graugrün
            'services rural biomass boiler': '#c6cf98',  # Sehr helles Graugrün
            'services urban decentral biomass boiler': '#dde5b5',  # Sehr helles Graugrün
            'biomass to liquid': '#32CD32',  # Lime Green
            'unsustainable solid biomass': '#998622',  # Gelbbraun
            'unsustainable bioliquids': '#32CD32',  # Lime Green
            'electrobiofuels': '#ff0000',  # Rot
            'BioSNG': '#123456',  # Dunkles Blaugrau
            'BioSNG CC': '#45233b',  # Dunkles Lila
            'solid biomass to hydrogen': '#654321',  # Braun
            'Biomasse': '#baa741',

            # Wasser
            'hydro': '#298c81',
            'ror': '#3dbfb0',
            'run of river': '#3dbfb0',  # NEU: Hinzugefügt für die Gruppierung
            'PHS': '#51dbcc',

            # Solar
            'solar': "#f9d002",
            'solar rooftop': '#ffea80',

            # Konventionell / Gas / Öl
            'OCGT': '#e0986c',
            'CCGT': '#a85522',
            'gas': '#e05b09',
            'oil': '#c9c9c9',
            'oil primary': '#7a7a7a',  # NEU: Hinzugefügt
            'nuclear': '#ff8c00',
            'coal': '#545454',
            'lignite': '#826837',

            # Biomasse
            'biomass': '#baa741',
            'solid biomass': '#baa741',
            'biogas': '#e3d37d',
            'waste': '#e3d37d',

            # Speicher & Wasserstoff
            'battery': '#ace37f',
            'home battery': '#80c944',
            'V2G': '#e5ffa8',
            'H2 turbine': '#991f83',
            'H2 Fuel Cell': '#c251ae',
            'hydrogen': '#bf13a0',  # NEU: Hinzugefügt für die Gruppierung
            'H2': '#bf13a0',
            'H2 Electrolysis': '#ff29d9',
            'H2 storage': '#bf13a0',

            # Sonstige
            'geothermal': '#ba91b1',
            'other': '#000000',

            # Stromverbrauch / Sektoren
            'load': '#1f77b4',  # Stromlast (dunkelblau)
            'battery charger': '#76c7a3',  # Batterie laden
            'H2 Electrolysis': '#e066ff',  # Elektrolyse (kräftig magenta)
            'methanation': '#a349a4',  # Methanisierung (violett)
            'Haber-Bosch': '#b565b0',  # Ammoniak-Synthese
            'heat pump': '#ff9966',  # Wärmepumpen
            'resistive heater': '#ff7f50',  # Widerstandsheizung
            'EV charger': '#9999ff',  # Elektrofahrzeuge
            'export': '#808080',  # Stromexport

            "battery discharger": "#2a9d8f",  # Ein sattes Petrolgrün für große Batterien
            "home battery discharger": "#8ecae6",  # Ein helleres Blau/Türkis (könnte man anpassen)
            # Alternativ für ein helleres Grün: "#90be6d"





        }
        self.DEFAULT_COLOR = '#a9a9a9'


    # --- DATENBANK ALLER VERFÜGBAREN SZENARIEN ---
    # (Dieser Teil bleibt unverändert)
    SCENARIOS = {
        "dunkelflaute_neu_2": [
            "/home/endata/pypsa-eur/results/start_new_myopic_28_bigrun_dunkelflaute_transmission_limited/networks/base_s_24___2025.nc",
            "/home/endata/pypsa-eur/results/start_new_myopic_28_bigrun_dunkelflaute_transmission_limited/networks/base_s_24___2030.nc",
            "/home/endata/pypsa-eur/results/start_new_myopic_28_bigrun_dunkelflaute_transmission_limited/networks/base_s_24___2035.nc",
            "/home/endata/pypsa-eur/results/start_new_myopic_28_bigrun_dunkelflaute_transmission_limited/networks/base_s_24___2040.nc",
            "/home/endata/pypsa-eur/results/start_new_myopic_28_bigrun_dunkelflaute_transmission_limited/networks/base_s_24___2045.nc",
            "/home/endata/pypsa-eur/results/start_new_myopic_28_bigrun_dunkelflaute_transmission_limited/networks/base_s_24___2050.nc"
        ],
        "new_avg": [
        "/home/endata/pypsa-eur/results/start_new_myopic_29_bigrun_transmission_limited/networks/base_s_24___2025.nc",
        "/home/endata/pypsa-eur/results/start_new_myopic_29_bigrun_transmission_limited/networks/base_s_24___2030.nc",
        "/home/endata/pypsa-eur/results/start_new_myopic_29_bigrun_transmission_limited/networks/base_s_24___2035.nc",
        "/home/endata/pypsa-eur/results/start_new_myopic_29_bigrun_transmission_limited/networks/base_s_24___2040.nc",
        "/home/endata/pypsa-eur/results/start_new_myopic_29_bigrun_transmission_limited/networks/base_s_24___2045.nc",
        "/home/endata/pypsa-eur/results/start_new_myopic_29_bigrun_transmission_limited/networks/base_s_24___2050.nc"]

    }

    def get_networks(self):
        """
        Gibt Netzwerkdateien basierend auf der oben definierten 'SCENARIO_SELECTION' Variable zurück.
        """
        selection = self.SCENARIO_SELECTION

        if selection == "both":
            print("Konfiguration im 'both'-Modus: Lade 'average' und 'dunkelflaute'.")
            return {
                'average': self.SCENARIOS.get('new_avg', []),
                'dunkelflaute': self.SCENARIOS.get('dunkelflaute_neu_2', [])
            }
        elif selection in self.SCENARIOS:
            print(f"Konfiguration im Single-Modus: Lade Szenario '{selection}'.")
            return self.SCENARIOS[selection]
        else:
            print(
                f"FEHLER: Das in 'SCENARIO_SELECTION' gewählte Szenario '{selection}' ist nicht in config.py definiert.")
            return None

    # =========================================================================
    # --- PLOTTING-STEUERUNG ---
    # =========================================================================

    # Welche Länder sollen geplottet werden:
    # -> Liste einzelner Länder (["DE", "FR", "ES"]) oder "ALL" für gesamtes Netzwerk
    COUNTRIES_TO_PLOT = ["ALL", "DE", "FR", "ES", "CH", "DK"]

    # Welche Plots sollen generiert werden:
    # -> "all" = alle bekannten Plots
    # -> Oder spezifische Liste, z. B. ["installed_capacity", "price", "storage_usage"]
    PLOTS_TO_RUN = "all"

    # Mapping aller verfügbaren Plottypen auf ihre Modulnamen oder Skripte
    AVAILABLE_PLOTS = {
        "installed_capacity": "plot_installed_capacity",
        "energy_generated": "plot_energy_generated",
        "storage": "plot_storage",
        "price": "plot_price",
        "losses": "plot_losses",
        "demand_vs_generation": "plot_demand_vs_generation",
        "price_setting_unit": "plot_price_setting_unit",
        "gas_h2_usage": "plot_gas_h2_usage",
        "capacity_expansion": "plot_capacity_expansion",
        "demand_profile": "plot_demand_profile",
        "generation_profile": "plot_generation_profile",
    }

    # Ausgabeordnerstruktur (automatisch pro Plot & Land)
    # =========================================================================
    # --- HELFERFUNKTIONEN ---
    # =========================================================================

    def get_countries(self):
        """
        Gibt die Länder zurück, die geplottet werden sollen.
        Falls 'ALL' angegeben ist, wird ein Platzhalter für Gesamtnetzwerk ergänzt.
        """
        if "ALL" in self.COUNTRIES_TO_PLOT:
            return ["ALL"] + [c for c in self.COUNTRIES_TO_PLOT if c != "ALL"]
        return self.COUNTRIES_TO_PLOT

    def get_plots_to_run(self):
        """
        Gibt die zu startenden Plots zurück (entweder alle oder ausgewählte).
        """
        if self.PLOTS_TO_RUN == "all":
            return list(self.AVAILABLE_PLOTS.keys())
        return [p for p in self.PLOTS_TO_RUN if p in self.AVAILABLE_PLOTS]


