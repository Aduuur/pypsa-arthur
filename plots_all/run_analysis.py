"""
Ausführungsskript für die PyPSA-Eur Länderanalyse
"""

from country_analysis import CountryAnalyzer
from pathlib import Path


def analyze_country(network_path: str, country: str, create_all_plots: bool = True):
    """
    Führt eine vollständige Analyse für ein Land durch

    Parameters:
    -----------
    network_path : str
        Pfad zur .nc Datei
    country : str
        Ländercode (z.B. "DE", "FR", "ES")
    create_all_plots : bool
        Ob alle Plots erstellt werden sollen
    """
    print(f"Starte Analyse für {country}...")

    # Output-Verzeichnis erstellen
    output_dir = f"/shared_endata/atlite_cutouts/results/PyPSA-eur results/Plots/testrun_1997/analysis_{country}"

    # Analyzer initialisieren
    analyzer = CountryAnalyzer(
        network_path=network_path,
        country=country,
        output_dir=output_dir
    )

    # Zusammenfassungsbericht
    analyzer.summary_report()

    if create_all_plots:
        # Alle Plots erstellen
        analyzer.generate_all_plots(save=True)
    else:
        # Nur spezifische Plots erstellen (Beispiel)
        analyzer.plot_installed_capacity()
        analyzer.plot_supply_demand_balance()

    print(f"Analyse für {country} abgeschlossen. Ergebnisse in: {output_dir}")

    return analyzer


def analyze_multiple_countries(network_path: str, countries: list):
    """
    Analysiert mehrere Länder nacheinander
    """
    results = {}

    for country in countries:
        try:
            results[country] = analyze_country(network_path, country)
            print(f"✓ {country} erfolgreich analysiert")
        except Exception as e:
            print(f"✗ Fehler bei {country}: {e}")
            results[country] = None

    return results


def main():
    """Hauptausführung"""

    # ========================================
    # KONFIGURATION - HIER ANPASSEN
    # ========================================

    # Pfad zu Ihrer .nc Datei
    NETWORK_PATH = "/home/endata/pypsa-eur/results/masterarbeit_myopic_heating_transport_biomass/networks/base_s_18__H-T-B_2030.nc"

    # Welche Länder analysieren? (ISO-2 Codes)
    COUNTRIES = ["DE", "FR", "ES", "IT", "NL", "CH"]

    # Alle Plots erstellen oder nur ausgewählte?
    CREATE_ALL_PLOTS = True

    # ========================================

    # Prüfen ob Datei existiert
    if not Path(NETWORK_PATH).exists():
        print(f"FEHLER: Datei {NETWORK_PATH} nicht gefunden!")
        print("Bitte den Pfad in der NETWORK_PATH Variable anpassen.")
        return

    print("PyPSA-Eur Länderanalyse")
    print("=" * 40)
    print(f"Netzwerk: {NETWORK_PATH}")
    print(f"Länder: {', '.join(COUNTRIES)}")
    print("=" * 40)

    # Einzelne Länder analysieren
    if len(COUNTRIES) == 1:
        analyzer = analyze_country(NETWORK_PATH, COUNTRIES[0], CREATE_ALL_PLOTS)
    else:
        # Mehrere Länder
        results = analyze_multiple_countries(NETWORK_PATH, COUNTRIES)

        # Kurze Zusammenfassung
        print("\n" + "=" * 40)
        print("ZUSAMMENFASSUNG")
        print("=" * 40)
        for country, result in results.items():
            if result is not None:
                print(f"✓ {country}: Analyse erfolgreich")
            else:
                print(f"✗ {country}: Fehler bei Analyse")


if __name__ == "__main__":
    main()