"""
Ausführungsskript für die PyPSA-Eur Myopic Länderanalyse
"""

from country_analysis_myopic import MyopicCountryAnalyzer
from pathlib import Path
import sys


def find_network_files(base_directory: str, pattern: str = "base_s_*"):
    """
    Findet PyPSA-Eur Netzwerk-Dateien in einem Verzeichnis

    Parameters:
    -----------
    base_directory : str
        Verzeichnis zum Suchen
    pattern : str
        Suchmuster für Dateien

    Returns:
    --------
    dict : Dictionary mit gefundenen Dateien
    """
    base_dir = Path(base_directory)

    if not base_dir.exists():
        print(f"Verzeichnis {base_directory} existiert nicht!")
        return {}

    # Suche nach .nc Dateien mit Jahreszahlen
    network_files = {}

    for nc_file in base_dir.glob("*.nc"):
        filename = nc_file.stem

        # Extrahiere Jahr aus Dateiname
        import re
        year_match = re.search(r'_(\d{4})$', filename)
        if year_match:
            year = int(year_match.group(1))
            base_name = filename[:-5]  # Entferne _YYYY

            if base_name not in network_files:
                network_files[base_name] = {}
            network_files[base_name][year] = str(nc_file)

    return network_files


def analyze_myopic_country(base_path: str, country: str, years: list = None):
    """
    Führt eine vollständige Myopic-Analyse für ein Land durch
    """
    print(f"Starte Myopic-Analyse für {country}...")
    print(f"Basis-Pfad: {base_path}")

    # Output-Verzeichnis erstellen
    output_dir = f"myopic_analysis_{country}"

    try:
        # Analyzer initialisieren
        analyzer = MyopicCountryAnalyzer(
            network_base_path=base_path,
            country=country,
            years=years,
            output_dir=output_dir
        )

        # Zusammenfassungsbericht
        analyzer.summary_report_myopic()

        # Alle Myopic-Plots erstellen
        analyzer.generate_all_myopic_plots(save=True)

        print(f"✓ Myopic-Analyse für {country} abgeschlossen.")
        print(f"  Ergebnisse in: {output_dir}")

        # Frage nach Detailanalyse für einzelne Jahre
        while True:
            user_input = input(f"\nDetailanalyse für einzelnes Jahr erstellen? (Jahr eingeben oder 'n' für nein): ")
            if user_input.lower() in ['n', 'no', 'nein', '']:
                break
            try:
                detail_year = int(user_input)
                if detail_year in analyzer.years:
                    print(f"Erstelle Detailanalyse für {detail_year}...")
                    analyzer.analyze_single_year(detail_year, create_detailed_plots=True)
                    print(f"✓ Detailanalyse für {detail_year} abgeschlossen.")
                else:
                    print(f"Jahr {detail_year} nicht verfügbar. Verfügbare Jahre: {analyzer.years}")
            except ValueError:
                print("Bitte geben Sie eine gültige Jahreszahl ein.")

        return analyzer

    except Exception as e:
        print(f"✗ Fehler bei Myopic-Analyse für {country}: {e}")
        return None


def interactive_setup():
    """Interaktives Setup für die Analyse"""
    print("PyPSA-Eur Myopic Länderanalyse - Interaktives Setup")
    print("=" * 60)

    # 1. Verzeichnis eingeben
    while True:
        network_dir = input("\nGeben Sie das Verzeichnis mit den .nc Dateien ein: ").strip()
        if Path(network_dir).exists():
            break
        print(f"Verzeichnis '{network_dir}' existiert nicht. Bitte erneut versuchen.")

    # 2. Verfügbare Netzwerk-Dateien finden
    print(f"\nSuche nach Netzwerk-Dateien in {network_dir}...")
    network_files = find_network_files(network_dir)

    if not network_files:
        print("Keine PyPSA-Eur Netzwerk-Dateien gefunden!")
        print("Dateien sollten das Format 'basename_YYYY.nc' haben")
        return None

    # 3. Basis-Netzwerk auswählen
    print(f"\nGefundene Netzwerk-Serien:")
    for i, (base_name, years_dict) in enumerate(network_files.items()):
        years = sorted(years_dict.keys())
        print(f"{i + 1}. {base_name} (Jahre: {years})")

    while True:
        try:
            choice = int(input(f"\nWählen Sie eine Serie (1-{len(network_files)}): ")) - 1
            if 0 <= choice < len(network_files):
                selected_base = list(network_files.keys())[choice]
                selected_files = network_files[selected_base]
                break
            else:
                print(f"Bitte wählen Sie eine Zahl zwischen 1 und {len(network_files)}")
        except ValueError:
            print("Bitte geben Sie eine gültige Zahl ein.")

    # 4. Land auswählen
    country = input("\nLändercode eingeben (z.B. DE, FR, ES): ").strip().upper()

    # 5. Jahre auswählen
    available_years = sorted(selected_files.keys())
    print(f"\nVerfügbare Jahre: {available_years}")
    years_input = input("Jahre für Analyse (leer für alle, oder z.B. '2030,2040,2050'): ").strip()

    if years_input:
        try:
            selected_years = [int(y.strip()) for y in years_input.split(',')]
            selected_years = [y for y in selected_years if y in available_years]
        except ValueError:
            print("Fehler beim Parsen der Jahre. Verwende alle verfügbaren Jahre.")
            selected_years = None
    else:
        selected_years = None

    # 6. Basis-Pfad konstruieren
    base_path = str(Path(network_dir) / selected_base)

    return {
        'base_path': base_path,
        'country': country,
        'years': selected_years,
        'available_files': selected_files
    }


def main():
    """Hauptausführung"""

    print("PyPSA-Eur Myopic Länderanalyse")
    print("=" * 40)

    # Prüfe Kommandozeilenargumente
    if len(sys.argv) >= 3:
        # Direkte Ausführung mit Argumenten
        base_path = sys.argv[1]
        country = sys.argv[2]
        years = None
        if len(sys.argv) > 3:
            try:
                years = [int(y) for y in sys.argv[3].split(',')]
            except ValueError:
                print("Fehler beim Parsen der Jahre aus Kommandozeile")

        print(f"Kommandozeilen-Modus:")
        print(f"  Basis-Pfad: {base_path}")
        print(f"  Land: {country}")
        print(f"  Jahre: {years if years else 'alle verfügbaren'}")

        analyze_myopic_country(base_path, country, years)

    else:
        # Interaktiver Modus
        config = interactive_setup()

        if config is None:
            print("Setup abgebrochen.")
            return

        print(f"\n{'=' * 40}")
        print("KONFIGURATION")
        print(f"{'=' * 40}")
        print(f"Basis-Pfad: {config['base_path']}")
        print(f"Land: {config['country']}")
        print(f"Jahre: {config['years'] if config['years'] else 'alle verfügbaren'}")
        print(f"Verfügbare Dateien: {list(config['available_files'].keys())}")

        # Bestätigung
        confirm = input("\nAnalyse starten? (j/n): ").strip().lower()
        if confirm in ['j', 'ja', 'y', 'yes', '']:
            analyze_myopic_country(
                config['base_path'],
                config['country'],
                config['years']
            )
        else:
            print("Analyse abgebrochen.")


if __name__ == "__main__":
    main()