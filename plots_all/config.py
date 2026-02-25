"""
Konfigurationsdatei für die PyPSA-Eur Analyse
"""

# Technologie-Farben (PyPSA-Eur Standard)
TECH_COLORS = {
    # Strom-Technologien
    'solar': '#FFDD00',
    'solar rooftop': '#FFE640',
    'solar-hsat': '#FFF080',
    'onwind': '#74CBEB',
    'offwind': '#004E8A',
    'offwind-ac': '#003366',
    'offwind-dc': '#002244',
    'offwind-float': '#001122',
    'ror': '#1E7DDF',
    'hydro': '#298DFF',
    'PHS': '#298DFF',

    # Fossil
    'gas': '#B08080',
    'OCGT': '#B08080',
    'CCGT': '#A07070',
    'coal': '#454545',
    'lignite': '#8B4513',
    'oil': '#8B0000',
    'nuclear': '#FF5200',

    # Biomasse (bessere Farben)
    'biogas': '#228B22',  # Forest Green
    'solid biomass': '#32CD32',  # Lime Green
    'unsustainable biogas': '#8FBC8F',  # Dark Sea Green
    'unsustainable solid biomass': '#9ACD32',  # Yellow Green
    'unsustainable bioliquids': '#ADFF2F',  # Green Yellow

    # Wasserstoff
    'H2': '#EA4CFA',
    'H2 Fuel Cell': '#DA3CEA',

    # Wärme-Technologien (deutlich unterscheidbare Farben)
    'rural heat vent': '#FF6B6B',  # Coral
    'urban central heat vent': '#FF4500',  # Orange Red
    'urban decentral heat vent': '#FF8C00',  # Dark Orange
    'rural solar thermal': '#DAA520',  # Goldenrod
    'urban central solar thermal': '#B8860B',  # Dark Goldenrod
    'urban decentral solar thermal': '#CD853F',  # Peru

    # Speicher
    'battery': '#b36b00',
    'load': '#d63031'
}

# Länder-Codes und Namen
COUNTRY_NAMES = {
    'DE': 'Deutschland',
    'FR': 'Frankreich',
    'ES': 'Spanien',
    'IT': 'Italien',
    'NL': 'Niederlande',
    'BE': 'Belgien',
    'AT': 'Österreich',
    'CH': 'Schweiz',
    'PL': 'Polen',
    'DK': 'Dänemark',
    'SE': 'Schweden',
    'NO': 'Norwegen',
    'FI': 'Finnland',
    'GB': 'Großbritannien',
    'IE': 'Irland',
    'PT': 'Portugal'
}

# Plot-Einstellungen
PLOT_SETTINGS = {
    'figsize_default': (12, 8),
    'figsize_large': (16, 12),
    'dpi': 300,
    'style': 'seaborn-v0_8-whitegrid',
    'font_size': 12,
    'title_size': 16
}

# Ausgabe-Einstellungen
OUTPUT_SETTINGS = {
    'save_format': 'png',
    'save_dpi': 300,
    'bbox_inches': 'tight'
}