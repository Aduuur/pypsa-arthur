import pandas as pd

# Lade die beiden CSV-Dateien (sicherstellen, dass sie korrekt im richtigen Verzeichnis gespeichert sind)
raw_data_path = '/home/endata/PycharmProjects/PythonProject/Compare-Cordex-Era5/metvar_timeseries_out/metrics_2010.csv'  # Stelle sicher, dass dies der korrekte Pfad ist
cutout_data_path = '//home/endata/PycharmProjects/PythonProject/Compare-Cordex-Era5/cutout_metvar_timeseries_out/metrics_2010.csv'  # Hier die Cutout-Datei eintragen

# Lade die Rohdaten
raw_data = pd.read_csv(raw_data_path)

cutout_data = pd.read_csv(cutout_data_path)

# Überblick über die Daten
print("Raw Data Head:")
print(raw_data.head())

# Falls die Cutout-Daten existieren:
if not cutout_data.empty:
    print("\nCutout Data Head:")
    print(cutout_data.head())

# Berechne Mittelwert und Standardabweichung für Rohdaten (ALL)
agg_raw_data = raw_data.groupby(['var', 'window_days']).agg({
    'corr_era5_rcp26': 'mean',
    'corr_era5_rcp45': 'mean',
    'corr_rcp26_rcp45': 'mean',
    'era5_roll_mean': 'mean',
    'era5_roll_std': 'mean',
    'rcp26_roll_mean': 'mean',
    'rcp26_roll_std': 'mean',
    'rcp45_roll_mean': 'mean',
    'rcp45_roll_std': 'mean'
}).reset_index()

# Falls auch Cutout-Daten vorhanden sind:
if not cutout_data.empty:
    # Berechne Mittelwert und Standardabweichung für Cutouts
    agg_cutout_data = cutout_data.groupby(['var', 'window_days']).agg({
        'corr_era5_rcp26': 'mean',
        'corr_era5_rcp45': 'mean',
        'corr_rcp26_rcp45': 'mean',
        'era5_roll_mean': 'mean',
        'era5_roll_std': 'mean',
        'rcp26_roll_mean': 'mean',
        'rcp26_roll_std': 'mean',
        'rcp45_roll_mean': 'mean',
        'rcp45_roll_std': 'mean'
    }).reset_index()

    # Zeige Ergebnisse
    print("\nAggregierte Werte für Raw Data:")
    print(agg_raw_data)

    print("\nAggregierte Werte für Cutout Data:")
    print(agg_cutout_data)

# Optional: Eine Übersicht der wichtigsten KPIs
def display_key_stats(data, data_type):
    print(f"\nStatistische Kennzahlen für {data_type} Daten:")
    print(data.describe())

# Zeige statistische Kennzahlen für beide Datensätze
display_key_stats(agg_raw_data, "Rohdaten")
if not cutout_data.empty:
    display_key_stats(agg_cutout_data, "Cutout-Daten")

# Optional: Speichern der aggregierten Daten als CSV
agg_raw_data.to_csv('aggregated_raw_data.csv', index=False)
if not cutout_data.empty:
    agg_cutout_data.to_csv('aggregated_cutout_data.csv', index=False)

print("\nDaten wurden erfolgreich aggregiert und gespeichert!")
