#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose: Strompreis-Statistik (Basisfall / Standardwetterjahr)
===============================================================
Analysiert die Preisstruktur in einem normalen Wetterjahr (2005).
Fokus:
- Jahresmittelwerte & Volatilität
- Häufigkeit von Null-Preisen (Überschuss)
- Saisonale Unterschiede (Winter vs. Sommer)
"""

import os, re
import pandas as pd
import numpy as np
import pypsa
from config_final import PlottingConfig

# --- KONFIGURATION ---
TARGET_COUNTRIES = ["DE", "FR", "ALL"]
TARGET_YEARS = [2025, 2030, 2040, 2050]  # Fokus auf Entwicklung


def get_price_series(n, country):
    """Holt gewichtete durchschnittliche Marktpreise."""
    if country == "ALL":
        buses = n.buses[n.buses.carrier == "AC"].index
    else:
        buses = n.buses[(n.buses.index.str.startswith(country)) & (n.buses.carrier == "AC")].index

    if len(buses) == 0: return pd.Series(dtype=float)
    # Einfacher Durchschnitt über alle Knoten des Landes
    return n.buses_t.marginal_price[buses].mean(axis=1)


def analyze_year_statistics(series, year, country):
    """Berechnet statistische Kennzahlen für das Jahr."""

    # 1. Basis-Statistik
    mean_price = series.mean()
    std_dev = series.std()  # Maß für Volatilität
    max_price = series.max()
    min_price = series.min()

    # 2. Häufigkeit von Extremen
    # Wie oft ist Strom "kostenlos" (<= 0.1 €)? -> Überschuss
    hours_zero = (series <= 0.1).sum()
    share_zero = (hours_zero / len(series)) * 100

    # Wie oft ist Strom "teuer" (> 100 €)? -> Knappheit / Gaseinsatz
    hours_high = (series > 100).sum()

    # 3. Saisonalität (Winter: Jan, Feb, Dez | Sommer: Jun, Jul, Aug)
    # Wir nutzen den Monat aus dem Index
    months = series.index.month
    winter_mask = (months == 1) | (months == 2) | (months == 12)
    summer_mask = (months == 6) | (months == 7) | (months == 8)

    mean_winter = series[winter_mask].mean()
    mean_summer = series[summer_mask].mean()

    print(f"\n📊 ANALYSE {country} {year} (Basisfall)")
    print(f"   ------------------------------------------------")
    print(f"   Ø Jahrespreis:      {mean_price:.2f} €/MWh")
    print(f"   Volatilität (Std):  {std_dev:.2f} €/MWh")
    print(f"   Maximaler Preis:    {max_price:.2f} €/MWh")
    print(f"   ------------------------------------------------")
    print(f"   Stunden ~0 €/MWh:   {hours_zero} h ({share_zero:.1f} %)")
    print(f"   Stunden >100 €/MWh: {hours_high} h")
    print(f"   ------------------------------------------------")
    print(f"   Ø Winter (DJF):     {mean_winter:.2f} €/MWh")
    print(f"   Ø Sommer (JJA):     {mean_summer:.2f} €/MWh")
    print(f"   Saison-Spread:      {mean_winter - mean_summer:.2f} €/MWh")


def main():
    config = PlottingConfig()
    networks = config.get_networks()

    # WICHTIG: Hier 'average' erzwingen oder sicherstellen, dass 'average' geladen wird
    # Falls networks ein Dict ist, wähle 'average'
    if isinstance(networks, dict):
        if 'average' in networks:
            paths = networks['average']
            print(">>> Lade Szenario: AVERAGE (Standardwetterjahr)")
        elif 'new_avg' in networks:
            paths = networks['new_avg']
            print(">>> Lade Szenario: NEW_AVG (Standardwetterjahr)")
        else:
            # Fallback
            paths = list(networks.values())[0]
            print(">>> WARNUNG: Nutze erstes verfügbares Szenario (Bitte prüfen ob Average!)")
    else:
        paths = networks

    for path in paths:
        if not os.path.isfile(path): continue
        m = re.search(r"_(\d{4})\.nc$", path)
        if not m: continue
        year = int(m.group(1))

        if year not in TARGET_YEARS: continue

        print(f"\nLade Netzwerk {year}...")
        n = pypsa.Network(path)

        # Zeitindex reparieren falls nötig
        try:
            n.snapshots = pd.to_datetime(n.snapshots)
        except:
            pass

        for country in TARGET_COUNTRIES:
            prices = get_price_series(n, country)
            if not prices.empty:
                analyze_year_statistics(prices, year, country)


if __name__ == "__main__":
    main()