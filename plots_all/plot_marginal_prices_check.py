#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose: Ursachenanalyse extremer Strompreise (>1000 €/MWh) für Deutschland
============================================================================

Ermittelt für jede Hochpreisstunde:
- Gesamtlast (MW)
- Gesamtstromerzeugung (MW)
- Differenz (Deckungslücke)
- Speicherstände (Batterie, H2)
- Gaslimit oder Fuel-Constraint-Infos

Ziel: Erkennen, ob Preispeaks durch Speicherleerstand, Limits oder Netzprobleme entstehen.
"""

import pypsa
import pandas as pd
import os
from config_final import PlottingConfig

# ============================================
# Einstellungen
# ============================================
COUNTRY = "DE"
PRICE_THRESHOLD = 1000  # €/MWh
SHOW_DETAILS = True     # Ausgabe für jede Stunde
SAVE_CSV = True

# ============================================
# Hauptlogik
# ============================================
config = PlottingConfig()
networks = config.get_networks()
if not isinstance(networks, list) or not networks:
    raise ValueError("❌ Keine Netzwerke in config_final.get_networks() gefunden!")

# Erstes Netz (oder anpassen)
path = networks[0]
print(f"📂 Lade Netzwerk: {path}")
n = pypsa.Network(path)

# === 1. Preiszeitreihe für DE ===
de_buses = [b for b in n.buses.index if b.startswith(COUNTRY)]
if not de_buses:
    raise ValueError("❌ Keine Busse für DE gefunden!")

de_price = n.buses_t.marginal_price[de_buses].mean(axis=1)
highprice_hours = de_price[de_price > PRICE_THRESHOLD]

if highprice_hours.empty:
    print(f"✅ Keine Stunden über {PRICE_THRESHOLD} €/MWh gefunden.")
    exit()

print(f"⚠️ {len(highprice_hours)} Hochpreisstunden gefunden:\n{highprice_hours}\n")

# === 2. Basisdaten vorbereiten ===

# Deutsche Lasten auswählen (nach Bus)
de_loads = n.loads.index[n.loads.bus.isin(de_buses)]

# Last in MW
de_loads_in_ts = de_loads.intersection(n.loads_t.p_set.columns)
load_DE = n.loads_t.p_set[de_loads_in_ts].sum(axis=1)
# Erzeugung (MW)
de_gens = n.generators.index[n.generators.bus.isin(de_buses)]
de_gens = de_gens.intersection(n.generators_t.p.columns)
gen_DE = n.generators_t.p[de_gens].sum(axis=1)
# Speicherstände
store_DE = n.stores[n.stores.bus.isin(de_buses)]
storage_states = n.stores_t.state_of_charge[store_DE.index] if not store_DE.empty else pd.DataFrame()

# === 3. Ergebnisse sammeln ===
records = []

for t in highprice_hours.index:
    load = load_DE.loc[t]
    gen = gen_DE.loc[t]
    gap = gen - load
    row = {
        "Zeit": t,
        "Preis_€/MWh": de_price.loc[t],
        "Last_MW": load,
        "Erzeugung_MW": gen,
        "Bilanz_(Gen-Load)_MW": gap,
    }

    # Speicherstände abfragen
    for s in store_DE.index:
        soc = n.stores_t.state_of_charge[s].loc[t]
        row[f"SoC_{s}"] = soc

    records.append(row)

    if SHOW_DETAILS:
        print(f"\n🕒 {t} — Preis {de_price.loc[t]:.1f} €/MWh")
        print(f"   ➤ Last: {load:,.0f} MW | Erzeugung: {gen:,.0f} MW | Bilanz: {gap:,.0f} MW")

        # Speicherzustände
        if not storage_states.empty:
            print("   🔋 Speicherstände:")
            for s in store_DE.index:
                soc = n.stores_t.state_of_charge[s].loc[t]
                print(f"      {s:25s}: {soc:,.1f} MWh")

# === 4. Als Tabelle speichern ===
df = pd.DataFrame(records)
df = df.round(2)

if SAVE_CSV:
    out_path = os.path.join(
        config.BASE_SAVE_PATH,
        f"diagnose_highprice_{COUNTRY}_{os.path.basename(path).replace('.nc','.csv')}"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n💾 Ergebnisse gespeichert unter: {out_path}")

print("\n✅ Diagnose abgeschlossen.")
