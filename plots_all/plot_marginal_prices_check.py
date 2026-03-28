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


def main():
    # ============================================
    # Hauptlogik
    # ============================================
    config = PlottingConfig()
    networks = config.get_networks()
    if isinstance(networks, dict):
        # Nimm erstes Szenario wenn dict
        networks = next(iter(networks.values()))
    if not isinstance(networks, list) or not networks:
        print("⚠️ Keine Netzwerke in config.get_networks() gefunden, skip.")
        return

    # Erstes Netz (oder anpassen)
    path = networks[0]
    print(f"📂 Lade Netzwerk: {path}")
    n = pypsa.Network(path)

    # === 1. Preiszeitreihe für DE ===
    de_buses = [b for b in n.buses.index if b.startswith(COUNTRY)]
    if not de_buses:
        print("⚠️ Keine Busse für DE gefunden, skip.")
        return

    de_price = n.buses_t.marginal_price[de_buses].mean(axis=1)
    highprice_hours = de_price[de_price > PRICE_THRESHOLD]

    if highprice_hours.empty:
        print(f"✅ Keine Stunden über {PRICE_THRESHOLD} €/MWh gefunden.")
        return

    print(f"⚠️ {len(highprice_hours)} Hochpreisstunden gefunden:\n{highprice_hours}\n")

    # === 2. Basisdaten vorbereiten ===
    de_loads = n.loads.index[n.loads.bus.isin(de_buses)]
    de_loads_in_ts = de_loads.intersection(
        n.loads_t.p_set.columns if hasattr(n.loads_t, 'p_set') and not n.loads_t.p_set.empty
        else pd.Index([])
    )
    load_DE = n.loads_t.p_set[de_loads_in_ts].sum(axis=1) if len(de_loads_in_ts) > 0 else pd.Series(0.0, index=n.snapshots)

    de_gens = n.generators.index[n.generators.bus.isin(de_buses)]
    de_gens_in_ts = de_gens.intersection(
        n.generators_t.p.columns if hasattr(n.generators_t, 'p') and not n.generators_t.p.empty
        else pd.Index([])
    )
    gen_DE = n.generators_t.p[de_gens_in_ts].sum(axis=1) if len(de_gens_in_ts) > 0 else pd.Series(0.0, index=n.snapshots)

    # FIX: stores_t.state_of_charge existiert in neueren PyPSA-Versionen nicht mehr.
    # Verwende stores_t.e stattdessen (energy level = SoC für Stores).
    store_DE = n.stores[n.stores.bus.isin(de_buses)]
    if not store_DE.empty:
        soc_attr = None
        for candidate in ("e", "state_of_charge"):
            ts = getattr(n.stores_t, candidate, None)
            if ts is not None and not ts.empty:
                avail = store_DE.index.intersection(ts.columns)
                if len(avail) > 0:
                    soc_attr = candidate
                    store_DE = store_DE.loc[avail]  # nur verfügbare
                    break
        if soc_attr is None:
            store_DE = store_DE.iloc[0:0]  # leer, kein SoC verfügbar
    storage_ts = getattr(n.stores_t, soc_attr, None) if not store_DE.empty else None

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
        if storage_ts is not None and not store_DE.empty:
            for s in store_DE.index:
                if s in storage_ts.columns:
                    row[f"SoC_{s}"] = storage_ts[s].loc[t]

        records.append(row)

        if SHOW_DETAILS:
            print(f"\n🕒 {t} — Preis {de_price.loc[t]:.1f} €/MWh")
            print(f"   ➤ Last: {load:,.0f} MW | Erzeugung: {gen:,.0f} MW | Bilanz: {gap:,.0f} MW")
            if storage_ts is not None and not store_DE.empty:
                print("   🔋 Speicherstände:")
                for s in store_DE.index:
                    if s in storage_ts.columns:
                        soc = storage_ts[s].loc[t]
                        print(f"      {s:25s}: {soc:,.1f} MWh")

    # === 4. Als Tabelle speichern ===
    df = pd.DataFrame(records)
    df = df.round(2)

    if SAVE_CSV and not df.empty:
        out_path = os.path.join(
            config.BASE_SAVE_PATH,
            f"diagnose_highprice_{COUNTRY}_{os.path.basename(path).replace('.nc','.csv')}"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n💾 Ergebnisse gespeichert unter: {out_path}")

    print("\n✅ Diagnose abgeschlossen.")


if __name__ == "__main__":
    main()
