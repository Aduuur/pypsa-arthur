import pypsa
import pandas as pd
import json

n = pypsa.Network("results/compare-robust-vol3/networks/aro/robust.nc")

print("="*80)
print("NETZWERK GRUNDLAGEN")
print("="*80)
print(f"Snapshots: {len(n.snapshots)}")
print(f"Zeitspanne: {n.snapshots[0]} bis {n.snapshots[-1]}")
print(f"Generatoren: {len(n.generators)}")
print(f"Links: {len(n.links)}")
print(f"Stores: {len(n.stores)}")
print(f"Busse: {len(n.buses)}")

load_gens = n.generators[n.generators.carrier == "load"]
print("\n" + "="*80)
print("LOAD-SHEDDING GENERATOREN")
print("="*80)
print(f"Anzahl Load-Generatoren: {len(load_gens)}")
if len(load_gens) > 0:
    print(f"Gesamt p_nom: {load_gens.p_nom.sum():.2f} MW")
    print(f"Gesamt p_nom_opt: {load_gens.p_nom_opt.sum():.2f} MW")
    print(f"Marginal Cost: {load_gens.marginal_cost.unique()}")

print("\n" + "="*80)
print("BIOMASSE & BIOGAS KAPAZITÄTEN")
print("="*80)
bio_gens = n.generators[n.generators.carrier.str.contains("bio", case=False, na=False)]
print(f"Biomasse-Generatoren: {len(bio_gens)}")
print(f"Gesamt p_nom_opt: {bio_gens.p_nom_opt.sum():.2e} MW")

print("\n" + "="*80)
print("KOSTEN-ANALYSE")
print("="*80)
print(f"Gesamt-Objektiv: {n.objective:.2e} EUR/a")
print(f"In Milliarden: {n.objective/1e9:.0f} Mrd EUR/a")

if len(load_gens) > 0:
    ls_cost = load_gens.p_nom_opt.sum() * 8760 * 100000
    print(f"Load-Shedding Kosten: {ls_cost:.2e} EUR/a")
    print(f"Das sind {ls_cost/n.objective*100:.1f}% des Objektivs")

print("\n" + "="*80)
print("ENERGIE-ERZEUGUNG")
print("="*80)
if hasattr(n, 'generators_t') and 'p' in n.generators_t and len(n.generators_t.p) > 0:
    print(f"Dispatch-Zeitschritte: {len(n.generators_t.p)}")
else:
    print("⚠️  KEINE DISPATCH-DATEN GEFUNDEN!")

print("\n" + "="*80)
print("DIAGNOSE ZUSAMMENFASSUNG")
print("="*80)
print(f"1. Snapshots: {len(n.snapshots)}")
print(f"2. Objektiv: {n.objective:.2e} EUR/a")
print(f"3. Load-Shedding: {load_gens.p_nom_opt.sum():.0f} MW")
print(f"4. Biomasse: {bio_gens.p_nom_opt.sum():.2e} MW")
