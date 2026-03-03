"""
Diagnose IIS: Find which PyPSA component causes
  x851080 >= 0  AND  x851080 = -595.557  (infeasible)

Run with:
  python3 diagnose_iis.py networks/prepared_cutout_mCNRM-CERFACS-CM5_rcp45_2010.nc
"""
import sys
import pypsa
import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "networks/prepared_cutout_mCNRM-CERFACS-CM5_rcp45_2010.nc"
n = pypsa.Network(path)

TARGET = -595.557
TOL = 5.0  # MWh

print("=" * 60)
print(f"Looking for component with value ≈ {TARGET} MWh")
print("=" * 60)

# 1. Stores: e_initial * e_nom ≈ TARGET
print("\n--- Stores: e_initial * e_nom ---")
s = n.stores.copy()
s["e_init_abs"] = s["e_initial"] * s["e_nom"].replace(np.inf, np.nan)
candidates = s[abs(s["e_init_abs"] - TARGET) < TOL]
print(candidates[["e_nom", "e_initial", "e_init_abs", "e_min_pu", "e_cyclic", "carrier"]] if len(candidates) else "none")

# 2. Stores: e_min_pu * e_nom ≈ TARGET
print("\n--- Stores: e_min_pu * e_nom ---")
s["e_min_abs"] = s["e_min_pu"] * s["e_nom"].replace(np.inf, np.nan)
candidates = s[abs(s["e_min_abs"] - TARGET) < TOL]
print(candidates[["e_nom", "e_min_pu", "e_min_abs", "carrier"]] if len(candidates) else "none")

# 3. Stores_t: any time-varying e_min_pu or e_set ≈ TARGET
if hasattr(n, "stores_t"):
    for col in ["e_min_pu", "e_set", "e"]:
        df = getattr(n.stores_t, col, None)
        if df is not None and len(df.columns) > 0:
            for asset in df.columns:
                e_nom = n.stores.at[asset, "e_nom"]
                if np.isinf(e_nom):
                    vals = df[asset]
                else:
                    vals = df[asset] * e_nom
                if abs(vals - TARGET).min() < TOL:
                    idx = abs(vals - TARGET).idxmin()
                    print(f"stores_t.{col}[{asset}] at {idx} = {vals[idx]:.3f}")

# 4. StorageUnits: state_of_charge_initial * p_nom ≈ TARGET
print("\n--- StorageUnits: soc_initial * p_nom * max_hours ---")
su = n.storage_units.copy()
su["e_init_abs"] = su["state_of_charge_initial"] * su["p_nom"] * su["max_hours"]
candidates = su[abs(su["e_init_abs"] - TARGET) < TOL]
print(candidates[["p_nom", "max_hours", "state_of_charge_initial", "e_init_abs", "carrier"]] if len(candidates) else "none")

# 5. Check e_nom_opt values (from previous solve stored in network)
print("\n--- Stores with e_nom_opt ≈ |TARGET| ---")
if "e_nom_opt" in n.stores.columns:
    candidates = n.stores[abs(n.stores["e_nom_opt"] - abs(TARGET)) < TOL]
    print(candidates[["e_nom_opt", "carrier"]] if len(candidates) else "none")

# 6. Show all stores with non-zero e_initial
print("\n--- All stores with e_initial != 0 ---")
nonzero = n.stores[n.stores["e_initial"] != 0]
print(nonzero[["e_nom", "e_initial", "carrier", "e_cyclic"]].to_string() if len(nonzero) else "none")

# 7. Check stores_t.e at t=last (non-cyclic boundary)
print("\n--- stores_t.e at last snapshot (non-cyclic boundary) ---")
if hasattr(n.stores_t, "e") and n.stores_t.e is not None and len(n.stores_t.e.columns) > 0:
    last_e = n.stores_t.e.iloc[-1]
    candidates = last_e[abs(last_e - TARGET) < TOL]
    print(candidates if len(candidates) else "none")
    # Also show first vs last difference
    diff = n.stores_t.e.iloc[-1] - n.stores_t.e.iloc[0]
    candidates = diff[abs(diff - TARGET) < TOL]
    print(f"\nstores_t.e: last - first ≈ {TARGET}:")
    print(candidates if len(candidates) else "none")

print("\nDone.")