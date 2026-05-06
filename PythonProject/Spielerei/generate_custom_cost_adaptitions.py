"""
generate_custom_cost_adaptions.py
---------------------------------
Creates the CSV file `data/custom_cost_adaptions_per_horizon.csv`
for PyPSA-Eur custom fuel cost overrides.

Expected columns:
Fuel | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | source | currency_year
"""

import os
import pandas as pd

# === CONFIGURATION ===
OUTPUT_DIR = "/mnt/endata/MA_Arthur"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "custom_cost_adaptions_per_horizon.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Data ===
data = [
    {
        "Fuel": "coal",
        "2020": 6.9, "2025": 6.6, "2030": 6.4, "2035": 6.2, "2040": 5.9, "2045": 5.7, "2050": 5.5,
        "source": "TYNDP 2024",
        "currency_year": 2022.0,
    },
    {
        "Fuel": "lignite",
        "2020": 6.5, "2025": 6.5, "2030": 6.5, "2035": 6.5, "2040": 6.5, "2045": 6.5, "2050": 6.5,
        "source": "TYNDP 2024",
        "currency_year": 2022.0,
    },
    {
        "Fuel": "gas",
        "2020": 24.9, "2025": 23.8, "2030": 22.6, "2035": 21.5, "2040": 20.3, "2045": 19.2, "2050": 18.1,
        "source": "TYNDP 2024",
        "currency_year": 2022.0,
    },
    {
        "Fuel": "oil",
        "2020": 43.6, "2025": 42.9, "2030": 42.3, "2035": 41.6, "2040": 41.0, "2045": 40.3, "2050": 39.6,
        "source": "TYNDP 2024",
        "currency_year": 2022.0,
    },
    {
        "Fuel": "nuclear",
        "2020": 6.1, "2025": 6.1, "2030": 6.1, "2035": 6.1, "2040": 6.1, "2045": 6.1, "2050": 6.1,
        "source": "TYNDP 2024",
        "currency_year": 2022.0,
    },
    {
        "Fuel": "solid biomass",
        "2020": 28.04, "2025": 29.18, "2030": 30.32, "2035": 31.46, "2040": 32.61, "2045": 33.75, "2050": 34.89,
        "source": "Heat Roadmap Europe",
        "currency_year": 2015,
    },
    {
        "Fuel": "biogas",
        "2020": 70.4, "2025": 69.1, "2030": 67.7, "2035": 66.3, "2040": 64.9, "2045": 63.6, "2050": 62.2,
        "source": "TYNDP 2024",
        "currency_year": 2022.0,
    },
]

# === Write CSV ===
df = pd.DataFrame(data)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ CSV successfully written to: {OUTPUT_FILE}")
print(f"{len(df)} rows, columns: {list(df.columns)}")
