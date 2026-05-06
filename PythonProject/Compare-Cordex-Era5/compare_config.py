# compare_config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass(frozen=True)
class CompareConfig:
    # Default: deine drei Run-Ordner
    run_dirs: Dict[str, Path] = None
    out_dir: Path = Path("./compare_analysis_out")

    # feste Farben je Run
    colors: Dict[str, str] = None

    # Jahr (für Zeitfilter/Monate, falls vorhanden)
    year: int = 2010

    def __post_init__(self):
        object.__setattr__(self, "run_dirs", self.run_dirs or {
            "cordex_rcp26": Path("/home/endata/PycharmProjects/pypsa-ee/resources/compare-cordex-rcp26"),
            "cordex_rcp45": Path("/home/endata/PycharmProjects/pypsa-ee/resources/compare-cordex-rcp45"),
            "era5":         Path("/home/endata/PycharmProjects/pypsa-ee/resources/compare-era5"),
        })
        object.__setattr__(self, "colors", self.colors or {
            "cordex_rcp26": "#1f77b4",
            "cordex_rcp45": "#ff7f0e",
            "era5":         "#2ca02c",
        })
