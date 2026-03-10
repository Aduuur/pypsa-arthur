# config_final.py — Kompatibilitäts-Shim
# Alle plot_*.py importieren diesen Namen historisch.
# Leitet alles auf master_config weiter, keine Skripte müssen geändert werden.

from master_config import (  # noqa: F401
    PlottingConfig,
    AROPlottingConfig,
    MasterConfig,
    MASTER_CONFIG,
)

__all__ = ["PlottingConfig", "AROPlottingConfig", "MasterConfig", "MASTER_CONFIG"]
