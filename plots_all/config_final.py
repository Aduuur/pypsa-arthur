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

def fill_leap_day(df):
    """Füllt fehlende Stunden (z.B. 29.02.) per forward-fill auf."""
    import pandas as _pd
    if df is None or df.empty:
        return df
    idx = _pd.to_datetime(df.index)
    full = _pd.date_range(start=idx[0], end=idx[-1], freq="h")
    if len(full) == len(idx):
        return df
    df2 = df.copy()
    df2.index = idx
    return df2.reindex(full).ffill().bfill()

