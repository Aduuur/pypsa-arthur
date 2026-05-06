from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================
BASE = Path("/home/endata/PycharmProjects/pypsa-ee/resources")
PATH_A = BASE / "demand-test-2048" / "electricity_demand_non_hist.csv"
PATH_B = BASE / "demand_2048" / "electricity_demand_non_hist.csv"

COUNTRIES = ["DE", "GB", "FR", "ES", "IT", "PL"]

# If years differ, we align by position (Jan..Dec) and re-label both series to a
# common synthetic hourly index (year-agnostic) for plotting.
SYNTHETIC_YEAR = 2048  # non-leap year, used only for the x-axis in plots


# =============================================================================
# IO + NORMALIZATION
# =============================================================================
def read_demand_csv(path: Path) -> pd.DataFrame:
    """
    Reads a demand CSV and returns a numeric DataFrame with:
      - DatetimeIndex if a time column exists/parseable, otherwise RangeIndex
      - Columns uppercased
    """
    df = pd.read_csv(path)

    # Identify time column (common variants) else assume first column is time if parseable
    time_col_candidates = [c for c in df.columns if c.lower() in {"snapshot", "time", "datetime", "date", "timestamp"}]
    tcol = time_col_candidates[0] if time_col_candidates else df.columns[0]

    # Try parse time
    t = pd.to_datetime(df[tcol], errors="coerce", utc=False)
    if t.notna().mean() > 0.95:  # mostly parseable -> treat as time index
        df = df.drop(columns=[tcol])
        df.index = pd.DatetimeIndex(t)
        df = df[~df.index.isna()].copy()
        df = df[~df.index.duplicated(keep="first")].sort_index()
    else:
        # No reliable time index; keep row order
        df = df.drop(columns=[tcol], errors="ignore")
        df.index = pd.RangeIndex(len(df))

    # Normalize columns
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Keep numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def ensure_columns(df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: Missing columns {missing}. Available (first 30): {list(df.columns)[:30]}")
    return df[cols].copy()


# =============================================================================
# ALIGNMENT (ROBUST TO DIFFERENT YEARS)
# =============================================================================
def make_synthetic_hourly_index(n: int, year: int = 2001) -> pd.DatetimeIndex:
    """
    Create a synthetic hourly index of length n starting Jan 1 of `year`.
    Purely for plotting/alignment display (year-agnostic).
    """
    start = f"{year}-01-01 00:00:00"
    return pd.date_range(start=start, periods=n, freq="h")


def align_by_position(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """
    Align two frames by row position (not by timestamps). Works even if years differ,
    as long as they represent the same Jan..Dec length.

    If lengths mismatch, both are truncated to the common minimum length (warn).
    Returns (a_aligned, b_aligned, synthetic_index).
    """
    n_a, n_b = len(a), len(b)
    n = min(n_a, n_b)

    if n_a != n_b:
        print(f"[WARN] Length mismatch: A={n_a}, B={n_b}. Truncating both to n={n} (common minimum).")

    a2 = a.iloc[:n].copy()
    b2 = b.iloc[:n].copy()

    syn = make_synthetic_hourly_index(n, year=SYNTHETIC_YEAR)
    a2.index = syn
    b2.index = syn

    return a2, b2, syn


# =============================================================================
# METRICS
# =============================================================================
def pearson_corr_per_country(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    out = {}
    for c in a.columns:
        x = a[c]
        y = b[c]
        mask = x.notna() & y.notna()
        out[c] = x[mask].corr(y[mask]) if mask.sum() >= 3 else np.nan
    return pd.Series(out).sort_values(ascending=False)


def diff_stats_per_country(diff: pd.DataFrame) -> pd.DataFrame:
    """
    Basic difference stats per country: mean, MAE, RMSE, max abs.
    """
    rows = []
    for c in diff.columns:
        d = diff[c].dropna()
        if len(d) == 0:
            rows.append((c, np.nan, np.nan, np.nan, np.nan))
            continue
        mae = float(np.mean(np.abs(d)))
        rmse = float(np.sqrt(np.mean(d**2)))
        rows.append((c, float(d.mean()), mae, rmse, float(np.max(np.abs(d)))))
    return pd.DataFrame(rows, columns=["country", "mean_diff", "mae", "rmse", "max_abs"]).set_index("country")


# =============================================================================
# PLOTTING
# =============================================================================
def plot_country_with_diff(a: pd.DataFrame, b: pd.DataFrame, country: str) -> None:
    """
    Plot full-year overlay (A vs B) and the difference (A - B) below.
    """
    diff = a[country] - b[country]

    fig, (ax1, ax2) = plt.subplots(figsize=(16, 6), nrows=2, sharex=True)

    ax1.plot(a.index, a[country], label="demand-test-2026 (A)", linewidth=1.0)
    ax1.plot(b.index, b[country], label="demand_2026 (B)", linewidth=1.0)
    ax1.set_title(f"{country} – electricity_demand_non_hist (aligned by position; synthetic year {SYNTHETIC_YEAR})")
    ax1.set_ylabel("Electricity demand")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(diff.index, diff, label="A - B", linewidth=1.0)
    ax2.axhline(0.0, linewidth=1.0)
    ax2.set_ylabel("Difference")
    ax2.set_xlabel("Time (synthetic, year-agnostic)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    plt.show()


def plot_all_countries_diff_heatmap(diff: pd.DataFrame) -> None:
    """
    Optional overview: heatmap-like image of differences over time (countries x time).
    Uses imshow to keep it lightweight (no seaborn).
    """
    # Convert to numpy, keep column order
    arr = diff.to_numpy().T  # countries x time
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(arr, aspect="auto", interpolation="nearest")
    ax.set_title("Differences (A - B) – overview (countries x time)")
    ax.set_yticks(range(len(diff.columns)))
    ax.set_yticklabels(diff.columns)
    ax.set_xlabel("Time index (synthetic hourly)")
    ax.set_ylabel("Country")
    fig.colorbar(im, ax=ax, label="A - B")
    fig.tight_layout()
    plt.show()


# =============================================================================
# RUN
# =============================================================================
COUNTRIES = [c.upper() for c in COUNTRIES]

df_a = read_demand_csv(PATH_A)
df_b = read_demand_csv(PATH_B)

df_a = ensure_columns(df_a, COUNTRIES, label="A (demand-test-2026)")
df_b = ensure_columns(df_b, COUNTRIES, label="B (demand_2026)")

# Align robustly by position (ignoring differing years)
a_aligned, b_aligned, _ = align_by_position(df_a, df_b)

# Correlations
corrs = pearson_corr_per_country(a_aligned, b_aligned)
print("\nPearson correlation per country (aligned by position):")
print(corrs.to_string(float_format=lambda x: f"{x:0.4f}"))

# Differences + stats
diff = a_aligned - b_aligned
stats = diff_stats_per_country(diff)
print("\nDifference statistics per country (A - B):")
print(stats.to_string(float_format=lambda x: f"{x:0.4f}"))

# Plots per country: overlay + difference
for c in COUNTRIES:
    plot_country_with_diff(a_aligned, b_aligned, c)

# Optional: overview plot of differences (countries x time)
plot_all_countries_diff_heatmap(diff)