from __future__ import annotations

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import seaborn as sns


# =============================================================================
# KONFIGURATION
# =============================================================================
COUNTRIES_TO_ANALYZE = ["DE", "FR", "PL", "CZ", "AT", "CH", "IT", "ES", "BE", "NL", "DK", "SE",
    "PT", "NO", "FI", "LT", "LV", "EE", "GB",
]


Y0, Y1 = 2025, 2050
YEARS_TO_ANALYZE = range(Y0, Y1 + 1)

CAPACITY_YEAR = 2024

# Score: True = SSE, False = MAE (für "typisches Jahr" i.d.R. MAE sinnvoller)
USE_SSE = False

AVG_PERIOD_START = Y0
AVG_PERIOD_END = Y1

# Shape- vs Level-Vergleich:
PROFILE_NORMALIZATION = "year_sum"  # "year_sum" oder None

# Optional: Blockweise Auswertung
DECADAL_MODE = False
DECADES = [(2025, 2034), (2035, 2044), (2045, 2050)]

# CF sanity
CF_RANGE_TOL = 0.05

# -------------------------
# NEU: Robustere Stress-/Dunkelflauten-Definition
# -------------------------
# Wir definieren "Stress" nicht nur als Peak einer 7d-Mean Residual Load,
# sondern ergänzen:
# - Max 14d rolling mean (längere Events)
# - Dauer oberhalb einer Schwelle (Stunden)
# - "Energie" oberhalb einer Schwelle (Fläche) auf Basis 7d-Mean
#
# Schwelle tau wird aus der Gesamtverteilung der 7d-Mean Residual Load bestimmt.
STRESS_ROLL_DAYS_MAIN = 7
STRESS_ROLL_DAYS_LONG = 14
STRESS_THRESHOLD_MODE = "quantile"  # "quantile" oder "absolute"
STRESS_THRESHOLD_Q = 0.90           # nur relevant bei mode="quantile"
STRESS_THRESHOLD_ABS_MW = None      # nur relevant bei mode="absolute"

# Daraus bauen wir eine kombinierte Stress-Skalar-Metrik pro Jahr und filtern Top-25%.
STRESS_WEIGHTS = {
    "peak_7d_sc": 0.35,
    "peak_14d_sc": 0.20,
    "hours_above_tau_sc": 0.20,
    "energy_above_tau_sc": 0.25,
}

# -------------------------
# 0.7 = Schwerpunkt "typisches Wetterjahr/Erzeugung", 0.3 = Systemstress-Komponente
FINAL_OBJECTIVE_WEIGHTS = {
    "gen_typical": 0.70,
    "residual_typical": 0.30,
}

# --- PFADE (bitte ggf. anpassen) ---
DEMAND_BASE_DIR = "/home/endata/PycharmProjects/pypsa-ee/resources"
DEMAND_DIR_PATTERN = "demand_{year}"
DEMAND_FILENAME = "electricity_demand_non_hist.csv"

GENERATION_DATA_DIR = "/mnt/endata/MA_Arthur/final_results_portfolio_uncorrected"
CAPACITY_DATA_PATH = "/home/endata/PycharmProjects/PythonProject/Residuallast/installed_capacities_europe.csv"

OUTPUT_DIR = "/mnt/endata/MA_Arthur/average_weather_year/plot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COUNTRY_CODE_TO_FILENAME_MAP = {
    "DE": "Germany",
    "FR": "France",
    "GB": "United_Kingdom",
    "ES": "Spain",
    "IT": "Italy",
    "PL": "Poland",
}

TECHNOLOGIES = ["solar", "wind_onshore", "wind_offshore"]


# =============================================================================
# Helpers: Alternative Metriken + Clustering
# =============================================================================
def weekly_profiles_by_year(df_generation: pd.DataFrame, value_col: str = "total") -> pd.DataFrame:
    """
    index=year, columns=week (1..52), values=weekly total (sum over hours).

    value_col: Spalte in df_generation (z.B. "total" oder "residual_load_mw")
    """
    if value_col not in df_generation.columns:
        raise KeyError(f"df_generation muss eine Spalte '{value_col}' enthalten.")
    if not isinstance(df_generation.index, pd.DatetimeIndex):
        raise TypeError("df_generation.index muss ein DatetimeIndex sein.")

    df = df_generation[[value_col]].copy()
    df["year"] = df.index.year
    df["week"] = df.index.isocalendar().week.astype(int)
    df = df[df["week"] != 53]

    weekly = (
        df.groupby(["year", "week"])[value_col]
        .sum()
        .unstack(level="week")
        .fillna(0.0)
        .sort_index()
    )
    weekly.index = weekly.index.astype(int)
    return weekly


def compute_alternative_metrics(weekly_yearly: pd.DataFrame, ref_weekly: pd.Series) -> pd.DataFrame:
    """
    Metriken pro Jahr gegen Referenz-Wochenprofil:
      - mae_level: MAE auf Rohwerten (Level)
      - mae_shape: MAE nach Normierung auf Summe=1 (Shape)
      - cos_sim: Cosine similarity (höher besser)
      - pearson_r: Pearson r (höher besser)

    Zusätzlich: min-max-skalierte Varianten *_sc, alle so gedreht, dass 0=best, 1=worst.
    Und combined_score als gewichtete Summe.
    """
    cols = weekly_yearly.columns
    ref = ref_weekly.reindex(cols)
    if ref.isna().any():
        missing = ref.index[ref.isna()].tolist()
        raise ValueError(f"ref_weekly hat nach Reindex NaNs. Fehlende Wochen: {missing}")

    ref_vec = ref.values

    rows = []
    for y in weekly_yearly.index:
        vec = weekly_yearly.loc[y].values

        mae_level = float(np.mean(np.abs(vec - ref_vec)))

        v_sum = float(vec.sum())
        r_sum = float(ref_vec.sum())
        if v_sum == 0.0 or r_sum == 0.0:
            mae_shape = np.nan
        else:
            vec_n = vec / v_sum
            ref_n = ref_vec / r_sum
            mae_shape = float(np.mean(np.abs(vec_n - ref_n)))

        try:
            cs = 1.0 - cosine(vec, ref_vec)
            cos_sim = float(cs) if np.isfinite(cs) else np.nan
        except Exception:
            cos_sim = np.nan

        try:
            r, _ = pearsonr(vec, ref_vec)
            pearson = float(r) if np.isfinite(r) else np.nan
        except Exception:
            pearson = np.nan

        rows.append(
            {
                "year": int(y),
                "mae_level": mae_level,
                "mae_shape": mae_shape,
                "cos_sim": cos_sim,
                "pearson_r": pearson,
            }
        )

    dfm = pd.DataFrame(rows).set_index("year").sort_index()

    # --- Skalierung (0..1) ---
    # kleinere besser
    for col in ["mae_level", "mae_shape"]:
        mn, mx = dfm[col].min(skipna=True), dfm[col].max(skipna=True)
        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            dfm[f"{col}_sc"] = (dfm[col] - mn) / (mx - mn)
        else:
            dfm[f"{col}_sc"] = 0.0

    # größere besser -> invertieren
    for col in ["cos_sim", "pearson_r"]:
        mn, mx = dfm[col].min(skipna=True), dfm[col].max(skipna=True)
        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            dfm[f"{col}_sc"] = 1.0 - (dfm[col] - mn) / (mx - mn)
        else:
            dfm[f"{col}_sc"] = 0.0

    # --- Combined score ---
    # Schwerpunkt: Shape + Musterähnlichkeit
    w = {"mae_shape_sc": 0.45, "mae_level_sc": 0.10, "cos_sim_sc": 0.25, "pearson_r_sc": 0.20}
    dfm["combined_score"] = 0.0
    for k, v in w.items():
        fill_value = dfm[k].max(skipna=True) if k in dfm.columns else 1.0
        dfm["combined_score"] += dfm[k].fillna(fill_value) * v

    return dfm


def cluster_years_and_medoid(weekly_yearly: pd.DataFrame, n_clusters=None):
    """
    Hierarchisches Clustering auf Wochenvektoren (cosine distance).
    """
    X = weekly_yearly.values
    Z = linkage(X, method="average", metric="cosine")

    if n_clusters is None:
        n_years = X.shape[0]
        n_clusters = max(2, min(6, max(2, n_years // 6)))

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    medoids = {}
    for lab in np.unique(labels):
        idxs = np.where(labels == lab)[0]
        sub = X[idxs]

        dmat = np.zeros((len(idxs), len(idxs)))
        for i in range(len(idxs)):
            for j in range(len(idxs)):
                dmat[i, j] = cosine(sub[i], sub[j])

        medoid_local = int(np.argmin(dmat.sum(axis=1)))
        medoid_idx = int(idxs[medoid_local])
        medoids[int(lab)] = int(weekly_yearly.index[medoid_idx])

    return labels, medoids, Z


# =============================================================================
# Generic Helpers (I/O)
# =============================================================================
def _read_timeseries_csv_firstcol(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Unerwartetes Format (zu wenige Spalten) in: {path}")

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].isna().all():
        raise ValueError(f"Konnte Zeitspalte nicht als datetime parsen in: {path}")

    df = df.set_index(time_col)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def _ensure_hourly_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name}: Index ist kein DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def _slice_years(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    return df[(df.index.year >= y0) & (df.index.year <= y1)].copy()


def _assert_cf_sane(df_cf: pd.DataFrame, name: str) -> None:
    mins = df_cf.min()
    maxs = df_cf.max()
    bad_min = mins[mins < (0.0 - CF_RANGE_TOL)]
    bad_max = maxs[maxs > (1.0 + CF_RANGE_TOL)]
    if (not bad_min.empty) or (not bad_max.empty):
        msg = [f"{name}: CF Range Check fehlgeschlagen."]
        if not bad_min.empty:
            msg.append("  Min < 0 (jenseits Toleranz):")
            msg.extend([f"    {k}: {v:.3f}" for k, v in bad_min.items()])
        if not bad_max.empty:
            msg.append("  Max > 1 (jenseits Toleranz):")
            msg.extend([f"    {k}: {v:.3f}" for k, v in bad_max.items()])
        msg.append(
            "Hinweis: Wenn diese Spalten keine echten Capacity Factors sind (0..1), "
            "dann multiplizierst du evtl. fälschlich nochmals mit Kapazitäten."
        )
        raise ValueError("\n".join(msg))


# =============================================================================
# Laden: Demand (pro Jahr)
# =============================================================================
def load_demand_timeseries() -> pd.DataFrame:
    dfs = []
    missing = []

    for year in YEARS_TO_ANALYZE:
        year_dir = os.path.join(DEMAND_BASE_DIR, DEMAND_DIR_PATTERN.format(year=year))
        path = os.path.join(year_dir, DEMAND_FILENAME)
        if not os.path.exists(path):
            missing.append(path)
            continue

        df_y = _read_timeseries_csv_firstcol(path)
        df_y = _ensure_hourly_index(df_y, f"Demand {year}")

        cols_missing = [c for c in COUNTRIES_TO_ANALYZE if c not in df_y.columns]
        if cols_missing:
            raise ValueError(
                f"Demand-Datei {path} fehlt Spalten: {cols_missing}. "
                f"Vorhanden: {list(df_y.columns)[:20]}..."
            )

        df_y = df_y[COUNTRIES_TO_ANALYZE].copy()
        dfs.append(df_y)

    if missing:
        raise FileNotFoundError(
            "Fehlende Demand-Dateien (pro Jahr muss eine existieren). Beispiele:\n- "
            + "\n- ".join(missing[:10])
            + ("\n..." if len(missing) > 10 else "")
        )

    df = pd.concat(dfs, axis=0).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = _slice_years(df, Y0, Y1)

    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        nan_share = float(df.isna().mean().max())
        raise ValueError(
            f"Demand enthält NaNs (max. Anteil in einer Spalte: {nan_share:.2%}). Betroffene Spalten: {nan_cols}"
        )

    return df


# =============================================================================
# Laden: Capacity Factors (pro Land)
# =============================================================================
def _find_cf_file_for_country(country_name: str) -> str:
    exact = os.path.join(
        GENERATION_DATA_DIR,
        f"final_portfolio_generation_{country_name}_weather{Y0}-{Y1}_cap{CAPACITY_YEAR}.csv",
    )
    if os.path.exists(exact):
        return exact

    pattern = os.path.join(
        GENERATION_DATA_DIR,
        f"final_portfolio_generation_{country_name}_weather*_cap{CAPACITY_YEAR}.csv",
    )
    cands = sorted(glob.glob(pattern))
    if not cands:
        raise FileNotFoundError(
            f"Keine CF-Datei gefunden für {country_name}. Erwartet z.B.:\n  {exact}\n"
            f"Oder Pattern:\n  {pattern}"
        )

    cands = sorted(cands, key=lambda p: os.path.getsize(p), reverse=True)
    return cands[0]


def load_capacity_factors() -> pd.DataFrame:
    all_cf_dfs = []

    for code in COUNTRIES_TO_ANALYZE:
        name = COUNTRY_CODE_TO_FILENAME_MAP[code]
        path = _find_cf_file_for_country(name)

        df = pd.read_csv(
            path,
            index_col="time",
            parse_dates=True,
            usecols=["time", "solar_cf", "wind_onshore_cf", "wind_offshore_cf"],
        )
        df = _ensure_hourly_index(df, f"CF {code}")
        df = _slice_years(df, Y0, Y1)

        df.columns = [f"{code}_solar", f"{code}_wind_onshore", f"{code}_wind_offshore"]
        all_cf_dfs.append(df)

    df_cf = pd.concat(all_cf_dfs, axis=1)

    if df_cf.isna().any().any():
        bad = df_cf.columns[df_cf.isna().any()].tolist()
        raise ValueError(f"CF-Zeitreihen enthalten NaNs. Betroffene Spalten: {bad}")

    _assert_cf_sane(df_cf, "Capacity Factors")
    return df_cf


# =============================================================================
# Laden: Kapazitäten
# =============================================================================
def load_installed_capacities() -> tuple[pd.DataFrame, pd.Series]:
    df_caps = pd.read_csv(CAPACITY_DATA_PATH)

    required_cols = {"country", "technology", "installed_gw"}
    if not required_cols.issubset(df_caps.columns):
        raise ValueError(
            f"Kapazitätsdatei fehlt Spalten {required_cols - set(df_caps.columns)}. "
            f"Vorhanden: {list(df_caps.columns)}"
        )

    df_caps = df_caps[df_caps["country"].isin(COUNTRIES_TO_ANALYZE)].copy()

    cap_matrix = (
        df_caps.pivot_table(
            index="country",
            columns="technology",
            values="installed_gw",
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )

    tech_caps = df_caps.groupby("technology")["installed_gw"].sum()
    tech_caps = tech_caps.reindex(TECHNOLOGIES).fillna(0.0)
    if tech_caps.sum() <= 0:
        raise ValueError("Kapazitäten summieren sich zu 0 – prüfe CAPACITY_DATA_PATH/Technologien.")

    tech_weights = tech_caps / tech_caps.sum()
    return cap_matrix, tech_weights


# =============================================================================
# Generation + Residual Load
# =============================================================================
def aggregate_generation_mw(df_cfs: pd.DataFrame, cap_matrix_gw: pd.DataFrame) -> pd.DataFrame:
    df_generation = pd.DataFrame(index=df_cfs.index)

    for tech in TECHNOLOGIES:
        tech_cols = [f"{code}_{tech}" for code in COUNTRIES_TO_ANALYZE]

        caps_mw = []
        for code in COUNTRIES_TO_ANALYZE:
            if tech in cap_matrix_gw.columns and code in cap_matrix_gw.index:
                caps_mw.append(float(cap_matrix_gw.loc[code, tech]) * 1000.0)
            else:
                caps_mw.append(0.0)

        df_gen_tech = df_cfs[tech_cols].mul(caps_mw, axis=1)
        df_generation[tech] = df_gen_tech.sum(axis=1)

    df_generation["total"] = df_generation[TECHNOLOGIES].sum(axis=1)
    return df_generation


def compute_residual_load_mw(df_demand: pd.DataFrame, df_generation: pd.DataFrame) -> pd.Series:
    demand_total = df_demand[COUNTRIES_TO_ANALYZE].sum(axis=1)
    demand_total, gen_total = demand_total.align(df_generation["total"], join="inner")
    residual = demand_total - gen_total
    residual.name = "residual_load_mw"
    return residual


# =============================================================================
# NEU: Stress-Metriken pro Jahr (Peak + Dauer + Fläche)
# =============================================================================
def _minmax_scale(s: pd.Series) -> pd.Series:
    mn = float(s.min())
    mx = float(s.max())
    if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
        return (s - mn) / (mx - mn)
    return pd.Series(0.0, index=s.index)


def compute_yearly_stress_metrics(residual_load_mw: pd.Series) -> tuple[pd.DataFrame, float]:
    """
    Liefert pro Jahr:
      - peak_7d: max 7d rolling mean
      - peak_14d: max 14d rolling mean
      - hours_above_tau: #Stunden, in denen 7d-Mean > tau
      - energy_above_tau: Summe (7d-Mean - tau)+ über Zeit (MW*h)
    plus skalierte *_sc und stress_score.

    tau wird global (über alle Jahre) gesetzt, damit die Jahre vergleichbar sind.
    """
    roll7 = residual_load_mw.rolling(window=STRESS_ROLL_DAYS_MAIN * 24, min_periods=STRESS_ROLL_DAYS_MAIN * 24).mean()
    roll14 = residual_load_mw.rolling(window=STRESS_ROLL_DAYS_LONG * 24, min_periods=STRESS_ROLL_DAYS_LONG * 24).mean()

    # tau bestimmen
    if STRESS_THRESHOLD_MODE == "absolute":
        if STRESS_THRESHOLD_ABS_MW is None:
            raise ValueError("STRESS_THRESHOLD_MODE='absolute' verlangt STRESS_THRESHOLD_ABS_MW.")
        tau = float(STRESS_THRESHOLD_ABS_MW)
    elif STRESS_THRESHOLD_MODE == "quantile":
        tau = float(roll7.quantile(STRESS_THRESHOLD_Q))
    else:
        raise ValueError(f"Unbekannter STRESS_THRESHOLD_MODE: {STRESS_THRESHOLD_MODE}")

    rows = []
    for y in YEARS_TO_ANALYZE:
        r7y = roll7[roll7.index.year == y].dropna()
        r14y = roll14[roll14.index.year == y].dropna()

        peak_7d = float(r7y.max()) if not r7y.empty else np.nan
        peak_14d = float(r14y.max()) if not r14y.empty else np.nan

        if r7y.empty:
            hours_above = np.nan
            energy_above = np.nan
        else:
            above = r7y - tau
            above_pos = above.clip(lower=0.0)
            hours_above = float((above_pos > 0).sum())
            # MW*h (weil stündlich)
            energy_above = float(above_pos.sum())

        rows.append(
            {
                "year": int(y),
                "peak_7d": peak_7d,
                "peak_14d": peak_14d,
                "hours_above_tau": hours_above,
                "energy_above_tau_mwh": energy_above,
            }
        )

    d = pd.DataFrame(rows).set_index("year").sort_index()

    # skalieren (0=best, 1=worst)
    d["peak_7d_sc"] = _minmax_scale(d["peak_7d"])
    d["peak_14d_sc"] = _minmax_scale(d["peak_14d"])
    d["hours_above_tau_sc"] = _minmax_scale(d["hours_above_tau"])
    d["energy_above_tau_sc"] = _minmax_scale(d["energy_above_tau_mwh"])

    # Stress-Score
    d["stress_score"] = 0.0
    for k, w in STRESS_WEIGHTS.items():
        if k not in d.columns:
            raise KeyError(f"STRESS_WEIGHTS referenziert unbekannte Spalte: {k}")
        fillv = float(d[k].max(skipna=True)) if d[k].notna().any() else 1.0
        d["stress_score"] += d[k].fillna(fillv) * float(w)

    return d, tau


def identify_extreme_years(residual_load_mw: pd.Series) -> tuple[pd.Index, pd.Index, pd.Index, pd.DataFrame, float]:
    """
    NEU: Filter basiert auf kombinierten Stressmetriken (Top-25% stress_score werden ausgeschlossen).
    Zur Transparenz liefern wir die Tabelle + tau zurück.

    Rückgabe:
      valid_years, invalid_years, all_years, stress_df, tau
    """
    stress_df, tau = compute_yearly_stress_metrics(residual_load_mw)

    # Top-25% raus
    q75 = float(stress_df["stress_score"].quantile(0.75))
    valid_years = stress_df[stress_df["stress_score"] <= q75].index
    invalid_years = stress_df[stress_df["stress_score"] > q75].index
    all_years = stress_df.index

    return valid_years, invalid_years, all_years, stress_df, tau


# =============================================================================
# Referenzprofile (wochenbasiert)
# =============================================================================
def calculate_average_weekly_profile_from_series(
    s: pd.Series,
    start_year: int,
    end_year: int,
    normalization: str | None,
) -> pd.Series:
    """
    Baut ein wöchentliches Referenzprofil (week->value) aus einer Zeitreihe.
    """
    df = s.to_frame("value")
    df = df[(df.index.year >= start_year) & (df.index.year <= end_year)].copy()
    df["week"] = df.index.isocalendar().week.astype(int)
    df = df[df["week"] != 53].copy()
    df["year"] = df.index.year

    weekly_by_year = df.groupby(["year", "week"])["value"].sum()

    if normalization == "year_sum":
        year_totals = weekly_by_year.groupby(level="year").sum()
        weekly_by_year = weekly_by_year.div(year_totals, level="year")

    weekly_avg = weekly_by_year.groupby("week").mean()
    weekly_avg.index = weekly_avg.index.astype(int)
    return weekly_avg


def calculate_average_profile(
    df_generation: pd.DataFrame,
    start_year: int,
    end_year: int,
    normalization: str | None,
) -> pd.DataFrame:
    """
    Wie bisher: wöchentliches Referenzprofil für Erzeugungstechnologien + total
    """
    df = df_generation[(df_generation.index.year >= start_year) & (df_generation.index.year <= end_year)].copy()
    df["week"] = df.index.isocalendar().week.astype(int)
    df = df[df["week"] != 53].copy()
    df["year"] = df.index.year

    cols = TECHNOLOGIES + ["total"]
    weekly_by_year = df.groupby(["year", "week"])[cols].sum()

    if normalization == "year_sum":
        year_totals = weekly_by_year.groupby(level="year").sum()
        weekly_by_year = weekly_by_year.div(year_totals, level="year")

    weekly_avg = weekly_by_year.groupby("week").mean()
    return weekly_avg


# =============================================================================
# AUSSAGEKRÄFTIGE PLOTS
# =============================================================================
def plot_scaled_metric_distributions(metrics_df: pd.DataFrame, b0: int, b1: int, tag: str) -> str:
    """
    Ein Plot, der wirklich lesbar ist: alle Metriken auf 0..1 skaliert (0=best, 1=worst).
    """
    cols = ["mae_shape_sc", "mae_level_sc", "cos_sim_sc", "pearson_r_sc"]
    df_plot = metrics_df[cols].melt(var_name="metric", value_name="value")

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_plot, x="metric", y="value", inner="box")
    plt.title(f"Scaled metric distributions (0=best, 1=worst) {tag} {b0}-{b1}")
    plt.ylim(-0.05, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"metrics_violin_scaled_{tag}_{b0}_{b1}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_raw_metrics_separate(metrics_df: pd.DataFrame, b0: int, b1: int, tag: str) -> tuple[str, str]:
    """
    Rohmetriken, aber sinnvoll getrennt:
      - MAEs auf Log-Skala
      - Similarities auf linearer Skala [-1,1]
    """
    # MAE raw (log)
    plt.figure(figsize=(10, 5))
    sns.violinplot(
        data=metrics_df[["mae_shape", "mae_level"]].melt(var_name="metric", value_name="value"),
        x="metric",
        y="value",
        inner="box",
    )
    plt.yscale("log")
    plt.title(f"MAE distributions (log-scale) {tag} {b0}-{b1}")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, f"metrics_violin_mae_log_{tag}_{b0}_{b1}.png")
    plt.savefig(p1, dpi=200)
    plt.close()

    # similarities raw
    plt.figure(figsize=(10, 5))
    sns.violinplot(
        data=metrics_df[["cos_sim", "pearson_r"]].melt(var_name="metric", value_name="value"),
        x="metric",
        y="value",
        inner="box",
    )
    plt.ylim(-1.05, 1.05)
    plt.title(f"Similarity distributions {tag} {b0}-{b1}")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, f"metrics_violin_sim_{tag}_{b0}_{b1}.png")
    plt.savefig(p2, dpi=200)
    plt.close()

    return p1, p2


def plot_gap_topk(metrics_df: pd.DataFrame, b0: int, b1: int, tag: str, k: int = 15) -> str:
    """
    Zeigt, ob es überhaupt ein "deutlich bestes" Jahr gibt:
    Top-K Combined Scores als Kurve (Gap/Elbow Check).
    """
    top = metrics_df.sort_values("combined_score").head(k)

    plt.figure(figsize=(10, 5))
    plt.plot(top.index.astype(int), top["combined_score"].values, marker="o")
    plt.title(f"Top-{k} combined scores (gap check) {tag} {b0}-{b1}")
    plt.xlabel("Year")
    plt.ylabel("combined_score (lower better)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"combined_score_top{k}_{tag}_{b0}_{b1}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_combined_bar(
    df_final: pd.DataFrame, valid_years: pd.Index, best_year: int, b0: int, b1: int
) -> str:
    """
    Barplot über alle Jahre: final_score.
    Farbe: best=rot, Kandidat=blau, ausgeschlossen=grau.
    """
    df = df_final.sort_index()
    years = df.index.astype(int)
    vals = df["final_score"].values

    valid_set = set(map(int, list(valid_years)))

    colors = []
    for y in years:
        if int(y) == int(best_year):
            colors.append("red")
        elif int(y) in valid_set:
            colors.append("steelblue")
        else:
            colors.append("lightgray")

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(years, vals, color=colors, alpha=0.9, width=0.7)

    ax.set_title(f"Final Score per Year {b0}-{b1}")
    ax.set_xlabel("Year")
    ax.set_ylabel("final_score (lower better)")

    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45)

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red", lw=6, label=f"Bestes Jahr: {best_year}"),
        Line2D([0], [0], color="steelblue", lw=6, label="Kandidaten (nicht-extrem nach Stress-Score)"),
        Line2D([0], [0], color="lightgray", lw=6, label="Ausgeschlossen (Top-25% Stress-Score)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=12, frameon=True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"final_score_bar_{b0}_{b1}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_weekly_heatmap_ordered(weekly_block: pd.DataFrame, order_years: list[int], b0: int, b1: int, tag: str) -> str:
    """
    Heatmap der Wochenprofile in einer Reihenfolge (z.B. final_score).
    """
    df = weekly_block.loc[order_years].copy()

    plt.figure(figsize=(14, 8))
    sns.heatmap(df, cmap="viridis", cbar_kws={"label": f"Weekly {tag} (sum over hours)"})
    plt.xlabel("Week")
    plt.ylabel("Year (ordered)")
    plt.title(f"Weekly profiles ordered {tag} {b0}-{b1}")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"weekly_profiles_heatmap_{tag}_{b0}_{b1}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_dendrogram(weekly_block: pd.DataFrame, Z, b0: int, b1: int, tag: str) -> str:
    plt.figure(figsize=(12, 4))
    dendrogram(Z, labels=weekly_block.index.astype(str), leaf_rotation=90)
    plt.title(f"Hierarchical clustering of years ({tag}) {b0}-{b1}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"years_dendrogram_{tag}_{b0}_{b1}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_stress_overview(stress_df: pd.DataFrame, b0: int, b1: int) -> str:
    """
    Schnell-Plot: Stress-Score & Komponenten (skaliert) über Jahre.
    """
    df = stress_df.loc[b0:b1].copy()
    cols = ["peak_7d_sc", "peak_14d_sc", "hours_above_tau_sc", "energy_above_tau_sc", "stress_score"]
    plt.figure(figsize=(14, 5))
    for c in cols:
        plt.plot(df.index.astype(int), df[c].values, marker="o", label=c)
    plt.title(f"Stress metrics (scaled) {b0}-{b1}")
    plt.xlabel("Year")
    plt.ylabel("0..1 (higher=worse), stress_score weighted")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"stress_metrics_scaled_{b0}_{b1}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ANALYSE: Standardjahr (2025–2050) – typisches Profil + robuste Stress-Filterung")
    print("=" * 80)

    # 1) Demand
    print("\n[1/6] Lade Demand (pro Jahr)...")
    df_demand = load_demand_timeseries()
    print(f"  Demand geladen: {df_demand.index.min()} -> {df_demand.index.max()} | shape={df_demand.shape}")

    # 2) CFs
    print("\n[2/6] Lade Capacity Factors...")
    df_cfs = load_capacity_factors()
    print(f"  CFs geladen:   {df_cfs.index.min()} -> {df_cfs.index.max()} | shape={df_cfs.shape}")

    # 3) Kapazitäten
    print("\n[3/6] Lade installierte Kapazitäten...")
    cap_matrix_gw, tech_weights = load_installed_capacities()
    print("  Technologie-Gewichtung (aus installierten GW):")
    for tech, w in tech_weights.items():
        print(f"    {tech:15s}: {w:6.1%}")

    # 4) Generation + Residual
    print("\n[4/6] Aggregiere Erzeugung + berechne Residual Load...")
    df_generation = aggregate_generation_mw(df_cfs, cap_matrix_gw)

    # Align index
    df_generation = df_generation.loc[df_demand.index.intersection(df_generation.index)].copy()
    df_demand = df_demand.loc[df_generation.index].copy()

    residual_load = compute_residual_load_mw(df_demand, df_generation)
    print(f"  Residual Load: {residual_load.index.min()} -> {residual_load.index.max()} | n={len(residual_load):,}")

    # 5) Filter (robust) + Stress-Tabelle
    print("\n[5/6] Robust-Filter (Stress-Score) ...")
    valid_years, invalid_years, all_years, stress_df, tau = identify_extreme_years(residual_load)

    print("\nStress-Filter (Top-25% stress_score ausgeschlossen):")
    print(f"  tau (Schwelle auf 7d-Mean, mode={STRESS_THRESHOLD_MODE}) = {tau:,.2f} MW")
    print(f"  Extreme Jahre (Top-25%): {len(invalid_years)} -> {list(map(int, invalid_years))}")
    print(f"  Kandidaten:             {len(valid_years)} -> {list(map(int, valid_years))}")
    print(f"  Gesamtjahre:            {len(all_years)}")

    stress_csv = os.path.join(OUTPUT_DIR, f"stress_metrics_{Y0}_{Y1}.csv")
    stress_df.to_csv(stress_csv)
    print("  Stress CSV gespeichert:", stress_csv)

    # 6) Standardjahr-Auswahl: Erzeugung + Residual typischer machen
    print("\n[6/6] Standardjahr-Auswahl: typisches Erzeugungsprofil + typisches Residualprofil...")

    print("\nProfil-Setup:")
    print(f"  PROFILE_NORMALIZATION = {PROFILE_NORMALIZATION}")
    print(f"  LOSS                 = {'SSE' if USE_SSE else 'MAE'}")
    print(f"  FINAL_OBJECTIVE_WEIGHTS = {FINAL_OBJECTIVE_WEIGHTS}")

    def run_block_with_alternatives(b0: int, b1: int) -> None:
        # --- Referenzprofile ---
        avg_gen_profile = calculate_average_profile(
            df_generation,
            start_year=b0,
            end_year=b1,
            normalization=PROFILE_NORMALIZATION,
        )

        avg_res_profile = calculate_average_weekly_profile_from_series(
            residual_load,
            start_year=b0,
            end_year=b1,
            normalization=PROFILE_NORMALIZATION,
        )

        # --- Wochenprofile pro Jahr ---
        weekly_gen_yearly = weekly_profiles_by_year(df_generation, value_col="total")
        weekly_gen_block = weekly_gen_yearly[(weekly_gen_yearly.index >= b0) & (weekly_gen_yearly.index <= b1)]

        # Residual als df (damit weekly_profiles_by_year passt)
        df_res = residual_load.to_frame("residual_load_mw")
        weekly_res_yearly = weekly_profiles_by_year(df_res, value_col="residual_load_mw")
        weekly_res_block = weekly_res_yearly[(weekly_res_yearly.index >= b0) & (weekly_res_yearly.index <= b1)]

        # --- Metriken vs Referenz ---
        metrics_gen = compute_alternative_metrics(weekly_gen_block, ref_weekly=avg_gen_profile["total"])
        metrics_res = compute_alternative_metrics(weekly_res_block, ref_weekly=avg_res_profile)

        # umbenennen für Klarheit
        metrics_gen = metrics_gen.add_prefix("gen_")
        metrics_res = metrics_res.add_prefix("res_")

        # zusammenführen
        dfm = metrics_gen.join(metrics_res, how="inner")

        # final_score (Multi-Objective)
        dfm["final_score"] = (
            FINAL_OBJECTIVE_WEIGHTS["gen_typical"] * dfm["gen_combined_score"]
            + FINAL_OBJECTIVE_WEIGHTS["residual_typical"] * dfm["res_combined_score"]
        )

        # Kandidatenfilter (nur nicht-extreme Stressjahre)
        dfm_candidates = dfm[dfm.index.isin(valid_years)].sort_values("final_score")
        if dfm_candidates.empty:
            raise RuntimeError(f"Keine Kandidaten (nach Stress-Filter) im Block {b0}-{b1}.")

        best_year = int(dfm_candidates.index[0])

        print(f"\nTop 10 Jahre (final_score) | Block {b0}-{b1} (nur Kandidaten):")
        cols_show = [
            "final_score",
            "gen_combined_score",
            "res_combined_score",
            "gen_mae_shape",
            "gen_cos_sim",
            "res_mae_shape",
            "res_cos_sim",
        ]
        cols_show = [c for c in cols_show if c in dfm_candidates.columns]
        print(dfm_candidates[cols_show].head(10).to_string(float_format="{:0.6f}".format))

        # --- Outputs: CSV ---
        csv_path = os.path.join(OUTPUT_DIR, f"scores_final_{b0}_{b1}.csv")
        dfm.sort_index().to_csv(csv_path)

        # --- Stress Plot (für Zeitraum) ---
        p_stress = plot_stress_overview(stress_df, b0, b1)

        # --- Heatmaps ordered by final_score ---
        order = dfm.sort_values("final_score").index.astype(int).tolist()
        p_heat_gen = plot_weekly_heatmap_ordered(weekly_gen_block, order, b0, b1, tag="generation_total")
        p_heat_res = plot_weekly_heatmap_ordered(weekly_res_block, order, b0, b1, tag="residual_load")

        # --- Metrikverteilungen (gen/res getrennt, skaliert) ---
        # Dafür brauchen wir wieder die unpräfixte Struktur -> wir nutzen metrics_gen/res vor Prefix
        mg = metrics_gen.copy()
        mr = metrics_res.copy()

        # Für die Plots erwarten die Funktionen Spaltennamen ohne Prefix.
        # Wir bauen temporäre Frames in "Originalschema" zurück:
        mg_plot = mg.rename(columns={k: k.replace("gen_", "") for k in mg.columns})
        mr_plot = mr.rename(columns={k: k.replace("res_", "") for k in mr.columns})

        p_scaled_gen = plot_scaled_metric_distributions(mg_plot, b0, b1, tag="GEN")
        p_scaled_res = plot_scaled_metric_distributions(mr_plot, b0, b1, tag="RES")

        p_mae_log_gen, p_sim_gen = plot_raw_metrics_separate(mg_plot, b0, b1, tag="GEN")
        p_mae_log_res, p_sim_res = plot_raw_metrics_separate(mr_plot, b0, b1, tag="RES")

        # --- Gap/Elbow ---
        # hierfür verwenden wir dfm mit final_score -> wir bauen ein kleines metrics_df-like
        df_gap = dfm[["final_score"]].rename(columns={"final_score": "combined_score"})
        p_gap = plot_gap_topk(df_gap, b0, b1, tag="FINAL", k=15)

        # --- Barplot final_score (alle Jahre) ---
        p_bar = plot_combined_bar(dfm, valid_years, best_year, b0, b1)

        # --- Clustering + Dendrogram (auf generation_total; alternativ residual möglich) ---
        labels, medoids, Z = cluster_years_and_medoid(weekly_gen_block)
        p_den = plot_dendrogram(weekly_gen_block, Z, b0, b1, tag="GEN")

        print("\nCluster medoids (GEN; representative years):", medoids)
        print("Best year by final_score (nur Kandidaten):", best_year)

        print("\nSaved outputs:")
        print("  Stress CSV :", stress_csv)
        print("  Final CSV  :", csv_path)
        print("  Stress plot:", p_stress)
        print("  Heat GEN   :", p_heat_gen)
        print("  Heat RES   :", p_heat_res)
        print("  Scaled GEN :", p_scaled_gen)
        print("  Scaled RES :", p_scaled_res)
        print("  Raw GEN MAE:", p_mae_log_gen)
        print("  Raw GEN sim:", p_sim_gen)
        print("  Raw RES MAE:", p_mae_log_res)
        print("  Raw RES sim:", p_sim_res)
        print("  Gap FINAL  :", p_gap)
        print("  Bar FINAL  :", p_bar)
        print("  Dendrogram :", p_den)

    if DECADAL_MODE:
        for b0, b1 in DECADES:
            run_block_with_alternatives(b0, b1)
    else:
        run_block_with_alternatives(AVG_PERIOD_START, AVG_PERIOD_END)

    print("\n" + "=" * 80)
    print("ANALYSE ABGESCHLOSSEN")
    print("=" * 80)
